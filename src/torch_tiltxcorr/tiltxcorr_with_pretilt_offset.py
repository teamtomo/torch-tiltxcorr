import math

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize_scalar
from torch_fourier_filter.bandpass import bandpass_filter

from torch_tiltxcorr.utils import (
    calculate_cross_correlation,
    get_shift_from_correlation_image,
    transform_shift_from_stretched_image,
    taper_image_edges,
    apply_stretch_perpendicular_to_tilt_axis
)


def tiltxcorr_with_pretilt_offset(
    tilt_series: torch.Tensor,  # (b, h, w)
    tilt_angles: torch.Tensor,  # (b, )
    tilt_axis_angle: float,
    low_pass_cutoff: float,  # cycles/px
    pretilt_range: tuple[float, float] = (-30.0, 30.0),  # search range in degrees
    max_iter: int = 10,  # max iterations for Brent's method
) -> tuple[torch.Tensor, float]:  # (b, 2) yx shifts and optimal pretilt offset
    """
    Find optimal pretilt offset by maximizing sum of inter-image tilted cross correlations.

    Uses scipy's Brent's method (bounded).

    Args:
        tilt_series: Stack of tilt images (b, h, w)
        tilt_angles: Tilt angles for each image (b,)
        tilt_axis_angle: Angle of tilt axis in degrees
        low_pass_cutoff: Low-pass filter cutoff in cycles/px
        pretilt_range: (min, max) range of pretilt offsets to search
        max_iter: Maximum iterations for Brent's method optimizer

    Returns:
        shifts: Optimal shifts for each image (b, 2) yx coords
        optimal_pretilt: Optimal pretilt offset in degrees
    """
    pretilt_min, pretilt_max = pretilt_range

    # Track history for correlation curve
    pretilt_history = []
    correlation_history = []

    def objective(pretilt_offset: float) -> float:
        """Objective function: negative correlation (to minimize)."""
        _, total_correlation = _compute_shifts_with_pretilt(
            tilt_series=tilt_series,
            tilt_angles=tilt_angles,
            tilt_axis_angle=tilt_axis_angle,
            low_pass_cutoff=low_pass_cutoff,
            pretilt_offset=pretilt_offset,
        )

        # Convert to float and store history
        correlation_val = float(total_correlation.item())
        pretilt_history.append(pretilt_offset)
        correlation_history.append(correlation_val)

        # Return negative for minimization
        return -correlation_val

    # Run Brent's method optimization
    result = minimize_scalar(
        objective,
        bounds=(pretilt_min, pretilt_max),
        method='bounded',
        options={'maxiter': max_iter}
    )

    optimal_pretilt = float(result.x)

    # Get final shifts with optimal pretilt
    final_shifts, final_correlation = _compute_shifts_with_pretilt(
        tilt_series=tilt_series,
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        low_pass_cutoff=low_pass_cutoff,
        pretilt_offset=optimal_pretilt,
    )

    return final_shifts, optimal_pretilt


def _compute_shifts_with_pretilt(
    tilt_series: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_axis_angle: float,
    low_pass_cutoff: float,
    pretilt_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute shifts for a given pretilt offset and return total correlation.

    Returns:
        shifts: Computed shifts (b, 2)
        total_correlation: Sum of all inter-image correlation peaks (as tensor)
    """
    # extract shape
    b, h, w = tilt_series.shape

    # sort input data by ORIGINAL tilt angle to maintain consistent reference
    tilt_angles = torch.as_tensor(tilt_angles).float()
    sorted_indices = torch.argsort(tilt_angles)
    sorted_tilt_series = tilt_series[sorted_indices]
    sorted_tilt_angles = tilt_angles[sorted_indices]

    # rfft & filter
    sorted_tilt_series_rfft = torch.fft.rfft2(sorted_tilt_series)
    filter = bandpass_filter(
        low=0.025,
        high=low_pass_cutoff,
        falloff=0.025,
        rfft=True,
        fftshift=False,
        image_shape=(h, w),
        device=tilt_series.device,
    )
    sorted_tilt_series_rfft *= filter
    sorted_tilt_series = torch.fft.irfft2(sorted_tilt_series_rfft, s=(h, w))

    # find index where input tilt angles transition from negative to positive
    transition_idx = torch.argmin(torch.abs(sorted_tilt_angles))

    # apply pretilt offset to tilt angles
    sorted_tilt_angles_with_pretilt = sorted_tilt_angles + pretilt_offset

    # grab positive branch: least positive tilt angle -> most positive tilt angles
    positive_branch_tilt_series = sorted_tilt_series[transition_idx:]
    positive_leaf_tilt_angles = sorted_tilt_angles_with_pretilt[transition_idx:]

    # grab negative branch: least positive tilt angle -> most negative tilt angles
    negative_branch_tilt_series = torch.flip(sorted_tilt_series[:transition_idx + 1], dims=[0])
    negative_branch_tilt_angles = torch.flip(sorted_tilt_angles_with_pretilt[:transition_idx + 1], dims=[0])

    # find shifts between images in positive branch
    positive_branch_shifts, positive_correlation = _find_shifts_for_branch(
        tilt_series=positive_branch_tilt_series,
        tilt_angles=positive_leaf_tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
    )

    # find shifts between images in negative branch
    negative_branch_shifts, negative_correlation = _find_shifts_for_branch(
        tilt_series=negative_branch_tilt_series,
        tilt_angles=negative_branch_tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
    )

    # Total correlation is sum from both branches
    total_correlation = positive_correlation + negative_correlation

    # shifts are between adjacent pairs, take cumulative sum in each leaf
    # to get shifts that center each image
    shifts = torch.zeros(size=(b, 2))
    negative_branch_shifts = torch.cumsum(negative_branch_shifts, dim=0)
    # skip one on positive branch as we added the reference tilt twice
    positive_branch_shifts = torch.cumsum(positive_branch_shifts, dim=0)

    # assemble ordered shifts for whole tilt series
    shifts[:transition_idx + 1, :] = (
        torch.flip(negative_branch_shifts, dims=(0,))
    )
    shifts[transition_idx:, :] = positive_branch_shifts

    # put shifts back in original order
    shifts_original_order = torch.zeros_like(shifts)
    for idx_unsorted, idx_sorted in enumerate(sorted_indices):
        shifts_original_order[idx_sorted] = shifts[idx_unsorted]

    return shifts_original_order, total_correlation


def _find_shifts_for_branch(
    tilt_series: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_axis_angle: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Initialize shifts tensor
    leaf_shifts = torch.zeros(
        size=(len(tilt_series), 2),
        dtype=torch.float32,
        device=tilt_series.device
    )

    # Accumulate total correlation across all pairs
    total_correlation = None

    # Iterate over pairs of images in the leaf
    for idx in range(len(tilt_series) - 1):
        img1, img2 = tilt_series[idx], tilt_series[idx + 1]
        tilt_angle1, tilt_angle2 = float(tilt_angles[idx]), float(tilt_angles[idx + 1])

        # store shift which aligns img2 with img1
        shift, correlation = _find_shift_between_adjacent_tilt_images(
            img1=img1, img2=img2,
            tilt_angle1=tilt_angle1, tilt_angle2=tilt_angle2,
            tilt_axis_angle=tilt_axis_angle,
        )
        leaf_shifts[idx + 1] = shift
        # Accumulate correlation (initialize on first iteration)
        if total_correlation is None:
            total_correlation = correlation
        else:
            total_correlation = total_correlation + correlation

    return leaf_shifts, total_correlation


def _find_shift_between_adjacent_tilt_images(
    img1: torch.Tensor,
    img2: torch.Tensor,
    tilt_angle1: float,
    tilt_angle2: float,
    tilt_axis_angle: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Get absolute tilt angles
    abs_tilt_angle1, abs_tilt_angle2 = abs(tilt_angle1), abs(tilt_angle2)

    # Stretch image with larger absolute tilt angle (always img2)
    scale_factor = math.cos(np.deg2rad(abs_tilt_angle1)) / math.cos(np.deg2rad(abs_tilt_angle2))
    img2_stretched = apply_stretch_perpendicular_to_tilt_axis(
        img2, tilt_axis_angle=tilt_axis_angle, scale_factor=scale_factor
    )
    img1, img2_stretched = (
        taper_image_edges(img1), taper_image_edges(img2_stretched)
    )

    # pad images for cross-correlation
    p = int(0.5 * min(img1.shape[-2:]))
    img1 = F.pad(img1, [p] * 4, value=img1.mean())
    img2_stretched = F.pad(img2_stretched, [p] * 4, value=img2_stretched.mean())

    # Calculate correlation and get shift
    correlation_image = calculate_cross_correlation(
        img1, img2_stretched
    )
    # remove padding from the result
    correlation_image = F.pad(correlation_image, [-p] * 4)

    shift = get_shift_from_correlation_image(correlation_image)

    # Get the peak correlation value as a quality metric
    correlation_peak = correlation_image.max()

    # Transform shift to account for the fact that img2 was stretched
    transformed_shift = transform_shift_from_stretched_image(
        shift=shift, tilt_axis_angle=tilt_axis_angle, scale_factor=scale_factor
    )

    return transformed_shift, correlation_peak
