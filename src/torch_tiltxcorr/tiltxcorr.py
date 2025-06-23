from math import cos

import numpy as np
import torch
from torch_fourier_filter.bandpass import bandpass_filter
from torch_grid_utils import rectangle

from torch_tiltxcorr.utils import (
    apply_stretch_perpendicular_to_tilt_axis,
    calculate_cross_correlation,
    get_shift_from_correlation_image,
    transform_shift_from_stretched_image,
)


def tiltxcorr(
    tilt_series: torch.Tensor,  # (b, h, w)
    tilt_angles: torch.Tensor,  # (b, )
    tilt_axis_angle: float,
    low_pass_cutoff: float,  # cycles/px
    taper_fraction: float = .1,  # how much of total image size is tapered
):
    # extract shape
    b, h, w = tilt_series.shape
    taper_width = taper_fraction * min(h, w)
    taper_mask_shape = (int(h - taper_width), int(w - taper_width))

    # create edge taper mask
    edge_taper_mask = rectangle(
        dimensions=taper_mask_shape,
        image_shape=(h, w),
        smoothing_radius=taper_width / 2,
        device=tilt_series.device,
    )

    # sort input data by tilt angle
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

    # find index where tilt angle transitions from negative to positive
    transition_idx = torch.where(sorted_tilt_angles >= 0)[0][0]

    # grab positive branch: least positive tilt angle -> most positive tilt angles
    # make sure the transition tilt is added to both by subtracting -1
    positive_branch_tilt_series = sorted_tilt_series[transition_idx - 1:]
    positive_leaf_tilt_angles = sorted_tilt_angles[transition_idx - 1:]

    # grab negative branch: least positive tilt angle -> most negative tilt angles
    negative_branch_tilt_series = torch.flip(sorted_tilt_series[:transition_idx + 1], dims=[0])
    negative_branch_tilt_angles = torch.flip(sorted_tilt_angles[:transition_idx + 1], dims=[0])

    # find shifts between images in positive branch
    positive_branch_shifts = _find_shifts_for_branch(
        tilt_series=positive_branch_tilt_series,
        tilt_angles=positive_leaf_tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        edge_taper_mask=edge_taper_mask,
    )

    # find shifts between images in negative branch
    negative_branch_shifts = _find_shifts_for_branch(
        tilt_series=negative_branch_tilt_series,
        tilt_angles=negative_branch_tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        edge_taper_mask=edge_taper_mask,
    )

    # shifts are between adjacent pairs, take cumulative sum in each leaf
    # to get shifts that center each image
    shifts = torch.zeros(size=(b, 2))
    negative_branch_shifts = torch.cumsum(negative_branch_shifts, dim=0)
    # skip one on positive branch as we added the reference tilt twice
    positive_branch_shifts = torch.cumsum(positive_branch_shifts[1:], dim=0)

    # assemble ordered shifts for whole tilt series
    shifts[:transition_idx + 1, :] = torch.flip(negative_branch_shifts, dims=(0,))
    shifts[transition_idx:, :] = positive_branch_shifts

    # put shifts back in original order
    shifts_original_order = torch.zeros_like(shifts)
    for idx_unsorted, idx_sorted in enumerate(sorted_indices):
        shifts_original_order[idx_sorted] = shifts[idx_unsorted]
    return shifts


def _find_shifts_for_branch(
    tilt_series: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_axis_angle: float,
    edge_taper_mask: torch.Tensor,
):
    # Initialize shifts tensor
    leaf_shifts = torch.zeros(
        size=(len(tilt_series), 2),
        dtype=torch.float32,
        device=tilt_series.device
    )

    # Iterate over pairs of images in the leaf
    for idx in range(len(tilt_series) - 1):
        img1, img2 = tilt_series[idx], tilt_series[idx + 1]
        tilt_angle1, tilt_angle2 = float(tilt_angles[idx]), float(tilt_angles[idx + 1])

        # store shift which aligns img2 with img1
        leaf_shifts[idx + 1] = _find_shift_between_adjacent_tilt_images(
            img1=img1, img2=img2,
            tilt_angle1=tilt_angle1, tilt_angle2=tilt_angle2,
            tilt_axis_angle=tilt_axis_angle,
            edge_taper_mask=edge_taper_mask,
        )

    return leaf_shifts


def _find_shift_between_adjacent_tilt_images(
    img1: torch.Tensor,
    img2: torch.Tensor,
    tilt_angle1: float,
    tilt_angle2: float,
    tilt_axis_angle: float,
    edge_taper_mask: torch.Tensor,
) -> torch.Tensor:
    # Get absolute tilt angles
    abs_tilt_angle1, abs_tilt_angle2 = abs(tilt_angle1), abs(tilt_angle2)

    # Stretch image with larger absolute tilt angle (always img2)
    scale_factor = cos(np.deg2rad(abs_tilt_angle1)) / cos(np.deg2rad(abs_tilt_angle2))
    img2_stretched = apply_stretch_perpendicular_to_tilt_axis(
        img2, tilt_axis_angle=tilt_axis_angle, scale_factor=scale_factor
    )

    img1 = img1 * edge_taper_mask
    img2_stretched = img2_stretched * edge_taper_mask

    # Calculate correlation and get shift
    correlation_image = calculate_cross_correlation(
        img1, img2_stretched
    )

    shift = get_shift_from_correlation_image(correlation_image)

    # Transform shift to account for the fact that img2 was stretched
    transformed_shift = transform_shift_from_stretched_image(
        shift=shift, tilt_axis_angle=tilt_axis_angle, scale_factor=scale_factor
    )

    return transformed_shift


def tiltxcorr_no_stretch(
        tilt_series: torch.Tensor,
        tilt_angles: torch.Tensor,  # (b, )
        low_pass_cutoff: float,  # cycles/px
        taper_fraction: float = .1,  # how much of total image size is tapered
) -> torch.Tensor:
    """Find coarse shifts of images without stretching along tilt axis."""
    # extract shape
    b, h, w = tilt_series.shape
    taper_width = taper_fraction * min(h, w)
    taper_mask_shape = (int(h - taper_width), int(w - taper_width))

    # create edge taper mask
    edge_taper_mask = rectangle(
        dimensions=taper_mask_shape,
        image_shape=(h, w),
        smoothing_radius=taper_width / 2,
        device=tilt_series.device,
    )

    # sort input data by tilt angle
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
    # we wont stretch images here, so we can already taper all the edges
    sorted_tilt_series *= edge_taper_mask

    # find index where tilt angle transitions from negative to positive
    transition_idx = torch.where(sorted_tilt_angles >= 0)[0][0]

    # grab positive branch: least positive tilt angle -> most positive tilt angles
    # make sure the transition tilt is added to both by subtracting -1
    positive_branch_tilt_series = sorted_tilt_series[transition_idx - 1:]

    # grab negative branch: least positive tilt angle -> most negative tilt angles
    negative_branch_tilt_series = torch.flip(
        sorted_tilt_series[:transition_idx + 1], dims=[0]
    )

    # find shifts between images in positive branch
    positive_branch_shifts = _find_shifts_for_branch_no_stretch(
        tilt_series=positive_branch_tilt_series
    )

    # find shifts between images in negative branch
    negative_branch_shifts = _find_shifts_for_branch_no_stretch(
        tilt_series=negative_branch_tilt_series
    )

    negative_branch_shifts = torch.cumsum(negative_branch_shifts, dim=0)
    # skip one on positive branch as we added the reference tilt twice
    positive_branch_shifts = torch.cumsum(positive_branch_shifts[1:], dim=0)

    # assemble ordered shifts for whole tilt series
    shifts = torch.zeros(size=(b, 2))
    shifts[:transition_idx + 1, :] = torch.flip(
        negative_branch_shifts, dims=(0,)
    )
    shifts[transition_idx:, :] = positive_branch_shifts

    # put shifts back in original order
    shifts_original_order = torch.zeros_like(shifts)
    for idx_unsorted, idx_sorted in enumerate(sorted_indices):
        shifts_original_order[idx_sorted] = shifts[idx_unsorted]
    return shifts


def _find_shifts_for_branch_no_stretch(
        tilt_series: torch.Tensor,
) -> torch.Tensor:
    # Initialize shifts tensor
    leaf_shifts = torch.zeros(
        size=(len(tilt_series), 2),
        dtype=torch.float32,
        device=tilt_series.device
    )

    # Iterate over pairs of images in the leaf
    for idx in range(len(tilt_series) - 1):
        img1, img2 = tilt_series[idx], tilt_series[idx + 1]

        # store shift which aligns img2 with img1
        leaf_shifts[idx + 1] = (
            _find_shift_between_adjacent_tilt_images_no_stretch(
                img1=img1, img2=img2,
            )
        )

    return leaf_shifts


def _find_shift_between_adjacent_tilt_images_no_stretch(
        img1: torch.Tensor,
        img2: torch.Tensor,
) -> torch.Tensor:
    correlation_image = calculate_cross_correlation(img1, img2)
    shift = get_shift_from_correlation_image(correlation_image)
    return shift
