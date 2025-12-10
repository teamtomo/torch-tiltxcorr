import torch
from torch.nn import functional as F
from torch_fourier_filter.bandpass import bandpass_filter

from torch_tiltxcorr.utils import taper_image_edges, calculate_cross_correlation, get_shift_from_correlation_image


def tiltxcorr_no_stretch(
    tilt_series: torch.Tensor,
    tilt_angles: torch.Tensor,  # (b, )
    low_pass_cutoff: float,  # cycles/px
) -> torch.Tensor:
    """Find coarse shifts of images without stretching along tilt axis."""
    # extract shape
    b, h, w = tilt_series.shape

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
    img1, img2 = (taper_image_edges(img1), taper_image_edges(img2))
    # pad images for cross-correlation
    p = int(0.5 * min(img1.shape[-2:]))
    img1 = F.pad(img1, [p] * 4, value=img1.mean())
    img2 = F.pad(img2, [p] * 4, value=img2.mean())
    correlation_image = calculate_cross_correlation(img1, img2)
    # remove padding from the result
    correlation_image = F.pad(correlation_image, [-p] * 4)
    shift = get_shift_from_correlation_image(correlation_image)
    return shift
