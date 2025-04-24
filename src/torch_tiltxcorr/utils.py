import einops
import torch
from torch_affine_utils.transforms_2d import R, S, T
from torch_affine_utils import homogenise_coordinates
from torch_transform_image import affine_transform_image_2d
from torch_fourier_filter.bandpass import bandpass_filter

def apply_stretch_perpendicular_to_tilt_axis(
    image: torch.Tensor,
    tilt_axis_angle: float,
    scale_factor: float,
) -> torch.Tensor:
    # grab image dimensions and calculate center
    h, w = image.shape[-2:]
    center = torch.tensor((h // 2, w // 2), device=image.device, dtype=image.dtype)

    # construct transforms
    T0 = T(-1 * center)  # origin from array [0, 0] to center of image
    R0 = R(-1 * tilt_axis_angle, yx=True)  # rotate coords so tilt axis is aligned along image Y
    S0 = S((1, 1 / scale_factor))  # scale coords in X (perpendicular to tilt axis)
    R1 = R(tilt_axis_angle, yx=True)  # rotate coords to align current Y with tilt axis in image
    T1 = T(center)  # origin back to [0, 0]

    M = (T1 @ R1 @ S0 @ R0 @ T0)
    M = M.to(image.device)

    # apply transform
    image = affine_transform_image_2d(
        image, matrices=M, interpolation="bicubic", yx_matrices=True
    )
    return image


def calculate_cross_correlation(img1, img2):
    img1_fft = torch.fft.rfft2(img1)
    img2_fft = torch.fft.rfft2(img2)
    cross_power = img1_fft * torch.conj(img2_fft)
    normalized_cross_power = cross_power / (torch.abs(cross_power) + 1e-8)
    return torch.fft.irfft2(normalized_cross_power, s=img1.shape)


def get_shift_from_correlation_image(correlation_image: torch.Tensor) -> torch.Tensor:
    """shift should be applied to img2 to align with img1"""
    flat_idx = torch.argmax(correlation_image)
    h, w = correlation_image.shape
    peak_y, peak_x = flat_idx // w, flat_idx % w

    # convert to shift (accounting for FFT centering)
    dy = (peak_y.item() - h // 2) % h - h // 2
    dx = (peak_x.item() - w // 2) % w - w // 2
    return torch.tensor([dy, dx], device=correlation_image.device, dtype=correlation_image.dtype)


def transform_shift_from_stretched_image(
    shift: torch.Tensor,
    tilt_axis_angle: float,
    scale_factor: float,
) -> torch.Tensor:
    # construct (3, 1) column vector with homogenous coords
    shift = homogenise_coordinates(shift)
    shift = einops.rearrange(shift, 'yxw -> yxw 1')

    # compose transforms
    R0 = R(-1 * tilt_axis_angle, yx=True)  # align tilt axis with Y
    S0 = S((1, 1 / scale_factor)) # scale X component
    R1 = R(tilt_axis_angle, yx=True) # rotate tilt axis back into

    M = R1 @ S0 @ R0
    M = M.to(shift.device)

    # apply transform
    transformed_shift = M @ shift
    transformed_shift = transformed_shift.view((3, ))[:2]
    return transformed_shift
