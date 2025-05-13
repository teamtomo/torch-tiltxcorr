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
    result = torch.fft.irfft2(normalized_cross_power, s=img1.shape)
    result = torch.real(torch.fft.ifftshift(result, dim=(-2, -1)))
    return result


def get_shift_from_correlation_image(correlation_image: torch.Tensor) -> torch.Tensor:
    """shift should be applied to img2 to align with img1"""
    h, w = correlation_image.shape  # for unraveling
    dtype, device = correlation_image.dtype, correlation_image.device
    image_shape = torch.as_tensor(
        correlation_image.shape,
        device=device,
        dtype=dtype
    )
    center = torch.divide(image_shape, 2, rounding_mode="floor")

    flat_idx = torch.argmax(correlation_image)
    peak_y, peak_x = (flat_idx // w), (flat_idx % w)

    # Ensure that the max index is not on the border
    if (
            peak_y == 0
            or peak_y == h - 1
            or peak_x == 0
            or peak_x == w - 1
    ):
        # convert to shift (accounting for FFT centering)
        shift = torch.tensor(
            [peak_y, peak_x],
            device=correlation_image.device,
            dtype=correlation_image.dtype
        ) - center
        return shift

    # Parabolic interpolation in the y direction
    f_y0 = correlation_image[peak_y - 1, peak_x]
    f_y1 = correlation_image[peak_y, peak_x]
    f_y2 = correlation_image[peak_y + 1, peak_x]
    subpixel_peak_y = peak_y + 0.5 * (f_y0 - f_y2) / (f_y0 - 2 * f_y1 + f_y2)

    # Parabolic interpolation in the x direction
    f_x0 = correlation_image[peak_y, peak_x - 1]
    f_x1 = correlation_image[peak_y, peak_x]
    f_x2 = correlation_image[peak_y, peak_x + 1]
    subpixel_peak_x = peak_x + 0.5 * (f_x0 - f_x2) / (f_x0 - 2 * f_x1 + f_x2)

    subpixel_shift = torch.tensor(
        [subpixel_peak_y, subpixel_peak_x],
        device=correlation_image.device,
        dtype=correlation_image.dtype
    ) - center
    return subpixel_shift


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
