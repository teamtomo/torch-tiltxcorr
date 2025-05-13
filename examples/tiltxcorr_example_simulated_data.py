# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "einops",
#   "numpy",
#   "torch",
#   "torch-grid-utils",
#   "torch-fourier-slice",
#   "torch-fourier-shift",
#   "torch-fourier-filter",
#   "scipy",
#   "napari[pyqt5]",
#   "torch-tiltxcorr",
# ]
# exclude-newer = "2025-03-29T00:00:00Z"
# [tool.uv.sources]
# torch-tiltxcorr = { path = "../" }
# ///
import einops
import numpy as np
import torch
from torch_grid_utils import coordinate_grid
from torch_fourier_slice import project_3d_to_2d
from torch_fourier_shift import fourier_shift_image_2d
from scipy.spatial.transform import Rotation as R
import napari

from torch_tiltxcorr import tiltxcorr


def simulate_volume() -> torch.Tensor:  # (128, 128, 128)
    point_positions = (
        [64, 32, 32],
        [64, 32, 96],
        [64, 96, 32],
        [64, 96, 96],
    )

    volume = torch.zeros(size=(128, 128, 128))
    for point in point_positions:
        _volume = coordinate_grid((128, 128, 128), center=point, norm=True)
        _volume = _volume < 2
        _volume = _volume.float()
        volume += _volume

    return volume


def simulate_tilt_series(
    volume: torch.Tensor,
    tilt_axis_angle: float,
    tilt_angles: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    # construct rotation matrices
    in_plane_rotation_angles = einops.repeat(
        np.array([tilt_axis_angle]), '1 -> b', b=len(tilt_angles)
    )
    euler_angles = einops.rearrange(
        [tilt_angles, in_plane_rotation_angles], pattern="angles b -> b angles"
    )
    rotation_matrices = R.from_euler(seq='yz', angles=euler_angles, degrees=True).inv().as_matrix()
    rotation_matrices = torch.tensor(rotation_matrices).float()

    # make tilt series
    tilt_series = project_3d_to_2d(volume, rotation_matrices=rotation_matrices)

    # shift resulting images in plane
    shifts = torch.tensor(shifts).float()
    tilt_series = fourier_shift_image_2d(tilt_series, shifts=shifts)
    return tilt_series


if __name__ == "__main__":
    # simulate volume with a few points in the xy plane
    volume = simulate_volume()

    # setup tilt series geometry
    tilt_axis_angle = 85
    tilt_angles = np.linspace(-60, 60, num=61, endpoint=True)
    rng = np.random.default_rng(1414)
    shifts = rng.uniform(low=-5, high=5, size=(len(tilt_angles), 2))
    shifts[tilt_angles == 0] = 0

    # simulate tilt series
    tilt_series = simulate_tilt_series(volume, tilt_axis_angle, tilt_angles, shifts)

    # run torch-tiltxcorr and apply shifts
    shifts = tiltxcorr(
        tilt_series=tilt_series,
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        low_pass_cutoff=0.5
    )
    shifted_tilt_series = fourier_shift_image_2d(tilt_series, shifts=shifts)

    # visiualize results
    viewer = napari.Viewer()
    viewer.add_image(tilt_series)
    viewer.add_image(shifted_tilt_series)
    napari.run()
