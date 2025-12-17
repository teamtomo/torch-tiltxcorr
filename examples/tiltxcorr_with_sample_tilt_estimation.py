# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "einops",
#   "numpy",
#   "torch",
#   "torch-affine-utils",
#   "torch-image-interpolation",
#   "torch-fourier-slice",
#   "torch-fourier-shift",
#   "torch-fourier-filter",
#   "scipy",
#   "torch-tiltxcorr",
# ]
# exclude-newer = "2025-12-20T00:00:00Z"
# [tool.uv.sources]
# torch-tiltxcorr = { path = "../" }
# ///
"""
Example: tiltxcorr_with_sample_tilt_estimation with simulated data

This example demonstrates how to use tiltxcorr_with_sample_tilt_estimation to estimate
the component of the sample tilt angle around the tilt axis.

The sample tilt represents the physical angle at which the sample sits
in the microscope. The effective stage tilt angle seen by the beam is:
true_tilt_angle = nominal_stage_angle + sample_tilt
"""
import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch_affine_utils.transforms_3d import Rx, Ry, Rz
from torch_affine_utils.utils import homogenise_coordinates
from torch_image_interpolation import insert_into_image_3d
from torch_fourier_slice import project_3d_to_2d
from torch_fourier_shift import fourier_shift_image_2d
import napari

from torch_tiltxcorr import tiltxcorr_with_sample_tilt_estimation


def generate_tilted_plane_tilt_series(
    sample_tilt_x: float = 0.0,
    sample_tilt_y: float = 0.0,
    d: int = 128,
    n_points_on_plane: int = 100,
    tilt_angles_deg: torch.Tensor | None = None,
    tilt_axis_angle: float = 85.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic tilt series from points on a tilted plane.

    Parameters
    ----------
    sample_tilt_x : float
        Sample tilt around x-axis in degrees
    sample_tilt_y : float
        Sample tilt around y-axis in degrees
    d : int
        Volume dimension (cubic volume d x d x d)
    n_points_on_plane : int
        Number of random points to place on the plane
    tilt_angles_deg : torch.Tensor, optional
        Tilt angles in degrees. If None, uses linspace(-60, 60, 41)
    tilt_axis_angle : float
        Tilt axis angle in degrees
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tilt_series : torch.Tensor
        Generated tilt series (n_tilts, h, w)
    tilt_angles : torch.Tensor
        Tilt angles in degrees (n_tilts,)
    """
    # Setup dimensions
    d2 = d // 2
    volume_center_zyx = torch.tensor([d2, d2, d2]).float()

    # Setup plane geometry - sample tilt in microscope
    M_tilt_sample_in_scope = Ry(sample_tilt_y, zyx=True) @ Rx(sample_tilt_x, zyx=True)

    # Generate points on a tilted xy plane in the microscope
    # Create regular grid across xy
    y = torch.linspace(-d2, d2, steps=d)
    x = torch.linspace(-d2, d2, steps=d)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yx = einops.rearrange([yy, xx], "yx h w -> h w yx")
    zyx = F.pad(yx, (1, 0), value=0)
    xy_plane_zyxw = homogenise_coordinates(zyx)
    xy_plane_zyxw_col = einops.rearrange(xy_plane_zyxw, "h w zyxw -> (h w) zyxw 1")

    # Randomly sample points from grid
    b = len(xy_plane_zyxw_col)
    rng = np.random.default_rng(seed=seed)
    idx_subset = rng.choice(b, size=(n_points_on_plane,), replace=False)

    # Apply sample tilt transformation to points
    points_in_scope_zyxw_col = xy_plane_zyxw_col[idx_subset]
    points_in_scope_zyxw_col = M_tilt_sample_in_scope @ points_in_scope_zyxw_col
    points_in_scope_zyxw = einops.rearrange(points_in_scope_zyxw_col, "b zyxw 1 -> b zyxw")
    points_in_scope_zyx = points_in_scope_zyxw[..., :3]

    # Create 3D volume with points on the tilted plane
    points_in_volume = points_in_scope_zyx + volume_center_zyx
    volume_in_scope = torch.zeros((d, d, d)).float()
    volume_in_scope, _ = insert_into_image_3d(
        values=torch.ones((n_points_on_plane,)),
        coordinates=points_in_volume,
        image=volume_in_scope
    )

    # Setup tilt angles
    if tilt_angles_deg is None:
        tilt_angles_deg = torch.linspace(-60, 60, steps=41)

    # Setup scope2detector transformation
    M_scope2detector = Rz(tilt_axis_angle, zyx=True) @ Ry(tilt_angles_deg, zyx=True)

    # Generate tilt series
    M_scope2detector_rot = M_scope2detector[:, :3, :3]
    M_scope2detector_rot_inv = torch.linalg.pinv(M_scope2detector_rot)
    tilt_series = project_3d_to_2d(
        volume=volume_in_scope,
        rotation_matrices=M_scope2detector_rot_inv,
        zyx_matrices=True,
    )

    return tilt_series, tilt_angles_deg


if __name__ == "__main__":
    print("=" * 70)
    print("Simulated Tilt Series with Sample Tilt")
    print("=" * 70)
    print()

    # Setup parameters
    true_sample_tilt_y = 15.0  # Ground truth: sample is tilted +15 degrees around Y-axis
    d = 128
    tilt_axis_angle = 85.0
    tilt_angles_deg = torch.linspace(-60, 60, steps=41)
    low_pass_cutoff = 0.5

    print(f"Simulation parameters:")
    print(f"  True sample tilt (Y-axis): {true_sample_tilt_y}°")
    print(f"  Volume size: {d}³")
    print(f"  Tilt axis angle: {tilt_axis_angle}°")
    print(f"  Tilt angles: {len(tilt_angles_deg)} images from {tilt_angles_deg.min()}° to {tilt_angles_deg.max()}°")
    print(f"  Low-pass cutoff: {low_pass_cutoff} cycles/px")
    print()

    # Simulate tilt series with known sample tilt
    print("Simulating tilt series with tilted plane...")
    tilt_series, tilt_angles = generate_tilted_plane_tilt_series(
        sample_tilt_x=0.0,
        sample_tilt_y=true_sample_tilt_y,
        d=d,
        n_points_on_plane=100,
        tilt_angles_deg=tilt_angles_deg,
        tilt_axis_angle=tilt_axis_angle,
        seed=42,
    )
    print(f"  Tilt series shape: {tilt_series.shape}")
    print()

    # Run tiltxcorr_with_sample_tilt to recover the sample tilt
    print("Running tiltxcorr_with_sample_tilt...")
    print("  Searching for optimal sample tilt angle...")
    shifts, estimated_sample_tilt = tiltxcorr_with_sample_tilt_estimation(
        tilt_series=tilt_series,
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        low_pass_cutoff=low_pass_cutoff,
        sample_tilt_range=(-30.0, 30.0),
        max_iter=15,
    )
    print()

    # Display results
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  Ground truth sample tilt (Y-axis):      {true_sample_tilt_y:6.2f}°")
    print(f"  Estimated sample tilt:                  {estimated_sample_tilt:6.2f}°")
    print(f"  Error:                                  {abs(estimated_sample_tilt - true_sample_tilt_y):6.2f}°")
    print()
    print(f"  Ground truth shifts:                     all 0")
    print(f"  Mean estimated shift magnitude:        {torch.norm(shifts, dim=1).mean():6.2f} pixels")
    print(f"  Max estimated shift magnitude:         {torch.norm(shifts, dim=1).max():6.2f} pixels")
    print()