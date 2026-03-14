"""
generate_mock_data.py
---------------------
Generates a synthetic 3D "CT scan" represented as NumPy arrays with realistic
Hounsfield Unit (HU) values and a companion lung mask.

HU reference ranges used:
  -1000 HU : air (outside body / trachea)
      0 HU : water / soft tissue
   -500 HU : normal lung parenchyma
   -800 HU : cavity-like low-density region inside the lung
"""

import numpy as np


def generate_mock_ct_and_mask(shape=(64, 64, 64), seed=42):
    """Create a synthetic 3D CT scan and a corresponding lung mask.

    The volume is divided into three conceptual zones:
      * Background (outside the body): ~-1000 HU
      * Lung parenchyma (the mask region): ~-500 HU with noise
      * Simulated cavity pockets inside the lung: ~-800 HU

    Parameters
    ----------
    shape : tuple of int
        (depth, height, width) dimensions of the synthetic volume.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ct_array : np.ndarray, dtype float32
        Synthetic CT scan in Hounsfield Units.
    lung_mask : np.ndarray, dtype bool
        Boolean mask where True indicates lung parenchyma.
    voxel_spacing_mm : tuple of float
        Physical size of each voxel in millimetres (z, y, x).
    """
    rng = np.random.default_rng(seed)

    # Start with an air-filled volume (-1000 HU)
    ct_array = np.full(shape, -1000.0, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Define a simple ellipsoidal lung region (one lobe for simplicity)   #
    # ------------------------------------------------------------------ #
    cz, cy, cx = [s // 2 for s in shape]
    rz, ry, rx = shape[0] // 3, shape[1] // 3, shape[2] // 3

    z_idx, y_idx, x_idx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    lung_mask = (
        ((z_idx - cz) / rz) ** 2
        + ((y_idx - cy) / ry) ** 2
        + ((x_idx - cx) / rx) ** 2
    ) <= 1.0

    # Fill lung parenchyma with ~-500 HU + Gaussian noise
    lung_hu = rng.normal(loc=-500.0, scale=50.0, size=shape).astype(np.float32)
    ct_array[lung_mask] = lung_hu[lung_mask]

    # ------------------------------------------------------------------ #
    # Carve out two small spherical "cavity" pockets inside the lung      #
    # ------------------------------------------------------------------ #
    cavity_centres = [
        (cz - rz // 3, cy - ry // 3, cx - rx // 3),
        (cz + rz // 3, cy + ry // 3, cx + rx // 3),
    ]
    cavity_radius = max(2, min(shape) // 12)

    for ccz, ccy, ccx in cavity_centres:
        cavity_mask = (
            ((z_idx - ccz) ** 2 + (y_idx - ccy) ** 2 + (x_idx - ccx) ** 2)
            <= cavity_radius ** 2
        )
        cavity_mask &= lung_mask  # cavities only exist inside the lung
        cavity_hu = rng.normal(loc=-800.0, scale=30.0, size=shape).astype(np.float32)
        ct_array[cavity_mask] = cavity_hu[cavity_mask]

    voxel_spacing_mm = (1.0, 1.0, 1.0)  # isotropic 1 mm³ voxels

    return ct_array, lung_mask, voxel_spacing_mm


if __name__ == "__main__":
    ct, mask, spacing = generate_mock_ct_and_mask()
    print(f"CT array shape      : {ct.shape}")
    print(f"CT value range      : [{ct.min():.1f}, {ct.max():.1f}] HU")
    print(f"Lung mask voxels    : {mask.sum():,}")
    print(f"Voxel spacing (mm)  : {spacing}")
