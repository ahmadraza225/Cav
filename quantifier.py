"""
quantifier.py
-------------
Calculates the physical volume of segmented cavity voxels.

In a real NIfTI-based pipeline the voxel spacing is read directly from the
image header (e.g. via ``nibabel``).  Here we accept the spacing as a plain
Python tuple so the module remains dependency-free.

Volume formula
~~~~~~~~~~~~~~
  volume_mm3 = number_of_cavity_voxels × voxel_volume_mm3

where
  voxel_volume_mm3 = spacing_z_mm × spacing_y_mm × spacing_x_mm
"""


def calculate_volume(segmentation, voxel_spacing_mm=(1.0, 1.0, 1.0)):
    """Return the total cavity volume in cubic millimetres.

    Parameters
    ----------
    segmentation : np.ndarray of bool
        Boolean segmentation mask produced by
        :func:`processor.segment_cavities`.
    voxel_spacing_mm : tuple of float
        Physical size of a single voxel as (z_mm, y_mm, x_mm).
        Defaults to 1 mm isotropic.

    Returns
    -------
    volume_mm3 : float
        Total volume of ``True`` voxels in mm³.
    num_voxels : int
        Raw count of cavity voxels.
    voxel_volume_mm3 : float
        Volume of a single voxel in mm³.
    """
    sz, sy, sx = voxel_spacing_mm
    voxel_volume_mm3 = sz * sy * sx
    num_voxels = int(segmentation.sum())
    volume_mm3 = num_voxels * voxel_volume_mm3
    return volume_mm3, num_voxels, voxel_volume_mm3


def volume_summary(volume_mm3, num_voxels, voxel_volume_mm3):
    """Return a human-readable summary string for the volume measurement.

    Parameters
    ----------
    volume_mm3 : float
    num_voxels : int
    voxel_volume_mm3 : float

    Returns
    -------
    str
    """
    volume_cm3 = volume_mm3 / 1000.0
    lines = [
        "── Volume Quantification ─────────────────────────",
        f"  Cavity voxels detected : {num_voxels:,}",
        f"  Voxel volume           : {voxel_volume_mm3:.4f} mm³",
        f"  Total cavity volume    : {volume_mm3:,.2f} mm³",
        f"                         : {volume_cm3:.4f} cm³",
        "──────────────────────────────────────────────────",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from generate_mock_data import generate_mock_ct_and_mask
    from processor import apply_lung_filter, segment_cavities

    ct, mask, spacing = generate_mock_ct_and_mask()
    filtered = apply_lung_filter(ct, mask)
    seg = segment_cavities(filtered, mask)

    vol, nvox, vvol = calculate_volume(seg, spacing)
    print(volume_summary(vol, nvox, vvol))
