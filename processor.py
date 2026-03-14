"""
processor.py
------------
Implements the two core processing steps of the cavity segmentation pipeline:

  1. **Filtering** – mirrors the logic in ``filternii.py`` from the original
     ``cavity-nnunet-pytorch`` project.  Voxels that lie *outside* the lung
     mask are set to -1024 HU so the downstream segmentation stage can safely
     ignore them.

  2. **Segmentation** – a simple intensity-threshold approach that approximates
     what the trained nnU-Net neural network would do.  Voxels with HU values
     between ``cavity_hu_low`` and ``cavity_hu_high`` that are *inside* the
     lung mask are labelled as potential cavity voxels.

     The default window (-900 to -700 HU) targets low-density regions that
     are characteristic of air-filled mycobacterial cavities.
"""

import numpy as np


# HU value assigned to non-lung regions after filtering (matches original project)
OUTSIDE_LUNG_HU = -1024


def apply_lung_filter(ct_array, lung_mask):
    """Set voxels outside the lung mask to OUTSIDE_LUNG_HU.

    This replicates the behaviour of ``filternii.py`` in the original project,
    which ensures the neural network (or, here, the threshold segmenter) only
    "sees" relevant lung tissue.

    Parameters
    ----------
    ct_array : np.ndarray
        Raw CT scan in Hounsfield Units.
    lung_mask : np.ndarray of bool
        Boolean mask; True where lung parenchyma is present.

    Returns
    -------
    filtered : np.ndarray, dtype float32
        A copy of ``ct_array`` with non-lung voxels set to OUTSIDE_LUNG_HU.
    """
    filtered = ct_array.copy().astype(np.float32)
    filtered[~lung_mask] = OUTSIDE_LUNG_HU
    return filtered


def segment_cavities(filtered_ct, lung_mask, cavity_hu_low=-900, cavity_hu_high=-700):
    """Identify potential cavity voxels via intensity thresholding.

    In the full pipeline a trained U-Net performs this step.  Here we use a
    straightforward HU window: cavity voxels are typically very low-density
    (air-filled) regions *within* the lung parenchyma.

    Parameters
    ----------
    filtered_ct : np.ndarray
        Filtered CT array (output of :func:`apply_lung_filter`).
    lung_mask : np.ndarray of bool
        Boolean mask; True where lung parenchyma is present.
    cavity_hu_low : float
        Lower HU bound of the cavity window (inclusive).
    cavity_hu_high : float
        Upper HU bound of the cavity window (inclusive).

    Returns
    -------
    segmentation : np.ndarray, dtype bool
        Boolean array; True where a cavity voxel was detected.
    """
    in_window = (filtered_ct >= cavity_hu_low) & (filtered_ct <= cavity_hu_high)
    segmentation = in_window & lung_mask
    return segmentation


if __name__ == "__main__":
    from generate_mock_data import generate_mock_ct_and_mask

    ct, mask, _ = generate_mock_ct_and_mask()
    filtered = apply_lung_filter(ct, mask)
    seg = segment_cavities(filtered, mask)

    print(f"Filtered CT – non-lung set to : {OUTSIDE_LUNG_HU} HU")
    print(f"Unique values outside mask    : {np.unique(filtered[~mask])}")
    print(f"Cavity voxels detected        : {seg.sum():,}")
