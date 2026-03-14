"""
main.py
-------
Entry point for the Lung Cavity Segmentation MVP.

Orchestrates the full pipeline:
  1. Generate  – create a synthetic 3D CT scan and lung mask
  2. Filter    – isolate lung parenchyma (non-lung → -1024 HU)
  3. Segment   – threshold-based cavity detection (-900 to -700 HU)
  4. Quantify  – compute cavity volume in mm³ and cm³
  5. Report    – print a structured summary to stdout

Run
---
    python main.py
"""

from generate_mock_data import generate_mock_ct_and_mask
from processor import apply_lung_filter, segment_cavities, OUTSIDE_LUNG_HU
from quantifier import calculate_volume, volume_summary

# ── Pipeline configuration ────────────────────────────────────────────────── #
CT_SHAPE = (64, 64, 64)          # synthetic scan dimensions (z, y, x)
RANDOM_SEED = 42                 # for reproducibility
CAVITY_HU_LOW = -900             # lower bound of cavity HU window
CAVITY_HU_HIGH = -700            # upper bound of cavity HU window


def run_pipeline():
    """Execute the end-to-end cavity segmentation MVP pipeline."""

    # ── Step 1 : Generate synthetic data ────────────────────────────────── #
    print("=" * 52)
    print("  Lung Cavity Segmentation MVP – Pipeline Report")
    print("=" * 52)
    print()
    print("Step 1 › Generating synthetic CT scan …")
    ct_array, lung_mask, voxel_spacing = generate_mock_ct_and_mask(
        shape=CT_SHAPE, seed=RANDOM_SEED
    )
    total_voxels = ct_array.size
    lung_voxels = int(lung_mask.sum())
    print(f"  Scan shape             : {ct_array.shape}")
    print(f"  Total voxels           : {total_voxels:,}")
    print(f"  Lung mask voxels       : {lung_voxels:,}  "
          f"({lung_voxels / total_voxels * 100:.1f}% of scan)")
    print(f"  CT HU range (raw)      : [{ct_array.min():.1f}, {ct_array.max():.1f}]")
    print()

    # ── Step 2 : Apply lung filter ───────────────────────────────────────── #
    print("Step 2 › Applying lung filter …")
    print(f"  Non-lung regions set to: {OUTSIDE_LUNG_HU} HU")
    filtered_ct = apply_lung_filter(ct_array, lung_mask)
    print(f"  CT HU range (filtered) : [{filtered_ct.min():.1f}, "
          f"{filtered_ct.max():.1f}]")
    print()

    # ── Step 3 : Segment cavities ────────────────────────────────────────── #
    print("Step 3 › Segmenting cavities …")
    print(f"  Cavity HU window       : [{CAVITY_HU_LOW}, {CAVITY_HU_HIGH}] HU")
    segmentation = segment_cavities(
        filtered_ct, lung_mask,
        cavity_hu_low=CAVITY_HU_LOW,
        cavity_hu_high=CAVITY_HU_HIGH,
    )
    cavity_voxels = int(segmentation.sum())
    print(f"  Cavity voxels found    : {cavity_voxels:,}  "
          f"({cavity_voxels / lung_voxels * 100:.2f}% of lung)")
    print()

    # ── Step 4 : Quantify volume ─────────────────────────────────────────── #
    print("Step 4 › Quantifying cavity volume …")
    volume_mm3, num_voxels, voxel_vol = calculate_volume(segmentation, voxel_spacing)
    print(volume_summary(volume_mm3, num_voxels, voxel_vol))
    print()

    # ── Step 5 : Final summary ───────────────────────────────────────────── #
    print("=" * 52)
    print("  FINAL REPORT")
    print("=" * 52)
    print(f"  Scan dimensions        : {ct_array.shape[0]} × "
          f"{ct_array.shape[1]} × {ct_array.shape[2]} voxels")
    print(f"  Voxel spacing          : {voxel_spacing[0]} × "
          f"{voxel_spacing[1]} × {voxel_spacing[2]} mm")
    print(f"  Lung volume (approx.)  : {lung_voxels:,} mm³  "
          f"({lung_voxels / 1000:.2f} cm³)")
    print(f"  Cavity volume          : {volume_mm3:,.2f} mm³  "
          f"({volume_mm3 / 1000:.4f} cm³)")
    print(f"  Cavity / Lung ratio    : "
          f"{cavity_voxels / lung_voxels * 100:.2f}%")
    print("=" * 52)


if __name__ == "__main__":
    run_pipeline()
