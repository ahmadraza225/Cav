# Cav – Lung Cavity Segmentation MVP

A **Minimum Viable Product (MVP)** that demonstrates the core workflow of the
[`cavity-nnunet-pytorch`](https://github.com/josephnw/cavity-nnunet-pytorch)
project **without** requiring deep learning, a GPU, or heavy medical-imaging
libraries. It is designed for presentations and stakeholder walkthroughs.

---

## Overview

The full `cavity-nnunet-pytorch` pipeline automatically detects and
volumetrically quantifies mycobacterial cavities in chest CT scans using a
trained [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) neural network.

This MVP replaces the neural network with a **simple intensity threshold**
while preserving the same logical data flow:

```
Input CT scan
     │
     ▼
[ Filter ]   – set non-lung voxels to -1024 HU   (processor.py)
     │
     ▼
[ Segment ]  – threshold: -900 to -700 HU         (processor.py)
     │
     ▼
[ Quantify ] – count voxels × voxel volume (mm³)  (quantifier.py)
     │
     ▼
Summary report
```

| MVP component | Full DL equivalent |
|---|---|
| `generate_mock_data.py` – NumPy synthetic CT | Real NIfTI CT scan from DICOM |
| `processor.apply_lung_filter()` | `filternii.py` HU masking (identical logic) |
| `processor.segment_cavities()` HU threshold | Trained nnU-Net inference |
| `quantifier.calculate_volume()` | Same voxel-counting formula |

---

## Project Structure

```
Cav/
├── generate_mock_data.py  # Synthetic 3-D CT scan + lung mask generator
├── processor.py           # Lung filter + threshold segmentation
├── quantifier.py          # Voxel volume calculation
├── main.py                # Pipeline orchestrator & report printer
└── README.md              # This file
```

---

## Requirements

Only [NumPy](https://numpy.org/) is required:

```bash
pip install numpy
```

---

## How to Run

```bash
python main.py
```

### Expected output (truncated)

```
====================================================
  Lung Cavity Segmentation MVP – Pipeline Report
====================================================

Step 1 › Generating synthetic CT scan …
  Scan shape             : (64, 64, 64)
  Total voxels           : 262,144
  Lung mask voxels       : 14,832  (5.7% of scan)
  CT HU range (raw)      : [-1000.0, 184.3]

Step 2 › Applying lung filter …
  Non-lung regions set to: -1024 HU

Step 3 › Segmenting cavities …
  Cavity HU window       : [-900, -700] HU
  Cavity voxels found    : 452  (3.05% of lung)

Step 4 › Quantifying cavity volume …
── Volume Quantification ─────────────────────────
  Cavity voxels detected : 452
  Voxel volume           : 1.0000 mm³
  Total cavity volume    : 452.00 mm³
                         : 0.4520 cm³
──────────────────────────────────────────────────

====================================================
  FINAL REPORT
====================================================
  Cavity volume          : 452.00 mm³  (0.4520 cm³)
  Cavity / Lung ratio    : 3.05%
====================================================
```

---

## How It Relates to the Full Implementation

| Aspect | MVP | Full pipeline |
|---|---|---|
| **Data source** | Synthetic NumPy array | Real patient NIfTI files |
| **Lung mask** | Simple ellipsoid formula | Automated lung segmentation |
| **Cavity detection** | HU intensity threshold | nnU-Net deep learning model |
| **Volume calculation** | Identical formula | Identical formula |
| **Dependencies** | `numpy` only | PyTorch, nnU-Net, nibabel, … |
| **GPU required** | No | Yes (recommended) |

The filtering logic in `processor.apply_lung_filter()` is a direct Python
equivalent of the `filternii.py` script in `cavity-nnunet-pytorch`.  The
volume calculation in `quantifier.py` uses the same voxel-counting method
that the full pipeline applies after neural-network inference.

---

## License

See [LICENSE](LICENSE) for details.