# WiseSpine Baseline Segmentation, Reconstruction & Spacing Evaluation Pipeline

This repository contains scripts for:

- Running vertebra segmentation models  
- Reconstructing full spine volumes from individual masks  
- Evaluating predictions using DICE, IoU, and vertebral spacing metrics  

The current baseline implementation supports:

- TotalSegmentator  
- TotalSpineSeg (optional, if installed)

The pipeline is configured for the VerSe20 training dataset.

---

## Directory Structure

```
wisespine/
├── SAM-Med3D/
├── TotalSegmentator/
├── nnUNet/
├── totalspineseg/
├── data/
│   └── Verse20/
│       └── dataset-01training/
│           ├── rawdata/
│           │   └── sub-xxx/
│           │       └── sub-xxx_dir-ax_ct.nii.gz
│           │
│           └── derivatives/
│               └── sub-xxx/
│                   ├── sub-xxx_dir-ax_seg-subreg_ctd.json
│                   └── sub-xxx_dir-ax_seg-vert_msk.nii.gz
│
└── wisespine_new/
    ├── baseline_outputs/
    │   ├── models/
    │   │   └── TotalSegmentator/
    │   │       └── sub-xxx/
    │   │           ├── sub-xxx_TotalSegmentator_vertebrae_*.nii.gz
    │   │           └── sub-xxx_TotalSegmentator_sacrum.nii.gz
    │   │
    │   ├── reconstructed/
    │   │   └── TotalSegmentator/
    │   │       └── sub-xxx/
    │   │           └── sub-xxx_TotalSegmentator_spine_union.nii.gz
    │   │
    │   ├── centroids/
    │   │   └── TotalSegmentator/
    │   │       └── sub-xxx/
    │   │           └── sub-xxx_TotalSegmentator_centroids.json
    │   │
    │   ├── evaluation_metrics.csv
    │   ├── spacings.csv
    │   └── spacings_errors.csv
    │
    └── scripts/
        ├── segmentation.py
        ├── reconstruct_spine.py
        ├── evaluate_segmentation.py
        ├── spacings.py
        └── evaluate_spacings.py
```

---

# Requirements

## Python Packages

```bash
pip install nibabel numpy pandas
```

For segmentation:

```bash
pip install TotalSegmentator
```

Optional:

```bash
pip install totalspineseg
```

---

# Segmentation

**Script:** `segmentation.py`

This script:

- Detects training cases automatically (or uses a manual list)
- Runs segmentation per case
- Renames output masks into standardized format

## Configuration Options

```python
RUN_ALL_TRAINING_CASES = True
CLEAR_EXISTING = False
MODEL_NAMES = ["TotalSegmentator"]
```

### Important Parameters

| Parameter | Description |
|-----------|-------------|
| `RUN_ALL_TRAINING_CASES` | Auto-detect all sub-* folders |
| `CLEAR_EXISTING` | Delete previous masks before running |
| `TS_TASK` | Set to `"vertebrae_mr"` |
| `TS_NR_THR_RESAMP` | Resampling threads |
| `TS_NR_THR_SAVING` | Saving threads |

## Run

```bash
python segmentation.py
```

Outputs saved to:

```
baseline_outputs/models/<MODEL_NAME>/<CASE>/
```

---

# Spine Reconstruction

**Script:** `reconstruct_spine.py`

This script:

- Combines individual vertebra masks
- Optionally preserves vertebra labels
- Saves a single union spine file

## Configuration

```python
PRESERVE_LABELS = True
CLEAR_EXISTING = False
```

### Output

```
sub-xxx_TotalSegmentator_spine_union.nii.gz
```

If `PRESERVE_LABELS = True`:
- Each vertebra receives a unique integer label

If `False`:
- Entire spine is binary

## Run

```bash
python reconstruct_spine.py
```

---

# Evaluation Script

**Script:** `evaluate_segmentation.py`

This script:

- Computes overall spine reconstruction metrics
- Computes per-vertebra metrics
- Saves results to CSV

## Metrics

- DICE coefficient  
- IoU (Intersection over Union)

### Label Mapping

The script maps:

- C1–C7 → labels 1–7  
- T1–T12 → labels 8–19  
- L1–L5 → labels 20–24  
- S1 → label 25  

## Run

```bash
python evaluate_segmentation.py
```

Output:

```
baseline_outputs/evaluation_metrics.csv
```

---

# Vertebral Spacing Analysis

This section adds centroid extraction and inter-vertebra spacing evaluation.

---

## 1. `spacings.py`

This script:

- Extracts vertebra centroids from predicted masks  
- Converts voxel coordinates to world coordinates (mm)  
- Saves JSON files containing centroids per case

### Configuration

```python
MODEL_NAME = "TotalSegmentator"

PRED_BASE = "/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/models"
OUTPUT_BASE = "/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/centroids"

VERTEBRA_ORDER = [
    "C1","C2","C3","C4","C5","C6","C7",
    "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
    "L1","L2","L3","L4","L5"
]
```

### Run

```bash
python spacings.py
```

### Output

```
baseline_outputs/centroids/<MODEL_NAME>/<CASE>/<CASE>_<MODEL_NAME>_centroids.json
```

---

## 2. `evaluate_spacings.py`

This script:

- Loads ground truth and predicted centroids  
- Computes inter-vertebra distances (gaps) in mm  
- Computes absolute error (mm) between predicted and ground truth spacings  
- Saves results to CSV files

### Configuration

```python
MODEL_NAME = "TotalSegmentator"

GT_BASE = "/gscratch/scrubbed/bhan830/wisespine/data/Verse20/dataset-01training/derivatives"
PRED_BASE = "/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/centroids/TotalSegmentator"
OUTPUT_BASE = "/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs"

CASES = ["sub-gl003", "sub-gl016", "sub-gl047", "sub-gl090", "sub-gl124"]

VERTEBRAE = [
    "C1","C2","C3","C4","C5","C6","C7",
    "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
    "L1","L2","L3","L4","L5"
]

SPACING_LABELS = [f"{VERTEBRAE[i]}-{VERTEBRAE[i+1]}" for i in range(len(VERTEBRAE)-1)]
```

### Run

```bash
python evaluate_spacings.py
```

### Output

1. **Predicted spacings CSV**

```
baseline_outputs/spacings.csv
```

2. **Absolute error CSV**

```
baseline_outputs/spacings_errors.csv
```

---

## Methodology

1. **Centroid Extraction (`spacings.py`)**  
   - Load individual vertebra mask  
   - Compute voxel centroid  
   - Convert to world coordinates (mm) using affine transformation  

2. **Spacing Computation (`evaluate_spacings.py`)**  
   - Compute Euclidean distance between successive vertebra centroids  
   - Convert voxel distances to mm using voxel size from CT header  

3. **Error Computation**  
   - Absolute error: `|predicted - ground_truth|`  

---

# Pipeline Overview

```
Step 1 → Run segmentation model
Step 2 → Rename & organize masks
Step 3 → Reconstruct full spine
Step 4 → Extract vertebra centroids
Step 5 → Compute spacings & errors
Step 6 → Evaluate against ground truth (DICE & IoU)
Step 7 → Export metrics CSV
```

---

# Reproducibility Notes

- Ensure input CT filenames follow:
  ```
  sub-xxx_dir-ax_ct.nii.gz
  ```
- Ensure GT mask naming:
  ```
  sub-xxx_dir-ax_seg-vert_msk.nii.gz
  ```
- Scripts assume VerSe20 directory structure.

---

# Extending to New Models

To add a new segmentation model:

1. Add model name to:

```python
MODEL_NAMES = ["TotalSegmentator", "NewModel"]
```

2. Implement a new function:

```python
def run_new_model(...):
    pass
```

3. Follow the naming convention:

```
{case}_{model}_vertebrae_*.nii.gz
```

The rest of the pipeline (reconstruction + evaluation + spacing) will work automatically.

---

# Summary

This pipeline provides:

- Automated batch segmentation
- Structured output management
- Spine reconstruction
- Centroid extraction & spacing analysis
- Quantitative evaluation (DICE, IoU, spacing errors)
- CSV export of metrics

It is modular, extendable, and ready for benchmarking additional spine segmentation models.
