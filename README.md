# WiseSpine Baseline Segmentation & Evaluation Pipeline

This repository contains scripts for:

- Running vertebra segmentation models  
- Reconstructing full spine volumes from individual masks  
- Evaluating predictions using DICE and IoU metrics  

The current baseline implementation supports:

- TotalSegmentator  
- TotalSpineSeg (optional, if installed)

The pipeline is configured for the VerSe20 training dataset.

---

## 📂 Directory Structure

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
    │   └── evaluation_metrics.csv
    │
    └── scripts/
        ├── run_segmentation.py
        ├── reconstruct_spine.py
        └── evaluate_segmentation.py
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

**Script:** `run_segmentation.py`

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
python run_segmentation.py
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

# Pipeline Overview

```
Step 1 → Run segmentation model
Step 2 → Rename & organize masks
Step 3 → Reconstruct full spine
Step 4 → Evaluate against ground truth
Step 5 → Export metrics CSV
```

---

# Evaluation Details

For each case:

## Spine-level Metrics
- DICE (union spine vs GT spine)
- IoU

## Vertebra-level Metrics
- DICE per vertebra
- IoU per vertebra

Missing labels are recorded as `NaN`.

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

The rest of the pipeline (reconstruction + evaluation) will work automatically.

---

# Summary

This pipeline provides:

- Automated batch segmentation
- Structured output management
- Spine reconstruction
- Quantitative evaluation
- CSV export of metrics

It is modular, extendable, and ready for benchmarking additional spine segmentation models.
