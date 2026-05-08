# test_cobb.py

Spine segmentation and Cobb angle validation pipeline for VerSe20 CT data using TotalSegmentator masks.

---

## Overview

This script evaluates how well TotalSegmentator segments individual vertebrae compared to VerSe20 ground truth masks, and whether those segmentation errors propagate into Cobb angle measurements. It does **not** compute a clinical Cobb angle — it validates the quality of the segmentation pipeline that feeds into Cobb angle computation.

---

## Requirements

```bash
pip install nibabel numpy pandas
```

Also requires `segmentation.py` (provides `get_case_list()`) and `cobb_angle_pipeline.py` (or symlinked as `cobb_debug_full.py`) in the same directory.

---

## Usage

**Single case:**
```bash
python test_cobb.py sub-gl003
```

**Full dataset:**
```bash
python test_cobb.py
```

Output is saved to:
```
/gpfs/projects/wisemind/bhan830/wisespine/wisespine_new/baseline_outputs/evaluation_metrics.csv
```

---

## Input Data Structure

```
derivatives/
└── sub-gl003/
    ├── sub-gl003_dir-ax_seg-vert_msk.nii.gz   # GT label volume (VerSe20)
    └── sub-gl003_dir-ax_seg-subreg_ctd.json    # Expert centroid annotations

models/TotalSegmentator/
└── sub-gl003/
    └── sub-gl003_TotalSegmentator_vertebrae_T1.nii.gz   # Individual vertebra masks

reconstructed/TotalSegmentator/
└── sub-gl003/
    └── sub-gl003_TotalSegmentator_spine_union.nii.gz    # Union of all vertebra masks
```

---

## Validation Layers

### Layer 1 — Whole-Spine Reconstruction DICE / IoU
Compares the union of all TotalSegmentator vertebra masks against the binarized VerSe20 GT mask. Measures overall spine coverage. Both volumes are canonicalized to RAS orientation before comparison to avoid axis mismatch.

### Layer 2 — Per-Vertebra DICE / IoU
Each individual TotalSegmentator mask is compared against the corresponding label extracted from the GT label volume. More informative than whole-spine DICE because it isolates quality per vertebra — the two vertebrae used in a Cobb measurement need high individual DICE.

### Layer 3A — Centroid Displacement vs GT Mask
For each vertebra, the physical-space centroid of the TotalSegmentator mask is compared against the centroid of the VerSe GT label. Both are computed in world RAS mm using `affine @ [mean_vox, 1]`. Low displacement confirms TotalSegmentator places vertebrae in the correct spatial location.

> ⚠️ Centroids are **not** computed as `voxel_index × spacing` — that omits the image origin translation and causes ~200mm errors.

### Layer 3B — Centroid Displacement vs VerSe Expert Annotations
Same TotalSegmentator centroid compared against the expert-annotated vertebral body centres from the VerSe centroid JSON. Expert centroids are converted from voxel space to world mm using the original NIfTI affine. Expected offset is ~10mm because expert annotations mark the vertebral body centre while TotalSegmentator mask centroids are pulled posteriorly by the spinous process and lamina.

### Layer 4 — Endplate Normal Angular Error
The top 25% of voxels by Z position are extracted as the superior endplate point cloud; bottom 25% as the inferior endplate. SVD is applied to each cloud to fit a plane. The angle between the TotalSegmentator normal and the GT normal is reported. This is the most direct proxy for Cobb angle accuracy — endplate orientation is exactly what determines the Cobb angle.

### Layer 5 — Centroid-Based Curvature (VerSe Expert)
For each consecutive pair of vertebrae with expert centroid annotations, computes the angle between the vector connecting the two centroids and the vertical axis (Z in RAS), projected onto the coronal plane. This is an independent geometric reference derived purely from VerSe expert annotations, used to cross-validate endplate-based Cobb angles.

### Layer 6 — Pseudo-GT Cobb Angles
The Cobb angle pipeline is run on both GT masks and TotalSegmentator masks for every consecutive vertebra pair. The difference between the two angles is the key metric: it directly measures whether segmentation errors change the computed Cobb angle. All differences below 5° are considered acceptable.

### Layer 7 — Cross-Validation: Endplate Cobb vs Centroid Curvature
Compares the endplate-based Cobb angle (from both GT and pred masks) against the centroid-based curvature from VerSe expert annotations.

| Pattern | Interpretation |
|---|---|
| GT Cobb vs centroid large, pred Cobb vs centroid large | Pipeline methodology issue |
| GT Cobb vs centroid small, pred Cobb vs centroid large | TotalSegmentator segmentation issue |
| Both small | Consistent and trustworthy |

### Layer 8 — Per-Vertebra Reliability Flag
Each vertebra is flagged as reliable or unreliable for Cobb measurement based on all four criteria:

| Threshold | Default |
|---|---|
| Per-vertebra DICE | ≥ 0.80 |
| Centroid displacement vs GT mask | ≤ 3.0 mm |
| Centroid displacement vs expert | ≤ 15.0 mm |
| Endplate normal error | ≤ 5.0° |

A vertebra fails if **any** threshold is breached. Expert centroid displacement is only checked when an expert annotation exists for that vertebra.

---

## Output Columns

Each row in the CSV corresponds to one case. Columns include:

| Column | Description |
|---|---|
| `spine_recon_DICE` | Whole-spine union DICE vs GT |
| `{V}_DICE` | Per-vertebra DICE (e.g. `T1_DICE`) |
| `{V}_centroid_disp_mask_mm` | Centroid displacement vs GT mask |
| `{V}_centroid_disp_expert_mm` | Centroid displacement vs VerSe expert |
| `{V}_endplate_max_normal_err_deg` | Max endplate normal angular error |
| `{V}_cobb_reliable` | Reliability flag for Cobb measurement |
| `ctd_tilt_{U}_{L}_deg` | Expert centroid segment tilt angle |
| `cobb_{U}_{L}_gt_deg` | Cobb angle from GT masks |
| `cobb_{U}_{L}_pred_deg` | Cobb angle from TotalSegmentator masks |
| `cobb_{U}_{L}_diff_deg` | Absolute difference between the two |
| `xval_{U}_{L}_gt_cobb_vs_ctd_deg` | GT Cobb vs expert centroid tilt |
| `xval_{U}_{L}_pred_cobb_vs_ctd_deg` | Pred Cobb vs expert centroid tilt |
| `mean_cobb_diff_deg` | Mean Cobb difference across all pairs |
| `n_unreliable_verts` | Count of vertebrae failing reliability check |

---

## Coordinate Convention

All physical-space computations use world RAS mm. Images are canonicalized to RAS via `nibabel.as_closest_canonical` before any array-level comparison. VerSe centroid JSON voxel indices are converted to world mm using the **original** (non-canonical) NIfTI affine, since world coordinates are orientation-independent.

---

## Interpreting Results (sub-gl003 example)

```
spine DICE                : 0.8691
mean vert DICE            : 0.8908
mean centroid disp (mask) : 0.39 mm
mean centroid disp (exp.) : 10.10 mm
mean endplate normal err  : 1.08 deg
mean Cobb diff (pred-GT)  : 0.58 deg
mean GT Cobb vs ctd tilt  : 3.21 deg
mean pred Cobb vs ctd     : 3.48 deg
unreliable verts          : 14
```

The 0.58° mean Cobb difference is the most important result — it shows TotalSegmentator's segmentation errors do not meaningfully change the computed Cobb angle. The 3.48° pred vs centroid tilt confirms geometric consistency with the independent VerSe expert annotations. The 1.08° endplate normal error means TotalSegmentator endplate orientations differ from GT by approximately 1°, contributing at most ~2° of potential Cobb angle error in the worst case — well within the 5° clinical threshold for meaningful disagreement.

---

## Notes

- `std=nan` in the dataset summary is expected when running a single case — it populates with multiple cases
- The C2 vertebra (axis) has an anatomically unusual shape due to the dens process; endplate fitting on C2 is unreliable and it should not be used as a Cobb endpoint
- Lumbar and lower thoracic vertebrae absent from VerSe20 annotations for a given case will show `NaN` metrics and `cobb_reliable=False`
