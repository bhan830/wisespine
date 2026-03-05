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
wisespine_new/
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
