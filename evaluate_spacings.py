#!/usr/bin/env python3

import os
import json
import csv
import numpy as np
import nibabel as nib

# =============================
# CONFIGURATION
# =============================

MODEL_NAME = "TotalSegmentator"

GT_BASE = "/gscratch/scrubbed/bhan830/wisespine/data/Verse20/dataset-01training/derivatives"
PRED_BASE = f"/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/centroids/{MODEL_NAME}"
OUTPUT_BASE = "/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs"

# Only process these 5 cases
CASES = ["sub-gl003", "sub-gl016", "sub-gl047", "sub-gl090", "sub-gl124"]

# Ordered vertebrae list
VERTEBRAE = [
    "C1","C2","C3","C4","C5","C6","C7",
    "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
    "L1","L2","L3","L4","L5"
]

# Create spacing labels (e.g., C1-C2)
SPACING_LABELS = [f"{VERTEBRAE[i]}-{VERTEBRAE[i+1]}" 
                  for i in range(len(VERTEBRAE)-1)]

# =============================
# HELPER FUNCTIONS
# =============================

def load_centroids(json_path):
    """Load centroid JSON and return dict {label_number: [X,Y,Z]}"""
    with open(json_path, "r") as f:
        data = json.load(f)

    centroids = {}
    for entry in data:
        if "label" in entry:
            centroids[entry["label"]] = np.array([entry["X"], entry["Y"], entry["Z"]])
    return centroids

def compute_spacings_mm(centroids, voxel_to_mm_scale):
    """Compute distances between successive vertebrae in mm"""
    spacings = []

    for i in range(1, len(VERTEBRAE)):
        if i not in centroids or (i+1) not in centroids:
            spacings.append("")
            continue

        p1_vox = centroids[i]
        p2_vox = centroids[i+1]

        # Convert voxel differences to mm using voxel spacing
        p1_mm = p1_vox * voxel_to_mm_scale
        p2_mm = p2_vox * voxel_to_mm_scale

        dist_mm = np.linalg.norm(p2_mm - p1_mm)
        spacings.append(round(float(dist_mm), 2))

    return spacings

def compute_errors(gt_spacings, pred_spacings):
    """Compute absolute error in mm and relative error %"""
    errors_mm = []
    errors_pct = []

    for gt, pred in zip(gt_spacings, pred_spacings):
        if gt == "" or pred == "":
            errors_mm.append("")
            errors_pct.append("")
        else:
            abs_err = round(abs(pred - gt), 2)
            pct_err = round(abs_err / gt * 100, 2) if gt != 0 else 0
            errors_mm.append(abs_err)
            errors_pct.append(pct_err)

    return errors_mm, errors_pct

# =============================
# MAIN
# =============================

def main():
    spacing_rows = []
    error_rows = []

    for case in CASES:
        print(f"Processing {case}")

        # Ground truth JSON
        gt_json = os.path.join(GT_BASE, case, f"{case}_dir-ax_seg-subreg_ctd.json")
        pred_json = os.path.join(PRED_BASE, case, f"{case}_{MODEL_NAME}_centroids.json")

        if not os.path.exists(gt_json):
            print(f"  [INFO] Missing GT centroids for {case}")
            continue
        if not os.path.exists(pred_json):
            print(f"  [INFO] Missing predicted centroids for {case}")
            continue

        gt_centroids = load_centroids(gt_json)
        pred_centroids = load_centroids(pred_json)

        # Compute voxel-to-mm scale from CT image (assume all cases have CT in rawdata)
        ct_path = os.path.join(GT_BASE.replace("derivatives","rawdata"), case, f"{case}_dir-ax_ct.nii.gz")
        if not os.path.exists(ct_path):
            print(f"  [INFO] Missing CT for {case}, defaulting voxel_to_mm_scale = 1.0")
            voxel_to_mm_scale = np.array([1.0, 1.0, 1.0])
        else:
            img = nib.load(ct_path)
            voxel_to_mm_scale = np.array(img.header.get_zooms()[:3])

        # Compute spacings in mm
        gt_spacings = compute_spacings_mm(gt_centroids, voxel_to_mm_scale)
        pred_spacings = compute_spacings_mm(pred_centroids, voxel_to_mm_scale)

        # Compute errors
        errors_mm, errors_pct = compute_errors(gt_spacings, pred_spacings)

        spacing_rows.append([case, MODEL_NAME] + pred_spacings)
        error_rows.append([case, MODEL_NAME] + errors_mm)  # store absolute mm errors; optionally add pct later

    # Write CSVs
    header = ["case", "model"] + SPACING_LABELS

    spacing_csv = os.path.join(OUTPUT_BASE, "spacings.csv")
    error_csv = os.path.join(OUTPUT_BASE, "spacings_errors.csv")

    with open(spacing_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(spacing_rows)

    with open(error_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(error_rows)

    print("\n✅ Spacing evaluation complete.")
    print(f"Saved: {spacing_csv}")
    print(f"Saved: {error_csv}")


if __name__ == "__main__":
    main()