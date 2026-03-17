#!/usr/bin/env python3
import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from segmentation import get_case_list

case_list = get_case_list()
# ==============================
# CONFIG
# ==============================
MODEL_NAME = "TotalSegmentator"

BASE_GT_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/data/Verse20/dataset-01training/derivatives")
BASE_MODELS_DIR = Path(f"/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/models/{MODEL_NAME}")
BASE_RECON_DIR = Path(f"/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/reconstructed/{MODEL_NAME}")

OUTPUT_CSV = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/evaluation_metrics.csv")

# ==============================
# UTILITY FUNCTIONS
# ==============================
def dice_coefficient(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return 2.0 * intersection / total if total > 0 else np.nan

def iou_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union > 0 else np.nan

def load_nifti_bool(path):
    return nib.load(path).get_fdata() > 0

# Map vertebra names to GT labels
def create_label_mapping():
    mapping = {}
    counter = 1
    for i in range(1, 8):   # C1-C7
        mapping[f"C{i}"] = counter
        counter += 1
    for i in range(1, 13):  # T1-T12
        mapping[f"T{i}"] = counter
        counter += 1
    for i in range(1, 6):   # L1-L5
        mapping[f"L{i}"] = counter
        counter += 1
    mapping["S1"] = counter
    return mapping

# ==============================
# MAIN EVALUATION
# ==============================
def evaluate_case(case_id, label_mapping):
    results = {"model": MODEL_NAME, "case": case_id}

    gt_path = BASE_GT_DIR / case_id / f"{case_id}_dir-ax_seg-vert_msk.nii.gz"
    recon_path = BASE_RECON_DIR / case_id / f"{case_id}_{MODEL_NAME}_spine_union.nii.gz"
    indiv_mask_dir = BASE_MODELS_DIR / case_id

    if not gt_path.exists():
        print(f"❌ Ground truth not found for {case_id}: {gt_path}")
        return None
    if not recon_path.exists():
        print(f"❌ Reconstructed spine not found for {case_id}: {recon_path}")
        return None

    # Load masks
    gt_mask = nib.load(gt_path).get_fdata()
    recon_mask = load_nifti_bool(recon_path)

    # Overall spine metrics
    results["spine reconstruction DICE"] = dice_coefficient(recon_mask, gt_mask > 0)
    results["spine reconstruction IoU"] = iou_score(recon_mask, gt_mask > 0)

    # Individual vertebra metrics
    vert_files = sorted(glob.glob(str(indiv_mask_dir / f"{case_id}_{MODEL_NAME}_vertebrae_*.nii.gz")))
    if not vert_files:
        print(f"❌ No individual vertebra masks found for {case_id}.")
        return results

    for vf in vert_files:
        vert_name = Path(vf).name.split(f"{case_id}_{MODEL_NAME}_vertebrae_")[-1].replace(".nii.gz", "")
        pred_mask = load_nifti_bool(vf)
        gt_label = label_mapping.get(vert_name, None)
        if gt_label is None or (gt_mask == gt_label).sum() == 0:
            results[f"{vert_name} DICE"] = np.nan
            results[f"{vert_name} IoU"] = np.nan
        else:
            gt_vert_mask = gt_mask == gt_label
            results[f"{vert_name} DICE"] = dice_coefficient(pred_mask, gt_vert_mask)
            results[f"{vert_name} IoU"] = iou_score(pred_mask, gt_vert_mask)

    return results

def main():
    # Delete previous CSV if exists
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()

    label_mapping = create_label_mapping()
    all_results = []

    for case_id in case_list:
        print(f"\n--- Evaluating case: {case_id} ---")
        case_results = evaluate_case(case_id, label_mapping)
        if case_results is not None:
            all_results.append(case_results)

    if not all_results:
        print("❌ No results to save.")
        return

    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Evaluation complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()