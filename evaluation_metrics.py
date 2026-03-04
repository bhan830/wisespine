#!/usr/bin/env python3
import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "TotalSegmentator"
CASE_ID = "sub-gl017"

GROUND_TRUTH = f"/gscratch/scrubbed/bhan830/wisespine/data/Verse20/dataset-02validation/derivatives/{CASE_ID}/{CASE_ID}_seg-vert_msk.nii.gz"
INDIV_MASKS_DIR = f"/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/models/{MODEL_NAME}/{CASE_ID}"
RECON_MASK = f"/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/reconstructed/{MODEL_NAME}/{CASE_ID}/{CASE_ID}_{MODEL_NAME}_spine_union.nii.gz"
OUTPUT_CSV = "/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/evaluation_metrics.csv"

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

def load_nifti(path):
    return nib.load(path).get_fdata().astype(bool)

# ==============================
# MAIN EVALUATION
# ==============================
def main():
    # Delete CSV if exists
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    # Load ground truth mask
    gt_mask = nib.load(GROUND_TRUTH).get_fdata()
    
    # Load reconstructed spine
    if not os.path.exists(RECON_MASK):
        print(f"❌ Reconstructed mask not found: {RECON_MASK}")
        return
    recon_mask = nib.load(RECON_MASK).get_fdata() > 0

    # Compute overall spine DICE / IoU
    spine_dice = dice_coefficient(recon_mask, gt_mask > 0)
    spine_iou = iou_score(recon_mask, gt_mask > 0)

    # Prepare results dictionary
    results = {
        "model": MODEL_NAME,
        "case": CASE_ID,
        "spine reconstruction DICE": spine_dice,
        "spine reconstruction IoU": spine_iou
    }

    # Define vertebra label mapping
    label_mapping = {}
    label_counter = 1
    # C1-C7
    for i in range(1, 8):
        label_mapping[f"C{i}"] = label_counter
        label_counter += 1
    # T1-T12
    for i in range(1, 13):
        label_mapping[f"T{i}"] = label_counter
        label_counter += 1
    # L1-L5
    for i in range(1, 6):
        label_mapping[f"L{i}"] = label_counter
        label_counter += 1
    # S1
    label_mapping["S1"] = label_counter

    # Load individual vertebra masks
    vert_files = sorted(glob.glob(os.path.join(INDIV_MASKS_DIR, f"{CASE_ID}_{MODEL_NAME}_vertebrae_*.nii.gz")))

    if not vert_files:
        print("❌ No individual vertebra masks found.")
    else:
        for vf in vert_files:
            vert_name = os.path.basename(vf).split(f"{CASE_ID}_{MODEL_NAME}_vertebrae_")[-1].replace(".nii.gz", "")
            pred_mask = nib.load(vf).get_fdata() > 0
            gt_label = label_mapping.get(vert_name, None)
            if gt_label is None:
                results[f"{vert_name} DICE"] = np.nan
                results[f"{vert_name} IoU"] = np.nan
                continue
            gt_vert_mask = gt_mask == gt_label
            if gt_vert_mask.sum() == 0:
                # Vertebra not present in GT, skip
                results[f"{vert_name} DICE"] = np.nan
                results[f"{vert_name} IoU"] = np.nan
            else:
                results[f"{vert_name} DICE"] = dice_coefficient(pred_mask, gt_vert_mask)
                results[f"{vert_name} IoU"] = iou_score(pred_mask, gt_vert_mask)

    # Save results to CSV
    df = pd.DataFrame([results])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Evaluation complete. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()