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
    return 2.0 * intersection / total if total > 0 else 1.0

def iou_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union > 0 else 1.0

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
    gt_mask_img = nib.load(GROUND_TRUTH)
    gt_mask = gt_mask_img.get_fdata().astype(int)

    # Load reconstructed spine
    recon_img = nib.load(RECON_MASK)
    recon_mask = recon_img.get_fdata().astype(bool)

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

    # Find individual vertebra predictions
    vert_files = sorted(glob.glob(os.path.join(INDIV_MASKS_DIR, f"{CASE_ID}_{MODEL_NAME}_vertebrae_*.nii.gz")))

    # Mapping vertebra names to GT label numbers
    label_mapping = {
        **{f"C{i}": i for i in range(1, 8)},        # C1-C7
        **{f"T{i}": 7 + i for i in range(1, 13)},   # T1-T12
        **{f"L{i}": 19 + i for i in range(1, 6)},   # L1-L5
        "S1": 25,
        "sacrum": 25
    }

    if not vert_files:
        print("❌ No individual vertebra masks found.")
    else:
        for vf in vert_files:
            basename = os.path.basename(vf)
            # Extract vertebra name (C1, T1, L1, etc.)
            vert_name = basename.split(f"{CASE_ID}_{MODEL_NAME}_vertebrae_")[-1].replace(".nii.gz", "")
            if vert_name.lower() == "sacrum":
                vert_name = "sacrum"

            gt_label = label_mapping.get(vert_name, None)
            if gt_label is None:
                print(f"⚠️  Unknown vertebra {vert_name}, skipping.")
                results[f"{vert_name} DICE"] = np.nan
                results[f"{vert_name} IoU"] = np.nan
                continue

            pred_mask_img = nib.load(vf)
            pred_mask = pred_mask_img.get_fdata().astype(bool)

            gt_vert_mask = (gt_mask == gt_label)

            results[f"{vert_name} DICE"] = dice_coefficient(pred_mask, gt_vert_mask)
            results[f"{vert_name} IoU"] = iou_score(pred_mask, gt_vert_mask)

    # Save results to CSV
    df = pd.DataFrame([results])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Evaluation complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()