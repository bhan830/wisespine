#!/usr/bin/env python3
"""
evaluation_metrics.py

Compute Dice score and Mean IoU for vertebrae reconstructions
against ground-truth segmentation masks.
"""

import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).parent
RECON_DIR = BASE_DIR / "baseline_outputs" / "reconstructed"
GT_DIR = BASE_DIR / "data" / "Verse20" / "dataset-02validation" / "derivatives"
CSV_FILE = BASE_DIR / "baseline_outputs" / "evaluation_metrics.csv"

# ----------------------------
# Metric functions
# ----------------------------
def dice_score(pred, gt, labels):
    dices = []
    for l in labels:
        p_mask = pred == l
        g_mask = gt == l
        if g_mask.sum() == 0 and p_mask.sum() == 0:
            dices.append(1.0)
        elif g_mask.sum() == 0 or p_mask.sum() == 0:
            dices.append(0.0)
        else:
            dice = 2 * (p_mask & g_mask).sum() / (p_mask.sum() + g_mask.sum())
            dices.append(dice)
    return np.mean(dices)

def mean_iou(pred, gt, labels):
    ious = []
    for l in labels:
        p_mask = pred == l
        g_mask = gt == l
        intersection = (p_mask & g_mask).sum()
        union = (p_mask | g_mask).sum()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)
    return np.mean(ious)

# ----------------------------
# Main evaluation
# ----------------------------
def main():
    results = []

    recon_files = sorted(RECON_DIR.glob("*_reconstructed_spine.nii.gz"))
    if not recon_files:
        print("❌ No reconstructed files found.")
        return

    for recon_file in recon_files:
        # case_name matches folder name in derivatives
        case_name = recon_file.stem.replace("_reconstructed_spine", "")
        print(f"🦴 Evaluating case: {recon_file.name}")

        # Correct GT path
        gt_file = GT_DIR / case_name / f"{case_name}_seg-vert_msk.nii.gz"

        if not gt_file.exists():
            print(f"❌ Ground-truth masks not found for {case_name}, skipping")
            continue

        # Load images
        recon_data = nib.load(recon_file).get_fdata()
        gt_data = nib.load(gt_file).get_fdata()

        labels = np.unique(gt_data)
        labels = labels[labels != 0]

        dice = dice_score(recon_data, gt_data, labels)
        iou = mean_iou(recon_data, gt_data, labels)

        print(f"✅ Dice: {dice:.4f}, Mean IoU: {iou:.4f}")

        results.append({
            "case": case_name,
            "dice": dice,
            "mean_iou": iou
        })

    if results:
        df = pd.DataFrame(results)
        df.to_csv(CSV_FILE, index=False)
        print(f"✅ Evaluation metrics saved to {CSV_FILE}")
    else:
        print("⚠️ No metrics to save.")


if __name__ == "__main__":
    main()
