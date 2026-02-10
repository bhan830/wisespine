#!/usr/bin/env python3
"""
baseline.py

Process a single spine CT case to generate individual vertebrae masks.
Can use existing segmentation masks from training/validation datasets.
Outputs are saved to baseline_outputs/<case_name>/
"""

import os
import sys
from pathlib import Path
import nibabel as nib
import numpy as np

# -----------------------------
# Configuration
# -----------------------------

# Case to process (change to your desired case)
CASE_NAME = "sub-gl017"

# Paths
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data" / "Verse20"
OUTPUT_DIR = BASE_DIR / "baseline_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Training or validation datasets
TRAIN_DIR = DATA_DIR / "dataset-01training" / "derivatives"
VALID_DIR = DATA_DIR / "dataset-02validation" / "derivatives"

# -----------------------------
# Helper functions
# -----------------------------

def find_segmentation(case_name):
    """Look for existing segmentation mask in training or validation sets."""
    # Check validation first
    val_mask = VALID_DIR / case_name / f"{case_name}_seg-vert_msk.nii.gz"
    train_mask = TRAIN_DIR / case_name / f"{case_name}_seg-vert_msk.nii.gz"

    if val_mask.exists():
        return val_mask
    elif train_mask.exists():
        return train_mask
    else:
        return None

def save_vertebrae_masks(seg_mask_path, output_dir):
    """
    Split the segmentation mask into individual vertebrae masks.
    Assumes labels:
      1-7: cervical (C1-C7)
      8-19: thoracic (T1-T12)
      20-25: lumbar (L1-L6)
    """
    img = nib.load(str(seg_mask_path))
    data = img.get_fdata()
    affine = img.affine

    vertebra_labels = {
        **{i: f"C{i}" for i in range(1, 8)},       # C1-C7
        **{i: f"T{i-7}" for i in range(8, 20)},   # T1-T12
        **{i: f"L{i-19}" for i in range(20, 26)} # L1-L6
    }

    output_dir.mkdir(exist_ok=True, parents=True)

    saved_files = []

    for label, name in vertebra_labels.items():
        mask_data = np.where(data == label, 1, 0).astype(np.uint8)
        if mask_data.sum() == 0:
            # Skip labels not present in this scan
            continue
        out_file = output_dir / f"{seg_mask_path.stem}_{name}.nii.gz"
        nib.save(nib.Nifti1Image(mask_data, affine), str(out_file))
        if out_file.exists():
            print(f"✅ Saved {out_file.resolve()}")
            saved_files.append(out_file.resolve())
        else:
            print(f"❌ Failed to save {out_file.resolve()}")

    if len(saved_files) == 0:
        print(f"⚠️ No vertebrae masks were produced for {seg_mask_path.name}")
    return saved_files

# -----------------------------
# Main processing
# -----------------------------

def main():
    print(f"🦴 Processing case: {CASE_NAME}")

    seg_mask = find_segmentation(CASE_NAME)
    if seg_mask is None:
        print(f"❌ No segmentation mask found for {CASE_NAME} in training or validation sets.")
        sys.exit(1)

    print(f"Found segmentation mask: {seg_mask.resolve()}")

    case_output_dir = OUTPUT_DIR / CASE_NAME
    saved_files = save_vertebrae_masks(seg_mask, case_output_dir)

    if saved_files:
        print(f"✅ Done! All masks saved to {case_output_dir.resolve()}")
        print("These files can now be used to reconstruct the spine using TotalSegmentator or other tools.")
    else:
        print(f"⚠️ No masks saved for {CASE_NAME}. Check the segmentation mask file.")

if __name__ == "__main__":
    main()
