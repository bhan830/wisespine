#!/usr/bin/env python3

import os
import glob
import shutil
import nibabel as nib
import numpy as np

# ==============================
# CONFIG
# ==============================

CASE_NAME = "sub-gl017"  # <-- your case name
MODEL_NAME = "TotalSegmentator"  # <-- model name

MODELS_DIR = f"/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/models/{MODEL_NAME}/{CASE_NAME}"
OUTPUT_DIR = f"/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/reconstructed/{MODEL_NAME}/{CASE_NAME}"

PRESERVE_LABELS = True  # True = each vertebra gets unique label

# ==============================
# Spine Reconstruction
# ==============================

def reconstruct_spine(models_dir, output_dir, case_name, model_name):

    # ---- Clear output directory ----
    if os.path.exists(output_dir):
        print(f"[INFO] Clearing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory ready: {output_dir}")

    # Grab all individual vertebra masks
    vertebra_files = sorted(
        glob.glob(os.path.join(models_dir, "*vertebrae_*.nii.gz"))
    )

    # Optionally include sacrum
    sacrum_file = glob.glob(os.path.join(models_dir, "*sacrum.nii.gz"))
    if sacrum_file:
        vertebra_files.extend(sacrum_file)

    if not vertebra_files:
        print("❌ No individual vertebra masks found.")
        return

    print(f"Found {len(vertebra_files)} vertebra masks.")

    combined = None
    affine = None
    header = None

    for idx, filepath in enumerate(vertebra_files, start=1):
        print(f"Adding: {os.path.basename(filepath)}")
        img = nib.load(filepath)
        data = img.get_fdata()

        if combined is None:
            combined = np.zeros_like(data)
            affine = img.affine
            header = img.header

        if PRESERVE_LABELS:
            combined[data > 0] = idx
        else:
            combined[data > 0] = 1

    # ---- Save union file with case name and model name ----
    output_filename = f"{case_name}_{model_name}_spine_union.nii.gz"
    output_path = os.path.join(output_dir, output_filename)

    out_img = nib.Nifti1Image(combined.astype(np.uint16), affine, header)
    nib.save(out_img, output_path)

    print("\n✅ Spine reconstruction complete.")
    print(f"Saved to: {output_path}")

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    reconstruct_spine(MODELS_DIR, OUTPUT_DIR, CASE_NAME, MODEL_NAME)