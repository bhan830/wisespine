#!/usr/bin/env python3
import os
import glob
import shutil
import nibabel as nib
import numpy as np
from pathlib import Path

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "TotalSegmentator"
MODELS_BASE_DIR = Path(f"/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/models/{MODEL_NAME}")
RECON_BASE_DIR = Path(f"/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/reconstructed/{MODEL_NAME}")

PRESERVE_LABELS = True  # True = each vertebra gets unique label
CLEAR_EXISTING = False   # True = clear all reconstructions before generating
# ==============================
# Spine Reconstruction
# ==============================
def reconstruct_spine(models_dir: Path, recon_dir: Path, case_name: str):
    """
    Combine individual vertebrae masks into a single union spine NIfTI file.
    """
    recon_dir.mkdir(parents=True, exist_ok=True)
    output_path = recon_dir / f"{case_name}_{MODEL_NAME}_spine_union.nii.gz"

    if output_path.exists() and not CLEAR_EXISTING:
        print(f"[INFO] Reconstruction already exists for {case_name}. Skipping.")
        return

    if output_path.exists() and CLEAR_EXISTING:
        print(f"[INFO] Clearing existing reconstruction for {case_name}.")
        output_path.unlink()

    # Grab all vertebra and sacrum masks
    vertebra_files = sorted(models_dir.glob(f"{case_name}_{MODEL_NAME}_vertebrae_*.nii.gz"))
    sacrum_files = sorted(models_dir.glob(f"{case_name}_{MODEL_NAME}_sacrum.nii.gz"))
    all_files = vertebra_files + sacrum_files

    if not all_files:
        print(f"❌ No individual vertebra masks found for {case_name}.")
        return

    print(f"[INFO] Found {len(all_files)} masks for {case_name}. Reconstructing spine...")

    combined = None
    affine = None
    header = None

    for idx, mask_path in enumerate(all_files, start=1):
        print(f"[INFO] Adding: {mask_path.name}")
        img = nib.load(mask_path)
        data = img.get_fdata()

        if combined is None:
            combined = np.zeros_like(data)
            affine = img.affine
            header = img.header

        if PRESERVE_LABELS:
            combined[data > 0] = idx
        else:
            combined[data > 0] = 1

    # Save union file
    nib.save(nib.Nifti1Image(combined.astype(np.uint16), affine, header), output_path)
    print(f"✅ Spine reconstruction complete for {case_name}. Saved to: {output_path}\n")

# ==============================
# MAIN
# ==============================
def main():
    # List all case directories in models base dir
    cases = [d.name for d in MODELS_BASE_DIR.iterdir() if d.is_dir()]
    print(f"[INFO] Found {len(cases)} cases in {MODELS_BASE_DIR}")

    for case in cases:
        print(f"\n--- Processing case: {case} ---")
        models_dir = MODELS_BASE_DIR / case
        recon_dir = RECON_BASE_DIR / case
        reconstruct_spine(models_dir, recon_dir, case)

    print("\n✅ All reconstructions completed.")

# ==============================
if __name__ == "__main__":
    main()