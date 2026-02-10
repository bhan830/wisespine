#!/usr/bin/env python3
"""
process_reconstruct.py

Split vertebra masks and reconstruct the spine with numeric labels.
"""

from pathlib import Path
import numpy as np
import nibabel as nib

# ---------------- CONFIG ----------------
CASE_NAME = "sub-gl017"

# Full path to derivatives folder on your cluster
DERIVATIVES_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/data/Verse20/dataset-02validation/derivatives")

BASELINE_OUT = Path("baseline_outputs")

VERTS = [
    "C1","C2","C3","C4","C5","C6","C7",
    "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
    "L1","L2","L3","L4","L5","L6"
]
# ----------------------------------------

def split_masks(case_name: str):
    """
    Split full segmentation mask into individual vertebra masks.
    Returns list of saved mask file paths.
    """
    case_dir = DERIVATIVES_DIR / case_name
    full_mask_file = case_dir / f"{case_name}_seg-vert_msk.nii.gz"

    if not full_mask_file.exists():
        print(f"❌ No segmentation mask found for {case_name}")
        return []

    img = nib.load(str(full_mask_file))
    data = img.get_fdata()
    affine = img.affine

    out_dir = BASELINE_OUT / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for idx, vert in enumerate(VERTS, start=1):
        mask_data = (data == idx).astype(np.uint8)
        if mask_data.sum() == 0:
            print(f"⚠️ No voxels found for {vert}, skipping")
            continue
        out_file = out_dir / f"{case_name}_seg-vert_msk.nii_{vert}.nii.gz"
        nib.save(nib.Nifti1Image(mask_data, affine), str(out_file))
        saved_files.append(out_file)
        print(f"✅ Saved {out_file.name}")

    return saved_files

def reconstruct_spine(case_name: str, vert_files: list):
    """
    Reconstruct spine from individual vertebra masks using numeric labels.
    """
    if not vert_files:
        print("⚠️ No vertebrae masks found to reconstruct spine.")
        return

    first_img = nib.load(str(vert_files[0]))
    recon_data = np.zeros(first_img.shape, dtype=np.uint8)
    affine = first_img.affine

    for idx, f in enumerate(vert_files, start=1):
        img = nib.load(str(f))
        mask = img.get_fdata() > 0
        recon_data[mask] = idx  # assign numeric label
        vert_name = f.name.split("_")[-1].replace(".nii.gz", "")
        print(f"✅ Added {vert_name} (label={idx}) to reconstructed volume")

    recon_dir = BASELINE_OUT / "reconstructed"
    recon_dir.mkdir(parents=True, exist_ok=True)
    out_file = recon_dir / f"{case_name}_reconstructed_spine.nii.gz"
    nib.save(nib.Nifti1Image(recon_data, affine), str(out_file))
    print(f"🎉 Reconstructed spine saved as {out_file}")

def main():
    print(f"🦴 Processing case: {CASE_NAME}")
    vert_files = split_masks(CASE_NAME)
    reconstruct_spine(CASE_NAME, vert_files)

if __name__ == "__main__":
    main()
