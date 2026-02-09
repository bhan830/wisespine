#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import nibabel as nib
import numpy as np
import shutil

# ----------------- CONFIG -----------------
DATA_DIR = "/gscratch/scrubbed/bhan830/wisespine/data/Verse20/dataset-02validation/rawdata"
OUTPUT_DIR = "/gscratch/scrubbed/bhan830/wisespine/baseline_outputs"
TMP_DIR = "/gscratch/scrubbed/bhan830/wisespine/tmp"
TOTALSEG_CMD = "TotalSegmentator"  # command-line TotalSegmentator
TASK_NAME = "vertebrae_body"
DEVICE = "cpu"  # use CPU since no GPU detected
MAX_WORKERS = 2  # number of parallel threads
STOP_ON_FIRST_SUCCESS = True  # stop after first successful segmentation
# ------------------------------------------

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

def process_case(case_folder: Path):
    ct_file = next(case_folder.glob("*_ct.nii*"), None)
    if ct_file is None:
        print(f"⚠️ No CT file found in {case_folder}. Skipping.")
        return False

    case_name = case_folder.name
    print(f"\nProcessing {case_name} ...")

    # temp output folder
    tmp_case_dir = Path(TMP_DIR) / f"ts_{case_name}"
    if tmp_case_dir.exists():
        shutil.rmtree(tmp_case_dir)
    tmp_case_dir.mkdir(parents=True, exist_ok=True)

    # final binary mask output
    mask_file = Path(OUTPUT_DIR) / f"{case_name}_vertebrae_mask.nii.gz"

    try:
        # Run TotalSegmentator CLI
        cmd = [
            TOTALSEG_CMD,
            "-i", str(ct_file),
            "-o", str(tmp_case_dir),
            "--task", TASK_NAME,
            "--device", DEVICE,
            "--ml"  # use machine learning model
        ]
        subprocess.run(cmd, check=True)

        # Look for per-vertebra files
        vert_files = list(tmp_case_dir.glob("vertebrae_*.nii.gz"))
        if not vert_files:
            print(f"⚠️ No vertebrae_* files found for {case_name}. Skipping.")
            return False

        # Create a binary mask by combining all vertebrae masks
        first_img = nib.load(vert_files[0])
        combined_data = np.zeros(first_img.shape, dtype=np.uint8)
        for vf in vert_files:
            img = nib.load(vf)
            combined_data[img.get_fdata() > 0] = 1

        # Save binary mask
        binary_mask = nib.Nifti1Image(combined_data, affine=first_img.affine)
        nib.save(binary_mask, mask_file)
        print(f"✅ Saved binary vertebra mask for {case_name} at {mask_file}")

        # Clean up temp folder
        shutil.rmtree(tmp_case_dir)

        return True

    except subprocess.CalledProcessError as e:
        print(f"⚠️ TotalSegmentator failed for {case_name}. Skipping.")
        return False

# ----------------- MAIN -----------------
def main():
    case_folders = [f for f in Path(DATA_DIR).iterdir() if f.is_dir()]
    print(f"Found {len(case_folders)} cases in {DATA_DIR}")

    success_found = False

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for case_folder, result in zip(case_folders, executor.map(process_case, case_folders)):
            if result and STOP_ON_FIRST_SUCCESS:
                print("\nStopping after first successful case as configured.")
                break

if __name__ == "__main__":
    main()
