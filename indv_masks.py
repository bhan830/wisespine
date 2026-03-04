#!/usr/bin/env python3
import subprocess
from pathlib import Path
import shutil

# ==============================
# Configuration
# ==============================
CASES = ["sub-gl003", "sub-gl016", "sub-gl047", "sub-gl090", "sub-gl124"]
RAWDATA_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/data/Verse20/dataset-01training/rawdata")
MODELS_BASE_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/models/TotalSegmentator")
MODEL_NAME = "TotalSegmentator"
TASK = "vertebrae_mr"  # MR/CT spine model
NR_THR_RESAMP = 4
NR_THR_SAVING = 4
CLEAR_EXISTING = False  # True = remove existing masks, False = skip cases that already have masks

# ==============================
# Run TotalSegmentator
# ==============================
def run_totalsegmentator(input_ct: Path, output_dir: Path, case: str):
    """
    Run TotalSegmentator with HPC-safe CPU mode.
    """
    # Skip if not clearing and folder already has masks
    existing_masks = list(output_dir.glob(f"{case}_*.nii.gz"))
    if output_dir.exists() and existing_masks and not CLEAR_EXISTING:
        print(f"[INFO] Masks already exist for {case}. Skipping segmentation.")
        return

    # Clear previous output if requested
    if output_dir.exists() and CLEAR_EXISTING:
        print(f"[INFO] Clearing existing output folder: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output folder ready: {output_dir}")

    cmd = [
        "TotalSegmentator",
        "-i", str(input_ct),
        "-o", str(output_dir),
        "--task", TASK,
        "--nr_thr_resamp", str(NR_THR_RESAMP),
        "--nr_thr_saving", str(NR_THR_SAVING),
    ]

    print("="*60)
    print(f"Running vertebrae segmentation (HPC-safe mode) for {input_ct.name}")
    print("="*60)

    subprocess.run(cmd, check=True)

# ==============================
# Rename individual masks
# ==============================
def save_individual_masks(output_dir: Path, case: str):
    """
    Rename all vertebrae_* and sacrum.nii.gz masks to include case name.
    """
    masks = list(output_dir.glob("vertebrae_*.nii.gz")) + list(output_dir.glob("sacrum.nii.gz"))
    if not masks:
        print(f"[WARNING] No vertebra masks found for {case}!")
        return

    for mask_file in masks:
        new_name = output_dir / f"{case}_{MODEL_NAME}_{mask_file.name}"
        mask_file.rename(new_name)
        print(f"[INFO] Renamed mask: {new_name}")

# ==============================
# Main
# ==============================
def main():
    for case in CASES:
        print(f"\n--- Processing case: {case} ---")
        input_ct = RAWDATA_DIR / case / f"{case}_dir-ax_ct.nii.gz"
        output_dir = MODELS_BASE_DIR / case

        if not input_ct.exists():
            print(f"[ERROR] Input CT not found for {case}: {input_ct}")
            continue

        try:
            run_totalsegmentator(input_ct, output_dir, case)
            save_individual_masks(output_dir, case)
            print(f"[INFO] Finished case: {case}")
        except subprocess.CalledProcessError:
            print(f"[ERROR] TotalSegmentator failed for {case}. Check input and environment.")
        except Exception as e:
            print(f"[ERROR] Unexpected error for {case}: {e}")

    print("\n✅ All cases processed.")

# ==============================
if __name__ == "__main__":
    main()