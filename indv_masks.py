#!/usr/bin/env python3
import subprocess
from pathlib import Path
import shutil

# -----------------------------
# Configuration
# -----------------------------
INPUT_CT = Path("/gscratch/scrubbed/bhan830/wisespine/data/Verse20/dataset-02validation/rawdata/sub-gl017/sub-gl017_ct.nii.gz")
OUTPUT_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/models/TotalSegmentator/sub-gl017")
CASE_NAME = "sub-gl017"
MODEL_NAME = "TotalSegmentator"
TASK = "vertebrae_mr"  # use the MR/CT spine model
NR_THR_RESAMP = 4
NR_THR_SAVING = 4

# -----------------------------
# Run TotalSegmentator
# -----------------------------
def run_totalsegmentator(input_ct: Path, output_dir: Path):
    """
    Run TotalSegmentator with HPC-safe CPU mode.
    """
    # Clear previous output
    if output_dir.exists():
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
    print(f"Running vertebrae segmentation (HPC-safe mode)")
    print("="*60)

    subprocess.run(cmd, check=True)

# -----------------------------
# Save individual masks
# -----------------------------
def save_individual_masks(segmentation_path: Path, case_name: str):
    """
    Rename all vertebrae_* and sacrum.nii.gz masks to include the case name.
    """
    masks = list(segmentation_path.glob("vertebrae_*.nii.gz")) + list(segmentation_path.glob("sacrum.nii.gz"))
    if not masks:
        print("[WARNING] No vertebra masks found!")
        return

    for mask_file in masks:
        new_name = segmentation_path / f"{case_name}_{MODEL_NAME}_{mask_file.name}"
        mask_file.rename(new_name)
        print(f"[INFO] Renamed mask: {new_name}")

# -----------------------------
# Main
# -----------------------------
def main():
    run_totalsegmentator(INPUT_CT, OUTPUT_DIR)
    save_individual_masks(OUTPUT_DIR, CASE_NAME)
    print("\n✅ All done. Outputs saved in:")
    print(f"{OUTPUT_DIR}")

if __name__ == "__main__":
    main()