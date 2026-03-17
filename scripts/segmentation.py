#!/usr/bin/env python3
import subprocess
from pathlib import Path
import shutil
import importlib

# ==============================
# Configuration
# ==============================

# If True → automatically process ALL cases in training dataset
# If False → only process cases listed in CASES
RUN_ALL_TRAINING_CASES = False

CASES = [
    "sub-gl003"
]

RAWDATA_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/data/Verse20/dataset-01training/rawdata")
MODELS_BASE_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/models")

MODEL_NAMES = ["TotalSegmentator", "TotalSpineSeg"]

CLEAR_EXISTING = False  # True = remove existing masks, False = skip if already exist

# TotalSegmentator settings
TS_TASK = "vertebrae_mr"
TS_NR_THR_RESAMP = 4
TS_NR_THR_SAVING = 4


# ==============================
# Helper Functions
# ==============================

def get_case_list():
    """Return case list based on RUN_ALL_TRAINING_CASES flag."""
    if RUN_ALL_TRAINING_CASES:
        print("[INFO] Auto-detecting all training cases...")
        detected_cases = sorted([
            p.name for p in RAWDATA_DIR.iterdir()
            if p.is_dir() and p.name.startswith("sub-")
        ])
        print(f"[INFO] Found {len(detected_cases)} cases.")
        return detected_cases
    else:
        print("[INFO] Using manually specified case list.")
        return CASES


def run_totalsegmentator(input_ct: Path, output_dir: Path, case: str):
    existing_masks = list(output_dir.glob(f"{case}_*.nii.gz"))

    if output_dir.exists() and existing_masks and not CLEAR_EXISTING:
        print(f"[INFO] Masks already exist for {case}. Skipping.")
        return

    if output_dir.exists() and CLEAR_EXISTING:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "TotalSegmentator",
        "-i", str(input_ct),
        "-o", str(output_dir),
        "--task", TS_TASK,
        "--nr_thr_resamp", str(TS_NR_THR_RESAMP),
        "--nr_thr_saving", str(TS_NR_THR_SAVING),
    ]

    print(f"[RUN] TotalSegmentator → {case}")
    subprocess.run(cmd, check=True)


def run_totalspineseg(input_ct: Path, output_dir: Path, case: str):
    try:
        ts_module = importlib.import_module("totalspineseg.inference")
    except ModuleNotFoundError:
        print("[ERROR] totalspineseg.inference not found.")
        return

    existing_masks = list(output_dir.glob(f"{case}_*.nii.gz"))

    if output_dir.exists() and existing_masks and not CLEAR_EXISTING:
        print(f"[INFO] Masks already exist for {case} (TotalSpineSeg). Skipping.")
        return

    if output_dir.exists() and CLEAR_EXISTING:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] TotalSpineSeg → {case}")

    try:
        ts_module.run_segmentation(str(input_ct), str(output_dir))
    except Exception as e:
        print(f"[ERROR] TotalSpineSeg failed for {case}: {e}")


def rename_masks(output_dir: Path, case: str, model_name: str):
    masks = list(output_dir.glob("vertebrae_*.nii.gz")) + \
            list(output_dir.glob("sacrum.nii.gz"))

    if not masks:
        print(f"[WARNING] No masks found for {case} ({model_name})")
        return

    for mask_file in masks:
        new_name = output_dir / f"{case}_{model_name}_{mask_file.name}"
        mask_file.rename(new_name)


# ==============================
# Main
# ==============================

def main():
    case_list = get_case_list()

    for model_name in MODEL_NAMES:
        print(f"\n===== Processing Model: {model_name} =====")

        for case in case_list:
            input_ct = RAWDATA_DIR / case / f"{case}_dir-ax_ct.nii.gz"
            output_dir = MODELS_BASE_DIR / model_name / case

            if not input_ct.exists():
                print(f"[WARNING] Missing CT for {case}")
                continue

            try:
                if model_name == "TotalSegmentator":
                    run_totalsegmentator(input_ct, output_dir, case)

                elif model_name == "TotalSpineSeg":
                    run_totalspineseg(input_ct, output_dir, case)

                else:
                    print(f"[ERROR] Unknown model: {model_name}")
                    continue

                rename_masks(output_dir, case, model_name)

            except subprocess.CalledProcessError:
                print(f"[ERROR] Segmentation failed for {case} ({model_name})")
            except Exception as e:
                print(f"[ERROR] Unexpected error for {case}: {e}")
    print("\n✅ All models processed.")


if __name__ == "__main__":
    main()