#!/usr/bin/env python3
import os
import json
import csv
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================
RUN_ALL_CASES = False
CASES = ["sub-gl003", "sub-gl016", "sub-gl047", "sub-gl090", "sub-gl124"]

MODELS_BASE_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/models")
COBB_JSON_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/cobb_angles")
OUTPUT_CSV = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs") / "cobb_angles.csv"

MODEL_NAMES = ["TotalSegmentator"]

# ==============================
# HELPERS
# ==============================
def get_case_list(model_name):
    """Return list of cases for a model."""
    if RUN_ALL_CASES:
        model_dir = MODELS_BASE_DIR / model_name
        detected_cases = sorted([p.name for p in model_dir.iterdir() if p.is_dir()])
        print(f"[INFO] Auto-detected {len(detected_cases)} cases for {model_name}")
        return detected_cases
    else:
        print(f"[INFO] Using manually specified cases: {CASES}")
        return CASES

# ==============================
# MAIN
# ==============================
def main():
    cobb_results = []

    for model_name in MODEL_NAMES:
        cases = get_case_list(model_name)
        for case in cases:
            json_file = COBB_JSON_DIR / model_name / case / f"{case}_{model_name}_cobb.json"
            if not json_file.exists():
                print(f"[WARN] JSON file not found for {case} ({model_name})")
                continue
            with open(json_file, "r") as f:
                data = json.load(f)
            if not data:  # empty list if calculation failed
                cobb_results.append([case, model_name, "", "", ""])
            else:
                cobb_results.append([
                    data.get("case", ""),
                    data.get("model", ""),
                    data.get("upper_vertebra", ""),
                    data.get("lower_vertebra", ""),
                    data.get("cobb_angle", "")
                ])

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "model", "upper_vertebra", "lower_vertebra", "cobb_angle"])
        writer.writerows(cobb_results)

    print(f"\n✅ Cobb angles CSV saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()