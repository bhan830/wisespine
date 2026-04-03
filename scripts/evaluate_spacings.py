#!/usr/bin/env python3
import json
import csv
from pathlib import Path

# ==============================
# CONFIG
# ==============================

MODELS = ["TotalSegmentator"]  # You can add more models here if needed

PRED_JSON_BASE = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/disc_heights")
OUTPUT_BASE = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs")

VERTEBRAE = [
    "C1","C2","C3","C4","C5","C6","C7",
    "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
    "L1","L2","L3","L4","L5"
]

SPACING_LABELS = [f"{VERTEBRAE[i]}-{VERTEBRAE[i+1]}" for i in range(len(VERTEBRAE)-1)]

# ==============================
# HELPERS
# ==============================

def load_json(path):
    """Load predicted spacing JSON."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {d["vertebra_pair"]: d["distance_mm"] for d in data}

def get_all_cases(model_name):
    """Automatically get all case folders for a given model."""
    model_dir = PRED_JSON_BASE / model_name
    if not model_dir.exists():
        return []
    return sorted([p.name for p in model_dir.iterdir() if p.is_dir()])

# ==============================
# MAIN
# ==============================

def main():
    spacing_rows = []

    for model in MODELS:
        case_list = get_all_cases(model)
        if not case_list:
            print(f"[WARN] No cases found for model {model}")
            continue

        for case in case_list:
            print(f"\nProcessing {case} ({model})")

            pred_json = PRED_JSON_BASE / model / case / f"{case}_{model}_disc_heights.json"
            pred = load_json(pred_json)

            row = [case, model]

            for label in SPACING_LABELS:
                val = pred.get(label)
                if val is None or val < 0:
                    row.append("")
                else:
                    row.append(round(val, 2))

            spacing_rows.append(row)

    # Write CSV
    header = ["case", "model"] + SPACING_LABELS
    output_file = OUTPUT_BASE / "spacings.csv"
    with open(output_file, "w", newline="") as f:
        csv.writer(f).writerows([header] + spacing_rows)

    print(f"\n✅ Done. Spacings CSV written to: {output_file}")

if __name__ == "__main__":
    main()