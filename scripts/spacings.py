#!/usr/bin/env python3
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion

# ==============================
# CONFIGURATION
# ==============================
RUN_ALL_CASES = True  # True → process all folders in CURRENT_DIR
CASES = ["sub-gl003", "sub-gl016", "sub-gl047", "sub-gl090", "sub-gl124"]  # Only used if RUN_ALL_CASES=False

MODELS_BASE_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/models")
OUTPUT_BASE_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/disc_heights")

MODEL_NAMES = ["TotalSegmentator", "TotalSpineSeg"]
CLEAR_EXISTING = False  # If True, overwrites existing JSON

VERTEBRA_ORDER = [
    "C1","C2","C3","C4","C5","C6","C7",
    "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
    "L1","L2","L3","L4","L5"
]

# Endplate parameters
TOP_P = 95
BOT_P = 5
MIN_ENDPLATE_POINTS = 10
EROSION_ITER = 1
OUTLIER_THRESHOLD_MM = 15.0

# ==============================
# HELPERS
# ==============================
def get_case_list(model_dir):
    if RUN_ALL_CASES:
        return sorted([p.name for p in model_dir.iterdir() if p.is_dir()])
    else:
        return CASES

def load_mask(mask_path):
    """Load NIfTI mask and return voxel coordinates transformed to mm"""
    img = nib.load(mask_path)
    data = img.get_fdata() > 0
    if np.sum(data) == 0:
        return None
    if EROSION_ITER > 0:
        data = binary_erosion(data, iterations=EROSION_ITER)
    coords = np.argwhere(data)
    if coords.size == 0:
        return None
    return nib.affines.apply_affine(img.affine, coords)

def get_endplate(coords, superior=True):
    """Return points on the superior or inferior endplate along Z-axis"""
    z = coords[:, 2]
    if superior:
        endplate = coords[z >= np.percentile(z, TOP_P)]
    else:
        endplate = coords[z <= np.percentile(z, BOT_P)]
    return endplate

def compute_disc_height(upper_coords, lower_coords, upper_label="", lower_label=""):
    """Compute minimal Euclidean distance between upper inferior and lower superior endplates"""
    upper_endplate = get_endplate(upper_coords, superior=False)
    lower_endplate = get_endplate(lower_coords, superior=True)
    if len(upper_endplate) < MIN_ENDPLATE_POINTS or len(lower_endplate) < MIN_ENDPLATE_POINTS:
        print(f"[WARN] Not enough endplate points for {upper_label}-{lower_label}")
        return None
    tree = cKDTree(lower_endplate)
    distance_mm = float(tree.query(upper_endplate, k=1)[0].min())
    if distance_mm > OUTLIER_THRESHOLD_MM:
        print(f"⚠️ outlier ({distance_mm:.2f} mm)")
    else:
        print(f"✅ {distance_mm:.2f} mm")
    return distance_mm

# ==============================
# MAIN
# ==============================
def process_case(model_name, case_name):
    print(f"\nProcessing {case_name} ({model_name})...")
    case_dir = MODELS_BASE_DIR / model_name / case_name
    output_dir = OUTPUT_BASE_DIR / model_name / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{case_name}_{model_name}_disc_heights.json"
    if output_file.exists() and not CLEAR_EXISTING:
        print(f"[INFO] JSON already exists. Skipping {case_name} ({model_name})")
        return

    results = []
    for i in range(len(VERTEBRA_ORDER) - 1):
        upper = VERTEBRA_ORDER[i]
        lower = VERTEBRA_ORDER[i + 1]

        upper_mask = case_dir / f"{case_name}_{model_name}_vertebrae_{upper}.nii.gz"
        lower_mask = case_dir / f"{case_name}_{model_name}_vertebrae_{lower}.nii.gz"

        print(f"{upper}-{lower}:", end=" ")

        if not upper_mask.exists() or not lower_mask.exists():
            print("❌ missing")
            continue

        upper_coords = load_mask(upper_mask)
        lower_coords = load_mask(lower_mask)
        if upper_coords is None or lower_coords is None:
            print("❌ empty")
            continue

        dist = compute_disc_height(upper_coords, lower_coords, upper, lower)
        if dist is None:
            continue

        results.append({"vertebra_pair": f"{upper}-{lower}", "distance_mm": dist})

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved JSON: {output_file}")

def main():
    for model_name in MODEL_NAMES:
        print(f"\n===== Processing Model: {model_name} =====")
        model_dir = MODELS_BASE_DIR / model_name
        case_list = get_case_list(model_dir)
        for case in case_list:
            process_case(model_name, case)
    print("\n🎉 All processing done!")

if __name__ == "__main__":
    main()