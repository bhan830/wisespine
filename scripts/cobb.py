#!/usr/bin/env python3
import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from itertools import combinations
import gc

# ==============================
# CONFIGURATION
# ==============================
RUN_ALL_CASES = True
CASES = ["sub-gl003", "sub-gl016", "sub-gl047", "sub-gl090", "sub-gl124"]

MODELS_BASE_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/models")
OUTPUT_BASE_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/cobb_angles")

MODEL_NAMES = ["TotalSegmentator"]
CLEAR_EXISTING = False  # True → overwrite existing JSON

VERTEBRAE = [
    "C1","C2","C3","C4","C5","C6","C7",
    "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
    "L1","L2","L3","L4","L5"
]

MIN_VERTEBRA_POINTS = 10   # minimal points for plane fitting
MAX_MASK_POINTS = 5000     # downsample large masks for memory efficiency

# ==============================
# HELPERS
# ==============================
def get_case_list(model_dir):
    if RUN_ALL_CASES:
        detected_cases = sorted([p.name for p in model_dir.iterdir() if p.is_dir()])
        print(f"[INFO] Auto-detected {len(detected_cases)} cases in {model_dir}")
        return detected_cases
    else:
        print(f"[INFO] Using manually specified cases: {CASES}")
        return CASES

def load_mask(mask_path):
    """Load vertebra mask and return world coordinates, downsample if too large."""
    if not mask_path.exists():
        return None
    img = nib.load(mask_path)
    data = img.get_fdata()
    coords = np.column_stack(np.where(data > 0)).astype(np.float32)
    if coords.shape[0] < MIN_VERTEBRA_POINTS:
        return None
    # Downsample large masks to save memory
    if coords.shape[0] > MAX_MASK_POINTS:
        idx = np.random.choice(coords.shape[0], MAX_MASK_POINTS, replace=False)
        coords = coords[idx, :]
    return nib.affines.apply_affine(img.affine, coords)

def fit_plane(coords):
    """Fit a plane to points using SVD."""
    if coords is None or len(coords) < 3:
        return None, None
    centroid = coords.mean(axis=0)
    coords_centered = coords - centroid
    _, _, vh = np.linalg.svd(coords_centered)
    normal = vh[2, :]
    return normal, centroid

def get_endplate_planes(coords):
    """Return top and bottom plane normals and centroids."""
    z_sorted = coords[coords[:, 2].argsort()]
    top_coords = z_sorted[-max(3, len(z_sorted)//10):]      # top 10% points
    bottom_coords = z_sorted[:max(3, len(z_sorted)//10)]    # bottom 10% points
    n_top, c_top = fit_plane(top_coords)
    n_bottom, c_bottom = fit_plane(bottom_coords)
    return {"top": (n_top, c_top), "bottom": (n_bottom, c_bottom)}

def plane_angle(n1, n2):
    """Compute angle between two plane normals in degrees."""
    cos_theta = np.clip(np.dot(n1, n2)/(np.linalg.norm(n1)*np.linalg.norm(n2)), -1, 1)
    return np.degrees(np.arccos(cos_theta))

def extract_planes(case_dir, model_name, case):
    """Extract top/bottom planes for all vertebrae in a case."""
    planes = {}
    for v in VERTEBRAE:
        mask_file = case_dir / f"{case}_{model_name}_vertebrae_{v}.nii.gz"
        coords = load_mask(mask_file)
        if coords is None:
            continue
        planes[v] = get_endplate_planes(coords)
        del coords  # free memory early
    return planes

def find_max_cobb(planes):
    """Find the vertebra pair with maximum Cobb angle."""
    max_angle = 0
    upper_v, lower_v = None, None
    for v1, v2 in combinations(planes.keys(), 2):
        n1, _ = planes[v1]["top"]
        n2, _ = planes[v2]["bottom"]
        if n1 is None or n2 is None:
            continue
        angle = plane_angle(n1, n2)
        if angle > max_angle:
            max_angle = angle
            upper_v, lower_v = v1, v2
    return upper_v, lower_v, max_angle

# ==============================
# MAIN PROCESSING
# ==============================
def process_case(model_name, case):
    case_dir = MODELS_BASE_DIR / model_name / case
    output_dir = OUTPUT_BASE_DIR / model_name / case
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{case}_{model_name}_cobb.json"

    # Skip completed cases unless CLEAR_EXISTING
    if output_file.exists() and not CLEAR_EXISTING:
        print(f"[INFO] Cobb JSON already exists for {case} ({model_name}). Skipping.")
        return

    print(f"\nProcessing {case} ({model_name})...")
    planes = extract_planes(case_dir, model_name, case)
    if len(planes) < 2:
        print(f"[WARN] Not enough vertebrae for Cobb calculation in {case}")
        with open(output_file, "w") as f:
            json.dump([], f)
        del planes
        gc.collect()
        return

    upper, lower, angle = find_max_cobb(planes)
    print(f"[INFO] Max Cobb angle: {angle:.2f}° ({upper}-{lower})")

    results = {
        "case": case,
        "model": model_name,
        "upper_vertebra": upper,
        "lower_vertebra": lower,
        "cobb_angle": float(angle)
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Saved Cobb angle JSON to {output_file}")

    # Clean up memory
    del planes
    gc.collect()

def main():
    for model_name in MODEL_NAMES:
        print(f"\n===== Processing Model: {model_name} =====")
        model_dir = MODELS_BASE_DIR / model_name
        cases = get_case_list(model_dir)
        for case in cases:
            process_case(model_name, case)
    print("\n🎉 Cobb calculation complete.")

if __name__ == "__main__":
    main()