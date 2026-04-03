#!/usr/bin/env python3
"""
Accurate 3D Cobb Angle Calculation and Endplate Line Overlay

- Computes Cobb angle between two vertebrae using actual top/bottom endplate points
- Draws lines along endplates in 3D NIfTI format
- Saves metrics and overlay NIfTI (Hyak-compatible)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import json

# ==============================
# USER INPUT
# ==============================
CASE = "sub-gl003"       # patient case
MODEL_NAME = "TotalSegmentator"
UPPER_V = "C1"           # upper vertebra
LOWER_V = "T1"           # lower vertebra

MODELS_BASE_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/models")
OUTPUT_DIR = Path(f"./cobb_output/{CASE}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_VERTEBRA_POINTS = 10

# ==============================
# HELPER FUNCTIONS
# ==============================
def load_mask(mask_path):
    """Load mask, return coords in world space, raw data, and NIfTI image."""
    if not mask_path.exists():
        print(f"[ERROR] Mask not found: {mask_path}")
        return None, None, None
    img = nib.load(mask_path)
    data = img.get_fdata()
    coords = np.column_stack(np.where(data > 0))
    if coords.shape[0] < MIN_VERTEBRA_POINTS:
        print(f"[WARN] Not enough points: {mask_path}")
        return None, None, None
    world_coords = nib.affines.apply_affine(img.affine, coords)
    return world_coords, data, img

def fit_plane(coords):
    """Fit plane using SVD, return normal and centroid."""
    centroid = coords.mean(axis=0)
    coords_centered = coords - centroid
    _, _, vh = np.linalg.svd(coords_centered)
    normal = vh[2, :]
    return normal, centroid

def fit_line(coords):
    """Fit principal axis (line) using SVD."""
    centroid = coords.mean(axis=0)
    coords_centered = coords - centroid
    u, s, vh = np.linalg.svd(coords_centered)
    line_dir = vh[0, :]
    return line_dir, centroid

def plane_angle_between_lines(dir1, dir2):
    """Angle between two 3D lines."""
    cos_theta = np.clip(np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2)), -1, 1)
    angle = np.degrees(np.arccos(cos_theta))
    return angle

def extract_endplate_lines(coords, top=True, fraction=0.1):
    """Return principal line (direction and centroid) along top or bottom endplate."""
    # Sort by Z
    z_sorted = coords[coords[:, 2].argsort()]
    n_points = max(3, int(len(z_sorted)*fraction))
    if top:
        points = z_sorted[-n_points:]
    else:
        points = z_sorted[:n_points]
    line_dir, centroid = fit_line(points)
    return line_dir, centroid, points

def draw_line_in_volume(mask_img, line_centroid, line_dir, length=50, thickness=1):
    """Draw a 3D line along line_dir centered at line_centroid, return binary volume."""
    shape = mask_img.shape
    line_vol = np.zeros(shape, dtype=np.uint8)
    inv_affine = np.linalg.inv(mask_img.affine)
    centroid_vox = nib.affines.apply_affine(inv_affine, line_centroid)

    num_points = 2*length + 1
    for t in np.linspace(-length, length, num_points):
        pt = centroid_vox + t*line_dir
        x, y, z = np.round(pt).astype(int)
        # Draw small cube of size thickness
        for dx in range(-thickness, thickness+1):
            for dy in range(-thickness, thickness+1):
                for dz in range(-thickness, thickness+1):
                    xi, yi, zi = x+dx, y+dy, z+dz
                    if 0 <= xi < shape[0] and 0 <= yi < shape[1] and 0 <= zi < shape[2]:
                        line_vol[xi, yi, zi] = 255
    return line_vol

# ==============================
# MAIN FUNCTION
# ==============================
def compute_and_draw_cobb(case, model_name, upper_v, lower_v):
    case_dir = MODELS_BASE_DIR / model_name / case
    upper_mask_path = case_dir / f"{case}_{model_name}_vertebrae_{upper_v}.nii.gz"
    lower_mask_path = case_dir / f"{case}_{model_name}_vertebrae_{lower_v}.nii.gz"

    # Load masks
    upper_coords, upper_data, upper_img = load_mask(upper_mask_path)
    lower_coords, lower_data, lower_img = load_mask(lower_mask_path)
    if upper_coords is None or lower_coords is None:
        print("[ERROR] Failed to load masks")
        return

    # Extract principal lines along top/bottom endplates
    dir_top, centroid_top, top_points = extract_endplate_lines(upper_coords, top=True)
    dir_bottom, centroid_bottom, bottom_points = extract_endplate_lines(lower_coords, top=False)

    # Compute Cobb angle
    cobb_angle = plane_angle_between_lines(dir_top, dir_bottom)
    print(f"[INFO] Cobb angle ({upper_v}-{lower_v}): {cobb_angle:.2f}°")

    # Draw lines in 3D volumes
    line_upper_vol = draw_line_in_volume(upper_img, centroid_top, dir_top)
    line_lower_vol = draw_line_in_volume(lower_img, centroid_bottom, dir_bottom)

    # Merge line volumes
    line_overlay = line_upper_vol + line_lower_vol
    line_overlay[line_overlay>255] = 255

    # Save overlay NIfTI
    overlay_img = nib.Nifti1Image(line_overlay, upper_img.affine)
    overlay_file = OUTPUT_DIR / f"{case}_{upper_v}_{lower_v}_endplate_lines.nii.gz"
    nib.save(overlay_img, overlay_file)
    print(f"[SUCCESS] Saved 3D endplate lines overlay: {overlay_file}")

    # Save metrics
    metrics = {
        "upper_vertebra": upper_v,
        "lower_vertebra": lower_v,
        "cobb_angle_degrees": float(cobb_angle),
        "top_endplate_points": len(top_points),
        "bottom_endplate_points": len(bottom_points)
    }
    metrics_file = OUTPUT_DIR / f"{case}_{upper_v}_{lower_v}_cobb_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[SUCCESS] Saved metrics: {metrics_file}")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    compute_and_draw_cobb(CASE, MODEL_NAME, UPPER_V, LOWER_V)