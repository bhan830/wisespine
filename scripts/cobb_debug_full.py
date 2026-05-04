"""
cobb_angle_pipeline.py

Cobb angle pipeline — fits endplate planes in 3D from vertebra segmentation masks,
then projects to the coronal plane for 2D angle measurement.

Why 3D plane fitting instead of 2D PCA:
  A 2D coronal projection collapses the superior-inferior tilt of each vertebra,
  making even a clearly rotated vertebra look rectangular.  Fitting a plane to
  the top/bottom voxels of each mask in 3D captures the true endplate orientation
  regardless of how the mask looks in projection.

Pipeline:
  Step 1 — Load CT volume and individual vertebra mask .nii.gz files
  Step 2 — Extract middle coronal slice (CT background) and per-mask slab
  Step 3 — Fit superior + inferior endplate planes in 3D (SVD)
  Step 4 — Project plane normals to coronal (X-Z) plane -> 2D endplate directions
  Step 5 — Compute Cobb angle; save JSON
  Step 6 — Save debug PNG (OBB polygons + endplate lines on CT background)
  Step 7 — [Optional] Launch Napari 3D viewer for interactive debugging

Usage:
  Set configuration below, then:
      python cobb_angle_pipeline.py

Dependencies:
  pip install nibabel numpy matplotlib scipy
  pip install napari[all]   # only needed if NAPARI_DEBUG = True
"""

import json
import math
import os
import re
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Arc, Polygon


# ── Configuration ─────────────────────────────────────────────────────────────

SUBJECT = "sub-gl003"

# Individual TotalSegmentator vertebra mask files
MASK_DIR = Path(
    "/gscratch/scrubbed/bhan830/wisespine/wisespine_new"
    "/baseline_outputs/models/TotalSegmentator/sub-gl003"
)

# Original CT scan — used as PNG background.
# If the file does not exist the pipeline falls back to a mask silhouette.
CT_PATH = Path(
    "/gscratch/scrubbed/bhan830/wisespine/data/Verse20"
    "/dataset-01training/rawdata/sub-gl003"
    "/sub-gl003_dir-ax_ct.nii.gz"
)

OUTPUT_DIR = Path(
    "/gscratch/scrubbed/bhan830/wisespine/wisespine_new"
    "/baseline_outputs/cobb_angles_dicom/debug"
)

# Vertebrae for the Cobb measurement (script auto-corrects anatomical order)
UPPER_VERTEBRA = "T1"
LOWER_VERTEBRA = "T6"

# Fraction of each vertebra Z extent used to fit each endplate plane.
# 0.25 = top/bottom 25% of voxels by Z. Increase if masks are noisy.
ENDPLATE_FRACTION = 0.25

# Set True to launch an interactive Napari 3D viewer after computing planes.
# Requires a display — automatically skipped on headless servers.
NAPARI_DEBUG = False

# PNG output: long side of the figure in inches
TARGET_LONG_SIDE_IN = 20


# ── Anatomical sort order ─────────────────────────────────────────────────────

SORT_ORDER = {
    **{f"C{i}": i - 1 for i in range(1, 8)},    # C1-C7  ->  0-6
    **{f"T{i}": i + 6 for i in range(1, 13)},   # T1-T12 ->  7-18
    **{f"L{i}": i + 18 for i in range(1, 6)},   # L1-L5  -> 19-23
}

# Visualisation palette
UPPER_COL = "#00d4ff"   # cyan   -- upper vertebra superior endplate
LOWER_COL = "#ff6b35"   # orange -- lower vertebra inferior endplate
BG_COL    = "#0d0d1a"
LABEL_PAD = dict(boxstyle="round,pad=0.18", linewidth=0)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_output(path: Path) -> None:
    """Create parent dirs; warn when overwriting an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"[Output] Overwriting: {path.name}")


def _parse_vertebra_name(path: Path) -> str | None:
    m = re.search(r"vertebrae_([A-Z]\d+)", path.name)
    return m.group(1) if m else None


def _validate_and_order(upper: str, lower: str,
                         available: list) -> tuple:
    for name, label in [(upper, "UPPER_VERTEBRA"), (lower, "LOWER_VERTEBRA")]:
        if name not in available:
            avail = "  ".join(
                f"{v:4s}" for v in
                sorted(available, key=lambda x: SORT_ORDER.get(x, 99))
            )
            sys.exit(f"\n[ERROR] '{name}' ({label}) not found.\n"
                     f"        Available: {avail}\n")
    if SORT_ORDER.get(upper, 99) > SORT_ORDER.get(lower, 99):
        print(f"[INFO] Reordered to anatomical order: {lower} -> {upper}")
        return lower, upper
    return upper, lower


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 -- Load data
# ─────────────────────────────────────────────────────────────────────────────

def _canonicalize(img: nib.Nifti1Image) -> tuple:
    """
    Reorient a NIfTI image to RAS (Right-Anterior-Superior) canonical
    orientation and return (data_array, voxel_spacing).

    Why this matters:
      NIfTI files can be stored in many axis orderings (LAS, PIL, RPI, etc.).
      TotalSegmentator masks are not guaranteed to be RAS.  The pipeline
      assumes axis 0 = left-right, axis 1 = anterior-posterior,
      axis 2 = inferior-superior.  Without canonicalization, projecting
      along axis 1 may give a sagittal or axial slice instead of coronal,
      and the bounding box X/Z coordinates are completely wrong.

    nib.as_closest_canonical reorders and flips axes so the image is as
    close as possible to RAS without resampling.
    """
    orig_codes = nib.aff2axcodes(img.affine)
    img_ras    = nib.as_closest_canonical(img)
    ras_codes  = nib.aff2axcodes(img_ras.affine)
    if orig_codes != ras_codes:
        print(f"         Reoriented {orig_codes} -> {ras_codes}")
    data    = np.asarray(img_ras.dataobj)
    spacing = np.array(img_ras.header.get_zooms()[:3], dtype=float)
    return data, spacing


def load_ct(path: Path):
    """
    Load and canonicalize the CT volume to RAS.
    Returns (data, voxel_spacing) or (None, None) if the file is missing.
    """
    if not path.exists():
        print(f"[Step 1] CT not found at:\n           {path}\n"
              f"         Using mask silhouette fallback.")
        return None, None
    img  = nib.load(str(path))
    orig = nib.aff2axcodes(img.affine)
    print(f"[Step 1] CT original orientation: {orig}")
    data, spacing = _canonicalize(img)
    print(f"[Step 1] CT loaded  shape={data.shape}  "
          f"spacing={np.round(spacing, 3)} mm")
    return data.astype(np.float32), spacing


def load_masks(mask_dir: Path) -> dict:
    """
    Load and canonicalize every vertebra mask to RAS.
    Returns {vertebra_name: (data_array_uint8, voxel_spacing)}.
    Storing pre-canonicalized arrays avoids repeated reorientation.
    """
    masks = {}
    for f in sorted(mask_dir.glob("*_vertebrae_*.nii.gz")):
        name = _parse_vertebra_name(f)
        if name:
            img  = nib.load(str(f))
            data, spacing = _canonicalize(img)
            masks[name] = (data.astype(np.uint8), spacing)
    if not masks:
        sys.exit(f"[ERROR] No vertebra masks found in:\n        {mask_dir}")
    sample_spacing = next(iter(masks.values()))[1]
    print(f"[Step 1] Loaded {len(masks)} masks  "
          f"spacing={np.round(sample_spacing, 3)} mm  "
          f"(all canonicalized to RAS)")
    print(f"         {', '.join(sorted(masks, key=lambda x: SORT_ORDER.get(x, 99)))}")
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 -- Coronal slice helpers
# ─────────────────────────────────────────────────────────────────────────────

def mid_coronal_ct(ct_data: np.ndarray) -> np.ndarray:
    """Single middle coronal (Y) slice of the CT -> (nx, nz)."""
    return ct_data[:, ct_data.shape[1] // 2, :]


def mid_coronal_mask_slab(mask_3d: np.ndarray) -> np.ndarray:
    """
    Max-project a thin slab (+-5% of ny) around the mask Y centre-of-mass.
    Returns (nx, nz) binary array.  More robust than a single slice for thin
    structures that may not intersect the exact mid-plane.
    """
    ny  = mask_3d.shape[1]
    ys  = np.where(mask_3d.any(axis=(0, 2)))[0]
    if len(ys) == 0:
        return mask_3d[:, ny // 2, :]
    mid_y  = int(ys.mean())
    slab_r = max(1, ny // 20)
    y_lo   = max(0, mid_y - slab_r)
    y_hi   = min(ny, mid_y + slab_r + 1)
    return mask_3d[:, y_lo:y_hi, :].max(axis=1)


def build_background(ct_data, masks: dict) -> np.ndarray:
    """
    Return (nx, nz) background for the PNG.
    Prefers the CT mid-coronal slice; falls back to mask max-projection.
    masks = {name: (data_uint8, spacing)}
    In RAS: axis 0 = R (left-right), axis 1 = A (front-back), axis 2 = S (up-down).
    Coronal view = project along axis 1, display (nx, nz) = (R, S).
    """
    if ct_data is not None:
        return mid_coronal_ct(ct_data)

    first_data = next(iter(masks.values()))[0]
    combo = np.zeros(first_data.shape, dtype=np.uint8)
    for data, _ in masks.values():
        np.maximum(combo, data, out=combo)
    print("[Step 2] Using mask silhouette as background.")
    return combo.max(axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 -- 3D endplate plane fitting
# ─────────────────────────────────────────────────────────────────────────────

def _fit_plane_svd(pts: np.ndarray):
    """
    Fit a plane to a point cloud using SVD.
    Returns (unit_normal, centroid).
    The normal is the right-singular vector with the smallest singular value
    (least-squares best-fit plane normal).
    Normal is oriented to point in the +Z direction.
    """
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal   = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    return normal / np.linalg.norm(normal), centroid


def _obb_endplate_corners(mask_3d: np.ndarray,
                           vox_x: float, vox_z: float,
                           top_half: bool):
    """
    Compute left and right corner of one endplate in coronal voxel space.

    1. Project the 3D mask to a coronal (X-Z) slab via max-projection.
    2. PCA on the 2D point cloud in mm -- PC1 is the long (left-right) axis.
    3. Split at PC2 median into superior / inferior halves.
    4. Within the chosen half take the outermost 10% along PC1 as corners.

    Returns shape (2, 2): [[x_left, z_left], [x_right, z_right]] in voxels,
    or None if the mask is empty.
    """
    mask_2d = mid_coronal_mask_slab(mask_3d)
    xs, zs  = np.where(mask_2d > 0)
    if len(xs) < 4:
        return None

    pts_mm   = np.column_stack([xs * vox_x, zs * vox_z])
    centroid = pts_mm.mean(axis=0)
    pts_c    = pts_mm - centroid

    _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
    pc1, pc2 = Vt[0], Vt[1]
    if pc1[0] < 0: pc1 = -pc1   # PC1 -> right
    if pc2[1] < 0: pc2 = -pc2   # PC2 -> superior

    proj1  = pts_c @ pc1
    proj2  = pts_c @ pc2
    median = np.median(proj2)

    half = (proj2 >= median) if top_half else (proj2 < median)
    if not half.any():
        half = np.ones(len(proj2), dtype=bool)

    p1h, p2h = proj1[half], proj2[half]
    n = max(1, len(p1h) // 10)

    idx_l   = np.argsort(p1h)[:n]
    left_mm = centroid + p1h[idx_l].mean() * pc1 + p2h[idx_l].mean() * pc2

    idx_r    = np.argsort(p1h)[-n:]
    right_mm = centroid + p1h[idx_r].mean() * pc1 + p2h[idx_r].mean() * pc2

    return np.array([
        left_mm  / np.array([vox_x, vox_z]),
        right_mm / np.array([vox_x, vox_z]),
    ])


def fit_endplate_planes(mask_3d: np.ndarray,
                         vox_x: float, vox_y: float, vox_z: float):
    """
    Fit superior and inferior endplate planes to a vertebra mask in 3D.

    The superior endplate is fitted to the top ENDPLATE_FRACTION of mask
    voxels by Z; inferior to the bottom fraction.

    Returns a dict with keys 'superior' and 'inferior', each containing:
        normal_3d   : unit normal of the fitted plane  (3,)
        centroid_mm : centroid of the endplate voxels  (3,)
        corners_vox : [[x_left, z_left], [x_right, z_right]]  (2, 2) or None
    Returns None if the mask has fewer than 20 voxels.
    """
    xs, ys, zs = np.where(mask_3d > 0)
    if len(xs) < 20:
        return None

    pts_mm = np.column_stack([xs * vox_x, ys * vox_y, zs * vox_z])
    z_vals = pts_mm[:, 2]

    sup_thresh = np.percentile(z_vals, 100 * (1 - ENDPLATE_FRACTION))
    inf_thresh = np.percentile(z_vals, 100 * ENDPLATE_FRACTION)

    sup_normal, sup_centroid = _fit_plane_svd(pts_mm[z_vals >= sup_thresh])
    inf_normal, inf_centroid = _fit_plane_svd(pts_mm[z_vals <= inf_thresh])

    return {
        "superior": {
            "normal_3d":   sup_normal,
            "centroid_mm": sup_centroid,
            "corners_vox": _obb_endplate_corners(
                                mask_3d, vox_x, vox_z, top_half=True),
        },
        "inferior": {
            "normal_3d":   inf_normal,
            "centroid_mm": inf_centroid,
            "corners_vox": _obb_endplate_corners(
                                mask_3d, vox_x, vox_z, top_half=False),
        },
    }


def process_all_vertebrae(masks: dict,
                           vox_x: float, vox_y: float, vox_z: float) -> dict:
    """Run Steps 2-3 for every loaded mask. Returns a nested result dict."""
    results = {}
    skipped = []

    for name, (mask_3d, _) in sorted(masks.items(),
                                      key=lambda x: SORT_ORDER.get(x[0], 99)):
        planes  = fit_endplate_planes(mask_3d, vox_x, vox_y, vox_z)

        if planes is None:
            skipped.append(name)
            continue

        mask_2d = mid_coronal_mask_slab(mask_3d)
        xs, zs  = np.where(mask_2d > 0)
        results[name] = {
            "planes":  planes,
            "mask_2d": mask_2d,
            "mask_3d": mask_3d,
            "bbox": {
                "x_min": int(xs.min()), "x_max": int(xs.max()),
                "z_min": int(zs.min()), "z_max": int(zs.max()),
                "cx":    float(xs.mean()), "cz": float(zs.mean()),
            },
            "vox_x": vox_x, "vox_y": vox_y, "vox_z": vox_z,
        }

    if skipped:
        print(f"[Step 3] Skipped (sparse masks): {', '.join(skipped)}")
    print(f"[Step 3] Processed {len(results)} vertebrae.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 -- Project 3D normal to coronal endplate direction
# ─────────────────────────────────────────────────────────────────────────────

def normal_to_coronal_direction(normal_3d: np.ndarray) -> np.ndarray:
    """
    Convert a 3D endplate plane normal to the direction of the endplate line
    in the coronal (X-Z) plane.

    Derivation:
      The endplate plane has unit normal n = [nx, ny, nz].
      Its intersection with any coronal plane (Y = const) is a line.
      That line's direction = n x Y_hat = [nx, ny, nz] x [0, 1, 0]
                                        = [-nz, 0, nx]
      In the 2D (X, Z) coronal projection this is [-nz, nx], normalised.

    If the normal is nearly parallel to Y (endplate faces directly forward),
    the direction degenerates -- [1, 0] (horizontal) is returned as a fallback.
    """
    nx, ny, nz = normal_3d
    d    = np.array([-nz, nx])
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return np.array([1.0, 0.0])
    return d / norm


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 -- Cobb angle + JSON
# ─────────────────────────────────────────────────────────────────────────────

def compute_cobb(vertebrae: dict, upper_name: str, lower_name: str) -> dict:
    """
    Compute the Cobb angle:
      - superior endplate of upper_name  (3D plane -> coronal direction)
      - inferior endplate of lower_name  (3D plane -> coronal direction)
    """
    sup_plane = vertebrae[upper_name]["planes"]["superior"]
    inf_plane = vertebrae[lower_name]["planes"]["inferior"]

    sup_dir = normal_to_coronal_direction(sup_plane["normal_3d"])
    inf_dir = normal_to_coronal_direction(inf_plane["normal_3d"])

    dot   = np.clip(np.abs(np.dot(sup_dir, inf_dir)), 0.0, 1.0)
    angle = math.degrees(math.acos(dot))

    vox_x = vertebrae[upper_name]["vox_x"]
    vox_z = vertebrae[upper_name]["vox_z"]

    def _serialise_corners(corners_vox):
        if corners_vox is None:
            return None
        return {
            "left_vox":  corners_vox[0].tolist(),
            "right_vox": corners_vox[1].tolist(),
            "left_mm":   (corners_vox[0] * [vox_x, vox_z]).tolist(),
            "right_mm":  (corners_vox[1] * [vox_x, vox_z]).tolist(),
        }

    return {
        "subject":        SUBJECT,
        "upper_vertebra": upper_name,
        "lower_vertebra": lower_name,
        "cobb_angle_deg": round(angle, 4),
        "superior_endplate": {
            "normal_3d":    sup_plane["normal_3d"].tolist(),
            "centroid_mm":  sup_plane["centroid_mm"].tolist(),
            "direction_2d": sup_dir.tolist(),
            "corners":      _serialise_corners(sup_plane["corners_vox"]),
        },
        "inferior_endplate": {
            "normal_3d":    inf_plane["normal_3d"].tolist(),
            "centroid_mm":  inf_plane["centroid_mm"].tolist(),
            "direction_2d": inf_dir.tolist(),
            "corners":      _serialise_corners(inf_plane["corners_vox"]),
        },
    }


def save_json(payload: dict, path: Path) -> None:
    _prepare_output(path)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[JSON] Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 -- Debug PNG
# ─────────────────────────────────────────────────────────────────────────────

def _line_span(p_l: np.ndarray, p_r: np.ndarray,
               x_max: float) -> tuple:
    """Extend a line through p_l and p_r to the full image width [0, x_max]."""
    d = p_r - p_l
    if abs(d[0]) < 1e-9:
        return np.array([p_l[0], 0.0]), np.array([p_l[0], float(1e9)])
    t0 = (0.0   - p_l[0]) / d[0]
    t1 = (x_max - p_l[0]) / d[0]
    return p_l + t0 * d, p_l + t1 * d


def _line_intersection(p1, d1, p2, d2):
    """Intersection of two 2D parametric lines. Returns None if parallel."""
    A = np.column_stack([d1, -d2])
    if abs(np.linalg.det(A)) < 1e-10:
        return None
    t = np.linalg.solve(A, p2 - p1)[0]
    return p1 + t * d1


def draw_debug_png(bg_slice: np.ndarray,
                   vertebrae: dict,
                   cobb: dict,
                   vox_x: float, vox_y: float, vox_z: float,
                   out_path: Path) -> None:
    """
    Render and save the debug PNG.

    Background: CT mid-coronal slice (or mask silhouette).
    No resampling -- raw voxels go to imshow with aspect=vox_z/vox_x so each
    voxel is rendered at its correct physical size without changing the array.
    All overlays are plotted in raw voxel coordinates.

    Overlays:
      - OBB polygon for every vertebra  (PCA-fitted, should be crooked)
      - Thick endplate lines for the two selected vertebrae
      - Dashed extensions to their intersection point
      - Angle arc + numeric label
    """
    upper_name    = cobb["upper_vertebra"]
    lower_name    = cobb["lower_vertebra"]
    angle_deg     = cobb["cobb_angle_deg"]
    imshow_aspect = vox_z / vox_x

    nx, nz = bg_slice.shape
    x_max  = float(nx - 1)

    # Figure size from physical dimensions, long side fixed to TARGET_LONG_SIDE_IN
    phys_w = nx * vox_x
    phys_h = nz * vox_z
    ratio  = phys_w / phys_h
    if ratio >= 1.0:
        fig_w, fig_h = TARGET_LONG_SIDE_IN, TARGET_LONG_SIDE_IN / ratio
    else:
        fig_w, fig_h = TARGET_LONG_SIDE_IN * ratio, TARGET_LONG_SIDE_IN

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG_COL)
    ax.set_facecolor(BG_COL)

    # Background -- percentile windowing
    fg      = bg_slice[bg_slice > 0]
    p2, p98 = (np.percentile(fg, [2, 98]) if len(fg) else (0.0, 1.0))
    disp    = np.clip((bg_slice - p2) / max(p98 - p2, 1e-9), 0.0, 1.0)

    ax.imshow(disp.T, cmap="gray", origin="lower",
              aspect=imshow_aspect, interpolation="none", vmin=0, vmax=1)

    cmap_v = plt.colormaps["tab20"].resampled(max(len(vertebrae), 1))

    # ── Compute view bounds from OBB corners BEFORE drawing ─────────────────
    # We need these up-front so we can clip the intersection point and
    # angle label to always land inside the visible area.
    all_x, all_z = [], []
    for vdata in vertebrae.values():
        sup_c = vdata["planes"]["superior"]["corners_vox"]
        inf_c = vdata["planes"]["inferior"]["corners_vox"]
        if sup_c is not None and inf_c is not None:
            for pt in [sup_c[0], sup_c[1], inf_c[0], inf_c[1]]:
                all_x.append(pt[0]); all_z.append(pt[1])

    if all_x:
        pad_x  = (max(all_x) - min(all_x)) * 0.15
        pad_z  = (max(all_z) - min(all_z)) * 0.10
        view_x0 = min(all_x) - pad_x
        view_x1 = max(all_x) + pad_x
        view_z0 = min(all_z) - pad_z
        view_z1 = max(all_z) + pad_z
    else:
        view_x0, view_x1 = 0.0, float(nx - 1)
        view_z0, view_z1 = 0.0, float(nz - 1)

    # ── OBB polygons + vertebra labels ───────────────────────────────────────
    for idx, (name, vdata) in enumerate(
        sorted(vertebrae.items(), key=lambda x: SORT_ORDER.get(x[0], 99))
    ):
        colour = cmap_v(idx)
        is_sel = name in (upper_name, lower_name)
        planes = vdata["planes"]

        sup_c = planes["superior"]["corners_vox"]
        inf_c = planes["inferior"]["corners_vox"]
        if sup_c is None or inf_c is None:
            continue

        # Four OBB corners: sup-left, sup-right, inf-right, inf-left
        quad = np.array([sup_c[0], sup_c[1], inf_c[1], inf_c[0]])

        ax.add_patch(Polygon(
            quad, closed=True,
            linewidth=2.2 if is_sel else 0.8,
            edgecolor=(*colour[:3], 0.95),
            facecolor=(*colour[:3], 0.22 if is_sel else 0.06),
            linestyle="-" if is_sel else "--",
            zorder=3,
        ))

        cx, cz = quad[:, 0].mean(), quad[:, 1].mean()
        ax.text(cx, cz, name, color="white",
                fontsize=7 if is_sel else 5,
                fontweight="bold" if is_sel else "normal",
                ha="center", va="center", zorder=4,
                bbox={**LABEL_PAD, "facecolor": colour,
                      "alpha": 0.85 if is_sel else 0.35})

    # ── Endplate lines ────────────────────────────────────────────────────────
    def centroid_to_vox(c_mm):
        # centroid_mm = [x_mm, y_mm, z_mm]; project to (x_vox, z_vox)
        return np.array([c_mm[0] / vox_x, c_mm[2] / vox_z])

    def dir_to_vox(d_mm):
        return np.array([d_mm[0] / vox_x, d_mm[1] / vox_z])

    def hw_from_corners(corners):
        if corners is None:
            return 20.0
        l = np.array(corners["left_vox"])
        r = np.array(corners["right_vox"])
        return np.linalg.norm(r - l) / 2.0

    sup_ep = cobb["superior_endplate"]
    inf_ep = cobb["inferior_endplate"]

    sup_anchor = centroid_to_vox(sup_ep["centroid_mm"])
    inf_anchor = centroid_to_vox(inf_ep["centroid_mm"])
    sup_dir    = dir_to_vox(sup_ep["direction_2d"])
    inf_dir    = dir_to_vox(inf_ep["direction_2d"])
    sup_hw     = hw_from_corners(sup_ep["corners"])
    inf_hw     = hw_from_corners(inf_ep["corners"])

    for anchor, d_vox, hw, col, lbl in [
        (sup_anchor, sup_dir, sup_hw, UPPER_COL,
         f"{upper_name} superior endplate (3D fit)"),
        (inf_anchor, inf_dir, inf_hw, LOWER_COL,
         f"{lower_name} inferior endplate (3D fit)"),
    ]:
        p_l = anchor - hw * d_vox
        p_r = anchor + hw * d_vox

        # Dashed extension clipped to the view bounds (avoids expanding limits)
        sl, sr = _line_span(p_l, p_r, x_max)
        sl[0] = np.clip(sl[0], view_x0, view_x1)
        sr[0] = np.clip(sr[0], view_x0, view_x1)
        ax.plot([sl[0], sr[0]], [sl[1], sr[1]],
                color=col, lw=1.0, ls="--", alpha=0.5, zorder=6)

        # Solid segment over the vertebra body
        ax.plot([p_l[0], p_r[0]], [p_l[1], p_r[1]],
                color=col, lw=3.5, ls="-", label=lbl, zorder=7)
        for pt in (p_l, p_r):
            ax.plot(pt[0], pt[1], "o", color=col,
                    ms=6, mec="white", mew=0.8, zorder=8)

    # ── Intersection + angle label ────────────────────────────────────────────
    intersect = _line_intersection(sup_anchor, sup_dir, inf_anchor, inf_dir)

    # Decide where to place the angle label:
    #   - If the intersection is inside the view, draw the arc there.
    #   - If it is outside (or lines are parallel), place the label at the
    #     midpoint between the two endplate anchors (always in view).
    view_x_range = view_x1 - view_x0
    view_z_range = view_z1 - view_z0

    def _in_view(pt):
        return (view_x0 <= pt[0] <= view_x1 and
                view_z0 <= pt[1] <= view_z1)

    avg_h = np.mean([v["bbox"]["z_max"] - v["bbox"]["z_min"]
                     for v in vertebrae.values()])

    if intersect is not None and _in_view(intersect):
        # Intersection is visible — draw arc there
        ax.plot(intersect[0], intersect[1], "x",
                color="yellow", ms=12, mew=2.0, zorder=9)

        ar = imshow_aspect
        a1 = math.degrees(math.atan2(sup_dir[1] * ar, sup_dir[0]))
        a2 = math.degrees(math.atan2(inf_dir[1] * ar, inf_dir[0]))
        a_lo, a_hi = sorted([a1 % 180, a2 % 180])
        if (a_hi - a_lo) > 90:
            a_lo, a_hi = a_hi, a_lo + 180

        arc_r = avg_h * 0.6
        ax.add_patch(Arc(intersect, 2 * arc_r, 2 * arc_r,
                         angle=0, theta1=a_lo, theta2=a_hi,
                         color="yellow", lw=2.0, zorder=9))

        mid_rad = math.radians((a_lo + a_hi) / 2)
        lx = intersect[0] + arc_r * 2.5 * math.cos(mid_rad)
        lz = intersect[1] + arc_r * 2.5 * math.sin(mid_rad)
        # Final safety clip so label stays inside view
        lx = np.clip(lx, view_x0 + view_x_range * 0.05,
                         view_x1 - view_x_range * 0.05)
        lz = np.clip(lz, view_z0 + view_z_range * 0.05,
                         view_z1 - view_z_range * 0.05)
    else:
        # Intersection off-screen — place label between the two endplate midpoints
        mid   = (sup_anchor + inf_anchor) / 2
        lx    = np.clip(mid[0], view_x0 + view_x_range * 0.1,
                                view_x1 - view_x_range * 0.1)
        lz    = np.clip(mid[1], view_z0 + view_z_range * 0.1,
                                view_z1 - view_z_range * 0.1)

    ax.text(lx, lz, f"{angle_deg:.1f}°",
            color="yellow", fontsize=14, fontweight="bold",
            ha="center", va="center", zorder=10,
            bbox={**LABEL_PAD, "facecolor": BG_COL, "alpha": 0.85})

    # ── Set axis limits LAST so nothing drawn above can expand them ───────────
    ax.set_xlim(view_x0, view_x1)
    ax.set_ylim(view_z0, view_z1)

    ax.legend(
        handles=[
            mpatches.Patch(color=UPPER_COL,
                           label=f"{upper_name}  superior endplate"),
            mpatches.Patch(color=LOWER_COL,
                           label=f"{lower_name}  inferior endplate"),
            mpatches.Patch(color="yellow",
                           label=f"Cobb angle = {angle_deg:.1f}°"),
        ],
        loc="upper right",
        facecolor="#1a1a2e", edgecolor="gray",
        labelcolor="white", fontsize=8,
    )
    ax.set_title(
        f"{SUBJECT}  --  Cobb Angle: {upper_name} -> {lower_name}\n"
        f"Angle = {angle_deg:.1f}  |  "
        f"spacing: {vox_x:.3f} x {vox_y:.3f} x {vox_z:.3f} mm  |  "
        f"mid coronal slice  |  3D SVD plane fitting",
        color="white", fontsize=10, fontweight="bold", pad=8,
    )
    ax.axis("off")
    plt.tight_layout(pad=0.5)

    _prepare_output(out_path)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), format="png")
    plt.close()
    print(f"[Step 6] PNG saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 -- Napari 3D viewer (optional)
# ─────────────────────────────────────────────────────────────────────────────

def launch_napari(ct_data, vertebrae: dict,
                  upper_name: str, lower_name: str,
                  vox_x: float, vox_y: float, vox_z: float) -> None:
    """
    Launch an interactive Napari 3D viewer showing:
      - CT image (if available)
      - All vertebra masks as a combined label volume
      - Superior + inferior endplate centroid points (cyan / orange)
      - Larger anchor points for the two selected vertebrae

    What to look for:
      - Do the mask labels correctly outline each vertebra body?
      - Are the cyan (superior) / orange (inferior) centroids on the endplate
        surfaces, or floating in empty space?
      - If centroids are off, increase ENDPLATE_FRACTION or check mask quality.
    """
    try:
        import napari
    except ImportError:
        print("[Step 7] Napari not installed.  Run:  pip install napari[all]")
        return

    spacing = (vox_x, vox_y, vox_z)
    viewer  = napari.Viewer(title=f"Cobb Debug -- {SUBJECT}")

    if ct_data is not None:
        viewer.add_image(ct_data, name="CT", colormap="gray",
                         scale=spacing, blending="additive")

    # Combine all masks into a single label volume
    ref_shape = next(iter(vertebrae.values()))["mask_3d"].shape
    label_vol = np.zeros(ref_shape, dtype=np.uint8)
    for idx, (name, vdata) in enumerate(
        sorted(vertebrae.items(),
               key=lambda x: SORT_ORDER.get(x[0], 99)), 1
    ):
        label_vol[vdata["mask_3d"] > 0] = idx   # mask_3d already stored in vertebrae dict

    viewer.add_labels(label_vol, name="Vertebra masks", scale=spacing)

    # Endplate centroids as Points layers
    sup_pts, inf_pts = [], []
    for vdata in vertebrae.values():
        sc = vdata["planes"]["superior"]["centroid_mm"] / np.array([vox_x, vox_y, vox_z])
        ic = vdata["planes"]["inferior"]["centroid_mm"] / np.array([vox_x, vox_y, vox_z])
        sup_pts.append(sc)
        inf_pts.append(ic)

    if sup_pts:
        viewer.add_points(np.array(sup_pts), name="Superior centroids",
                          size=4, face_color="cyan", scale=spacing)
    if inf_pts:
        viewer.add_points(np.array(inf_pts), name="Inferior centroids",
                          size=4, face_color="orange", scale=spacing)

    # Highlight the two selected vertebrae with larger markers
    for name, col in [(upper_name, "cyan"), (lower_name, "orange")]:
        if name not in vertebrae:
            continue
        sc = (vertebrae[name]["planes"]["superior"]["centroid_mm"]
              / np.array([vox_x, vox_y, vox_z]))
        ic = (vertebrae[name]["planes"]["inferior"]["centroid_mm"]
              / np.array([vox_x, vox_y, vox_z]))
        viewer.add_points(np.array([sc, ic]),
                          name=f"{name} selected endplates",
                          size=8, face_color=col, scale=spacing)

    print("[Step 7] Napari viewer launched -- close the window to continue.")
    napari.run()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    upper_name = UPPER_VERTEBRA.upper()
    lower_name = LOWER_VERTEBRA.upper()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load
    print("\n-- Step 1: Load CT and masks ----------------------------------------")
    ct_data, ct_spacing = load_ct(CT_PATH)
    masks               = load_masks(MASK_DIR)

    if ct_spacing is not None:
        vox_x, vox_y, vox_z = ct_spacing
    else:
        # Spacing already extracted during canonicalization
        vox_x, vox_y, vox_z = next(iter(masks.values()))[1]
        print(f"[Step 1] Spacing from mask: "
              f"{vox_x:.3f} x {vox_y:.3f} x {vox_z:.3f} mm")

    # Step 2: Build coronal background
    print("\n-- Step 2: Build coronal background ---------------------------------")
    bg_slice = build_background(ct_data, masks)
    print(f"[Step 2] Background shape: {bg_slice.shape}")

    # Steps 3-4: Fit 3D endplate planes for all vertebrae
    print("\n-- Steps 3-4: 3D endplate plane fitting -----------------------------")
    vertebrae = process_all_vertebrae(masks, vox_x, vox_y, vox_z)

    planes_json = {
        "subject":       SUBJECT,
        "voxel_spacing": {"x_mm": vox_x, "y_mm": vox_y, "z_mm": vox_z},
        "vertebrae": {
            name: {
                "bbox": v["bbox"],
                "superior": {
                    "normal_3d":   v["planes"]["superior"]["normal_3d"].tolist(),
                    "centroid_mm": v["planes"]["superior"]["centroid_mm"].tolist(),
                },
                "inferior": {
                    "normal_3d":   v["planes"]["inferior"]["normal_3d"].tolist(),
                    "centroid_mm": v["planes"]["inferior"]["centroid_mm"].tolist(),
                },
            }
            for name, v in vertebrae.items()
        },
    }
    save_json(planes_json,
              OUTPUT_DIR / f"{SUBJECT}_endplate_planes.json")

    # Step 5: Cobb angle
    print("\n-- Step 5: Cobb angle -----------------------------------------------")
    upper_name, lower_name = _validate_and_order(
        upper_name, lower_name, list(vertebrae.keys())
    )
    cobb      = compute_cobb(vertebrae, upper_name, lower_name)
    angle_deg = cobb["cobb_angle_deg"]

    border = "-" * 44
    print(f"\n  +{border}+")
    print(f"  |  Cobb angle ({upper_name} -> {lower_name}):  {angle_deg:6.2f}  |")
    print(f"  +{border}+\n")

    save_json(cobb,
              OUTPUT_DIR / f"{SUBJECT}_cobb_{upper_name}_{lower_name}.json")

    # Step 6: Debug PNG
    print("\n-- Step 6: Debug PNG ------------------------------------------------")
    draw_debug_png(bg_slice, vertebrae, cobb,
                   vox_x, vox_y, vox_z,
                   OUTPUT_DIR / f"{SUBJECT}_cobb_{upper_name}_{lower_name}.png")

    # Step 7: Napari (optional)
    if NAPARI_DEBUG:
        print("\n-- Step 7: Napari 3D viewer -----------------------------------------")
        launch_napari(ct_data, vertebrae, upper_name, lower_name,
                      vox_x, vox_y, vox_z)

    print("\n✓ Pipeline complete.")
    print(f"  Planes JSON : {OUTPUT_DIR / f'{SUBJECT}_endplate_planes.json'}")
    print(f"  Cobb JSON   : {OUTPUT_DIR / f'{SUBJECT}_cobb_{upper_name}_{lower_name}.json'}")
    print(f"  Debug PNG   : {OUTPUT_DIR / f'{SUBJECT}_cobb_{upper_name}_{lower_name}.png'}\n")


if __name__ == "__main__":
    main()