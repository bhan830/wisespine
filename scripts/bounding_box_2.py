"""
bounding_box.py

Task: find the mid coronal slice of the spine CT, draw an oriented bounding
box (OBB) around each vertebra on that slice, label every box, and save all
corner coordinates to a JSON file.

Per meeting guidance (Vikash):
  - Focus on the coronal view and extract the mid-slice to visualise
    segmentation as blobs.
  - Draw bounding boxes around each disjointed mask and label them with
    coordinates saved to a JSON file.
  - Break the task into specific steps and approach it methodically.

Steps
-----
  1. Load CT volume and vertebra mask files, reorient all to RAS.
  2. Find the mid coronal Y-slice (highest combined mask density).
  3. Project a slab around that Y onto the coronal plane.
  4. Fit an oriented bounding box (OBB) to each vertebra via PCA.
  5. Save OBB corner coordinates to JSON (overwrite if exists).
  6. Draw OBBs on CT background and save as PNG.

Outputs
-------
  <OUTPUT_DIR>/<SUBJECT>_spine_coronal.png   — debug image
  <OUTPUT_DIR>/<SUBJECT>_vertebra_boxes.json — OBB corner coordinates

Usage
-----
  python bounding_box.py

Dependencies
------------
  pip install nibabel numpy matplotlib scipy
"""

import json
import os
import re
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Polygon


# ── Configuration ─────────────────────────────────────────────────────────────

SUBJECT = "sub-gl003"

MASK_DIR = Path(
    "/gscratch/scrubbed/bhan830/wisespine/wisespine_new"
    "/baseline_outputs/models/TotalSegmentator/sub-gl003"
)

CT_PATH = Path(
    "/gscratch/scrubbed/bhan830/wisespine/data/Verse20"
    "/dataset-01training/rawdata/sub-gl003"
    "/sub-gl003_dir-ax_ct.nii.gz"
)

OUTPUT_DIR = Path(
    "/gscratch/scrubbed/bhan830/wisespine/wisespine_new"
    "/baseline_outputs/cobb_angles_dicom/debug"
)

# Reconstructed spine union mask — union of all individual vertebra masks.
# Used to spatially filter each individual mask before OBB fitting,
# removing ribs, transverse processes, and other surrounding structures
# that fall outside the spine column and bloat the bounding boxes.
# Set to None to skip filtering.
SPINE_MASK_PATH = Path(
    "/gscratch/scrubbed/bhan830/wisespine/wisespine_new"
    "/baseline_outputs/reconstructed/TotalSegmentator/sub-gl003"
    "/sub-gl003_TotalSegmentator_spine_union.nii.gz"
)

# Half-width of the coronal slab in voxels (±SLAB_HALF around the mid Y).
# Wider slab = more vertebrae captured at the cost of slight blurring.
# Increase if some vertebrae are missing from the image.
SLAB_HALF = 10

# OBB fitting: clip each mask to its central Z fraction before PCA.
# This removes protruding anatomy (e.g. C2 odontoid, spinous processes)
# that would otherwise make individual boxes appear too large.
OBB_Z_CLIP_LO = 20   # percentile
OBB_Z_CLIP_HI = 80   # percentile

# OBB corner robustness: average the outermost N% of projected points
# instead of the single extreme, to suppress stray voxels.
CORNER_PERCENTILE = 5   # %

# PNG long-side size in inches
FIGURE_LONG_IN = 20


# ── Anatomical sort order ─────────────────────────────────────────────────────

SORT_ORDER = {
    **{f"C{i}": i - 1  for i in range(1, 8)},
    **{f"T{i}": i + 6  for i in range(1, 13)},
    **{f"L{i}": i + 18 for i in range(1, 6)},
}

BG_COL    = "#0d0d1a"
LABEL_PAD = dict(boxstyle="round,pad=0.2", linewidth=0)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load and canonicalize to RAS
# ─────────────────────────────────────────────────────────────────────────────

def _to_ras(img: nib.Nifti1Image) -> tuple:
    """
    Reorient a NIfTI image to RAS orientation and return (data, zooms).

    nib.as_closest_canonical reorders and flips axes so that:
      axis 0 -> R (left to right)
      axis 1 -> A (posterior to anterior)
      axis 2 -> S (inferior to superior)

    This guarantees that projecting along axis 1 gives a true coronal view
    and that X = left-right, Z = inferior-superior in all 2D outputs.
    Without this, masks stored in different orientations produce bounding
    boxes that are completely misaligned with the CT background.
    """
    ras   = nib.as_closest_canonical(img)
    data  = np.asarray(ras.dataobj)
    zooms = np.array(ras.header.get_zooms()[:3], dtype=float)
    return data, zooms


def load_ct(path: Path) -> tuple:
    """Load CT, reorient to RAS. Returns (data_float32, zooms) or (None, None)."""
    if not path.exists():
        print(f"[CT] Not found: {path}\n     Using mask silhouette as background.")
        return None, None
    data, zooms = _to_ras(nib.load(str(path)))
    print(f"[CT] Loaded  shape={data.shape}  zooms={np.round(zooms, 3)} mm")
    return data.astype(np.float32), zooms


def load_masks(mask_dir: Path) -> dict:
    """
    Return {vertebra_name: data_uint8} for every mask in mask_dir.
    All masks are reoriented to RAS so they share the same axis convention
    as the CT.
    """
    masks   = {}
    pattern = re.compile(r"vertebrae_([A-Z]\d+)")
    for f in sorted(mask_dir.glob("*_vertebrae_*.nii.gz")):
        m = pattern.search(f.name)
        if not m:
            continue
        data, _ = _to_ras(nib.load(str(f)))
        masks[m.group(1)] = (data > 0).astype(np.uint8)

    if not masks:
        sys.exit(f"[ERROR] No vertebra masks found in {mask_dir}")

    print(f"[Masks] Loaded {len(masks)}: "
          f"{', '.join(sorted(masks, key=lambda x: SORT_ORDER.get(x, 99)))}")
    return masks


def load_spine_mask(path) -> np.ndarray | None:
    """
    Load the reconstructed spine union mask and reorient to RAS.
    Returns a binary uint8 array of the same shape as the CT/individual masks,
    or None if path is None or file does not exist.

    This mask covers only the spine column (union of all vertebra masks).
    ANDing each individual mask with it removes voxels that belong to the
    individual vertebra mask but fall outside the spine column footprint —
    primarily ribs, transverse processes, and pedicle extensions that appear
    in the coronal projection and cause the OBBs to be too wide.
    """
    if path is None:
        return None
    if not path.exists():
        print(f"[Spine mask] Not found: {path}  (skipping filter)")
        return None
    data, _ = _to_ras(nib.load(str(path)))
    spine = (data > 0).astype(np.uint8)
    print(f"[Spine mask] Loaded  shape={spine.shape}  "
          f"voxels={int(spine.sum())}")
    return spine


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Find the mid coronal slice
# ─────────────────────────────────────────────────────────────────────────────

def best_coronal_y(masks: dict) -> int:
    """
    Find the Y-index (anterior-posterior axis in RAS) where the combined
    mask voxel count is highest.

    This is the mid-slice per Vikash's guidance — the coronal plane that
    shows the most complete cross-section of the vertebra bodies.
    """
    ny    = next(iter(masks.values())).shape[1]
    y_sum = np.zeros(ny, dtype=np.float64)
    for data in masks.values():
        y_sum += data.sum(axis=(0, 2))
    y_idx = int(y_sum.argmax())
    print(f"[Slice] Mid coronal Y = {y_idx} / {ny}  "
          f"(peak density = {int(y_sum[y_idx])} voxels)")
    return y_idx


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Coronal slab projection
# ─────────────────────────────────────────────────────────────────────────────

def slab_projection(volume: np.ndarray, y_idx: int,
                    half: int, binary: bool = False) -> np.ndarray:
    """
    Max-project a slab of +/-half voxels around y_idx along axis 1.
    Returns a 2D array (nx, nz).

    A slab rather than a single slice captures vertebrae whose bodies do
    not fall exactly on y_idx due to the spine's sagittal curve.
    SLAB_HALF controls the trade-off: wider = more anatomy, slightly blurrier.
    """
    ny   = volume.shape[1]
    y_lo = max(0,  y_idx - half)
    y_hi = min(ny, y_idx + half + 1)
    proj = volume[:, y_lo:y_hi, :].max(axis=1)
    return (proj > 0).astype(np.uint8) if binary else proj


def build_background(ct_data, masks: dict, y_idx: int) -> np.ndarray:
    """
    Build the (nx, nz) CT background for the PNG at y_idx.
    Falls back to a mask max-projection silhouette if CT is unavailable.
    """
    if ct_data is not None:
        bg = slab_projection(ct_data, y_idx, SLAB_HALF)
        print(f"[BG] CT slab Y={y_idx}+/-{SLAB_HALF}  shape={bg.shape}")
        return bg
    first = next(iter(masks.values()))
    combo = np.zeros(first.shape, dtype=np.uint8)
    for data in masks.values():
        np.maximum(combo, data, out=combo)
    bg = combo.max(axis=1).astype(np.float32)
    print(f"[BG] Mask silhouette fallback  shape={bg.shape}")
    return bg


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Fit oriented bounding boxes via PCA
# ─────────────────────────────────────────────────────────────────────────────

def fit_obb(mask_2d: np.ndarray, vox_x: float, vox_z: float) -> dict | None:
    """
    Fit an oriented bounding box to a 2D binary mask using PCA.

    Algorithm
    ---------
    1. Clip the mask to the central Z fraction (OBB_Z_CLIP_LO to OBB_Z_CLIP_HI
       percentile) before fitting.  This removes protruding anatomy such as
       the C2 odontoid process and spinous processes that extend well beyond
       the vertebra body and would otherwise produce boxes that are too large.

    2. Scale all foreground coordinates to mm before running PCA.
       Voxels are anisotropic (x=0.291 mm, z=1.25 mm — 4.3x difference).
       Skipping this biases PC1 toward the X axis regardless of actual
       vertebra orientation.

    3. SVD on the centred point cloud gives:
         PC1 = long axis (left <-> right across the vertebra body)
         PC2 = short axis (inferior <-> superior)
       The box will be visibly tilted for any rotated vertebra.

    4. Project all points; use CORNER_PERCENTILE averaging at extremes
       to suppress stray voxels pulling corners outward.

    5. Reconstruct four corners in mm, convert back to voxel coordinates.

    Corner order: [bottom_left, top_left, top_right, bottom_right]
      bottom = inferior (low Z in RAS)
      top    = superior (high Z in RAS)
      left/right = patient left/right

    Returns dict or None if fewer than 4 foreground pixels remain after clipping.
    """
    xs, zs = np.where(mask_2d > 0)
    if len(xs) < 4:
        return None

    # Clip to central Z fraction to remove protruding anatomy
    z_lo = np.percentile(zs, OBB_Z_CLIP_LO)
    z_hi = np.percentile(zs, OBB_Z_CLIP_HI)
    keep = (zs >= z_lo) & (zs <= z_hi)
    xs, zs = xs[keep], zs[keep]
    if len(xs) < 4:
        return None

    # Scale to mm — essential for correct PCA with anisotropic voxels
    pts_mm   = np.column_stack([xs * vox_x, zs * vox_z])
    centroid = pts_mm.mean(axis=0)
    pts_c    = pts_mm - centroid

    # SVD: rows of Vt are the principal components
    _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
    pc1, pc2 = Vt[0], Vt[1]

    # Orient consistently: PC1 -> right (+x), PC2 -> superior (+z)
    if pc1[0] < 0: pc1 = -pc1
    if pc2[1] < 0: pc2 = -pc2

    # Project all points; clip extremes for robustness
    proj1 = pts_c @ pc1
    proj2 = pts_c @ pc2
    n      = max(1, len(proj1) * CORNER_PERCENTILE // 100)
    p1_min = np.sort(proj1)[:n].mean()
    p1_max = np.sort(proj1)[-n:].mean()
    p2_min = np.sort(proj2)[:n].mean()
    p2_max = np.sort(proj2)[-n:].mean()

    # Four corners in mm, then convert back to voxels
    corners_mm = np.array([
        centroid + p1_min * pc1 + p2_min * pc2,   # bottom-left
        centroid + p1_min * pc1 + p2_max * pc2,   # top-left
        centroid + p1_max * pc1 + p2_max * pc2,   # top-right
        centroid + p1_max * pc1 + p2_min * pc2,   # bottom-right
    ])
    corners_vox = corners_mm / np.array([vox_x, vox_z])
    center_vox  = centroid   / np.array([vox_x, vox_z])

    return {
        "corners_vox": corners_vox,          # (4, 2) for plotting
        "corners_mm":  corners_mm,           # (4, 2) for JSON / angle calc
        "center_vox":  center_vox,           # (2,)   label position
        "center_mm":   centroid,             # (2,)
        "angle_deg":   float(np.degrees(np.arctan2(pc1[1], pc1[0]))),
    }


def process_vertebrae(masks: dict, y_idx: int,
                      vox_x: float, vox_z: float,
                      spine_mask: np.ndarray | None = None) -> dict:
    """
    Project every vertebra mask to 2D at y_idx and fit an OBB.
    If spine_mask is provided, each individual mask is ANDed with it
    before projection to remove surrounding structures.
    Returns {name: obb_dict} for all non-empty masks.
    """
    results, skipped = {}, []
    for name in sorted(masks, key=lambda x: SORT_ORDER.get(x, 99)):
        mask = masks[name]
        if spine_mask is not None and mask.shape == spine_mask.shape:
            mask = mask & spine_mask
        mask_2d = slab_projection(mask, y_idx, SLAB_HALF, binary=True)
        obb     = fit_obb(mask_2d, vox_x, vox_z)
        if obb is None:
            skipped.append(name)
        else:
            results[name] = obb
    if skipped:
        print(f"[OBB] Skipped (empty at Y={y_idx}): {', '.join(skipped)}")
    print(f"[OBB] Fitted {len(results)} oriented bounding boxes.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Save JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_json(obbs: dict, y_idx: int,
              vox_x: float, vox_y: float, vox_z: float,
              out_path: Path) -> None:
    """
    Write OBB corner coordinates to JSON. Overwrites any existing file.

    Corner order: [bottom_left, top_left, top_right, bottom_right]
    Coordinates stored in both voxel space and physical mm.
    """
    payload = {
        "subject":       SUBJECT,
        "coronal_y_idx": y_idx,
        "slab_half":     SLAB_HALF,
        "voxel_spacing": {"x_mm": vox_x, "y_mm": vox_y, "z_mm": vox_z},
        "corner_order":  ["bottom_left", "top_left", "top_right", "bottom_right"],
        "vertebrae": {
            name: {
                "corners_vox": obb["corners_vox"].tolist(),
                "corners_mm":  obb["corners_mm"].tolist(),
                "center_vox":  obb["center_vox"].tolist(),
                "center_mm":   obb["center_mm"].tolist(),
                "obb_angle_deg": obb["angle_deg"],
            }
            for name, obb in sorted(
                obbs.items(), key=lambda x: SORT_ORDER.get(x[0], 99)
            )
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[JSON] Overwriting {out_path.name}")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[JSON] Saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Draw and save PNG
# ─────────────────────────────────────────────────────────────────────────────

def draw_png(bg: np.ndarray, obbs: dict,
             vox_x: float, vox_z: float,
             y_idx: int, out_path: Path) -> None:
    """
    Render CT coronal background with every OBB overlaid and save as PNG.

    - imshow uses aspect=vox_z/vox_x so voxels render at correct physical size.
    - OBBs are tilted Polygon patches — visibly crooked for rotated vertebrae.
    - Labels show only the vertebra name.
    - View is cropped tightly to the combined OBB extent.
    """
    nx, nz = bg.shape
    aspect = vox_z / vox_x

    phys_w = nx * vox_x
    phys_h = nz * vox_z
    ratio  = phys_w / phys_h
    fig_w  = FIGURE_LONG_IN if ratio >= 1.0 else FIGURE_LONG_IN * ratio
    fig_h  = FIGURE_LONG_IN / ratio if ratio >= 1.0 else FIGURE_LONG_IN

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG_COL)
    ax.set_facecolor(BG_COL)

    fg      = bg[bg > 0]
    p2, p98 = (np.percentile(fg, [2, 98]) if len(fg) else (0.0, 1.0))
    disp    = np.clip((bg - p2) / max(p98 - p2, 1e-9), 0.0, 1.0)

    ax.imshow(disp.T, cmap="gray", origin="lower",
              aspect=aspect, interpolation="bilinear", vmin=0, vmax=1)

    cmap_v         = plt.colormaps["tab20"].resampled(max(len(obbs), 1))
    all_cx, all_cz = [], []

    for idx, (name, obb) in enumerate(
        sorted(obbs.items(), key=lambda x: SORT_ORDER.get(x[0], 99))
    ):
        colour  = cmap_v(idx)
        corners = np.array(obb["corners_vox"])
        center  = obb["center_vox"]

        all_cx.extend(corners[:, 0])
        all_cz.extend(corners[:, 1])

        ax.add_patch(Polygon(
            corners, closed=True,
            linewidth=1.5,
            edgecolor=(*colour[:3], 0.95),
            facecolor=(*colour[:3], 0.12),
            zorder=3,
        ))
        ax.text(center[0], center[1], name,
                color="white", fontsize=5.5, fontweight="bold",
                ha="center", va="center", zorder=4,
                bbox={**LABEL_PAD, "facecolor": colour, "alpha": 0.75})

    if all_cx:
        span_x = max(all_cx) - min(all_cx)
        span_z = max(all_cz) - min(all_cz)
        ax.set_xlim(min(all_cx) - span_x * 0.12,
                    max(all_cx) + span_x * 0.12)
        ax.set_ylim(min(all_cz) - span_z * 0.08,
                    max(all_cz) + span_z * 0.08)

    ax.axis("off")
    plt.tight_layout(pad=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[PNG] Overwriting {out_path.name}")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), format="png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load
    print("\n── Step 1: Load ──────────────────────────────────────────────────")
    ct_data, ct_zooms = load_ct(CT_PATH)
    masks             = load_masks(MASK_DIR)

    if ct_zooms is not None:
        vox_x, vox_y, vox_z = ct_zooms
    else:
        _, zooms = _to_ras(nib.load(str(
            next(MASK_DIR.glob("*_vertebrae_*.nii.gz"))
        )))
        vox_x, vox_y, vox_z = zooms
        print(f"[Spacing] From mask: {vox_x:.3f} x {vox_y:.3f} x {vox_z:.3f} mm")

    # Step 2: Find mid coronal slice
    print("\n── Step 2: Find mid coronal slice ───────────────────────────────")
    y_idx = best_coronal_y(masks)

    # Step 3: Build background
    print("\n── Step 3: Build coronal background ─────────────────────────────")
    bg = build_background(ct_data, masks, y_idx)

    # Step 4: Fit OBBs
    print("\n── Step 4: Fit oriented bounding boxes ──────────────────────────")
    spine_mask = load_spine_mask(SPINE_MASK_PATH)
    obbs = process_vertebrae(masks, y_idx, vox_x, vox_z, spine_mask)

    # Step 5: Save JSON
    print("\n── Step 5: Save JSON ─────────────────────────────────────────────")
    json_path = OUTPUT_DIR / f"{SUBJECT}_vertebra_boxes.json"
    save_json(obbs, y_idx, vox_x, vox_y, vox_z, json_path)

    # Step 6: Draw PNG
    print("\n── Step 6: Draw PNG ──────────────────────────────────────────────")
    png_path = OUTPUT_DIR / f"{SUBJECT}_spine_coronal.png"
    draw_png(bg, obbs, vox_x, vox_z, y_idx, png_path)

    names_in_image = ", ".join(
        sorted(obbs.keys(), key=lambda x: SORT_ORDER.get(x, 99))
    )
    print(f"[PNG] Saved -> {png_path}")
    print(f"\n[Result] {SUBJECT}  |  coronal Y={y_idx}+/-{SLAB_HALF}  |  "
          f"spacing: {vox_x:.3f} x {vox_z:.3f} mm")
    print(f"[Result] {len(obbs)} vertebrae with bounding boxes: {names_in_image}")

    print("\n Done.")
    print(f"  JSON : {json_path}")
    print(f"  PNG  : {png_path}\n")


if __name__ == "__main__":
    main()