"""
fitted_lines.py

Step 2 of the Cobb angle pipeline.

Reads the bounding box corners from the JSON produced by bounding_box.py,
draws a line across the superior endplate of the upper vertebra and the
inferior endplate of the lower vertebra, extends both lines to their
intersection, and labels the Cobb angle.

Usage
-----
    python fitted_lines.py <UPPER> <LOWER>

    python fitted_lines.py C2 C5
    python fitted_lines.py T1 T6

The upper vertebra contributes its SUPERIOR endplate (top edge of its OBB).
The lower vertebra contributes its INFERIOR endplate (bottom edge of its OBB).
The script will auto-correct if you pass them in the wrong anatomical order.

Outputs
-------
    <OUTPUT_DIR>/<SUBJECT>_cobb_<UPPER>_<LOWER>.png

Corner convention (from bounding_box.py JSON)
---------------------------------------------
    index 0 = bottom_left   (inferior left  corner of OBB)
    index 1 = top_left      (superior left  corner of OBB)
    index 2 = top_right     (superior right corner of OBB)
    index 3 = bottom_right  (inferior right corner of OBB)

    Superior endplate = edge from index 1 -> index 2
    Inferior endplate = edge from index 0 -> index 3

Dependencies
------------
    pip install nibabel numpy matplotlib scipy
"""

import json
import math
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Arc, Polygon

# ── Configuration — must match bounding_box.py ────────────────────────────────

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

# Vertebrae to measure — change these to adjust the Cobb angle pair
UPPER_VERTEBRA = "C2"
LOWER_VERTEBRA = "C5"

# JSON produced by bounding_box.py
BOXES_JSON = OUTPUT_DIR / f"{SUBJECT}_vertebra_boxes.json"

# ── Colours ───────────────────────────────────────────────────────────────────

UPPER_COL = "#00d4ff"   # cyan   — superior endplate of upper vertebra
LOWER_COL = "#ff6b35"   # orange — inferior endplate of lower vertebra
ARC_COL   = "#f5e642"   # yellow — angle arc and label
BG_COL    = "#0d0d1a"
LABEL_PAD = dict(boxstyle="round,pad=0.2", linewidth=0)

FIGURE_LONG_IN = 20

# ── Anatomical sort order ─────────────────────────────────────────────────────

SORT_ORDER = {
    **{f"C{i}": i - 1  for i in range(1, 8)},
    **{f"T{i}": i + 6  for i in range(1, 13)},
    **{f"L{i}": i + 18 for i in range(1, 6)},
}

SPINE_Y_MARGIN_PCT = 5


# ─────────────────────────────────────────────────────────────────────────────
# Load helpers  (identical to bounding_box.py so background is pixel-perfect)
# ─────────────────────────────────────────────────────────────────────────────

def _to_ras(img: nib.Nifti1Image) -> tuple:
    ras   = nib.as_closest_canonical(img)
    data  = np.asarray(ras.dataobj)
    zooms = np.array(ras.header.get_zooms()[:3], dtype=float)
    return data, zooms


def load_ct(path: Path) -> tuple:
    if not path.exists():
        print(f"[CT] Not found — using mask silhouette fallback.")
        return None, None
    img  = nib.load(str(path))
    data, zooms = _to_ras(img)
    print(f"[CT] shape={data.shape}  zooms={np.round(zooms, 3)} mm")
    return data.astype(np.float32), zooms


def load_masks(mask_dir: Path) -> dict:
    import re
    pattern = re.compile(r"vertebrae_([A-Z]\d+)")
    masks   = {}
    for f in sorted(mask_dir.glob("*_vertebrae_*.nii.gz")):
        m = pattern.search(f.name)
        if m:
            data, _ = _to_ras(nib.load(str(f)))
            masks[m.group(1)] = (data > 0).astype(np.uint8)
    if not masks:
        sys.exit(f"[ERROR] No masks in {mask_dir}")
    return masks


def spine_y_range(masks: dict) -> tuple:
    first = next(iter(masks.values()))
    ny    = first.shape[1]
    y_sum = np.zeros(ny, dtype=np.float64)
    for data in masks.values():
        y_sum += data.sum(axis=(0, 2))
    occupied = np.where(y_sum > 0)[0]
    if len(occupied) == 0:
        return 0, ny - 1
    cum   = np.cumsum(y_sum[occupied])
    total = cum[-1]
    y_lo  = int(occupied[np.searchsorted(cum, total * SPINE_Y_MARGIN_PCT / 100)])
    y_hi  = int(occupied[np.searchsorted(cum, total * (1 - SPINE_Y_MARGIN_PCT / 100),
                                          side="right") - 1])
    return y_lo, y_hi


def build_background(ct_data, masks: dict, y_lo: int, y_hi: int) -> np.ndarray:
    if ct_data is not None:
        return ct_data[:, y_lo:y_hi + 1, :].max(axis=1)
    first = next(iter(masks.values()))
    combo = np.zeros(first.shape, dtype=np.uint8)
    for data in masks.values():
        np.maximum(combo, data, out=combo)
    return combo.max(axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def line_intersection(p1: np.ndarray, d1: np.ndarray,
                      p2: np.ndarray, d2: np.ndarray) -> np.ndarray | None:
    """
    Intersection of two infinite 2D lines:
        L1 = p1 + t * d1
        L2 = p2 + s * d2
    Returns the intersection point or None if lines are parallel.
    """
    A = np.column_stack([d1, -d2])
    if abs(np.linalg.det(A)) < 1e-10:
        return None
    t = np.linalg.solve(A, p2 - p1)[0]
    return p1 + t * d1


def cobb_angle(d1: np.ndarray, d2: np.ndarray) -> float:
    """
    Angle in degrees between two direction vectors.
    Returns the acute angle (0–90°) following standard Cobb convention.
    """
    v1  = d1 / np.linalg.norm(d1)
    v2  = d2 / np.linalg.norm(d2)
    dot = np.clip(abs(np.dot(v1, v2)), 0.0, 1.0)
    return math.degrees(math.acos(dot))


def extend_line(p_a: np.ndarray, p_b: np.ndarray,
                x_min: float, x_max: float) -> tuple:
    """
    Extend the line through p_a and p_b to the full horizontal extent
    [x_min, x_max].  Returns (left_pt, right_pt).
    """
    d = p_b - p_a
    if abs(d[0]) < 1e-9:                    # vertical line
        return (np.array([p_a[0], x_min]),
                np.array([p_a[0], x_max]))
    t0 = (x_min - p_a[0]) / d[0]
    t1 = (x_max - p_a[0]) / d[0]
    return p_a + t0 * d, p_a + t1 * d


# ─────────────────────────────────────────────────────────────────────────────
# Main drawing function
# ─────────────────────────────────────────────────────────────────────────────

def draw(bg: np.ndarray,
         all_obbs: dict,
         upper_name: str,
         lower_name: str,
         vox_x: float,
         vox_z: float,
         out_path: Path) -> None:
    """
    Draw the CT background, all OBB outlines, the two endplate lines,
    their intersection, and the Cobb angle arc on a single PNG.
    """

    nx, nz        = bg.shape
    imshow_aspect = vox_z / vox_x

    # ── Figure ────────────────────────────────────────────────────────────────
    phys_w = nx * vox_x
    phys_h = nz * vox_z
    ratio  = phys_w / phys_h
    if ratio >= 1.0:
        fig_w, fig_h = FIGURE_LONG_IN, FIGURE_LONG_IN / ratio
    else:
        fig_w, fig_h = FIGURE_LONG_IN * ratio, FIGURE_LONG_IN

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG_COL)
    ax.set_facecolor(BG_COL)

    # ── CT background ─────────────────────────────────────────────────────────
    fg      = bg[bg > 0]
    p2, p98 = (np.percentile(fg, [2, 98]) if len(fg) else (0.0, 1.0))
    disp    = np.clip((bg - p2) / max(p98 - p2, 1e-9), 0.0, 1.0)

    ax.imshow(disp.T, cmap="gray", origin="lower",
              aspect=imshow_aspect, interpolation="bilinear",
              vmin=0, vmax=1)

    cmap_v = plt.colormaps["tab20"].resampled(max(len(all_obbs), 1))

    # ── All OBBs (light outlines for context) ────────────────────────────────
    all_cx, all_cz = [], []

    for idx, (name, obb) in enumerate(
        sorted(all_obbs.items(), key=lambda x: SORT_ORDER.get(x[0], 99))
    ):
        colour  = cmap_v(idx)
        corners = np.array(obb["corners_vox"])   # (4, 2)
        is_sel  = name in (upper_name, lower_name)

        all_cx.extend(corners[:, 0])
        all_cz.extend(corners[:, 1])

        ax.add_patch(Polygon(
            corners, closed=True,
            linewidth=2.0 if is_sel else 0.8,
            edgecolor=(*colour[:3], 0.95 if is_sel else 0.5),
            facecolor=(*colour[:3], 0.18 if is_sel else 0.05),
            zorder=3,
        ))

        cx, cz = corners[:, 0].mean(), corners[:, 1].mean()
        ax.text(cx, cz, name, color="white",
                fontsize=7 if is_sel else 5,
                fontweight="bold" if is_sel else "normal",
                ha="center", va="center", zorder=4,
                bbox={**LABEL_PAD, "facecolor": colour,
                      "alpha": 0.80 if is_sel else 0.35})

    # ── Endplate corners ──────────────────────────────────────────────────────
    # Corner order: [bottom_left(0), top_left(1), top_right(2), bottom_right(3)]
    # Superior endplate = top edge    = corners[1] -> corners[2]
    # Inferior endplate = bottom edge = corners[0] -> corners[3]

    upper_corners = np.array(all_obbs[upper_name]["corners_vox"])
    lower_corners = np.array(all_obbs[lower_name]["corners_vox"])

    sup_l = upper_corners[1]   # top-left  of upper OBB
    sup_r = upper_corners[2]   # top-right of upper OBB
    inf_l = lower_corners[0]   # bottom-left  of lower OBB
    inf_r = lower_corners[3]   # bottom-right of lower OBB

    sup_dir = sup_r - sup_l
    inf_dir = inf_r - inf_l

    # Scale voxel direction vectors to mm before computing angle.
    # Voxels are anisotropic (x=0.291mm, z=1.25mm — 4.3x difference).
    # Computing the angle in raw voxel space underweights the Z component
    # and produces a significantly wrong angle (e.g. 2.7° instead of 11.6°).
    sup_dir_mm = sup_dir * np.array([vox_x, vox_z])
    inf_dir_mm = inf_dir * np.array([vox_x, vox_z])
    angle_deg  = cobb_angle(sup_dir_mm, inf_dir_mm)

    # ── Compute intersection before setting any limits ────────────────────────
    intersect = line_intersection(sup_l, sup_dir, inf_l, inf_dir)

    # ── Build view bounds that always include the intersection ────────────────
    # Start from the spine OBB extent, then expand to include the intersection
    # point (plus margin) so there is never empty clipping on that side.
    if all_cx:
        pad_x = (max(all_cx) - min(all_cx)) * 0.12
        pad_z = (max(all_cz) - min(all_cz)) * 0.08
        view_x0 = min(all_cx) - pad_x
        view_x1 = max(all_cx) + pad_x
        view_z0 = min(all_cz) - pad_z
        view_z1 = max(all_cz) + pad_z
    else:
        view_x0, view_x1 = 0.0, float(nx - 1)
        view_z0, view_z1 = 0.0, float(nz - 1)

    avg_h = np.mean([
        max(np.array(v["corners_vox"])[:, 1]) -
        min(np.array(v["corners_vox"])[:, 1])
        for v in all_obbs.values()
    ])
    arc_r = avg_h * 0.8

    if intersect is not None:
        # Expand view to contain intersection + arc + label with comfortable margin
        margin = arc_r * 4.0
        view_x0 = min(view_x0, intersect[0] - margin)
        view_x1 = max(view_x1, intersect[0] + margin)
        view_z0 = min(view_z0, intersect[1] - margin)
        view_z1 = max(view_z1, intersect[1] + margin)

    # ── Extend lines to the full (possibly expanded) view width ──────────────
    for p_l, p_r, col, label in [
        (sup_l, sup_r, UPPER_COL, f"{upper_name} superior endplate"),
        (inf_l, inf_r, LOWER_COL, f"{lower_name} inferior endplate"),
    ]:
        ext_l, ext_r = extend_line(p_l, p_r, view_x0, view_x1)

        # Dashed extended line
        ax.plot([ext_l[0], ext_r[0]], [ext_l[1], ext_r[1]],
                color=col, lw=1.2, ls="--", alpha=0.6, zorder=6)

        # Solid line across the vertebra body
        ax.plot([p_l[0], p_r[0]], [p_l[1], p_r[1]],
                color=col, lw=3.5, ls="-", label=label, zorder=7)

        # Corner dots
        for pt in (p_l, p_r):
            ax.plot(pt[0], pt[1], "o", color=col,
                    ms=7, mec="white", mew=1.0, zorder=8)

    # ── Intersection + angle arc + label inside the angle ────────────────────
    if intersect is not None:
        ax.plot(intersect[0], intersect[1], "+",
                color=ARC_COL, ms=14, mew=2.0, zorder=9)

        # Arc angles corrected for imshow aspect ratio
        ar = imshow_aspect
        a1 = math.degrees(math.atan2(sup_dir[1] * ar, sup_dir[0]))
        a2 = math.degrees(math.atan2(inf_dir[1] * ar, inf_dir[0]))

        a_lo, a_hi = sorted([a1 % 180, a2 % 180])
        if (a_hi - a_lo) > 90:
            a_lo, a_hi = a_hi, a_lo + 180

        ax.add_patch(Arc(
            intersect, 2 * arc_r, 2 * arc_r,
            angle=0, theta1=a_lo, theta2=a_hi,
            color=ARC_COL, lw=2.0, zorder=9,
        ))

        # Label placed INSIDE the angle at the arc midpoint
        mid_rad = math.radians((a_lo + a_hi) / 2)
        lx = intersect[0] + arc_r * 1.5 * math.cos(mid_rad)
        lz = intersect[1] + arc_r * 1.5 * math.sin(mid_rad)
    else:
        # Parallel endplates — place label between midpoints
        mid = ((sup_l + sup_r) / 2 + (inf_l + inf_r) / 2) / 2
        lx, lz = float(mid[0]), float(mid[1])

    ax.text(lx, lz, f"{angle_deg:.1f}°",
            color=ARC_COL, fontsize=15, fontweight="bold",
            ha="center", va="center", zorder=10,
            bbox={**LABEL_PAD, "facecolor": BG_COL, "alpha": 0.85})

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.legend(
        handles=[
            mpatches.Patch(color=UPPER_COL,
                           label=f"{upper_name}  superior endplate"),
            mpatches.Patch(color=LOWER_COL,
                           label=f"{lower_name}  inferior endplate"),
            mpatches.Patch(color=ARC_COL,
                           label=f"Cobb angle = {angle_deg:.1f}°"),
        ],
        loc="upper right",
        facecolor="#1a1a2e", edgecolor="gray",
        labelcolor="white", fontsize=9,
    )

    # Set axis limits LAST — expanded to include intersection if needed
    ax.set_xlim(view_x0, view_x1)
    ax.set_ylim(view_z0, view_z1)
    ax.axis("off")
    plt.tight_layout(pad=0.5)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[PNG] Overwriting {out_path.name}")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), format="png")
    plt.close()
    print(f"[PNG] Saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Vertebrae from config ─────────────────────────────────────────────────
    upper_arg = UPPER_VERTEBRA.upper()
    lower_arg = LOWER_VERTEBRA.upper()

    # ── Load JSON ─────────────────────────────────────────────────────────────
    if not BOXES_JSON.exists():
        sys.exit(
            f"[ERROR] Boxes JSON not found:\n        {BOXES_JSON}\n"
            f"        Run bounding_box.py first."
        )

    with open(BOXES_JSON) as f:
        data = json.load(f)

    all_obbs  = data["vertebrae"]
    vox_x     = data["voxel_spacing"]["x_mm"]
    vox_y     = data["voxel_spacing"]["y_mm"]
    vox_z     = data["voxel_spacing"]["z_mm"]
    # Support both key names for backwards compatibility
    y_range  = data.get("coronal_y_range") or data.get("coronal_y_idx")
    if isinstance(y_range, list):
        y_lo, y_hi = y_range[0], y_range[1]
    else:
        y_lo = y_hi = int(y_range)

    available = sorted(all_obbs.keys(), key=lambda x: SORT_ORDER.get(x, 99))

    # ── Validate ──────────────────────────────────────────────────────────────
    for name, label in [(upper_arg, "UPPER"), (lower_arg, "LOWER")]:
        if name not in all_obbs:
            print(f"[ERROR] '{name}' ({label}) not found in JSON.")
            print(f"        Available: {', '.join(available)}")
            sys.exit(1)

    # Auto-correct anatomical order
    if SORT_ORDER.get(upper_arg, 99) > SORT_ORDER.get(lower_arg, 99):
        upper_arg, lower_arg = lower_arg, upper_arg
        print(f"[INFO] Reordered to anatomical order: "
              f"{upper_arg} (upper) -> {lower_arg} (lower)")

    upper_name, lower_name = upper_arg, lower_arg

    # ── Rebuild background (identical to bounding_box.py) ────────────────────
    print(f"\n[Load] Rebuilding CT background  (Y={y_lo}–{y_hi})")
    ct_data, ct_zooms = load_ct(CT_PATH)
    masks             = load_masks(MASK_DIR)
    bg                = build_background(ct_data, masks, y_lo, y_hi)

    # ── Compute angle (printed to CLI) ────────────────────────────────────────
    upper_corners = np.array(all_obbs[upper_name]["corners_vox"])
    lower_corners = np.array(all_obbs[lower_name]["corners_vox"])

    sup_dir    = upper_corners[2] - upper_corners[1]
    inf_dir    = lower_corners[3] - lower_corners[0]
    sup_dir_mm = sup_dir * np.array([vox_x, vox_z])
    inf_dir_mm = inf_dir * np.array([vox_x, vox_z])
    angle      = cobb_angle(sup_dir_mm, inf_dir_mm)

    border = "─" * 46
    print(f"\n  ┌{border}┐")
    print(f"  │  Cobb angle  {upper_name} → {lower_name} :  {angle:6.2f}°"
          f"{'':>{46 - 27 - len(upper_name) - len(lower_name)}}│")
    print(f"  └{border}┘")
    print(f"\n  Superior endplate : {upper_name}  "
          f"(corners {upper_corners[1].tolist()} -> {upper_corners[2].tolist()})")
    print(f"  Inferior endplate : {lower_name}  "
          f"(corners {lower_corners[0].tolist()} -> {lower_corners[3].tolist()})\n")

    # ── Draw and save ─────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / f"{SUBJECT}_cobb_{upper_name}_{lower_name}.png"
    draw(bg, all_obbs, upper_name, lower_name, vox_x, vox_z, out_path)


if __name__ == "__main__":
    main()