"""
cobb_curve.py  —  SpineCurve-net pipeline
==========================================
Outputs two PNGs (overwritten if they exist):

  <SUBJECT_ID>_sagittal.png
      Mean slab ± SAG_SLAB_HALF slices along the L-R axis.

  <SUBJECT_ID>_coronal.png
      Centroid-guided curved MPR along the A-P axis.
      For each SI row in the image the AP slab centre is interpolated from
      the centroid AP positions, so the view follows the spine's curvature
      (cervical lordosis, thoracic kyphosis) and every vertebra is visible.
      A fixed flat slab fails here because this patient's centroids span
      ~122 voxels in AP depth due to natural spinal curvature.

Change only SUBJECT_ID to run on a different subject.

Run:
    python cobb_curve.py
"""

import json
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

# =============================================================================
# PARAMETERS — only SUBJECT_ID needs to change between subjects
# =============================================================================

SUBJECT_ID = "sub-gl003"

VERSE20_ROOT = "/gscratch/scrubbed/bhan830/wisespine/data/Verse20/dataset-01training"
OUT_ROOT     = "/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/cobb_angles/TotalSegmentator"

CT_PATH  = f"{VERSE20_ROOT}/rawdata/{SUBJECT_ID}/{SUBJECT_ID}_dir-ax_ct.nii.gz"
CTD_PATH = f"{VERSE20_ROOT}/derivatives/{SUBJECT_ID}/{SUBJECT_ID}_dir-ax_seg-subreg_ctd.json"
OUT_DIR  = f"{OUT_ROOT}/{SUBJECT_ID}"

# Sagittal: mean slab ± this many slices around median centroid L-R position
SAG_SLAB_HALF = 5

# Coronal curved MPR: slab half-width at each SI level (follows centroid curve)
COR_SLAB_HALF = 5

# CT display window (Hounsfield units) — bone window
CT_WL = 400
CT_WW = 1800

# Centroid markers
MARKER_SIZE     = 8
LABEL_FONT_SIZE = 8.0
LABEL_OFFSET_PX = 6

FIG_DPI      = 150
FIG_BG       = "#111111"
BORDER_COLOR = "#444444"

# =============================================================================
# VERTEBRA LABELS & COLOURS
# =============================================================================

VERTEBRA_LABELS = {
    1:  "C1",  2:  "C2",  3:  "C3",  4:  "C4",  5:  "C5",  6:  "C6",  7:  "C7",
    8:  "T1",  9:  "T2",  10: "T3",  11: "T4",  12: "T5",  13: "T6",  14: "T7",
    15: "T8",  16: "T9",  17: "T10", 18: "T11", 19: "T12",
    20: "L1",  21: "L2",  22: "L3",  23: "L4",  24: "L5",
    25: "S1",  26: "S2",  28: "L6",
}
REGION_COLORS = {
    "C": "#4A90D9",
    "T": "#E67E22",
    "L": "#27AE60",
    "S": "#8E44AD",
}

def region_color(name: str) -> str:
    return REGION_COLORS.get(name[0], "#AAAAAA")


# =============================================================================
# I/O
# =============================================================================

def load_ct(path: str):
    img  = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img


def load_centroids(json_path: str) -> list:
    with open(json_path) as f:
        raw = json.load(f)
    direction = None
    centroids = []
    for entry in raw:
        if "direction" in entry:
            direction = entry["direction"]
            continue
        lbl  = int(entry["label"])
        vox  = np.array([entry["X"], entry["Y"], entry["Z"]], dtype=np.float64)
        name = VERTEBRA_LABELS.get(lbl, f"V{lbl}")
        centroids.append({"label": lbl, "name": name, "vox": vox})
    print(f"  Direction : {direction}")
    return centroids


# =============================================================================
# AXIS ROLES FROM AFFINE
# =============================================================================

def get_axis_roles(affine: np.ndarray):
    rot           = np.abs(affine[:3, :3])
    sag_axis      = int(np.argmax(rot[0, :]))
    horiz_axis    = int(np.argmax(rot[1, :]))
    vert_axis     = int(np.argmax(rot[2, :]))
    sup_ascending = affine[2, vert_axis] > 0
    return sag_axis, horiz_axis, vert_axis, sup_ascending


# =============================================================================
# CT WINDOWING
# =============================================================================

def window_ct(data: np.ndarray) -> np.ndarray:
    lo = CT_WL - CT_WW / 2.0
    hi = CT_WL + CT_WW / 2.0
    return (np.clip(data, lo, hi) - lo) / (hi - lo)


# =============================================================================
# SPLINE
# =============================================================================

def fit_spline(cols, rows, n_pts=600):
    if len(cols) < 2:
        return None, None
    pts   = np.c_[cols, rows].astype(float)
    diffs = np.diff(pts, axis=0)
    chord = np.concatenate([[0.], np.cumsum(np.linalg.norm(diffs, axis=1))])
    t     = chord / chord[-1]
    t_f   = np.linspace(0., 1., n_pts)
    return CubicSpline(t, cols)(t_f), CubicSpline(t, rows)(t_f)


# =============================================================================
# RENDER
# =============================================================================

def save_png(img, centroids, col_fn, row_fn, title, suptitle, out_path):
    n_rows, n_cols = img.shape
    aspect = n_rows / max(n_cols, 1)
    fig, ax = plt.subplots(figsize=(7, max(8, 7 * aspect * 0.9)),
                           facecolor=FIG_BG)
    fig.subplots_adjust(left=0.02, right=0.93, top=0.93, bottom=0.08)

    ax.imshow(img, cmap="gray", aspect="auto", vmin=0, vmax=1, origin="upper")
    ax.set_facecolor(FIG_BG)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER_COLOR)
    ax.set_title(title, color="white", fontsize=8, pad=5)

    cols = [col_fn(c["vox"]) for c in centroids]
    rows = [row_fn(c["vox"]) for c in centroids]

    cx, cy = fit_spline(cols, rows)
    if cx is not None:
        ax.plot(cx, cy, "-", color="#FFD700", lw=2.0, alpha=0.85, zorder=4)

    for c, px, py in zip(centroids, cols, rows):
        clr = region_color(c["name"])
        ax.plot(px, py, "o", color=clr, markersize=MARKER_SIZE,
                markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        ax.text(px + LABEL_OFFSET_PX, py, c["name"],
                color=clr, fontsize=LABEL_FONT_SIZE,
                va="center", fontweight="bold", zorder=6)

    fig.suptitle(suptitle, color="white", fontsize=11, y=0.97)

    patches = [
        mpatches.Patch(color=REGION_COLORS["C"], label="Cervical"),
        mpatches.Patch(color=REGION_COLORS["T"], label="Thoracic"),
        mpatches.Patch(color=REGION_COLORS["L"], label="Lumbar"),
        mpatches.Patch(color=REGION_COLORS["S"], label="Sacral"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               framealpha=0.3, facecolor="#222222",
               labelcolor="white", fontsize=9,
               bbox_to_anchor=(0.5, 0.005))

    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor=FIG_BG)
    print(f"Saved  ->  {out_path}")
    plt.close(fig)


# =============================================================================
# VIEW BUILDERS
# =============================================================================

def build_sagittal(ct_win, centroids,
                   sag_axis, horiz_axis, vert_axis, sup_ascending, out_path):
    """
    Centroid-guided curved MPR along the L-R axis.

    Scoliosis causes lateral curvature, so each vertebra sits at a different
    L-R position.  A fixed flat slab centred at the median L-R misses
    off-centre vertebrae the same way the flat coronal slab did in AP.

    For every SI row in the output image the L-R slab centre is interpolated
    from the centroid (vert, sag) pairs, so the view follows the spine's
    lateral curve and every vertebra is visible.

    Screen cols = A-P axis (horiz_axis), rows = S-I axis (vert_axis).
    """
    # reorder volume to (horiz, sag, vert) for straightforward indexing
    order      = [horiz_axis, sag_axis, vert_axis]
    ct_reorder = np.transpose(ct_win, order)          # (n_horiz, n_sag, n_vert)
    n_horiz, n_sag, n_vert = ct_reorder.shape

    # build L-R position interpolator from centroid (vert, sag) pairs
    verts = np.array([c["vox"][vert_axis] for c in centroids])
    sags  = np.array([c["vox"][sag_axis]  for c in centroids])
    sort  = np.argsort(verts)
    lr_interp = interp1d(
        verts[sort], sags[sort],
        kind="linear",
        bounds_error=False,
        fill_value=(sags[sort][0], sags[sort][-1]),   # clamp at endpoints
    )

    # build curved MPR row by row
    mpr = np.zeros((n_horiz, n_vert), dtype=np.float32)
    for z in range(n_vert):
        lr_centre = float(lr_interp(z))
        lo = max(0,        int(round(lr_centre)) - SAG_SLAB_HALF)
        hi = min(n_sag-1,  int(round(lr_centre)) + SAG_SLAB_HALF)
        mpr[:, z] = ct_reorder[:, lo:hi+1, z].mean(axis=1)

    # orient for display: (n_vert, n_horiz), Superior at top
    img = mpr.T                      # (n_vert, n_horiz)
    if sup_ascending:
        img    = img[::-1, :]
        row_fn = lambda v: n_vert - 1 - v[vert_axis]
    else:
        row_fn = lambda v: v[vert_axis]
    col_fn = lambda v: v[horiz_axis]

    lr_range = f"{sags.min():.0f}–{sags.max():.0f}"
    save_png(
        img, centroids, col_fn, row_fn,
        title=(f"Curved MPR  ± {SAG_SLAB_HALF} slices  |  "
               f"axis {sag_axis} (L-R)  centroid range [{lr_range}]  |  "
               f"all {len(centroids)} vertebrae"),
        suptitle=f"{SUBJECT_ID}  —  Sagittal  ({len(centroids)} vertebrae)",
        out_path=out_path,
    )


def build_coronal(ct_win, centroids,
                  sag_axis, horiz_axis, vert_axis, sup_ascending, out_path):
    """
    Centroid-guided curved MPR.

    The spine's cervical lordosis and thoracic kyphosis mean vertebral bodies
    span a wide AP range — a fixed flat slab captures only a fraction of them.

    Instead, for every SI row in the output image the AP slab centre is
    linearly interpolated from the centroid (vert, horiz) pairs.  The result
    follows the spine's own curvature so every vertebra is visible while the
    local slab half-width (COR_SLAB_HALF) stays narrow enough to exclude
    most off-spine structures.

    Outside the SI range covered by centroids the AP position is held constant
    at the nearest endpoint centroid value.

    Screen cols = L-R axis (sag_axis), rows = S-I axis (vert_axis).
    """
    # reorder volume to (sag, horiz, vert) for straightforward indexing
    order      = [sag_axis, horiz_axis, vert_axis]
    ct_reorder = np.transpose(ct_win, order)          # (n_sag, n_horiz, n_vert)
    n_sag, n_horiz, n_vert = ct_reorder.shape

    # build AP-position interpolator from centroid (vert, horiz) pairs
    verts  = np.array([c["vox"][vert_axis]  for c in centroids])
    horizs = np.array([c["vox"][horiz_axis] for c in centroids])
    sort   = np.argsort(verts)
    ap_interp = interp1d(
        verts[sort], horizs[sort],
        kind="linear",
        bounds_error=False,
        fill_value=(horizs[sort][0], horizs[sort][-1]),  # clamp at endpoints
    )

    # build curved MPR row by row
    mpr = np.zeros((n_sag, n_vert), dtype=np.float32)
    for z in range(n_vert):
        ap_centre = float(ap_interp(z))
        lo = max(0,         int(round(ap_centre)) - COR_SLAB_HALF)
        hi = min(n_horiz-1, int(round(ap_centre)) + COR_SLAB_HALF)
        mpr[:, z] = ct_reorder[:, lo:hi+1, z].mean(axis=1)

    # orient for display: (n_vert, n_sag), Superior at top
    img = mpr.T                      # (n_vert, n_sag)
    if sup_ascending:
        img    = img[::-1, :]
        row_fn = lambda v: n_vert - 1 - v[vert_axis]
    else:
        row_fn = lambda v: v[vert_axis]
    col_fn = lambda v: v[sag_axis]

    ap_range = f"{horizs.min():.0f}–{horizs.max():.0f}"
    save_png(
        img, centroids, col_fn, row_fn,
        title=(f"Curved MPR  ± {COR_SLAB_HALF} slices  |  "
               f"axis {horiz_axis} (A-P)  centroid range [{ap_range}]  |  "
               f"all {len(centroids)} vertebrae"),
        suptitle=f"{SUBJECT_ID}  —  Coronal  ({len(centroids)} vertebrae)",
        out_path=out_path,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n=== cobb_curve.py — {SUBJECT_ID} ===")

    for label, fp in [("CT", CT_PATH), ("Centroids", CTD_PATH)]:
        if not Path(fp).exists():
            sys.exit(f"ERROR: {label} file not found:\n  {fp}")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading CT:\n  {CT_PATH}")
    ct_data, ct_img = load_ct(CT_PATH)
    zooms = np.array(ct_img.header.get_zooms()[:3])
    print(f"  Shape      : {ct_data.shape}")
    print(f"  Voxel size : {zooms} mm")

    sag_axis, horiz_axis, vert_axis, sup_ascending = get_axis_roles(ct_img.affine)
    print(f"\nAxis roles  — sag={sag_axis} (L-R)  horiz={horiz_axis} (A-P)  "
          f"vert={vert_axis} (S-I)  sup_asc={sup_ascending}")

    print(f"\nLoading centroids:\n  {CTD_PATH}")
    centroids = load_centroids(CTD_PATH)
    centroids.sort(key=lambda c: -c["vox"][vert_axis] if sup_ascending
                                 else  c["vox"][vert_axis])
    print(f"  {len(centroids)} vertebrae: " + "  ".join(c["name"] for c in centroids))
    print()
    for c in centroids:
        v = c["vox"]
        print(f"  {c['name']:4s}  sag={v[sag_axis]:.1f}  "
              f"horiz={v[horiz_axis]:.1f}  vert={v[vert_axis]:.1f}")

    ct_win = window_ct(ct_data)

    print(f"\nRendering sagittal...")
    build_sagittal(ct_win, centroids,
                   sag_axis, horiz_axis, vert_axis, sup_ascending,
                   out_dir / f"{SUBJECT_ID}_sagittal.png")

    print(f"\nRendering coronal (curved MPR)...")
    build_coronal(ct_win, centroids,
                  sag_axis, horiz_axis, vert_axis, sup_ascending,
                  out_dir / f"{SUBJECT_ID}_coronal.png")

    print("\nDone.\n")


if __name__ == "__main__":
    main()