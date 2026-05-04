"""
cobb_curve.py  —  SpineCurve-net pipeline
==========================================
Outputs two PNGs (overwritten if they exist):

  <SUBJECT_ID>_sagittal.png   — centroid-guided curved MPR, L-R axis
  <SUBJECT_ID>_coronal.png    — centroid-guided curved MPR, A-P axis

Both views:
  • Centroid dots + labels + cubic spline through all centroids
  • Endplate lines extended to full image width at VERT_A and VERT_B
  • If lines intersect inside image: arc at intersection (classical Cobb diagram)
  • If lines are nearly parallel: mini angle indicator at midpoint
  • Zoomed inset of the measurement region in bottom-right corner
  • Severity label (Normal / Mild / Moderate / Severe)

Change SUBJECT_ID, VERT_A, VERT_B to run on a different subject or segment.

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
# PARAMETERS — edit these
# =============================================================================

SUBJECT_ID = "sub-gl003"

VERT_A = "C4"   # superior vertebra for Cobb measurement
VERT_B = "T5"   # inferior vertebra for Cobb measurement

VERSE20_ROOT = "/gscratch/scrubbed/bhan830/wisespine/data/Verse20/dataset-01training"
OUT_ROOT     = "/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/cobb_angles/TotalSegmentator"

CT_PATH  = f"{VERSE20_ROOT}/rawdata/{SUBJECT_ID}/{SUBJECT_ID}_dir-ax_ct.nii.gz"
CTD_PATH = f"{VERSE20_ROOT}/derivatives/{SUBJECT_ID}/{SUBJECT_ID}_dir-ax_seg-subreg_ctd.json"
OUT_DIR  = f"{OUT_ROOT}/{SUBJECT_ID}"

# Curved MPR slab half-width in slices
SAG_SLAB_HALF = 5
COR_SLAB_HALF = 5

# Zoom inset: extra pixels around the two Cobb vertebrae
INSET_PAD = 45

# CT window — bone window
CT_WL = 400
CT_WW = 1800

MARKER_SIZE     = 8
LABEL_FONT_SIZE = 8.0
LABEL_OFFSET_PX = 6
FIG_DPI         = 150
FIG_BG          = "#111111"
BORDER_COLOR    = "#444444"

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

def load_ct(path):
    img  = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img

def load_centroids(json_path):
    with open(json_path) as f:
        raw = json.load(f)
    direction, centroids = None, []
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

def get_axis_roles(affine):
    rot           = np.abs(affine[:3, :3])
    sag_axis      = int(np.argmax(rot[0, :]))
    horiz_axis    = int(np.argmax(rot[1, :]))
    vert_axis     = int(np.argmax(rot[2, :]))
    sup_ascending = affine[2, vert_axis] > 0
    return sag_axis, horiz_axis, vert_axis, sup_ascending

def window_ct(data):
    lo = CT_WL - CT_WW / 2.0
    hi = CT_WL + CT_WW / 2.0
    return (np.clip(data, lo, hi) - lo) / (hi - lo)


# =============================================================================
# CURVED MPR
# =============================================================================

def curved_mpr(ct_win, centroids, slice_axis, col_axis, row_axis,
               sup_ascending, slab_half):
    order = [col_axis, slice_axis, row_axis]
    ct_t  = np.transpose(ct_win, order)
    n_col, n_slice, n_row = ct_t.shape

    rows_c   = np.array([c["vox"][row_axis]   for c in centroids])
    slices_c = np.array([c["vox"][slice_axis] for c in centroids])
    sort     = np.argsort(rows_c)
    interp   = interp1d(rows_c[sort], slices_c[sort], kind="linear",
                        bounds_error=False,
                        fill_value=(slices_c[sort][0], slices_c[sort][-1]))

    mpr = np.zeros((n_col, n_row), dtype=np.float32)
    for z in range(n_row):
        centre = float(interp(z))
        lo = max(0,         int(round(centre)) - slab_half)
        hi = min(n_slice-1, int(round(centre)) + slab_half)
        mpr[:, z] = ct_t[:, lo:hi+1, z].mean(axis=1)

    img = mpr.T
    if sup_ascending:
        img    = img[::-1, :]
        row_fn = lambda v: n_row - 1 - v[row_axis]
    else:
        row_fn = lambda v: v[row_axis]
    col_fn = lambda v: v[col_axis]
    return img, col_fn, row_fn


# =============================================================================
# SPLINE (mm space)
# =============================================================================

def fit_spline_mm(cols_px, rows_px, zoom_col, zoom_row, n_pts=600):
    """
    Fit spline in mm space to correctly handle anisotropic voxel spacing.
    Returns dense pixel curve arrays + spline objects for tangent queries.
    """
    cols_mm = np.array(cols_px, dtype=float) * zoom_col
    rows_mm = np.array(rows_px, dtype=float) * zoom_row
    pts     = np.c_[cols_mm, rows_mm]
    diffs   = np.diff(pts, axis=0)
    chord   = np.concatenate([[0.], np.cumsum(np.linalg.norm(diffs, axis=1))])
    t       = chord / chord[-1]
    cs_col  = CubicSpline(t, cols_mm)
    cs_row  = CubicSpline(t, rows_mm)
    t_f     = np.linspace(0., 1., n_pts)
    return cs_col(t_f)/zoom_col, cs_row(t_f)/zoom_row, cs_col, cs_row, t


# =============================================================================
# ENDPLATE DIRECTION
# =============================================================================

def endplate_direction(cs_col_mm, cs_row_mm, t_val, zoom_col, zoom_row):
    """
    Perpendicular to the spline tangent at t_val.
    Returns pixel-space direction (for drawing) and mm-space direction (for angle).
    """
    dc = float(cs_col_mm.derivative()(t_val))
    dr = float(cs_row_mm.derivative()(t_val))
    # perpendicular in mm space
    ep_c_mm, ep_r_mm = -dr, dc
    n = np.hypot(ep_c_mm, ep_r_mm)
    ep_c_mm /= n;  ep_r_mm /= n
    # convert to pixel space
    ep_c_px = ep_c_mm / zoom_col
    ep_r_px = ep_r_mm / zoom_row
    n = np.hypot(ep_c_px, ep_r_px)
    ep_c_px /= n;  ep_r_px /= n
    return ep_c_px, ep_r_px, ep_c_mm, ep_r_mm


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def clip_line_to_box(px, py, dx, dy, n_rows, n_cols):
    """Clip infinite line through (px,py) with direction (dx,dy) to image box."""
    t_lo, t_hi = -1e9, 1e9
    if abs(dx) > 1e-10:
        ta, tb = (0-px)/dx, (n_cols-1-px)/dx
        t_lo = max(t_lo, min(ta, tb));  t_hi = min(t_hi, max(ta, tb))
    elif not (0 <= px <= n_cols-1):
        return None
    if abs(dy) > 1e-10:
        ta, tb = (0-py)/dy, (n_rows-1-py)/dy
        t_lo = max(t_lo, min(ta, tb));  t_hi = min(t_hi, max(ta, tb))
    elif not (0 <= py <= n_rows-1):
        return None
    if t_lo >= t_hi:
        return None
    return (px+t_lo*dx, py+t_lo*dy), (px+t_hi*dx, py+t_hi*dy)


def line_intersect(p1, d1, p2, d2):
    A = np.array([[d1[0],-d2[0]],[d1[1],-d2[1]]], dtype=float)
    b = np.array([p2[0]-p1[0], p2[1]-p1[1]], dtype=float)
    if abs(np.linalg.det(A)) < 1e-10:
        return None
    ts = np.linalg.solve(A, b)
    return p1 + ts[0]*d1


def draw_arc(ax, center, d1, d2, radius, color, lw=2.0):
    """Draw arc between two directions at center. Returns bisector angle (deg)."""
    a1 = float(np.degrees(np.arctan2(d1[1], d1[0])))
    a2 = float(np.degrees(np.arctan2(d2[1], d2[0])))
    diff = ((a2 - a1) + 180) % 360 - 180
    if diff < 0:
        a1, diff = a2, -diff
    ax.add_patch(mpatches.Arc(center, 2*radius, 2*radius,
                              theta1=a1, theta2=a1+diff,
                              color=color, lw=lw, zorder=9))
    return a1 + diff/2.0


# =============================================================================
# SEVERITY
# =============================================================================

def severity_label(angle):
    if angle < 10:   return "Normal (<10°)",   "#4CAF50"
    elif angle < 25: return "Mild (10–25°)",   "#FFC107"
    elif angle < 40: return "Moderate (25–40°)", "#FF9800"
    else:            return "Severe (>40°)",   "#F44336"


# =============================================================================
# COBB ANGLE DRAWING
# =============================================================================

def draw_endplate_and_arc(ax, img_shape, p, ec_px, er_px,
                          LINE_COLOR, LINE_LW):
    """Draw one full-width endplate line + diamond marker."""
    n_rows, n_cols = img_shape
    seg = clip_line_to_box(p[0], p[1], ec_px, er_px, n_rows, n_cols)
    if seg:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                "-", color=LINE_COLOR, lw=LINE_LW,
                solid_capstyle="round", zorder=7)
    ax.plot(p[0], p[1], "D", color=LINE_COLOR, markersize=8,
            markeredgecolor="white", markeredgewidth=0.8, zorder=8)


def annotate_cobb(ax, pa, pb, ea_px, er_a, eb_px, er_b,
                  angle, img_shape, ARC_COLOR):
    """
    Draw the angle arc + label.

    The intersection of the two endplate lines is always shown — even when
    it falls outside the image bounds.  ax.set_xlim / set_ylim are NOT
    changed here; instead the lines are clipped to a generous expanded box
    so they visually converge at the intersection whether it is inside or
    outside the original image area.  The arc is drawn at the intersection
    and the angle label sits along the bisector.
    """
    n_rows, n_cols = img_shape
    da_px     = np.array([ea_px, er_a])
    db_px     = np.array([eb_px, er_b])
    intersect = line_intersect(pa, da_px, pb, db_px)

    box = dict(boxstyle="round,pad=0.45", facecolor="#111111",
               edgecolor="#888888", alpha=0.95)

    if intersect is None:
        # perfectly parallel — just label the midpoint
        mid = (pa + pb) / 2.0
        ax.text(mid[0] + 12, mid[1],
                f"{angle:.1f}°\n({VERT_A}\u2013{VERT_B})",
                color=ARC_COLOR, fontsize=13, fontweight="bold",
                va="center", ha="left", zorder=11, bbox=box)
        return

    # always draw both lines all the way to the intersection
    # (clip_line_to_box already handles in-image portion;
    #  here we additionally draw from centroid to intersection
    #  so the converging geometry is unambiguous)
    LINE_COLOR = "#FF4444"
    for p, d in [(pa, da_px), (pb, db_px)]:
        # direction from centroid toward intersection
        toward = intersect - p
        dist   = np.linalg.norm(toward)
        if dist > 1:
            td = toward / dist
            ax.plot([p[0], intersect[0]], [p[1], intersect[1]],
                    "-", color=LINE_COLOR, lw=2.5,
                    solid_capstyle="round", zorder=7)

    # arc at intersection
    arc_r = max(28, min(65, float(np.linalg.norm(pb - pa)) * 0.14))
    da    = pa - intersect;  da /= np.linalg.norm(da)
    db    = pb - intersect;  db /= np.linalg.norm(db)
    bis   = draw_arc(ax, intersect, da, db, arc_r, ARC_COLOR, lw=2.5)
    ax.plot(*intersect, "o", color=ARC_COLOR, markersize=6, zorder=10)

    # label along bisector, offset outward from intersection
    lr = arc_r + 22
    lx = intersect[0] + lr * np.cos(np.radians(bis))
    ly = intersect[1] + lr * np.sin(np.radians(bis))
    ax.text(lx, ly, f"{angle:.1f}°", color=ARC_COLOR,
            fontsize=14, fontweight="bold",
            va="center", ha="center", zorder=11, bbox=box)


# =============================================================================
# DRAW COBB ON MAIN AXES
# =============================================================================

def draw_cobb(ax, centroids, col_fn, row_fn,
              cs_col_mm, cs_row_mm, t_knots,
              zoom_col, zoom_row, img_shape):
    """Draw Cobb angle on ax; return angle in degrees or None."""
    names = [c["name"] for c in centroids]
    for v in (VERT_A, VERT_B):
        if v not in names:
            print(f"  WARNING: {v} not found — skipping Cobb.")
            return None

    ia, ib = names.index(VERT_A), names.index(VERT_B)

    pa = np.array([col_fn(centroids[ia]["vox"]),
                   row_fn(centroids[ia]["vox"])], dtype=float)
    pb = np.array([col_fn(centroids[ib]["vox"]),
                   row_fn(centroids[ib]["vox"])], dtype=float)

    ec_a_px, er_a_px, ec_a_mm, er_a_mm = endplate_direction(
        cs_col_mm, cs_row_mm, t_knots[ia], zoom_col, zoom_row)
    ec_b_px, er_b_px, ec_b_mm, er_b_mm = endplate_direction(
        cs_col_mm, cs_row_mm, t_knots[ib], zoom_col, zoom_row)

    # Cobb angle in mm space
    dot   = float(np.clip(ec_a_mm*ec_b_mm + er_a_mm*er_b_mm, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(abs(dot))))

    LINE_COLOR = "#FF4444"
    LINE_LW    = 2.5
    ARC_COLOR  = "#FFD700"

    # endplate lines on main view
    draw_endplate_and_arc(ax, img_shape, pa, ec_a_px, er_a_px, LINE_COLOR, LINE_LW)
    draw_endplate_and_arc(ax, img_shape, pb, ec_b_px, er_b_px, LINE_COLOR, LINE_LW)

    # arc + label on main view
    annotate_cobb(ax, pa, pb, ec_a_px, er_a_px, ec_b_px, er_b_px,
                  angle, img_shape, ARC_COLOR)

    # severity badge
    sev_text, sev_col = severity_label(angle)
    ax.text(8, 8, sev_text, color=sev_col, fontsize=9, fontweight="bold",
            va="top", ha="left", zorder=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111",
                      edgecolor=sev_col, alpha=0.88))

    return angle


# =============================================================================
# RENDER
# =============================================================================

def _setup_ax(ax):
    ax.set_facecolor(FIG_BG)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER_COLOR)


def _draw_spine(ax, img, centroids, col_fn, row_fn, zoom_col, zoom_row,
                cs_col_mm, cs_row_mm, t_knots, img_shape, draw_angle=True):
    """
    Draw CT background, spline, centroid dots, and optionally Cobb annotation
    onto ax.  Returns the computed Cobb angle (or None).
    """
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="upper",
              aspect=zoom_row / zoom_col)
    _setup_ax(ax)

    cols_px = [col_fn(c["vox"]) for c in centroids]
    rows_px = [row_fn(c["vox"]) for c in centroids]

    cx, cy, _, _, _ = fit_spline_mm(cols_px, rows_px, zoom_col, zoom_row)
    if cx is not None:
        ax.plot(cx, cy, "-", color="#FFD700", lw=2.0, alpha=0.85, zorder=4)

    for c, px, py in zip(centroids, cols_px, rows_px):
        clr = region_color(c["name"])
        ax.plot(px, py, "o", color=clr, markersize=MARKER_SIZE,
                markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        ax.text(px + LABEL_OFFSET_PX, py, c["name"], color=clr,
                fontsize=LABEL_FONT_SIZE, va="center",
                fontweight="bold", zorder=6)

    angle = None
    if draw_angle and cs_col_mm is not None:
        angle = draw_cobb(ax, centroids, col_fn, row_fn,
                          cs_col_mm, cs_row_mm, t_knots,
                          zoom_col, zoom_row, img_shape)
    return angle


def save_png(img, centroids, col_fn, row_fn,
             zoom_col, zoom_row, title, suptitle, out_path):
    """
    Two-panel PNG:
      Left  — full spine overview with spline, centroids, endplate lines
      Right — close-up of the VERT_A–VERT_B region with the angle clearly visible
    """
    n_rows, n_cols = img.shape

    # ── spline (computed once, shared by both panels) ─────────────────────────
    cols_px = [col_fn(c["vox"]) for c in centroids]
    rows_px = [row_fn(c["vox"]) for c in centroids]
    _, _, cs_col_mm, cs_row_mm, t_knots = fit_spline_mm(
        cols_px, rows_px, zoom_col, zoom_row)

    # ── close-up crop around the two Cobb vertebrae ───────────────────────────
    names = [c["name"] for c in centroids]
    PAD   = INSET_PAD
    if VERT_A in names and VERT_B in names:
        ia, ib = names.index(VERT_A), names.index(VERT_B)
        pa_col = col_fn(centroids[ia]["vox"])
        pa_row = row_fn(centroids[ia]["vox"])
        pb_col = col_fn(centroids[ib]["vox"])
        pb_row = row_fn(centroids[ib]["vox"])
        cr_lo  = max(0,        int(min(pa_row, pb_row)) - PAD)
        cr_hi  = min(n_rows-1, int(max(pa_row, pb_row)) + PAD)
        cc_lo  = max(0,        int(min(pa_col, pb_col)) - PAD)
        cc_hi  = min(n_cols-1, int(max(pa_col, pb_col)) + PAD)
        crop   = img[cr_lo:cr_hi+1, cc_lo:cc_hi+1]
        # shifted col/row functions for the crop
        col_fn_c = lambda v: col_fn(v) - cc_lo
        row_fn_c = lambda v: row_fn(v) - cr_lo
        has_crop = True
    else:
        crop, col_fn_c, row_fn_c = img, col_fn, row_fn
        cr_lo = cc_lo = 0
        has_crop = False

    # ── figure layout: two panels side by side ────────────────────────────────
    # left panel height is fixed; right panel matches the crop aspect ratio
    phys_full_h = n_rows * zoom_row
    phys_full_w = n_cols * zoom_col
    left_w  = 5.5
    left_h  = min(18.0, left_w * phys_full_h / phys_full_w)

    if has_crop:
        cr_h, cr_w = crop.shape
        right_h = left_h
        right_w = right_h * (cr_w * zoom_col) / (cr_h * zoom_row)
        right_w = min(right_w, 6.0)
    else:
        right_w = left_w
        right_h = left_h

    fig_w = left_w + right_w + 0.4   # 0.4 gap
    fig_h = max(left_h, right_h) + 1.2   # room for suptitle + legend

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=FIG_BG)

    # manually place axes so heights are physically correct
    top    = 1.0 - 0.6 / fig_h     # suptitle space
    bot    = 0.9 / fig_h            # legend space
    usable = top - bot

    # left panel
    lx = 0.02
    lw = (left_w / fig_w) * 0.96
    lh = usable * (left_h / fig_h) / (usable)   # fill usable height
    lh = usable
    ax_full = fig.add_axes([lx, bot, lw, lh])

    # right panel — vertically centred
    rw    = (right_w / fig_w) * 0.96
    rh    = usable * (right_h / fig_h) / (usable)
    rh    = min(usable, usable * (crop.shape[0] * zoom_row) /
                (n_rows * zoom_row) * 1.0)
    rh    = max(0.15, min(usable, rh))
    ry    = bot + (usable - rh) / 2.0
    rx    = lx + lw + 0.4 / fig_w
    ax_crop = fig.add_axes([rx, ry, rw, rh])

    # ── left panel: full spine ────────────────────────────────────────────────
    angle = _draw_spine(ax_full, img, centroids, col_fn, row_fn,
                        zoom_col, zoom_row,
                        cs_col_mm, cs_row_mm, t_knots,
                        (n_rows, n_cols), draw_angle=True)
    ax_full.set_title("Full spine", color="white", fontsize=9, pad=4)

    # ── right panel: close-up ─────────────────────────────────────────────────
    # filter centroids to only those inside the crop
    crop_centroids = [c for c in centroids
                      if (cc_lo <= col_fn(c["vox"]) <= cc_hi
                          and cr_lo <= row_fn(c["vox"]) <= cr_hi)]

    _, _, cs_c, cs_r, t_c = fit_spline_mm(
        [col_fn_c(c["vox"]) for c in crop_centroids],
        [row_fn_c(c["vox"]) for c in crop_centroids],
        zoom_col, zoom_row)

    _draw_spine(ax_crop, crop, crop_centroids, col_fn_c, row_fn_c,
                zoom_col, zoom_row,
                cs_c, cs_r, t_c,
                crop.shape, draw_angle=True)
    ax_crop.set_title(f"Close-up: {VERT_A}–{VERT_B}",
                      color="#FFD700", fontsize=9, pad=4)

    # ── shared title + legend ─────────────────────────────────────────────────
    angle_str = (f"  |  Cobb ({VERT_A}–{VERT_B}) = {angle:.1f}°"
                 if angle is not None else "")
    fig.suptitle(suptitle + angle_str, color="white", fontsize=12,
                 y=1.0 - 0.25/fig_h)

    patches = [
        mpatches.Patch(color=REGION_COLORS["C"], label="Cervical"),
        mpatches.Patch(color=REGION_COLORS["T"], label="Thoracic"),
        mpatches.Patch(color=REGION_COLORS["L"], label="Lumbar"),
        mpatches.Patch(color=REGION_COLORS["S"], label="Sacral"),
        mpatches.Patch(color="#FF4444",          label="Endplate lines"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               framealpha=0.3, facecolor="#222222",
               labelcolor="white", fontsize=8,
               bbox_to_anchor=(0.5, 0.0))

    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor=FIG_BG)
    print(f"Saved  ->  {out_path}")
    plt.close(fig)
    return angle


# =============================================================================
# VIEW BUILDERS
# =============================================================================

def build_sagittal(ct_win, zooms, centroids,
                   sag_axis, horiz_axis, vert_axis, sup_ascending, out_path):
    img, col_fn, row_fn = curved_mpr(
        ct_win, centroids,
        slice_axis=sag_axis, col_axis=horiz_axis, row_axis=vert_axis,
        sup_ascending=sup_ascending, slab_half=SAG_SLAB_HALF)
    lr = [c["vox"][sag_axis] for c in centroids]
    return save_png(
        img, centroids, col_fn, row_fn,
        zoom_col=zooms[horiz_axis], zoom_row=zooms[vert_axis],
        title=(f"Curved MPR ±{SAG_SLAB_HALF} sl  |  "
               f"axis {sag_axis}(L-R) [{min(lr):.0f}–{max(lr):.0f}]  |  "
               f"{len(centroids)} vertebrae"),
        suptitle=f"{SUBJECT_ID}  —  Sagittal",
        out_path=out_path)


def build_coronal(ct_win, zooms, centroids,
                  sag_axis, horiz_axis, vert_axis, sup_ascending, out_path):
    img, col_fn, row_fn = curved_mpr(
        ct_win, centroids,
        slice_axis=horiz_axis, col_axis=sag_axis, row_axis=vert_axis,
        sup_ascending=sup_ascending, slab_half=COR_SLAB_HALF)
    ap = [c["vox"][horiz_axis] for c in centroids]
    return save_png(
        img, centroids, col_fn, row_fn,
        zoom_col=zooms[sag_axis], zoom_row=zooms[vert_axis],
        title=(f"Curved MPR ±{COR_SLAB_HALF} sl  |  "
               f"axis {horiz_axis}(A-P) [{min(ap):.0f}–{max(ap):.0f}]  |  "
               f"{len(centroids)} vertebrae"),
        suptitle=f"{SUBJECT_ID}  —  Coronal",
        out_path=out_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n=== cobb_curve.py — {SUBJECT_ID}  ({VERT_A}→{VERT_B}) ===")

    for label, fp in [("CT", CT_PATH), ("Centroids", CTD_PATH)]:
        if not Path(fp).exists():
            sys.exit(f"ERROR: {label} not found:\n  {fp}")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"\nLoading CT:\n  {CT_PATH}")
    ct_data, ct_img = load_ct(CT_PATH)
    zooms = np.array(ct_img.header.get_zooms()[:3])
    print(f"  Shape      : {ct_data.shape}")
    print(f"  Voxel size : {zooms} mm")

    sag_axis, horiz_axis, vert_axis, sup_ascending = get_axis_roles(ct_img.affine)
    print(f"\nAxis roles  — sag={sag_axis}(L-R)  horiz={horiz_axis}(A-P)  "
          f"vert={vert_axis}(S-I)  sup_asc={sup_ascending}")

    print(f"\nLoading centroids:\n  {CTD_PATH}")
    centroids = load_centroids(CTD_PATH)
    centroids.sort(key=lambda c: -c["vox"][vert_axis] if sup_ascending
                                 else  c["vox"][vert_axis])
    print(f"  {len(centroids)} vertebrae: " +
          "  ".join(c["name"] for c in centroids))

    ct_win = window_ct(ct_data)

    print(f"\nRendering sagittal...")
    sag_angle = build_sagittal(ct_win, zooms, centroids,
                               sag_axis, horiz_axis, vert_axis, sup_ascending,
                               Path(OUT_DIR) / f"{SUBJECT_ID}_sagittal.png")

    print(f"\nRendering coronal...")
    cor_angle = build_coronal(ct_win, zooms, centroids,
                              sag_axis, horiz_axis, vert_axis, sup_ascending,
                              Path(OUT_DIR) / f"{SUBJECT_ID}_coronal.png")

    sev_sag = severity_label(sag_angle)[0] if sag_angle else "N/A"
    sev_cor = severity_label(cor_angle)[0] if cor_angle else "N/A"
    print(f"\n{'='*52}")
    print(f"  Subject  : {SUBJECT_ID}")
    print(f"  Vertebrae: {VERT_A} → {VERT_B}")
    if sag_angle is not None:
        print(f"  Sagittal Cobb : {sag_angle:.2f}°  [{sev_sag}]")
    if cor_angle is not None:
        print(f"  Coronal  Cobb : {cor_angle:.2f}°  [{sev_cor}]")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()