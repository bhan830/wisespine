#!/usr/bin/env python3
"""
visualize_cobb_2d.py
====================
Builds a geometrically-correct coronal DICOM Secondary Capture
with bounding-box and endplate-line overlays for Cobb angle review.

Key design decisions
--------------------
1. Resample CT volume to isotropic voxels (scipy.ndimage.zoom) using the
   actual SliceThickness and PixelSpacing from the DICOM headers before
   extracting any coronal slice.  Without this the coronal image is
   12 px tall vs 704 px wide — completely squashed.

2. Align the SEG (93 frames) to the CT (12 slices) by matching physical
   Z positions (ImagePositionPatient[2]), then resample the label masks
   with the same zoom factors.

3. Draw overlays (bounding boxes, endplate lines, Cobb arc) directly on
   a uint8 numpy array using PIL — no matplotlib re-encoding artefacts.

4. Save as a DICOM Secondary Capture with:
     - PixelSpacing  = [target_spacing_mm, target_spacing_mm]
     - SamplesPerPixel = 3 (RGB, burned-in colour overlays)
   ITK-SNAP and 3D Slicer open this correctly.

Authors : Benjamin Han, Mary (guidance: Vikash)
"""

import numpy as np
import pydicom
import pydicom.uid
from pydicom.dataset import Dataset, FileDataset
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.ndimage import zoom as nd_zoom
from PIL import Image, ImageDraw, ImageFont
import datetime

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
DICOM_BASE_DIR   = Path("/gscratch/scrubbed/bhan830/wisespine/data/DICOM-Dataset")
IMAGE_SERIES_DIR = DICOM_BASE_DIR / "DICOM-RT"
SEG_DIR          = DICOM_BASE_DIR / "DICOM-SEG"
OUTPUT_DIR       = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new"
                        "/baseline_outputs/cobb_angles_dicom")
DEBUG_DIR        = OUTPUT_DIR / "debug"

TOP_VERTEBRA    = "T10"
BOTTOM_VERTEBRA = "L5"

# Overlay colours (R,G,B uint8)
COLORS = {
    TOP_VERTEBRA:    (230, 57,  70),    # red
    BOTTOM_VERTEBRA: (69, 123, 157),    # blue
}
SUP_COLOR  = (244, 162,  97)            # orange — superior endplate
INF_COLOR  = (42,  157, 143)            # teal   — inferior endplate
ARC_COLOR  = (255, 230,   0)            # yellow — Cobb arc
TEXT_COLOR = (255, 255, 255)            # white  — labels
MASK_ALPHA = 100                        # 0–255


# ══════════════════════════════════════════════════════════════
# STEP 1 — Load CT slices, sort by Z, return volume + metadata
# ══════════════════════════════════════════════════════════════
def load_ct_series():
    files = [f for f in sorted(IMAGE_SERIES_DIR.glob("*.dcm"))]
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            mod = getattr(ds, "Modality", "")
            if mod in ("RTSTRUCT", "SEG", "SR"):
                continue
            z = float(getattr(ds, "ImagePositionPatient", [0,0,0])[2])
            slices.append((z, f))
        except Exception:
            continue

    slices.sort(key=lambda x: x[0], reverse=True)   # superior first
    print(f"  {len(slices)} CT slices sorted by Z position")

    arrays, z_positions, ref_ds = [], [], None
    for z, f in slices:
        ds  = pydicom.dcmread(f)
        arr = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope",     1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arrays.append(arr * slope + intercept)
        z_positions.append(z)
        if ref_ds is None:
            ref_ds = ds

    ct_vol = np.stack(arrays, axis=0)                # (n, rows, cols)

    ps  = getattr(ref_ds, "PixelSpacing", [1.0, 1.0])
    row_sp  = float(ps[0])
    col_sp  = float(ps[1])
    thick   = float(getattr(ref_ds, "SliceThickness", 5.0))

    print(f"  CT volume : {ct_vol.shape}")
    print(f"  Spacing   : row={row_sp:.3f}mm  col={col_sp:.3f}mm  thick={thick:.1f}mm")

    return ct_vol, np.array(z_positions), row_sp, col_sp, thick, ref_ds


# ══════════════════════════════════════════════════════════════
# STEP 2 — Resample volume to isotropic voxels
# ══════════════════════════════════════════════════════════════
def resample_isotropic(ct_vol, z_positions, row_sp, col_sp, thick):
    """
    Zoom the CT volume so all three axes have the same voxel size
    (target = col_sp, the finest in-plane spacing).

    Returns
    -------
    ct_iso        : resampled (nz, ny, nx) float32 volume, values [0..1]
    zoom_factors  : (zf, rf, cf) used, needed to rescale masks
    target_sp     : isotropic voxel size in mm
    new_z_pos     : interpolated Z positions for the resampled slices
    """
    target_sp = col_sp                        # finest in-plane spacing
    zf = thick  / target_sp                   # e.g. 5.0/0.488 ≈ 10.25
    rf = row_sp / target_sp                   # usually ≈ 1.0
    cf = 1.0

    print(f"  Resampling: zoom_z={zf:.2f}  zoom_row={rf:.2f}  zoom_col={cf:.2f}")

    # Window to [0,1] first
    lo, hi = np.percentile(ct_vol, 2), np.percentile(ct_vol, 98)
    ct_w   = np.clip((ct_vol - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    ct_iso  = nd_zoom(ct_w, (zf, rf, cf), order=1)
    print(f"  Resampled : {ct_iso.shape}")

    # Interpolate Z positions for the new slices
    old_idx = np.arange(len(z_positions))
    new_idx = np.linspace(0, len(z_positions)-1, ct_iso.shape[0])
    new_z   = np.interp(new_idx, old_idx, z_positions)

    return ct_iso, (zf, rf, cf), target_sp, new_z


# ══════════════════════════════════════════════════════════════
# STEP 3 — Build aligned label volumes and resample
# ══════════════════════════════════════════════════════════════
def label_matches(label, target):
    return target.upper() in label.upper()


def find_seg_files():
    out = []
    for f in SEG_DIR.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if getattr(ds, "Modality", "") == "SEG":
                out.append(f)
        except Exception:
            continue
    return out


def build_and_resample_volumes(seg_path, ct_z_orig, ref_rows, ref_cols,
                                zoom_factors, new_shape, targets):
    """
    1. Map every SEG frame to the nearest original CT slice by Z.
    2. Build (n_orig_ct, ref_rows, ref_cols) bool volumes.
    3. Resample with the same zoom_factors as the CT.
    Returns {label: (nz_iso, ny_iso, nx_iso) bool ndarray}.
    """
    seg_ds = pydicom.dcmread(seg_path)
    px     = seg_ds.pixel_array                   # (n_seg, seg_rows, seg_cols)
    n_seg, seg_rows, seg_cols = px.shape
    n_orig_ct = len(ct_z_orig)

    # Scale SEG pixel coords → CT pixel coords
    row_sc = ref_rows / seg_rows
    col_sc = ref_cols / seg_cols

    # Segment number → label
    sid_map = {}
    for seg in seg_ds.SegmentSequence:
        for t in targets:
            if label_matches(seg.SegmentLabel, t):
                sid_map[seg.SegmentNumber] = t

    raw_vols = {t: np.zeros((n_orig_ct, ref_rows, ref_cols), dtype=np.uint8)
                for t in targets}

    for i, fmeta in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
        ref = fmeta.SegmentIdentificationSequence[0].ReferencedSegmentNumber
        label = sid_map.get(ref)
        if label is None:
            continue
        try:
            ipp_z = float(fmeta.PlanePositionSequence[0].ImagePositionPatient[2])
        except Exception:
            continue

        ct_idx = int(np.argmin(np.abs(ct_z_orig - ipp_z)))
        mask   = px[i] > 0
        if not np.any(mask):
            continue
        sr, sc = np.where(mask)
        cr = np.clip(np.round(sr * row_sc).astype(int), 0, ref_rows-1)
        cc = np.clip(np.round(sc * col_sc).astype(int), 0, ref_cols-1)
        raw_vols[label][ct_idx, cr, cc] = 1

    # Resample masks with same zoom as CT (nearest-neighbour)
    zf, rf, cf = zoom_factors
    iso_vols = {}
    for t, vol in raw_vols.items():
        iso_vols[t] = nd_zoom(vol.astype(np.float32),
                               (zf, rf, cf), order=0) > 0.5
        print(f"  {t}: {iso_vols[t].sum()} voxels after resampling")

    return iso_vols


# ══════════════════════════════════════════════════════════════
# STEP 4 — Best coronal row in isotropic space
# ══════════════════════════════════════════════════════════════
def best_coronal_row(volumes):
    combined = sum(v.astype(np.uint16) for v in volumes.values())
    sums     = combined.sum(axis=(0, 2))    # sum over Z and X
    return int(np.argmax(sums))


# ══════════════════════════════════════════════════════════════
# STEP 5 — Endplate line fitting in coronal (X, Z) space
# ══════════════════════════════════════════════════════════════
def volume_to_points(vol):
    pts = np.argwhere(vol).astype(float)
    if len(pts) > 8000:
        pts = pts[np.random.choice(len(pts), 8000, replace=False)]
    return pts


def extract_endplates(pts, pct=10.0):
    """pts[:,0]=Z(slice).  Low Z = superior."""
    z   = pts[:, 0]
    sup = pts[z <= np.percentile(z, pct)]
    inf = pts[z >= np.percentile(z, 100-pct)]
    return sup, inf


def fit_line(ep_pts, best_row, n_rows, margin=0.15):
    """
    Fit a line through endplate points near `best_row`.
    Coronal image: x = col (axis 2), y = Z-slice (axis 0).
    Returns (x0, y0, x1, y1).
    """
    tol  = max(n_rows * 0.12, 3)
    near = ep_pts[np.abs(ep_pts[:, 1] - best_row) <= tol]
    if len(near) < 3:
        near = ep_pts

    xp = near[:, 2]    # col  → image x
    yp = near[:, 0]    # Z    → image y

    pts2 = np.column_stack([xp, yp])
    c2   = pts2.mean(axis=0)
    d2   = PCA(n_components=1).fit(pts2).components_[0]

    dx  = max(xp.max() - xp.min(), 1.0)
    pad = dx * margin
    if abs(d2[0]) > 1e-8:
        t0 = (xp.min() - pad - c2[0]) / d2[0]
        t1 = (xp.max() + pad - c2[0]) / d2[0]
        if t0 > t1: t0, t1 = t1, t0
    else:
        t0, t1 = -dx * 0.6, dx * 0.6

    p0 = c2 + t0 * d2
    p1 = c2 + t1 * d2
    return float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])


def cobb_from_lines(x0a,y0a,x1a,y1a, x0b,y0b,x1b,y1b):
    d1 = np.array([x1a-x0a, y1a-y0a], dtype=float)
    d2 = np.array([x1b-x0b, y1b-y0b], dtype=float)
    n1 = np.linalg.norm(d1); n2 = np.linalg.norm(d2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    d1 /= n1; d2 /= n2
    a = float(np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1.0, 1.0))))
    return 180.0 - a if a > 90 else a


# ══════════════════════════════════════════════════════════════
# STEP 6 — PIL drawing helpers
# ══════════════════════════════════════════════════════════════
def gray_to_rgb(gray_2d):
    """(H,W) float [0,1] → (H,W,3) uint8."""
    g = (np.clip(gray_2d, 0.0, 1.0) * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def overlay_mask(rgb, mask_2d, color_rgb, alpha=100):
    """Blend coloured mask onto RGB array (in-place)."""
    a = alpha / 255.0
    for c, v in enumerate(color_rgb):
        ch = rgb[:, :, c].astype(np.float32)
        ch[mask_2d] = ch[mask_2d] * (1-a) + v * a
        rgb[:, :, c] = ch.clip(0, 255).astype(np.uint8)


def draw_bbox(draw, mask_2d, color, lw=3):
    """Draw dashed bounding rectangle around mask_2d."""
    rows, cols = np.where(mask_2d)
    if len(rows) == 0:
        return None
    r0,r1 = rows.min(), rows.max()
    c0,c1 = cols.min(), cols.max()
    # PIL ImageDraw rectangle
    draw.rectangle([c0, r0, c1, r1], outline=color, width=lw)
    return (c0, r0, c1, r1)


def draw_line_pil(draw, x0, y0, x1, y1, color, width=3):
    draw.line([(int(x0), int(y0)), (int(x1), int(y1))],
              fill=color, width=width)


def draw_arc_pil(draw, sup_line, inf_line, cobb_deg):
    """Draw arc at intersection of two endplate lines."""
    x0s,y0s,x1s,y1s = sup_line
    x0i,y0i,x1i,y1i = inf_line

    ds_ = np.array([x1s-x0s, y1s-y0s], dtype=float)
    di_ = np.array([x1i-x0i, y1i-y0i], dtype=float)
    ns = np.linalg.norm(ds_); ni = np.linalg.norm(di_)
    if ns < 1e-8 or ni < 1e-8:
        return
    ds_ /= ns; di_ /= ni

    A = np.array([(x0s+x1s)/2, (y0s+y1s)/2])
    B = np.array([(x0i+x1i)/2, (y0i+y1i)/2])
    denom = ds_[0]*di_[1] - ds_[1]*di_[0]
    if abs(denom) < 1e-6:
        return
    t  = ((B-A)[0]*di_[1] - (B-A)[1]*di_[0]) / denom
    ix = A + t * ds_

    r  = int(max(abs(x1s-x0s), abs(x1i-x0i)) * 0.20)
    if r < 5:
        return

    a1_deg = float(np.degrees(np.arctan2(ds_[1], ds_[0])))
    a2_deg = float(np.degrees(np.arctan2(di_[1], di_[0])))
    bbox   = [ix[0]-r, ix[1]-r, ix[0]+r, ix[1]+r]
    draw.arc(bbox, start=min(a1_deg,a2_deg),
             end=max(a1_deg,a2_deg), fill=ARC_COLOR, width=3)

    mid_rad = np.radians((a1_deg + a2_deg) / 2)
    lx = int(ix[0] + r * 1.8 * np.cos(mid_rad))
    ly = int(ix[1] + r * 1.8 * np.sin(mid_rad))
    draw.text((lx, ly), f"{cobb_deg:.1f}\u00b0",
              fill=ARC_COLOR)


def label_at(draw, x, y, text, color):
    draw.text((int(x)+2, int(y)-14), text, fill=color)


# ══════════════════════════════════════════════════════════════
# STEP 7 — Render coronal image with overlays
# ══════════════════════════════════════════════════════════════
def render_coronal(ct_iso, iso_vols, best_row,
                   sup_line, inf_line, cobb_deg,
                   row_offset=0):
    """
    Extract coronal slice at (best_row + row_offset) and draw overlays.
    Returns (H, W, 3) uint8 numpy array.
    """
    row_idx = best_row + row_offset
    row_idx = max(0, min(ct_iso.shape[1]-1, row_idx))

    gray   = ct_iso[:, row_idx, :]           # (nZ, nX)
    rgb    = gray_to_rgb(gray)

    for v, vol in iso_vols.items():
        mask = vol[:, row_idx, :]
        overlay_mask(rgb, mask, COLORS[v], alpha=MASK_ALPHA)

    img  = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)

    for v, vol in iso_vols.items():
        mask = vol[:, row_idx, :]
        bbox = draw_bbox(draw, mask, COLORS[v], lw=3)
        if bbox:
            c0, r0, c1, r1 = bbox
            label_at(draw, c0, r0, v, COLORS[v])

    # Endplate lines
    if sup_line is not None:
        draw_line_pil(draw, *sup_line, SUP_COLOR, width=3)
    if inf_line is not None:
        draw_line_pil(draw, *inf_line, INF_COLOR, width=3)
    if sup_line and inf_line:
        draw_arc_pil(draw, sup_line, inf_line, cobb_deg)

    return np.array(img, dtype=np.uint8)


# ══════════════════════════════════════════════════════════════
# STEP 8 — Save as DICOM Secondary Capture
# ══════════════════════════════════════════════════════════════
def save_dicom_sc(rgb_array, ref_ds, out_path, spacing_mm, description=""):
    """
    rgb_array : (H, W, 3) uint8
    spacing_mm: isotropic voxel size in mm (both row and col identical)
    """
    H, W = rgb_array.shape[:2]

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID  = "1.2.840.10008.5.1.4.1.1.7"
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID        = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID   = pydicom.uid.generate_uid()

    ds = FileDataset(str(out_path), {}, file_meta=file_meta,
                     preamble=b"\x00" * 128)
    ds.is_implicit_VR   = False
    ds.is_little_endian = True

    # Copy patient/study identifiers
    for tag in ("PatientID", "PatientName", "StudyInstanceUID",
                "StudyDate", "StudyTime", "AccessionNumber"):
        if hasattr(ref_ds, tag):
            setattr(ds, tag, getattr(ref_ds, tag))

    ds.SOPClassUID       = "1.2.840.10008.5.1.4.1.1.7"
    ds.SOPInstanceUID    = file_meta.MediaStorageSOPInstanceUID
    ds.Modality          = "SC"
    ds.ConversionType    = "WSD"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.InstanceNumber    = "1"
    ds.ContentDate       = datetime.date.today().strftime("%Y%m%d")
    ds.ContentTime       = datetime.datetime.now().strftime("%H%M%S")
    ds.SeriesDescription = description
    ds.ImageComments     = description

    # Pixel geometry
    ds.Rows                        = H
    ds.Columns                     = W
    ds.SamplesPerPixel             = 3
    ds.PhotometricInterpretation   = "RGB"
    ds.PlanarConfiguration         = 0      # interleaved RGBRGB...
    ds.BitsAllocated               = 8
    ds.BitsStored                  = 8
    ds.HighBit                     = 7
    ds.PixelRepresentation         = 0
    ds.PixelSpacing                = [f"{spacing_mm:.4f}", f"{spacing_mm:.4f}"]

    ds.PixelData = rgb_array.tobytes()
    ds.save_as(str(out_path), write_like_original=False)
    print(f"  [DCM]  {out_path.name}  ({H}x{W} px, spacing={spacing_mm:.3f}mm)")


# ══════════════════════════════════════════════════════════════
# TOP-LEVEL
# ══════════════════════════════════════════════════════════════
def visualize(seg_path, ct_iso, iso_vols, best_row,
              sup_line, inf_line, cobb_deg,
              target_sp, ref_ds):
    case_id = seg_path.stem
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # Main annotated coronal slice
    rgb_main = render_coronal(ct_iso, iso_vols, best_row,
                               sup_line, inf_line, cobb_deg, row_offset=0)

    save_dicom_sc(rgb_main, ref_ds,
                  DEBUG_DIR / f"{case_id}_cobb_coronal.dcm",
                  target_sp,
                  f"Cobb {cobb_deg:.1f}deg {TOP_VERTEBRA}-{BOTTOM_VERTEBRA}")

    # Survey: 3 adjacent rows side-by-side
    step  = max(1, ct_iso.shape[1] // 8)
    panels = []
    for off in [-step, 0, step]:
        panels.append(render_coronal(
            ct_iso, iso_vols, best_row,
            sup_line if off == 0 else None,
            inf_line if off == 0 else None,
            cobb_deg, row_offset=off))
    survey = np.concatenate(panels, axis=1)

    save_dicom_sc(survey, ref_ds,
                  DEBUG_DIR / f"{case_id}_coronal_survey.dcm",
                  target_sp,
                  f"Coronal survey {TOP_VERTEBRA}-{BOTTOM_VERTEBRA}")


def main():
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load and resample CT ──────────────────────────────────
    print("[INFO] Loading CT series ...")
    ct_vol, ct_z_orig, row_sp, col_sp, thick, ref_ds = load_ct_series()

    print("[INFO] Resampling to isotropic voxels ...")
    ct_iso, zoom_factors, target_sp, new_z = resample_isotropic(
        ct_vol, ct_z_orig, row_sp, col_sp, thick)

    ref_rows, ref_cols = ct_vol.shape[1], ct_vol.shape[2]
    targets = [TOP_VERTEBRA, BOTTOM_VERTEBRA]

    seg_files = find_seg_files()
    print(f"[INFO] Found {len(seg_files)} SEG file(s)\n")

    for seg_path in seg_files:
        case_id = seg_path.stem
        print(f"\n{'─'*60}\nCase: {case_id}\n{'─'*60}")
        try:
            # ── Align + resample SEG masks ────────────────────
            iso_vols = build_and_resample_volumes(
                seg_path, ct_z_orig, ref_rows, ref_cols,
                zoom_factors, ct_iso.shape, targets)

            empty = [t for t, v in iso_vols.items() if v.sum() == 0]
            if empty:
                print(f"  [ERROR] Empty after alignment: {empty}")
                print("  Check that SEG ImagePositionPatient Z values overlap CT")
                continue

            # ── Best coronal row + endplate lines ─────────────
            best_row = best_coronal_row(iso_vols)
            print(f"  Best coronal row: {best_row} / {ct_iso.shape[1]}")

            endplates = {}
            for v, vol in iso_vols.items():
                pts = volume_to_points(vol)
                sup, inf = extract_endplates(pts)
                endplates[v] = (sup, inf)

            sup_line = fit_line(
                endplates[TOP_VERTEBRA][0], best_row, ct_iso.shape[1])
            inf_line = fit_line(
                endplates[BOTTOM_VERTEBRA][1], best_row, ct_iso.shape[1])
            cobb = cobb_from_lines(*sup_line, *inf_line)
            print(f"  2D Cobb angle: {cobb:.2f}°")

            # ── Render + save ─────────────────────────────────
            visualize(seg_path, ct_iso, iso_vols, best_row,
                      sup_line, inf_line, cobb, target_sp, ref_ds)

        except Exception as e:
            import traceback
            print(f"[ERROR] {seg_path.stem}: {e}")
            traceback.print_exc()

    print(f"\nDone. Outputs in: {DEBUG_DIR}")


if __name__ == "__main__":
    main()