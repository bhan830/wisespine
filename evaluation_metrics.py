#!/usr/bin/env python3
"""
Evaluation (heuristic-only, C/T/L):

- Canonicalize pred & GT
- Resample pred -> GT (NN)
- Crop pred to GT axial extent
- Global (binary) Dice/IoU
- Per-vertebra Dice/IoU via optimal matching (Hungarian if available, else greedy)
- Append to:
    wisespine_new/baseline_outputs/evaluation_metrics.csv
- Write per-case detail CSV:
    wisespine_new/baseline_outputs/evaluation_details_<CASE>.csv

Run:
  python wisespine_new/evaluation_metrics.py
"""

from pathlib import Path
from datetime import datetime
import csv
import re
import numpy as np
import nibabel as nib

try:
    from nibabel.processing import resample_from_to
    NIB_RESAMPLE_OK = True
except Exception:
    NIB_RESAMPLE_OK = False

try:
    import scipy.ndimage as ndi
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# Progress bars
try:
    from tqdm import tqdm
    def pbar(it, desc): return tqdm(it, desc=desc, leave=False)
except Exception:
    def pbar(it, desc): return it

# ----------------------------
# Config
# ----------------------------
CASE = "sub-gl017"

BASE_DIR = Path(__file__).resolve().parent.parent

RECON_DIR = BASE_DIR / "wisespine_new" / "baseline_outputs" / "reconstructed" / CASE
PRED_DIR  = BASE_DIR / "wisespine_new" / "baseline_outputs" / CASE

HEUR_UNION = RECON_DIR / f"{CASE}_predicted_spine_heuristic.nii.gz"
INDIV_RE   = re.compile(rf"^{re.escape(CASE)}_pred-vert_([CTL]\d+)\.nii(\.gz)?$", re.IGNORECASE)

GT_PATH = (
    BASE_DIR / "data" / "Verse20" / "dataset-02validation" / "derivatives" / CASE /
    f"{CASE}_seg-vert_msk.nii.gz"
)

MAIN_CSV   = BASE_DIR / "wisespine_new" / "baseline_outputs" / "evaluation_metrics.csv"
DETAIL_CSV = BASE_DIR / "wisespine_new" / "baseline_outputs" / f"evaluation_details_{CASE}.csv"

CROP_TO_GT_AXIAL_EXTENT = True
CROP_MARGIN_SLICES = 2
EPS = 1e-8

# ----------------------------
# Helpers
# ----------------------------
def _to_can(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(img)

def _resample_like(src_img: nib.Nifti1Image, ref_img: nib.Nifti1Image, order: int = 0):
    if NIB_RESAMPLE_OK:
        return resample_from_to(src_img, ref_img, order=order)
    if src_img.shape != ref_img.shape or not np.allclose(src_img.affine, ref_img.affine):
        raise RuntimeError("Resampling unavailable and spaces differ.")
    return src_img

def crop_to_gt_axial(pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    if not CROP_TO_GT_AXIAL_EXTENT:
        return pred_bin
    nz = np.where(gt_bin)
    if len(nz[0]) == 0:
        return pred_bin
    zmin, zmax = int(np.min(nz[2])), int(np.max(nz[2]))
    zmin = max(0, zmin - CROP_MARGIN_SLICES)
    zmax = min(pred_bin.shape[2] - 1, zmax + CROP_MARGIN_SLICES)
    mask = np.zeros_like(pred_bin, dtype=bool)
    mask[:, :, zmin:zmax+1] = True
    return pred_bin & mask

def dice_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.sum(pred & gt); denom = np.sum(pred) + np.sum(gt)
    return float(2.0 * inter / (denom + EPS)) if denom > 0 else 1.0

def iou_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.sum(pred & gt); union = np.sum(pred | gt)
    return float(inter / (union + EPS)) if union > 0 else 1.0

# ----------------------------
# Loaders
# ----------------------------
def load_gt():
    gt_img = _to_can(nib.load(GT_PATH))
    gt_lab = gt_img.get_fdata().astype(np.int32)
    gt_bin = gt_lab > 0
    gt_labels = np.unique(gt_lab); gt_labels = gt_labels[gt_labels > 0]
    return gt_bin, gt_lab, gt_img, gt_labels

def load_union_to_gt(union_path: Path, gt_img: nib.Nifti1Image, gt_bin: np.ndarray) -> np.ndarray:
    if not union_path.exists():
        return np.zeros_like(gt_bin, dtype=bool)
    pred_img = _to_can(nib.load(union_path))
    pred_img = _resample_like(pred_img, gt_img, order=0)
    pred_bin = pred_img.get_fdata() > 0
    pred_bin = crop_to_gt_axial(pred_bin, gt_bin)
    return pred_bin.astype(bool)

def collect_per_vertebra() -> list[Path]:
    if not PRED_DIR.exists(): return []
    return sorted([p for p in PRED_DIR.glob("*.nii*") if INDIV_RE.match(p.name)])

def build_pred_components_to_gt(gt_img: nib.Nifti1Image, gt_bin: np.ndarray):
    indiv = collect_per_vertebra()
    comps, names = [], []
    for fp in pbar(indiv, "Load per-vertebra masks"):
        m = INDIV_RE.match(fp.name)
        if not m: continue
        name = m.group(1).upper()  # C#, T#, L#
        img = _to_can(nib.load(fp))
        try: img = _resample_like(img, gt_img, order=0)
        except RuntimeError: continue
        comp = img.get_fdata() > 0
        comp = crop_to_gt_axial(comp, gt_bin)
        if np.any(comp):
            comps.append(comp.astype(bool)); names.append(name)
    return comps, names

# ----------------------------
# Matching & metrics
# ----------------------------
def _overlap_matrix(pred_components, gt_lab, gt_labels):
    n_pred, n_gt = len(pred_components), len(gt_labels)
    M = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i in pbar(range(n_pred), "Overlap matrix"):
        pc = pred_components[i]
        if not np.any(pc): continue
        pc_sum = pc.sum()
        for j, g in enumerate(gt_labels):
            gmask = (gt_lab == int(g))
            inter = np.sum(pc & gmask)
            union = pc_sum + gmask.sum() - inter
            M[i, j] = (inter / (union + EPS)) if union > 0 else 0.0
    return M

def match_and_score(pred_components, pred_names, gt_lab, gt_labels):
    if len(pred_components) == 0 or len(gt_labels) == 0: return []
    M = _overlap_matrix(pred_components, gt_lab, gt_labels)
    # Hungarian if available, else greedy
    if SCIPY_OK and M.size > 0:
        from scipy.optimize import linear_sum_assignment
        cost = 1.0 - M
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        used, row_ind, col_ind = set(), [], []
        for i in pbar(range(M.shape[0]), "Greedy match"):
            j = int(np.argmax(M[i,:])) if M.shape[1] > 0 else -1
            if j >= 0 and M[i,j] > 0 and j not in used:
                row_ind.append(i); col_ind.append(j); used.add(j)
    results = []
    for i, j in pbar(list(zip(row_ind, col_ind)), "Score pairs"):
        if i < 0 or j < 0: continue
        pc = pred_components[i]
        gmask = (gt_lab == int(gt_labels[j]))
        inter = int(np.sum(pc & gmask))
        pv = int(np.sum(pc)); gv = int(np.sum(gmask))
        dice = (2.0 * inter) / (pv + gv + EPS) if (pv + gv) > 0 else 1.0
        iou  = float(M[i, j])
        results.append({
            "pred_idx": i,
            "pred_name": pred_names[i] if i < len(pred_names) else f"pred_{i}",
            "gt_label": int(gt_labels[j]),
            "dice": float(dice),
            "iou": iou,
            "pred_voxels": pv,
            "gt_voxels": gv,
            "intersection": inter
        })
    return results

# ----------------------------
# CSV writers
# ----------------------------
def append_main_csv(case, global_dice, global_iou, mean_vert_dice, mean_vert_iou):
    MAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp","case","global_dice","global_iou","mean_vert_dice","mean_vert_iou"]
    row = [
        datetime.now().isoformat(timespec="seconds"),
        case,
        f"{global_dice:.4f}",
        f"{global_iou:.4f}",
        "" if mean_vert_dice is None else f"{mean_vert_dice:.4f}",
        "" if mean_vert_iou  is None else f"{mean_vert_iou:.4f}",
    ]
    write_header = not MAIN_CSV.exists()
    with open(MAIN_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow(row)

def write_detail_csv(case, detail_rows):
    if not detail_rows: return
    header = ["case","pred_idx","pred_name","gt_label","dice","iou","pred_voxels","gt_voxels","intersection"]
    with open(DETAIL_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in detail_rows:
            w.writerow([
                case, r["pred_idx"], r["pred_name"], r["gt_label"],
                f"{r['dice']:.4f}", f"{r['iou']:.4f}",
                r["pred_voxels"], r["gt_voxels"], r["intersection"]
            ])

# ----------------------------
# Main
# ----------------------------
def main():
    print(f"🦴 Evaluating {CASE}")

    # GT
    gt_bin, gt_lab, gt_img, gt_labels = load_gt()

    # Global
    pred_bin = load_union_to_gt(HEUR_UNION, gt_img, gt_bin)
    global_dice = dice_binary(pred_bin, gt_bin)
    global_iou  = iou_binary(pred_bin, gt_bin)
    print(f"✅ Global Dice: {global_dice:.4f}")
    print(f"✅ Global IoU:  {global_iou:.4f}")

    # Per-vertebra (C/T/L)
    pred_components, pred_names = build_pred_components_to_gt(gt_img, gt_bin)
    detail_rows = match_and_score(pred_components, pred_names, gt_lab, gt_labels)
    if detail_rows:
        mean_vert_dice = float(np.mean([r["dice"] for r in detail_rows]))
        mean_vert_iou  = float(np.mean([r["iou"]  for r in detail_rows]))
        print(f"🔎 Per-vertebra matched pairs: {len(detail_rows)}")
        print(f"   Mean per-vertebra Dice: {mean_vert_dice:.4f}")
        print(f"   Mean per-vertebra IoU:  {mean_vert_iou:.4f}")
    else:
        mean_vert_dice = None; mean_vert_iou = None
        print("⚠️ No per-vertebra matches computed.")

    # CSV
    append_main_csv(CASE, global_dice, global_iou, mean_vert_dice, mean_vert_iou)
    write_detail_csv(CASE, detail_rows)

if __name__ == "__main__":
    main()