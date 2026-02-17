#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation (per-model, single case) → ONE CSV only (always overwritten)

Writes ONE file per run & case:
  baseline_outputs/evaluation_metrics_<CASE>.csv

Each row is either:
  - a global 'Spine' row for a model (global Dice/IoU), or
  - a per-vertebra row for a matched GT level (C1..L5) with Dice/IoU.

Key points:
  • TotalSegmentator (TS) is evaluated in TS ref space (first per-level NIfTI):
      - GT is resampled → TS ref (not the other way).
      - No axial crop for TS.
      - Name-based matching for TS (guarantees C1–T4 rows if present in GT).
      - min_pred_voxels=1 for TS (don’t filter).
  • Other models: CT is the ref (resample GT/preds → CT; optional axial crop).
  • Union is rebuilt from per-level files in the chosen ref space.
  • Robust discovery: standardized per-level + TS-native vertebrae_C*.nii.gz.
  • CSV writing:
      - Always opened with "w" (overwrite if exists).
      - If the FS is full: use --stdout or set EVAL_OUTDIR to a path with space.

Run:
  python evaluation_metrics.py
  python evaluation_metrics.py --case sub-gl017 --models "TotalSegmentator,NNUnet Framework"
"""

from __future__ import annotations
import os
import re
import sys
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib

# Optional resampling / matching
try:
    from nibabel.processing import resample_from_to
    NIB_RESAMPLE_OK = True
except Exception:
    NIB_RESAMPLE_OK = False

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# Progress bars (quiet fallback)
try:
    from tqdm import tqdm
    def pbar(it, desc): return tqdm(it, desc=desc, leave=False)
except Exception:
    def pbar(it, desc): return it


# ----------------------------
# Defaults & paths
# ----------------------------
def _norm_case(c: str) -> str:
    c = (c or "").strip()
    return c if c.startswith("sub-") else f"sub-{c}"

DEFAULT_CASE = _norm_case("gl017")
DEFAULT_MODELS = ["TotalSegmentator", "NNUnet Framework", "TotalSpineSeg", "SamMed3D"]

MATCH_POLICY = "optimal"    # default for non-TS
MIN_PRED_VOXELS_DEFAULT = 800
CROP_MARGIN_SLICES = 6

THIS_FILE = Path(__file__).resolve()
WSN_DIR   = THIS_FILE.parent
BO_DIR    = WSN_DIR / "baseline_outputs"
MODELS_ROOT = BO_DIR / "models"
RECON_ROOT  = BO_DIR / "reconstructed"

DATA_CAND_ROOTS = [
    WSN_DIR.parent / "data" / "Verse20" / "dataset-02validation",
    WSN_DIR.parent.parent / "wisespine" / "data" / "Verse20" / "dataset-02validation",
]

def find_ct_path(case: str) -> Optional[Path]:
    for base in DATA_CAND_ROOTS:
        raw = base / "rawdata" / case
        for fn in (f"{case}_ct.nii.gz", f"{case}_CT.nii.gz", "ct.nii.gz", f"{case}_ct.nii", f"{case}_CT.nii"):
            p = raw / fn
            if p.exists():
                return p
    return None

def gt_roots_for(case: str) -> List[Path]:
    base_dir = WSN_DIR.parent
    return [
        BO_DIR / "ground_truth" / case,
        WSN_DIR / "ground_truth" / case,
        base_dir / "data" / "Verse20" / "dataset-02validation" / "derivatives" / case,
        base_dir / "data" / "Verse20" / "dataset-02validation" / "segmentations" / case,
        base_dir / "data" / "Verse20" / "dataset-02validation" / "rawdata" / case / "segmentations",
    ]

EPS = 1e-8

LEVELS = [*(f"C{i}" for i in range(1, 8)),
          *(f"T{i}" for i in range(1, 13)),
          *(f"L{i}" for i in range(1, 6))]

PRED_NAME_PAT = re.compile(r"[._-]pred-vert_([CTL]\d+)\.nii(?:\.gz)?$", re.IGNORECASE)


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

def dice_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.sum(pred & gt); denom = np.sum(pred) + np.sum(gt)
    return float(2.0 * inter / (denom + EPS)) if denom > 0 else 1.0

def iou_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.sum(pred & gt); union = np.sum(pred | gt)
    return float(inter / (union + EPS)) if union > 0 else 1.0

def _extract_level_from_name(name: str) -> Optional[str]:
    m = PRED_NAME_PAT.search(name)
    if m:
        lvl = m.group(1).upper()
        return lvl if lvl[0] in ("C","T","L") else None
    m2 = re.search(r"([CTL]\d+)", name, re.IGNORECASE)
    return m2.group(1).upper() if m2 else None


# ----------------------------
# Ground truth loaders → arbitrary REF
# ----------------------------
def _find_gt_per_vertebra_dir(gt_roots: List[Path]) -> Optional[Path]:
    for root in gt_roots:
        if not root.exists():
            continue
        files = list(root.glob("*.nii*"))
        if any(any(k in p.name for k in LEVELS) for p in files):
            return root
    return None

def _find_gt_labeled_pair(gt_roots: List[Path]) -> Optional[Tuple[Path, Path]]:
    for root in gt_roots:
        for nm in ("segmentations.nii.gz", "segmentations.nii"):
            seg = root / nm
            names = root / "segment_names.json"
            if seg.exists() and names.exists():
                return seg, names
    return None

def _find_gt_verse_mask(gt_roots: List[Path]) -> Optional[Path]:
    for root in gt_roots:
        if not root.exists():
            continue
        hits = list(root.glob("*seg-vert_msk.nii*"))
        if hits:
            return hits[0]
    return None

def _extract_level_from_label_name(name: str) -> Optional[str]:
    m = re.search(r"((?:[CTL]\d+))", str(name), re.IGNORECASE)
    return m.group(1).upper() if m else None

def load_gt_masks_to_ref(ref_img: nib.Nifti1Image, gt_roots: List[Path]) -> Tuple[Dict[str, np.ndarray], nib.Nifti1Image]:
    gt: Dict[str, np.ndarray] = {}

    # (1) Per-vertebra GT files
    vdir = _find_gt_per_vertebra_dir(gt_roots)
    if vdir:
        print(f"🔎 GT per-vertebra dir: {vdir}")
        for lvl in LEVELS:
            cands = [p for p in vdir.glob(f"*{lvl}*.nii*")]
            if not cands:
                continue
            img = _to_can(nib.load(cands[0]))
            if img.shape != ref_img.shape or not np.allclose(img.affine, ref_img.affine):
                img = _resample_like(img, ref_img, order=0)
            gt[lvl] = (img.get_fdata() > 0)
        if gt:
            return gt, ref_img

    # (2) Labeled + names
    pair = _find_gt_labeled_pair(gt_roots)
    if pair:
        seg_nii, seg_map = pair
        print(f"🔎 GT labeled: {seg_nii} with {seg_map}")
        with open(seg_map) as f:
            names = json.load(f)
        mapping = {int(k): str(v) for k, v in names.items()}
        img = _to_can(nib.load(seg_nii))
        if img.shape != ref_img.shape or not np.allclose(img.affine, ref_img.affine):
            img = _resample_like(img, ref_img, order=0)
        lab = img.get_fdata().astype(np.int32)
        for idx, nm in mapping.items():
            lvl = _extract_level_from_label_name(nm)
            if not lvl:
                continue
            gt[lvl] = (lab == int(idx))
        if gt:
            return gt, img

    # (3) VERSe single mask
    verse = _find_gt_verse_mask(gt_roots)
    if verse:
        print(f"🔎 GT VERSe mask: {verse}")
        img = _to_can(nib.load(verse))
        if img.shape != ref_img.shape or not np.allclose(img.affine, ref_img.affine):
            img = _resample_like(img, ref_img, order=0)
        lab = img.get_fdata().astype(np.int32)
        labels = [int(v) for v in np.unique(lab) if int(v) > 0]
        comps = []
        for li in labels:
            coords = np.argwhere(lab == li)
            if coords.size == 0:
                continue
            comps.append((li, float(coords[:, 2].mean())))
        comps.sort(key=lambda t: t[1])  # head -> tail
        for idx, (li, _) in enumerate(comps):
            if idx >= len(LEVELS):
                break
            gt[LEVELS[idx]] = (lab == int(li))
        if gt:
            return gt, img

    raise FileNotFoundError("❌ Ground truth not found in expected locations.")


# ----------------------------
# Predictions → REF
# ----------------------------
def _model_dirs(model: str, case: str) -> Tuple[Path, Path, Path]:
    model_dir = MODELS_ROOT / model / case
    recon_dir = RECON_ROOT / model / case
    union_primary = recon_dir / f"{case}_{model}_predicted_spine_union.nii.gz"
    union_alt     = recon_dir / f"{case}_predicted_spine_union.nii.gz"
    union_path = union_primary if union_primary.exists() else (union_alt if union_alt.exists() else union_primary)
    return model_dir, recon_dir, union_path

def collect_pred_masks(model: str, case: str) -> List[Path]:
    model_dir, _, _ = _model_dirs(model, case)
    if not model_dir.exists():
        return []
    files: List[Path] = []
    files = sorted(model_dir.glob(f"{case}_{model}_pred-vert_[CTL][0-9]*.nii*"))
    if not files:
        files = sorted(model_dir.glob(f"{case}_pred-vert_[CTL][0-9]*.nii*"))
    ts_native = sorted(model_dir.glob("vertebrae_[cCtTlL][0-9]*.nii*"))
    seen = set(p.resolve() for p in files)
    for p in ts_native:
        rp = p.resolve()
        if rp not in seen:
            files.append(p)
            seen.add(rp)
    return files

def load_perlevel_in_ref(
    files: List[Path],
    ref_img: nib.Nifti1Image,
    min_vox: int,
    do_crop: bool,
    crop_gate: Optional[np.ndarray],
    crop_margin: int
) -> Tuple[List[np.ndarray], List[str]]:
    comps, names = [], []
    for fp in pbar(files, "per-level → REF"):
        lvl = _extract_level_from_name(fp.name) or "UNK"
        try:
            img = _to_can(nib.load(fp))
            if img.shape != ref_img.shape or not np.allclose(img.affine, ref_img.affine):
                img = _resample_like(img, ref_img, order=0)
        except Exception:
            continue
        comp = (img.get_fdata() > 0)
        if do_crop and crop_gate is not None:
            nz = np.where(crop_gate)
            if len(nz[0]) > 0:
                zmin, zmax = int(np.min(nz[2])), int(np.max(nz[2]))
                zmin = max(0, zmin - crop_margin)
                zmax = min(comp.shape[2] - 1, zmax + crop_margin)
                gate = np.zeros_like(comp, dtype=bool)
                gate[:, :, zmin:zmax+1] = True
                comp &= gate
        if int(np.count_nonzero(comp)) < min_vox:
            continue
        comps.append(comp.astype(bool))
        names.append(lvl)
    return comps, names

def build_union_from_perlevels(
    files: List[Path],
    ref_img: nib.Nifti1Image,
    do_crop: bool,
    crop_gate: Optional[np.ndarray],
    crop_margin: int
) -> np.ndarray:
    union = np.zeros(ref_img.shape, dtype=bool)
    for fp in pbar(files, "Build union"):
        try:
            img = _to_can(nib.load(fp))
            if img.shape != ref_img.shape or not np.allclose(img.affine, ref_img.affine):
                img = _resample_like(img, ref_img, order=0)
            mask = (img.get_fdata() > 0)
            if do_crop and crop_gate is not None:
                nz = np.where(crop_gate)
                if len(nz[0]) > 0:
                    zmin, zmax = int(np.min(nz[2])), int(np.max(nz[2]))
                    zmin = max(0, zmin - crop_margin)
                    zmax = min(mask.shape[2] - 1, zmax + crop_margin)
                    gate = np.zeros_like(mask, dtype=bool)
                    gate[:, :, zmin:zmax+1] = True
                    mask &= gate
            union |= mask
        except Exception:
            continue
    return union


# ----------------------------
# Matching & metrics
# ----------------------------
def overlap_matrix(
    pred_components: List[np.ndarray],
    gt_levels: Dict[str, np.ndarray],
    gt_ordered: List[str]
) -> np.ndarray:
    n_pred, n_gt = len(pred_components), len(gt_ordered)
    M = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i in range(n_pred):
        pc = pred_components[i]
        if not np.any(pc):
            continue
        pc_sum = pc.sum()
        for j, lvl in enumerate(gt_ordered):
            g = gt_levels.get(lvl)
            if g is None:
                continue
            inter = np.sum(pc & g)
            union = pc_sum + g.sum() - inter
            M[i, j] = (inter / (union + EPS)) if union > 0 else 0.0
    return M

def match_and_score(
    pred_components: List[np.ndarray],
    pred_names: List[str],
    gt_levels: Dict[str, np.ndarray],
    gt_ordered_levels: List[str],
    policy: str
):
    if len(pred_components) == 0 or len(gt_ordered_levels) == 0:
        return []

    results = []

    if policy == "name":
        name_to_idx = {pred_names[i]: i for i in range(len(pred_names))}
        for lvl in gt_ordered_levels:
            if lvl in name_to_idx:
                i = name_to_idx[lvl]
                pc = pred_components[i]
                g = gt_levels.get(lvl)
                if g is None:
                    continue
                inter = int(np.sum(pc & g))
                pv = int(np.sum(pc))
                gv = int(np.sum(g))
                dice = (2.0 * inter) / (pv + gv + EPS) if (pv + gv) > 0 else 1.0
                union = pv + gv - inter
                iou = inter / (union + EPS) if union > 0 else 1.0
                results.append({
                    "pred_idx": i,
                    "pred_name": pred_names[i],
                    "level": lvl,
                    "dice": float(dice),
                    "iou": float(iou),
                    "pred_voxels": pv,
                    "gt_voxels": gv,
                    "intersection": inter
                })
        return results

    # Optimal assignment by IoU
    if SCIPY_OK:
        M = overlap_matrix(pred_components, gt_levels, gt_ordered_levels)
        if M.size > 0:
            cost = 1.0 - M
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if i < 0 or j < 0:
                    continue
                pc = pred_components[i]
                lvl = gt_ordered_levels[j]
                g = gt_levels.get(lvl)
                if g is None:
                    continue
                inter = int(np.sum(pc & g))
                pv = int(np.sum(pc))
                gv = int(np.sum(g))
                dice = (2.0 * inter) / (pv + gv + EPS) if (pv + gv) > 0 else 1.0
                iou = float(M[i, j])
                results.append({
                    "pred_idx": i,
                    "pred_name": pred_names[i] if i < len(pred_names) else f"pred_{i}",
                    "level": lvl,
                    "dice": float(dice),
                    "iou": iou,
                    "pred_voxels": pv,
                    "gt_voxels": gv,
                    "intersection": inter
                })
        return results

    # Greedy fallback
    M = overlap_matrix(pred_components, gt_levels, gt_ordered_levels)
    used = set()
    for i in range(M.shape[0]):
        if M.shape[1] == 0:
            break
        j = int(np.argmax(M[i, :]))
        if j >= 0 and M[i, j] > 0 and j not in used:
            pc = pred_components[i]
            lvl = gt_ordered_levels[j]
            g = gt_levels.get(lvl)
            if g is None:
                continue
            inter = int(np.sum(pc & g))
            pv = int(np.sum(pc))
            gv = int(np.sum(g))
            dice = (2.0 * inter) / (pv + gv + EPS) if (pv + gv) > 0 else 1.0
            iou = float(M[i, j])
            results.append({
                "pred_idx": i,
                "pred_name": pred_names[i] if i < len(pred_names) else f"pred_{i}",
                "level": lvl,
                "dice": float(dice),
                "iou": iou,
                "pred_voxels": pv,
                "gt_voxels": gv,
                "intersection": inter
            })
            used.add(j)
    return results


# ----------------------------
# CSV writer with overwrite, stdout option, and env fallback
# ----------------------------
def _open_csv_overwrite_with_fallback(path: Path, to_stdout: bool):
    """
    Open CSV for overwrite, or to stdout, or to EVAL_OUTDIR/$HOME fallback if FS is full.
    Returns (handle, real_path_or_None, is_stdout).
    """
    if to_stdout:
        return (sys.stdout, None, True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # "w" → always overwrite if exists
        return (path.open("w", newline=""), path, False)
    except OSError as e:
        print(f"⚠️ Cannot write to {path} ({e}). Trying fallback...")
        alt_env = os.environ.get("EVAL_OUTDIR", "").strip()
        if alt_env:
            alt = Path(alt_env) / path.name
            try:
                alt.parent.mkdir(parents=True, exist_ok=True)
                return (alt.open("w", newline=""), alt, False)
            except OSError as e2:
                print(f"⚠️ Cannot write to {alt} ({e2}).")
        home_alt = Path(os.path.expanduser("~")) / "wisespine_eval" / path.name
        try:
            home_alt.parent.mkdir(parents=True, exist_ok=True)
            return (home_alt.open("w", newline=""), home_alt, False)
        except OSError as e3:
            print(f"⚠️ Cannot write to {home_alt} ({e3}). Falling back to stdout.")
            return (sys.stdout, None, True)

class SafeCSVWriter:
    """Wrapper that can switch to stdout if a mid-write OSError occurs (quota/full)."""
    def __init__(self, fh, real_path, is_stdout):
        self.fh = fh
        self.real_path = real_path
        self.is_stdout = is_stdout
        self.writer = csv.writer(fh)

    def write_row(self, row):
        try:
            self.writer.writerow(row)
            if self.is_stdout:
                self.fh.flush()
        except OSError as e:
            if not self.is_stdout:
                print(f"⚠️ Write failed to {self.real_path} ({e}). Switching to stdout.")
                self.fh = sys.stdout
                self.real_path = None
                self.is_stdout = True
                self.writer = csv.writer(self.fh)
                self.writer.writerow(row)
            else:
                raise

    def close(self):
        try:
            if (not self.is_stdout) and self.fh:
                self.fh.close()
        except Exception:
            pass


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate spine segmentation per model (single case).")
    ap.add_argument("--case", type=str, default=DEFAULT_CASE, help="Case ID (e.g., sub-gl017)")
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                    help="Comma-separated list of models to evaluate")
    ap.add_argument("--policy", type=str, default=MATCH_POLICY, choices=["optimal", "name"],
                    help="Default matching policy (TS will use name regardless)")
    ap.add_argument("--no-crop", action="store_true", help="Disable axial crop (non-TS models only)")
    ap.add_argument("--min-pred-voxels", type=int, default=MIN_PRED_VOXELS_DEFAULT)
    ap.add_argument("--crop-margin", type=int, default=CROP_MARGIN_SLICES)
    ap.add_argument("--stdout", action="store_true", help="Write CSV to stdout (never touch disk)")
    return ap.parse_args()


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    case = _norm_case(args.case)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    min_vox_default = int(args.min_pred_voxels)
    crop_margin = int(args.crop_margin)

    print(f"🦴 Evaluating case: {case}")
    gt_roots = gt_roots_for(case)

    # CT reference (for non-TS)
    ct_path = find_ct_path(case)
    ct_img: Optional[nib.Nifti1Image] = _to_can(nib.load(ct_path)) if ct_path else None
    if ct_img is not None:
        print(f"📌 CT reference available: {ct_path}")
    else:
        print("⚠️ CT not found; non-TS models will fall back to GT-derived reference.")

    # Prepare CSV (overwrite, with fallback)
    out_csv = BO_DIR / f"evaluation_metrics_{case}.csv"
    header = [
        "timestamp","model","case","level","dice","iou",
        "pred_idx","pred_name","pred_voxels","gt_voxels","intersection",
        "global_dice","global_iou","mean_vert_dice","mean_vert_iou",
        "n_pred_levels","n_gt_levels","union_voxels"
    ]
    fh, real_path, is_stdout = _open_csv_overwrite_with_fallback(out_csv, args.stdout)
    w_safe = SafeCSVWriter(fh, real_path, is_stdout)
    w_safe.write_row(header)

    for model in models:
        files = collect_pred_masks(model, case)

        print("\n==============================")
        print(f"▶️ Model: {model}")
        print(f"   found per-level: {len(files)}")
        print("==============================")

        if model == "TotalSegmentator":
            if not files:
                print("⚠️ No per-level files for TotalSegmentator.")
                continue

            # REF = first TS per-level NIfTI (evaluate in TS space)
            ref_img = _to_can(nib.load(files[0]))
            gt_levels, _ = load_gt_masks_to_ref(ref_img, gt_roots)
            gt_union = np.zeros(ref_img.shape, dtype=bool)
            for _, msk in gt_levels.items():
                gt_union |= msk
            ordered_gt = [lvl for lvl in LEVELS if lvl in gt_levels]

            # TS evaluation settings
            do_crop = False
            crop_gate = None
            policy_to_use = "name"     # force name matching for TS
            min_vox = 1                # do not filter away components

            pred_components, pred_names = load_perlevel_in_ref(
                files, ref_img, min_vox, do_crop, crop_gate, crop_margin
            )
            pred_union = build_union_from_perlevels(files, ref_img, do_crop, crop_gate, crop_margin)

        else:
            # REF = CT for non-TS (or GT-derived if CT missing)
            if ct_img is not None:
                ref_img = ct_img
            else:
                ref_img = nib.Nifti1Image(np.zeros((2,2,2)), np.eye(4))
            gt_levels, ref_or_gt = load_gt_masks_to_ref(ref_img, gt_roots)
            if ct_img is None:
                ref_img = ref_or_gt

            gt_union = np.zeros(ref_img.shape, dtype=bool)
            for _, msk in gt_levels.items():
                gt_union |= msk
            ordered_gt = [lvl for lvl in LEVELS if lvl in gt_levels]

            do_crop = not args.no_crop
            crop_gate = gt_union if do_crop else None
            policy_to_use = args.policy
            min_vox = min_vox_default

            pred_components, pred_names = load_perlevel_in_ref(
                files, ref_img, min_vox, do_crop, crop_gate, crop_margin
            )
            pred_union = build_union_from_perlevels(files, ref_img, do_crop, crop_gate, crop_margin)

        n_pred_levels = len(pred_components)
        union_vox = int(pred_union.sum())
        print(f"   Union voxels (REF): {union_vox}")

        global_dice = dice_binary(pred_union, gt_union)
        global_iou  = iou_binary(pred_union, gt_union)
        print(f"✅ Global Dice: {global_dice:.6f}")
        print(f"✅ Global IoU : {global_iou:.6f}")

        details = match_and_score(pred_components, pred_names, gt_levels, ordered_gt, policy_to_use)
        mean_vert_dice = float(np.mean([r["dice"] for r in details])) if details else None
        mean_vert_iou  = float(np.mean([r["iou"]  for r in details])) if details else None

        if model == "TotalSegmentator":
            expected = [f"C{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 5)]
            missing = [lvl for lvl in expected if lvl in ordered_gt and lvl not in [d["level"] for d in details]]
            if missing:
                print(f"⚠️ Missing TS levels in output rows (but present in GT): {', '.join(missing)}")
            else:
                print("✅ TS per-vertebra rows include all present levels (C1–T4 for this case).")

        now_str = datetime.now().isoformat(timespec="seconds")

        # Spine row
        w_safe.write_row([
            now_str,
            model, case, "Spine",
            f"{global_dice:.6f}", f"{global_iou:.6f}",
            "", "", "", "", "",
            f"{global_dice:.6f}", f"{global_iou:.6f}",
            "" if mean_vert_dice is None else f"{mean_vert_dice:.6f}",
            "" if mean_vert_iou  is None else f"{mean_vert_iou:.6f}",
            n_pred_levels, len(ordered_gt), union_vox
        ])
        # Per-level rows
        for r in details:
            w_safe.write_row([
                now_str,
                model, case, r["level"],
                f"{r['dice']:.6f}", f"{r['iou']:.6f}",
                r["pred_idx"], r["pred_name"], r["pred_voxels"], r["gt_voxels"], r["intersection"],
                f"{global_dice:.6f}", f"{global_iou:.6f}",
                "" if mean_vert_dice is None else f"{mean_vert_dice:.6f}",
                "" if mean_vert_iou  is None else f"{mean_vert_iou:.6f}",
                n_pred_levels, len(ordered_gt), union_vox
            ])

    w_safe.close()
    if w_safe.real_path is not None:
        print(f"\n✅ ONE CSV written (overwritten if existed): {w_safe.real_path}")
    else:
        print("\n⚠️ CSV printed to stdout (no file was written).")
    print("   Contains one 'Spine' row (global) + per-vertebra rows per model.")


if __name__ == "__main__":
    main()