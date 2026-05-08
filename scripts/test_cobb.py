#!/usr/bin/env python3
"""
evaluate.py

Full spine segmentation + Cobb angle evaluation pipeline for VerSe20 data.

Validation layers
-----------------
  1. Whole-spine reconstruction  DICE / IoU  (vs GT binary mask)
  2. Per-vertebra DICE / IoU                 (vs GT label volume)
  3. Per-vertebra centroid displacement      (vs GT mask-derived AND vs VerSe expert centroids)
  4. Per-vertebra endplate normal error      (angle between SVD normals from pred vs GT masks)
  5. Centroid-based curvature               (from VerSe expert centroids -> coronal segment tilts)
  6. Pseudo-GT Cobb angles                  (pipeline run on GT masks vs pred masks)
  7. Cross-validation                       (centroid curvature vs endplate Cobb, both sources)
  8. Per-vertebra reliability flag          (composite gate for Cobb measurement trust)

Coordinate convention
---------------------
  All physical-space computations are in world RAS mm via the NIfTI affine.
  Centroids are computed as affine @ [mean_vox, 1] so they share the same
  reference frame as VerSe expert centroids (also affine-converted from JSON).
  Do NOT compute centroids as voxel_index * spacing — that omits the origin
  translation and causes ~200 mm errors vs expert annotations.
"""

import glob
import json
import math
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

from segmentation import get_case_list
from cobb_angle_pipeline import (
    ENDPLATE_FRACTION,
    SORT_ORDER,
    _fit_plane_svd,
    normal_to_coronal_direction,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "TotalSegmentator"

BASE_GT_DIR     = Path("/gpfs/projects/wisemind/bhan830/wisespine/data/Verse20/dataset-01training/derivatives")
BASE_MODELS_DIR = Path(f"/gpfs/projects/wisemind/bhan830/wisespine/wisespine_new/baseline_outputs/models/{MODEL_NAME}")
BASE_RECON_DIR  = Path(f"/gpfs/projects/wisemind/bhan830/wisespine/wisespine_new/baseline_outputs/reconstructed/{MODEL_NAME}")

OUTPUT_CSV = Path("/gpfs/projects/wisemind/bhan830/wisespine/wisespine_new/baseline_outputs/evaluation_metrics.csv")

# Reliability thresholds
RELIABLE_DICE_MIN          = 0.80   # per-vertebra DICE
RELIABLE_CENTROID_DISP_MAX = 3.0    # mm — vs GT mask centroid
RELIABLE_EXPERT_DISP_MAX   = 15.0    # mm — vs VerSe expert centroid
RELIABLE_NORMAL_ANG_MAX    = 5.0    # degrees — endplate normal error

# Cobb cross-validation warning thresholds
COBB_PSEUDO_GT_DIFF_WARN   = 5.0    # degrees — pred vs GT-mask Cobb
COBB_CENTROID_CROSS_WARN   = 8.0    # degrees — endplate Cobb vs centroid curvature


# ══════════════════════════════════════════════════════════════════════════════
# LABEL MAPPING  (VerSe convention)
# ══════════════════════════════════════════════════════════════════════════════

def _build_label_mapping() -> dict:
    m, n = {}, 1
    for i in range(1,  8): m[f"C{i}"] = n; n += 1   # C1-C7   ->  1-7
    for i in range(1, 13): m[f"T{i}"] = n; n += 1   # T1-T12  ->  8-19
    for i in range(1,  6): m[f"L{i}"] = n; n += 1   # L1-L5   -> 20-24
    m["S1"] = n                                       # S1      -> 25
    return m

LABEL_MAP     = _build_label_mapping()
LABEL_TO_NAME = {v: k for k, v in LABEL_MAP.items()}
SORTED_VERTS  = sorted(LABEL_MAP.keys(), key=lambda x: SORT_ORDER.get(x, 99))


# ══════════════════════════════════════════════════════════════════════════════
# BASIC METRICS
# ══════════════════════════════════════════════════════════════════════════════

def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return 2.0 * inter / total if total > 0 else float("nan")

def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union > 0 else float("nan")

def load_bool(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata() > 0


# ══════════════════════════════════════════════════════════════════════════════
# CENTROID — world RAS mm via affine
# ══════════════════════════════════════════════════════════════════════════════

def mask_centroid_world_mm(mask_3d: np.ndarray,
                            affine: np.ndarray) -> Optional[np.ndarray]:
    """
    Centroid of a binary mask in world RAS mm.

    Uses affine @ [mean_vox_i, mean_vox_j, mean_vox_k, 1] so the result is
    in the same reference frame as VerSe expert centroids (also affine-derived).
    Computing as voxel_index * spacing omits the image origin translation and
    causes ~200 mm discrepancies against expert annotations.
    """
    coords = np.argwhere(mask_3d)
    if len(coords) == 0:
        return None
    mean_vox = np.append(coords.mean(axis=0), 1.0)
    return (affine @ mean_vox)[:3]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3A — Centroid displacement vs GT mask
# ══════════════════════════════════════════════════════════════════════════════

def centroid_displacement_vs_mask(pred_mask: np.ndarray,
                                   gt_mask: np.ndarray,
                                   affine: np.ndarray) -> dict:
    c_pred = mask_centroid_world_mm(pred_mask, affine)
    c_gt   = mask_centroid_world_mm(gt_mask,   affine)
    if c_pred is None or c_gt is None:
        return dict(centroid_disp_mask_mm=float("nan"),
                    centroid_disp_mask_x_mm=float("nan"),
                    centroid_disp_mask_z_mm=float("nan"))
    diff = c_pred - c_gt
    return dict(centroid_disp_mask_mm=float(np.linalg.norm(diff)),
                centroid_disp_mask_x_mm=float(abs(diff[0])),
                centroid_disp_mask_z_mm=float(abs(diff[2])))


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3B — VerSe expert centroid loading + displacement
# ══════════════════════════════════════════════════════════════════════════════

def load_verse_centroids(ctd_json_path: Path,
                          orig_nifti_path: Path) -> dict:
    """
    Load VerSe expert centroid annotations and convert to world RAS mm.

    JSON voxel indices are in the original image space. Applying the original
    NIfTI affine converts them to world mm — the same space produced by
    mask_centroid_world_mm when using the canonical affine, because world
    coordinates are orientation-independent.
    """
    if not ctd_json_path.exists():
        return {}
    orig_affine = nib.load(str(orig_nifti_path)).affine
    with open(ctd_json_path) as f:
        ctd = json.load(f)
    centroids = {}
    for entry in ctd:
        if "label" not in entry:
            continue
        vox_h = np.array([entry["X"], entry["Y"], entry["Z"], 1.0])
        centroids[entry["label"]] = (orig_affine @ vox_h)[:3]
    return centroids


def centroid_displacement_vs_expert(pred_mask: np.ndarray,
                                     affine: np.ndarray,
                                     expert_centroid_mm: Optional[np.ndarray]) -> dict:
    """
    Displacement between pred-mask centroid (world mm) and VerSe expert centroid.
    Both are now in the same world RAS mm reference frame.
    """
    if expert_centroid_mm is None:
        return dict(centroid_disp_expert_mm=float("nan"),
                    centroid_disp_expert_x_mm=float("nan"),
                    centroid_disp_expert_z_mm=float("nan"))
    c_pred = mask_centroid_world_mm(pred_mask, affine)
    if c_pred is None:
        return dict(centroid_disp_expert_mm=float("nan"),
                    centroid_disp_expert_x_mm=float("nan"),
                    centroid_disp_expert_z_mm=float("nan"))
    diff = c_pred - expert_centroid_mm
    return dict(centroid_disp_expert_mm=float(np.linalg.norm(diff)),
                centroid_disp_expert_x_mm=float(abs(diff[0])),
                centroid_disp_expert_z_mm=float(abs(diff[2])))


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — Endplate normal angular error
# ══════════════════════════════════════════════════════════════════════════════

def _endplate_pts(mask_3d: np.ndarray,
                  spacing: np.ndarray,
                  top: bool) -> Optional[np.ndarray]:
    xs, ys, zs = np.where(mask_3d > 0)
    if len(xs) < 20:
        return None
    pts   = np.column_stack([xs, ys, zs]) * spacing
    z_val = pts[:, 2]
    pct   = 100 * (1 - ENDPLATE_FRACTION) if top else 100 * ENDPLATE_FRACTION
    sel   = z_val >= np.percentile(z_val, pct) if top else z_val <= np.percentile(z_val, pct)
    return pts[sel]


def endplate_normal_error(pred_mask: np.ndarray,
                           gt_mask: np.ndarray,
                           spacing: np.ndarray) -> dict:
    out = {}
    for ep, top in [("superior", True), ("inferior", False)]:
        pts_p = _endplate_pts(pred_mask, spacing, top)
        pts_g = _endplate_pts(gt_mask,   spacing, top)
        key   = f"endplate_{ep}_normal_err_deg"
        if pts_p is None or pts_g is None:
            out[key] = float("nan")
            continue
        n_p, _ = _fit_plane_svd(pts_p)
        n_g, _ = _fit_plane_svd(pts_g)
        dot    = float(np.clip(abs(np.dot(n_p, n_g)), 0.0, 1.0))
        out[key] = math.degrees(math.acos(dot))
    sup = out.get("endplate_superior_normal_err_deg", float("nan"))
    inf = out.get("endplate_inferior_normal_err_deg", float("nan"))
    out["endplate_max_normal_err_deg"] = float(np.nanmax([sup, inf]))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — Centroid-based curvature from VerSe expert centroids
# ══════════════════════════════════════════════════════════════════════════════

def _segment_tilt_deg(c_upper: np.ndarray, c_lower: np.ndarray) -> float:
    """
    Angle (degrees) between the vector connecting two centroids and vertical
    (Z axis in RAS), projected onto the coronal (X-Z) plane.
    """
    vec = c_lower - c_upper
    vec_coronal = np.array([vec[0], vec[2]])
    if np.linalg.norm(vec_coronal) < 1e-6:
        return float("nan")
    vec_coronal /= np.linalg.norm(vec_coronal)
    dot = float(np.clip(abs(np.dot(vec_coronal, np.array([0.0, -1.0]))), 0.0, 1.0))
    return math.degrees(math.acos(dot))


def centroid_curvature_from_experts(expert_centroids: dict,
                                     present_labels: list) -> tuple:
    out, tilt_map = {}, {}
    annotated = sorted([l for l in present_labels if l in expert_centroids])
    for i in range(len(annotated) - 1):
        ul, ll  = annotated[i], annotated[i + 1]
        uname   = LABEL_TO_NAME.get(ul, str(ul))
        lname   = LABEL_TO_NAME.get(ll, str(ll))
        tilt    = _segment_tilt_deg(expert_centroids[ul], expert_centroids[ll])
        out[f"ctd_tilt_{uname}_{lname}_deg"] = round(tilt, 3)
        tilt_map[(ul, ll)] = tilt
    return out, tilt_map


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — Pseudo-GT Cobb angles
# ══════════════════════════════════════════════════════════════════════════════

def _cobb_from_masks(mask_upper: np.ndarray,
                     mask_lower: np.ndarray,
                     spacing: np.ndarray) -> Optional[float]:
    pts_sup = _endplate_pts(mask_upper, spacing, top=True)
    pts_inf = _endplate_pts(mask_lower, spacing, top=False)
    if pts_sup is None or pts_inf is None:
        return None
    n_sup, _ = _fit_plane_svd(pts_sup)
    n_inf, _ = _fit_plane_svd(pts_inf)
    d_sup    = normal_to_coronal_direction(n_sup)
    d_inf    = normal_to_coronal_direction(n_inf)
    dot      = float(np.clip(abs(np.dot(d_sup, d_inf)), 0.0, 1.0))
    return math.degrees(math.acos(dot))


def pseudo_gt_cobb(gt_label_vol: np.ndarray,
                   pred_masks: dict,
                   spacing: np.ndarray) -> tuple:
    out, pair_map = {}, {}
    present = [v for v in SORTED_VERTS
               if v in pred_masks and (gt_label_vol == LABEL_MAP[v]).any()]
    for i in range(len(present) - 1):
        upper, lower = present[i], present[i + 1]
        key = f"cobb_{upper}_{lower}"
        gt_upper = gt_label_vol == LABEL_MAP[upper]
        gt_lower = gt_label_vol == LABEL_MAP[lower]
        gt_angle = _cobb_from_masks(gt_upper,          gt_lower,          spacing)
        pd_angle = _cobb_from_masks(pred_masks[upper], pred_masks[lower], spacing)
        out[f"{key}_gt_deg"]   = round(gt_angle, 3) if gt_angle is not None else float("nan")
        out[f"{key}_pred_deg"] = round(pd_angle, 3) if pd_angle is not None else float("nan")
        if gt_angle is not None and pd_angle is not None:
            diff = abs(pd_angle - gt_angle)
            out[f"{key}_diff_deg"] = round(diff, 3)
            out[f"{key}_flagged"]  = diff > COBB_PSEUDO_GT_DIFF_WARN
        else:
            out[f"{key}_diff_deg"] = float("nan")
            out[f"{key}_flagged"]  = True
        ul, ll = LABEL_MAP.get(upper), LABEL_MAP.get(lower)
        if ul and ll:
            pair_map[(ul, ll)] = (gt_angle, pd_angle)
    return out, pair_map


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 7 — Cross-validation: endplate Cobb vs centroid curvature
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate_cobb_vs_centroids(cobb_pair_map: dict,
                                      tilt_map: dict) -> dict:
    """
    Compare GT-mask Cobb and pred-mask Cobb against expert centroid tilt.

    Interpretation guide:
      gt_cobb_vs_ctd large, pred_cobb_vs_ctd large  -> pipeline methodology issue
      gt_cobb_vs_ctd small, pred_cobb_vs_ctd large  -> TotalSegmentator segmentation issue
      both small                                     -> consistent and trustworthy
    """
    out = {}
    for pair, (gt_cobb, pd_cobb) in cobb_pair_map.items():
        if pair not in tilt_map:
            continue
        tilt   = tilt_map[pair]
        ul, ll = pair
        prefix = f"xval_{LABEL_TO_NAME.get(ul, ul)}_{LABEL_TO_NAME.get(ll, ll)}"
        out[f"{prefix}_ctd_tilt_deg"] = round(tilt, 3)
        if gt_cobb is not None and not math.isnan(tilt):
            d = abs(gt_cobb - tilt)
            out[f"{prefix}_gt_cobb_vs_ctd_deg"]     = round(d, 3)
            out[f"{prefix}_gt_cobb_vs_ctd_flagged"] = d > COBB_CENTROID_CROSS_WARN
        else:
            out[f"{prefix}_gt_cobb_vs_ctd_deg"]     = float("nan")
            out[f"{prefix}_gt_cobb_vs_ctd_flagged"] = True
        if pd_cobb is not None and not math.isnan(tilt):
            d = abs(pd_cobb - tilt)
            out[f"{prefix}_pred_cobb_vs_ctd_deg"]     = round(d, 3)
            out[f"{prefix}_pred_cobb_vs_ctd_flagged"] = d > COBB_CENTROID_CROSS_WARN
        else:
            out[f"{prefix}_pred_cobb_vs_ctd_deg"]     = float("nan")
            out[f"{prefix}_pred_cobb_vs_ctd_flagged"] = True
    return out


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 8 — Per-vertebra reliability flag
# ══════════════════════════════════════════════════════════════════════════════

def reliability_flag(dice_val: float,
                     disp_mask_mm: float,
                     disp_expert_mm: float,
                     max_normal_err_deg: float) -> bool:
    if math.isnan(dice_val) or dice_val < RELIABLE_DICE_MIN:
        return False
    if math.isnan(disp_mask_mm) or disp_mask_mm > RELIABLE_CENTROID_DISP_MAX:
        return False
    if not math.isnan(disp_expert_mm) and disp_expert_mm > RELIABLE_EXPERT_DISP_MAX:
        return False
    if math.isnan(max_normal_err_deg) or max_normal_err_deg > RELIABLE_NORMAL_ANG_MAX:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# CASE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_case(case_id: str) -> Optional[dict]:
    r = {"model": MODEL_NAME, "case": case_id}

    gt_seg_path = BASE_GT_DIR / case_id / f"{case_id}_dir-ax_seg-vert_msk.nii.gz"
    gt_ctd_path = BASE_GT_DIR / case_id / f"{case_id}_dir-ax_seg-subreg_ctd.json"
    recon_path  = BASE_RECON_DIR / case_id / f"{case_id}_{MODEL_NAME}_spine_union.nii.gz"
    mask_dir    = BASE_MODELS_DIR / case_id

    for label, path in [("GT seg", gt_seg_path), ("recon", recon_path)]:
        if not path.exists():
            print(f"  x {label} not found: {path}")
            return None

    # Load GT label volume — canonicalize to RAS
    gt_img      = nib.load(str(gt_seg_path))
    gt_ras_img  = nib.as_closest_canonical(gt_img)
    gt_vol      = gt_ras_img.get_fdata()
    canon_affine = gt_ras_img.affine                          # world RAS mm reference
    spacing      = np.array(gt_ras_img.header.get_zooms()[:3], dtype=float)

    # Load VerSe expert centroids using the *original* affine
    expert_centroids = load_verse_centroids(gt_ctd_path, gt_seg_path)
    n_expert = len(expert_centroids)
    r["n_expert_centroids"] = n_expert
    if n_expert == 0:
        print(f"  ! No expert centroid JSON: {gt_ctd_path}")
    else:
        names = [LABEL_TO_NAME.get(l, str(l)) for l in sorted(expert_centroids)]
        print(f"  Expert centroids ({n_expert}): {', '.join(names)}")

    # Layer 1 — whole-spine reconstruction
    recon_mask = nib.as_closest_canonical(nib.load(str(recon_path))).get_fdata() > 0
    r["spine_recon_DICE"] = dice(recon_mask, gt_vol > 0)
    r["spine_recon_IoU"]  = iou(recon_mask,  gt_vol > 0)

    # Per-vertebra loop
    vert_files = sorted(
        glob.glob(str(mask_dir / f"{case_id}_{MODEL_NAME}_vertebrae_*.nii.gz"))
    )
    if not vert_files:
        print(f"  x No individual vertebra masks: {mask_dir}")
        return r

    pred_masks: dict  = {}
    present_labels: list = []

    for vf in vert_files:
        stem   = Path(vf).name
        vname  = stem.split(f"{case_id}_{MODEL_NAME}_vertebrae_")[-1].replace(".nii.gz", "")
        vlabel = LABEL_MAP.get(vname)

        pred_ras  = nib.as_closest_canonical(nib.load(vf))
        pred_mask = pred_ras.get_fdata() > 0
        pred_masks[vname] = pred_mask

        gt_exists = vlabel is not None and (gt_vol == vlabel).any()
        gt_vert   = (gt_vol == vlabel) if gt_exists else None

        # Layer 2 — DICE / IoU
        d_val = dice(pred_mask, gt_vert) if gt_exists else float("nan")
        i_val = iou( pred_mask, gt_vert) if gt_exists else float("nan")
        r[f"{vname}_DICE"] = d_val
        r[f"{vname}_IoU"]  = i_val

        # Layer 3a — centroid vs GT mask (world mm)
        disp_mask = (centroid_displacement_vs_mask(pred_mask, gt_vert, canon_affine)
                     if gt_exists
                     else dict(centroid_disp_mask_mm=float("nan"),
                               centroid_disp_mask_x_mm=float("nan"),
                               centroid_disp_mask_z_mm=float("nan")))
        for k, v in disp_mask.items():
            r[f"{vname}_{k}"] = v

        # Layer 3b — centroid vs VerSe expert (world mm, same frame)
        expert_c    = expert_centroids.get(vlabel)
        disp_expert = centroid_displacement_vs_expert(pred_mask, canon_affine, expert_c)
        for k, v in disp_expert.items():
            r[f"{vname}_{k}"] = v
        if expert_c is not None:
            present_labels.append(vlabel)

        # Layer 4 — endplate normal error
        nrm = (endplate_normal_error(pred_mask, gt_vert, spacing)
               if gt_exists
               else dict(endplate_superior_normal_err_deg=float("nan"),
                         endplate_inferior_normal_err_deg=float("nan"),
                         endplate_max_normal_err_deg=float("nan")))
        for k, v in nrm.items():
            r[f"{vname}_{k}"] = v

        # Layer 8 — reliability flag
        r[f"{vname}_cobb_reliable"] = reliability_flag(
            d_val,
            disp_mask["centroid_disp_mask_mm"],
            disp_expert["centroid_disp_expert_mm"],
            nrm["endplate_max_normal_err_deg"],
        )

    # Layer 5 — centroid curvature from VerSe experts
    ctd_metrics, tilt_map = centroid_curvature_from_experts(expert_centroids, present_labels)
    r.update(ctd_metrics)
    r["n_expert_curvature_pairs"] = len(tilt_map)

    # Layer 6 — pseudo-GT Cobb
    cobb_metrics, cobb_pair_map = pseudo_gt_cobb(gt_vol, pred_masks, spacing)
    r.update(cobb_metrics)

    # Layer 7 — cross-validation
    r.update(cross_validate_cobb_vs_centroids(cobb_pair_map, tilt_map))

    # Case-level summary scalars
    def _nanmean(suffix):
        vals = [v for k, v in r.items()
                if k.endswith(suffix) and isinstance(v, float) and not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    r["mean_vert_DICE"]               = _nanmean("_DICE")
    r["mean_centroid_disp_mask_mm"]   = _nanmean("centroid_disp_mask_mm")
    r["mean_centroid_disp_expert_mm"] = _nanmean("centroid_disp_expert_mm")
    r["mean_endplate_normal_err_deg"] = _nanmean("endplate_max_normal_err_deg")
    r["mean_cobb_diff_deg"]           = _nanmean("_diff_deg")
    r["max_cobb_diff_deg"]            = max(
        (v for k, v in r.items()
         if k.endswith("_diff_deg") and isinstance(v, float) and not math.isnan(v)),
        default=float("nan"),
    )
    r["mean_gt_cobb_vs_ctd_deg"]    = _nanmean("_gt_cobb_vs_ctd_deg")
    r["mean_pred_cobb_vs_ctd_deg"]  = _nanmean("_pred_cobb_vs_ctd_deg")
    r["n_unreliable_verts"]         = sum(
        1 for k, v in r.items() if k.endswith("_cobb_reliable") and v is False
    )
    return r


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    case_list = sys.argv[1:] if len(sys.argv) > 1 else get_case_list()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()

    all_results = []

    for case_id in case_list:
        print(f"\n-- {case_id} {'-' * (52 - len(case_id))}")
        row = evaluate_case(case_id)
        if row is None:
            continue
        all_results.append(row)

        def _g(k): return row.get(k, float("nan"))
        print(f"  spine DICE                : {_g('spine_recon_DICE'):.4f}")
        print(f"  mean vert DICE            : {_g('mean_vert_DICE'):.4f}")
        print(f"  mean centroid disp (mask) : {_g('mean_centroid_disp_mask_mm'):.2f} mm")
        print(f"  mean centroid disp (exp.) : {_g('mean_centroid_disp_expert_mm'):.2f} mm")
        print(f"  mean endplate normal err  : {_g('mean_endplate_normal_err_deg'):.2f} deg")
        print(f"  mean Cobb diff (pred-GT)  : {_g('mean_cobb_diff_deg'):.2f} deg")
        print(f"  mean GT Cobb vs ctd tilt  : {_g('mean_gt_cobb_vs_ctd_deg'):.2f} deg")
        print(f"  mean pred Cobb vs ctd     : {_g('mean_pred_cobb_vs_ctd_deg'):.2f} deg")
        print(f"  unreliable verts          : {int(_g('n_unreliable_verts'))}")
        print(f"  expert centroid coverage  : {int(_g('n_expert_centroids'))} vertebrae")

    if not all_results:
        print("\nx No results to save.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n+ Saved -> {OUTPUT_CSV}  ({len(df)} cases, {len(df.columns)} columns)")

    summary_cols = [
        "spine_recon_DICE",
        "mean_vert_DICE",
        "mean_centroid_disp_mask_mm",
        "mean_centroid_disp_expert_mm",
        "mean_endplate_normal_err_deg",
        "mean_cobb_diff_deg",
        "max_cobb_diff_deg",
        "mean_gt_cobb_vs_ctd_deg",
        "mean_pred_cobb_vs_ctd_deg",
        "n_unreliable_verts",
    ]
    print("\n== Dataset Summary " + "=" * 42)
    for col in summary_cols:
        if col in df:
            vals = df[col].dropna()
            if len(vals):
                print(f"  {col:<42}  mean={vals.mean():.3f}  std={vals.std():.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()