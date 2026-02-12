#!/usr/bin/env python3
"""
Heuristic-only pipeline (no TSS). Produces ONLY vertebrae C/T/L (no sacrum):

Flow:
  1) Run TotalSegmentator with best-available task:
       vertebrae -> total -> vertebrae_body
  2) If TS per-vertebra exist: copy them to C/T/L masks (skip S*) as *_pred-vert_<LEVEL>.nii.gz
  3) Else (vertebrae_body fallback):
       - spacing-aware split (erosion -> label -> per-component redilation)
       - min-component FILTER by PHYSICAL volume (mm^3)
       - CTL classification by CROSS-SECTION AREA near centroid
       - per-vertebra POSTERIOR COMPLETION within bone HU clamp [170,1000] & radius cap (5 mm), no overlaps
       - save per-vertebra as *_pred-vert_C#, T#, L#.nii.gz
  4) Build union: <CASE>_predicted_spine_heuristic.nii.gz

Strict I/O:
  - TS raw & union:   wisespine_new/baseline_outputs/reconstructed/<CASE>/
  - Per-vertebra CTL: wisespine_new/baseline_outputs/<CASE>/

Run:
  python wisespine_new/process_reconstruct.py
"""

import os
import re
import sys
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import nibabel as nib

# Optional resampling / morphology
try:
    from nibabel.processing import resample_from_to
    NIB_RESAMPLE_OK = True
except Exception:
    NIB_RESAMPLE_OK = False

try:
    import scipy.ndimage as ndi
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# Progress bars (fallback if tqdm missing)
try:
    from tqdm import tqdm
    def pbar(it, desc): return tqdm(it, desc=desc, leave=False)
except Exception:
    def pbar(it, desc): return it

# ----------------------------
# Config (tune if needed)
# ----------------------------
CASE = os.environ.get("WISESPINE_CASE", "sub-gl017")

BASE_DIR   = Path(__file__).resolve().parent.parent
RECON_ROOT = BASE_DIR / "wisespine_new" / "baseline_outputs" / "reconstructed"
OUTPUT_DIR = RECON_ROOT / CASE                     # TS & union live here

INDIV_DIR  = BASE_DIR / "wisespine_new" / "baseline_outputs" / CASE  # per-vertebra CTL live here

HEUR_UNION_PATH = OUTPUT_DIR / f"{CASE}_predicted_spine_heuristic.nii.gz"

# Data root(s)
DATA_CAND_ROOTS = [
    BASE_DIR.parent / "wisespine"     / "data" / "Verse20" / "dataset-02validation" / "rawdata" / CASE,
    BASE_DIR          / "data"        / "Verse20" / "dataset-02validation" / "rawdata" / CASE,
]

# Posterior completion (heuristic only)
BONE_HU_MIN    = 170.0
BONE_HU_MAX    = 1000.0
GROW_RADIUS_MM = 5.0       # 4–6 mm recommended

# Geometry-only CTL thresholds (mm²)
AREA_C_MAX_MM2 = 380.0
AREA_L_MIN_MM2 = 680.0
MAX_C, MAX_T, MAX_L = 7, 12, 6

# Bodies splitting strength (mm)
ERODE_MM    = 1.5
REDILATE_MM = 2.0

# Min component volume (mm³) after splitting (prevents tiny fragments)
MIN_COMP_MM3 = 800.0

# ----------------------------
# Utilities
# ----------------------------
def find_ct_path() -> Path:
    for root in DATA_CAND_ROOTS:
        for fn in (f"{CASE}_ct.nii.gz", f"{CASE}_CT.nii.gz"):
            p = root / fn
            if p.exists(): return p
    print("❌ CT not found for", CASE)
    for root in DATA_CAND_ROOTS:
        print("   looked in:", root)
    sys.exit(1)

def _to_can(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(img)

def _resample_like(src_img: nib.Nifti1Image, ref_img: nib.Nifti1Image, order: int = 0):
    if NIB_RESAMPLE_OK: return resample_from_to(src_img, ref_img, order=order)
    if src_img.shape != ref_img.shape or not np.allclose(src_img.affine, ref_img.affine):
        raise RuntimeError("Resampling unavailable and spaces differ.")
    return src_img

def _save_bin_like(ref_img: nib.Nifti1Image, mask: np.ndarray, out_path: Path):
    can = _to_can(ref_img)
    out = nib.Nifti1Image(mask.astype(np.uint8), can.affine, can.header)
    out.set_sform(can.affine, code=1); out.set_qform(can.affine, code=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists(): out_path.unlink()
    nib.save(out, out_path)

def _mm_to_vox(mm: float, spacing: tuple[float, float, float]) -> int:
    return max(1, int(round(mm / max(min(spacing), 1e-6))))

def _min_voxels_from_mm3(img: nib.Nifti1Image, mm3: float) -> int:
    dx, dy, dz = _to_can(img).header.get_zooms()[:3]
    vv = float(dx) * float(dy) * float(dz)
    return max(1, int(round(mm3 / max(vv, 1e-6))))

# ----------------------------
# TotalSegmentator (for heuristic)
# ----------------------------
PAT_VERTE = re.compile(r"^vertebrae_((?:c|t|l|s)\d+)(?:\.nii(?:\.gz)?)?$", re.IGNORECASE)

def detect_ts_tasks_to_try():
    try:
        help_text = subprocess.run(["TotalSegmentator", "-h"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, check=False).stdout.lower()
    except Exception:
        return ["vertebrae", "total", "vertebrae_body"]
    tasks = []
    if " vertebrae " in help_text or "'vertebrae'" in help_text: tasks.append("vertebrae")
    if " total "     in help_text or "'total'"     in help_text: tasks.append("total")
    if "vertebrae_body" in help_text: tasks.append("vertebrae_body")
    if not tasks: tasks = ["vertebrae", "total", "vertebrae_body"]
    return tasks

def run_totalsegmentator(ct_path: Path) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks_to_try = detect_ts_tasks_to_try()
    last_err = None
    for task in tasks_to_try:
        print(f"🚀 Running TotalSegmentator (--task {task}) into: {OUTPUT_DIR}")
        cmd = ["TotalSegmentator", "-i", str(ct_path), "-o", str(OUTPUT_DIR), "--task", task, "--ml"]
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ TotalSegmentator completed with task '{task}'")
            return task
        except subprocess.CalledProcessError as e:
            last_err = e
            print(f"❗ Task '{task}' failed. Trying next...")
    if last_err: raise last_err
    raise RuntimeError("TotalSegmentator failed without an error.")

# ----------------------------
# Heuristic pipeline helpers
# ----------------------------
def _collect_per_vertebra_files(seg_dir: Path):
    if not seg_dir.is_dir(): return []
    return sorted([p for p in seg_dir.glob("*.nii*") if PAT_VERTE.match(p.name)])

def copy_ts_per_vertebra_to_ctl() -> int:
    """Copy TS per-vertebra files, saving ONLY C/T/L (skip S*)."""
    seg_dir = OUTPUT_DIR / "segmentations"
    files = _collect_per_vertebra_files(seg_dir)
    if not files: return 0
    written = 0
    for fp in pbar(files, "Copy TS per-vertebra → CTL"):
        m = PAT_VERTE.match(fp.name)
        if not m: continue
        lvl = m.group(1).upper()
        if lvl.startswith("S"):  # skip sacrum
            continue
        img = nib.load(fp)
        data = _to_can(img).get_fdata() > 0
        _save_bin_like(img, data, INDIV_DIR / f"{CASE}_pred-vert_{lvl}.nii.gz")
        written += 1
    return written

def copy_ts_labeled_to_ctl() -> int:
    """Copy from labeled TS volume if present (segmentations.nii.gz + names)."""
    seg_nii = OUTPUT_DIR / "segmentations" / "segmentations.nii.gz"
    seg_map = OUTPUT_DIR / "segmentations" / "segment_names.json"
    if not (seg_nii.exists() and seg_map.exists()): return 0
    try:
        with open(seg_map, "r") as f:
            names = json.load(f)
        mapping = {int(k): str(v) for k, v in names.items()}
    except Exception:
        return 0
    img = nib.load(seg_nii)
    lab = _to_can(img).get_fdata().astype(np.int32)
    written = 0
    for idx, name in pbar(list(mapping.items()), "Extract TS labeled → CTL"):
        m = PAT_VERTE.match(name)
        if not m: continue
        lvl = m.group(1).upper()
        if lvl.startswith("S"): continue
        mask = (lab == idx)
        if not np.any(mask): continue
        _save_bin_like(img, mask, INDIV_DIR / f"{CASE}_pred-vert_{lvl}.nii.gz")
        written += 1
    return written

def split_bodies(ts_body_img: nib.Nifti1Image):
    """Erode → label → per-component redilation; returns (labeled_kept, n_kept)."""
    can = _to_can(ts_body_img)
    data = can.get_fdata() > 0
    if not SCIPY_OK:
        labeled, n = ndi.label(data) if hasattr(ndi, "label") else (data.astype(np.int32), int(data.any()))
        return labeled, n

    dx, dy, dz = can.header.get_zooms()[:3]
    e_iters = _mm_to_vox(ERODE_MM, (dx, dy, dz))
    d_iters = _mm_to_vox(REDILATE_MM, (dx, dy, dz))
    struct = ndi.generate_binary_structure(3, 1)

    eroded = ndi.binary_erosion(data, structure=struct, iterations=e_iters)
    labeled, n = ndi.label(eroded)
    if n == 0: return labeled, 0

    # redilate each component; then drop tiny ones by physical volume
    redil = np.zeros_like(labeled, dtype=np.int32)
    for lab in pbar(range(1, n+1), "Redilate comps"):
        comp = (labeled == lab)
        comp = ndi.binary_dilation(comp, structure=struct, iterations=d_iters)
        redil[comp] = lab

    min_vox = _min_voxels_from_mm3(ts_body_img, MIN_COMP_MM3)
    keep_mask = np.zeros(redil.shape, dtype=bool)
    kept = 0
    for lab in pbar(range(1, n+1), "Filter small comps"):
        comp = (redil == lab)
        if int(comp.sum()) >= min_vox:
            keep_mask |= comp
            kept += 1
    if kept == 0: return redil, 0
    labeled_kept, n_kept = ndi.label(keep_mask)
    return labeled_kept, n_kept

def compute_cs_areas(labeled: np.ndarray, ts_img: nib.Nifti1Image):
    """Return list of (lab, z_mean, area_mm2) ordered cranio→caudal."""
    can = _to_can(ts_img)
    dx, dy, dz = can.header.get_zooms()[:3]
    pix2mm2 = float(dx) * float(dy)
    comps = []
    labels = np.unique(labeled); labels = labels[labels > 0]
    for lab in pbar(labels, "Compute CS areas"):
        coords = np.argwhere(labeled == lab)
        if coords.size == 0: continue
        z_mean = coords[:, 2].mean()
        k = int(round(z_mean))
        k0 = max(0, k-1); k1 = min(labeled.shape[2]-1, k); k2 = min(labeled.shape[2]-1, k+1)
        area_vox = 0; cnt = 0
        for kk in {k0, k1, k2}:
            area_vox += int(np.count_nonzero(labeled[:, :, kk] == lab)); cnt += 1
        area_mm2 = (area_vox / max(cnt,1)) * pix2mm2
        comps.append((int(lab), float(z_mean), float(area_mm2)))
    comps.sort(key=lambda x: x[1])
    return comps

def assign_ctl(areas_mm2: list[float]) -> tuple[int,int,int]:
    """Return (c_count, t_count, l_count) from area sequence."""
    n = len(areas_mm2)
    if n == 0: return 0,0,0
    c = 0
    for a in areas_mm2:
        if a <= AREA_C_MAX_MM2: c += 1
        else: break
    c = min(c, MAX_C)
    l = 0
    for a in reversed(areas_mm2):
        if a >= AREA_L_MIN_MM2: l += 1
        else: break
    l = min(l, MAX_L)
    if c + l > n:
        overflow = c + l - n
        reduce_l = min(l, overflow); l -= reduce_l; overflow -= reduce_l
        if overflow > 0: c -= min(c, overflow)
    t = max(0, n - c - l); t = min(t, MAX_T)
    if c + t + l > n:
        overflow = c + t + l - n
        reduce_l = min(l, overflow); l -= reduce_l; overflow -= reduce_l
        if overflow > 0: c -= min(c, overflow)
        t = n - c - l
    return c, t, l

def posterior_completion(ct_like_ts: nib.Nifti1Image, seeds_list: list[np.ndarray]) -> list[np.ndarray]:
    """Bone-clamped geodesic growth per component, radius-capped; no overlaps."""
    if not SCIPY_OK: return [s.copy() for s in seeds_list]
    can = _to_can(ct_like_ts)
    ct = np.nan_to_num(can.get_fdata(), nan=-1000.0)
    bone = (ct >= BONE_HU_MIN) & (ct <= BONE_HU_MAX)
    dx, dy, dz = can.header.get_zooms()[:3]
    spacing = (float(dx), float(dy), float(dz))
    assigned = np.zeros_like(bone, dtype=bool)
    grown_list = []
    for seed in pbar(seeds_list, "Posterior completion"):
        if not np.any(seed): grown_list.append(seed); continue
        dist = ndi.distance_transform_edt(~seed, sampling=spacing)
        within = dist <= float(GROW_RADIUS_MM)
        allowed = bone & within & (~assigned)
        grown = ndi.binary_propagation(seed, mask=allowed)
        assigned |= grown | seed
        grown_list.append(grown | seed)
    return grown_list

# ----------------------------
# Main heuristic pipeline
# ----------------------------
def main():
    ct_path = find_ct_path()
    print(f"📄 Using CT: {ct_path}")

    used_task = run_totalsegmentator(ct_path)
    INDIV_DIR.mkdir(parents=True, exist_ok=True)

    # 1) If TS per-vertebra exist, copy ONLY C/T/L
    wrote = 0
    if used_task in ("vertebrae", "total"):
        w = copy_ts_per_vertebra_to_ctl()
        if w == 0:
            w = copy_ts_labeled_to_ctl()
        wrote += w

    # 2) Bodies-only fallback → split → filter → CTL → completion → save C/T/L
    if wrote == 0:
        body_path = OUTPUT_DIR / "vertebrae_body.nii.gz"
        if not body_path.exists():
            print("❌ No vertebrae output found (vertebrae_body/vertebrae).")
            sys.exit(1)
        ts_body_img = nib.load(body_path)

        labeled_kept, n_kept = split_bodies(ts_body_img)
        if n_kept == 0:
            print("⚠️ vertebrae_body produced 0 kept components after filtering.")
            sys.exit(0)

        comps = compute_cs_areas(labeled_kept, ts_body_img)
        areas = [a for (_, _, a) in comps]
        c_count, t_count, l_count = assign_ctl(areas)

        ordered_labels = [lab for (lab, _, _) in comps]
        seeds = [(labeled_kept == lab) for lab in ordered_labels]

        # intensities for completion
        ct_img = nib.load(ct_path)
        try:
            ct_like_ts = _resample_like(ct_img, ts_body_img, order=1)  # linear for HU
        except RuntimeError:
            ct_like_ts = ts_body_img
            print("⚠️ Resampling CT->TS unavailable; skipping intensity-based completion.")

        seeds = posterior_completion(ct_like_ts, seeds)

        # save C/T/L masks
        idx = 0
        for i in range(c_count):
            _save_bin_like(ts_body_img, seeds[idx], INDIV_DIR / f"{CASE}_pred-vert_C{i+1}.nii.gz"); idx += 1
        for j in range(t_count):
            _save_bin_like(ts_body_img, seeds[idx], INDIV_DIR / f"{CASE}_pred-vert_T{j+1}.nii.gz"); idx += 1
        for k in range(l_count):
            _save_bin_like(ts_body_img, seeds[idx], INDIV_DIR / f"{CASE}_pred-vert_L{k+1}.nii.gz"); idx += 1

        print(f"✅ Heuristic CTL: C={c_count}, T={t_count}, L={l_count} (components_kept={n_kept})")

    # 3) Build union from saved C/T/L masks
    ctl_files = sorted(INDIV_DIR.glob(f"{CASE}_pred-vert_[CTL]*.nii*"))
    if ctl_files:
        ref = _to_can(nib.load(ctl_files[0]))
        union = np.zeros(ref.shape, dtype=bool)
        for fp in pbar(ctl_files, "Build union"):
            img = _to_can(nib.load(fp))
            dat = img.get_fdata() > 0
            if img.shape != ref.shape or not np.allclose(img.affine, ref.affine):
                img = _resample_like(img, ref, order=0); dat = img.get_fdata() > 0
            union |= dat
        _save_bin_like(ref, union, HEUR_UNION_PATH)
        print(f"✅ Saved union: {HEUR_UNION_PATH}")
    else:
        print("⚠️ No CTL masks found; union not generated.")

if __name__ == "__main__":
    main()