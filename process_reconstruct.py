#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spine Reconstruction (single case, multi-model) — minimal reconstructed/ outputs

Goal:
  • Keep 'reconstructed/' clean and minimal.
  • Only create per-model/case directories under 'reconstructed/'.
  • Only write a union mask if explicitly enabled (BUILD_UNION=1).
  • No intermediates (no vertebrae_body, discs, heuristics, raw dumps) are ever written to reconstructed/.

Run:
    python process_reconstruct.py

Environment overrides (optional):
  CASE=sub-gl017                # default: gl017 → normalized to 'sub-gl017'
  BUILD_UNION=1|0               # default 0: don't write unions; evaluator rebuilds union anyway
  REGENERATE_ALWAYS=1|0         # default 0: reuse existing per-level outputs if found
  STRICT_TOTALSEGMENTATOR=1|0   # default 1
  STRICT_OTHER_MODELS=1|0       # default 1
  SEED_FROM_DERIVATIVES=1|0     # default 1 (fallback for non-TS if no external cmd)
  TS_CMD="TotalSegmentator"     # or "python -m totalsegmentator", etc.
  TOTALSEG_EXTRA_ARGS="--gpu 0"
  EXTERNAL_MODEL_CMDS='{"NNUnet Framework":"nnUNetv2_predict -i {ct} -o {outdir} ..."}'
     # Or: "NNUnet Framework=...;TotalSpineSeg=...;SamMed3D=..."

Outputs:
  baseline_outputs/models/<MODEL>/<CASE>/<CASE>_<MODEL>_pred-vert_<LEVEL>.nii.gz
  baseline_outputs/reconstructed/<MODEL>/<CASE>/<CASE>_<MODEL>_predicted_spine_union.nii.gz   (only if BUILD_UNION=1)
  baseline_outputs/labels/<CASE>/<MODEL>_labels.csv
  baseline_outputs/labels/<CASE>/labels_index.csv
"""

from __future__ import annotations
import os, re, sys, shlex, json, subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib

# Optional libs
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

# Progress bar
try:
    from tqdm import tqdm
    def pbar(it, desc): return tqdm(it, desc=desc, leave=False)
except Exception:
    def pbar(it, desc): return it

# --------------------------------
# Config
# --------------------------------
def _norm_case(c: str) -> str:
    c = (c or "").strip()
    return c if c.startswith("sub-") else f"sub-{c}"

def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1","true","yes","y"): return True
    if v in ("0","false","no","n"): return False
    return default

CASE = _norm_case(os.environ.get("CASE", "gl017"))

MODELS_TO_RUN = ["TotalSegmentator", "NNUnet Framework", "TotalSpineSeg", "SamMed3D"]
DISPLAY2KEY = {m: m for m in MODELS_TO_RUN}

STRICT_MODELS = {
    "TotalSegmentator": env_bool("STRICT_TOTALSEGMENTATOR", True),
    "NNUnet Framework": env_bool("STRICT_OTHER_MODELS", True),
    "TotalSpineSeg":    env_bool("STRICT_OTHER_MODELS", True),
    "SamMed3D":         env_bool("STRICT_OTHER_MODELS", True),
}

# Critical: keep reconstructed minimal
BUILD_UNION       = env_bool("BUILD_UNION", False)   # default OFF → evaluator rebuilds
REGENERATE_ALWAYS = env_bool("REGENERATE_ALWAYS", False)
SEED_FROM_DERIV   = env_bool("SEED_FROM_DERIVATIVES", True)

POSTPROC_CLOSE_ITERS = 1
LEVELS = [*(f"C{i}" for i in range(1, 8)),
          *(f"T{i}" for i in range(1, 13)),
          *(f"L{i}" for i in range(1, 6))]

# Paths
WISE_DIR     = Path(__file__).resolve().parent
BASE_DIR     = WISE_DIR / "baseline_outputs"
RECON_ROOT   = BASE_DIR / "reconstructed"
MODELS_ROOT  = BASE_DIR / "models"
LABELS_CASE_ROOT  = BASE_DIR / "labels" / CASE

DATA_CAND_ROOTS = [
    WISE_DIR.parent / "data" / "Verse20" / "dataset-02validation",
    WISE_DIR.parent.parent / "wisespine" / "data" / "Verse20" / "dataset-02validation",
]

PAT_LEVEL = re.compile(r"^(?:.*_)?(?:vertebrae_)?((?:c|t|l)\d+)(?:\.nii(?:\.gz)?)?$", re.IGNORECASE)

# --------------------------------
# I/O helpers
# --------------------------------
def find_ct_path() -> Path:
    for base in DATA_CAND_ROOTS:
        raw = base / "rawdata" / CASE
        for fn in (f"{CASE}_ct.nii.gz", f"{CASE}_CT.nii.gz", "ct.nii.gz", f"{CASE}_ct.nii", f"{CASE}_CT.nii"):
            p = raw / fn
            if p.exists(): return p
    print("❌ CT not found for", CASE)
    for base in DATA_CAND_ROOTS:
        print("   looked in:", base / "rawdata" / CASE)
    sys.exit(1)

def find_deriv_seg_path() -> Optional[Path]:
    for base in DATA_CAND_ROOTS:
        p = base / "derivatives" / CASE / f"{CASE}_seg-vert_msk.nii.gz"
        if p.exists(): return p
    return None

def _to_can(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(img)

def _resample_like(src_img: nib.Nifti1Image, ref_img: nib.Nifti1Image, order: int = 0) -> nib.Nifti1Image:
    if NIB_RESAMPLE_OK: return resample_from_to(src_img, ref_img, order=order)
    if src_img.shape != ref_img.shape or not np.allclose(src_img.affine, ref_img.affine):
        raise RuntimeError("Resampling unavailable and spaces differ.")
    return src_img

def _extract_level_from_name(name: str) -> Optional[str]:
    m = PAT_LEVEL.match(name)
    if not m: return None
    lvl = m.group(1).upper()
    return lvl if lvl[0] in ("C","T","L") else None

def _make_level_filename(model_key: str, level: str) -> str:
    return f"{CASE}_{model_key}_pred-vert_{level}.nii.gz"

def _save_bin_like(ref_img: nib.Nifti1Image, mask: np.ndarray, out_path: Path):
    can = _to_can(ref_img)
    out = nib.Nifti1Image(mask.astype(np.uint8), can.affine, can.header)
    out.set_sform(can.affine, code=1); out.set_qform(can.affine, code=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists(): out_path.unlink()
    nib.save(out, out_path)

def _largest_component(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask) or not SCIPY_OK: return mask.astype(bool)
    lab, n = ndi.label(mask.astype(bool))
    if n <= 1: return mask.astype(bool)
    sizes = ndi.sum(mask, lab, index=range(1, n+1))
    return lab == (1 + int(np.argmax(sizes)))

def _morph_cleanup(mask: np.ndarray, close_iters: int = 1) -> np.ndarray:
    if not np.any(mask) or not SCIPY_OK: return mask.astype(bool)
    struct = ndi.generate_binary_structure(3, 1)
    filled = ndi.binary_fill_holes(mask)
    closed = ndi.binary_closing(filled, structure=struct, iterations=close_iters)
    return closed

def refine_and_save_level(src_img: nib.Nifti1Image, mask: np.ndarray, out_fp: Path):
    m = _largest_component(mask)
    m = _morph_cleanup(m, POSTPROC_CLOSE_ITERS)
    _save_bin_like(src_img, m, out_fp)

# --------------------------------
# Harvesting from model outputs
# --------------------------------
def _candidate_dirs(model_dir: Path) -> List[Path]:
    return [d for d in [model_dir, model_dir/"segmentations", model_dir/CASE, model_dir/"segmentations"/CASE] if d.exists()]

def _likely_vertebra_mask_name(name: str) -> bool:
    n = name.lower()
    if "pred-vert_" in n: return False
    if "vertebrae_body" in n or "vertebrae-body" in n: return False
    if "segmentations" in n or "segmentation" in n or "labels" in n: return False
    return bool(re.search(r"(?:^|[_\-\.])([ctl][0-9]{1,2})(?:[_\-\.]|\.nii)", n))

def harvest_from_perfile(model_dir: Path, model_key: str) -> int:
    written, used = 0, set()
    for d in _candidate_dirs(model_dir):
        per_files = list(d.glob("vertebrae_[cCtTlL][0-9]*.nii*"))
        per_files += [p for p in d.glob("*.nii*") if _likely_vertebra_mask_name(p.name)]
        for fp in sorted(set(per_files)):
            lvl = _extract_level_from_name(fp.name)
            if not lvl or lvl in used: continue
            try:
                img = _to_can(nib.load(fp)); m = img.get_fdata() > 0
                out_fp = model_dir / _make_level_filename(model_key, lvl)
                if out_fp.exists() and not REGENERATE_ALWAYS:
                    written += 1; used.add(lvl); continue
                refine_and_save_level(img, m, out_fp)
                written += 1; used.add(lvl)
            except Exception:
                continue
    return written

def _labeled_candidates(model_dir: Path) -> List[Path]:
    cands: List[Path] = []
    for nm in ("segmentations.nii.gz","segmentations.nii","segmentation.nii.gz","segmentation.nii",
               "ct_seg.nii.gz","ct_seg.nii","labels.nii.gz","labels.nii"):
        for d in _candidate_dirs(model_dir):
            p = d / nm
            if p.exists(): cands.append(p)
    return cands

def harvest_from_labeled(seg_path: Path, model_dir: Path, model_key: str) -> int:
    """Map labeled volumes head→tail to C1..L5 and save standardized per-level."""
    try:
        img = _to_can(nib.load(seg_path))
    except Exception:
        return 0
    lab = img.get_fdata().astype(np.int32)
    labels = [int(x) for x in np.unique(lab) if int(x) > 0]
    comps=[]
    for li in labels:
        cc=np.argwhere(lab==li)
        if cc.size==0: continue
        comps.append((li, float(cc[:,2].mean())))
    comps.sort(key=lambda t:t[1])
    wrote=0
    for idx,(li,_) in enumerate(comps):
        if idx>=len(LEVELS): break
        lvl=LEVELS[idx]; mask=(lab==li)
        if not np.any(mask): continue
        out_fp = model_dir/_make_level_filename(model_key, lvl)
        if out_fp.exists() and not REGENERATE_ALWAYS: wrote+=1; continue
        refine_and_save_level(img, mask, out_fp); wrote+=1
    return wrote

# --------------------------------
# TotalSegmentator
# --------------------------------
def _ts_cli_candidates() -> List[str]:
    c = os.environ.get("TS_CMD", "").strip()
    cands = [c] if c else []
    cands += ["TotalSegmentator", "totalsegmentator", "python -m totalsegmentator", f"{shlex.quote(sys.executable)} -m totalsegmentator"]
    # unique preserve order
    out=[]; seen=set()
    for s in cands:
        if s and s not in seen: out.append(s); seen.add(s)
    return out

def _run_cmd(cmd_str: str, args: List[str]) -> Tuple[bool, str, str, int]:
    argv = shlex.split(cmd_str) + args
    try:
        cp = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return (cp.returncode==0, cp.stdout, cp.stderr, cp.returncode)
    except FileNotFoundError:
        return (False, "", f"Not found: {argv[0]}", 127)
    except Exception as e:
        return (False, "", str(e), 1)

def run_totalsegmentator(ct_path: Path, model_dir: Path, model_key: str):
    # Reuse if existing per-levels present and not forcing regenerate
    existing = list(model_dir.glob(f"{CASE}_{model_key}_pred-vert_[CTL]*.nii*"))
    if existing and not REGENERATE_ALWAYS:
        print(f"ℹ️ Reusing {len(existing)} TS per-level files.")
        return

    # Try TS '--task vertebrae_body' and harvest per-file outputs
    extra = os.environ.get("TOTALSEG_EXTRA_ARGS", "").strip()
    ran=False; errors=[]
    for cmd in _ts_cli_candidates():
        for ml in (False, True):
            args=["-i", str(ct_path), "-o", str(model_dir), "--task", "vertebrae_body"]
            if ml: args.append("--ml")
            if extra: args += shlex.split(extra)
            ok, out, err, rc = _run_cmd(cmd, args)
            print(f"▶️ TS: {cmd} {' '.join(args)}  rc={rc}")
            if ok:
                ran=True
                break
            else:
                errors.append(f"[{cmd}] ml={ml} rc={rc}\n{(err or out).strip()}")
        if ran: break

    if not ran:
        msg = "[TotalSegmentator] could not run TS CLI.\n" + "\n---\n".join(errors)
        if STRICT_MODELS["TotalSegmentator"]: raise RuntimeError(msg)
        print("⚠️ " + msg)

    wrote = harvest_from_perfile(model_dir, model_key)
    if wrote == 0:
        for seg in _labeled_candidates(model_dir):
            wrote = harvest_from_labeled(seg, model_dir, model_key)
            if wrote>0: break

    # Log result
    files = list(model_dir.glob(f"{CASE}_{model_key}_pred-vert_[CTL]*.nii*"))
    print(f"   TS per-level (standardized): {len(files)} files")

# --------------------------------
# Other models (external or seeding)
# --------------------------------
def _run_external_cmd(cmd_template: str, ct_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = cmd_template.format(ct=str(ct_path), outdir=str(out_dir))
    print(f"🚀 External model: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def process_seg_as_source(seg_path: Path, model_dir: Path, model_key: str) -> int:
    if not seg_path or not seg_path.exists(): return 0
    if REGENERATE_ALWAYS:
        for fp in model_dir.glob(f"{CASE}_{model_key}_pred-vert_[CTL]*.nii*"):
            try: fp.unlink()
            except Exception: pass
    img=_to_can(nib.load(seg_path)); lab = img.get_fdata().astype(np.int32)
    labels=[int(v) for v in np.unique(lab) if int(v)>0]
    comps=[]
    for li in labels:
        cc=np.argwhere(lab==li)
        if cc.size==0: continue
        comps.append((li, float(cc[:,2].mean())))
    comps.sort(key=lambda t:t[1])
    wrote=0
    for idx,(li,_) in enumerate(comps):
        if idx>=len(LEVELS): break
        lvl=LEVELS[idx]; mask=(lab==li)
        if not np.any(mask): continue
        out_fp = model_dir/_make_level_filename(model_key, lvl)
        if out_fp.exists() and not REGENERATE_ALWAYS: wrote+=1; continue
        refine_and_save_level(img, mask, out_fp); wrote+=1
    return wrote

def run_model_external_or_seed(model_name: str, ct_path: Path, model_dir: Path, model_key: str, deriv_seg: Optional[Path]):
    # optional external inference
    ext_map: Dict[str,str] = {}
    ext_env = os.environ.get("EXTERNAL_MODEL_CMDS", "").strip()
    if ext_env:
        try:
            parsed=json.loads(ext_env)
            if isinstance(parsed, dict): ext_map={str(k):str(v) for k,v in parsed.items()}
        except Exception:
            for chunk in ext_env.split(";"):
                if "=" in chunk:
                    k,v=chunk.split("=",1)
                    ext_map[k.strip()]=v.strip()

    if model_name in ext_map and ext_map[model_name]:
        try:
            _run_external_cmd(ext_map[model_name], ct_path, model_dir)
        except subprocess.CalledProcessError as e:
            if STRICT_MODELS.get(model_name, True): raise
            print(f"⚠️ External cmd failed for {model_name}: {e}")

    wrote = harvest_from_perfile(model_dir, model_key)
    if wrote == 0:
        for seg in _labeled_candidates(model_dir):
            wrote = harvest_from_labeled(seg, model_dir, model_key)
            if wrote>0: break
    if wrote == 0 and SEED_FROM_DERIV and deriv_seg and deriv_seg.exists():
        print(f"ℹ️ Seeding [{model_name}] from {deriv_seg}")
        wrote = process_seg_as_source(deriv_seg, model_dir, model_key)

    if wrote == 0 and STRICT_MODELS.get(model_name, True):
        raise RuntimeError(f"[{model_name}] No per-vertebra masks produced.")

# --------------------------------
# Union (optional; minimal reconstructed/)
# --------------------------------
def build_union_from_models_dir(model_dir: Path, out_path: Path):
    """ OR-union across standardized per-level masks. """
    files = sorted(model_dir.glob(f"{CASE}_*_pred-vert_[CTL][0-9]*.nii*"))
    if not files:
        print(f"⚠️ No per-level files in {model_dir}; skip union.")
        return False
    ref = _to_can(nib.load(files[0]))
    union = np.zeros(ref.shape, dtype=bool)
    for fp in pbar(files, "Union"):
        img = _to_can(nib.load(fp)); dat = img.get_fdata()>0
        if img.shape!=ref.shape or not np.allclose(img.affine, ref.affine):
            img=_resample_like(img, ref, order=0); dat=img.get_fdata()>0
        union |= dat
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_bin_like(ref, union, out_path)
    print(f"✅ Union saved: {out_path}")
    return True

# --------------------------------
# Labels CSV
# --------------------------------
def save_labels_csv(model_disp: str, case: str, indiv_dir: Path, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(indiv_dir.glob(f"{case}_{model_disp}_pred-vert_[CTL]*.nii*"))
    rows=[]
    for fp in files:
        lvl=_extract_level_from_name(fp.name)
        if not lvl: continue
        rows.append([model_disp, case, lvl, str(fp)])
    with out_csv.open("w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["model","case","level","mask_path"])
        w.writerows(rows)
    print(f"📝 Labels CSV saved: {out_csv}")

def append_labels_index(model_disp: str, case: str, indiv_dir: Path, index_csv: Path):
    files = sorted(indiv_dir.glob(f"{case}_{model_disp}_pred-vert_[CTL]*.nii*"))
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not index_csv.exists()
    with index_csv.open("a", newline="") as f:
        import csv
        w = csv.writer(f)
        if write_header: w.writerow(["model","case","level","mask_path"])
        for fp in files:
            lvl=_extract_level_from_name(fp.name)
            if not lvl: continue
            w.writerow([model_disp, case, lvl, str(fp)])

# --------------------------------
# Main
# --------------------------------
def main():
    ct_path   = find_ct_path()
    deriv_seg = find_deriv_seg_path()
    print(f"📄 CT: {ct_path}")
    if deriv_seg: print(f"📄 Derivative seg (optional): {deriv_seg}")

    LABELS_CASE_ROOT.mkdir(parents=True, exist_ok=True)
    combined_index_csv = LABELS_CASE_ROOT / "labels_index.csv"
    if combined_index_csv.exists(): combined_index_csv.unlink()

    for model_disp in MODELS_TO_RUN:
        model_key = DISPLAY2KEY[model_disp]
        model_dir = MODELS_ROOT / model_key / CASE
        model_dir.mkdir(parents=True, exist_ok=True)

        recon_dir = RECON_ROOT / model_key / CASE
        # Important: only create recon_dir if building union; otherwise leave it absent or empty
        if BUILD_UNION:
            recon_dir.mkdir(parents=True, exist_ok=True)
        union_path = recon_dir / f"{CASE}_{model_key}_predicted_spine_union.nii.gz"

        print("\n==============================")
        print(f"▶️ {model_disp}")
        print(f"   models dir: {model_dir}")
        if BUILD_UNION:
            print(f"   union path: {union_path}")
        print("==============================")

        # Generate/harvest per-vertebra masks into models/<MODEL>/<CASE>/
        if model_key == "TotalSegmentator":
            run_totalsegmentator(ct_path, model_dir, model_key)
        else:
            run_model_external_or_seed(model_disp, ct_path, model_dir, model_key, deriv_seg)

        # Sanity
        files = sorted(model_dir.glob(f"{CASE}_{model_key}_pred-vert_[CTL]*.nii*"))
        if not files:
            raise RuntimeError(f"[{model_disp}] No per-level masks found in {model_dir}")

        # Optional minimal union (only one file)
        if BUILD_UNION:
            build_union_from_models_dir(model_dir, union_path)

        # Labels CSVs
        model_labels_csv = LABELS_CASE_ROOT / f"{model_key}_labels.csv"
        save_labels_csv(model_disp, CASE, model_dir, model_labels_csv)
        append_labels_index(model_disp, CASE, model_dir, combined_index_csv)

    print("\n✅ Done. 'reconstructed/' contains only per-model/case folders and (optionally) one union file each.")
    if not BUILD_UNION:
        print("ℹ️ BUILD_UNION=0 → evaluator will rebuild unions from per-level masks.")

if __name__ == "__main__":
    main()