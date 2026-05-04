#!/usr/bin/env python3
"""
Cobb Angle Pipeline (DICOM SEG)
================================
Calculates Cobb angles from DICOM segmentation files using two methods:

  Method 1 - 3D Plane Fitting (SVD):
      Fits planes to the superior/inferior endplates of each vertebra
      and computes the angle between the superior plate of the top vertebra
      and the inferior plate of the bottom vertebra.

  Method 2 - 2D Bounding Box (Coronal Projection):
      Projects each vertebra onto the coronal plane (X-Z), fits a bounding
      box, and uses the long axis as vertebral tilt. Recommended as the
      primary visual-verification method (team meeting 2026-04-03).

Outputs (per case):
  debug/<case>_validation.png  - 3D + 2D validation figure
  debug/<case>_points_planes.txt - raw coords + plane params for review
  <case>.json                  - angle results + review flag

Authors : Benjamin Han, Mary (guidance: Vikash)
Repo    : https://github.com/bhan830/wisespine
"""

import json
import numpy as np
import pydicom
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
DICOM_BASE_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/data/DICOM-Dataset")
OUTPUT_DIR     = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/cobb_angles_dicom")
DEBUG_DIR      = OUTPUT_DIR / "debug"

TOP_VERTEBRA    = "T10"
BOTTOM_VERTEBRA = "L5"

# Palette
VERT_COLORS = {"T10": "#e63946", "L5": "#457b9d"}
SUP_COLOR   = "#f4a261"   # orange — superior endplate
INF_COLOR   = "#2a9d8f"   # teal   — inferior endplate
BBOX_COLOR  = "#adb5bd"   # grey   — bounding box wireframe


# ──────────────────────────────────────────────────────────────
# CASE DISCOVERY
# ──────────────────────────────────────────────────────────────
def find_cases():
    seg_dir = DICOM_BASE_DIR / "DICOM-SEG"
    cases = []
    for f in seg_dir.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if getattr(ds, "Modality", "") == "SEG":
                cases.append(f)
        except Exception:
            continue
    print(f"[INFO] Found {len(cases)} DICOM-SEG file(s)")
    return cases


# ──────────────────────────────────────────────────────────────
# LABEL MATCHING
# ──────────────────────────────────────────────────────────────
def label_matches(label: str, target: str) -> bool:
    return target.upper() in label.upper()


# ──────────────────────────────────────────────────────────────
# POINT CLOUD LOADER
# ──────────────────────────────────────────────────────────────
def load_point_cloud(ds, pixel_array: np.ndarray, target_label: str):
    """
    Extract voxel coordinates for a named vertebra segment.
    Returns (N, 3) float array [row, col, frame_index], or None.
    """
    seg_id = None
    for seg in ds.SegmentSequence:
        if label_matches(seg.SegmentLabel, target_label):
            seg_id = seg.SegmentNumber
            break
    if seg_id is None:
        print(f"  [WARN] Segment '{target_label}' not found")
        return None

    pts_all = []
    for i, frame in enumerate(ds.PerFrameFunctionalGroupsSequence):
        ref = frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber
        if ref != seg_id:
            continue
        mask = pixel_array[i] > 0
        if not np.any(mask):
            continue
        rows, cols = np.where(mask)
        pts_all.append(np.column_stack([rows, cols, np.full(len(rows), i)]))

    if not pts_all:
        print(f"  [WARN] No frames found for '{target_label}'")
        return None

    pts = np.vstack(pts_all).astype(float)
    if len(pts) > 5000:
        pts = pts[np.random.choice(len(pts), 5000, replace=False)]
    return pts


# ──────────────────────────────────────────────────────────────
# SPINE NORMALIZATION  — spine long axis → Z
# ──────────────────────────────────────────────────────────────
def normalize_spine(coords_dict: dict) -> dict:
    """
    1. Centre all clouds at the global centroid.
    2. PCA: assign the highest-variance axis (sup-inf) to Z,
            the next to X (left-right), last to Y (ant-post).
    3. Flip Z so that superior is +Z.

    Returns a new dict {label: (N,3) ndarray} in the normalized frame.
    """
    all_pts = np.vstack(list(coords_dict.values()))
    centre  = all_pts.mean(axis=0)
    centred = {k: v - centre for k, v in coords_dict.items()}

    pca = PCA(n_components=3)
    pca.fit(np.vstack(list(centred.values())))
    # components_[0] = max-variance = sup-inf axis
    # Reorder so that axis mapping is: PC0->Z, PC1->X, PC2->Y
    R = pca.components_[[2, 1, 0], :]   # (3,3) row-permuted rotation

    rotated = {k: (v @ R.T) for k, v in centred.items()}

    # Ensure superior is at +Z: the top vertebra should have higher Z mean
    z_top = rotated[TOP_VERTEBRA][:, 2].mean()
    z_bot = rotated[BOTTOM_VERTEBRA][:, 2].mean()
    if z_top < z_bot:
        rotated = {k: v * np.array([1, 1, -1]) for k, v in rotated.items()}

    return rotated


# ──────────────────────────────────────────────────────────────
# ENDPLATE EXTRACTION  (uses Z after normalization)
# ──────────────────────────────────────────────────────────────
def extract_endplates(points: np.ndarray, pct: float = 10.0):
    """
    Superior endplate = top-Z percentile (highest Z = top of vertebra).
    Inferior endplate = bottom-Z percentile.

    Returns (superior_pts, inferior_pts).
    """
    z   = points[:, 2]
    sup = points[z >= np.percentile(z, 100 - pct)]
    inf = points[z <= np.percentile(z, pct)]
    return sup, inf


# ──────────────────────────────────────────────────────────────
# PLANE FITTING (SVD)
# ──────────────────────────────────────────────────────────────
def fit_plane(points: np.ndarray):
    """Fit best-fit plane via SVD. Returns (unit_normal, centroid)."""
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    return vh[2], centroid


# ──────────────────────────────────────────────────────────────
# ANGLE BETWEEN TWO VECTORS
# ──────────────────────────────────────────────────────────────
def vec_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))))


# ──────────────────────────────────────────────────────────────
# COBB  —  METHOD 1: 3D PLANE NORMALS
# ──────────────────────────────────────────────────────────────
def cobb_3d(planes: dict, top: str, bottom: str) -> float:
    """Angle between superior endplate of TOP and inferior endplate of BOTTOM."""
    n1, _ = planes[top]["superior"]
    n2, _ = planes[bottom]["inferior"]
    if np.dot(n1, n2) < 0:
        n2 = -n2
    a = vec_angle(n1, n2)
    return 180.0 - a if a > 90 else a


# ──────────────────────────────────────────────────────────────
# COBB  —  METHOD 2: 2D BOUNDING BOX (coronal X-Z)
# ──────────────────────────────────────────────────────────────
def fit_bbox_long_axis(pts_2d: np.ndarray) -> np.ndarray:
    pca     = PCA(n_components=2).fit(pts_2d)
    proj    = pts_2d @ pca.components_.T
    extents = proj.max(axis=0) - proj.min(axis=0)
    return pca.components_[int(np.argmax(extents))]


def cobb_2d_bbox(coords: dict, top: str, bottom: str) -> float:
    """Coronal plane after normalization = X (left-right) vs Z (sup-inf)."""
    def coronal(pts):
        return pts[:, [0, 2]]

    ax_top    = fit_bbox_long_axis(coronal(coords[top]))
    ax_bottom = fit_bbox_long_axis(coronal(coords[bottom]))
    return vec_angle(ax_top, ax_bottom)


# ──────────────────────────────────────────────────────────────
# TEXT EXPORT
# ──────────────────────────────────────────────────────────────
def export_txt(coords: dict, planes: dict, case_id: str):
    out = DEBUG_DIR / f"{case_id}_points_planes.txt"
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for v, pts in coords.items():
            f.write(f"# VERTEBRA: {v}  ({len(pts)} points, normalized frame)\n")
            np.savetxt(f, pts, fmt="%.4f", delimiter=",",
                       header="X,Y,Z", comments="")
            f.write("\n")
            ns, cs = planes[v]["superior"]
            ni, ci = planes[v]["inferior"]
            f.write(f"# SUPERIOR PLANE  normal={ns.round(5).tolist()}  centroid={cs.round(3).tolist()}\n")
            f.write(f"# INFERIOR PLANE  normal={ni.round(5).tolist()}  centroid={ci.round(3).tolist()}\n\n")
    print(f"  [TXT] {out}")


# ──────────────────────────────────────────────────────────────
# VIZ HELPERS
# ──────────────────────────────────────────────────────────────

def _bbox_wireframe(ax, pts, color, lw=0.9, alpha=0.65):
    """Draw 3D axis-aligned bounding box wireframe."""
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    x0,y0,z0 = mn;  x1,y1,z1 = mx
    corners = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        ax.plot([corners[i,0],corners[j,0]],
                [corners[i,1],corners[j,1]],
                [corners[i,2],corners[j,2]],
                color=color, lw=lw, alpha=alpha)


def _plane_patch(ax, normal, centroid, pts_for_size, color, alpha=0.38):
    """
    Draw a filled rectangular plane patch in 3D, sized to the endplate cloud.
    Patch lies in the best-fit plane; a normal arrow marks the orientation.
    """
    # Plane size = spread of the endplate points (.ptp removed in NumPy 2)
    size = max(pts_for_size[:, 0].max() - pts_for_size[:, 0].min(),
               pts_for_size[:, 1].max() - pts_for_size[:, 1].min(),
               pts_for_size[:, 2].max() - pts_for_size[:, 2].min()) * 0.7
    size = max(size, 4)

    # Two orthogonal in-plane unit vectors
    ref = np.array([0, 0, 1]) if abs(normal[2]) < 0.9 else np.array([1, 0, 0])
    u   = np.cross(normal, ref);  u /= (np.linalg.norm(u) + 1e-8)
    v   = np.cross(normal, u);    v /= (np.linalg.norm(v) + 1e-8)

    h = size / 2
    corners = np.array([
        centroid + h*u + h*v,
        centroid - h*u + h*v,
        centroid - h*u - h*v,
        centroid + h*u - h*v,
    ])
    poly = Poly3DCollection([corners], alpha=alpha,
                            facecolor=color, edgecolor=color, linewidth=0.8)
    ax.add_collection3d(poly)

    # Normal vector arrow (half-size for clarity)
    ax.quiver(*centroid, *(normal * h * 0.55),
              color=color, linewidth=1.3, arrow_length_ratio=0.25)


# ──────────────────────────────────────────────────────────────
# MAIN VISUALIZATION
# ──────────────────────────────────────────────────────────────
def plot_results(coords: dict, planes: dict,
                 a3d: float, a2d: float, case_id: str):
    """
    Two-panel validation figure.

    LEFT (3D)
      - Spine oriented so Z = superior-inferior axis (superior = +Z).
      - Per-vertebra bounding box wireframe.
      - Superior endplate points (orange, top of each vertebra) and
        inferior endplate points (teal, bottom of each vertebra).
      - Fitted plane + normal arrow for each endplate.

    RIGHT (2D coronal, X-Z plane)
      - Bounding rectangles per vertebra.
      - Superior (orange) / inferior (teal) endplate point bands.
      - Bounding-box long-axis arrow = tilt used for 2D Cobb.
    """
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(
        f"Cobb Angle Validation  —  {case_id}\n"
        f"3D plane method: {a3d:.1f}deg     2D bbox method: {a2d:.1f}deg",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── LEFT: 3D ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(121, projection="3d")
    ax3.set_title("3D View  (Z = superior-inferior axis)", fontsize=11, pad=10)

    for v, pts in coords.items():
        vcol = VERT_COLORS.get(v, "#888")
        sup, inf = extract_endplates(pts)

        # 1. Full vertebra body (very faint)
        ax3.scatter(pts[:,0], pts[:,1], pts[:,2],
                    s=1, alpha=0.12, color=vcol)

        # 2. Bounding box wireframe around whole vertebra
        _bbox_wireframe(ax3, pts, color=vcol, lw=1.0, alpha=0.55)

        # 3. Endplate point clouds:
        #    superior = top of vertebra (high Z) → orange
        #    inferior = bottom of vertebra (low Z) → teal
        ax3.scatter(sup[:,0], sup[:,1], sup[:,2],
                    s=10, color=SUP_COLOR, alpha=0.95, zorder=5)
        ax3.scatter(inf[:,0], inf[:,1], inf[:,2],
                    s=10, color=INF_COLOR, alpha=0.95, zorder=5)

        # 4. Fitted planes
        n_s, c_s = planes[v]["superior"]
        n_i, c_i = planes[v]["inferior"]
        _plane_patch(ax3, n_s, c_s, sup, color=SUP_COLOR, alpha=0.35)
        _plane_patch(ax3, n_i, c_i, inf, color=INF_COLOR, alpha=0.35)

        # 5. Label at vertebra centroid
        cen = pts.mean(axis=0)
        ax3.text(cen[0], cen[1], cen[2], f"  {v}",
                 color=vcol, fontsize=9, fontweight="bold")

    ax3.set_xlabel("X (left-right)")
    ax3.set_ylabel("Y (ant-post)")
    ax3.set_zlabel("Z (inferior → superior)")
    ax3.view_init(elev=15, azim=75)

    legend_elems = [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=SUP_COLOR, markersize=9,
               label="Superior endplate points"),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=INF_COLOR, markersize=9,
               label="Inferior endplate points"),
        patches.Patch(facecolor=SUP_COLOR, alpha=0.45,
                      label="Superior plane (SVD fit)"),
        patches.Patch(facecolor=INF_COLOR, alpha=0.45,
                      label="Inferior plane (SVD fit)"),
        Line2D([0],[0], color=BBOX_COLOR, lw=1.2,
               label="Vertebra bounding box"),
    ]
    ax3.legend(handles=legend_elems, loc="lower left", fontsize=8)

    # ── RIGHT: 2D coronal ──────────────────────────────────────────────
    ax2 = fig.add_subplot(122)
    ax2.set_title("2D Coronal View  (X-Z plane)", fontsize=11, pad=10)
    ax2.set_xlabel("X  (left to right)")
    ax2.set_ylabel("Z  (inferior to superior)")

    first = True
    for v, pts in coords.items():
        vcol = VERT_COLORS.get(v, "#888")
        sup, inf = extract_endplates(pts)

        xz_all = pts[:, [0, 2]]
        xz_sup = sup[:, [0, 2]]
        xz_inf = inf[:, [0, 2]]

        # Body cloud
        ax2.scatter(xz_all[:,0], xz_all[:,1], s=1, alpha=0.15, color=vcol)

        # Bounding box
        mn, mx = xz_all.min(axis=0), xz_all.max(axis=0)
        w, h = mx - mn
        ax2.add_patch(patches.Rectangle(
            mn, w, h,
            linewidth=1.8, edgecolor=vcol, facecolor="none", zorder=3))

        # Endplate point bands
        ax2.scatter(xz_sup[:,0], xz_sup[:,1], s=12,
                    color=SUP_COLOR, zorder=4,
                    label="Superior endplate" if first else "")
        ax2.scatter(xz_inf[:,0], xz_inf[:,1], s=12,
                    color=INF_COLOR, zorder=4,
                    label="Inferior endplate" if first else "")
        first = False

        # Long-axis arrow for bounding-box Cobb method
        centre   = xz_all.mean(axis=0)
        long_ax  = fit_bbox_long_axis(xz_all)
        half_len = max(w, h) * 0.55
        ax2.annotate(
            "", xy=centre + long_ax * half_len,
            xytext=centre - long_ax * half_len,
            arrowprops=dict(arrowstyle="<->", color=vcol, lw=2.2),
            zorder=5)

        # Vertebra label above bounding box
        ax2.text(mn[0] + w/2, mx[1] + 0.8, v,
                 ha="center", color=vcol, fontsize=10, fontweight="bold")

    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_aspect("equal")
    ax2.autoscale()

    plt.tight_layout()
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out = DEBUG_DIR / f"{case_id}_validation.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [PNG] {out}")


# ──────────────────────────────────────────────────────────────
# PROCESS ONE CASE
# ──────────────────────────────────────────────────────────────
def process(path: Path):
    case = path.stem
    print(f"\n{'─'*60}\nCase: {case}\n{'─'*60}")

    ds = pydicom.dcmread(path)
    px = ds.pixel_array

    coords_raw = {}
    for v in [TOP_VERTEBRA, BOTTOM_VERTEBRA]:
        pts = load_point_cloud(ds, px, v)
        if pts is not None:
            coords_raw[v] = pts

    if len(coords_raw) < 2:
        print("[WARN] Could not load both vertebrae - skipping")
        return

    # Normalize so spine runs along Z
    coords = normalize_spine(coords_raw)

    # Fit endplate planes (in normalized space)
    planes = {}
    for v, pts in coords.items():
        sup, inf = extract_endplates(pts)
        planes[v] = {
            "superior": fit_plane(sup),
            "inferior": fit_plane(inf),
        }

    a3d = cobb_3d(planes, TOP_VERTEBRA, BOTTOM_VERTEBRA)
    a2d = cobb_2d_bbox(coords, TOP_VERTEBRA, BOTTOM_VERTEBRA)

    print(f"  3D plane-fitting Cobb : {a3d:.2f} deg")
    print(f"  2D bounding-box Cobb  : {a2d:.2f} deg")
    print(f"  Difference            : {abs(a3d - a2d):.2f} deg")
    if a3d > 50 or a2d > 50:
        print("  [FLAG] Angle >50 deg - visually verify")

    export_txt(coords, planes, case)
    plot_results(coords, planes, a3d, a2d, case)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "case": case,
        "top_vertebra": TOP_VERTEBRA,
        "bottom_vertebra": BOTTOM_VERTEBRA,
        "cobb_3d_plane_fitting_deg": round(float(a3d), 3),
        "cobb_2d_bbox_deg": round(float(a2d), 3),
        "difference_deg": round(abs(float(a3d) - float(a2d)), 3),
        "flag_needs_review": bool(a3d > 50 or a2d > 50),
    }
    with open(OUTPUT_DIR / f"{case}.json", "w") as f:
        json.dump(result, f, indent=4)
    print(f"  [JSON] {OUTPUT_DIR / case}.json")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    cases = find_cases()
    if not cases:
        print("[ERROR] No DICOM-SEG files found. Check DICOM_BASE_DIR.")
        return
    for c in tqdm(cases, desc="Processing cases"):
        try:
            process(c)
        except Exception as e:
            print(f"[ERROR] {c.stem}: {e}")
    print("\nDone. Results in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()