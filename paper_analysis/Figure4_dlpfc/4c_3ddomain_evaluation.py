"""
DLPFC 3D domain-continuity and neighborhood-connectivity evaluation.

This script compares SpatialZ and AXIS virtual slices by asking whether a spot's
cross-slice 3D neighbors preserve the same spatial-domain/layer label.

Main outputs
------------
1. overall_cross_slice_domain_continuity.csv
2. per_domain_cross_slice_domain_continuity.csv
3. per_spot_cross_slice_domain_continuity.csv          optional, controlled by SAVE_PER_SPOT
4. neighborhood_connectivity_long.csv                  source-domain -> neighbor-domain table
5. neighborhood_connectivity_matrix_fraction.csv       row-normalized matrix form
6. combined_3d_domain_continuity.png
7. neighborhood_connectivity_heatmap.png

Interpretation of neighborhood_connectivity_heatmap.png
-------------------------------------------------------
Rows are the source spot's domain/layer. Columns are the labels of selected
cross-slice 3D neighbors. Values are row-normalized fractions. By default, each
source spot uses the top-k nearest neighbors within the physical radius in each
eligible target slice; set N_NEIGHBORS_PER_TARGET_SLICE=None to use all
radius-neighbors. A clean 3D domain-continuity pattern should concentrate mass
near the diagonal.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree


# ============================================================
# CONFIG
# ============================================================
ROOT = "/media/huifang/data/registration/DLPFC/huifang"
OUTDIR = os.path.join(ROOT, "virtual_tissue", "evaluation_domain")
os.makedirs(OUTDIR, exist_ok=True)

COORD_KEY = "spatial"
DOMAIN_KEY = "layer_guess_reordered"
# DOMAIN_KEY = "axis_joint_domain"

REFERENCE_REAL_Z = 10
VISIUM_NEAREST_SPOT_DISTANCE_UM = 100.0

# Fair comparison: use the common subset available for both methods.
# If both methods are available every 10 µm, replace with np.arange(20, 310, 10).
VIRTUAL_DEPTHS = np.arange(40, 310, 30)
Z_ORDER = [10] + VIRTUAL_DEPTHS.tolist() + [310]

REAL_SLICE_PATHS = {
    10: os.path.join(ROOT, "h5ad", "1_1.h5ad"),
    310: os.path.join(ROOT, "h5ad", "1_2.h5ad"),
}

SPATIALZ_SLICE_PATHS = {
    depth: os.path.join(
        ROOT,
        "virtual_tissue",
        "spatialz",
        "dense_reconstruction",
        "group1",
        f"spatialz_{depth}.000.h5ad",
    )
    for depth in VIRTUAL_DEPTHS
}

AXIS_SLICE_PATHS = {
    depth: os.path.join(
        ROOT,
        "virtual_tissue",
        "dlpfc_z_group1_grid80_spatial1",
        "query_full_gene_set",
        f"virtual_dlpfc_z_{depth}.000.h5ad",
    )
    for depth in VIRTUAL_DEPTHS
}

METHOD_ORDER = ["SpatialZ", "AXIS"]
METHOD_PALETTE = {
    "SpatialZ": "#7C8DA6",
    "AXIS": "#C4473A",
}

# Score only virtual slices by default. Set True to include shared real anchors.
USE_REAL_ANCHORS_IN_SCORE = False
Z_ORDER_SCORE = Z_ORDER if USE_REAL_ANCHORS_IN_SCORE else VIRTUAL_DEPTHS.tolist()

# Cross-slice-only scoring is usually more meaningful for 3D continuity.
INCLUDE_SAME_SLICE_NEIGHBORS = False

# Number of neighbors considered for each source spot in each eligible target slice.
# - Set to an integer, e.g. 6, to use the k nearest neighbors within the physical
#   radius in each adjacent/eligible slice.
# - Set to None to use all radius-neighbors, matching the original radius-only
#   implementation.
N_NEIGHBORS_PER_TARGET_SLICE = None

# Nearby-slice rule. max_z_diff_um = MAX_Z_DIFF_FACTOR * median z step.
MAX_Z_DIFF_FACTOR = 2.

# Real physical 3D radius.
# radius_um = RADIUS_FACTOR * sqrt((100 µm)^2 + (median dz)^2)
RADIUS_FACTOR = 2.

MIN_POINTS_PER_DOMAIN = 80
INVALID_DOMAIN_LABELS = {"nan", "NaN", "None", "", "unknown"}

# I/O switches.
RUN_MODE = "plot"  # "compute", "plot", or "both"
SAVE_PER_SPOT = True
FIG_DPI = 300

# Output files.
OVERALL_CSV = os.path.join(OUTDIR, "overall_cross_slice_domain_continuity.csv")
PER_DOMAIN_CSV = os.path.join(OUTDIR, "per_domain_cross_slice_domain_continuity.csv")
PER_SPOT_CSV = os.path.join(OUTDIR, "per_spot_cross_slice_domain_continuity.csv")
CONNECTIVITY_LONG_CSV = os.path.join(OUTDIR, "neighborhood_connectivity_long.csv")
CONNECTIVITY_MATRIX_CSV = os.path.join(OUTDIR, "neighborhood_connectivity_matrix_fraction.csv")
COMBINED_CONTINUITY_FIG = os.path.join(OUTDIR, "combined_3d_domain_continuity.png")
CONNECTIVITY_HEATMAP_FIG = os.path.join(OUTDIR, "neighborhood_connectivity_heatmap.png")
CONNECTIVITY_DIAGNOSIS_FIG = os.path.join(OUTDIR, "neighborhood_connectivity_diagnosis.png")

# Diagnosis figure configuration.
DIAG_SOURCE_DOMAIN = "Layer5"
DIAG_LEAP_DOMAIN = "WM"
DIAG_LOCAL_CONTEXT_FACTOR = 1.75   # local background window size relative to radius
DIAG_MIN_NEIGHBORS = 4
DIAG_VIEW_ELEV = 22
DIAG_VIEW_AZIM = -58


# ============================================================
# BASIC HELPERS
# ============================================================
def domain_sort_key(x: object) -> Tuple[int, object]:
    s = str(x)
    try:
        return (0, int(float(s)))
    except Exception:
        return (1, s)


def sort_domains(domains: Iterable[object]) -> List[str]:
    vals = [str(x) for x in domains if str(x) not in INVALID_DOMAIN_LABELS]
    return sorted(pd.unique(pd.Series(vals)).tolist(), key=domain_sort_key)


def load_slice_dict(path_dict: Dict[int, str]) -> Dict[int, sc.AnnData]:
    out: Dict[int, sc.AnnData] = {}
    for z, path in path_dict.items():
        if path is None or str(path).strip() == "":
            continue
        if not os.path.exists(path):
            print(f"[WARN] missing file for z={z}: {path}")
            continue
        ad = sc.read_h5ad(path)
        if "valid" in ad.obs.columns:
            ad = ad[ad.obs["valid"]].copy()
        out[int(z)] = ad
    return out


def get_coords(adata: sc.AnnData, coord_key: str = COORD_KEY) -> np.ndarray:
    if coord_key not in adata.obsm:
        raise KeyError(f"{coord_key!r} not found in adata.obsm")
    return np.asarray(adata.obsm[coord_key], dtype=np.float32)


def get_labels(adata: sc.AnnData, domain_key: str = DOMAIN_KEY) -> np.ndarray:
    if domain_key not in adata.obs:
        raise KeyError(f"{domain_key!r} not found in adata.obs")
    return adata.obs[domain_key].astype(str).to_numpy()


def get_slice_for_z(
    real_ads: Dict[int, sc.AnnData],
    method_ads: Dict[int, sc.AnnData],
    z: int,
) -> Tuple[Optional[sc.AnnData], bool]:
    if z in real_ads:
        return real_ads[z], True
    if z in method_ads:
        return method_ads[z], False
    return None, False


def infer_network_nn_distance(adata_ref: sc.AnnData, coord_key: str = COORD_KEY) -> float:
    coords = get_coords(adata_ref, coord_key=coord_key)
    coords = coords[np.all(np.isfinite(coords), axis=1)]
    if coords.shape[0] < 3:
        raise ValueError("Too few spots to infer nearest-neighbor spacing.")
    tree = cKDTree(coords)
    dist, _ = tree.query(coords, k=2)
    nn = dist[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size == 0:
        raise ValueError("Failed to infer nearest-neighbor distance.")
    return float(np.median(nn))


def collect_domain_order(*ads_groups: Dict[int, sc.AnnData], domain_key: str = DOMAIN_KEY) -> List[str]:
    labels: List[str] = []
    for ads in ads_groups:
        for ad in ads.values():
            labels.extend(get_labels(ad, domain_key=domain_key).tolist())
    return sort_domains(labels)


def compute_radius_and_z_threshold(
    z_order_score: Sequence[int],
    visium_nn_um: float = VISIUM_NEAREST_SPOT_DISTANCE_UM,
    radius_factor: float = RADIUS_FACTOR,
    max_z_diff_factor: float = MAX_Z_DIFF_FACTOR,
) -> Tuple[float, float, float]:
    z_sorted = np.sort(np.asarray(z_order_score, dtype=np.float32))
    dz_med = float(np.median(np.diff(z_sorted))) if z_sorted.size >= 2 else 0.0
    radius_um = float(radius_factor * np.sqrt(visium_nn_um ** 2 + dz_med ** 2))
    max_z_diff_um = float(max_z_diff_factor * dz_med) if dz_med > 0 else np.inf
    return radius_um, max_z_diff_um, dz_med


# ============================================================
# PHYSICAL SLICE DATA
# ============================================================
def build_physical_slice_data(
    real_ads: Dict[int, sc.AnnData],
    method_ads: Dict[int, sc.AnnData],
    z_order_score: Sequence[int],
    xy_scale_to_um: float,
    domain_key: str = DOMAIN_KEY,
    coord_key: str = COORD_KEY,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Build per-slice XY coordinates in physical µm and valid domain labels."""
    slice_data: Dict[int, Dict[str, np.ndarray]] = {}
    for z in z_order_score:
        ad, is_real = get_slice_for_z(real_ads, method_ads, int(z))
        if ad is None:
            print(f"[WARN] missing scored slice at z={z}")
            continue
        xy = get_coords(ad, coord_key=coord_key)
        labels = get_labels(ad, domain_key=domain_key)
        valid = np.all(np.isfinite(xy), axis=1) & np.array(
            [str(lab) not in INVALID_DOMAIN_LABELS for lab in labels], dtype=bool
        )
        xy_um = xy[valid].astype(np.float32) * float(xy_scale_to_um)
        slice_data[int(z)] = {
            "xy_um": xy_um,
            "labels": labels[valid].astype(str),
            "z_um": np.float32(z),
            "is_real": np.array(bool(is_real)),
        }
    return slice_data


# ============================================================
# NEIGHBORHOOD CONTINUITY + CONNECTIVITY
# ============================================================
def eligible_target_slices(
    z_src: int,
    z_available: Sequence[int],
    radius_um: float,
    max_z_diff_um: float,
    include_same_slice_neighbors: bool,
) -> List[int]:
    out: List[int] = []
    for z_tgt in z_available:
        dz = abs(float(z_tgt) - float(z_src))
        if (not include_same_slice_neighbors) and (z_tgt == z_src):
            continue
        if dz > max_z_diff_um + 1e-8:
            continue
        if dz >= radius_um:
            continue
        out.append(int(z_tgt))
    return out


def query_neighbors_with_optional_topk(
    tree: cKDTree,
    xy_src: np.ndarray,
    radius_xy: float,
    n_neighbors: Optional[int] = N_NEIGHBORS_PER_TARGET_SLICE,
) -> List[np.ndarray]:
    """Return neighbor indices for each source point.

    If n_neighbors is None, return all points within radius_xy. Otherwise, return
    the k nearest points within radius_xy. This makes the connectivity report
    easier to control and compare across methods because each source spot
    contributes at most k neighbor pairs per eligible target slice.
    """
    if n_neighbors is None:
        raw = tree.query_ball_point(xy_src, r=float(radius_xy))
        return [np.asarray(x, dtype=np.int64) for x in raw]

    k = int(n_neighbors)
    if k <= 0:
        raise ValueError("N_NEIGHBORS_PER_TARGET_SLICE must be positive or None.")

    dist, idx = tree.query(xy_src, k=k, distance_upper_bound=float(radius_xy))
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    n_target = tree.n
    out: List[np.ndarray] = []
    for d_row, i_row in zip(dist, idx):
        keep = np.isfinite(d_row) & (i_row >= 0) & (i_row < n_target)
        out.append(i_row[keep].astype(np.int64, copy=False))
    return out


def compute_neighbor_purity_and_connectivity(
    slice_data: Dict[int, Dict[str, np.ndarray]],
    z_order_score: Sequence[int],
    domain_order: Sequence[str],
    radius_um: float,
    max_z_diff_um: float,
    include_same_slice_neighbors: bool = INCLUDE_SAME_SLICE_NEIGHBORS,
    n_neighbors_per_target_slice: Optional[int] = N_NEIGHBORS_PER_TARGET_SLICE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-spot purity and source-domain -> neighbor-domain connectivity.

    Per-spot purity uses:
        same-domain selected cross-slice neighbors / all selected cross-slice neighbors.

    Neighbor selection is controlled by N_NEIGHBORS_PER_TARGET_SLICE:
      - integer k: use up to k nearest neighbors within the physical 3D radius
        in each eligible target slice;
      - None: use all radius-neighbors, matching the original implementation.
    """
    z_available = [int(z) for z in z_order_score if int(z) in slice_data]
    if not z_available:
        raise ValueError("No scored slices available.")

    domain_order = list(domain_order)
    domain_to_idx = {d: i for i, d in enumerate(domain_order)}
    n_domains = len(domain_order)
    connectivity_counts = np.zeros((n_domains, n_domains), dtype=np.int64)

    trees = {z: cKDTree(slice_data[z]["xy_um"]) for z in z_available}
    label_codes = {
        z: np.asarray([domain_to_idx[str(lab)] for lab in slice_data[z]["labels"]], dtype=np.int32)
        for z in z_available
    }

    per_spot_parts: List[pd.DataFrame] = []

    for z_src in z_available:
        xy_src = slice_data[z_src]["xy_um"]
        lab_src = slice_data[z_src]["labels"].astype(str)
        src_codes = label_codes[z_src]
        n_src = xy_src.shape[0]
        total_neighbors = np.zeros(n_src, dtype=np.int32)
        same_neighbors = np.zeros(n_src, dtype=np.int32)

        for z_tgt in eligible_target_slices(
            z_src=z_src,
            z_available=z_available,
            radius_um=radius_um,
            max_z_diff_um=max_z_diff_um,
            include_same_slice_neighbors=include_same_slice_neighbors,
        ):
            dz = abs(float(z_tgt) - float(z_src))
            r_xy = np.sqrt(max(radius_um ** 2 - dz ** 2, 0.0))
            if r_xy <= 0:
                continue

            target_codes = label_codes[z_tgt]
            nbr_lists = query_neighbors_with_optional_topk(
                tree=trees[z_tgt],
                xy_src=xy_src,
                radius_xy=float(r_xy),
                n_neighbors=n_neighbors_per_target_slice,
            )

            for i, idxs in enumerate(nbr_lists):
                if idxs.size == 0:
                    continue
                if include_same_slice_neighbors and z_tgt == z_src:
                    idxs = idxs[idxs != i]
                    if idxs.size == 0:
                        continue

                tgt_codes_i = target_codes[idxs]
                src_code = int(src_codes[i])
                counts = np.bincount(tgt_codes_i, minlength=n_domains)

                connectivity_counts[src_code, :] += counts
                total_neighbors[i] += int(tgt_codes_i.size)
                same_neighbors[i] += int(counts[src_code])

        has_neighbors = total_neighbors > 0
        purity = np.full(n_src, np.nan, dtype=np.float32)
        purity[has_neighbors] = same_neighbors[has_neighbors] / total_neighbors[has_neighbors]

        per_spot_parts.append(pd.DataFrame({
            "slice_z": float(z_src),
            "domain": lab_src,
            "n_neighbors": total_neighbors.astype(int),
            "n_same_domain_neighbors": same_neighbors.astype(int),
            "neighbor_purity": purity.astype(float),
            "has_neighbors": has_neighbors.astype(bool),
        }))

    per_spot_df = pd.concat(per_spot_parts, axis=0, ignore_index=True)
    connectivity_df = connectivity_counts_to_long(connectivity_counts, domain_order)
    return per_spot_df, connectivity_df

def connectivity_counts_to_long(counts: np.ndarray, domain_order: Sequence[str]) -> pd.DataFrame:
    domain_order = list(domain_order)
    row_total = counts.sum(axis=1)
    all_total = int(counts.sum())
    rows: List[Dict[str, object]] = []
    for i, src in enumerate(domain_order):
        for j, tgt in enumerate(domain_order):
            n = int(counts[i, j])
            rows.append({
                "source_domain": str(src),
                "target_domain": str(tgt),
                "n_neighbor_pairs": n,
                "source_domain_neighbor_pairs": int(row_total[i]),
                "fraction_of_source_domain_neighbors": float(n / row_total[i]) if row_total[i] > 0 else np.nan,
                "fraction_of_all_neighbor_pairs": float(n / all_total) if all_total > 0 else np.nan,
                "is_same_domain": bool(str(src) == str(tgt)),
            })
    return pd.DataFrame(rows)


def summarize_neighbor_continuity(
    per_spot_df: pd.DataFrame,
    connectivity_df: pd.DataFrame,
    min_points_per_domain: int = MIN_POINTS_PER_DOMAIN,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Summarize per-domain and overall continuity.

    continuity_score is the spot-averaged purity, matching the original design.
    neighbor_pair_purity is the pair-count weighted diagonal fraction from the
    connectivity matrix. Reporting both makes it easier to diagnose whether a
    score is driven by many weakly connected spots or by many neighbor pairs.
    """
    df_valid = per_spot_df[per_spot_df["has_neighbors"]].copy()
    overall_score = float(df_valid["neighbor_purity"].mean()) if len(df_valid) else np.nan
    overall_coverage = float(per_spot_df["has_neighbors"].mean()) if len(per_spot_df) else np.nan

    conn_diag = connectivity_df[connectivity_df["is_same_domain"]].copy()
    conn_diag = conn_diag.set_index("source_domain")

    rows: List[Dict[str, object]] = []
    for dom, sub in per_spot_df.groupby("domain", sort=False):
        n_total = int(len(sub))
        if n_total < min_points_per_domain:
            continue
        sub_valid = sub[sub["has_neighbors"]]
        n_valid = int(len(sub_valid))
        dom_score = float(sub_valid["neighbor_purity"].mean()) if n_valid else np.nan
        dom_cov = n_valid / max(n_total, 1)
        dom_pair_purity = np.nan
        dom_pair_count = 0
        if str(dom) in conn_diag.index:
            rr = conn_diag.loc[str(dom)]
            dom_pair_purity = float(rr["fraction_of_source_domain_neighbors"])
            dom_pair_count = int(rr["source_domain_neighbor_pairs"])
        rows.append({
            "domain": str(dom),
            "n_points": n_total,
            "n_points_with_neighbors": n_valid,
            "neighbor_coverage": float(dom_cov),
            "continuity_score": dom_score,
            "neighbor_pair_purity": dom_pair_purity,
            "source_domain_neighbor_pairs": dom_pair_count,
        })

    same_pairs = connectivity_df.loc[connectivity_df["is_same_domain"], "n_neighbor_pairs"].sum()
    all_pairs = connectivity_df["n_neighbor_pairs"].sum()
    overall = {
        "overall_continuity_score": overall_score,
        "overall_neighbor_coverage": overall_coverage,
        "overall_neighbor_pair_purity": float(same_pairs / all_pairs) if all_pairs > 0 else np.nan,
        "n_total_points": int(len(per_spot_df)),
        "n_points_with_neighbors": int(per_spot_df["has_neighbors"].sum()),
        "n_neighbor_pairs": int(all_pairs),
    }
    return pd.DataFrame(rows), overall


# ============================================================
# PLOTTING
# ============================================================
def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.8,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    })


def domain_palette(domain_order: Sequence[str]) -> Dict[str, object]:
    preferred = {
        "Layer3": "#4C78A8",  # blue
        "Layer4": "#72B7B2",  # teal
        "Layer5": "#54A24B",  # green
        "Layer6": "#ECAE5F",  # mustard
        "WM": "#B279A2",      # mauve
    }
    cmap = plt.get_cmap("tab10")
    out = {}
    for i, dom in enumerate(domain_order):
        out[str(dom)] = preferred.get(str(dom), cmap(i % 10))
    return out


def plot_domain_continuity_combined(
    per_domain_df: pd.DataFrame,
    overall_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
) -> None:
    set_plot_style()
    if per_domain_df.empty:
        print("[WARN] empty per_domain_df; skip continuity plot")
        return

    method_order = [m for m in METHOD_ORDER if m in set(per_domain_df["method"])]
    x_map = {m: i for i, m in enumerate(method_order)}
    dom_order = sort_domains(per_domain_df["domain"].unique())
    dom_palette = domain_palette(dom_order)

    fig, ax = plt.subplots(figsize=(4.8, 6.0))
    sns.boxplot(
        data=per_domain_df,
        x="method",
        y="continuity_score",
        order=method_order,
        palette=METHOD_PALETTE,
        showfliers=False,
        width=0.52,
        linewidth=1.1,
        ax=ax,
        zorder=1,
    )

    wide = per_domain_df.pivot(index="domain", columns="method", values="continuity_score")
    for dom in wide.index:
        if all(m in wide.columns and pd.notnull(wide.loc[dom, m]) for m in method_order):
            ax.plot(
                [x_map[m] for m in method_order],
                [wide.loc[dom, m] for m in method_order],
                color="0.83",
                lw=1.0,
                alpha=0.95,
                zorder=2,
            )

    rng = np.random.default_rng(0)
    for _, r in per_domain_df.iterrows():
        if r["method"] not in x_map:
            continue
        ax.scatter(
            x_map[r["method"]] + rng.uniform(-0.08, 0.08),
            r["continuity_score"],
            s=95,
            color=dom_palette[str(r["domain"])],
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Spot-averaged same-domain neighbor fraction")
    ax.set_title("3D domain continuity score", fontsize=16, weight="bold")
    ax.grid(axis="y", alpha=0.18)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               markerfacecolor=dom_palette[d], markeredgecolor="black", label=str(d))
        for d in dom_order
    ]
    ax.legend(handles=handles, title="Domain", bbox_to_anchor=(1.02, 1.0), loc="upper left",
              frameon=False, fontsize=10, title_fontsize=13)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)




def make_connectivity_cmap():
    """Cell/Nature-style sequential palette with light low-end and elegant teal-blue highs."""
    # Low values stay light so off-diagonal small fractions remain visible.
    colors = [
        "#fbfbfd",  # near-white
        "#e8eff1",  # pale blue-gray
        "#cfe5df",  # pale sage-teal
        "#95cdbd",  # muted mint
        "#4fa8a1",  # teal
        "#2f6c9f",  # steel blue
        "#1f3f75",  # deep blue
    ]
    return sns.blend_palette(colors, as_cmap=True)

def connectivity_matrix(
    connectivity_df: pd.DataFrame,
    method: str,
    domain_order: Sequence[str],
    value_col: str = "fraction_of_source_domain_neighbors",
) -> pd.DataFrame:
    sub = connectivity_df[connectivity_df["method"] == method]
    mat = sub.pivot(index="source_domain", columns="target_domain", values=value_col)
    mat = mat.reindex(index=list(domain_order), columns=list(domain_order))
    return mat.astype(float)


def plot_neighborhood_connectivity_heatmap(
    connectivity_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Plot row-normalized source-domain -> target-domain connectivity vertically.

    SpatialZ is shown on top and AXIS on bottom. This is easier to read than a
    left-right layout for annotated matrices and is more suitable for a paper
    panel.
    """
    sns.set_theme(style="white", context="paper")
    if connectivity_df.empty:
        print("[WARN] empty connectivity_df; skip heatmap")
        return

    domain_order = sort_domains(
        list(connectivity_df["source_domain"].unique())
        + list(connectivity_df["target_domain"].unique())
    )
    methods = [m for m in METHOD_ORDER if m in set(connectivity_df["method"])]
    if not methods:
        print("[WARN] no valid methods in connectivity_df; skip heatmap")
        return

    n_methods = len(methods)
    fig = plt.figure(figsize=(8.4, 4.65 * n_methods))
    gs = GridSpec(
        n_methods,
        2,
        width_ratios=[1.0, 0.045],
        hspace=0.34,
        wspace=0.08,
        figure=fig,
    )
    cbar_ax = fig.add_subplot(gs[:, 1])
    cmap = make_connectivity_cmap()

    for i, method in enumerate(methods):
        ax = fig.add_subplot(gs[i, 0])
        mat = connectivity_matrix(connectivity_df, method, domain_order)
        mat_values = mat.to_numpy(dtype=float)
        hm = sns.heatmap(
            mat,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 10.0, "fontweight": "normal"},
            square=True,
            linewidths=1.05,
            linecolor="#f3f3f3",
            cbar=(i == 0),
            cbar_ax=cbar_ax if i == 0 else None,
            cbar_kws={"label": "Fraction of selected neighbors"},
        )

        nrow, ncol = mat_values.shape
        for k, txt in enumerate(hm.texts):
            r = k // ncol
            c = k % ncol
            val = float(mat_values[r, c]) if np.isfinite(mat_values[r, c]) else 0.0
            txt.set_color("white" if val >= 0.45 else "#22313f")

        ax.set_title(method, fontsize=20, weight="bold", pad=12)
        ax.set_xlabel("Neighbor layer/domain", fontsize=14)
        ax.set_ylabel("Source layer/domain", fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor", fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.tick_params(axis="both", length=0)
        ax.set_facecolor("#fcfcfc")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("0.50")

    cbar_ax.tick_params(labelsize=11)
    cbar_ax.set_ylabel("Fraction of selected neighbors", fontsize=13)

    neighbor_text = (
        "all radius-limited neighbors per target slice"
        if N_NEIGHBORS_PER_TARGET_SLICE is None
        else f"top-{N_NEIGHBORS_PER_TARGET_SLICE} radius-limited neighbors per target slice"
    )
    fig.suptitle(
        f"Cross-slice neighborhood connectivity by layer/domain\n({neighbor_text})",
        y=0.985,
        fontsize=18,
        weight="bold",
    )
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _gather_spot_neighbor_records(
    slice_data: Dict[int, Dict[str, np.ndarray]],
    z_order_score: Sequence[int],
    radius_um: float,
    max_z_diff_um: float,
    include_same_slice_neighbors: bool = INCLUDE_SAME_SLICE_NEIGHBORS,
    n_neighbors_per_target_slice: Optional[int] = N_NEIGHBORS_PER_TARGET_SLICE,
) -> pd.DataFrame:
    """Recompute lightweight source-spot neighbor composition for case selection.

    The saved continuity CSVs contain aggregate counts, but they do not store
    source spot indices or exact neighbor identities. The diagnosis figure needs
    those local identities, so it performs this small recalculation.
    """
    z_available = [int(z) for z in z_order_score if int(z) in slice_data]
    if not z_available:
        return pd.DataFrame()
    trees = {z: cKDTree(slice_data[z]["xy_um"]) for z in z_available}
    rows = []
    for z_src in z_available:
        xy_src = slice_data[z_src]["xy_um"]
        labels_src = slice_data[z_src]["labels"].astype(str)
        for z_tgt in eligible_target_slices(
            z_src=z_src,
            z_available=z_available,
            radius_um=radius_um,
            max_z_diff_um=max_z_diff_um,
            include_same_slice_neighbors=include_same_slice_neighbors,
        ):
            dz = abs(float(z_tgt) - float(z_src))
            r_xy = np.sqrt(max(radius_um ** 2 - dz ** 2, 0.0))
            if r_xy <= 0:
                continue
            nbr_lists = query_neighbors_with_optional_topk(
                tree=trees[z_tgt],
                xy_src=xy_src,
                radius_xy=float(r_xy),
                n_neighbors=n_neighbors_per_target_slice,
            )
            labels_tgt = slice_data[z_tgt]["labels"].astype(str)
            for i, idxs in enumerate(nbr_lists):
                if idxs.size == 0:
                    continue
                if include_same_slice_neighbors and z_tgt == z_src:
                    idxs = idxs[idxs != i]
                    if idxs.size == 0:
                        continue
                tgt_labs = labels_tgt[idxs]
                counts = pd.Series(tgt_labs).value_counts()
                rec = {
                    "z_src": int(z_src),
                    "src_idx": int(i),
                    "src_domain": str(labels_src[i]),
                    "n_neighbors": int(idxs.size),
                }
                for dom, cnt in counts.items():
                    rec[f"count__{dom}"] = int(cnt)
                    rec[f"frac__{dom}"] = float(cnt / idxs.size)
                rows.append(rec)
    return pd.DataFrame(rows)


def _collect_neighbors_for_one_spot(
    slice_data: Dict[int, Dict[str, np.ndarray]],
    z_src: int,
    src_idx: int,
    radius_um: float,
    max_z_diff_um: float,
    include_same_slice_neighbors: bool = INCLUDE_SAME_SLICE_NEIGHBORS,
    n_neighbors_per_target_slice: Optional[int] = N_NEIGHBORS_PER_TARGET_SLICE,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    z_available = [int(z) for z in Z_ORDER_SCORE if int(z) in slice_data]
    trees = {z: cKDTree(slice_data[z]["xy_um"]) for z in z_available}
    src_xy = slice_data[z_src]["xy_um"][int(src_idx)]
    src_lab = str(slice_data[z_src]["labels"][int(src_idx)])
    source = {"z_src": int(z_src), "src_idx": int(src_idx), "src_xy": src_xy, "src_domain": src_lab}

    parts = []
    for z_tgt in eligible_target_slices(
        z_src=z_src,
        z_available=z_available,
        radius_um=radius_um,
        max_z_diff_um=max_z_diff_um,
        include_same_slice_neighbors=include_same_slice_neighbors,
    ):
        dz = abs(float(z_tgt) - float(z_src))
        r_xy = np.sqrt(max(radius_um ** 2 - dz ** 2, 0.0))
        if r_xy <= 0:
            continue
        nbr_lists = query_neighbors_with_optional_topk(
            tree=trees[z_tgt],
            xy_src=src_xy[None, :],
            radius_xy=float(r_xy),
            n_neighbors=n_neighbors_per_target_slice,
        )
        idxs = nbr_lists[0]
        if idxs.size == 0:
            continue
        if include_same_slice_neighbors and z_tgt == z_src:
            idxs = idxs[idxs != src_idx]
            if idxs.size == 0:
                continue
        xy_tgt = slice_data[z_tgt]["xy_um"][idxs]
        labs_tgt = slice_data[z_tgt]["labels"][idxs].astype(str)
        dx = xy_tgt[:, 0] - src_xy[0]
        dy = xy_tgt[:, 1] - src_xy[1]
        dist3 = np.sqrt(dx ** 2 + dy ** 2 + (float(z_tgt) - float(z_src)) ** 2)
        parts.append(pd.DataFrame({
            "z_tgt": float(z_tgt),
            "tgt_idx": idxs.astype(int),
            "x_um": xy_tgt[:, 0],
            "y_um": xy_tgt[:, 1],
            "z_um": np.full(len(idxs), float(z_tgt)),
            "domain": labs_tgt,
            "dist3_um": dist3,
        }))
    neigh_df = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame(
        columns=["z_tgt", "tgt_idx", "x_um", "y_um", "z_um", "domain", "dist3_um"]
    )
    return source, neigh_df


def _choose_diagnosis_spots(
    spatialz_slice_data: Dict[int, Dict[str, np.ndarray]],
    axis_slice_data: Dict[int, Dict[str, np.ndarray]],
    radius_um: float,
    max_z_diff_um: float,
) -> Dict[str, Tuple[int, int]]:
    sp_summary = _gather_spot_neighbor_records(spatialz_slice_data, Z_ORDER_SCORE, radius_um, max_z_diff_um)
    if sp_summary.empty:
        raise RuntimeError("No neighbor-summary records available for diagnosis figure.")

    bad_frac_col = f"frac__{DIAG_LEAP_DOMAIN}"
    bad_count_col = f"count__{DIAG_LEAP_DOMAIN}"
    if bad_frac_col not in sp_summary.columns:
        sp_summary[bad_frac_col] = 0.0
    if bad_count_col not in sp_summary.columns:
        sp_summary[bad_count_col] = 0

    candidates = sp_summary[
        (sp_summary["src_domain"] == DIAG_SOURCE_DOMAIN)
        & (sp_summary["n_neighbors"] >= DIAG_MIN_NEIGHBORS)
    ].copy()
    if candidates.empty:
        # Fall back to any source-domain spot if Layer5 naming differs.
        candidates = sp_summary[sp_summary["n_neighbors"] >= DIAG_MIN_NEIGHBORS].copy()
    if candidates.empty:
        raise RuntimeError("No spots with sufficient neighbors found for diagnosis figure.")

    candidates = candidates.sort_values([bad_frac_col, bad_count_col, "n_neighbors"], ascending=[False, False, False])
    chosen = candidates.iloc[0]
    sp_z = int(chosen["z_src"])
    sp_idx = int(chosen["src_idx"])
    sp_xy = spatialz_slice_data[sp_z]["xy_um"][sp_idx]
    chosen_domain = str(chosen["src_domain"])

    # Pick an AXIS spot in a similar location and same source domain, preferably at
    # the same z. This creates a fair paired diagnosis panel.
    ax_z = sp_z if sp_z in axis_slice_data else sorted(axis_slice_data.keys(), key=lambda z: abs(z - sp_z))[0]
    ax_xy = axis_slice_data[ax_z]["xy_um"]
    ax_labels = axis_slice_data[ax_z]["labels"].astype(str)
    same_domain = ax_labels == chosen_domain
    if not np.any(same_domain):
        same_domain = np.ones(len(ax_labels), dtype=bool)
    cand_idx = np.where(same_domain)[0]
    d2 = np.sum((ax_xy[cand_idx] - sp_xy[None, :]) ** 2, axis=1)
    ax_idx = int(cand_idx[np.argmin(d2)])
    return {"SpatialZ": (sp_z, sp_idx), "AXIS": (int(ax_z), ax_idx)}


def _draw_radius_ball(ax, center_xyz: np.ndarray, radius_um: float) -> None:
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 18)
    x = center_xyz[0] + radius_um * np.outer(np.cos(u), np.sin(v))
    y = center_xyz[1] + radius_um * np.outer(np.sin(u), np.sin(v))
    z = center_xyz[2] + radius_um * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=2, cstride=2, color="#9fb3c8", linewidth=0.45, alpha=0.18)


def plot_neighborhood_connectivity_diagnosis(
    real_ads: Dict[int, sc.AnnData],
    spatialz_ads: Dict[int, sc.AnnData],
    axis_ads: Dict[int, sc.AnnData],
    xy_scale_to_um: float,
    radius_um: float,
    max_z_diff_um: float,
    save_path: Optional[str] = None,
) -> None:
    """Fancy case-study figure of the local 3D radius-neighborhood.

    It selects a SpatialZ source-domain spot with the largest leakage into the
    biologically implausible target domain, then compares it with a nearby AXIS
    spot. The saved aggregate tables are not enough for this plot because exact
    neighbor identities are required; this function recomputes only the local
    neighbor information needed for the figure.
    """
    # if save_path is None:
    #     save_path = CONNECTIVITY_DIAGNOSIS_FIG
    sns.set_theme(style="white", context="paper")

    domain_order = collect_domain_order(real_ads, spatialz_ads, axis_ads, domain_key=DOMAIN_KEY)
    pal = domain_palette(domain_order)
    spatialz_slice_data = build_physical_slice_data(real_ads, spatialz_ads, Z_ORDER_SCORE, xy_scale_to_um, domain_key=DOMAIN_KEY, coord_key=COORD_KEY)
    axis_slice_data = build_physical_slice_data(real_ads, axis_ads, Z_ORDER_SCORE, xy_scale_to_um, domain_key=DOMAIN_KEY, coord_key=COORD_KEY)
    spot_choice = _choose_diagnosis_spots(spatialz_slice_data, axis_slice_data, radius_um, max_z_diff_um)

    fig = plt.figure(figsize=(13.3, 10.8))
    gs = GridSpec(2, 2, width_ratios=[1.85, 1.0], height_ratios=[1, 1], hspace=0.28, wspace=0.12, figure=fig)

    for row, (method, slice_data) in enumerate([("SpatialZ", spatialz_slice_data), ("AXIS", axis_slice_data)]):
        z_src, src_idx = spot_choice[method]
        source, neigh_df = _collect_neighbors_for_one_spot(slice_data, z_src, src_idx, radius_um, max_z_diff_um)
        src_xy = np.asarray(source["src_xy"], dtype=float)
        src_xyz = np.array([src_xy[0], src_xy[1], float(z_src)], dtype=float)

        ax3d = fig.add_subplot(gs[row, 0], projection="3d")
        axbar = fig.add_subplot(gs[row, 1])

        # Light local context points from eligible nearby slices.
        context_r = DIAG_LOCAL_CONTEXT_FACTOR * radius_um
        context_parts = []
        for z in [int(z) for z in Z_ORDER_SCORE if int(z) in slice_data and abs(float(z) - float(z_src)) <= max_z_diff_um + 1e-8]:
            xy = slice_data[z]["xy_um"]
            labs = slice_data[z]["labels"].astype(str)
            dxy = np.sqrt(np.sum((xy - src_xy[None, :]) ** 2, axis=1))
            keep = dxy <= context_r
            if np.any(keep):
                context_parts.append(pd.DataFrame({
                    "x_um": xy[keep, 0],
                    "y_um": xy[keep, 1],
                    "z_um": np.full(np.sum(keep), float(z)),
                    "domain": labs[keep],
                }))
        context_df = pd.concat(context_parts, axis=0, ignore_index=True) if context_parts else pd.DataFrame(columns=["x_um", "y_um", "z_um", "domain"])
        if len(context_df):
            ax3d.scatter(context_df["x_um"], context_df["y_um"], context_df["z_um"], s=7, c="#d7dde5", alpha=0.24, depthshade=False)

        _draw_radius_ball(ax3d, src_xyz, radius_um)

        # Selected neighbors colored by domain.
        for dom in domain_order:
            sub = neigh_df[neigh_df["domain"] == str(dom)]
            if len(sub) == 0:
                continue
            ax3d.scatter(sub["x_um"], sub["y_um"], sub["z_um"], s=42, color=pal[str(dom)], edgecolor="white", linewidth=0.45, alpha=0.96, depthshade=False)

        # Explicitly draw line segments to implausible leap-domain neighbors.
        leap = neigh_df[neigh_df["domain"] == DIAG_LEAP_DOMAIN]
        for _, rr in leap.iterrows():
            ax3d.plot([src_xyz[0], rr["x_um"]], [src_xyz[1], rr["y_um"]], [src_xyz[2], rr["z_um"]], color=pal.get(DIAG_LEAP_DOMAIN, "#B279A2"), linewidth=1.7, alpha=0.82)

        ax3d.scatter([src_xyz[0]], [src_xyz[1]], [src_xyz[2]], s=190, marker="*", color="#111111", edgecolor="gold", linewidth=1.0, depthshade=False)
        ax3d.set_title(f"{method}: local 3D neighborhood around a {source['src_domain']} spot", fontsize=14.2, fontweight="bold", pad=12)
        ax3d.set_xlabel("x (µm)", labelpad=6)
        ax3d.set_ylabel("y (µm)", labelpad=6)
        ax3d.set_zlabel("z (µm)", labelpad=7)
        ax3d.view_init(elev=DIAG_VIEW_ELEV, azim=DIAG_VIEW_AZIM)
        ax3d.set_xlim(src_xyz[0] - context_r, src_xyz[0] + context_r)
        ax3d.set_ylim(src_xyz[1] - context_r, src_xyz[1] + context_r)
        ax3d.set_zlim(src_xyz[2] - radius_um * 1.2, src_xyz[2] + radius_um * 1.2)
        ax3d.set_box_aspect((1.0, 1.0, 0.55))
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.grid(False)

        counts = neigh_df["domain"].value_counts().reindex(domain_order).fillna(0).astype(int)
        frac = (counts / max(int(counts.sum()), 1)).astype(float)
        y = np.arange(len(domain_order))
        axbar.barh(y, frac.values, color=[pal[str(d)] for d in domain_order], edgecolor="white", height=0.72)
        axbar.set_yticks(y)
        axbar.set_yticklabels(domain_order, fontsize=11)
        axbar.invert_yaxis()
        xmax = max(0.55, min(1.0, float(frac.max()) * 1.20 if len(frac) else 0.55))
        axbar.set_xlim(0, xmax)
        axbar.set_xlabel("Fraction of selected neighbors", fontsize=11)
        axbar.set_title("Neighbor domain composition", fontsize=13.0, fontweight="bold", pad=8)
        axbar.grid(axis="x", alpha=0.16)
        for yi, (cnt, fr) in enumerate(zip(counts.values, frac.values)):
            if cnt > 0:
                axbar.text(fr + 0.01, yi, f"{fr:.2f}  (n={cnt})", va="center", ha="left", fontsize=10)

        source_dom = str(source["src_domain"])
        leap_frac = float(frac.loc[DIAG_LEAP_DOMAIN]) if DIAG_LEAP_DOMAIN in frac.index else 0.0
        same_frac = float(frac.loc[source_dom]) if source_dom in frac.index else 0.0
        note = (
            f"Source spot: z={z_src}, idx={src_idx}\n"
            f"Source domain: {source_dom}\n"
            f"Selected neighbors: n={len(neigh_df)}\n"
            f"Same-domain fraction: {same_frac:.2f}\n"
            f"{DIAG_LEAP_DOMAIN} fraction: {leap_frac:.2f}"
        )
        axbar.text(0.02, -0.24, note, transform=axbar.transAxes, ha="left", va="top", fontsize=10.2,
                   bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.82", alpha=0.95))
        axbar.spines["top"].set_visible(False)
        axbar.spines["right"].set_visible(False)

        if row == 0:
            handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=pal[str(d)], markeredgecolor="white", markersize=8, label=str(d)) for d in domain_order]
            handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="#111111", markeredgecolor="gold", markersize=14, label="Source spot"))
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, 0.966), ncol=min(6, len(handles)), frameon=False, fontsize=10.2)

    rule = (
        "all neighbors within the 3D radius"
        if N_NEIGHBORS_PER_TARGET_SLICE is None
        else f"top-{N_NEIGHBORS_PER_TARGET_SLICE} nearest neighbors within the 3D radius"
    )
    fig.suptitle(
        f"Diagnosis of cross-slice 3D neighborhood connectivity\n"
        f"A {DIAG_SOURCE_DOMAIN} spot should connect locally to nearby layers; {DIAG_LEAP_DOMAIN} neighbors indicate leap-layer leakage ({rule})",
        fontsize=16.2,
        fontweight="bold",
        y=0.995,
    )
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================
# RUN PIPELINE
# ============================================================
def evaluate_method(
    method: str,
    method_ads: Dict[int, sc.AnnData],
    real_ads: Dict[int, sc.AnnData],
    domain_order: Sequence[str],
    xy_scale_to_um: float,
    radius_um: float,
    max_z_diff_um: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], pd.DataFrame]:
    print(f"[RUN] {method}")
    slice_data = build_physical_slice_data(
        real_ads=real_ads,
        method_ads=method_ads,
        z_order_score=Z_ORDER_SCORE,
        xy_scale_to_um=xy_scale_to_um,
        domain_key=DOMAIN_KEY,
        coord_key=COORD_KEY,
    )
    per_spot_df, connectivity_df = compute_neighbor_purity_and_connectivity(
        slice_data=slice_data,
        z_order_score=Z_ORDER_SCORE,
        domain_order=domain_order,
        radius_um=radius_um,
        max_z_diff_um=max_z_diff_um,
        include_same_slice_neighbors=INCLUDE_SAME_SLICE_NEIGHBORS,
        n_neighbors_per_target_slice=N_NEIGHBORS_PER_TARGET_SLICE,
    )
    per_domain_df, overall = summarize_neighbor_continuity(
        per_spot_df=per_spot_df,
        connectivity_df=connectivity_df,
        min_points_per_domain=MIN_POINTS_PER_DOMAIN,
    )

    per_spot_df.insert(0, "method", method)
    per_domain_df.insert(0, "method", method)
    connectivity_df.insert(0, "method", method)
    overall = {"method": method, **overall}
    return per_spot_df, per_domain_df, overall, connectivity_df


def load_inputs_and_metadata():
    real_ads = load_slice_dict(REAL_SLICE_PATHS)
    spatialz_ads = load_slice_dict(SPATIALZ_SLICE_PATHS)
    axis_ads = load_slice_dict(AXIS_SLICE_PATHS)
    if REFERENCE_REAL_Z not in real_ads:
        raise ValueError(f"REFERENCE_REAL_Z={REFERENCE_REAL_Z} not found in REAL_SLICE_PATHS")
    nn_current = infer_network_nn_distance(real_ads[REFERENCE_REAL_Z], coord_key=COORD_KEY)
    xy_scale_to_um = float(VISIUM_NEAREST_SPOT_DISTANCE_UM / nn_current)
    radius_um, max_z_diff_um, dz_med = compute_radius_and_z_threshold(Z_ORDER_SCORE)
    domain_order = collect_domain_order(real_ads, spatialz_ads, axis_ads, domain_key=DOMAIN_KEY)
    meta = {
        "nn_current": nn_current,
        "xy_scale_to_um": xy_scale_to_um,
        "radius_um": radius_um,
        "max_z_diff_um": max_z_diff_um,
        "dz_med": dz_med,
        "domain_order": domain_order,
    }
    return real_ads, spatialz_ads, axis_ads, meta


def compute_all() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real_ads, spatialz_ads, axis_ads, meta = load_inputs_and_metadata()
    nn_current = meta["nn_current"]
    xy_scale_to_um = meta["xy_scale_to_um"]
    radius_um = meta["radius_um"]
    max_z_diff_um = meta["max_z_diff_um"]
    dz_med = meta["dz_med"]
    domain_order = meta["domain_order"]

    print(f"[INFO] current network-space NN distance: {nn_current:.4f}")
    print(f"[INFO] XY scale factor to physical µm: {xy_scale_to_um:.6f}")
    print(f"[INFO] scored slices: {Z_ORDER_SCORE}")
    print(f"[INFO] median z step: {dz_med:.2f} µm")
    print(f"[INFO] 3D neighborhood radius: {radius_um:.2f} µm")
    print(f"[INFO] max z difference: {max_z_diff_um:.2f} µm")
    print(f"[INFO] include same-slice neighbors: {INCLUDE_SAME_SLICE_NEIGHBORS}")
    print(f"[INFO] neighbors per eligible target slice: {N_NEIGHBORS_PER_TARGET_SLICE}")
    print(f"[INFO] domain order: {domain_order}")

    outputs = []
    for method, ads in [("SpatialZ", spatialz_ads), ("AXIS", axis_ads)]:
        outputs.append(evaluate_method(
            method=method,
            method_ads=ads,
            real_ads=real_ads,
            domain_order=domain_order,
            xy_scale_to_um=xy_scale_to_um,
            radius_um=radius_um,
            max_z_diff_um=max_z_diff_um,
        ))

    per_spot_df = pd.concat([x[0] for x in outputs], axis=0, ignore_index=True)
    per_domain_df = pd.concat([x[1] for x in outputs], axis=0, ignore_index=True)
    overall_df = pd.DataFrame([x[2] for x in outputs])
    connectivity_df = pd.concat([x[3] for x in outputs], axis=0, ignore_index=True)

    for df in [overall_df, per_domain_df, connectivity_df]:
        df["xy_scale_to_um"] = xy_scale_to_um
        df["radius_um"] = radius_um
        df["max_z_diff_um"] = max_z_diff_um
        df["z_order_score"] = ",".join(map(str, Z_ORDER_SCORE))
        df["include_same_slice_neighbors"] = INCLUDE_SAME_SLICE_NEIGHBORS
        df["n_neighbors_per_target_slice"] = N_NEIGHBORS_PER_TARGET_SLICE if N_NEIGHBORS_PER_TARGET_SLICE is not None else "all_radius_neighbors"

    overall_df.to_csv(OVERALL_CSV, index=False)
    per_domain_df.to_csv(PER_DOMAIN_CSV, index=False)
    connectivity_df.to_csv(CONNECTIVITY_LONG_CSV, index=False)
    print(f"[OK] saved: {OVERALL_CSV}")
    print(f"[OK] saved: {PER_DOMAIN_CSV}")
    print(f"[OK] saved: {CONNECTIVITY_LONG_CSV}")

    matrix_rows = []
    for method in METHOD_ORDER:
        if method not in set(connectivity_df["method"]):
            continue
        mat = connectivity_matrix(connectivity_df, method, domain_order)
        mat.insert(0, "source_domain", mat.index.astype(str))
        mat.insert(0, "method", method)
        matrix_rows.append(mat.reset_index(drop=True))
    matrix_df = pd.concat(matrix_rows, axis=0, ignore_index=True) if matrix_rows else pd.DataFrame()
    matrix_df.to_csv(CONNECTIVITY_MATRIX_CSV, index=False)
    print(f"[OK] saved: {CONNECTIVITY_MATRIX_CSV}")

    if SAVE_PER_SPOT:
        per_spot_df.to_csv(PER_SPOT_CSV, index=False)
        print(f"[OK] saved: {PER_SPOT_CSV}")

    print("[RESULT] overall continuity:")
    print(overall_df[["method", "overall_continuity_score", "overall_neighbor_pair_purity", "overall_neighbor_coverage"]])
    return overall_df, per_domain_df, per_spot_df, connectivity_df


def load_existing_outputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_df = pd.read_csv(OVERALL_CSV)
    per_domain_df = pd.read_csv(PER_DOMAIN_CSV)
    connectivity_df = pd.read_csv(CONNECTIVITY_LONG_CSV)
    return overall_df, per_domain_df, connectivity_df


def plot_all(
    per_domain_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    connectivity_df: pd.DataFrame,
    real_ads: Dict[int, sc.AnnData],
    spatialz_ads: Dict[int, sc.AnnData],
    axis_ads: Dict[int, sc.AnnData],
    xy_scale_to_um: float,
    radius_um: float,
    max_z_diff_um: float,
) -> None:
    # plot_domain_continuity_combined(
    #     per_domain_df=per_domain_df,
    #     overall_df=overall_df,
    #     save_path=COMBINED_CONTINUITY_FIG,
    # )
    # plot_neighborhood_connectivity_heatmap(
    #     connectivity_df=connectivity_df,
    #     save_path=CONNECTIVITY_HEATMAP_FIG,
    # )
    plot_neighborhood_connectivity_diagnosis(
        real_ads=real_ads,
        spatialz_ads=spatialz_ads,
        axis_ads=axis_ads,
        xy_scale_to_um=xy_scale_to_um,
        radius_um=radius_um,
        max_z_diff_um=max_z_diff_um,
        # save_path=CONNECTIVITY_DIAGNOSIS_FIG,
    )


def main() -> None:
    if RUN_MODE not in {"compute", "plot", "both"}:
        raise ValueError("RUN_MODE must be 'compute', 'plot', or 'both'.")

    # We still load the input slices in plot mode because the new diagnosis figure
    # requires source-spot positions and local neighbor identities, which are not
    # preserved in the saved CSV summaries.
    real_ads, spatialz_ads, axis_ads, meta = load_inputs_and_metadata()

    if RUN_MODE in {"compute", "both"}:
        overall_df, per_domain_df, _, connectivity_df = compute_all()
    else:
        overall_df, per_domain_df, connectivity_df = load_existing_outputs()

    if RUN_MODE in {"plot", "both"}:
        plot_all(
            per_domain_df=per_domain_df,
            overall_df=overall_df,
            connectivity_df=connectivity_df,
            real_ads=real_ads,
            spatialz_ads=spatialz_ads,
            axis_ads=axis_ads,
            xy_scale_to_um=meta["xy_scale_to_um"],
            radius_um=meta["radius_um"],
            max_z_diff_um=meta["max_z_diff_um"],
        )


if __name__ == "__main__":
    main()
