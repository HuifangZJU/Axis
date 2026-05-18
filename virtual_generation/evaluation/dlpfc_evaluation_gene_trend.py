import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import pickle
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# CONFIG
# ============================================================
root = "/media/huifang/data/registration/DLPFC/huifang"
OUTDIR = os.path.join(root, "virtual_tissue", "evaluation")
os.makedirs(OUTDIR, exist_ok=True)

COORD_KEY = "spatial"
LOG1P = True
NORMALIZE_TOTAL = True
TARGET_SUM = 1e4

REFERENCE_REAL_Z = 10
RESULTS_CACHE = os.path.join(OUTDIR, "eval_cache.pkl")
RUN_MODE = "plot"   # "eval", "plot", or "both"
# ------------------------------------------------------------
# Real slices
# ------------------------------------------------------------
REAL_SLICE_PATHS = {
    10: os.path.join(root, "h5ad", "1_1.h5ad"),
    310: os.path.join(root, "h5ad", "1_2.h5ad"),
}

# ------------------------------------------------------------
# Virtual slice sequence
# ------------------------------------------------------------
# virtual_depth = np.arange(40, 310, 30)
virtual_depth = np.arange(20, 310, 10)
Z_ORDER = [10] + virtual_depth.tolist() + [310]

# ------------------------------------------------------------
# SpatialZ virtual slices
# ------------------------------------------------------------
SPATIALZ_SLICE_PATHS = {
    depth: os.path.join(
        root,
        "virtual_tissue",
        "spatialz",
        "dense_reconstruction",
        "group1",
        f"spatialz_{depth}.000.h5ad",
    )
    for depth in virtual_depth
}

# ------------------------------------------------------------
# AXIS virtual slices
# ------------------------------------------------------------
AXIS_SLICE_PATHS = {
    depth: os.path.join(
        root,
        "virtual_tissue",
        "dlpfc_z_group1_grid80_spatial1",
        "query_full_gene_set",
        f"virtual_dlpfc_z_{depth}.000.h5ad",
    )
    for depth in virtual_depth
}

# ------------------------------------------------------------
# Manual labels are used ONLY on the reference real slice
# to fit the single reference curve.
# ------------------------------------------------------------
DOMAIN_KEY = "layer_guess_reordered"
REFERENCE_CURVE_LABEL = "Layer5"   # change to "L5" if needed

# ------------------------------------------------------------
# Reference-curve fitting
# ------------------------------------------------------------
CURVE_BINS = 180
CURVE_SMOOTH_SIGMA = 2.5
CURVE_MIN_POINTS_TOTAL = 30
CURVE_BANDWIDTH_BINS = 6.0

# ------------------------------------------------------------
# Case-study gene
# ------------------------------------------------------------
CASE_GENE = "PCP4"
CASE_EXPR_QUANTILE = 0.50   # top 20% positive spots
CASE_DISTANCE_BINS = 140
CASE_DENSITY_SMOOTH_SIGMA = 1.2

# ------------------------------------------------------------
# Whole-gene metrics
# ------------------------------------------------------------
GENE_MODE = "hvg"           # "hvg", "all", or "custom"
N_TOP_GENES = 800
CUSTOM_GENES = None

# ------------------------------------------------------------
# Metric plot display
# ------------------------------------------------------------
DISPLAY_Y_QUANTILE = 99.0   # display clipping only; all genes still used in calculation

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
FIG_DPI = 300
TAB10_FIRST5 = [plt.get_cmap("tab10")(i) for i in range(5)]


# ============================================================
# BASIC HELPERS
# ============================================================
def load_slice_dict(path_dict, coord_key="spatial", flip_y=True, keep_positive_y=False):
    out = {}
    for z, p in path_dict.items():
        if p is None or str(p).strip() == "":
            continue
        if not os.path.exists(p):
            print(f"[WARN] missing file for z={z}: {p}")
            continue

        ad = sc.read_h5ad(p)

        if "valid" in ad.obs.columns:
            ad = ad[ad.obs["valid"]].copy()

        if flip_y:
            if coord_key not in ad.obsm:
                raise KeyError(f"'{coord_key}' not found in ad.obsm for z={z}")

            coords = np.asarray(ad.obsm[coord_key], dtype=np.float32).copy()

            if keep_positive_y:
                coords[:, 1] = coords[:, 1].max() - coords[:, 1]
            else:
                coords[:, 1] = -coords[:, 1]

            ad.obsm[coord_key] = coords

        out[z] = ad

    return out

def preprocess_adata(
    adata: sc.AnnData,
    normalize_total: bool = NORMALIZE_TOTAL,
    target_sum: float = TARGET_SUM,
    log1p: bool = LOG1P,
) -> sc.AnnData:
    ad = adata.copy()
    if normalize_total:
        sc.pp.normalize_total(ad, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(ad)
    return ad


def get_coords(adata: sc.AnnData, coord_key: str = COORD_KEY) -> np.ndarray:
    if coord_key not in adata.obsm:
        raise KeyError(f"'{coord_key}' not found in adata.obsm")
    return np.asarray(adata.obsm[coord_key], dtype=np.float32)


def get_labels(adata: sc.AnnData, domain_key: str = DOMAIN_KEY) -> np.ndarray:
    if domain_key not in adata.obs:
        raise KeyError(f"'{domain_key}' not found in adata.obs")
    return adata.obs[domain_key].astype(str).values


def get_display_slice(real_ads: Dict[int, sc.AnnData], method_ads: Dict[int, sc.AnnData], z: int):
    if z in real_ads:
        return real_ads[z], True
    if z in method_ads:
        return method_ads[z], False
    return None, False


def common_gene_intersection(adatas: Sequence[sc.AnnData]) -> List[str]:
    common = set(adatas[0].var_names.astype(str))
    for ad in adatas[1:]:
        common &= set(ad.var_names.astype(str))
    return sorted(common)


def subset_to_genes(adata: sc.AnnData, genes: Sequence[str]) -> sc.AnnData:
    keep = [g for g in genes if g in adata.var_names]
    return adata[:, keep].copy()


def extract_gene_vector(adata: sc.AnnData, gene: str) -> np.ndarray:
    if gene not in adata.var_names:
        raise KeyError(f"Gene '{gene}' not found in var_names")
    x = adata[:, [gene]].X
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x).reshape(-1).astype(np.float32)


def sparse_column_variance(X) -> np.ndarray:
    if sparse.issparse(X):
        X = X.tocsr()
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
        var = mean_sq - mean ** 2
    else:
        X = np.asarray(X)
        var = np.var(X, axis=0)
    return np.asarray(var, dtype=np.float32)


# ============================================================
# GENE SELECTION
# ============================================================
def select_genes(
    real_ads: Dict[int, sc.AnnData],
    spatialz_ads: Dict[int, sc.AnnData],
    axis_ads: Dict[int, sc.AnnData],
    mode: str = GENE_MODE,
    n_top: int = N_TOP_GENES,
    custom_genes: Optional[Sequence[str]] = CUSTOM_GENES,
) -> List[str]:
    all_ads = list(real_ads.values()) + list(spatialz_ads.values()) + list(axis_ads.values())
    common_genes = common_gene_intersection(all_ads)

    if mode == "all":
        return common_genes

    if mode == "custom":
        if custom_genes is None:
            raise ValueError("GENE_MODE='custom' but CUSTOM_GENES is None")
        return [g for g in custom_genes if g in common_genes]

    if mode == "hvg":
        ref_ads = [subset_to_genes(real_ads[z], common_genes) for z in sorted(real_ads.keys())]
        ad_ref = sc.concat(ref_ads, join="outer", merge="same")
        var = sparse_column_variance(ad_ref.X)
        idx = np.argsort(var)[::-1][:n_top]
        return ad_ref.var_names.values[idx].tolist()

    raise ValueError(f"Unknown GENE_MODE: {mode}")


# ============================================================
# FIT SINGLE REFERENCE CURVE FROM LAYER5 OF REFERENCE SLICE
# ============================================================
def fit_reference_curve_from_label(
    adata_ref: sc.AnnData,
    ref_label: str = REFERENCE_CURVE_LABEL,
    domain_key: str = DOMAIN_KEY,
    coord_key: str = COORD_KEY,
    n_bins: int = CURVE_BINS,
    smooth_sigma: float = CURVE_SMOOTH_SIGMA,
    min_points_total: int = CURVE_MIN_POINTS_TOTAL,
    bandwidth_bins: float = CURVE_BANDWIDTH_BINS,
):
    """
    Fit a single smooth reference curve directly in original spot coordinates:
      u := x
      v := y
    so the reference curve is y_ref(x).
    """
    coords_xy = get_coords(adata_ref, coord_key=coord_key)
    labels = get_labels(adata_ref, domain_key=domain_key).astype(str)

    u = coords_xy[:, 0].astype(np.float32)   # x
    v = coords_xy[:, 1].astype(np.float32)   # y

    mask_ref = labels == str(ref_label)
    u_ref = u[mask_ref]
    v_ref = v[mask_ref]

    if len(u_ref) < min_points_total:
        raise ValueError(
            f"Reference label '{ref_label}' has too few points ({len(u_ref)} < {min_points_total})."
        )

    u_min, u_max = np.min(u), np.max(u)
    u_grid = np.linspace(u_min, u_max, n_bins).astype(np.float32)
    du = np.median(np.diff(u_grid))
    bandwidth = float(max(bandwidth_bins * du, 1e-6))

    curve = np.full(n_bins, np.nan, dtype=np.float32)
    for i, ug in enumerate(u_grid):
        w = np.exp(-0.5 * ((u_ref - ug) / bandwidth) ** 2).astype(np.float32)
        keep = w > np.exp(-0.5 * (2.5 ** 2))
        if np.sum(keep) < 3:
            continue
        ww = w[keep]
        vv = v_ref[keep]
        curve[i] = np.sum(ww * vv) / np.maximum(np.sum(ww), 1e-8)

    valid = np.isfinite(curve)
    if np.sum(valid) < 4:
        raise ValueError(
            f"Reference curve from '{ref_label}' has too little support ({np.sum(valid)} valid grid points)."
        )

    f_fill = interp1d(
        u_grid[valid],
        curve[valid],
        kind="linear",
        bounds_error=False,
        fill_value=(curve[valid][0], curve[valid][-1]),
    )
    curve_fill = f_fill(u_grid).astype(np.float32)
    curve_smooth = gaussian_filter1d(curve_fill, sigma=smooth_sigma, mode="nearest").astype(np.float32)

    du_arr = np.diff(u_grid)
    dv_arr = np.diff(curve_smooth)
    ds = np.sqrt(du_arr ** 2 + dv_arr ** 2)
    s_grid = np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float32)

    v_of_u = interp1d(
        u_grid,
        curve_smooth,
        kind="linear",
        bounds_error=False,
        fill_value=(curve_smooth[0], curve_smooth[-1]),
    )
    s_of_u = interp1d(
        u_grid,
        s_grid,
        kind="linear",
        bounds_error=False,
        fill_value=(s_grid[0], s_grid[-1]),
    )

    return {
        "u_grid": u_grid,
        "v_grid": curve_smooth,
        "s_grid": s_grid,
        "v_of_u": v_of_u,
        "s_of_u": s_of_u,
        "u_min": float(u_grid[0]),
        "u_max": float(u_grid[-1]),
        "coord_key": coord_key,
        "reference_label": ref_label,
    }

# ============================================================
# SIGNED DISTANCE TO REFERENCE CURVE
# ============================================================
def estimate_spot_spacing(
    adatas: Sequence[sc.AnnData],
    coord_key: str = COORD_KEY,
    trim_percentile: Tuple[float, float] = (5, 95),
) -> float:
    """
    Estimate the Visium spot spacing from the median nearest-neighbor distance
    across one or more anchor slices.

    We compute the nearest non-self neighbor distance for each spot, trim extreme
    values, and take the global median.
    """
    nn_all = []

    for ad in adatas:
        coords = get_coords(ad, coord_key=coord_key)
        if coords.shape[0] < 2:
            continue

        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)   # first neighbor is self, second is true NN
        nn = dists[:, 1].astype(np.float32)

        lo, hi = np.percentile(nn, trim_percentile)
        nn = nn[(nn >= lo) & (nn <= hi)]
        if len(nn) > 0:
            nn_all.append(nn)

    if len(nn_all) == 0:
        raise ValueError("Failed to estimate spot spacing: no valid nearest-neighbor distances found.")

    nn_all = np.concatenate(nn_all)
    return float(np.median(nn_all))

def compute_signed_distance_to_reference_curve(
    adata: sc.AnnData,
    ref_curve,
    coord_key: str = COORD_KEY,
):
    """
    Compute vertical distance to the reference curve directly in original coordinates:
      d = y_spot - y_ref(x_spot)
    """
    coords_xy = get_coords(adata, coord_key=coord_key)

    u = coords_xy[:, 0].astype(np.float32)   # x
    v = coords_xy[:, 1].astype(np.float32)   # y

    u_clip = np.clip(u, ref_curve["u_min"], ref_curve["u_max"])
    v_ref = ref_curve["v_of_u"](u_clip).astype(np.float32)

    # optional along-curve coordinate
    s = ref_curve["s_of_u"](u_clip).astype(np.float32)

    d = (v - v_ref).astype(np.float32)
    return s, d


def weighted_quantile(values, weights, qs):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    qs = np.asarray(qs, dtype=float)

    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[keep]
    weights = weights[keep]

    if len(values) == 0:
        return np.full_like(qs, np.nan, dtype=float)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    return np.interp(qs, cdf, values)



def infer_global_distance_extent_scaled(
    adata_list: Sequence[sc.AnnData],
    gene: str,
    ref_curve,
    spot_spacing: float,
    expr_quantile: Optional[float] = CASE_EXPR_QUANTILE,
    q: Tuple[float, float] = (0.5, 99.5),
) -> Tuple[float, float]:
    all_d = []
    for ad in adata_list:
        x = extract_gene_vector(ad, gene)
        _, d = compute_signed_distance_to_reference_curve(ad, ref_curve, coord_key=COORD_KEY)

        mask = np.isfinite(d) & np.isfinite(x) & (x > 0)
        d = d[mask]
        x = x[mask]
        if len(x) == 0:
            continue

        if expr_quantile is not None:
            thr = np.quantile(x, expr_quantile)
            keep = x >= thr
            d = d[keep]

        if len(d) == 0:
            continue
        all_d.append(d / spot_spacing)

    if len(all_d) == 0:
        return -1.0, 1.0

    all_d = np.concatenate(all_d)
    d_min = float(np.percentile(all_d, q[0]))
    d_max = float(np.percentile(all_d, q[1]))
    return d_min, d_max


def z_to_edges(z_order: Sequence[float]) -> np.ndarray:
    z = np.asarray(z_order, dtype=np.float32)
    if len(z) == 1:
        return np.array([z[0] - 0.5, z[0] + 0.5], dtype=np.float32)
    mids = 0.5 * (z[:-1] + z[1:])
    left = z[0] - 0.5 * (z[1] - z[0])
    right = z[-1] + 0.5 * (z[-1] - z[-2])
    return np.concatenate([[left], mids, [right]]).astype(np.float32)


# ============================================================
# REFERENCE CURVE + ANCHOR PCP4 FIGURE
# ============================================================
def plot_reference_curve_and_anchor_maps(
    adata_ref: sc.AnnData,
    real_ads: Dict[int, sc.AnnData],
    ref_curve,
    gene: str = CASE_GENE,
    domain_key: str = DOMAIN_KEY,
    coord_key: str = COORD_KEY,
    save_path: Optional[str] = None,
):
    ref_coords = get_coords(adata_ref, coord_key=coord_key)
    ref_labels = get_labels(adata_ref, domain_key=domain_key).astype(str)

    ref_curve_xy = np.column_stack([ref_curve["u_grid"], ref_curve["v_grid"]]).astype(np.float32)

    anchor10 = real_ads[10]
    anchor310 = real_ads[310]
    coords10 = get_coords(anchor10, coord_key=coord_key)
    vals10 = extract_gene_vector(anchor10, gene)
    coords310 = get_coords(anchor310, coord_key=coord_key)
    vals310 = extract_gene_vector(anchor310, gene)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # left: reference geometry
    ax = axes[0]
    uniq = [u for u in pd.unique(pd.Series(ref_labels)) if pd.notnull(u)]
    uniq_sorted = sorted(uniq)
    color_map = {lab: TAB10_FIRST5[i % 5] for i, lab in enumerate(uniq_sorted)}

    for lab in uniq_sorted:
        m = ref_labels == lab
        ax.scatter(
            ref_coords[m, 0], ref_coords[m, 1],
            s=48, c=[color_map[lab]], edgecolors="white", linewidths=0.5,
            rasterized=True, label=lab
        )

    ax.plot(ref_curve_xy[:, 0], ref_curve_xy[:, 1], color="black", lw=4)
    ax.set_title(f"Reference curve from {ref_curve['reference_label']}", fontsize=24, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    # ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=14, loc="best")

    # shared color range for anchors
    vmin = 0.0
    vmax = max(np.percentile(np.concatenate([vals10, vals310]), 99), 1e-6)

    # middle: anchor 10
    ax = axes[1]
    sc1 = ax.scatter(
        coords10[:, 0], coords10[:, 1],
        c=vals10, cmap="Reds", vmin=vmin, vmax=vmax,
        s=48, edgecolors="white", linewidths=0.35, rasterized=True
    )
    ax.plot(ref_curve_xy[:, 0], ref_curve_xy[:, 1], color="#00cfd1", lw=4)
    ax.set_title(f"{gene} on real anchor (10 µm)", fontsize=24, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    # ax.invert_yaxis()

    # right: anchor 310
    ax = axes[2]
    sc2 = ax.scatter(
        coords310[:, 0], coords310[:, 1],
        c=vals310, cmap="Reds", vmin=vmin, vmax=vmax,
        s=48, edgecolors="white", linewidths=0.35, rasterized=True
    )
    ax.plot(ref_curve_xy[:, 0], ref_curve_xy[:, 1], color="#00cfd1", lw=4)
    ax.set_title(f"{gene} on real anchor (310 µm)", fontsize=24, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    # ax.invert_yaxis()

    cbar = fig.colorbar(sc2, ax=axes[1:], shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(f"{gene} expression", fontsize=18)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()


# ============================================================
# CASE GENE: DISTANCE DENSITY HEATMAP
# ============================================================
def compute_gene_distance_density_column(
    adata: sc.AnnData,
    gene: str,
    ref_curve,
    spot_spacing: float,
    coord_key: str = COORD_KEY,
    distance_edges: Optional[np.ndarray] = None,
    expr_quantile: Optional[float] = CASE_EXPR_QUANTILE,
    smooth_sigma_bins: float = CASE_DENSITY_SMOOTH_SIGMA,
):
    x = extract_gene_vector(adata, gene)
    _, d = compute_signed_distance_to_reference_curve(
        adata=adata,
        ref_curve=ref_curve,
        coord_key=coord_key,
    )

    mask = np.isfinite(d) & np.isfinite(x) & (x > 0)
    d = d[mask] / spot_spacing
    x = x[mask]

    if len(x) == 0:
        density = np.zeros(len(distance_edges) - 1, dtype=np.float32)
        return density, np.nan, np.nan

    if expr_quantile is not None:
        thr = np.quantile(x, expr_quantile)
        keep = x >= thr
        d = d[keep]
        x = x[keep]

    density, _ = np.histogram(d, bins=distance_edges, weights=x)
    density = density.astype(np.float32)

    if smooth_sigma_bins is not None and smooth_sigma_bins > 0:
        density = gaussian_filter1d(density, sigma=smooth_sigma_bins, mode="nearest")

    density = density / max(density.sum(), 1e-8)

    centers = 0.5 * (distance_edges[:-1] + distance_edges[1:])
    weighted_mean = float(np.sum(d * x) / max(np.sum(x), 1e-8))
    peak_distance = float(centers[np.argmax(density)])

    return density, weighted_mean, peak_distance


def build_gene_distance_density_matrix_for_sequence(
    real_ads: Dict[int, sc.AnnData],
    method_ads: Dict[int, sc.AnnData],
    gene: str,
    z_order: Sequence[int],
    ref_curve,
    spot_spacing: float,
    distance_edges: np.ndarray,
):
    cols = []
    means = []
    peaks = []

    for z in z_order:
        ad, _ = get_display_slice(real_ads, method_ads, z)
        if ad is None:
            cols.append(np.full(len(distance_edges) - 1, np.nan, dtype=np.float32))
            means.append(np.nan)
            peaks.append(np.nan)
            continue

        dens, mu, pk = compute_gene_distance_density_column(
            adata=ad,
            gene=gene,
            ref_curve=ref_curve,
            spot_spacing=spot_spacing,
            distance_edges=distance_edges,
        )
        cols.append(dens)
        means.append(mu)
        peaks.append(pk)

    return (
        np.stack(cols, axis=1).astype(np.float32),
        np.asarray(means, dtype=np.float32),
        np.asarray(peaks, dtype=np.float32),
    )


def make_light_distance_cmap():
    colors = [
        "#f8fbff",
        "#dcecf7",
        "#b8d7ea",
        "#e1d7f3",
        "#efc7dc",
        "#f8dfe6",
    ]
    return LinearSegmentedColormap.from_list("light_distance", colors, N=256)


def plot_case_gene_distance_density_heatmap(
    real_ads: Dict[int, sc.AnnData],
    spatialz_ads: Dict[int, sc.AnnData],
    axis_ads: Dict[int, sc.AnnData],
    gene: str,
    z_order: Sequence[int],
    ref_curve,
    spot_spacing: float,
    save_path: Optional[str] = None,
):
    all_ads = list(real_ads.values()) + list(spatialz_ads.values()) + list(axis_ads.values())
    d_min, d_max = infer_global_distance_extent_scaled(
        all_ads, gene, ref_curve, spot_spacing, expr_quantile=CASE_EXPR_QUANTILE, q=(0.5, 99.5)
    )
    distance_edges = np.linspace(d_min, d_max, CASE_DISTANCE_BINS + 1)

    M_spatialz, mean_spatialz, peak_spatialz = build_gene_distance_density_matrix_for_sequence(
        real_ads, spatialz_ads, gene, z_order, ref_curve, spot_spacing, distance_edges
    )
    M_axis, mean_axis, peak_axis = build_gene_distance_density_matrix_for_sequence(
        real_ads, axis_ads, gene, z_order, ref_curve, spot_spacing, distance_edges
    )

    vals = np.concatenate([
        M_spatialz[np.isfinite(M_spatialz)],
        M_axis[np.isfinite(M_axis)],
    ])
    vmax = max(float(np.percentile(vals, 99.5)), 1e-8)

    cmap = make_light_distance_cmap()
    y_edges = distance_edges
    x_edges = z_to_edges(z_order)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.1), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.08})

    row_defs = [
        ("SpatialZ", M_spatialz, mean_spatialz, peak_spatialz),
        ("AXIS", M_axis, mean_axis, peak_axis),
    ]

    im_last = None
    for ax, (row_name, M, mu_line, pk_line) in zip(axes, row_defs):
        im = ax.pcolormesh(
            x_edges,
            y_edges,
            M,
            cmap=cmap,
            shading="flat",
            vmin=0.0,
            vmax=vmax,
            rasterized=True,
        )
        im_last = im

        ax.plot(z_order, mu_line, color="#4C9F9F", lw=1.8, marker="o", ms=4.5, zorder=4)
        ax.plot(z_order, pk_line, color="#C76D7E", lw=1.5, linestyle="--",
                marker="s", ms=3.8, zorder=4)

        ax.axhline(0.0, color="white", lw=0.9, alpha=0.55)
        ax.set_ylabel(
            f"{row_name}\nRelative position\naround PCP4 band",
            fontsize=13,
            fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xticks(z_order)
    axes[-1].set_xticklabels([str(z) for z in z_order], fontsize=11)
    axes[-1].set_xlabel("Position along anatomical axis (μm)", fontsize=13)

    fig.suptitle(f"{gene}: distance density heatmap", fontsize=17, fontweight="semibold", y=0.97)

    handles = [
        Line2D([0], [0], color="#4C9F9F", lw=1.8, marker="o", ms=5, label="Weighted mean position"),
        Line2D([0], [0], color="#C76D7E", lw=1.5, linestyle="--", marker="s", ms=4, label="Peak position"),
    ]

    fig.subplots_adjust(left=0.15, right=0.82, top=0.86, bottom=0.11, hspace=0.10)
    cax = fig.add_axes([0.845, 0.20, 0.022, 0.52])
    cbar = fig.colorbar(im_last, cax=cax)
    cbar.set_label("Expression-weighted density", fontsize=11)

    fig.legend(handles=handles, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.84, 0.80), loc="upper left")

    if save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()


# ============================================================
# CASE GENE: DISTANCE VIOLIN MAP
# ============================================================
def kde_on_grid(values, weights, y_grid, bw_scale=0.85):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if len(values) < 3 or np.unique(values).size < 2:
        return np.zeros_like(y_grid, dtype=np.float32)

    try:
        kde = gaussian_kde(values, weights=weights)
        kde.set_bandwidth(kde.factor * bw_scale)
        dens = kde(y_grid)
        dens = dens / max(np.max(dens), 1e-12)
        return dens.astype(np.float32)
    except Exception:
        return np.zeros_like(y_grid, dtype=np.float32)


def mode_from_kde(values, weights, y_grid, bw_scale=0.85):
    dens = kde_on_grid(values, weights, y_grid, bw_scale=bw_scale)
    if np.all(dens <= 0):
        return np.nan
    return float(y_grid[np.argmax(dens)])


def plot_case_gene_distance_violin_map(
    dist_dict_spatialz: Dict[int, np.ndarray],
    dist_dict_axis: Dict[int, np.ndarray],
    weight_dict_spatialz: Dict[int, np.ndarray],
    weight_dict_axis: Dict[int, np.ndarray],
    z_order: Sequence[int],
    spot_spacing: float,
    save_path: Optional[str] = None,
    title: str = "PCP4: distance violin map",
):
    def draw_violin(ax, x0, y_grid, dens, color, ls, lw=1.8, alpha_fill=0.20):
        if dens.max() > 0:
            dens = dens / dens.max()
        width = 0.28 * median_step * dens
        ax.fill_betweenx(y_grid, x0 - width, x0 + width, color=color, alpha=alpha_fill, lw=0, zorder=2)
        ax.plot(x0 - width, y_grid, color=color, lw=lw, linestyle=ls, zorder=3)
        ax.plot(x0 + width, y_grid, color=color, lw=lw, linestyle=ls, zorder=3)

    median_step = np.median(np.diff(np.asarray(z_order, dtype=np.float32))) if len(z_order) > 1 else 1.0

    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)

    method_specs = [
        ("SpatialZ", dist_dict_spatialz, weight_dict_spatialz, "#7f8fa9"),
        ("AXIS", dist_dict_axis, weight_dict_axis, "#cc4c39"),
    ]

    for ax, (method_name, dist_dict, weight_dict, color) in zip(axes, method_specs):
        peak_line = []

        for z in z_order:
            d = np.asarray(dist_dict[z], dtype=float) / spot_spacing
            w = np.asarray(weight_dict[z], dtype=float)

            keep = np.isfinite(d) & np.isfinite(w) & (w > 0)
            d = d[keep]
            w = w[keep]

            if len(d) == 0:
                peak_line.append(np.nan)
                continue

            q_lo, q_hi = weighted_quantile(d, w, [0.005, 0.995])
            pad = 0.15 * max(q_hi - q_lo, 0.25)
            y_grid = np.linspace(q_lo - pad, q_hi + pad, 250)

            dens = kde_on_grid(d, w, y_grid, bw_scale=0.85)

            ls = "-" if z in [10, 310] else (0, (4, 4))
            lw = 2.0 if z in [10, 310] else 1.8
            alpha_fill = 0.22 if z in [10, 310] else 0.18

            draw_violin(ax=ax, x0=z, y_grid=y_grid, dens=dens, color=color, ls=ls, lw=lw, alpha_fill=alpha_fill)

            pk = mode_from_kde(d, w, y_grid, bw_scale=0.85)
            peak_line.append(pk)

        peak_line = np.asarray(peak_line, dtype=float)

        ax.plot(
            z_order,
            peak_line,
            color="black",
            lw=2.0,
            marker="o",
            ms=6,
            zorder=4,
        )

        ax.axhline(0, color="0.75", lw=1.1)
        ax.set_ylabel(
            f"{method_name}\nRelative position\naround PCP4 band",
            fontsize=22,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xticks(z_order)
    axes[-1].set_xticklabels([str(z) for z in z_order], fontsize=18)
    axes[-1].set_xlabel("Position along anatomical axis (µm)", fontsize=24)

    fig.suptitle(title, fontsize=30, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()


# ============================================================
# WHOLE-GENE DISTANCE METRICS
# ============================================================
def compute_weighted_mean_distance_for_genes(
    adata: sc.AnnData,
    genes: Sequence[str],
    ref_curve,
    spot_spacing: float,
    coord_key: str = COORD_KEY,
) -> np.ndarray:
    _, d = compute_signed_distance_to_reference_curve(
        adata=adata,
        ref_curve=ref_curve,
        coord_key=coord_key,
    )
    d = d / spot_spacing

    valid = np.isfinite(d)
    d = d[valid]

    ad = subset_to_genes(adata, genes)
    X = ad.X

    if sparse.issparse(X):
        X = X.tocsr()[valid]
        denom = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        numer = np.asarray(X.T @ d).ravel().astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)[valid]
        denom = X.sum(axis=0).astype(np.float32)
        numer = (d[:, None] * X).sum(axis=0).astype(np.float32)

    mu = numer / np.maximum(denom, 1e-8)
    mu[denom <= 1e-8] = np.nan
    return mu.astype(np.float32)


def build_gene_distance_matrix_for_sequence(
    real_ads: Dict[int, sc.AnnData],
    method_ads: Dict[int, sc.AnnData],
    genes: Sequence[str],
    z_order: Sequence[int],
    ref_curve,
    spot_spacing: float,
):
    cols = []
    for z in z_order:
        ad, _ = get_display_slice(real_ads, method_ads, z)
        if ad is None:
            cols.append(np.full(len(genes), np.nan, dtype=np.float32))
            continue
        mu = compute_weighted_mean_distance_for_genes(
            adata=ad,
            genes=genes,
            ref_curve=ref_curve,
            spot_spacing=spot_spacing,
        )
        cols.append(mu)
    return np.stack(cols, axis=1).astype(np.float32)


def gene_wise_trajectory_roughness(M: np.ndarray) -> np.ndarray:
    if M.shape[1] < 3:
        return np.full(M.shape[0], np.nan, dtype=np.float32)
    d2 = np.abs(M[:, 2:] - 2 * M[:, 1:-1] + M[:, :-2])
    return np.nanmean(d2, axis=1).astype(np.float32)


def gene_wise_adjacent_shift(M: np.ndarray) -> np.ndarray:
    if M.shape[1] < 2:
        return np.full(M.shape[0], np.nan, dtype=np.float32)
    step = np.abs(np.diff(M, axis=1))
    return np.nanmean(step, axis=1).astype(np.float32)


def build_metric_dataframe(
    genes: Sequence[str],
    M_spatialz: np.ndarray,
    M_axis: np.ndarray,
):
    rows = []

    rough_sz = gene_wise_trajectory_roughness(M_spatialz)
    rough_ax = gene_wise_trajectory_roughness(M_axis)

    step_sz = gene_wise_adjacent_shift(M_spatialz)
    step_ax = gene_wise_adjacent_shift(M_axis)

    for i, g in enumerate(genes):
        rows.append({
            "gene": g,
            "method": "SpatialZ",
            "roughness": rough_sz[i],
            "adjacent_shift": step_sz[i],
        })
        rows.append({
            "gene": g,
            "method": "AXIS",
            "roughness": rough_ax[i],
            "adjacent_shift": step_ax[i],
        })

    return pd.DataFrame(rows)


from matplotlib.patches import Patch

def plot_overall_gene_distance_metrics(
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    order = ["SpatialZ", "AXIS"]
    palette = {"SpatialZ": "#8fa0b8", "AXIS": "#d45a45"}

    specs = [
        ("roughness", "Gene-distance trajectory roughness"),
        ("adjacent_shift", "Adjacent distance shift"),
    ]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(5.4, 8),
        gridspec_kw={"hspace": 0.2}   # key spacing control
    )
    if len(specs) == 1:
        axes = [axes]

    for ax_i, (ax, (metric, ylabel)) in enumerate(zip(axes, specs)):
        dfm = metrics_df.copy()
        pooled = dfm[metric].values
        y_hi = np.nanpercentile(pooled, DISPLAY_Y_QUANTILE)
        y_hi = max(y_hi, 1e-8)

        # paired lines
        wide = dfm.pivot(index="gene", columns="method", values=metric)
        for _, row in wide.iterrows():
            if pd.notnull(row.get("SpatialZ", np.nan)) and pd.notnull(row.get("AXIS", np.nan)):
                ax.plot(
                    [0, 1], [row["SpatialZ"], row["AXIS"]],
                    color="0.86", lw=0.6, alpha=0.25, zorder=1
                )

        sns.boxplot(
            data=dfm,
            x="method",
            y=metric,
            order=order,
            palette=palette,
            width=0.48,
            showfliers=False,
            linewidth=1.5,
            ax=ax,
        )

        sns.stripplot(
            data=dfm,
            x="method",
            y=metric,
            order=order,
            palette=palette,
            size=4,
            alpha=0.55,
            jitter=0.18,
            linewidth=0,
            ax=ax,
        )

        ax.set_ylim(0, y_hi * 1.1)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(ylabel, fontsize=18, pad=12,weight="bold")   # add title padding
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(axis="y", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # only show x tick labels on the bottom panel
        if ax_i == 0:
            ax.tick_params(axis="x", labelbottom=False, bottom=False)
        else:
            ax.tick_params(axis="x", labelsize=17, rotation=10, pad=6)

        for i, m in enumerate(order):
            vals = dfm.loc[dfm["method"] == m, metric].values
            n_trunc = np.sum(vals > y_hi)
            if n_trunc > 0:
                ax.text(
                    i, y_hi * 1.01,
                    f"{n_trunc} high outliers",
                    ha="center", va="bottom",
                    fontsize=14, color="0.25"
                )

    # handles = [Patch(facecolor=palette[k], edgecolor="black", label=k) for k in order]
    # fig.legend(
    #     handles=handles,
    #     loc="center left",
    #     bbox_to_anchor=(0.8, 0.5),
    #     frameon=False,
    #     fontsize=16
    # )
    #
    # # reserve room for legend and avoid clipping
    # fig.subplots_adjust(left=0.12, right=0.80, top=0.96, bottom=0.08)

    if save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()


def export_gene_distance_matrix_csv(
    genes: Sequence[str],
    M: np.ndarray,
    z_order: Sequence[int],
    save_path: str,
):
    df = pd.DataFrame(M, index=genes, columns=[str(z) for z in z_order])
    df.index.name = "gene"
    df.to_csv(save_path)
    print(f"[OK] saved: {save_path}")


# ============================================================
# CASE GENE DISTANCE EXTRACTION FOR VIOLINS
# ============================================================
def build_case_gene_distance_dicts(
    real_ads: Dict[int, sc.AnnData],
    method_ads: Dict[int, sc.AnnData],
    gene: str,
    z_order: Sequence[int],
    ref_curve,
    expr_quantile: Optional[float] = CASE_EXPR_QUANTILE,
):
    dist_dict = {}
    weight_dict = {}

    for z in z_order:
        ad, _ = get_display_slice(real_ads, method_ads, z)
        if ad is None:
            dist_dict[z] = np.array([], dtype=np.float32)
            weight_dict[z] = np.array([], dtype=np.float32)
            continue

        x = extract_gene_vector(ad, gene)
        _, d = compute_signed_distance_to_reference_curve(ad, ref_curve, coord_key=COORD_KEY)

        mask = np.isfinite(d) & np.isfinite(x) & (x > 0)
        d = d[mask]
        x = x[mask]

        if len(x) > 0 and expr_quantile is not None:
            thr = np.quantile(x, expr_quantile)
            keep = x >= thr
            d = d[keep]
            x = x[keep]

        dist_dict[z] = d.astype(np.float32)
        weight_dict[z] = x.astype(np.float32)

    return dist_dict, weight_dict

def save_cache(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] saved cache: {path}")


def load_cache(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[OK] loaded cache: {path}")
    return obj

def plot_reference_curve_and_anchor_maps_from_cache(
    cache,
    save_path: Optional[str] = None,
):
    ref_coords = cache["reference_plot"]["ref_coords"]
    ref_labels = cache["reference_plot"]["ref_labels"]
    ref_curve_xy = cache["reference_plot"]["ref_curve_xy"]
    coords10 = cache["reference_plot"]["coords10"]
    vals10 = cache["reference_plot"]["vals10"]
    coords310 = cache["reference_plot"]["coords310"]
    vals310 = cache["reference_plot"]["vals310"]
    gene = cache["meta"]["case_gene"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # left: reference geometry
    ax = axes[0]
    uniq = [u for u in pd.unique(pd.Series(ref_labels)) if pd.notnull(u)]
    uniq_sorted = sorted(uniq)
    color_map = {lab: TAB10_FIRST5[i % 5] for i, lab in enumerate(uniq_sorted)}

    for lab in uniq_sorted:
        m = ref_labels == lab
        ax.scatter(
            ref_coords[m, 0], ref_coords[m, 1],
            s=20, c=[color_map[lab]], edgecolors="white", linewidths=0.5,
            rasterized=True, label=lab
        )

    ax.plot(ref_curve_xy[:, 0], ref_curve_xy[:, 1], color="black", lw=4)
    ax.set_title(f"Reference curve from {cache['meta']['reference_curve_label']}", fontsize=20, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=14, loc="best")

    # shared color range
    vmin = 0.0
    vmax = max(np.percentile(np.concatenate([vals10, vals310]), 99), 1e-6)

    # middle: anchor 10
    ax = axes[1]
    ax.scatter(
        coords10[:, 0], coords10[:, 1],
        c=vals10, cmap="Reds", vmin=vmin, vmax=vmax,
        s=35, edgecolors="white", linewidths=0.35, rasterized=True
    )
    ax.plot(ref_curve_xy[:, 0], ref_curve_xy[:, 1], color="#00cfd1", lw=4)
    ax.set_title(f"Reference line on {gene} (10 µm)", fontsize=20, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    # right: anchor 310
    ax = axes[2]
    sc2 = ax.scatter(
        coords310[:, 0], coords310[:, 1],
        c=vals310, cmap="Reds", vmin=vmin, vmax=vmax,
        s=35, edgecolors="white", linewidths=0.35, rasterized=True
    )
    ax.plot(ref_curve_xy[:, 0], ref_curve_xy[:, 1], color="#00cfd1", lw=4)
    ax.set_title(f"Reference line on {gene} (310 µm)", fontsize=20, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    # cbar = fig.colorbar(sc2, ax=axes[1:], shrink=0.85, pad=0.02)
    # cbar.ax.tick_params(labelsize=13)
    # cbar.set_label(f"{gene} expression", fontsize=18)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()

def plot_case_gene_distance_density_heatmap_from_cache(
    cache,
    save_path: Optional[str] = None,
):
    z_order = cache["meta"]["z_order"]

    gene = cache["meta"]["case_gene"]

    M_spatialz = cache["heatmap"]["M_spatialz"]
    M_axis = cache["heatmap"]["M_axis"]
    mean_spatialz = cache["heatmap"]["mean_spatialz"]
    peak_spatialz = cache["heatmap"]["peak_spatialz"]
    mean_axis = cache["heatmap"]["mean_axis"]
    peak_axis = cache["heatmap"]["peak_axis"]
    distance_edges = cache["heatmap"]["distance_edges"]

    vals = np.concatenate([
        M_spatialz[np.isfinite(M_spatialz)],
        M_axis[np.isfinite(M_axis)],
    ])
    vmax = max(float(np.percentile(vals, 99.5)), 1e-8)

    cmap = make_light_distance_cmap()
    y_edges = distance_edges
    x_edges = z_to_edges(z_order)

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.1), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.08})

    row_defs = [
        ("SpatialZ", M_spatialz, mean_spatialz, peak_spatialz),
        ("AXIS", M_axis, mean_axis, peak_axis),
    ]

    im_last = None
    for ax, (row_name, M, mu_line, pk_line) in zip(axes, row_defs):
        im = ax.pcolormesh(
            x_edges,
            y_edges,
            M,
            cmap=cmap,
            shading="flat",
            vmin=0.0,
            vmax=vmax,
            rasterized=True,
        )
        im_last = im
        ax.plot(z_order, pk_line, color="#C76D7E", lw=1.5, linestyle="--",
                marker="s", ms=3.8, zorder=4)

        ax.axhline(0.0, color="white", lw=0.9, alpha=0.55)
        ax.set_ylabel(
            # f"{row_name}"
        "Signed spot spacings",
            fontsize=13,
            # fontweight="bold",
            # rotation=0
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    virtual_depth_label = np.arange(70, 310, 60)
    x_ticks_positions= [10] + virtual_depth_label.tolist() + [310]
    x_ticks_labels = ["10 (real)"] + [f"{depth} (virtual)" for depth in virtual_depth_label.tolist()] + ["310 (real)"]

    axes[-1].set_xticks(x_ticks_positions)
    axes[-1].set_xticklabels(x_ticks_labels, fontsize=10.5)
    axes[-1].set_xlabel("Position along anatomical axis (μm)", fontsize=13)

    fig.suptitle(f"{gene}: weighted distance to reference curve", fontsize=17, fontweight="semibold", y=0.92)

    handles = [
        Line2D([0], [0], color="#C76D7E", lw=1.5, linestyle="--", marker="s", ms=4, label="Peak\nposition"),
    ]

    fig.subplots_adjust(left=0.15, right=0.82, top=0.86, bottom=0.11, hspace=0.10)
    cax = fig.add_axes([0.845, 0.20, 0.022, 0.52])
    cbar = fig.colorbar(im_last, cax=cax)
    cbar.set_label("Expression-weighted distance density", fontsize=11)

    fig.legend(handles=handles, frameon=False, fontsize=12,
               bbox_to_anchor=(0.82, 0.86), loc="upper left")

    if save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()

def evaluate_and_save(cache_path=RESULTS_CACHE):
    # 1) Load and preprocess
    real_ads = load_slice_dict(REAL_SLICE_PATHS)
    spatialz_ads = load_slice_dict(SPATIALZ_SLICE_PATHS)
    axis_ads = load_slice_dict(AXIS_SLICE_PATHS)

    if REFERENCE_REAL_Z not in real_ads:
        raise ValueError(f"REFERENCE_REAL_Z={REFERENCE_REAL_Z} not found in REAL_SLICE_PATHS")

    real_ads = {z: preprocess_adata(ad) for z, ad in real_ads.items()}
    spatialz_ads = {z: preprocess_adata(ad) for z, ad in spatialz_ads.items()}
    axis_ads = {z: preprocess_adata(ad) for z, ad in axis_ads.items()}

    # 2) Fit reference curve
    ref_curve = fit_reference_curve_from_label(
        adata_ref=real_ads[REFERENCE_REAL_Z],
        ref_label=REFERENCE_CURVE_LABEL,
        domain_key=DOMAIN_KEY,
        coord_key=COORD_KEY,
        n_bins=CURVE_BINS,
        smooth_sigma=CURVE_SMOOTH_SIGMA,
        min_points_total=CURVE_MIN_POINTS_TOTAL,
        bandwidth_bins=CURVE_BANDWIDTH_BINS,
    )
    print(f"[INFO] reference curve fitted from label: {ref_curve['reference_label']}")


    spot_spacing = estimate_spot_spacing(
        [real_ads[10], real_ads[310]],
        coord_key=COORD_KEY,
    )
    print(f"[INFO] estimated spot spacing: {spot_spacing:.4f}")

    # 4) Cache reference-plot arrays
    ref_coords = get_coords(real_ads[REFERENCE_REAL_Z], coord_key=COORD_KEY)
    ref_labels = get_labels(real_ads[REFERENCE_REAL_Z], domain_key=DOMAIN_KEY).astype(str)

    # current PCA-based version:
    ref_curve_xy = np.column_stack([ref_curve["u_grid"], ref_curve["v_grid"]]).astype(np.float32)

    coords10 = get_coords(real_ads[10], coord_key=COORD_KEY)
    vals10 = extract_gene_vector(real_ads[10], CASE_GENE)
    coords310 = get_coords(real_ads[310], coord_key=COORD_KEY)
    vals310 = extract_gene_vector(real_ads[310], CASE_GENE)

    # 5) Cache heatmap arrays
    all_ads = list(real_ads.values()) + list(spatialz_ads.values()) + list(axis_ads.values())
    d_min, d_max = infer_global_distance_extent_scaled(
        all_ads, CASE_GENE, ref_curve, spot_spacing,
        expr_quantile=CASE_EXPR_QUANTILE, q=(0.5, 99.5)
    )
    distance_edges = np.linspace(d_min, d_max, CASE_DISTANCE_BINS + 1)

    M_spatialz, mean_spatialz, peak_spatialz = build_gene_distance_density_matrix_for_sequence(
        real_ads, spatialz_ads, CASE_GENE, Z_ORDER, ref_curve, spot_spacing, distance_edges
    )
    M_axis, mean_axis, peak_axis = build_gene_distance_density_matrix_for_sequence(
        real_ads, axis_ads, CASE_GENE, Z_ORDER, ref_curve, spot_spacing, distance_edges
    )

    # 6) Cache violin arrays
    dist_dict_spatialz, weight_dict_spatialz = build_case_gene_distance_dicts(
        real_ads=real_ads,
        method_ads=spatialz_ads,
        gene=CASE_GENE,
        z_order=Z_ORDER,
        ref_curve=ref_curve,
        expr_quantile=CASE_EXPR_QUANTILE,
    )
    dist_dict_axis, weight_dict_axis = build_case_gene_distance_dicts(
        real_ads=real_ads,
        method_ads=axis_ads,
        gene=CASE_GENE,
        z_order=Z_ORDER,
        ref_curve=ref_curve,
        expr_quantile=CASE_EXPR_QUANTILE,
    )

    # 7) Whole-gene metrics
    genes = select_genes(
        real_ads=real_ads,
        spatialz_ads=spatialz_ads,
        axis_ads=axis_ads,
        mode=GENE_MODE,
        n_top=N_TOP_GENES,
        custom_genes=CUSTOM_GENES,
    )
    print(f"[INFO] using {len(genes)} genes for whole-gene distance metrics")

    real_ads_gene = {z: subset_to_genes(ad, genes) for z, ad in real_ads.items()}
    spatialz_ads_gene = {z: subset_to_genes(ad, genes) for z, ad in spatialz_ads.items()}
    axis_ads_gene = {z: subset_to_genes(ad, genes) for z, ad in axis_ads.items()}

    M_gene_spatialz = build_gene_distance_matrix_for_sequence(
        real_ads=real_ads_gene,
        method_ads=spatialz_ads_gene,
        genes=genes,
        z_order=Z_ORDER,
        ref_curve=ref_curve,
        spot_spacing=spot_spacing,
    )
    M_gene_axis = build_gene_distance_matrix_for_sequence(
        real_ads=real_ads_gene,
        method_ads=axis_ads_gene,
        genes=genes,
        z_order=Z_ORDER,
        ref_curve=ref_curve,
        spot_spacing=spot_spacing,
    )

    metric_df = build_metric_dataframe(
        genes=genes,
        M_spatialz=M_gene_spatialz,
        M_axis=M_gene_axis,
    )

    # keep your existing CSV exports too
    metric_df.to_csv(os.path.join(OUTDIR, "whole_gene_distance_metrics.csv"), index=False)
    export_gene_distance_matrix_csv(
        genes=genes,
        M=M_gene_spatialz,
        z_order=Z_ORDER,
        save_path=os.path.join(OUTDIR, "gene_distance_matrix_spatialz.csv"),
    )
    export_gene_distance_matrix_csv(
        genes=genes,
        M=M_gene_axis,
        z_order=Z_ORDER,
        save_path=os.path.join(OUTDIR, "gene_distance_matrix_axis.csv"),
    )

    cache = {
        "meta": {
            "z_order": list(Z_ORDER),
            "case_gene": CASE_GENE,
            "reference_real_z": REFERENCE_REAL_Z,
            "reference_curve_label": REFERENCE_CURVE_LABEL,
            "spot_spacing": float(spot_spacing),
        },
        "ref_curve": ref_curve,
        "reference_plot": {
            "ref_coords": ref_coords,
            "ref_labels": ref_labels,
            "ref_curve_xy": ref_curve_xy,
            "coords10": coords10,
            "vals10": vals10,
            "coords310": coords310,
            "vals310": vals310,
        },
        "heatmap": {
            "distance_edges": distance_edges,
            "M_spatialz": M_spatialz,
            "M_axis": M_axis,
            "mean_spatialz": mean_spatialz,
            "peak_spatialz": peak_spatialz,
            "mean_axis": mean_axis,
            "peak_axis": peak_axis,
        },
        "violin": {
            "dist_dict_spatialz": dist_dict_spatialz,
            "dist_dict_axis": dist_dict_axis,
            "weight_dict_spatialz": weight_dict_spatialz,
            "weight_dict_axis": weight_dict_axis,
        },
        "metrics": {
            "metric_df": metric_df,
            "genes": genes,
            "M_gene_spatialz": M_gene_spatialz,
            "M_gene_axis": M_gene_axis,
        },
    }

    save_cache(cache, cache_path)

def plot_from_cache(cache_path=RESULTS_CACHE):
    cache = load_cache(cache_path)

    # plot_reference_curve_and_anchor_maps_from_cache(
    #     cache,
    #     save_path=os.path.join(OUTDIR, "reference_curve_layer5_geometry.png"),
    # )
    #
    # plot_case_gene_distance_density_heatmap_from_cache(
    #     cache,
    #     save_path=os.path.join(OUTDIR, "pcp4_distance_density_heatmap.png"),
    # )

    # plot_case_gene_distance_violin_map(
    #     dist_dict_spatialz=cache["violin"]["dist_dict_spatialz"],
    #     dist_dict_axis=cache["violin"]["dist_dict_axis"],
    #     weight_dict_spatialz=cache["violin"]["weight_dict_spatialz"],
    #     weight_dict_axis=cache["violin"]["weight_dict_axis"],
    #     z_order=cache["meta"]["z_order"],
    #     spot_spacing=cache["meta"]["spot_spacing"],
    #     save_path=os.path.join(OUTDIR, "pcp4_distance_violin_map.png"),
    #     title="PCP4: distance violin map",
    # )

    plot_overall_gene_distance_metrics(
        metrics_df=cache["metrics"]["metric_df"],
        save_path=os.path.join(OUTDIR, "whole_gene_distance_metrics_boxplot.png"),
    )



# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if RUN_MODE == "eval":
        evaluate_and_save(cache_path=RESULTS_CACHE)

    elif RUN_MODE == "plot":
        plot_from_cache(cache_path=RESULTS_CACHE)

    elif RUN_MODE == "both":
        evaluate_and_save(cache_path=RESULTS_CACHE)
        plot_from_cache(cache_path=RESULTS_CACHE)

    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")