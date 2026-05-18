import os
from pathlib import Path
from collections import Counter
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, kruskal
from sklearn.decomposition import NMF

from spatial_arithmetic_utils import aggregate_virtual_slices_max
from matplotlib import pyplot as plt

# ============================================================
# CONFIG
# ============================================================

BIOLOGICAL_AXIS_NAME = "dlpfc_z"
GROUP_ID = 1
GRID_SIZE = 80
SPATIAL_BANDWIDTH = 1

ROOT = "/media/huifang/data/registration/DLPFC/huifang"
H5AD_DIR = "/media/huifang/data/registration/DLPFC/huifang/h5ad"

ANALYSIS_DIR = os.path.join(
    ROOT,
    "virtual_tissue",
    f"{BIOLOGICAL_AXIS_NAME}_group{GROUP_ID}_grid{GRID_SIZE}_spatial{SPATIAL_BANDWIDTH}",
    "analysis",
)

OUTDIR = os.path.join(ANALYSIS_DIR, "aggregation_evaluation")
os.makedirs(OUTDIR, exist_ok=True)

LAYER_KEY = "layer_guess_reordered"
REFERENCE_AXIS_VALUE = 0.0
SAMPLE_IDS = [0, 1, 2, 3]
MARKER_GENES = ["MFGE8", "MOBP","CUX2","RAI14","RORB", "ENC1","PCP4", "KCNIP2","SYT6", "FOXP2","MBP"]

USE_LOG1P_FOR_PLOT = True
RUN_NMF = True
NMF_COMPONENTS = 20

# output files
LONG_CSV = os.path.join(OUTDIR, "marker_expression_long.csv")
SUMMARY_CSV = os.path.join(OUTDIR, "marker_slice_summary.csv")
ADJ_CSV = os.path.join(OUTDIR, "marker_adjacent_stats.csv")


# ============================================================
# BASIC HELPERS
# ============================================================

def canonical_layer_order(labels) -> List[str]:
    labels = [str(x) for x in pd.unique(pd.Series(labels).dropna())]
    preferred = ["WM", "Layer6", "Layer5", "Layer4", "Layer3", "Layer2", "Layer1"]
    out = [x for x in preferred if x in labels]
    extras = [x for x in labels if x not in out]
    return out + sorted(extras)


def get_gene_vector(ad: AnnData, gene: str, value_layer: Optional[str] = None) -> np.ndarray:
    if gene not in ad.var_names:
        raise ValueError(f"Gene {gene} not found in ad.var_names")

    gene_idx = ad.var_names.get_loc(gene)

    if value_layer is None:
        X = ad.X
    else:
        if value_layer not in ad.layers:
            raise ValueError(f"Layer '{value_layer}' not found in ad.layers")
        X = ad.layers[value_layer]

    if sp.issparse(X):
        vals = X[:, gene_idx].toarray().ravel()
    else:
        vals = np.asarray(X[:, gene_idx]).ravel()

    return vals.astype(np.float32)


def format_p_value(p: float) -> str:
    if np.isnan(p):
        return "P = NA"
    if p < 1e-300:
        return r"$P < 1\times 10^{-300}$"
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        coeff = p / (10 ** exp)
        return rf"$P \leq {coeff:.0f}\times 10^{{{exp}}}$"
    return rf"$P = {p:.2f}$"


# ============================================================
# DATA LOADING
# ============================================================

def load_original_single_slices(
    h5ad_dir: str,
    group_id: int,
    sample_ids: List[int],
) -> Dict[int, AnnData]:
    out = {}
    for sid in sample_ids:
        path = os.path.join(h5ad_dir, f"{group_id}_{sid}.h5ad")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        out[sid] = sc.read_h5ad(path)

    return out


def load_aggregated_slice(
    analysis_dir: str,
    biological_axis_name: str,
    reference_axis_value: float = 0.0,
    output_subdir: str = "aggregation",
    output_name: Optional[str] = None,
) -> str:
    out_path = aggregate_virtual_slices_max(
        input_dir=analysis_dir,
        biological_axis_name=biological_axis_name,
        reference_axis_value=reference_axis_value,
        output_subdir=output_subdir,
        output_name=output_name,
    )
    return out_path


# ============================================================
# NMF WITH LABEL PROPAGATION
# ============================================================

def spatial_resample_nmf_with_labels(
    adata: AnnData,
    grid_size: int = GRID_SIZE,
    n_components: int = 20,
    layer_key: str = LAYER_KEY,
) -> AnnData:
    """
    Resample to a regular grid, fill empty cells from nearest non-empty cells,
    run NMF smoothing, and propagate spatial-domain labels to the grid.

    Returns a new AnnData with:
      - X: NMF-smoothed expression on the grid
      - obsm['spatial']: grid coordinates
      - obs[layer_key]: propagated layer labels
      - var_names copied from input
    """
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] not found")
    if layer_key not in adata.obs:
        raise KeyError(f"adata.obs['{layer_key}'] not found")

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = np.asarray(X, dtype=np.float32)
    X = np.clip(X, 0, None)  # keep NMF-safe

    labels = adata.obs[layer_key].astype(str).values

    # grid boundaries
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)

    x_edges = np.linspace(min_x, max_x, grid_size + 1)
    y_edges = np.linspace(min_y, max_y, grid_size + 1)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T.astype(np.float32)

    # assign spots to bins
    x_bin = np.digitize(coords[:, 0], x_edges) - 1
    y_bin = np.digitize(coords[:, 1], y_edges) - 1
    x_bin = np.clip(x_bin, 0, grid_size - 1)
    y_bin = np.clip(y_bin, 0, grid_size - 1)

    # accumulate average expression per grid cell
    grid_sum = np.zeros((grid_size, grid_size, X.shape[1]), dtype=np.float32)
    counts = np.zeros((grid_size, grid_size), dtype=np.float32)

    label_buckets = [[[] for _ in range(grid_size)] for _ in range(grid_size)]

    for i in range(coords.shape[0]):
        yy = y_bin[i]
        xx = x_bin[i]
        grid_sum[yy, xx] += X[i]
        counts[yy, xx] += 1
        label_buckets[yy][xx].append(labels[i])

    grid_mean = np.zeros_like(grid_sum)
    non_empty = counts > 0
    grid_mean[non_empty] = grid_sum[non_empty] / counts[non_empty, None]

    # majority label per non-empty cell
    grid_labels = np.empty((grid_size, grid_size), dtype=object)
    grid_labels[:] = None
    for yy in range(grid_size):
        for xx in range(grid_size):
            if label_buckets[yy][xx]:
                grid_labels[yy, xx] = Counter(label_buckets[yy][xx]).most_common(1)[0][0]

    # fill empty cells by nearest non-empty cell
    filled_grid = grid_mean.copy()
    filled_labels = grid_labels.copy()

    non_empty_idx = np.argwhere(counts > 0)
    empty_idx = np.argwhere(counts == 0)

    if len(non_empty_idx) == 0:
        raise ValueError("No non-empty grid cells found; cannot run NMF.")

    if len(empty_idx) > 0:
        tree = cKDTree(non_empty_idx)
        _, nearest_indices = tree.query(empty_idx)

        for i, empty in enumerate(empty_idx):
            src = non_empty_idx[nearest_indices[i]]
            filled_grid[tuple(empty)] = grid_mean[tuple(src)]
            filled_labels[tuple(empty)] = grid_labels[tuple(src)]

    # flatten and run NMF
    flat_grid = filled_grid.reshape(-1, X.shape[1])

    model = NMF(
        n_components=n_components,
        solver="mu",
        beta_loss="kullback-leibler",
        init="random",
        random_state=0,
        verbose=False,
        max_iter=500,
    )

    W = model.fit_transform(flat_grid)
    H = model.components_
    smoothed_flat = W @ H

    # output adata
    new_expr = smoothed_flat.astype(np.float32)
    new_labels = filled_labels.reshape(-1)

    new_adata = AnnData(X=new_expr)
    new_adata.var_names = adata.var_names.copy()
    new_adata.obsm["spatial"] = grid_coords
    new_adata.obs[layer_key] = pd.Categorical(new_labels)
    new_adata.uns["derived_operation"] = "spatial_resample_nmf"
    new_adata.uns["derived_grid_size"] = int(grid_size)
    new_adata.uns["derived_nmf_components"] = int(n_components)

    return new_adata


# ============================================================
# DATAFRAME BUILDING
# ============================================================

def build_gene_distribution_df(
    ad: AnnData,
    gene: str,
    slice_label: str,
    mode: str,
    layer_key: str = LAYER_KEY,
    use_log1p_for_plot: bool = True,
) -> pd.DataFrame:
    if layer_key not in ad.obs:
        raise KeyError(f"{layer_key} not found in adata.obs")

    vals = get_gene_vector(ad, gene, value_layer=None)
    layers = ad.obs[layer_key].astype(str).values

    df = pd.DataFrame({
        "gene": gene,
        "mode": mode,  # raw / nmf
        "value_raw": vals,
        "layer": layers,
        "slice_label": slice_label,
    })

    df = df[df["layer"].notna()].copy()
    df = df[df["layer"] != "nan"].copy()

    if use_log1p_for_plot:
        df["value_plot"] = np.log1p(np.clip(df["value_raw"].values, 0, None))
        df["y_label"] = "Log counts"
    else:
        df["value_plot"] = df["value_raw"].values
        df["y_label"] = "Counts"

    return df


# ============================================================
# STATISTICS
# ============================================================
import numpy as np
import pandas as pd


def compute_adjacent_mean_log1p_contrast(
    df: pd.DataFrame,
    layer_order,
    value_col: str = "value_plot",
    normalize: str = "zscore",   # options: "none", "zscore", "minmax"
    absolute: bool = True,
):
    """
    Compute an effect-size style adjacent-layer contrast based on layer-wise mean log1p expression.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
          - "layer"
          - value_col

    layer_order : list
        Ordered layer names, e.g. ["Layer3", "Layer4", "Layer5", "Layer6", "WM"]

    value_col : str
        Usually "value_plot", which is log1p(counts)

    normalize : str
        "none"   -> use raw layer means
        "zscore" -> z-score layer means within this slice
        "minmax" -> min-max normalize layer means within this slice

    absolute : bool
        If True, use absolute adjacent differences.
        If False, keep signed differences.

    Returns
    -------
    out : dict
        {
          "adjacent_mean_log1p_contrast": ...,
          "adjacent_mean_log1p_maxdiff": ...,
          "adjacent_mean_log1p_profile_std": ...,
          "dominant_layer": ...,
          "layer_means": np.ndarray,
          "layer_means_norm": np.ndarray,
          "adjacent_diffs": np.ndarray,
        }
    """
    means = []
    valid_layers = []

    for layer in layer_order:
        vals = df.loc[df["layer"] == layer, value_col].dropna().values
        if len(vals) == 0:
            means.append(np.nan)
        else:
            means.append(float(np.mean(vals)))
        valid_layers.append(layer)

    means = np.asarray(means, dtype=float)

    # normalize within slice
    means_norm = means.copy()
    finite = np.isfinite(means_norm)

    if np.sum(finite) < 2:
        return {
            "adjacent_mean_log1p_contrast": np.nan,
            "adjacent_mean_log1p_maxdiff": np.nan,
            "adjacent_mean_log1p_profile_std": np.nan,
            "dominant_layer": np.nan,
            "layer_means": means,
            "layer_means_norm": means_norm,
            "adjacent_diffs": np.array([], dtype=float),
        }

    if normalize == "zscore":
        mu = np.nanmean(means_norm)
        sd = np.nanstd(means_norm)
        if sd < 1e-12:
            means_norm = means_norm - mu
        else:
            means_norm = (means_norm - mu) / sd

    elif normalize == "minmax":
        mn = np.nanmin(means_norm)
        mx = np.nanmax(means_norm)
        if mx - mn < 1e-12:
            means_norm = means_norm - mn
        else:
            means_norm = (means_norm - mn) / (mx - mn)

    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize mode: {normalize}")

    diffs = np.diff(means_norm)
    if absolute:
        diffs_use = np.abs(diffs)
    else:
        diffs_use = diffs

    dominant_layer = valid_layers[int(np.nanargmax(means))] if np.any(np.isfinite(means)) else np.nan

    return {
        "adjacent_mean_log1p_contrast": float(np.nanmean(diffs_use)),
        "adjacent_mean_log1p_maxdiff": float(np.nanmax(diffs_use)),
        "adjacent_mean_log1p_profile_std": float(np.nanstd(means_norm)),
        "dominant_layer": dominant_layer,
        "layer_means": means,
        "layer_means_norm": means_norm,
        "adjacent_diffs": diffs,
    }



def compute_adjacent_layer_stats(
    df: pd.DataFrame,
    layer_order: List[str],
    value_col: str = "value_plot",
) -> pd.DataFrame:
    rows = []
    for i in range(len(layer_order) - 1):
        l1 = layer_order[i]
        l2 = layer_order[i + 1]

        v1 = df.loc[df["layer"] == l1, value_col].dropna().values
        v2 = df.loc[df["layer"] == l2, value_col].dropna().values

        if len(v1) == 0 or len(v2) == 0:
            p = np.nan
            delta_mean = np.nan
        else:
            p = mannwhitneyu(v1, v2, alternative="two-sided").pvalue
            delta_mean = float(np.mean(v1) - np.mean(v2))

        rows.append({
            "layer_1": l1,
            "layer_2": l2,
            "p_value": p,
            "neglog10_p": -np.log10(p) if (np.isfinite(p) and p > 0) else np.nan,
            "mean_diff": delta_mean,
        })
    return pd.DataFrame(rows)




def compute_gene_separation_summary(
    df: pd.DataFrame,
    layer_order: List[str],
    value_col: str = "value_plot",
) -> pd.DataFrame:
    grouped = (
        df.groupby("layer", as_index=False)[value_col]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .rename(columns={"count": "n"})
    )

    grouped["layer"] = pd.Categorical(grouped["layer"], categories=layer_order, ordered=True)
    grouped = grouped.sort_values("layer").reset_index(drop=True)

    # Kruskal-Wallis across all layers
    values_list = []
    for layer in layer_order:
        vv = df.loc[df["layer"] == layer, value_col].dropna().values
        if len(vv) > 0:
            values_list.append(vv)

    if len(values_list) >= 2:
        H, p_kw = kruskal(*values_list)
    else:
        H, p_kw = np.nan, np.nan

    layer_means = grouped["mean"].values.astype(float)
    if np.sum(np.isfinite(layer_means)) >= 2:
        order_idx = np.argsort(layer_means)[::-1]
        dominant_idx = order_idx[0]
        second_idx = order_idx[1]
        dominant_layer = str(grouped.loc[dominant_idx, "layer"])
        second_layer = str(grouped.loc[second_idx, "layer"])
        dominant_margin = float(layer_means[dominant_idx] - layer_means[second_idx])
    else:
        dominant_layer = np.nan
        second_layer = np.nan
        dominant_margin = np.nan

    adjacent_stats = compute_adjacent_layer_stats(df, layer_order, value_col=value_col)

    return pd.DataFrame([{
        "kruskal_H": H,
        "kruskal_p": p_kw,
        "adjacent_mean_neglog10p": np.nanmean(adjacent_stats["neglog10_p"].values),
        "adjacent_min_p": np.nanmin(adjacent_stats["p_value"].values) if len(adjacent_stats) > 0 else np.nan,
        "dominant_layer": dominant_layer,
        "second_layer": second_layer,
        "dominant_margin": dominant_margin,
    }])


# ============================================================
# MAIN COLLECTION
# ============================================================

def load_gene_expression(adata, genes):
    gene_expr = adata[:, genes].X
    if hasattr(gene_expr, 'toarray'):
        gene_expr = gene_expr.toarray()

    # Double log1p transform
    gene_expr = np.log1p(np.log1p(gene_expr))

    # Min-max normalization per gene
    gene_expr_norm = (gene_expr - gene_expr.min(axis=0)) / (gene_expr.max(axis=0) - gene_expr.min(axis=0) + 1e-6)
    return gene_expr_norm

def plot_marker_genes(adata, genes, gene_expr_norm):
    n = len(genes)
    # fig, axes = plt.subplots(1, n, figsize=(8 * n, 10), constrained_layout=True)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)

    for i, gene in enumerate(genes):
        ax = axes[i] if n > 1 else axes
        expr = gene_expr_norm[:, i]
        cmap = plt.cm.Reds
        colors = cmap(expr)
        colors[:, -1] = expr  # set alpha as expression level

        ax.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1],
                   color=colors, s=10)
        ax.set_title(gene,fontsize=26)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

    plt.show()

def collect_aggregation_marker_metrics(
    group_id: int,
    sample_ids: List[int],
    h5ad_dir: str,
    analysis_dir: str,
    biological_axis_name: str,
    reference_axis_value: float,
    marker_genes: List[str],
    layer_key: str = LAYER_KEY,
    outdir: str = OUTDIR,
    run_nmf: bool = False,
    nmf_components: int = 20,
):
    os.makedirs(outdir, exist_ok=True)
    # 2) load original single slices
    # 1) load original single slices
    single_ads = load_original_single_slices(
        h5ad_dir=h5ad_dir,
        group_id=group_id,
        sample_ids=sample_ids,
    )

    # keep deterministic order
    ordered_sids = sorted(single_ads.keys())

    # optionally harmonize genes across slices
    common_genes = None
    for sid in ordered_sids:
        ad = single_ads[sid]
        common_genes = ad.var_names if common_genes is None else common_genes.intersection(ad.var_names)

    single_ads = {
        sid: single_ads[sid][:, common_genes].copy()
        for sid in ordered_sids
    }

    # 2) load AXIS aggregated (max) slice
    agg_path = load_aggregated_slice(
        analysis_dir=analysis_dir,
        biological_axis_name=biological_axis_name,
        reference_axis_value=reference_axis_value,
        output_subdir="aggregation",
        output_name=None,
    )
    agg_ad_virtual_max = sc.read_h5ad(agg_path)

    # subset AXIS aggregate to the same common genes for fair comparison
    agg_ad_virtual_max = agg_ad_virtual_max[:, common_genes].copy()

    # 3) concatenated(raw): direct pooling, keep original spatial coords
    concat_raw_list = [single_ads[sid].copy() for sid in ordered_sids]
    agg_ad_concat_raw = sc.concat(
        concat_raw_list,
        axis=0,
        join="inner",
        label="batch",
        keys=[str(sid) for sid in ordered_sids],
        index_unique="-",
    )

    # 4) concatenated(warped): direct pooling after replacing spatial with spatial_axis
    concat_warped_list = []
    for sid in ordered_sids:
        ad = single_ads[sid].copy()
        if "spatial_axis" not in ad.obsm:
            raise KeyError(f"Slice {sid} missing obsm['spatial_axis']")
        ad.obsm["spatial"] = np.asarray(ad.obsm["spatial_axis"]).copy()
        concat_warped_list.append(ad)

    agg_ad_concat_raw_warped = sc.concat(
        concat_warped_list,
        axis=0,
        join="inner",
        label="batch",
        keys=[str(sid) for sid in ordered_sids],
        index_unique="-",
    )

    # optional metadata
    agg_ad_concat_raw.uns["derived_operation"] = "concat_raw"
    agg_ad_concat_raw_warped.uns["derived_operation"] = "concat_raw_warped"
    agg_ad_virtual_max.uns["derived_operation"] = "axis_aggregation_max"

    # 5) prepare analysis objects in the exact order you want in the figure
    analysis_objects = []
    for sid in ordered_sids:
        analysis_objects.append((f"Single {sid}", "raw", single_ads[sid]))

    analysis_objects.append(("Pool (raw)", "raw", agg_ad_concat_raw))
    analysis_objects.append(("Pool (warped)", "raw", agg_ad_concat_raw_warped))
    analysis_objects.append(("AXIS", "raw", agg_ad_virtual_max))

    # debug
    # exprs = load_gene_expression(agg_ad, ["MOBP"])
    # plot_marker_genes(agg_ad, ["MOBP"], exprs)

    # if run_nmf:
    #     print("run nmf_aggregation")
    #     nmf_agg = spatial_resample_nmf_with_labels(
    #         adata=agg_ad,
    #         grid_size=GRID_SIZE,
    #         n_components=nmf_components,
    #         layer_key=layer_key,
    #     )
    #     print("run nmf_single")
    #     nmf_single = {
    #         sid: spatial_resample_nmf_with_labels(
    #             adata=single_ads[sid],
    #             grid_size=GRID_SIZE,
    #             n_components=nmf_components,
    #             layer_key=layer_key,
    #         )
    #         for sid in sorted(single_ads.keys())
    #     }
    #
    #
    #     for sid in sorted(nmf_single.keys()):
    #         analysis_objects.append((f"Single {sid}", "nmf", nmf_single[sid]))
    #     analysis_objects.append(("Aggregated", "nmf", nmf_agg))

    # 4) collect long-form expression + metrics
    long_rows = []
    summary_rows = []
    adjacent_rows = []

    for slice_label, mode, ad in analysis_objects:
        if layer_key not in ad.obs:
            raise KeyError(f"{layer_key} missing in {slice_label} ({mode})")

        layer_order = canonical_layer_order(ad.obs[layer_key].astype(str).values)

        for gene in marker_genes:
            if gene not in ad.var_names:
                print(f"[Warn] {gene} not found in {slice_label} ({mode}), skipping.")
                continue

            df = build_gene_distribution_df(
                ad=ad,
                gene=gene,
                slice_label=slice_label,
                mode=mode,
                layer_key=layer_key,
                use_log1p_for_plot=USE_LOG1P_FOR_PLOT,
            )
            long_rows.append(df)


            global_summary = compute_gene_separation_summary(
                df,
                layer_order=layer_order,
                value_col="value_plot",
            )
            global_summary["gene"] = gene
            global_summary["slice_label"] = slice_label
            global_summary["mode"] = mode

            layer_order = canonical_layer_order(df["layer"].values)

            contrast_out = compute_adjacent_mean_log1p_contrast(
                df,
                layer_order=layer_order,
                value_col="value_plot",
                normalize="zscore",
                absolute=True,
            )

            global_summary["adjacent_mean_log1p_contrast_z"] = contrast_out["adjacent_mean_log1p_contrast"]
            global_summary["adjacent_mean_log1p_maxdiff_z"] = contrast_out["adjacent_mean_log1p_maxdiff"]
            global_summary["adjacent_mean_log1p_profile_std_z"] = contrast_out["adjacent_mean_log1p_profile_std"]

            summary_rows.append(global_summary)

            adj_stats = compute_adjacent_layer_stats(
                df,
                layer_order=layer_order,
                value_col="value_plot",
            )
            adj_stats["gene"] = gene
            adj_stats["slice_label"] = slice_label
            adj_stats["mode"] = mode
            adjacent_rows.append(adj_stats)

    long_df = pd.concat(long_rows, axis=0, ignore_index=True)
    summary_df = pd.concat(summary_rows, axis=0, ignore_index=True)
    adjacent_df = pd.concat(adjacent_rows, axis=0, ignore_index=True)

    long_df.to_csv(LONG_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    adjacent_df.to_csv(ADJ_CSV, index=False)

    print(f"[OK] saved long-form expression: {LONG_CSV}")
    print(f"[OK] saved summary metrics: {SUMMARY_CSV}")
    print(f"[OK] saved adjacent stats: {ADJ_CSV}")


# ============================================================
# RUN
# ============================================================




aggregate_virtual_slices_max(
    input_dir=ANALYSIS_DIR,
    biological_axis_name=BIOLOGICAL_AXIS_NAME,
    reference_axis_value=0,   # choose whichever z you want to keep attrs from
    output_subdir="aggregation",
)
#

# collect_aggregation_marker_metrics(
#         group_id=GROUP_ID,
#         sample_ids=SAMPLE_IDS,
#         h5ad_dir=H5AD_DIR,
#         analysis_dir=ANALYSIS_DIR,
#         biological_axis_name=BIOLOGICAL_AXIS_NAME,
#         reference_axis_value=REFERENCE_AXIS_VALUE,
#         marker_genes=MARKER_GENES,
#         layer_key=LAYER_KEY,
#         outdir=OUTDIR,
#         run_nmf=RUN_NMF,
#         nmf_components=NMF_COMPONENTS,
#     )
