#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal gene analysis from three spatially aligned raw/count-like h5ad files.

Each timepoint is preprocessed as library-size normalization -> optional log1p
before any temporal subtraction or trend analysis.

The streamlined workflow contains:
1. Coordinate alignment and common-gene extraction.
2. Per-domain temporal gene statistics and pattern fractions.
3. Persisted per-location spatial maps for selected marker genes.
4. Marker-gene spatial triptychs.
5. Per-domain top-gene composition summaries.
6. Cortical-layer and hippocampal representative temporal-pattern maps selected
   from the per-domain top-gene results. Each representative gene is saved as
   an independent figure, with region-level temporal-pattern fractions shown in a
   right-side dot-and-percentage annotation without repeating pattern names.

For each gene listed in SPATIAL_MAP_GENES, evaluation also saves a compact NPZ
under OUTDIR/spatial_maps/. The NPZ contains the exact matched coordinates,
three preprocessed timepoint values, two spatial deltas, the per-location trend
pattern, amplitude, domains, and observation names. This makes downstream
pipeline plotting independent of the analysis code.

Heatmap clustering, heatmap plotting, and their intermediate tables have been
removed. In trend maps, every selected anatomical location is first plotted in
gray; Increasing, Decreasing, Peak, and Trough locations are then overlaid in
color. Therefore near-zero, stable, mixed, or otherwise unclassified locations
remain visible rather than appearing as empty holes.
"""

from axis.settings import data_path
import os
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import kruskal
import argparse

from axis.matching.contrast import extract_preprocessed_matrix

# ============================================================
# CONFIG
# ============================================================
OUTDIR = data_path("paper_analysis/Figure56_mouse/5ghi/")
os.makedirs(OUTDIR, exist_ok=True)

TIMEPOINTS = [
    {"label": "2p5",  "time": 2.5,  "path": data_path("mouse_brain/virtual_tissue/AD_time_grid256_spatial2.0/analysis/virtual_AD_time_2.500.h5ad")},
    {"label": "5p7",  "time": 5.7,  "path": data_path("mouse_brain/virtual_tissue/AD_time_grid256_spatial2.0/analysis/virtual_AD_time_5.700.h5ad")},
    {"label": "17p9", "time": 17.9, "path": data_path("mouse_brain/virtual_tissue/AD_time_grid256_spatial2.0/analysis/virtual_AD_time_17.900.h5ad")},
]

COORD_KEY = "spatial"
DOMAIN_KEY = "axis_joint_domain"

HIPPOCAMPUS_DOMAINS = ["8", "11"]
CORTICAL_LAYER_DOMAINS = ["1", "4", "6", "10"]
MARKER_CASE_GENES = ["Gfap", "Arc"]

# Location-level products persisted for downstream plot-only pipelines.
# By default this matches the marker genes used by plot_gene_triptych().
SPATIAL_MAP_GENES = MARKER_CASE_GENES.copy()
SPATIAL_MAP_DIRNAME = "spatial_maps"

COORD_ROUND_DECIMALS = 4

# Unified count preprocessing. Each location is normalized using the FULL gene
# library in its source h5ad, then optionally log1p-transformed before temporal
# subtraction.
LIBRARY_SIZE_NORMALIZE = True
NORMALIZATION_TARGET_SUM = 253
LOG1P_BEFORE_ANALYSIS = True

MIN_LOCATIONS_PER_DOMAIN = 30
TOP_GENES_PER_DOMAIN =10

# Simple, explicit pixel-level trend rule threshold
PATTERN_MIN_DELTA = 0.05
PATTERN_RELATIVE_FACTOR = 0.1

FIG_DPI = 500
INVERT_X = True
INVERT_Y = False

BG_COLOR = "#D9D9D9"
BG_ALPHA = 0.62
BG_SIZE = 1.0


# ============================================================
# STYLE
# ============================================================
PATTERN_COLORS = {
    "Increase": "#D55E00",
    "Decrease": "#0072B2",
    "Mid-stage peak": "#009E73",
    "Mid-stage trough": "#CC79A7",
    "Stable": "#9E9E9E",
}


def set_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.15,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_name(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def sort_domains(domains: Sequence[str]) -> List[str]:
    def _key(x):
        sx = str(x)
        try:
            return (0, float(sx))
        except Exception:
            return (1, sx)
    return sorted([str(d) for d in domains], key=_key)


def apply_axis_orientation(ax):
    if INVERT_X:
        ax.invert_xaxis()
    if INVERT_Y:
        ax.invert_yaxis()


def robust_sym_vmax(vals: np.ndarray, q: float = 99.0) -> float:
    vals = np.asarray(vals, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 1.0
    vmax = np.percentile(np.abs(vals), q)
    return max(float(vmax), 1e-8)


# ============================================================
# DATA CONTAINER
# ============================================================
@dataclass
class AlignedTimepoint:
    label: str
    time: float
    coords: np.ndarray
    domains: np.ndarray
    X: np.ndarray
    genes: np.ndarray
    obs_names: np.ndarray


# ============================================================
# BASIC HELPERS
# ============================================================
def read_h5ad(path: str) -> sc.AnnData:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ad = sc.read_h5ad(path)
    return ad


def get_coords(ad: sc.AnnData, coord_key: str = COORD_KEY) -> np.ndarray:
    if coord_key not in ad.obsm:
        raise KeyError(f"{coord_key!r} not found in ad.obsm")
    arr = np.asarray(ad.obsm[coord_key], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Invalid coordinate shape: {arr.shape}")
    return arr[:, 1::-1].copy()


def get_domains(ad: sc.AnnData, domain_key: str = DOMAIN_KEY) -> np.ndarray:
    if domain_key not in ad.obs:
        raise KeyError(f"{domain_key!r} not found in ad.obs")
    return ad.obs[domain_key].astype(str).to_numpy()


def make_dense_X(
    ad: sc.AnnData,
    genes: Optional[Sequence[str]] = None,
    row_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Load selected genes after the shared count preprocessing.

    Library sizes are calculated from ALL genes in the original h5ad, even
    when only the common-gene subset is returned.
    """
    return extract_preprocessed_matrix(
        ad,
        row_indices=row_indices,
        genes=genes,
        normalize_total=LIBRARY_SIZE_NORMALIZE,
        target_sum=NORMALIZATION_TARGET_SUM,
        use_log1p=LOG1P_BEFORE_ANALYSIS,
    )


def intersect_genes(adatas: Sequence[sc.AnnData]) -> List[str]:
    common = set(adatas[0].var_names.astype(str))
    for ad in adatas[1:]:
        common &= set(ad.var_names.astype(str))
    genes = sorted(common)
    if len(genes) == 0:
        raise ValueError("No common genes across the input h5ad files.")
    return genes


def coord_keys(coords: np.ndarray, decimals: int = COORD_ROUND_DECIMALS) -> np.ndarray:
    c = np.round(np.asarray(coords, dtype=float), decimals=decimals)
    return np.array([f"{x:.{decimals}f}|{y:.{decimals}f}" for x, y in c], dtype=object)


def valid_domain_mask(domain_array: np.ndarray) -> np.ndarray:
    s = pd.Series(domain_array.astype(str))
    invalid = s.str.lower().isin({"nan", "none", "na", ""})
    return (~invalid).to_numpy()


def valid_domains(domain_array: np.ndarray) -> List[str]:
    mask = valid_domain_mask(domain_array)
    return sort_domains(pd.unique(pd.Series(domain_array[mask].astype(str))))


# ============================================================
# ALIGNMENT
# ============================================================
def align_target_to_reference(
    ref_ad: sc.AnnData,
    tgt_ad: sc.AnnData,
    common_genes: Sequence[str],
    coord_key: str = COORD_KEY,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref_coords = get_coords(ref_ad, coord_key)
    tgt_coords = get_coords(tgt_ad, coord_key)

    ref_keys = coord_keys(ref_coords, decimals=COORD_ROUND_DECIMALS)
    tgt_keys = coord_keys(tgt_coords, decimals=COORD_ROUND_DECIMALS)

    ref_index = pd.Index(ref_keys)
    tgt_index = pd.Index(tgt_keys)
    common = ref_index.intersection(tgt_index)

    if len(common) == 0:
        raise ValueError("No overlapping rounded coordinates found between reference and target.")

    ref_pos = ref_index.get_indexer(common)
    tgt_pos = tgt_index.get_indexer(common)

    order = np.argsort(ref_pos)
    ref_pos = ref_pos[order]
    tgt_pos = tgt_pos[order]

    X_tgt = make_dense_X(
        tgt_ad,
        genes=common_genes,
        row_indices=tgt_pos,
    )
    dom_tgt = get_domains(tgt_ad)[tgt_pos]
    obs_tgt = tgt_ad.obs_names.astype(str).to_numpy()[tgt_pos]

    coords_ref_sub = ref_coords[ref_pos]
    return coords_ref_sub, X_tgt, dom_tgt, obs_tgt


def load_aligned_timepoints() -> List[AlignedTimepoint]:
    ads = [read_h5ad(tp["path"]) for tp in TIMEPOINTS]

    common_genes = intersect_genes(ads)

    ref_meta = TIMEPOINTS[0]
    ref_ad = ads[0]
    ref_coords = get_coords(ref_ad)
    ref_domains = get_domains(ref_ad)
    ref_X = make_dense_X(
        ref_ad,
        genes=common_genes,
    )
    ref_obs = ref_ad.obs_names.astype(str).to_numpy()

    data = [
        AlignedTimepoint(
            label=ref_meta["label"],
            time=float(ref_meta["time"]),
            coords=ref_coords,
            domains=ref_domains,
            X=ref_X,
            genes=np.asarray(common_genes, dtype=object),
            obs_names=ref_obs,
        )
    ]

    for meta, ad in zip(TIMEPOINTS[1:], ads[1:]):
        coords_ref_sub, X_tgt, dom_tgt, obs_tgt = align_target_to_reference(
            ref_ad=ads[0],
            tgt_ad=ad,
            common_genes=common_genes,
            coord_key=COORD_KEY,
        )

        if len(data) == 1 and len(coords_ref_sub) != len(data[0].coords):
            keys_ref_all = coord_keys(data[0].coords)
            keys_keep = coord_keys(coords_ref_sub)
            keep_mask = pd.Index(keys_ref_all).isin(pd.Index(keys_keep))
            data[0] = AlignedTimepoint(
                label=data[0].label,
                time=data[0].time,
                coords=data[0].coords[keep_mask],
                domains=data[0].domains[keep_mask],
                X=data[0].X[keep_mask],
                genes=data[0].genes,
                obs_names=data[0].obs_names[keep_mask],
            )

        ref_keys_keep = coord_keys(data[0].coords)
        cur_keys = coord_keys(coords_ref_sub)
        idx_cur = pd.Index(cur_keys).get_indexer(ref_keys_keep)
        if np.any(idx_cur < 0):
            raise ValueError("Current target failed to align to the retained reference set.")

        data.append(
            AlignedTimepoint(
                label=meta["label"],
                time=float(meta["time"]),
                coords=data[0].coords.copy(),
                domains=dom_tgt[idx_cur],
                X=X_tgt[idx_cur],
                genes=np.asarray(common_genes, dtype=object),
                obs_names=obs_tgt[idx_cur],
            )
        )

    ref_domains_final = data[0].domains.copy()
    for i in range(1, len(data)):
        data[i].domains = ref_domains_final.copy()

    print(f"[INFO] Loaded times: {[tp.time for tp in data]}")
    print(f"[INFO] Matched locations: {data[0].coords.shape[0]}")
    print(f"[INFO] Valid domains: {valid_domains(data[0].domains)}")
    print(f"[INFO] Common genes: {len(data[0].genes)}")
    print(
        "[INFO] Preprocessing: "
        f"normalize_total={LIBRARY_SIZE_NORMALIZE}, "
        f"target_sum={NORMALIZATION_TARGET_SUM:g}, "
        f"log1p={LOG1P_BEFORE_ANALYSIS}"
    )
    print(
        "[INFO] Temporal delta rule: preprocess(later) - preprocess(earlier)"
    )
    return data


# ============================================================
# SIMPLE, INTUITIVE PIXEL-LEVEL TREND RULE
# ============================================================
def classify_location_pattern(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Explicit and intuitive rule based on the two interval deltas.

    Let:
        d12 = v2 - v1
        d23 = v3 - v2

    Threshold:
        tau = max(PATTERN_MIN_DELTA, PATTERN_RELATIVE_FACTOR * median amplitude)

    Then:
        Increasing : d12 > +tau and d23 > +tau
        Decreasing : d12 < -tau and d23 < -tau
        Peak       : d12 > +tau and d23 < -tau
        Trough     : d12 < -tau and d23 > +tau
        Mixed      : otherwise
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)

    d12 = v2 - v1
    d23 = v3 - v2
    amp = np.maximum(np.abs(d12), np.abs(d23))

    tau = max(PATTERN_MIN_DELTA, float(np.nanmedian(amp) * PATTERN_RELATIVE_FACTOR))

    pat = np.full(v1.shape[0], "Stable", dtype=object)
    pat[(d12 > tau) & (d23 > tau)] = "Increase"
    pat[(d12 < -tau) & (d23 < -tau)] = "Decrease"
    pat[(d12 > tau) & (d23 < -tau)] = "Mid-stage peak"
    pat[(d12 < -tau) & (d23 > tau)] = "Mid-stage trough"

    return pat, amp.astype(np.float32)


# ============================================================
# PERSIST LOCATION-LEVEL SPATIAL MAPS
# ============================================================
def save_spatial_gene_maps(
    data: List[AlignedTimepoint],
    genes: Sequence[str],
    outdir: str,
) -> pd.DataFrame:
    """
    Save the per-location temporal products used by spatial plotting.

    This function is intentionally part of the *evaluation* stage. It stores
    the already-computed location-level products so later pipeline figures can
    be strictly plot-only and do not need to repeat normalization, alignment,
    subtraction, or temporal-pattern classification.

    One compressed NPZ is written per requested gene with these arrays:

        gene
        coords
        domains
        obs_names
        valid_domain_mask
        times
        time_labels
        value_2p5
        value_5p7
        value_17p9
        delta_5p7_minus_2p5
        delta_17p9_minus_5p7
        delta_17p9_minus_2p5
        pattern
        amplitude

    The spatial rows are in exactly the same matched order as ``data[0]``.
    """
    if len(data) != 3:
        raise ValueError(
            f"Expected exactly three aligned timepoints, got {len(data)}."
        )

    ensure_dir(outdir)

    genes_available = data[0].genes.astype(str).tolist()
    gene_to_index = {gene: i for i, gene in enumerate(genes_available)}

    coords = np.asarray(data[0].coords, dtype=np.float32)
    domains = np.asarray(data[0].domains, dtype=str)
    obs_names = np.asarray(data[0].obs_names, dtype=str)
    valid_mask = valid_domain_mask(domains).astype(bool)

    times = np.asarray([tp.time for tp in data], dtype=np.float32)
    time_labels = np.asarray([tp.label for tp in data], dtype=str)

    manifest_rows = []

    for gene in genes:
        gene = str(gene)

        if gene not in gene_to_index:
            print(f"[WARN] Spatial-map gene {gene!r} not found. Skip.")
            continue

        j = gene_to_index[gene]

        value_1 = np.asarray(data[0].X[:, j], dtype=np.float32)
        value_2 = np.asarray(data[1].X[:, j], dtype=np.float32)
        value_3 = np.asarray(data[2].X[:, j], dtype=np.float32)

        delta_12 = (value_2 - value_1).astype(np.float32)
        delta_23 = (value_3 - value_2).astype(np.float32)
        delta_13 = (value_3 - value_1).astype(np.float32)

        patterns, amplitude = classify_location_pattern(
            value_1,
            value_2,
            value_3,
        )
        patterns = np.asarray(patterns, dtype=str)
        amplitude = np.asarray(amplitude, dtype=np.float32)

        # Record the exact threshold used for this gene for provenance.
        tau = max(
            PATTERN_MIN_DELTA,
            float(np.nanmedian(amplitude) * PATTERN_RELATIVE_FACTOR),
        )

        save_path = os.path.join(
            outdir,
            f"{safe_name(gene)}_spatial_temporal_map.npz",
        )

        np.savez_compressed(
            save_path,
            gene=np.asarray(gene),
            coords=coords,
            domains=domains,
            obs_names=obs_names,
            valid_domain_mask=valid_mask,
            times=times,
            time_labels=time_labels,
            value_2p5=value_1,
            value_5p7=value_2,
            value_17p9=value_3,
            delta_5p7_minus_2p5=delta_12,
            delta_17p9_minus_5p7=delta_23,
            delta_17p9_minus_2p5=delta_13,
            pattern=patterns,
            amplitude=amplitude,
            pattern_threshold=np.asarray(tau, dtype=np.float32),
            normalization_target_sum=np.asarray(
                NORMALIZATION_TARGET_SUM,
                dtype=np.float32,
            ),
            library_size_normalize=np.asarray(
                LIBRARY_SIZE_NORMALIZE,
                dtype=bool,
            ),
            log1p_before_analysis=np.asarray(
                LOG1P_BEFORE_ANALYSIS,
                dtype=bool,
            ),
        )

        unique_patterns, pattern_counts = np.unique(
            patterns,
            return_counts=True,
        )
        count_map = dict(zip(unique_patterns.tolist(), pattern_counts.tolist()))

        manifest_rows.append({
            "gene": gene,
            "file": save_path,
            "n_locations": int(len(coords)),
            "n_valid_domain_locations": int(valid_mask.sum()),
            "pattern_threshold": float(tau),
            "n_increasing": int(count_map.get("Increase", 0)),
            "n_decreasing": int(count_map.get("Decrease", 0)),
            "n_peak": int(count_map.get("Peak", 0)),
            "n_trough": int(count_map.get("Trough", 0)),
            "n_mixed": int(count_map.get("Stable", 0)),
        })

        print(f"[SAVE] Spatial map {gene}: {save_path}")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(outdir, "spatial_maps_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[SAVE] Spatial-map manifest: {manifest_path}")

    return manifest_df


def load_saved_spatial_gene_map(
    path: str,
) -> dict:
    """
    Lightweight loader for downstream plot-only scripts.

    No analysis is performed here; arrays are returned exactly as persisted.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    payload = np.load(path, allow_pickle=False)
    return {key: payload[key] for key in payload.files}


# ============================================================
# DOMAIN-LEVEL RANKING + LOCATION-LEVEL PATTERN FRACTIONS
# ============================================================
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)

    good = np.isfinite(pvals)
    if good.sum() == 0:
        return out

    p = pvals[good]
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    tmp = np.empty_like(q)
    tmp[order] = q
    out[good] = tmp
    return out


def compute_domain_gene_stats(data: List[AlignedTimepoint]) -> pd.DataFrame:
    X1, X2, X3 = data[0].X, data[1].X, data[2].X
    genes = data[0].genes.astype(str).tolist()
    domains = valid_domains(data[0].domains)

    rows = []
    for dom in domains:
        mask = data[0].domains == str(dom)
        nloc = int(mask.sum())
        if nloc < MIN_LOCATIONS_PER_DOMAIN:
            print(f"[WARN] Skip domain {dom}: only {nloc} matched locations.")
            continue

        A = X1[mask]
        B = X2[mask]
        C = X3[mask]

        mean1 = A.mean(axis=0)
        mean2 = B.mean(axis=0)
        mean3 = C.mean(axis=0)

        delta12 = mean2 - mean1
        delta23 = mean3 - mean2
        delta13 = mean3 - mean1

        temporal_range = np.max(np.vstack([mean1, mean2, mean3]).T, axis=1) - np.min(np.vstack([mean1, mean2, mean3]).T, axis=1)
        max_abs_mean_delta = np.maximum(np.abs(delta12), np.abs(delta23))

        kw_stat = np.full(len(genes), np.nan, dtype=float)
        kw_p = np.full(len(genes), np.nan, dtype=float)

        inc_frac = np.full(len(genes), np.nan, dtype=float)
        dec_frac = np.full(len(genes), np.nan, dtype=float)
        peak_frac = np.full(len(genes), np.nan, dtype=float)
        trough_frac = np.full(len(genes), np.nan, dtype=float)
        mixed_frac = np.full(len(genes), np.nan, dtype=float)

        for j, gene in enumerate(genes):
            v1 = A[:, j]
            v2 = B[:, j]
            v3 = C[:, j]

            try:
                if not np.allclose(np.concatenate([v1, v2, v3]), np.concatenate([v1, v2, v3])[0]):
                    stat, p = kruskal(v1, v2, v3)
                    kw_stat[j] = float(stat)
                    kw_p[j] = float(p)
            except Exception:
                pass

            pat, _ = classify_location_pattern(v1, v2, v3)
            inc_frac[j] = np.mean(pat == "Increase")
            dec_frac[j] = np.mean(pat == "Decrease")
            peak_frac[j] = np.mean(pat == "Mid-stage peak")
            trough_frac[j] = np.mean(pat == "Mid-stage trough")
            mixed_frac[j] = np.mean(pat == "Stable")

        kw_fdr = bh_fdr(kw_p)

        rank_kw = pd.Series(-np.nan_to_num(kw_stat, nan=-np.inf)).rank(method="average", ascending=True).to_numpy()
        rank_range = pd.Series(-temporal_range).rank(method="average", ascending=True).to_numpy()
        rank_delta = pd.Series(-max_abs_mean_delta).rank(method="average", ascending=True).to_numpy()
        priority_score = (rank_kw + rank_range + rank_delta) / 3.0

        for j, gene in enumerate(genes):
            rows.append({
                "domain": str(dom),
                "gene": str(gene),
                "n_locations": nloc,

                "mean_2p5": float(mean1[j]),
                "mean_5p7": float(mean2[j]),
                "mean_17p9": float(mean3[j]),

                "mean_delta_5p7_minus_2p5": float(delta12[j]),
                "mean_delta_17p9_minus_5p7": float(delta23[j]),
                "mean_delta_17p9_minus_2p5": float(delta13[j]),

                "temporal_range": float(temporal_range[j]),
                "max_abs_mean_delta": float(max_abs_mean_delta[j]),

                "kw_stat": float(kw_stat[j]) if np.isfinite(kw_stat[j]) else np.nan,
                "kw_pval": float(kw_p[j]) if np.isfinite(kw_p[j]) else np.nan,
                "kw_fdr": float(kw_fdr[j]) if np.isfinite(kw_fdr[j]) else np.nan,

                "inc_frac": float(inc_frac[j]),
                "dec_frac": float(dec_frac[j]),
                "peak_frac": float(peak_frac[j]),
                "trough_frac": float(trough_frac[j]),
                "mixed_frac": float(mixed_frac[j]),

                "priority_score": float(priority_score[j]),
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("No domain-level gene statistics were computed.")

    df = df.sort_values(
        ["domain", "priority_score", "kw_stat", "max_abs_mean_delta"],
        ascending=[True, True, False, False]
    ).reset_index(drop=True)
    return df


# ============================================================
# PLOTTING
# ============================================================
def scatter_delta_panel(ax, coords, vals, title, cmap="vlag", s=1.5, alpha=0.95, mask=None):
    vals = np.asarray(vals, dtype=float)
    vmax = robust_sym_vmax(vals, q=99.0)

    if mask is None:
        mask = np.ones(coords.shape[0], dtype=bool)

    mask = np.asarray(mask, dtype=bool)
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=vals[mask], cmap=cmap, vmin=-vmax, vmax=vmax,
               s=s, alpha=alpha, linewidths=0, rasterized=True)

    ax.set_title(title,fontsize=18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    apply_axis_orientation(ax)


def scatter_trend_panel(
    ax,
    coords,
    patterns,
    title,
    mask=None,
    s=1.5,
    background_color=BG_COLOR,
    background_alpha=BG_ALPHA,
):
    """Plot a continuous trend map with mixed/near-zero locations in gray."""
    coords = np.asarray(coords, dtype=float)
    patterns = np.asarray(patterns, dtype=object)

    if mask is None:
        mask = np.ones(coords.shape[0], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    plot_mask = mask & np.all(np.isfinite(coords[:, :2]), axis=1)

    # Draw every selected anatomical location first. Locations classified as
    # Mixed include stable and near-zero temporal changes, so this gray layer
    # prevents the map from looking artificially sparse.
    if np.any(plot_mask):
        ax.scatter(
            coords[plot_mask, 0],
            coords[plot_mask, 1],
            color=background_color,
            s=max(float(s), float(BG_SIZE)),
            alpha=background_alpha,
            linewidths=0,
            label="Stable",
            rasterized=True,
            zorder=1,
        )

    # Overlay only locations with a directional temporal pattern.
    for pattern in ["Increase", "Decrease", "Mid-stage peak", "Mid-stage trough","Stable"]:
        pattern_mask = plot_mask & (patterns == pattern)
        if np.any(pattern_mask):
            ax.scatter(
                coords[pattern_mask, 0],
                coords[pattern_mask, 1],
                color=PATTERN_COLORS[pattern],
                s=s,
                alpha=0.92,
                linewidths=0,
                label=pattern,
                rasterized=True,
                zorder=2,
            )

    ax.set_title(title,fontsize=18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    apply_axis_orientation(ax)


def scatter_delta_panel(
    ax,
    coords,
    vals,
    title,
    cmap="vlag",
    s=1.5,
    alpha=0.95,
    mask=None,
):
    vals = np.asarray(vals, dtype=float)

    if mask is None:
        mask = np.ones(coords.shape[0], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    plot_mask = (
        mask
        & np.isfinite(vals)
        & np.all(np.isfinite(coords[:, :2]), axis=1)
    )

    vmax = robust_sym_vmax(vals[plot_mask], q=99.0)

    scatter = ax.scatter(
        coords[plot_mask, 0],
        coords[plot_mask, 1],
        c=vals[plot_mask],
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        s=s,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    apply_axis_orientation(ax)

    return scatter


def plot_gene_triptych(
    data: List[AlignedTimepoint],
    gene: str,
    save_path: str,
    mask: Optional[np.ndarray] = None,
    title_suffix: Optional[str] = None,
    remove_invalid_background: bool = False,
):
    genes = data[0].genes.astype(str).tolist()

    if gene not in genes:
        print(f"[WARN] Gene {gene} not found. Skip figure.")
        return

    gene_index = genes.index(gene)
    coords = data[0].coords

    valid_mask = valid_domain_mask(data[0].domains)

    if mask is None:
        mask = (
            valid_mask
            if remove_invalid_background
            else np.ones(coords.shape[0], dtype=bool)
        )
    else:
        mask = np.asarray(mask, dtype=bool)

        if remove_invalid_background:
            mask = mask & valid_mask

    value_1 = data[0].X[:, gene_index]
    value_2 = data[1].X[:, gene_index]
    value_3 = data[2].X[:, gene_index]

    delta_12 = value_2 - value_1
    delta_23 = value_3 - value_2

    patterns, _ = classify_location_pattern(
        value_1,
        value_2,
        value_3,
    )

    subtitle = (
        title_suffix
        if title_suffix is not None
        else "Whole landscape"
    )

    set_style()

    # Increased width to accommodate two colorbars and the external trend legend.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.5, 4.8),
    )

    scatter_12 = scatter_delta_panel(
        axes[0],
        coords,
        delta_12,
        "5.7m − 2.5m",
        mask=mask,
    )

    scatter_23 = scatter_delta_panel(
        axes[1],
        coords,
        delta_23,
        "17.9m − 5.7m",
        mask=mask,
    )

    scatter_trend_panel(
        axes[2],
        coords,
        patterns,
        "Per-location trend",
        mask=mask,
    )

    # Continuous legends for the two delta maps.
    colorbar_12 = fig.colorbar(
        scatter_12,
        ax=axes[0],
        fraction=0.042,
        pad=0.035,
    )
    colorbar_12.set_label(
        "Expression change",
        fontsize=9,
    )
    colorbar_12.ax.tick_params(labelsize=8)

    colorbar_23 = fig.colorbar(
        scatter_23,
        ax=axes[1],
        fraction=0.042,
        pad=0.035,
    )
    colorbar_23.set_label(
        "Expression change",
        fontsize=9,
    )
    colorbar_23.ax.tick_params(labelsize=8)

    # Use the actual gray background color for Mixed/stable locations.
    pattern_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="none",
            markerfacecolor=(
                BG_COLOR
                if pattern == "Stable"
                else PATTERN_COLORS[pattern]
            ),
            markeredgecolor="none",
            markersize=7,
            label=pattern,
        )
        for pattern in [
            "Increase",
            "Decrease",
            "Mid-stage peak",
            "Mid-stage trough",
            "Stable",
        ]
    ]

    # Put the categorical legend outside the third subplot.
    axes[2].legend(
        handles=pattern_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        title="Pattern",
        borderaxespad=0,
    )

    fig.suptitle(
        f"{gene} | {subtitle}",
        y=0.97,
        fontsize=16,
    )

    # More separation between panels and reserved space on the right.
    fig.subplots_adjust(
        left=0.035,
        right=0.88,
        bottom=0.08,
        top=0.84,
        wspace=0.52,
    )

    fig.savefig(
        save_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    print(f"[SAVE] {save_path}")
    plt.close(fig)
from pathlib import Path
import re

def plot_all_domains_top_genes_summary(
    stats_df: pd.DataFrame,
    save_path: str,
    top_n_show: int = TOP_GENES_PER_DOMAIN,
):
    """
    Plot the top temporal genes for each domain separately.

    Each domain is saved as one PNG file with the same figure size/aspect ratio.

    Parameters
    ----------
    stats_df
        DataFrame containing per-domain temporal gene statistics.

    save_path
        Can be either:

        1. A directory:
               ".../temporal_gene_summary"

           outputs:
               domain_0.png
               domain_1.png
               ...

        2. A PNG path:
               ".../top_temporal_genes.png"

           outputs:
               top_temporal_genes_domain_0.png
               top_temporal_genes_domain_1.png
               ...

    top_n_show
        Number of top-ranked genes shown for each domain.
    """

    set_style()

    # ------------------------------------------------------------------
    # Normalize domain IDs
    # ------------------------------------------------------------------
    plot_df = stats_df.copy()
    plot_df["domain"] = plot_df["domain"].astype(str)

    domains = sort_domains(pd.unique(plot_df["domain"]))

    if len(domains) == 0:
        print("[WARN] No domains available for plotting.")
        return

    # ------------------------------------------------------------------
    # Resolve output naming
    # ------------------------------------------------------------------
    save_path = Path(save_path)

    if save_path.suffix.lower() == ".png":
        out_dir = save_path.parent
        base_stem = save_path.stem
    else:
        out_dir = save_path
        base_stem = None

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plotted_patterns = [
        "Increase",
        "Decrease",
        "Mid-stage peak",
        "Mid-stage trough",
        "Stable",
    ]

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=PATTERN_COLORS[patt],
            lw=6,
            label=patt,
        )
        for patt in plotted_patterns
    ]

    # ------------------------------------------------------------------
    # Fixed figure geometry
    # ------------------------------------------------------------------
    # Preserve the original main-plot geometry exactly and add only extra
    # canvas width on the right for the one-column pattern legend.
    MAIN_FIGSIZE_SINGLE = (4.5, 5.2)
    RIGHT_LEGEND_EXTRA_WIDTH = 1.90

    # Original subplot margins, retained in PHYSICAL units after widening.
    MAIN_LEFT = 0.30
    MAIN_RIGHT = 0.90
    MAIN_BOTTOM = 0.20
    MAIN_TOP = 0.90

    for dom in domains:
        # ==============================================================
        # Prepare one domain
        # ==============================================================

        dfd = plot_df[
            plot_df["domain"] == str(dom)
        ].copy()

        dfd = (
            dfd.sort_values(
                [
                    "priority_score",
                    "kw_stat",
                    "max_abs_mean_delta",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ],
            )
            .head(top_n_show)
            .iloc[::-1]
            .reset_index(drop=True)
        )

        # ==============================================================
        # Make one independent figure
        # ==============================================================

        base_width, base_height = MAIN_FIGSIZE_SINGLE
        figure_width = base_width + RIGHT_LEGEND_EXTRA_WIDTH

        fig = plt.figure(
            figsize=(figure_width, base_height),
        )

        # Keep the main stacked-bar panel at exactly the same physical size
        # and position it had before the canvas was widened.
        ax = fig.add_axes(
            [
                MAIN_LEFT * base_width / figure_width,
                MAIN_BOTTOM,
                (MAIN_RIGHT - MAIN_LEFT) * base_width / figure_width,
                MAIN_TOP - MAIN_BOTTOM,
            ]
        )

        if dfd.empty:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="0.5",
            )

            ax.set_title(
                f"Domain {dom}",
                fontsize=13,
            )

            ax.axis("off")

        else:

            y = np.arange(
                len(dfd)
            )

            left = np.zeros(
                len(dfd),
                dtype=float,
            )

            frac_map = {
                "Increase": pd.to_numeric(
                    dfd["inc_frac"],
                    errors="coerce",
                )
                .fillna(0.0)
                .to_numpy(dtype=float),

                "Decrease": pd.to_numeric(
                    dfd["dec_frac"],
                    errors="coerce",
                )
                .fillna(0.0)
                .to_numpy(dtype=float),

                "Mid-stage peak": pd.to_numeric(
                    dfd["peak_frac"],
                    errors="coerce",
                )
                .fillna(0.0)
                .to_numpy(dtype=float),

                "Mid-stage trough": pd.to_numeric(
                    dfd["trough_frac"],
                    errors="coerce",
                )
                .fillna(0.0)
                .to_numpy(dtype=float),
                "Stable": pd.to_numeric(
                    dfd["mixed_frac"],
                    errors="coerce",
                )
                .fillna(0.0)
                .to_numpy(dtype=float),
            }

            # ----------------------------------------------------------
            # Stacked temporal-pattern fractions
            # ----------------------------------------------------------

            for patt in plotted_patterns:

                vals = frac_map[patt]

                ax.barh(
                    y,
                    vals,
                    left=left,
                    color=PATTERN_COLORS[patt],
                    alpha=0.92,
                    height=0.72,
                    label=patt,
                )

                left += vals

            # ----------------------------------------------------------
            # Gene labels
            # ----------------------------------------------------------

            ax.set_yticks(y)

            ax.set_yticklabels(
                dfd["gene"]
                .astype(str)
                .tolist(),
                fontsize=14

            )
            for tick_label in ax.get_yticklabels():
                tick_label.set_fontstyle("italic")
            if dom in ["1","4","6","10"]:
                ax.set_title(
                    f"Cortex (Domain {dom})",
                    fontsize=18,
                    pad=8,
                )
            elif dom in ["8","11"]:
                ax.set_title(
                    f"Hippocampus (Domain {dom})",
                    fontsize=18,
                    pad=8,
                    x=0.5,
                    ha="center",
                )
            else:
                ax.set_title(
                    f"Domain {dom}",
                    fontsize=18,
                    pad=8,
                )


            ax.set_xlabel(
                "Fraction of matched locations",
                fontsize=12
            )

            # Keep the actual stacked-bar scale at [0, 1].
            ax.set_xlim(
                0.0,
                1.0,
            )

            # ----------------------------------------------------------
            # Δ annotations
            #
            # Put them just outside the axis rather than letting them
            # change the effective PNG bounding box.
            # ----------------------------------------------------------

            # for i, (_, row) in enumerate(dfd.iterrows()):
            #     d12 = pd.to_numeric(
            #         row["mean_delta_5p7_minus_2p5"],
            #         errors="coerce",
            #     )
            #
            #     d23 = pd.to_numeric(
            #         row["mean_delta_17p9_minus_5p7"],
            #         errors="coerce",
            #     )
            #
            #     annotation = (
            #         f"Δ12={d12:.2f}, "
            #         f"Δ23={d23:.2f}"
            #     )
            #
            #     bar_end = float(left[i])
            #
            #     ax.text(
            #         bar_end + 0.015,
            #         i,
            #         annotation,
            #         va="center",
            #         ha="left",
            #         fontsize=10,
            #         color="0.35",
            #         transform=ax.transData,
            #         clip_on=False,
            #     )

            # Optional visual guide for 100%
            ax.axvline(
                1.0,
                color="0.82",
                linewidth=0.7,
                zorder=0,
            )

            # Clean Cell/Nature-style axis
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # ==============================================================
        # Pattern legend on the RIGHT in one column.
        #
        # Every independently saved domain plot contains the same legend so
        # redundant copies can be cropped during multi-panel assembly.
        # The main stacked-bar panel itself is unchanged.
        # ==============================================================
        if dom =='11':
            legend_x = (
                MAIN_RIGHT * base_width
                + 0.12
            ) / figure_width

            fig.legend(
                handles=handles,
                title="Age-associated\n   pattern   ",
                loc="upper left",
                bbox_to_anchor=(legend_x, 0.8),
                ncol=1,
                frameon=False,
                borderaxespad=0.0,
                fontsize=11,
                title_fontsize=11,
                handlelength=1.3,
                handletextpad=0.65,
                labelspacing=0.65,
                columnspacing=0.0,
            )

            # # General x-axis annotation, replacing the per-panel x-label
            # # ----------------------------------------------------------
            # # General x-axis key: a mini stacked bar with equal shares of
            # # every pattern, followed by the explanatory text.
            # # ----------------------------------------------------------
            # key_w = 0.19  # mini-bar width  (figure fraction)
            # key_h = 0.028  # mini-bar height (figure fraction)
            # key_y = 0.40  # mini-bar bottom (figure fraction)
            #
            # ax_key = fig.add_axes([legend_x, key_y, key_w, key_h])
            # n_pat = len(plotted_patterns)
            # ax_key.barh(
            #     [0] * n_pat,
            #     [1.0 / n_pat] * n_pat,
            #     left=[i / n_pat for i in range(n_pat)],
            #     color=[PATTERN_COLORS[p] for p in plotted_patterns],
            #     alpha=0.92,
            #     height=1.0,
            # )
            # ax_key.set_xlim(0, 1)
            # ax_key.set_ylim(-0.5, 0.5)
            # ax_key.axis("off")
            #
            # # End labels "0" and "1" under the mini bar
            # fig.text(legend_x, key_y - 0.012, "0",
            #          ha="center", va="top", fontsize=9, color="0.3")
            # fig.text(legend_x + key_w, key_y - 0.012, "1",
            #          ha="center", va="top", fontsize=9, color="0.3")
            #
            # # Explanatory text under the bar
            # fig.text(
            #     legend_x,
            #     key_y - 0.07,
            #     "Fraction of\nmatched locations",
            #     ha="left",
            #     va="top",
            #     fontsize=11,
            #     color="0.2",
            # )

        # ==============================================================
        # Output filename
        # ==============================================================

        safe_dom = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(dom),
        )

        if base_stem is None:
            domain_save_path = (
                out_dir
                / f"domain_{safe_dom}.png"
            )
        else:
            domain_save_path = (
                out_dir
                / f"{base_stem}_domain_{safe_dom}.png"
            )

        # IMPORTANT:
        # Do NOT use bbox_inches="tight".
        # Otherwise labels/annotations of different lengths can result
        # in slightly different PNG dimensions.
        fig.savefig(
            domain_save_path,
            dpi=FIG_DPI,
            facecolor="white",
        )

        print(
            f"[SAVE] Domain {dom}: "
            f"{domain_save_path}"
        )

        plt.close(fig)


def _select_region_representatives_from_top_domain_genes(
    stats_df: pd.DataFrame,
    group_name: str,
    group_domains: Sequence[str],
    top_n_per_domain: int = TOP_GENES_PER_DOMAIN,
    n_representatives: int = 4,
) -> pd.DataFrame:
    """
    Select region-level plotting genes from the already reported top genes of
    each constituent domain.

    This helper is used only by the plotting function. It does not modify the
    per-domain evaluation or its saved tables.

    Selection procedure
    -------------------
    1. Reproduce the per-domain top-gene ranking used by the summary barplots.
    2. Pool only those top-N candidate rows across the domains in the region.
    3. For each gene, aggregate its pattern fractions and delta support across
       the domains in which it appears among the top-N candidates.
    4. Assign each gene to the one pattern with its largest aggregate
       pattern-and-delta score, preventing one gene from representing multiple
       patterns.
    5. Select the highest-scoring gene for every available pattern.
    6. When fewer than four patterns are represented, add the second-ranked
       genes from represented patterns until exactly four unique genes remain.
    """
    pattern_order = ["Increase", "Decrease", "Mid-stage peak", "Mid-stage trough"]
    fraction_columns = {
        "Increase": "inc_frac",
        "Decrease": "dec_frac",
        "Mid-stage peak": "peak_frac",
        "Mid-stage trough": "trough_frac",
    }

    required_columns = {
        "domain",
        "gene",
        "priority_score",
        "kw_stat",
        "max_abs_mean_delta",
        "mean_delta_5p7_minus_2p5",
        "mean_delta_17p9_minus_5p7",
        *fraction_columns.values(),
    }
    missing = sorted(required_columns.difference(stats_df.columns))
    if missing:
        raise KeyError(
            "stats_df is missing columns required for plotting-only "
            f"representative selection: {missing}"
        )

    domains = [str(domain) for domain in group_domains]
    candidates = stats_df[
        stats_df["domain"].astype(str).isin(domains)
    ].copy()

    if candidates.empty:
        raise ValueError(
            f"No gene statistics found for region {group_name!r} "
            f"and domains {domains}."
        )

    candidates["domain"] = candidates["domain"].astype(str)
    candidates["gene"] = candidates["gene"].astype(str)

    numeric_columns = [
        "priority_score",
        "kw_stat",
        "max_abs_mean_delta",
        "mean_delta_5p7_minus_2p5",
        "mean_delta_17p9_minus_5p7",
        *fraction_columns.values(),
    ]
    for column in numeric_columns:
        candidates[column] = pd.to_numeric(
            candidates[column],
            errors="coerce",
        )

    # Reproduce exactly the top-gene definition used in the per-domain report.
    candidates = (
        candidates.sort_values(
            [
                "domain",
                "priority_score",
                "kw_stat",
                "max_abs_mean_delta",
            ],
            ascending=[True, True, False, False],
        )
        .groupby("domain", group_keys=False, observed=True)
        .head(top_n_per_domain)
        .copy()
    )

    delta_12 = candidates[
        "mean_delta_5p7_minus_2p5"
    ].fillna(0.0).to_numpy(dtype=float)
    delta_23 = candidates[
        "mean_delta_17p9_minus_5p7"
    ].fillna(0.0).to_numpy(dtype=float)

    # Pattern-specific delta support. A large value requires the mean deltas to
    # follow the direction expected for that temporal pattern.
    pattern_delta_support = {
        "Increase": np.maximum(delta_12, 0.0)
        + np.maximum(delta_23, 0.0),
        "Decrease": np.maximum(-delta_12, 0.0)
        + np.maximum(-delta_23, 0.0),
        "Mid-stage peak": np.maximum(delta_12, 0.0)
        + np.maximum(-delta_23, 0.0),
        "Mid-stage trough": np.maximum(-delta_12, 0.0)
        + np.maximum(delta_23, 0.0),
    }

    for pattern in pattern_order:
        fraction_column = fraction_columns[pattern]
        candidates[f"{pattern}_delta_support"] = (
            pattern_delta_support[pattern]
        )
        candidates[f"{pattern}_score"] = (
            candidates[fraction_column].fillna(0.0).to_numpy(dtype=float)
            * pattern_delta_support[pattern]
        )

    # Aggregate the same gene across the region's domains, but only after each
    # domain has contributed its independently selected top-N genes.
    aggregate_spec = {
        "priority_score": "min",
        "kw_stat": "max",
        "max_abs_mean_delta": "max",
        "mean_delta_5p7_minus_2p5": "mean",
        "mean_delta_17p9_minus_5p7": "mean",
        "domain": lambda values: ",".join(
            sorted(pd.unique(values.astype(str)), key=str)
        ),
    }
    for pattern in pattern_order:
        aggregate_spec[fraction_columns[pattern]] = "mean"
        aggregate_spec[f"{pattern}_delta_support"] = "mean"
        aggregate_spec[f"{pattern}_score"] = "mean"

    gene_summary = (
        candidates.groupby("gene", as_index=False, observed=True)
        .agg(aggregate_spec)
        .rename(
            columns={
                "domain": "candidate_domains",
            }
        )
    )

    domain_support = (
        candidates.groupby("gene", observed=True)["domain"]
        .nunique()
        .rename("domain_support")
        .reset_index()
    )
    gene_summary = gene_summary.merge(
        domain_support,
        on="gene",
        how="left",
    )

    score_columns = [
        f"{pattern}_score"
        for pattern in pattern_order
    ]
    score_matrix = gene_summary[
        score_columns
    ].fillna(0.0).to_numpy(dtype=float)

    dominant_index = np.argmax(score_matrix, axis=1)
    gene_summary["pattern_name"] = [
        pattern_order[index]
        for index in dominant_index
    ]
    gene_summary["pattern_score"] = score_matrix[
        np.arange(len(gene_summary)),
        dominant_index,
    ]

    gene_summary["pattern_fraction"] = [
        gene_summary.iloc[row_index][
            fraction_columns[pattern_order[pattern_index]]
        ]
        for row_index, pattern_index in enumerate(dominant_index)
    ]
    gene_summary["pattern_delta_support"] = [
        gene_summary.iloc[row_index][
            f"{pattern_order[pattern_index]}_delta_support"
        ]
        for row_index, pattern_index in enumerate(dominant_index)
    ]

    # Identify the domain in which the selected gene has its strongest evidence
    # for its assigned pattern. This is for annotation only; maps still show the
    # full anatomical region.
    source_domains = []
    for _, gene_row in gene_summary.iterrows():
        gene = str(gene_row["gene"])
        pattern = str(gene_row["pattern_name"])

        gene_domain_rows = candidates[
            candidates["gene"] == gene
        ].sort_values(
            [
                f"{pattern}_score",
                fraction_columns[pattern],
                "max_abs_mean_delta",
                "kw_stat",
                "priority_score",
            ],
            ascending=[False, False, False, False, True],
        )

        source_domains.append(
            str(gene_domain_rows.iloc[0]["domain"])
        )

    gene_summary["source_domain"] = source_domains

    # Rank genes inside their single assigned pattern.
    gene_summary = gene_summary.sort_values(
        [
            "pattern_name",
            "pattern_score",
            "pattern_fraction",
            "pattern_delta_support",
            "domain_support",
            "kw_stat",
            "priority_score",
            "gene",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
        ],
    )
    gene_summary["pattern_rank"] = (
        gene_summary.groupby(
            "pattern_name",
            observed=True,
        ).cumcount()
        + 1
    )

    # First take the strongest gene of every represented temporal pattern.
    selected_rows = []
    selected_genes = set()

    for pattern in pattern_order:
        pattern_candidates = gene_summary[
            gene_summary["pattern_name"] == pattern
        ].sort_values(
            [
                "pattern_rank",
                "pattern_score",
                "pattern_fraction",
                "pattern_delta_support",
            ],
            ascending=[True, False, False, False],
        )

        if pattern_candidates.empty:
            continue

        row = pattern_candidates.iloc[0]
        selected_rows.append(row)
        selected_genes.add(str(row["gene"]))

    # If a pattern is absent, take second-ranked genes from the represented
    # patterns. Rank 2 from each pattern is considered before rank 3, which
    # preserves pattern diversity as much as possible.
    while len(selected_rows) < n_representatives:
        remaining = gene_summary[
            ~gene_summary["gene"].isin(selected_genes)
        ].copy()

        if remaining.empty:
            break

        remaining = remaining.sort_values(
            [
                "pattern_rank",
                "pattern_score",
                "pattern_fraction",
                "pattern_delta_support",
                "domain_support",
                "kw_stat",
                "priority_score",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        )

        row = remaining.iloc[0]
        selected_rows.append(row)
        selected_genes.add(str(row["gene"]))

    if len(selected_rows) < n_representatives:
        raise ValueError(
            f"Only {len(selected_rows)} unique genes could be selected for "
            f"{group_name!r}; {n_representatives} are required."
        )

    selected = pd.DataFrame(
        selected_rows[:n_representatives]
    ).copy()

    pattern_sort = {
        pattern: index
        for index, pattern in enumerate(pattern_order)
    }
    selected["_pattern_order"] = selected[
        "pattern_name"
    ].map(pattern_sort)
    selected = selected.sort_values(
        [
            "_pattern_order",
            "pattern_rank",
            "pattern_score",
        ],
        ascending=[True, True, False],
    ).drop(columns="_pattern_order")

    selected["group_name"] = str(group_name)
    selected["selection_rank"] = np.arange(
        1,
        len(selected) + 1,
    )

    return selected.reset_index(drop=True)


REGION_PLOT_CONFIG = {
    "cortex": {
        "group_name": "Cortex",
        "group_domains": ["1", "4", "6", "10"],
        "point_size": 10,
        "xlim": None,
        "ylim": None,
        "figsize": (4.8, 4.5),
        "equal_aspect": True,
    },
    "hippocampus": {
        "group_name": "Hippocampus",
        "group_domains": ["8", "11"],
        "point_size": 14,
        "xlim": None,
        "ylim": None,
        "figsize": (4.8, 4.5),
        "equal_aspect": True,
    },
}


def _apply_axis_style(
    ax,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    equal_aspect: bool = True,
):
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")


def _region_pattern_fractions(
    patterns: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """Return pooled region-level fractions for every temporal pattern present."""
    pattern_order = ["Increase", "Decrease", "Mid-stage peak", "Mid-stage trough", "Stable"]
    patterns = np.asarray(patterns, dtype=object)
    group_mask = np.asarray(group_mask, dtype=bool)
    valid = group_mask & pd.notna(patterns)

    if not np.any(valid):
        return {pattern: np.nan for pattern in pattern_order}

    region_patterns = patterns[valid]
    return {
        pattern: float(np.mean(region_patterns == pattern))
        for pattern in pattern_order
    }


def _hex_to_rgba01(color, alpha: float = 1.0) -> np.ndarray:
    rgba = np.array(plt.matplotlib.colors.to_rgba(color), dtype=float)
    rgba[3] = float(alpha)
    return rgba



def _render_pattern_pixel_image(
    coords: np.ndarray,
    patterns: np.ndarray,
    mask: np.ndarray,
    background_color: str = BG_COLOR,
) -> tuple:
    """Render selected locations as a pixel-based RGBA image on the regularized grid."""
    coords = np.asarray(coords, dtype=float)
    patterns = np.asarray(patterns, dtype=object)
    mask = np.asarray(mask, dtype=bool)

    valid = mask & np.all(np.isfinite(coords[:, :2]), axis=1)
    coords_valid = coords[valid, :2]
    patterns_valid = patterns[valid]
    if coords_valid.shape[0] == 0:
        raise ValueError("No valid selected locations available for pixel rendering.")

    x_vals = np.unique(np.round(coords_valid[:, 0], 8))
    y_vals = np.unique(np.round(coords_valid[:, 1], 8))
    x_vals.sort()
    y_vals.sort()

    x_to_col = {value: idx for idx, value in enumerate(x_vals)}
    y_to_row = {value: idx for idx, value in enumerate(y_vals)}

    image = np.ones((len(y_vals), len(x_vals), 4), dtype=float)
    image[..., :] = _hex_to_rgba01("white", alpha=1.0)

    for (x, y), pattern in zip(coords_valid, patterns_valid):
        col = x_to_col[np.round(x, 8)]
        row = y_to_row[np.round(y, 8)]
        color = background_color if pattern == "Stable" else PATTERN_COLORS.get(pattern, background_color)
        image[row, col, :] = _hex_to_rgba01(color, alpha=1.0)

    dx = float(np.median(np.diff(x_vals))) if len(x_vals) > 1 else 1.0
    dy = float(np.median(np.diff(y_vals))) if len(y_vals) > 1 else 1.0
    extent = (
        float(x_vals.min() - dx / 2.0),
        float(x_vals.max() + dx / 2.0),
        float(y_vals.min() - dy / 2.0),
        float(y_vals.max() + dy / 2.0),
    )
    return image, extent



def _ordered_region_legend_items(fractions: dict) -> list:
    """Return pattern fractions in one fixed figure-wide color order.

    Panels h/i intentionally omit the pattern names and show only a colored dot
    followed by the percentage. Keeping a fixed order makes those dot-only
    annotations directly comparable across representative genes.
    """
    pattern_order = [
        "Increase",
        "Decrease",
        "Mid-stage peak",
        "Mid-stage trough",
        "Stable",
    ]

    items = []
    for pattern in pattern_order:
        fraction = fractions.get(pattern, np.nan)
        if np.isfinite(fraction) and fraction > 0:
            items.append((pattern, float(fraction)))

    return items


def _region_pattern_fraction(
    patterns: np.ndarray,
    group_mask: np.ndarray,
    pattern_name: str,
) -> float:
    """Compatibility wrapper returning one pooled region-level pattern fraction."""
    return _region_pattern_fractions(patterns, group_mask).get(
        str(pattern_name),
        np.nan,
    )


def _resolve_gene_figure_path(
    save_path: str,
    gene: str,
) -> Path:
    """Convert the historical panel save_path into one output path per gene."""
    save_path = Path(save_path)
    safe_gene = safe_name(gene)

    if save_path.suffix.lower() == ".png":
        out_dir = save_path.parent
        stem = save_path.stem
    else:
        out_dir = save_path
        stem = "representative_gene"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{safe_gene}.png"


def _plot_region_pattern_genes_individually(
    data: List[AlignedTimepoint],
    stats_df: pd.DataFrame,
    group_name: str,
    group_domains: Sequence[str],
    save_path: str,
    top_n_per_domain: int = TOP_GENES_PER_DOMAIN,
    n_representatives: int = 4,
    selection_csv_path: str = None,
    point_size: float = 10,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (4.8, 4.5),
    equal_aspect: bool = True,
):
    """
    Save one temporal-pattern map per representative gene for an anatomical region.

    Only the former last row of the 3 x N panel is retained: the per-location
    temporal-pattern map. The two month-to-month delta maps are not generated.

    The representative gene selection itself is unchanged. For each selected
    gene, the right-side annotation reports all region-present pattern fractions
    using colored dots followed only by percentages; pattern names are omitted.
    """
    selected = _select_region_representatives_from_top_domain_genes(
        stats_df=stats_df,
        group_name=group_name,
        group_domains=group_domains,
        top_n_per_domain=top_n_per_domain,
        n_representatives=n_representatives,
    )

    genes = data[0].genes.astype(str).tolist()
    gene_to_index = {gene: i for i, gene in enumerate(genes)}
    coords = data[0].coords
    group_mask = np.isin(
        data[0].domains.astype(str),
        [str(domain) for domain in group_domains],
    )

    # Compute pooled region-level fractions for every trend of each selected gene.
    # These fractions are location-weighted across all domains in the region.
    pattern_order = ["Increase", "Decrease", "Mid-stage peak", "Mid-stage trough", "Stable"]
    pattern_cache = {}
    fraction_cache = {}

    selected = selected.copy()
    for pattern_name in pattern_order:
        selected[f"region_{pattern_name.lower()}_fraction"] = np.nan

    for row_index, row in selected.iterrows():
        gene = str(row["gene"])

        if gene not in gene_to_index:
            continue

        j = gene_to_index[gene]
        patterns, _ = classify_location_pattern(
            data[0].X[:, j],
            data[1].X[:, j],
            data[2].X[:, j],
        )
        pattern_cache[gene] = patterns

        fractions = _region_pattern_fractions(
            patterns=patterns,
            group_mask=group_mask,
        )
        fraction_cache[gene] = fractions

        for pattern_name, fraction in fractions.items():
            selected.loc[
                row_index,
                f"region_{pattern_name.lower()}_fraction",
            ] = fraction

    # Retain the earlier convenience column for the selected representative
    # pattern, while also exporting every region-level trend fraction above.
    selected["region_pattern_fraction"] = [
        fraction_cache.get(str(row["gene"]), {}).get(
            str(row["pattern_name"]),
            np.nan,
        )
        for _, row in selected.iterrows()
    ]

    if selection_csv_path is None:
        selection_csv_path = os.path.splitext(save_path)[0] + "_selected_genes.csv"
    ensure_dir(os.path.dirname(selection_csv_path) or ".")
    selected.to_csv(selection_csv_path, index=False)
    print(f"[SAVE] {selection_csv_path}")

    set_style()

    for _, row in selected.iterrows():
        gene = str(row["gene"])
        pattern_name = str(row["pattern_name"])

        if gene not in gene_to_index:
            print(f"[WARN] Gene {gene!r} not found in aligned data. Skip figure.")
            continue

        patterns = pattern_cache.get(gene)
        if patterns is None:
            j = gene_to_index[gene]
            patterns, _ = classify_location_pattern(
                data[0].X[:, j],
                data[1].X[:, j],
                data[2].X[:, j],
            )

        # =============================================================
        # Preserve the original spatial-map geometry and add only extra
        # canvas width on the right for dot + percentage annotations.
        # =============================================================
        base_width, base_height = figsize
        right_annotation_extra_width = 1.90
        figure_width = base_width + right_annotation_extra_width

        fig = plt.figure(
            figsize=(figure_width, base_height),
        )

        # Reproduce the original subplot margins in physical units:
        # left=0.04, right=0.98, bottom=0.20, top=0.88.
        # Widening therefore does not resize the spatial map itself.
        ax = fig.add_axes(
            [
                0.04 * base_width / figure_width,
                0.20,
                (0.98 - 0.04) * base_width / figure_width,
                0.88 - 0.20,
            ]
        )

        pixel_image, pixel_extent = _render_pattern_pixel_image(
            coords=coords,
            patterns=patterns,
            mask=group_mask,
            background_color=BG_COLOR,
        )
        ax.imshow(
            pixel_image,
            origin="lower",
            extent=pixel_extent,
            interpolation="nearest",
            aspect="equal",
            zorder=1,
        )
        # ax.set_title(f"{group_name} {gene}", fontsize=18)
        ax.set_title(rf"{group_name} | $\it{{{gene}}}$", fontsize=18)
        _apply_axis_style(
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            equal_aspect=equal_aspect,
        )

        # Spatial maps should not show Cartesian axes, ticks, or spines.
        # The title and legend remain visible.
        ax.set_axis_off()

        # =============================================================
        # Right-side pattern-fraction annotation.
        #
        # Requirement for panels h/i:
        #   - do NOT print pattern names;
        #   - show only a colored dot followed by the percentage;
        #   - keep the annotation outside the spatial map on the right.
        #
        # The color order is fixed by _ordered_region_legend_items(), so one
        # retained panel-g legend can serve as the color key after assembly.
        # =============================================================
        fractions = fraction_cache.get(gene, {})
        legend_handles = []

        for pattern_name_for_legend, fraction in _ordered_region_legend_items(
            fractions
        ):
            color = (
                BG_COLOR
                if pattern_name_for_legend == "Stable"
                else PATTERN_COLORS[pattern_name_for_legend]
            )

            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=color,
                    markeredgecolor="none",
                    markersize=8.0,
                    # Percentage only: no pattern-name text.
                    label=f"{100.0 * fraction:.1f}%",
                )
            )

        if legend_handles:
            # annotation_x = (
            #     0.98 * base_width
            # ) / figure_width
            annotation_x=0.4

            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(annotation_x, 0.01),
                ncol=3,
                frameon=False,
                borderaxespad=0.0,
                fontsize=15.0,
                handlelength=0.8,
                handletextpad=0.55,
                labelspacing=0.8,
                columnspacing=0.0,
            )

            # =============================================================
            # Right-side color key: pattern name + color
            # =============================================================
            key_patterns = [
                "Increase",
                "Decrease",
                "Mid-stage peak",
                "Mid-stage trough",
                "Stable",
            ]

            key_handles = [
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=(
                        BG_COLOR if p == "Stable" else PATTERN_COLORS[p]
                    ),
                    markeredgecolor="none",
                    markersize=9,
                    label=p,
                )
                for p in key_patterns
            ]

            key_x = (0.98 * base_width + 0.12) / figure_width

            fig.legend(
                handles=key_handles,
                title="Age-associated\n   pattern   ",
                loc="center left",
                bbox_to_anchor=(key_x, 0.55),
                ncol=1,
                frameon=False,
                borderaxespad=0.0,
                fontsize=11,
                title_fontsize=11,
                handlelength=0.8,
                handletextpad=0.55,
                labelspacing=0.65,
            )

        gene_save_path = _resolve_gene_figure_path(
            save_path=save_path,
            gene=gene,
        )
        fig.savefig(
            gene_save_path,
            dpi=FIG_DPI,
            facecolor="white",
        )
        representative_fraction = fraction_cache.get(gene, {}).get(
            pattern_name,
            np.nan,
        )
        representative_percent = (
            f"{100.0 * representative_fraction:.1f}%"
            if np.isfinite(representative_fraction)
            else "NA"
        )
        print(
            f"[SAVE] {group_name} | {gene} | {pattern_name} "
            f"({representative_percent}): {gene_save_path}"
        )
        plt.close(fig)


def plot_cortical_layer_pattern_gene_panel(
    data: List[AlignedTimepoint],
    stats_df: pd.DataFrame,
    save_path: str,
    top_n_per_domain: int = TOP_GENES_PER_DOMAIN,
    n_representatives: int = 4,
    selection_csv_path: str = None,
    point_size: Optional[float] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    cfg = REGION_PLOT_CONFIG["cortex"]
    _plot_region_pattern_genes_individually(
        data=data,
        stats_df=stats_df,
        group_name=cfg["group_name"],
        group_domains=cfg["group_domains"],
        save_path=save_path,
        top_n_per_domain=top_n_per_domain,
        n_representatives=n_representatives,
        selection_csv_path=selection_csv_path,
        point_size=cfg["point_size"] if point_size is None else point_size,
        xlim=cfg["xlim"] if xlim is None else xlim,
        ylim=cfg["ylim"] if ylim is None else ylim,
        figsize=cfg["figsize"] if figsize is None else figsize,
        equal_aspect=cfg["equal_aspect"],
    )


def plot_hippocampus_pattern_gene_panel(
    data: List[AlignedTimepoint],
    stats_df: pd.DataFrame,
    save_path: str,
    top_n_per_domain: int = TOP_GENES_PER_DOMAIN,
    n_representatives: int = 4,
    selection_csv_path: str = None,
    point_size: Optional[float] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    cfg = REGION_PLOT_CONFIG["hippocampus"]
    _plot_region_pattern_genes_individually(
        data=data,
        stats_df=stats_df,
        group_name=cfg["group_name"],
        group_domains=cfg["group_domains"],
        save_path=save_path,
        top_n_per_domain=top_n_per_domain,
        n_representatives=n_representatives,
        selection_csv_path=selection_csv_path,
        point_size=cfg["point_size"] if point_size is None else point_size,
        xlim=cfg["xlim"] if xlim is None else xlim,
        ylim=cfg["ylim"] if ylim is None else ylim,
        figsize=cfg["figsize"] if figsize is None else figsize,
        equal_aspect=cfg["equal_aspect"],
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["eval", "plot", "both"],
        default="plot",
        help=(
            "'eval': compute and save domain statistics plus location-level "
            "spatial maps; 'plot': load the saved domain table and generate "
            "figures; 'both': evaluate/save and then plot."
        ),
    )
    parser.add_argument(
        "--spatial-map-genes",
        nargs="+",
        default=SPATIAL_MAP_GENES,
        help=(
            "Genes whose per-location temporal values, deltas, and trend "
            "patterns are persisted under OUTDIR/spatial_maps/."
        ),
    )
    args = parser.parse_args()

    figdir = os.path.join(OUTDIR, "figures")
    tbldir = os.path.join(OUTDIR, "tables")
    spatial_map_dir = os.path.join(OUTDIR, SPATIAL_MAP_DIRNAME)
    stats_csv = os.path.join(tbldir, "domain_gene_stats.csv")

    ensure_dir(tbldir)
    data = load_aligned_timepoints()

    if args.mode in {"eval", "both"}:
        stats_df = compute_domain_gene_stats(data)
        stats_df.to_csv(stats_csv, index=False)
        print(f"[OK] saved: {stats_csv}")

        # Persist the exact location-level products used by whole-spatial
        # delta/trend plotting. Downstream pipeline scripts can now load these
        # NPZ files directly without repeating the analysis.
        save_spatial_gene_maps(
            data=data,
            genes=args.spatial_map_genes,
            outdir=spatial_map_dir,
        )

    if args.mode in {"plot", "both"}:
        ensure_dir(figdir)
        set_style()

        if args.mode == "plot":
            if not os.path.isfile(stats_csv):
                raise FileNotFoundError(
                    f"Missing saved table: {stats_csv}. "
                    "Run with --mode eval or --mode both first."
                )
            stats_df = pd.read_csv(stats_csv)

        marker_dir = os.path.join(figdir, "marker_case_studies")
        ensure_dir(marker_dir)
        # for gene in MARKER_CASE_GENES:
        #     plot_gene_triptych(
        #         data=data,
        #         gene=gene,
        #         save_path=os.path.join(
        #             marker_dir,
        #             f"whole_landscape_{safe_name(gene)}.png",
        #         ),
        #         title_suffix="Whole landscape (valid domains only)",
        #         remove_invalid_background=True,
        #     )
        #
        # plot_all_domains_top_genes_summary(
        #     stats_df=stats_df,
        #     save_path=os.path.join(
        #         figdir,
        #         "all_domains_top_genes_summary.png",
        #     ),
        #     top_n_show=TOP_GENES_PER_DOMAIN,
        # )
        #
        selected_dir = os.path.join(
            figdir,
            "selected_domain_spatial_maps",
        )
        ensure_dir(selected_dir)

        # Saves four independent cortex trend-map figures, one per representative gene.
        plot_cortical_layer_pattern_gene_panel(
            data=data,
            stats_df=stats_df,
            save_path=os.path.join(selected_dir, "cortical_layers_representative_genes.png"),
            point_size=2,
            xlim=(1024, 0),
            ylim=(0, 1024),  # if your scatter functions do not already invert y
        )

        # Saves four independent hippocampus trend-map figures, one per representative gene.
        plot_hippocampus_pattern_gene_panel(
            data=data,
            stats_df=stats_df,
            save_path=os.path.join(selected_dir, "hippocampus_representative_genes.png"),
            point_size=5,
            xlim=(724, -100),
            ylim=(400, 924),
        )

    print(
        f"[DONE] Mode '{args.mode}' completed. "
        f"Results saved under: {OUTDIR}"
    )


if __name__ == "__main__":
    main()

