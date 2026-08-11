#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid selection of coherent and domain-specific disease-trending genes.

Purpose
-------
A single whole-domain signed mean is easy to interpret, but it can miss genes
with spatially localized or mixed-sign disease responses. Conversely, ranking
only from positive/negative differential components can overemphasize derived
domain contrasts and makes the final heatmap harder to interpret.

This script separates GENE SELECTION from HEATMAP DISPLAY:

1. Coherent-response track
   Rank genes from the ORIGINAL domain-mean disease-WT delta using the
   strongest response observed across the three ages:

       coherent_score
         = max_t(abs(domain_mean_delta))

   This retains genes with a strong domain-level disease response at at least
   one age, while keeping the final heatmap directly interpretable.

2. Domain-specific signed-component track
   At the location level, separate positive and negative deltas:

       positive_component = mean(max(delta, 0))
       negative_component = mean(min(delta, 0))

   The means are over ALL locations in the domain, so each component contains
   both prevalence and magnitude.

   For each target domain, subtract an equal-domain-weight reference formed
   from ALL OTHER annotated domains available at that age:

       differential_positive
         = target_positive - mean(other-domain positive)

       differential_negative
         = target_negative - mean(other-domain negative)

   For each age, combine the positive and negative differential components:

       heterogeneous_signal_t
         = abs(differential_positive_t)
           + abs(differential_negative_t)

   Then rank genes by the strongest combined directional response across ages:

       heterogeneous_score
         = max_t(heterogeneous_signal_t)

   This favors genes whose positive and/or negative location-level disease
   response is unusually strong in the target domain and avoids cancellation
   between opposite-signed subregions.

3. Hybrid selection
   Select a fixed number of coherent genes first, then select heterogeneous
   genes after excluding genes already chosen by the coherent track. This
   creates two non-overlapping, explicitly labeled marker blocks.

4. Main heatmap
   Plot ONLY the original domain-mean disease-WT delta for both marker blocks.
   Coherent and heterogeneous rows use separate symmetric color scales so
   strong coherent markers do not visually compress heterogeneous markers.
   Colored left-side brackets and labels identify the two marker classes.

This gives a directly interpretable main figure while allowing both:
- coherent domain-wide genes; and
- domain-specific / spatially heterogeneous genes

to enter the displayed set through a predefined, reproducible rule.

Important
---------
- Input h5ads are disease/normal virtual-tissue pairs; signed deltas are computed on the fly.
- Raw/count-like inputs are library-size normalized per location before subtraction.
- LOG1P_BEFORE_ANALYSIS controls whether log1p follows normalization before subtraction.
- The canonical spatial delta comes from spatial_delta_utils.
- No additional clipping, smoothing, or outlier removal is applied.
- The percentile color limit affects visualization only.
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from spatial_delta_utils import compute_spatial_delta_from_files


# ============================================================
# CONFIG
# ============================================================

DISEASE_NORMAL_FILES: List[Tuple[str, str, str]] = [
    (
        "2.5m",
        "/media/huifang/data/registration/xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_2.500.h5ad",
        "/media/huifang/data/registration/xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_102.500.h5ad",
    ),
    (
        "5.7m",
        "/media/huifang/data/registration/xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_5.700.h5ad",
        "/media/huifang/data/registration/xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_105.700.h5ad",
    ),
    (
        "13.4m",
        "/media/huifang/data/registration/xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_13.400.h5ad",
        "/media/huifang/data/registration/xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_113.400.h5ad",
    ),
]

# Unified count preprocessing used by all disease-related analyses.
# raw counts -> per-location library-size normalization -> optional log1p
#            -> disease - normal
LIBRARY_SIZE_NORMALIZE = True
NORMALIZATION_TARGET_SUM = 253
LOG1P_BEFORE_ANALYSIS = True
DELTA_USE_COMMON_VALID = True
COORDINATE_DECIMALS = 5

AGE_DISPLAY_LABELS = {
    "2.5m": r"$\Delta$ 2.5 m",
    "5.7m": r"$\Delta$ 5.7 m",
    "13.4m": r"$\Delta$ 13.4 m",
}

OUTDIR = (
    "/media/huifang/data/registration/paper_figure/"
    "Figure6_disease/trend_heatmap_hybrid"
)

DOMAIN_KEY = "axis_joint_domain"
VALID_KEY = "valid"

SELECTED_DOMAINS: Optional[List[str]] = None
MIN_SPOTS_PER_DOMAIN = 20

# Initial gene filtering.
MANUAL_GENES: Optional[List[str]] = None
MIN_GLOBAL_NONZERO_FRAC = 0.005

# IMPORTANT:
# None means keep ALL common genes passing the nonzero filter.
# This avoids globally preselecting only high-amplitude genes and thereby
# dropping region-specific candidates before domain-level analysis.
MAX_GENES_FOR_ANALYSIS: Optional[int] = None

# Final number of displayed genes per domain.
TOP_GENES_PER_DOMAIN = 20

# Fixed quotas for the two independent biological response modes.
N_COHERENT_GENES = 10
N_DOMAIN_SPECIFIC_GENES = 10

if N_COHERENT_GENES + N_DOMAIN_SPECIFIC_GENES > TOP_GENES_PER_DOMAIN:
    raise ValueError(
        "N_COHERENT_GENES + N_DOMAIN_SPECIFIC_GENES must be <= "
        "TOP_GENES_PER_DOMAIN."
    )

# Ranking is peak-based rather than mean/variance based:
#   coherent:       max(abs(preprocessed signed domain mean)) across ages
#   heterogeneous:  max(abs(diff positive) + abs(diff negative)) across ages

# Heterogeneous track is always target versus ALL OTHER ANNOTATED domains
# available at that age (subject only to MIN_SPOTS_PER_DOMAIN support).
# The domains retained for plotting do NOT define the reference set.
MIN_REFERENCE_DOMAINS = 1

DOMAIN_GROUPS = [
    {
        "name": "Cortex",
        "domains": ["1", "4", "6", "9", "10"],
    },
    {
        "name": "Hippocampus",
        "domains": ["8", "11"],
    },
    {
        "name": "Thalamus",
        "domains": ["0"],
    },
    {
        "name": "Hypothalamus",
        "domains": ["3"],
    },
]

# Heatmap: display the ORIGINAL mean disease-WT delta.
HEATMAP_CMAP = sns.color_palette("vlag", as_cmap=True)
COLOR_SCALE_PERCENTILE: Optional[float] = 99.0
CELL_EDGE_COLOR = "white"
CELL_EDGE_WIDTH = 0.7

# Visually separate the two marker classes in each heatmap panel.
SHOW_MARKER_CLASS_LABELS = True
MARKER_CLASS_LABEL_FONTSIZE = 7.5

# Muted publication-style annotation colors. These colors are used only for
# marker-class labels/side headers, not for the heatmap values themselves.
COHERENT_ANNOTATION_COLOR = "#2F6F8F"
HETEROGENEOUS_ANNOTATION_COLOR = "#A45A3A"
MARKER_CLASS_HEADER_WIDTH = 3.2
MARKER_CLASS_HEADER_TEXT_OFFSET = 0.075
MARKER_CLASS_HEADER_LINE_OFFSET = 0.012

# Keep the same panel width/height across all anatomical groups by matching
# the cortex layout width. Groups with fewer domains leave later panels blank
# rather than stretching each heatmap.
FIXED_DOMAIN_SLOTS = max(len(group["domains"]) for group in DOMAIN_GROUPS)
PANEL_WIDTH = 2.65
PANEL_HEIGHT = 7
TOTAL_DOMAIN_SLOTS = sum(len(group["domains"]) for group in DOMAIN_GROUPS)

FIG_DPI = 400
SAVE_PNG = True
SAVE_PDF = False

# Saved evaluation outputs used by plot-only mode.
ALL_SUMMARY_CSV = os.path.join(
    OUTDIR, "domain_gene_age_raw_components_all_domains.csv"
)
VIS_SUMMARY_CSV = os.path.join(
    OUTDIR, "domain_gene_age_raw_components_visualization_domains.csv"
)
SPECIFIC_CSV = os.path.join(
    OUTDIR, "domain_gene_age_domain_specific_components.csv"
)
RANKING_CSV = os.path.join(
    OUTDIR, "hybrid_domain_gene_rankings.csv"
)
SELECTED_CSV = os.path.join(
    OUTDIR, "hybrid_selected_genes.csv"
)
AUDIT_CSV = os.path.join(
    OUTDIR, "hybrid_selected_genes_audit.csv"
)
ANALYSIS_CONFIG_JSON = os.path.join(
    OUTDIR, "analysis_config.json"
)


# ============================================================
# STYLE
# ============================================================

def set_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def domain_sort_key(domain: str):
    try:
        return (0, float(domain))
    except (TypeError, ValueError):
        return (1, str(domain))


def rename_cracdl_symbol(adata):
    """Directly rename 2010300C02Rik to Cracdl."""
    old_symbol = "2010300C02Rik"
    new_symbol = "Cracdl"

    if old_symbol not in adata.var_names:
        return adata

    if new_symbol in adata.var_names:
        raise ValueError(
            f"Cannot rename {old_symbol} to {new_symbol}: "
            f"{new_symbol} already exists in adata.var_names."
        )

    adata.var_names = pd.Index(
        [new_symbol if gene == old_symbol else gene for gene in adata.var_names]
    )
    return adata


def compute_delta_h5ad(
    disease_path: str,
    normal_path: str,
):
    """Compute the signed spatial disease-minus-WT delta on the fly."""
    adata = compute_spatial_delta_from_files(
        disease_path=disease_path,
        normal_path=normal_path,
        normalize_total=LIBRARY_SIZE_NORMALIZE,
        target_sum=NORMALIZATION_TARGET_SUM,
        use_log1p=LOG1P_BEFORE_ANALYSIS,
        valid_key=VALID_KEY,
        spatial_key="spatial",
        domain_key=DOMAIN_KEY,
        genes=None,
        use_common_valid=DELTA_USE_COMMON_VALID,
        coordinate_decimals=COORDINATE_DECIMALS,
        return_result=False,
    )
    adata = rename_cracdl_symbol(adata)

    if DOMAIN_KEY not in adata.obs.columns:
        raise KeyError(
            f"Missing adata.obs[{DOMAIN_KEY!r}] after spatial-delta calculation."
        )

    return adata


def get_domain_labels(adata) -> np.ndarray:
    return adata.obs[DOMAIN_KEY].astype(str).to_numpy()


def extract_gene_matrix(
    adata,
    genes: Sequence[str],
) -> np.ndarray:
    """
    Extract selected genes without normalization or transformation.
    """
    genes = list(genes)

    missing = [gene for gene in genes if gene not in adata.var_names]
    if missing:
        raise KeyError(f"Missing genes: {missing[:5]}")

    indices = adata.var_names.get_indexer(genes)
    matrix = adata.X[:, indices]

    if sp.issparse(matrix):
        matrix = matrix.toarray()

    return np.asarray(matrix, dtype=np.float32)


# ============================================================
# GENE / DOMAIN SELECTION
# ============================================================

def get_common_genes(
    ad_dict: Dict[str, "AnnData"],
) -> List[str]:
    common = None

    for adata in ad_dict.values():
        genes = set(adata.var_names.astype(str))
        common = genes if common is None else common.intersection(genes)

    return sorted(common) if common is not None else []


def select_genes_for_analysis(
    ad_dict: Dict[str, "AnnData"],
) -> List[str]:
    """
    Keep common genes with sufficient global nonzero fraction.

    By default, DO NOT globally truncate by mean absolute delta. That kind of
    preselection can remove spatially/heterogeneous genes before the actual
    domain analysis.
    """
    common_genes = get_common_genes(ad_dict)

    if not common_genes:
        raise ValueError("No genes are shared by all delta files.")

    if MANUAL_GENES is not None:
        selected = [gene for gene in MANUAL_GENES if gene in common_genes]
        if not selected:
            raise ValueError("None of MANUAL_GENES are shared by all files.")
        return selected

    nonzero_values = []
    mean_abs_values = []

    for adata in ad_dict.values():
        matrix = extract_gene_matrix(adata, common_genes)
        nonzero_values.append(np.mean(matrix != 0, axis=0))
        mean_abs_values.append(np.mean(np.abs(matrix), axis=0))

    nonzero_fraction = np.nanmean(
        np.vstack(nonzero_values),
        axis=0,
    )
    mean_abs_delta = np.nanmean(
        np.vstack(mean_abs_values),
        axis=0,
    )

    gene_table = pd.DataFrame(
        {
            "gene": common_genes,
            "nonzero_fraction": nonzero_fraction,
            "global_mean_abs_delta": mean_abs_delta,
        }
    )

    gene_table = gene_table[
        gene_table["nonzero_fraction"] >= MIN_GLOBAL_NONZERO_FRAC
    ].copy()

    # Only apply an optional computational cap when explicitly requested.
    if MAX_GENES_FOR_ANALYSIS is not None:
        gene_table = (
            gene_table.sort_values(
                ["global_mean_abs_delta", "gene"],
                ascending=[False, True],
                kind="mergesort",
            )
            .head(MAX_GENES_FOR_ANALYSIS)
        )

    selected = gene_table["gene"].tolist()

    if not selected:
        raise ValueError("No genes passed the initial nonzero filter.")

    return selected


def get_all_domains(
    ad_dict: Dict[str, "AnnData"],
) -> List[str]:
    """
    Return every annotated domain present in any input age.

    These domains are used for the heterogeneous target-vs-reference analysis.
    """
    domains = set()

    for adata in ad_dict.values():
        domains.update(get_domain_labels(adata))

    return sorted(
        [domain for domain in domains if domain != "nan"],
        key=domain_sort_key,
    )


def get_visualization_domains(
    ad_dict: Dict[str, "AnnData"],
    domain_groups: Sequence[Dict],
) -> List[str]:
    """
    Domains retained for visualization only.

    SELECTED_DOMAINS, when provided, restricts only the plotted target domains.
    It does NOT restrict domains used as references in the heterogeneous
    differential analysis.

    When SELECTED_DOMAINS is None, use the union of DOMAIN_GROUPS.
    """
    available = set(get_all_domains(ad_dict))

    if SELECTED_DOMAINS is not None:
        requested = [str(domain) for domain in SELECTED_DOMAINS]
    else:
        requested = []
        for group in domain_groups:
            requested.extend(
                [str(domain) for domain in group["domains"]]
            )

    # Preserve requested order while removing duplicates.
    seen = set()
    ordered = []
    for domain in requested:
        if domain in available and domain not in seen:
            ordered.append(domain)
            seen.add(domain)

    return ordered


# ============================================================
# LOCATION-LEVEL -> DOMAIN SUMMARY
# ============================================================

def summarize_domain_matrix(
    domain_matrix: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Summarize signed location-level deltas without losing directional burden.

    For each gene:
        signed_mean       = mean(delta)
        positive_component = mean(max(delta, 0))
        negative_component = mean(min(delta, 0))
        absolute_burden    = mean(abs(delta))

    Note:
        signed_mean = positive_component + negative_component
        absolute_burden = positive_component - negative_component

    cancellation_index is a diagnostic:
        1 - abs(signed_mean) / absolute_burden

    It approaches 1 when positive and negative responses strongly cancel.
    """
    matrix = np.asarray(domain_matrix, dtype=np.float32)

    positive = np.maximum(matrix, 0.0)
    negative = np.minimum(matrix, 0.0)

    signed_mean = np.mean(matrix, axis=0)
    positive_component = np.mean(positive, axis=0)
    negative_component = np.mean(negative, axis=0)
    absolute_burden = np.mean(np.abs(matrix), axis=0)

    positive_fraction = np.mean(matrix > 0, axis=0)
    negative_fraction = np.mean(matrix < 0, axis=0)

    eps = 1e-8
    cancellation_index = 1.0 - (
        np.abs(signed_mean) / (absolute_burden + eps)
    )
    cancellation_index = np.clip(
        cancellation_index,
        0.0,
        1.0,
    )

    return {
        "signed_mean": signed_mean.astype(np.float32),
        "positive_component": positive_component.astype(np.float32),
        "negative_component": negative_component.astype(np.float32),
        "absolute_burden": absolute_burden.astype(np.float32),
        "positive_fraction": positive_fraction.astype(np.float32),
        "negative_fraction": negative_fraction.astype(np.float32),
        "cancellation_index": cancellation_index.astype(np.float32),
    }


def build_domain_gene_summary_table(
    ad_dict: Dict[str, "AnnData"],
    age_order: Sequence[str],
    domains: Sequence[str],
    genes: Sequence[str],
) -> pd.DataFrame:
    """
    Build all preprocessed-domain summaries required by both ranking tracks.
    """
    rows = []

    for age in age_order:
        adata = ad_dict[age]
        domain_labels = get_domain_labels(adata)
        matrix = extract_gene_matrix(adata, genes)

        for domain in domains:
            domain_mask = domain_labels == str(domain)
            n_spots = int(domain_mask.sum())

            if n_spots < MIN_SPOTS_PER_DOMAIN:
                continue

            summary = summarize_domain_matrix(
                matrix[domain_mask]
            )

            rows.extend(
                {
                    "age": age,
                    "domain": str(domain),
                    "gene": gene,
                    "signed_mean": float(
                        summary["signed_mean"][index]
                    ),
                    "positive_component": float(
                        summary["positive_component"][index]
                    ),
                    "negative_component": float(
                        summary["negative_component"][index]
                    ),
                    "absolute_burden": float(
                        summary["absolute_burden"][index]
                    ),
                    "positive_fraction": float(
                        summary["positive_fraction"][index]
                    ),
                    "negative_fraction": float(
                        summary["negative_fraction"][index]
                    ),
                    "cancellation_index": float(
                        summary["cancellation_index"][index]
                    ),
                    "n_spots": n_spots,
                }
                for index, gene in enumerate(genes)
            )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "No domain-gene summaries were generated."
        )

    return result


def retain_complete_domains(
    summary_df: pd.DataFrame,
    age_order: Sequence[str],
) -> pd.DataFrame:
    required_ages = set(age_order)
    keep_domains = []

    for domain, subset in summary_df.groupby("domain"):
        if required_ages.issubset(set(subset["age"].unique())):
            keep_domains.append(domain)

    result = summary_df[
        summary_df["domain"].isin(keep_domains)
    ].copy()

    if result.empty:
        raise ValueError(
            "No domain is sufficiently represented at every age."
        )

    return result


# ============================================================
# TARGET-vs-ALL-OTHER DOMAIN SPECIFICITY
# ============================================================

def build_domain_specific_component_table(
    summary_df: pd.DataFrame,
    age_order: Sequence[str],
    target_domains: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Compare each requested target domain with the equal-weight mean of ALL
    OTHER annotated domains available at the same age.

    Crucially, `summary_df` should contain summaries for ALL annotated domains.
    `target_domains` restricts only which targets are evaluated/exported; it
    never restricts the reference pool.

    Domain means are averaged with equal domain weights. Reference locations
    are not pooled, preventing large domains from dominating solely because
    they contain more locations.
    """
    value_columns = [
        "positive_component",
        "negative_component",
        "absolute_burden",
        "positive_fraction",
        "negative_fraction",
        "signed_mean",
        "cancellation_index",
    ]

    requested_targets = (
        None
        if target_domains is None
        else {str(domain) for domain in target_domains}
    )

    rows = []

    for age in age_order:
        age_df = summary_df[
            summary_df["age"] == age
        ].copy()

        available_domains = sorted(
            age_df["domain"].astype(str).unique(),
            key=domain_sort_key,
        )

        if len(available_domains) <= MIN_REFERENCE_DOMAINS:
            raise ValueError(
                f"Not enough domains at {age} for target-vs-reference analysis."
            )

        if requested_targets is None:
            targets_this_age = available_domains
        else:
            targets_this_age = [
                domain
                for domain in available_domains
                if domain in requested_targets
            ]

        indexed = age_df.set_index(["domain", "gene"])

        for target_domain in targets_this_age:
            # Reference = ALL other supported domains at this age.
            reference_domains = [
                domain
                for domain in available_domains
                if domain != target_domain
            ]

            if len(reference_domains) < MIN_REFERENCE_DOMAINS:
                continue

            target = indexed.loc[target_domain][value_columns]

            reference = (
                age_df[
                    age_df["domain"].isin(reference_domains)
                ]
                .groupby("gene", as_index=True)[value_columns]
                .mean()
            )

            common_genes = target.index.intersection(
                reference.index
            )

            target = target.loc[common_genes]
            reference = reference.loc[common_genes]

            diff_positive = (
                target["positive_component"]
                - reference["positive_component"]
            )
            diff_negative = (
                target["negative_component"]
                - reference["negative_component"]
            )
            diff_absolute_burden = (
                target["absolute_burden"]
                - reference["absolute_burden"]
            )

            rows.extend(
                {
                    "age": age,
                    "domain": target_domain,
                    "gene": gene,
                    "target_signed_mean": float(
                        target.loc[gene, "signed_mean"]
                    ),
                    "reference_signed_mean": float(
                        reference.loc[gene, "signed_mean"]
                    ),
                    "target_positive_component": float(
                        target.loc[gene, "positive_component"]
                    ),
                    "reference_positive_component": float(
                        reference.loc[gene, "positive_component"]
                    ),
                    "differential_positive_component": float(
                        diff_positive.loc[gene]
                    ),
                    "target_negative_component": float(
                        target.loc[gene, "negative_component"]
                    ),
                    "reference_negative_component": float(
                        reference.loc[gene, "negative_component"]
                    ),
                    "differential_negative_component": float(
                        diff_negative.loc[gene]
                    ),
                    "target_absolute_burden": float(
                        target.loc[gene, "absolute_burden"]
                    ),
                    "reference_absolute_burden": float(
                        reference.loc[gene, "absolute_burden"]
                    ),
                    "differential_absolute_burden": float(
                        diff_absolute_burden.loc[gene]
                    ),
                    "target_positive_fraction": float(
                        target.loc[gene, "positive_fraction"]
                    ),
                    "target_negative_fraction": float(
                        target.loc[gene, "negative_fraction"]
                    ),
                    "target_cancellation_index": float(
                        target.loc[gene, "cancellation_index"]
                    ),
                    "reference_domains": ",".join(reference_domains),
                    "n_reference_domains": len(reference_domains),
                }
                for gene in common_genes
            )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "No domain-specific component rows were generated."
        )

    return result


# ============================================================
# TWO INDEPENDENT GENE RANKING TRACKS
# ============================================================

def rank_genes_hybrid(
    domain_summary_df: pd.DataFrame,
    domain_specific_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute two independent ranking scores.

    Coherent marker score
    ---------------------
    Rank by the largest absolute ORIGINAL domain-mean delta across ages:

        coherent_score = max_t(abs(signed_mean_t))

    Heterogeneous marker score
    --------------------------
    Differential analysis is target domain versus the equal-weight mean of
    ALL OTHER retained domains.

    For each age:

        heterogeneous_signal_t
            = abs(differential_positive_component_t)
              + abs(differential_negative_component_t)

    Rank by:

        heterogeneous_score = max_t(heterogeneous_signal_t)

    Thus, positive and negative location-level responses both contribute,
    without allowing opposite signs to cancel.
    """
    coherent_rows = []

    for gene, gene_df in domain_summary_df.groupby("gene", sort=False):
        values = gene_df["signed_mean"].to_numpy(dtype=float)
        abs_values = np.abs(values)

        peak_index = int(np.nanargmax(abs_values))
        peak_row = gene_df.iloc[peak_index]

        coherent_rows.append(
            {
                "gene": gene,
                "coherent_score": float(abs_values[peak_index]),
                "coherent_peak_signed_delta": float(
                    peak_row["signed_mean"]
                ),
                "coherent_peak_age": str(
                    peak_row["age"]
                ),
                "coherent_mean_abs_delta": float(
                    np.nanmean(abs_values)
                ),
                "mean_absolute_burden": float(
                    np.nanmean(
                        gene_df["absolute_burden"].to_numpy(dtype=float)
                    )
                ),
                "mean_cancellation_index": float(
                    np.nanmean(
                        gene_df["cancellation_index"].to_numpy(dtype=float)
                    )
                ),
            }
        )

    coherent = pd.DataFrame(coherent_rows)

    specific_work = domain_specific_df.copy()
    specific_work["heterogeneous_signal"] = (
        np.abs(
            specific_work["differential_positive_component"].to_numpy(
                dtype=float
            )
        )
        + np.abs(
            specific_work["differential_negative_component"].to_numpy(
                dtype=float
            )
        )
    )

    heterogeneous_rows = []

    for gene, gene_df in specific_work.groupby("gene", sort=False):
        signal = gene_df["heterogeneous_signal"].to_numpy(dtype=float)
        peak_index = int(np.nanargmax(signal))
        peak_row = gene_df.iloc[peak_index]

        heterogeneous_rows.append(
            {
                "gene": gene,
                "heterogeneous_score": float(signal[peak_index]),
                "heterogeneous_peak_age": str(
                    peak_row["age"]
                ),
                "heterogeneous_peak_positive": float(
                    peak_row["differential_positive_component"]
                ),
                "heterogeneous_peak_negative": float(
                    peak_row["differential_negative_component"]
                ),
                "heterogeneous_mean_signal": float(
                    np.nanmean(signal)
                ),
                "mean_positive_burden_enrichment": float(
                    np.nanmean(
                        gene_df["differential_absolute_burden"].to_numpy(
                            dtype=float
                        )
                    )
                ),
            }
        )

    heterogeneous = pd.DataFrame(heterogeneous_rows)

    ranking = coherent.merge(
        heterogeneous,
        on="gene",
        how="inner",
        validate="one_to_one",
    )

    ranking["coherent_rank"] = (
        ranking["coherent_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    ranking["heterogeneous_rank"] = (
        ranking["heterogeneous_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Percentile fields are diagnostics only; they are not used for selection.
    ranking["coherent_percentile"] = (
        ranking["coherent_score"]
        .rank(method="average", pct=True)
    )
    ranking["heterogeneous_percentile"] = (
        ranking["heterogeneous_score"]
        .rank(method="average", pct=True)
    )

    ranking = ranking.sort_values(
        [
            "coherent_rank",
            "heterogeneous_rank",
            "gene",
        ],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return ranking

def select_hybrid_genes(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select two explicit, non-overlapping marker blocks.

    1. Take the top N_COHERENT_GENES by coherent_score.
    2. Exclude those genes from the heterogeneous candidate pool.
    3. Take the top N_DOMAIN_SPECIFIC_GENES by heterogeneous_score.
    4. Display coherent genes first, followed by heterogeneous genes.

    A gene that is highly ranked by both tracks is assigned to the coherent
    block because the coherent selection is performed first.
    """
    coherent_selected = (
        ranking.sort_values(
            ["coherent_score", "gene"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(N_COHERENT_GENES)
        .copy()
    )

    coherent_genes = set(
        coherent_selected["gene"].tolist()
    )

    heterogeneous_candidates = ranking[
        ~ranking["gene"].isin(coherent_genes)
    ].copy()

    heterogeneous_selected = (
        heterogeneous_candidates.sort_values(
            ["heterogeneous_score", "gene"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(N_DOMAIN_SPECIFIC_GENES)
        .copy()
    )

    coherent_selected["selection_source"] = "Coherent"
    coherent_selected["rank_within_source"] = np.arange(
        1,
        len(coherent_selected) + 1,
    )

    heterogeneous_selected["selection_source"] = "Heterogeneous"
    heterogeneous_selected["rank_within_source"] = np.arange(
        1,
        len(heterogeneous_selected) + 1,
    )

    selected = pd.concat(
        [
            coherent_selected,
            heterogeneous_selected,
        ],
        ignore_index=True,
    )

    if len(selected) < TOP_GENES_PER_DOMAIN:
        already_selected = set(selected["gene"])
        remaining = ranking[
            ~ranking["gene"].isin(already_selected)
        ].copy()

        # Fill any missing rows from the stronger of the two percentile ranks.
        remaining["fallback_score"] = np.maximum(
            remaining["coherent_percentile"],
            remaining["heterogeneous_percentile"],
        )

        remaining = remaining.sort_values(
            ["fallback_score", "gene"],
            ascending=[False, True],
            kind="mergesort",
        )

        n_missing = TOP_GENES_PER_DOMAIN - len(selected)
        fill = remaining.head(n_missing).copy()

        # Assign fill genes to the track on which they rank more strongly.
        fill["selection_source"] = np.where(
            fill["coherent_percentile"]
            >= fill["heterogeneous_percentile"],
            "Coherent",
            "Heterogeneous",
        )

        # Rebuild the two blocks after filling.
        selected = pd.concat(
            [selected, fill],
            ignore_index=True,
        )

        selected = pd.concat(
            [
                selected[
                    selected["selection_source"] == "Coherent"
                ].sort_values(
                    ["coherent_score", "gene"],
                    ascending=[False, True],
                    kind="mergesort",
                ),
                selected[
                    selected["selection_source"] == "Heterogeneous"
                ].sort_values(
                    ["heterogeneous_score", "gene"],
                    ascending=[False, True],
                    kind="mergesort",
                ),
            ],
            ignore_index=True,
        )

        selected["rank_within_source"] = (
            selected.groupby(
                "selection_source",
                sort=False,
            ).cumcount()
            + 1
        )

    selected["display_rank"] = np.arange(
        1,
        len(selected) + 1,
    )

    return selected.head(TOP_GENES_PER_DOMAIN)


# ============================================================
# HEATMAP MATRIX: ORIGINAL DISEASE-WT DELTA
# ============================================================

def build_hybrid_heatmap_matrix(
    summary_df: pd.DataFrame,
    specific_df: pd.DataFrame,
    domain: str,
    age_order: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select genes using both tracks but display the original signed_mean delta.
    """
    domain_summary = summary_df[
        summary_df["domain"] == str(domain)
    ].copy()

    domain_specific = specific_df[
        specific_df["domain"] == str(domain)
    ].copy()

    if domain_summary.empty or domain_specific.empty:
        return pd.DataFrame(), pd.DataFrame()

    ranking = rank_genes_hybrid(
        domain_summary_df=domain_summary,
        domain_specific_df=domain_specific,
    )
    selected = select_hybrid_genes(ranking)

    selected_genes = selected["gene"].tolist()

    matrix = (
        domain_summary[
            domain_summary["gene"].isin(selected_genes)
        ]
        .pivot_table(
            index="gene",
            columns="age",
            values="signed_mean",
            aggfunc="mean",
        )
        .reindex(
            index=selected_genes,
            columns=list(age_order),
        )
    )

    return matrix, selected


# ============================================================
# PLOTTING
# ============================================================

def calculate_marker_class_vmax(
    matrices: Sequence[pd.DataFrame],
    selected_tables: Sequence[pd.DataFrame],
    marker_class: str,
) -> float:
    """
    Calculate a group-level symmetric color limit for one marker class only.

    Coherent and heterogeneous markers therefore have independent display
    scales, while all domains within the same anatomical group remain directly
    comparable for a given marker class.
    """
    value_blocks = []

    for matrix, selected in zip(
        matrices,
        selected_tables,
    ):
        source = selected["selection_source"].to_numpy()
        mask = source == marker_class

        if not np.any(mask):
            continue

        values = matrix.to_numpy(dtype=float)[mask, :]
        values = np.abs(values[np.isfinite(values)])

        if values.size:
            value_blocks.append(values)

    if not value_blocks:
        return 1.0

    values = np.concatenate(value_blocks)

    if COLOR_SCALE_PERCENTILE is None:
        vmax = float(np.max(values))
    else:
        vmax = float(
            np.percentile(
                values,
                COLOR_SCALE_PERCENTILE,
            )
        )

    return max(vmax, 1e-8)


def format_gene_labels(
    selected: pd.DataFrame,
) -> List[str]:
    """Return plain gene symbols; marker class is shown separately."""
    return selected["gene"].tolist()


def _draw_marker_class_side_header(
    ax,
    row_start: int,
    row_end_exclusive: int,
    label: str,
    color: str,
) -> None:
    """
    Draw a straight colored side header tightly attached to the RIGHT side of
    the heatmap, plus a vertical class label just outside it.

    The left side is reserved for gene-name annotations.
    """
    if row_end_exclusive <= row_start:
        return

    y_top = row_start - 0.46
    y_bottom = row_end_exclusive - 0.54
    y_center = 0.5 * (y_top + y_bottom)

    transform = ax.get_yaxis_transform()
    x_line = 1.0 + MARKER_CLASS_HEADER_LINE_OFFSET
    x_text = 1.0 + MARKER_CLASS_HEADER_TEXT_OFFSET

    ax.plot(
        [x_line, x_line],
        [y_top, y_bottom],
        transform=transform,
        color=color,
        linewidth=MARKER_CLASS_HEADER_WIDTH,
        solid_capstyle="butt",
        clip_on=False,
        zorder=8,
    )

    if SHOW_MARKER_CLASS_LABELS:
        ax.text(
            x_text,
            y_center,
            label,
            transform=transform,
            rotation=90,
            ha="center",
            va="center",
            fontsize=MARKER_CLASS_LABEL_FONTSIZE,
            fontweight="bold",
            color=color,
            clip_on=False,
        )


def draw_heatmap_panel(
    ax,
    matrix: pd.DataFrame,
    selected: pd.DataFrame,
    domain: str,
    age_order: Sequence[str],
    coherent_vmax: float,
    heterogeneous_vmax: float,
):
    """
    Draw the ORIGINAL mean disease-WT delta with independent color scales for
    coherent and heterogeneous marker blocks.

    Both blocks use the same `vlag` colormap and zero-centered interpretation,
    but each block has its own symmetric vmin/vmax.
    """
    values = matrix.to_numpy(dtype=float)
    sources = selected["selection_source"].to_numpy()

    coherent_mask = sources == "Coherent"
    heterogeneous_mask = sources == "Heterogeneous"

    # Two transparent layers allow different normalizations in one axes.
    coherent_values = np.full_like(
        values,
        np.nan,
        dtype=float,
    )
    heterogeneous_values = np.full_like(
        values,
        np.nan,
        dtype=float,
    )

    coherent_values[coherent_mask, :] = values[coherent_mask, :]
    heterogeneous_values[heterogeneous_mask, :] = values[
        heterogeneous_mask,
        :
    ]

    # Make NaN cells transparent so the two class-specific layers can overlay.
    cmap = HEATMAP_CMAP.copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    coherent_image = ax.imshow(
        np.ma.masked_invalid(coherent_values),
        cmap=cmap,
        norm=Normalize(
            vmin=-coherent_vmax,
            vmax=coherent_vmax,
        ),
        aspect="auto",
        interpolation="nearest",
    )

    heterogeneous_image = ax.imshow(
        np.ma.masked_invalid(heterogeneous_values),
        cmap=cmap,
        norm=Normalize(
            vmin=-heterogeneous_vmax,
            vmax=heterogeneous_vmax,
        ),
        aspect="auto",
        interpolation="nearest",
    )

    n_rows, n_cols = values.shape

    # White cell borders.
    ax.set_xticks(
        np.arange(-0.5, n_cols, 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, n_rows, 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        color=CELL_EDGE_COLOR,
        linewidth=CELL_EDGE_WIDTH,
    )
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        [
            AGE_DISPLAY_LABELS.get(age, age)
            for age in age_order
        ],
        rotation=0,
        ha="center",
    )

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(
        format_gene_labels(selected)
    )

    for tick_label in ax.get_yticklabels():
        tick_label.set_fontstyle("italic")

    ax.tick_params(
        axis="both",
        which="major",
        length=0,
        pad=3,
    )

    n_coherent = int(np.sum(coherent_mask))
    n_heterogeneous = int(np.sum(heterogeneous_mask))

    # The selection function places coherent genes first and heterogeneous
    # genes second, allowing two clear right-side marker-class headers.
    if n_coherent > 0:
        _draw_marker_class_side_header(
            ax=ax,
            row_start=0,
            row_end_exclusive=n_coherent,
            label="Coherent",
            color=COHERENT_ANNOTATION_COLOR,
        )

    if n_heterogeneous > 0:
        _draw_marker_class_side_header(
            ax=ax,
            row_start=n_coherent,
            row_end_exclusive=n_coherent + n_heterogeneous,
            label="Heterogeneous",
            color=HETEROGENEOUS_ANNOTATION_COLOR,
        )

    # A strong two-color boundary makes the class transition immediately clear.
    if n_coherent > 0 and n_heterogeneous > 0:
        divider_y = n_coherent - 0.5

        # Left half of the separator uses coherent color.
        ax.plot(
            [-0.5, (n_cols - 1) / 2.0],
            [divider_y, divider_y],
            color=COHERENT_ANNOTATION_COLOR,
            linewidth=2.2,
            solid_capstyle="butt",
            zorder=9,
        )

        # Right half uses heterogeneous color.
        ax.plot(
            [(n_cols - 1) / 2.0, n_cols - 0.5],
            [divider_y, divider_y],
            color=HETEROGENEOUS_ANNOTATION_COLOR,
            linewidth=2.2,
            solid_capstyle="butt",
            zorder=9,
        )

    ax.set_title(
        f"Domain {domain}",
        fontweight="bold",
        pad=7,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    return coherent_image, heterogeneous_image

def save_combined_figure(
    fig,
    save_dir: str,
    figure_name: str = "all_regions",
) -> None:
    safe_name = (
        figure_name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    if SAVE_PNG:
        path = os.path.join(
            save_dir,
            f"{safe_name}_hybrid_peak_ranked_raw_delta_separate_scales.png",
        )
        fig.savefig(
            path,
            dpi=FIG_DPI,
            bbox_inches="tight",
        )
        print(f"[SAVE] {path}")

    if SAVE_PDF:
        path = os.path.join(
            save_dir,
            f"{safe_name}_hybrid_peak_ranked_raw_delta_separate_scales.pdf",
        )
        fig.savefig(
            path,
            bbox_inches="tight",
        )
        print(f"[SAVE] {path}")



def build_saved_selection_matrix(
    summary_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    group_name: str,
    domain: str,
    age_order: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct one displayed heatmap matrix from saved evaluation outputs.

    No filtering, ranking, or gene selection is performed here.
    """
    selected = selected_df[
        (selected_df["group"].astype(str) == str(group_name))
        & (selected_df["domain"].astype(str) == str(domain))
    ].copy()

    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected = selected.sort_values(
        "display_rank",
        kind="mergesort",
    )
    selected_genes = selected["gene"].astype(str).tolist()

    domain_summary = summary_df[
        summary_df["domain"].astype(str) == str(domain)
    ].copy()

    matrix = (
        domain_summary[
            domain_summary["gene"].astype(str).isin(selected_genes)
        ]
        .pivot_table(
            index="gene",
            columns="age",
            values="signed_mean",
            aggfunc="mean",
        )
        .reindex(
            index=selected_genes,
            columns=list(age_order),
        )
    )
    return matrix, selected
#
#
# def plot_grouped_hybrid_heatmaps(
#     summary_df: pd.DataFrame,
#     selected_df: pd.DataFrame,
#     age_order: Sequence[str],
#     domain_groups: Sequence[Dict],
#     save_dir: str,
# ) -> None:
#     """
#     Plot the saved hybrid-selection results only.
#
#     The plotting path does not recompute ranking, domain specificity,
#     or marker selection.
#     """
#     set_figure_style()
#     ensure_dir(save_dir)
#
#     matrices = []
#     selected_per_domain = []
#     kept_domains = []
#     group_ranges = []
#
#     available_domains = set(
#         summary_df["domain"].astype(str).unique()
#     )
#
#     panel_index = 0
#     for group in domain_groups:
#         group_name = str(group["name"])
#         group_start = panel_index
#
#         for domain in [str(x) for x in group["domains"]]:
#             if domain not in available_domains:
#                 continue
#
#             matrix, selected = build_saved_selection_matrix(
#                 summary_df=summary_df,
#                 selected_df=selected_df,
#                 group_name=group_name,
#                 domain=domain,
#                 age_order=age_order,
#             )
#             if matrix.empty:
#                 continue
#
#             matrices.append(matrix)
#             selected_per_domain.append(selected)
#             kept_domains.append(domain)
#             panel_index += 1
#
#         if panel_index > group_start:
#             group_ranges.append(
#                 (group_name, group_start, panel_index)
#             )
#
#     if not matrices:
#         print("[WARN] No valid heatmap panels were generated.")
#         return
#
#     coherent_vmax = calculate_marker_class_vmax(
#         matrices=matrices,
#         selected_tables=selected_per_domain,
#         marker_class="Coherent",
#     )
#     heterogeneous_vmax = calculate_marker_class_vmax(
#         matrices=matrices,
#         selected_tables=selected_per_domain,
#         marker_class="Heterogeneous",
#     )
#
#     n_domains = len(matrices)
#     n_slots = max(TOTAL_DOMAIN_SLOTS, n_domains)
#     figure_width = max(
#         8.0,
#         PANEL_WIDTH * n_slots + 1.35,
#     )
#
#     fig = plt.figure(
#         figsize=(figure_width, PANEL_HEIGHT)
#     )
#
#     grid = fig.add_gridspec(
#         1,
#         n_slots + 1,
#         width_ratios=[1.0] * n_slots + [0.085],
#         left=0.035,
#         right=0.97,
#         top=0.80,
#         bottom=0.12,
#         wspace=0.6,
#     )
#
#     axes = [
#         fig.add_subplot(grid[0, index])
#         for index in range(n_slots)
#     ]
#
#     cbar_grid = grid[0, -1].subgridspec(
#         2, 1, hspace=0.48
#     )
#     coherent_cbar_ax = fig.add_subplot(cbar_grid[0, 0])
#     heterogeneous_cbar_ax = fig.add_subplot(cbar_grid[1, 0])
#
#     for ax, domain, matrix, selected in zip(
#         axes[:n_domains],
#         kept_domains,
#         matrices,
#         selected_per_domain,
#     ):
#         draw_heatmap_panel(
#             ax=ax,
#             matrix=matrix,
#             selected=selected.drop(
#                 columns=["group", "domain"],
#                 errors="ignore",
#             ),
#             domain=domain,
#             age_order=age_order,
#             coherent_vmax=coherent_vmax,
#             heterogeneous_vmax=heterogeneous_vmax,
#         )
#
#     for ax in axes[n_domains:]:
#         ax.axis("off")
#
#     coherent_mappable = ScalarMappable(
#         norm=Normalize(
#             vmin=-coherent_vmax,
#             vmax=coherent_vmax,
#         ),
#         cmap=HEATMAP_CMAP,
#     )
#     heterogeneous_mappable = ScalarMappable(
#         norm=Normalize(
#             vmin=-heterogeneous_vmax,
#             vmax=heterogeneous_vmax,
#         ),
#         cmap=HEATMAP_CMAP,
#     )
#
#     coherent_colorbar = fig.colorbar(
#         coherent_mappable,
#         cax=coherent_cbar_ax,
#     )
#     coherent_colorbar.set_label(
#         r"Mean disease$-$WT $\Delta$",
#         rotation=90,
#         labelpad=8,
#         fontsize=8.5,
#     )
#     coherent_colorbar.ax.tick_params(
#         labelsize=7.5,
#         length=2.2,
#     )
#     coherent_colorbar.outline.set_visible(False)
#
#     heterogeneous_colorbar = fig.colorbar(
#         heterogeneous_mappable,
#         cax=heterogeneous_cbar_ax,
#     )
#     heterogeneous_colorbar.set_label(
#         r"Mean disease$-$WT $\Delta$",
#         rotation=90,
#         labelpad=8,
#         fontsize=8.5,
#     )
#     heterogeneous_colorbar.ax.tick_params(
#         labelsize=7.5,
#         length=2.2,
#     )
#     heterogeneous_colorbar.outline.set_visible(False)
#
#     for group_name, start_idx, end_idx in group_ranges:
#         left_box = axes[start_idx].get_position()
#         right_box = axes[end_idx - 1].get_position()
#         x_center = 0.5 * (left_box.x0 + right_box.x1)
#         y_top = max(left_box.y1, right_box.y1)
#
#         fig.text(
#             x_center,
#             y_top + 0.055,
#             group_name,
#             ha="center",
#             va="bottom",
#             fontsize=13.5,
#             fontweight="bold",
#         )
#
#     save_combined_figure(
#         fig=fig,
#         save_dir=save_dir,
#         figure_name="all_regions",
#     )
#     plt.close(fig)

def plot_individual_domain_hybrid_heatmaps(
    summary_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    age_order: Sequence[str],
    domain_groups: Sequence[Dict],
    save_dir: str,
) -> None:
    """
    Plot each anatomical domain as one independent figure.

    Important
    ---------
    - Every figure has exactly the same figsize and layout.
    - Coherent and heterogeneous marker color limits are calculated globally
      across ALL plotted domains, preserving cross-domain comparability.
    - Gene selection/order is read from the saved selected_df; nothing is
      re-ranked or re-selected here.
    - Each figure is saved as:

          [region_name]_[domain_id].png

      e.g.
          Cortex_1.png
          Hippocampus_8.png
          Thalamus_0.png
    """

    set_figure_style()
    ensure_dir(save_dir)

    # ============================================================
    # 1. Collect all domain matrices first
    #
    # This is necessary so all separately saved figures use the
    # SAME coherent and heterogeneous color scales.
    # ============================================================

    plot_records = []

    available_domains = set(
        summary_df["domain"]
        .astype(str)
        .unique()
    )

    for group in domain_groups:

        region_name = str(
            group["name"]
        )

        for domain in [
            str(x)
            for x in group["domains"]
        ]:

            if domain not in available_domains:
                continue

            matrix, selected = build_saved_selection_matrix(
                summary_df=summary_df,
                selected_df=selected_df,
                group_name=region_name,
                domain=domain,
                age_order=age_order,
            )

            if matrix.empty:
                continue

            plot_records.append(
                {
                    "region": region_name,
                    "domain": domain,
                    "matrix": matrix,
                    "selected": selected,
                }
            )

    if not plot_records:
        print(
            "[WARN] No valid domain heatmaps were generated."
        )
        return

    # ============================================================
    # 2. Global color limits
    #
    # Same logic as the original combined figure.
    # ============================================================

    matrices = [
        record["matrix"]
        for record in plot_records
    ]

    selected_tables = [
        record["selected"]
        for record in plot_records
    ]

    coherent_vmax = calculate_marker_class_vmax(
        matrices=matrices,
        selected_tables=selected_tables,
        marker_class="Coherent",
    )

    heterogeneous_vmax = calculate_marker_class_vmax(
        matrices=matrices,
        selected_tables=selected_tables,
        marker_class="Heterogeneous",
    )

    print(
        "[INFO] Shared color scales: "
        f"coherent ±{coherent_vmax:.3f}; "
        f"heterogeneous ±{heterogeneous_vmax:.3f}"
    )

    # ============================================================
    # 3. Fixed figure geometry
    #
    # IMPORTANT:
    # Every domain uses exactly the same figure dimensions.
    # ============================================================

    SINGLE_DOMAIN_FIGSIZE = (
        4.2,
        PANEL_HEIGHT,
    )

    # ============================================================
    # 4. Plot each domain independently
    # ============================================================

    for record in plot_records:

        region_name = record[
            "region"
        ]

        domain = record[
            "domain"
        ]

        matrix = record[
            "matrix"
        ]

        selected = record[
            "selected"
        ]

        selection_for_plot = selected.drop(
            columns=[
                "group",
                "domain",
            ],
            errors="ignore",
        )

        # --------------------------------------------------------
        # Fixed canvas for every domain
        # --------------------------------------------------------

        fig = plt.figure(
            figsize=SINGLE_DOMAIN_FIGSIZE
        )

        # Main heatmap + one narrow colorbar column.
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[
                1.0,
                0.075,
            ],

            # Fixed margins for every image.
            left=0.25,
            right=0.91,
            top=0.90,
            bottom=0.11,

            # Leave enough room for the Coherent/Heterogeneous
            # side annotations drawn by draw_heatmap_panel().
            wspace=0.75,
        )

        ax = fig.add_subplot(
            grid[0, 0]
        )

        # Two vertically stacked colorbars, same as original figure.
        cbar_grid = grid[
            0,
            1
        ].subgridspec(
            2,
            1,
            hspace=0.48,
        )

        coherent_cbar_ax = fig.add_subplot(
            cbar_grid[0, 0]
        )

        heterogeneous_cbar_ax = fig.add_subplot(
            cbar_grid[1, 0]
        )

        # --------------------------------------------------------
        # Heatmap
        # --------------------------------------------------------

        draw_heatmap_panel(
            ax=ax,
            matrix=matrix,
            selected=selection_for_plot,
            domain=domain,
            age_order=age_order,
            coherent_vmax=coherent_vmax,
            heterogeneous_vmax=heterogeneous_vmax,
        )

        # --------------------------------------------------------
        # Shared coherent scale
        # --------------------------------------------------------

        coherent_mappable = ScalarMappable(
            norm=Normalize(
                vmin=-coherent_vmax,
                vmax=coherent_vmax,
            ),
            cmap=HEATMAP_CMAP,
        )

        coherent_colorbar = fig.colorbar(
            coherent_mappable,
            cax=coherent_cbar_ax,
        )

        coherent_colorbar.set_label(
            r"Mean disease$-$WT $\Delta$",
            rotation=90,
            labelpad=7,
            fontsize=8.5,
        )

        coherent_colorbar.ax.tick_params(
            labelsize=7.5,
            length=2.2,
        )

        coherent_colorbar.outline.set_visible(
            False
        )

        # --------------------------------------------------------
        # Shared heterogeneous scale
        # --------------------------------------------------------

        heterogeneous_mappable = ScalarMappable(
            norm=Normalize(
                vmin=-heterogeneous_vmax,
                vmax=heterogeneous_vmax,
            ),
            cmap=HEATMAP_CMAP,
        )

        heterogeneous_colorbar = fig.colorbar(
            heterogeneous_mappable,
            cax=heterogeneous_cbar_ax,
        )

        heterogeneous_colorbar.set_label(
            r"Mean disease$-$WT $\Delta$",
            rotation=90,
            labelpad=7,
            fontsize=8.5,
        )

        heterogeneous_colorbar.ax.tick_params(
            labelsize=7.5,
            length=2.2,
        )

        heterogeneous_colorbar.outline.set_visible(
            False
        )

        # --------------------------------------------------------
        # Region annotation
        #
        # Keep this consistent across every panel.
        # --------------------------------------------------------

        fig.suptitle(
            region_name,
            fontsize=13.5,
            fontweight="bold",
            y=0.975,
        )

        # --------------------------------------------------------
        # Output filename:
        # [region_name]_[domain_id].png
        # --------------------------------------------------------

        safe_region_name = (
            region_name
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
        )

        safe_domain = (
            str(domain)
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
        )

        png_path = os.path.join(
            save_dir,
            f"{safe_region_name}_{safe_domain}.png",
        )

        fig.savefig(
            png_path,
            dpi=FIG_DPI,

            # IMPORTANT:
            # Do not use bbox_inches="tight".
            # Otherwise different gene-name lengths can create
            # PNGs with slightly different physical dimensions.
            facecolor="white",
        )

        print(
            f"[SAVE] {png_path}"
        )

        if SAVE_PDF:

            pdf_path = os.path.join(
                save_dir,
                f"{safe_region_name}_{safe_domain}.pdf",
            )

            fig.savefig(
                pdf_path,
                facecolor="white",
            )

            print(
                f"[SAVE] {pdf_path}"
            )

        plt.close(fig)

    print(
        f"[DONE] Saved {len(plot_records)} "
        f"individual domain figures to: {save_dir}"
    )


# ============================================================
# EVALUATION / SAVED-RESULT WORKFLOW
# ============================================================

def get_active_inputs() -> Tuple[List[Tuple[str, str, str]], List[str]]:
    pairs = [
        (label, disease_path, normal_path)
        for label, disease_path, normal_path in DISEASE_NORMAL_FILES
        if str(disease_path).strip() and str(normal_path).strip()
    ]
    if len(pairs) < 2:
        raise ValueError(
            "At least two disease/normal pairs are required."
        )
    return pairs, [label for label, _, _ in pairs]


def compute_hybrid_selection_tables(
    summary_df: pd.DataFrame,
    specific_df: pd.DataFrame,
    domain_groups: Sequence[Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the same hybrid ranking/selection calculation as the original script,
    but independently of plotting.
    """
    selected_tables = []
    ranking_tables = []

    available_domains = set(
        summary_df["domain"].astype(str).unique()
    )

    for group in domain_groups:
        group_name = str(group["name"])

        for domain in [str(x) for x in group["domains"]]:
            if domain not in available_domains:
                continue

            domain_summary = summary_df[
                summary_df["domain"].astype(str) == domain
            ].copy()
            domain_specific = specific_df[
                specific_df["domain"].astype(str) == domain
            ].copy()

            if domain_summary.empty or domain_specific.empty:
                continue

            ranking = rank_genes_hybrid(
                domain_summary_df=domain_summary,
                domain_specific_df=domain_specific,
            )
            selected = select_hybrid_genes(ranking)

            ranking = ranking.copy()
            ranking.insert(0, "domain", domain)
            ranking.insert(0, "group", group_name)
            ranking_tables.append(ranking)

            selected = selected.copy()
            selected.insert(0, "domain", domain)
            selected.insert(0, "group", group_name)
            selected_tables.append(selected)

    selected_df = (
        pd.concat(selected_tables, ignore_index=True)
        if selected_tables
        else pd.DataFrame()
    )
    ranking_df = (
        pd.concat(ranking_tables, ignore_index=True)
        if ranking_tables
        else pd.DataFrame()
    )
    return selected_df, ranking_df


def save_analysis_config(age_order: Sequence[str]) -> None:
    config = {
        "age_order": list(age_order),
        "domain_groups": DOMAIN_GROUPS,
        "preprocessing": {
            "library_size_normalize": LIBRARY_SIZE_NORMALIZE,
            "normalization_target_sum": NORMALIZATION_TARGET_SUM,
            "log1p_before_analysis": LOG1P_BEFORE_ANALYSIS,
            "delta_use_common_valid": DELTA_USE_COMMON_VALID,
            "coordinate_decimals": COORDINATE_DECIMALS,
        },
    }
    with open(
        ANALYSIS_CONFIG_JSON,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(config, handle, indent=2)


def evaluate_and_save() -> None:
    """
    Run all calculations and save numerical results only.
    """
    ensure_dir(OUTDIR)

    pairs, age_order = get_active_inputs()

    ad_dict = {
        label: compute_delta_h5ad(
            disease_path=disease_path,
            normal_path=normal_path,
        )
        for label, disease_path, normal_path in pairs
    }

    print(
        "[INFO] Preprocessing: "
        f"normalize_total={LIBRARY_SIZE_NORMALIZE}, "
        f"target_sum={NORMALIZATION_TARGET_SUM:g}, "
        f"log1p={LOG1P_BEFORE_ANALYSIS}"
    )
    print(
        "[INFO] Spatial delta definition: "
        "preprocess(disease) - preprocess(normal)"
    )

    all_domains = get_all_domains(ad_dict)
    visualization_domains = get_visualization_domains(
        ad_dict=ad_dict,
        domain_groups=DOMAIN_GROUPS,
    )
    genes = select_genes_for_analysis(ad_dict)

    print(f"[INFO] Ages: {age_order}")
    print(
        "[INFO] All domains used for heterogeneous references: "
        f"{all_domains}"
    )
    print(
        "[INFO] Visualization target domains: "
        f"{visualization_domains}"
    )
    print(f"[INFO] Genes retained: {len(genes)}")

    # All annotated domains: unchanged reference universe.
    all_domain_summary_df = build_domain_gene_summary_table(
        ad_dict=ad_dict,
        age_order=age_order,
        domains=all_domains,
        genes=genes,
    )
    all_domain_summary_df.to_csv(
        ALL_SUMMARY_CSV,
        index=False,
    )
    print(f"[SAVE] {ALL_SUMMARY_CSV}")

    # Visualization domains only.
    summary_df = all_domain_summary_df[
        all_domain_summary_df["domain"].isin(
            visualization_domains
        )
    ].copy()
    summary_df = retain_complete_domains(
        summary_df=summary_df,
        age_order=age_order,
    )

    retained_domains = sorted(
        summary_df["domain"].astype(str).unique(),
        key=domain_sort_key,
    )
    print(
        "[INFO] Retained visualization domains: "
        f"{retained_domains}"
    )

    summary_df.to_csv(
        VIS_SUMMARY_CSV,
        index=False,
    )
    print(f"[SAVE] {VIS_SUMMARY_CSV}")

    # Heterogeneous target-vs-all-other calculation.
    specific_df = build_domain_specific_component_table(
        summary_df=all_domain_summary_df,
        age_order=age_order,
        target_domains=retained_domains,
    )
    specific_df.to_csv(
        SPECIFIC_CSV,
        index=False,
    )
    print(f"[SAVE] {SPECIFIC_CSV}")

    # Same original hybrid ranking and quota-based selection.
    selected_df, ranking_df = compute_hybrid_selection_tables(
        summary_df=summary_df,
        specific_df=specific_df,
        domain_groups=DOMAIN_GROUPS,
    )

    if not ranking_df.empty:
        ranking_df.to_csv(
            RANKING_CSV,
            index=False,
        )
        print(f"[SAVE] {RANKING_CSV}")

    if not selected_df.empty:
        selected_df.to_csv(
            SELECTED_CSV,
            index=False,
        )
        print(f"[SAVE] {SELECTED_CSV}")

        audit_columns = [
            "group",
            "domain",
            "gene",
            "display_rank",
            "selection_source",
            "rank_within_source",
            "coherent_rank",
            "heterogeneous_rank",
            "coherent_score",
            "coherent_peak_signed_delta",
            "coherent_peak_age",
            "heterogeneous_score",
            "heterogeneous_peak_positive",
            "heterogeneous_peak_negative",
            "heterogeneous_peak_age",
            "coherent_percentile",
            "heterogeneous_percentile",
            "mean_absolute_burden",
            "mean_cancellation_index",
        ]
        audit_columns = [
            column
            for column in audit_columns
            if column in selected_df.columns
        ]
        selected_df[audit_columns].to_csv(
            AUDIT_CSV,
            index=False,
        )
        print(f"[SAVE] {AUDIT_CSV}")

    save_analysis_config(age_order)
    print(f"[SAVE] {ANALYSIS_CONFIG_JSON}")
    print(f"[DONE] Evaluation outputs saved to: {OUTDIR}")


def load_saved_plot_data():
    """
    Load only saved results required for figure generation.
    """
    required = [
        VIS_SUMMARY_CSV,
        SELECTED_CSV,
        ANALYSIS_CONFIG_JSON,
    ]
    missing = [
        path
        for path in required
        if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(
            "Missing saved result(s). Run --mode eval first:\n  "
            + "\n  ".join(missing)
        )

    summary_df = pd.read_csv(VIS_SUMMARY_CSV)
    selected_df = pd.read_csv(SELECTED_CSV)

    # Keep identifier dtypes identical to evaluation mode.
    summary_df["domain"] = summary_df["domain"].astype(str)
    summary_df["gene"] = summary_df["gene"].astype(str)
    summary_df["age"] = summary_df["age"].astype(str)

    selected_df["group"] = selected_df["group"].astype(str)
    selected_df["domain"] = selected_df["domain"].astype(str)
    selected_df["gene"] = selected_df["gene"].astype(str)

    with open(
        ANALYSIS_CONFIG_JSON,
        "r",
        encoding="utf-8",
    ) as handle:
        config = json.load(handle)

    age_order = [
        str(age)
        for age in config["age_order"]
    ]
    return summary_df, selected_df, age_order


def plot_saved_results() -> None:
    """
    Plot only from saved evaluation results.

    No h5ad is read and no gene filtering, domain specificity, ranking,
    or marker selection is recalculated.
    """
    summary_df, selected_df, age_order = load_saved_plot_data()

    figure_dir = os.path.join(
        OUTDIR,
        "grouped_hybrid_heatmaps",
    )
    ensure_dir(figure_dir)

    plot_individual_domain_hybrid_heatmaps(
        summary_df=summary_df,
        selected_df=selected_df,
        age_order=age_order,
        domain_groups=DOMAIN_GROUPS,
        save_dir=figure_dir,
    )
    print(f"[DONE] Figures saved to: {figure_dir}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("eval", "plot", "all"),
        default="plot",
        help=(
            "eval: calculate and save results only; "
            "plot: load saved results and plot only; "
            "all: evaluate then plot"
        ),
    )
    args = parser.parse_args()

    if args.mode in {"eval", "all"}:
        evaluate_and_save()

    if args.mode in {"plot", "all"}:
        plot_saved_results()


if __name__ == "__main__":
    main()
