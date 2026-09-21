#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate anatomical neighborhood preservation in AXIS and SpatialZ virtual
mouse-brain slices.

The analysis computes anatomical neighborhood component agreement and detailed
domain-neighborhood composition matrices.

The plotting workflow generates the anatomical neighborhood component-agreement
boxplot and separate average neighborhood heatmaps for each anatomical region and
method. Heatmaps are not vertically stacked.

For the heatmaps, all available input tissues are used. The real-reference
matrix is the mean across all real slices, while SpatialZ and AXIS matrices are
the entry-wise means across all available virtual ages. All independently saved
heatmaps share one positive-only color normalization. Because neighborhood
component fractions are nonnegative, only the white-to-red half of the seaborn
``vlag`` palette is used; the blue half is excluded. Hippocampus uses domains
8 and 11. Cortical layers use the requested order 1, 6, 4, and 11.
"""


from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================================================================
# CONFIGURATION
# =============================================================================

METHODS: Dict[str, Dict[float, str]] = {
    "AXIS": {
        # 3.3: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial1.0/query/virtual_AD_time_3.300.h5ad",
        4.1: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial1.0/query/virtual_AD_time_4.100.h5ad",
        # 4.9: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial1.0/query/virtual_AD_time_4.900.h5ad",
        # 8.75: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial1.0/query/virtual_AD_time_8.750.h5ad",
        11.8: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial1.0/query/virtual_AD_time_11.800.h5ad",
        # 14.85: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial1.0/query/virtual_AD_time_14.850.h5ad",
    },
    "SpatialZ": {
        # 3.3: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/3.3.h5ad",
        4.1: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/4.1.h5ad",
        # 4.9: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/4.9.h5ad",
        # 8.75: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/8.75.h5ad",
        11.8: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/11.8.h5ad",
        # 14.85: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/14.85.h5ad",
    },
}

REAL_SLICES: Dict[float, str] = {
    2.5: "/media/huifang/data/registration/xenium/mouse_brain/h5ads/Tg/0_0.h5ad",
    5.7: "/media/huifang/data/registration/xenium/mouse_brain/h5ads/Tg/0_1.h5ad",
    17.9: "/media/huifang/data/registration/xenium/mouse_brain/h5ads/Tg/0_2.h5ad",
}


@dataclass(frozen=True)
class Config:
    outdir: str = (
        "/media/huifang/data/registration/paper_analysis/Figure56_mouse/5de/"
    )

    coord_key: str = "spatial"
    domain_key: str = "axis_joint_domain"
    filter_valid: bool = True
    valid_key: str = "valid"
    knn_k: int = 50

    # None uses the union of domains present in all evaluated objects.
    selected_domains: Optional[Tuple[str, ...]] = None

    case_times: Tuple[float, ...] = (4.1, 11.8)
    hippocampus_domains: Tuple[str, ...] = ("8", "11")
    cortical_domains: Tuple[str, ...] = ("1", "4", "6", "9","10")

    figure_formats: Tuple[str, ...] = ("png", "pdf")
    figure_dpi: int = 350

    @property
    def agreement_csv(self) -> Path:
        return Path(self.outdir) / "anatomical_neighborhood_component_agreement.csv"

    @property
    def matrices_csv(self) -> Path:
        return Path(self.outdir) / "anatomical_neighborhood_matrices.csv.gz"

    @property
    def region_summary_csv(self) -> Path:
        return Path(self.outdir) / "anatomical_neighborhood_region_summary.csv"

    @property
    def config_json(self) -> Path:
        return Path(self.outdir) / "analysis_config.json"

    @property
    def figure_dir(self) -> Path:
        return Path(self.outdir) / "figures"


CFG = Config()

METHOD_COLORS = {
    "Real": "#7A7A7A",
    "SpatialZ": "#8EA4B8",
    "AXIS": "#C95A49",
}

# Kept exactly as in the original boxplot code.
FIG_DPI = 300
GROUP_PALETTE = {
    "Real": "#7f7f7f",
    "AXIS": "#d45a45",
    "SpatialZ": "#8fa0b8",
}

_VLAG_CMAP = sns.color_palette("vlag", as_cmap=True)
HEATMAP_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "vlag_white_to_red",
    _VLAG_CMAP(np.linspace(0.5, 1.0, 256)),
)

# Anatomical-region strip styling, following the compact side-header style used
# in the coherent/heterogeneous marker heatmaps.
CORTEX_REGION_COLOR = "#2F6F8F"
HIPPOCAMPUS_REGION_COLOR = "#A45A3A"
REGION_STRIP_WIDTH = 4.0
REGION_LABEL_FONTSIZE = 11.5
REGION_X_STRIP_OFFSET = 0.0
REGION_X_TEXT_OFFSET = 0.08
REGION_Y_STRIP_OFFSET = 0
REGION_Y_TEXT_OFFSET = 0.1

# =============================================================================
# DATA ACCESS
# =============================================================================

def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_h5ad(path: str, cfg: Config):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    adata = sc.read_h5ad(path)
    if cfg.filter_valid and cfg.valid_key in adata.obs:
        adata = adata[adata.obs[cfg.valid_key].astype(bool)].copy()

    if cfg.coord_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{cfg.coord_key!r}] in {path}")
    if cfg.domain_key not in adata.obs:
        raise KeyError(f"Missing adata.obs[{cfg.domain_key!r}] in {path}")
    return adata


def get_coordinates(adata, cfg: Config) -> np.ndarray:
    coords = np.asarray(adata.obsm[cfg.coord_key], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid coordinate shape: {coords.shape}")
    coords = coords[:, :2]
    if not np.all(np.isfinite(coords)):
        raise ValueError("Spatial coordinates contain non-finite values.")
    return coords


def get_domains(adata, cfg: Config) -> np.ndarray:
    return np.asarray(
        adata.obs[cfg.domain_key].astype("string").to_numpy(),
        dtype=object,
    )


def domain_sort_key(value: str) -> Tuple[int, float, str]:
    text = str(value)
    try:
        return 0, float(text), text
    except ValueError:
        return 1, np.inf, text


def determine_domain_order(
    real_slices: Mapping[float, str],
    methods: Mapping[str, Mapping[float, str]],
    cfg: Config,
) -> List[str]:
    if cfg.selected_domains is not None:
        return [str(value) for value in cfg.selected_domains]

    observed = set()
    all_paths = list(real_slices.values()) + [
        path
        for mapping in methods.values()
        for path in mapping.values()
    ]
    for path in all_paths:
        adata = read_h5ad(path, cfg)
        observed.update(
            str(value)
            for value in pd.unique(get_domains(adata, cfg))
            if pd.notna(value) and str(value).lower() != "nan"
        )

    return sorted(observed, key=domain_sort_key)


# =============================================================================
# ANATOMICAL NEIGHBORHOOD MATRICES
# =============================================================================

def compute_knn_indices(coords: np.ndarray, k: int) -> np.ndarray:
    n_obs = coords.shape[0]
    if n_obs < 2:
        return np.empty((n_obs, 0), dtype=np.int64)

    model = NearestNeighbors(n_neighbors=min(int(k) + 1, n_obs))
    model.fit(coords)
    return model.kneighbors(coords, return_distance=False)[:, 1:]


def compute_anatomical_neighborhood_matrix(
    adata,
    domain_order: Sequence[str],
    cfg: Config,
) -> np.ndarray:
    """
    Entry (i, j) is the mean fraction of KNN neighbors assigned to domain j
    among observations whose focal domain is i.
    """
    coords = get_coordinates(adata, cfg)
    labels = get_domains(adata, cfg)

    domain_to_index = {
        str(domain): index
        for index, domain in enumerate(domain_order)
    }
    label_index = np.asarray(
        [domain_to_index.get(str(label), -1) for label in labels],
        dtype=np.int64,
    )

    neighbor_indices = compute_knn_indices(coords, cfg.knn_k)
    n_domains = len(domain_order)
    per_observation = np.zeros((len(labels), n_domains), dtype=np.float32)

    for observation_index in range(len(labels)):
        neighbors = label_index[neighbor_indices[observation_index]]
        neighbors = neighbors[neighbors >= 0]
        if neighbors.size == 0:
            continue
        counts = np.bincount(neighbors, minlength=n_domains).astype(np.float32)
        per_observation[observation_index] = counts / counts.sum()

    matrix = np.full((n_domains, n_domains), np.nan, dtype=np.float32)
    for domain_index in range(n_domains):
        focal_mask = label_index == domain_index
        if np.any(focal_mask):
            matrix[domain_index] = np.mean(
                per_observation[focal_mask],
                axis=0,
            )
    return matrix


def mean_matrices(matrices: Sequence[np.ndarray]) -> np.ndarray:
    if not matrices:
        raise ValueError("At least one matrix is required.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.stack(matrices), axis=0).astype(np.float32)


def component_agreement(query_row: np.ndarray, reference_row: np.ndarray) -> float:
    """Agreement = 1 - 0.5 * L1 distance between composition vectors."""
    query_row = np.asarray(query_row, dtype=float)
    reference_row = np.asarray(reference_row, dtype=float)
    valid = np.isfinite(query_row) & np.isfinite(reference_row)
    if not np.any(valid):
        return np.nan
    return float(
        1.0 - 0.5 * np.sum(np.abs(query_row[valid] - reference_row[valid]))
    )


def matrix_to_long_rows(
    matrix: np.ndarray,
    group: str,
    time: Optional[float],
    slice_id: str,
    domain_order: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for focal_index, focal_domain in enumerate(domain_order):
        for neighbor_index, neighbor_domain in enumerate(domain_order):
            value = matrix[focal_index, neighbor_index]
            rows.append(
                {
                    "group": group,
                    "time": np.nan if time is None else float(time),
                    "slice_id": slice_id,
                    "focal_domain": str(focal_domain),
                    "neighbor_domain": str(neighbor_domain),
                    "component_fraction": (
                        float(value) if np.isfinite(value) else np.nan
                    ),
                }
            )
    return rows


def compute_evaluation_tables(
    real_slices: Mapping[float, str],
    methods: Mapping[str, Mapping[float, str]],
    domain_order: Sequence[str],
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute agreement scores and detailed neighborhood matrices."""
    real_matrices: Dict[float, np.ndarray] = {}
    for time, path in sorted(real_slices.items()):
        print(f"[Real] {time:g} months")
        real_matrices[float(time)] = compute_anatomical_neighborhood_matrix(
            read_h5ad(path, cfg), domain_order, cfg
        )

    real_reference = mean_matrices(list(real_matrices.values()))
    agreement_rows: List[Dict[str, object]] = []
    matrix_rows: List[Dict[str, object]] = matrix_to_long_rows(
        real_reference,
        group="Real reference",
        time=None,
        slice_id="real_reference",
        domain_order=domain_order,
    )

    # Real leave-one-out baseline.
    for query_time in sorted(real_matrices):
        query = real_matrices[query_time]
        others = [
            matrix
            for time, matrix in real_matrices.items()
            if time != query_time
        ]
        reference = mean_matrices(others) if others else real_reference
        matrix_rows.extend(
            matrix_to_long_rows(
                query,
                group="Real",
                time=query_time,
                slice_id=f"real_{query_time:g}",
                domain_order=domain_order,
            )
        )
        for domain_index, domain in enumerate(domain_order):
            agreement_rows.append(
                {
                    "group": "Real",
                    "time": query_time,
                    "slice_id": f"real_{query_time:g}",
                    "domain": str(domain),
                    "component_agreement": component_agreement(
                        query[domain_index], reference[domain_index]
                    ),
                }
            )

    # Virtual slices compared with the pooled real reference.
    for method, mapping in methods.items():
        for time, path in sorted(mapping.items()):
            print(f"[{method}] {time:g} months")
            query = compute_anatomical_neighborhood_matrix(
                read_h5ad(path, cfg), domain_order, cfg
            )
            matrix_rows.extend(
                matrix_to_long_rows(
                    query,
                    group=method,
                    time=float(time),
                    slice_id=f"{method}_{time:g}",
                    domain_order=domain_order,
                )
            )
            for domain_index, domain in enumerate(domain_order):
                agreement_rows.append(
                    {
                        "group": method,
                        "time": float(time),
                        "slice_id": f"{method}_{time:g}",
                        "domain": str(domain),
                        "component_agreement": component_agreement(
                            query[domain_index], real_reference[domain_index]
                        ),
                    }
                )

    return pd.DataFrame(agreement_rows), pd.DataFrame(matrix_rows)


# =============================================================================
# REGION-SPECIFIC SUMMARIES
# =============================================================================

def matrix_from_long(
    matrix_df: pd.DataFrame,
    group: str,
    domain_order: Sequence[str],
    time: Optional[float] = None,
) -> pd.DataFrame:
    subset = matrix_df[matrix_df["group"] == group].copy()
    if time is None:
        subset = subset[subset["time"].isna()]
    else:
        subset = subset[
            np.isclose(subset["time"].to_numpy(dtype=float), float(time))
        ]
    if subset.empty:
        raise KeyError(f"No matrix found for group={group!r}, time={time!r}.")

    matrix = subset.pivot(
        index="focal_domain",
        columns="neighbor_domain",
        values="component_fraction",
    )
    labels = [str(value) for value in domain_order]
    return matrix.reindex(index=labels, columns=labels)


def region_connectivity_statistics(
    matrix: pd.DataFrame,
    region_domains: Sequence[str],
) -> Dict[str, float]:
    domains = [str(value) for value in region_domains]
    values = matrix.reindex(index=domains, columns=domains).to_numpy(dtype=float)
    diagonal = np.diag(values)
    off_diagonal = values[~np.eye(len(domains), dtype=bool)]
    diagonal = diagonal[np.isfinite(diagonal)]
    off_diagonal = off_diagonal[np.isfinite(off_diagonal)]

    diagonal_mean = float(np.mean(diagonal)) if diagonal.size else np.nan
    off_diagonal_mean = float(np.mean(off_diagonal)) if off_diagonal.size else np.nan
    separation_margin = (
        diagonal_mean - off_diagonal_mean
        if np.isfinite(diagonal_mean) and np.isfinite(off_diagonal_mean)
        else np.nan
    )
    return {
        "diagonal_mean": diagonal_mean,
        "off_diagonal_mean": off_diagonal_mean,
        "separation_margin": separation_margin,
    }


def summarized_matrix_from_long(
    matrix_df: pd.DataFrame,
    group: str,
    domain_order: Sequence[str],
) -> pd.DataFrame:
    """
    Return one summarized neighborhood matrix per condition.

    ``Real reference`` is already the mean across real anchors. For AXIS and
    SpatialZ, each matrix entry is averaged across all available virtual ages.
    """
    labels = [str(value) for value in domain_order]

    # Keep matrix identifiers type-stable even if matrix_df came from CSV
    # or another external source.
    working_df = matrix_df.copy()
    working_df["group"] = working_df["group"].astype(str)
    working_df["focal_domain"] = working_df["focal_domain"].astype(str)
    working_df["neighbor_domain"] = working_df["neighbor_domain"].astype(str)

    subset = working_df[working_df["group"] == str(group)].copy()

    if group == "Real reference":
        subset = subset[subset["time"].isna()]
        if subset.empty:
            raise KeyError("The real-reference matrix is unavailable.")
        summarized = subset.pivot(
            index="focal_domain",
            columns="neighbor_domain",
            values="component_fraction",
        )
    else:
        if subset.empty:
            raise KeyError(f"No neighborhood matrices found for {group!r}.")
        summarized = (
            subset.groupby(
                ["focal_domain", "neighbor_domain"],
                as_index=False,
                observed=True,
            )["component_fraction"]
            .mean()
            .pivot(
                index="focal_domain",
                columns="neighbor_domain",
                values="component_fraction",
            )
        )

    return summarized.reindex(index=labels, columns=labels)


def matrix_at_condition(
    matrix_df: pd.DataFrame,
    group: str,
    domain_order: Sequence[str],
    time: Optional[float] = None,
) -> pd.DataFrame:
    """Return one real-reference or age-specific virtual neighborhood matrix."""
    labels = [str(value) for value in domain_order]
    subset = matrix_df[matrix_df["group"] == group].copy()

    if group == "Real reference":
        subset = subset[subset["time"].isna()]
    else:
        if time is None:
            raise ValueError(f"time is required for virtual group {group!r}.")
        subset = subset[
            np.isclose(
                subset["time"].to_numpy(dtype=float),
                float(time),
                equal_nan=False,
            )
        ]

    if subset.empty:
        raise KeyError(
            f"No neighborhood matrix found for group={group!r}, time={time!r}."
        )

    matrix = subset.pivot(
        index="focal_domain",
        columns="neighbor_domain",
        values="component_fraction",
    )
    return matrix.reindex(index=labels, columns=labels)


def compute_region_summary(
    matrix_df: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Save age-specific region statistics for the selected case times."""
    domain_order = sorted(
        matrix_df["focal_domain"].astype(str).unique(),
        key=domain_sort_key,
    )
    region_definitions = {
        "Hippocampus": cfg.hippocampus_domains,
        "Cortex": cfg.cortical_domains,
    }

    rows = []
    for region_name, region_domains in region_definitions.items():
        real_matrix = matrix_at_condition(
            matrix_df,
            group="Real reference",
            domain_order=domain_order,
            time=None,
        )
        rows.append(
            {
                "region": region_name,
                "domains": ",".join(region_domains),
                "group": "Real reference",
                "time": np.nan,
                **region_connectivity_statistics(
                    real_matrix,
                    region_domains,
                ),
            }
        )

        for time in cfg.case_times:
            for group in ("SpatialZ", "AXIS"):
                matrix = matrix_at_condition(
                    matrix_df,
                    group=group,
                    domain_order=domain_order,
                    time=float(time),
                )
                rows.append(
                    {
                        "region": region_name,
                        "domains": ",".join(region_domains),
                        "group": group,
                        "time": float(time),
                        **region_connectivity_statistics(
                            matrix,
                            region_domains,
                        ),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# SAVE NUMERICAL RESULTS
# =============================================================================

def save_results(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_directory(Path(cfg.outdir))
    domain_order = determine_domain_order(REAL_SLICES, METHODS, cfg)
    print("[Domains]", domain_order)

    agreement_df, matrix_df = compute_evaluation_tables(
        REAL_SLICES, METHODS, domain_order, cfg
    )
    region_summary = compute_region_summary(matrix_df, cfg)

    agreement_df.to_csv(cfg.agreement_csv, index=False)
    matrix_df.to_csv(cfg.matrices_csv, index=False, compression="gzip")
    region_summary.to_csv(cfg.region_summary_csv, index=False)

    config_payload = {
        "domain_order": domain_order,
        "coord_key": cfg.coord_key,
        "domain_key": cfg.domain_key,
        "knn_k": cfg.knn_k,
        "case_times": list(cfg.case_times),
        "hippocampus_domains": list(cfg.hippocampus_domains),
        "cortical_domains": list(cfg.cortical_domains),
        "real_slices": {str(time): path for time, path in REAL_SLICES.items()},
        "methods": {
            method: {str(time): path for time, path in mapping.items()}
            for method, mapping in METHODS.items()
        },
    }
    cfg.config_json.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    for path in (
        cfg.agreement_csv,
        cfg.matrices_csv,
        cfg.region_summary_csv,
        cfg.config_json,
    ):
        print(f"[Saved] {path}")
    return agreement_df, matrix_df


def load_results(
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    required = [cfg.agreement_csv, cfg.matrices_csv, cfg.config_json]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing saved outputs: " + ", ".join(missing))

    agreement_df = pd.read_csv(cfg.agreement_csv)
    matrix_df = pd.read_csv(cfg.matrices_csv)
    config = json.loads(cfg.config_json.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # IMPORTANT:
    # CSV loading automatically converts numeric-looking domain labels
    # such as "0", "1", ... to integers. In summarize/all mode these
    # columns are still strings in memory, but in plot mode the integer
    # labels no longer match the string-valued domain_order loaded from
    # analysis_config.json. Reindexing then produces an all-NaN matrix.
    #
    # Normalize identifiers immediately after loading so plot mode has
    # exactly the same data types as summarize/all mode.
    # ------------------------------------------------------------------
    for column in ("group", "slice_id", "domain"):
        if column in agreement_df.columns:
            agreement_df[column] = agreement_df[column].astype(str)

    for column in (
        "group",
        "slice_id",
        "focal_domain",
        "neighbor_domain",
    ):
        if column in matrix_df.columns:
            matrix_df[column] = matrix_df[column].astype(str)

    if "time" in agreement_df.columns:
        agreement_df["time"] = pd.to_numeric(
            agreement_df["time"],
            errors="coerce",
        )
    if "component_agreement" in agreement_df.columns:
        agreement_df["component_agreement"] = pd.to_numeric(
            agreement_df["component_agreement"],
            errors="coerce",
        )

    if "time" in matrix_df.columns:
        matrix_df["time"] = pd.to_numeric(
            matrix_df["time"],
            errors="coerce",
        )
    if "component_fraction" in matrix_df.columns:
        matrix_df["component_fraction"] = pd.to_numeric(
            matrix_df["component_fraction"],
            errors="coerce",
        )

    # Also normalize the saved config for robustness.
    if "domain_order" in config:
        config["domain_order"] = [
            str(value)
            for value in config["domain_order"]
        ]

    return agreement_df, matrix_df, config


# =============================================================================
# FIGURE HELPERS
# =============================================================================

def make_domain_palette(domains: Sequence[str]):
    base = sns.color_palette("tab20", n_colors=max(20, len(domains)))
    return {d: base[i % len(base)] for i, d in enumerate(domains)}


# The following three functions are kept exactly as in the original script.
def set_pub_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_or_show(fig, save_path=None):
    if save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_domain_component_agreement(
    comp_df: pd.DataFrame,
    save_path=None,
):
    if len(comp_df) == 0:
        print("[WARN] component agreement table empty; skip plot")
        return

    set_pub_style()

    # ------------------------------------------------------------------
    # Sort groups by mean score
    # ------------------------------------------------------------------
    order = (
        comp_df.groupby(
            "group",
            as_index=False,
        )["component_agreement"]
        .mean()
        .sort_values(
            "component_agreement",
            ascending=False,
        )["group"]
        .tolist()
    )

    domain_order = sorted(
        comp_df["domain"].unique().tolist()
    )

    domain_pal = make_domain_palette(
        domain_order
    )

    # ------------------------------------------------------------------
    # Figure layout
    #
    # Left  : main boxplot
    # Right : dedicated legend panel
    #
    # This prevents the legends from overlapping the data while ensuring
    # that everything remains within the exported figure.
    # ------------------------------------------------------------------
    fig = plt.figure(
        figsize=(8.6, 5.7),
    )

    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[4.5, 1.45],
        left=0.10,
        right=0.98,
        bottom=0.12,
        top=0.90,
        wspace=0.06,
    )

    ax = fig.add_subplot(
        gs[0, 0]
    )

    legend_ax = fig.add_subplot(
        gs[0, 1]
    )

    # Legend panel is purely for layout.
    legend_ax.axis("off")

    # ------------------------------------------------------------------
    # Main boxplot
    # ------------------------------------------------------------------
    sns.boxplot(
        data=comp_df,
        x="group",
        y="component_agreement",
        order=order,
        palette=GROUP_PALETTE,
        showfliers=False,
        width=0.58,
        linewidth=1.4,
        ax=ax,
    )

    # ------------------------------------------------------------------
    # Per-domain observations
    # ------------------------------------------------------------------
    xmap = {
        g: i
        for i, g in enumerate(order)
    }

    rng = np.random.default_rng(0)

    for _, r in comp_df.iterrows():

        if pd.isnull(
            r["component_agreement"]
        ):
            continue

        x = (
            xmap[r["group"]]
            + rng.uniform(
                -0.12,
                0.12,
            )
        )

        ax.scatter(
            x,
            r["component_agreement"],
            s=28,
            color=domain_pal[r["domain"]],
            alpha=0.84,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )

    # ------------------------------------------------------------------
    # Main-axis typography
    # ------------------------------------------------------------------
    ax.set_title(
        "Anatomical neighborhood component agreement",
        fontsize=18,
        pad=12,
    )

    ax.set_xlabel("")

    ax.set_ylabel(
        "Neighborhood composition agreement",
        fontsize=14,
        labelpad=8,
    )

    ax.tick_params(
        axis="x",
        labelsize=15,
        pad=5,
    )

    ax.tick_params(
        axis="y",
        labelsize=10.5,
    )

    ax.grid(
        axis="y",
        alpha=0.16,
        linewidth=0.8,
    )

    # Slightly stronger axis lines
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # ------------------------------------------------------------------
    # Legend handles
    # ------------------------------------------------------------------
    group_handles = [
        Patch(
            facecolor=GROUP_PALETTE[g],
            edgecolor="black",
            linewidth=0.7,
            label=g,
        )
        for g in order
    ]

    domain_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=domain_pal[d],
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=str(d),
            markersize=7,
        )
        for d in domain_order
    ]

    # ------------------------------------------------------------------
    # Right-side legend panel
    #
    # Group legend at top.
    # Domain legend directly beneath it.
    # ------------------------------------------------------------------
    group_legend = legend_ax.legend(
        handles=group_handles,
        # title="Method",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
        frameon=False,
        borderaxespad=0.0,
        fontsize=14,
        title_fontsize=15,
        handlelength=1.6,
        handletextpad=0.6,
        labelspacing=0.7,
    )

    legend_ax.add_artist(
        group_legend
    )

    # Estimate where the second legend should begin.
    # This leaves sufficient separation from the method legend while keeping
    # all legend content inside the reserved right-side panel.
    domain_legend_y = 0.72

    legend_ax.legend(
        handles=domain_handles,
        title="Domain",
        loc="upper left",
        bbox_to_anchor=(
            0.0,
            domain_legend_y,
        ),
        frameon=False,
        borderaxespad=0.0,
        fontsize=14,
        title_fontsize=15,
        markerscale=1.0,
        handletextpad=0.5,
        labelspacing=0.2,
        columnspacing=0.8,
        ncol=1,
    )

    # ------------------------------------------------------------------
    # Save / show
    # ------------------------------------------------------------------
    save_or_show(
        fig,
        save_path,
    )

# =============================================================================
# AGE-SPECIFIC NEIGHBORHOOD HEATMAPS
# =============================================================================

def select_region_matrix(
    matrix: pd.DataFrame,
    domains: Sequence[str],
) -> pd.DataFrame:
    ordered = [str(value) for value in domains]
    missing = [
        domain
        for domain in ordered
        if domain not in matrix.index or domain not in matrix.columns
    ]
    if missing:
        raise KeyError(f"Domains missing from neighborhood matrix: {missing}")
    return matrix.reindex(index=ordered, columns=ordered)


def ordered_unique(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            ordered.append(text)
            seen.add(text)
    return ordered



def annotate_heatmap(
    axis: mpl.axes.Axes,
    matrix: pd.DataFrame,
    vmax: float,
    fontsize: float,
) -> None:
    values = matrix.to_numpy(dtype=float)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                continue
            text_color = "white" if value >= 0.58 * vmax else "#222222"
            axis.text(
                column + 0.5,
                row + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
            )



def _draw_region_axis_headers(
    axis: mpl.axes.Axes,
    partition_index: int,
    n_domains: int,
) -> None:
    """
    Draw tight colored region strips plus grouped x/y anatomical labels.

    X-axis groups describe NEIGHBOR domains.
    Y-axis groups describe FOCAL domains.
    """
    if not (0 < int(partition_index) < int(n_domains)):
        return

    partition_index = int(partition_index)
    n_domains = int(n_domains)

    x_transform = axis.get_xaxis_transform()
    y_transform = axis.get_yaxis_transform()

    inset = -0.00

    groups = [
        {
            "name": "Cortex",
            "start": 0.0,
            "end": float(partition_index),
            "color": CORTEX_REGION_COLOR,
        },
        {
            "name": "Hippocampus",
            "start": float(partition_index),
            "end": float(n_domains),
            "color": HIPPOCAMPUS_REGION_COLOR,
        },
    ]

    for group in groups:
        start = group["start"]
        end = group["end"]
        center = 0.5 * (start + end)
        color = group["color"]
        name = group["name"]

        # Horizontal strip immediately below the heatmap.
        axis.plot(
            [start + inset, end - inset],
            [-REGION_X_STRIP_OFFSET, -REGION_X_STRIP_OFFSET],
            transform=x_transform,
            color=color,
            linewidth=REGION_STRIP_WIDTH,
            solid_capstyle="butt",
            clip_on=False,
            zorder=8,
        )
        axis.text(
            center-0.5,
            -REGION_X_TEXT_OFFSET,
            f"{name} (Neighbor)",
            transform=x_transform,
            ha="center",
            va="top",
            fontsize=REGION_LABEL_FONTSIZE,
            # fontweight="bold",
            color=color,
            clip_on=False,
        )

        # Vertical strip immediately left of the heatmap.
        axis.plot(
            [-REGION_Y_STRIP_OFFSET, -REGION_Y_STRIP_OFFSET],
            [start + inset, end - inset],
            transform=y_transform,
            color=color,
            linewidth=REGION_STRIP_WIDTH,
            solid_capstyle="butt",
            clip_on=False,
            zorder=8,
        )
        axis.text(
            -REGION_Y_TEXT_OFFSET,
            center+1,
            f"{name} (Focal)",
            transform=y_transform,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=REGION_LABEL_FONTSIZE,
            # fontweight="bold",
            color=color,
            clip_on=False,
        )


def draw_condition_heatmap(
    axis: mpl.axes.Axes,
    matrix: pd.DataFrame,
    method_label: str,
    region_title: str,
    norm: mpl.colors.Normalize,
    *,
    cbar_ax: Optional[mpl.axes.Axes] = None,
    annotate: bool = True,
    grid_fontsize: float = 12.0,
    tick_fontsize: float = 11.0,
    title_fontsize: float = 14.0,
    axis_fontsize: float = 12.0,
    partition_index: Optional[int] = None,
    partition_color: str = "#8A8A8A",
) -> None:
    """Plot one method-specific anatomical-neighborhood heatmap."""
    sns.heatmap(
        matrix,
        ax=axis,
        cmap=HEATMAP_CMAP,
        norm=norm,
        linewidths=0.55,
        linecolor="white",
        mask=matrix.isna(),
        cbar=(cbar_ax is not None),
        cbar_ax=cbar_ax,
        cbar_kws=(
            {"label": "Neighborhood component fraction"}
            if cbar_ax is not None
            else None
        ),
        annot=False,
    )

    axis.set_yticklabels(
        matrix.index.astype(str),
        rotation=0,
        fontsize=tick_fontsize,
    )
    axis.set_xticklabels(
        matrix.columns.astype(str),
        rotation=0,
        fontsize=tick_fontsize,
    )
    axis.tick_params(
        axis="both",
        labelsize=tick_fontsize,
        pad=5,
        length=0,
    )

    # The anatomical region is encoded directly on the x/y grouped labels.
    axis.set_title(
        method_label,
        fontsize=title_fontsize,
        # fontweight="bold",
        pad=10,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")

    if partition_index is not None and 0 < int(partition_index) < matrix.shape[0]:
        # boundary = float(partition_index)
        #
        # # Gray matrix partitions retained from the previous version.
        # axis.axhline(
        #     boundary,
        #     color=partition_color,
        #     linewidth=2.0,
        #     zorder=6,
        # )
        # axis.axvline(
        #     boundary,
        #     color=partition_color,
        #     linewidth=2.0,
        #     zorder=6,
        # )

        # Publication-style colored strips and grouped axis labels.
        _draw_region_axis_headers(
            axis=axis,
            partition_index=int(partition_index),
            n_domains=matrix.shape[0],
        )

    if annotate:
        annotate_heatmap(
            axis,
            matrix,
            vmax=float(norm.vmax),
            fontsize=grid_fontsize,
        )

    if cbar_ax is not None:
        cbar_ax.tick_params(labelsize=10.5)
        cbar_ax.set_ylabel(
            "Neighborhood component fraction",
            fontsize=11.5,
            labelpad=10,
        )


def plot_average_neighborhood_heatmaps(
    matrix_df: pd.DataFrame,
    domain_order: Sequence[str],
    cfg: Config,
) -> None:
    """
    Plot one combined cortex+hippocampus neighborhood heatmap per method.

    Domains are grouped by anatomical region, with cortical layers first and
    hippocampus second. Gray partition lines separate the two groups. All
    methods share the same positive-only normalization.
    """
    method_matrices = {
        "Real reference": summarized_matrix_from_long(
            matrix_df,
            group="Real",
            domain_order=domain_order,
        ),
        "SpatialZ": summarized_matrix_from_long(
            matrix_df,
            group="SpatialZ",
            domain_order=domain_order,
        ),
        "AXIS": summarized_matrix_from_long(
            matrix_df,
            group="AXIS",
            domain_order=domain_order,
        ),
    }

    cortical_domains = [str(value) for value in cfg.cortical_domains]
    hippocampus_domains = [str(value) for value in cfg.hippocampus_domains]
    combined_domains = ordered_unique(cortical_domains + hippocampus_domains)
    partition_index = len(ordered_unique(cortical_domains))

    plot_records = []
    for method_label, full_matrix in method_matrices.items():
        plot_records.append(
            {
                "method_label": method_label,
                "matrix": select_region_matrix(full_matrix, combined_domains),
            }
        )

    finite_arrays = []
    for record in plot_records:
        values = record["matrix"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_arrays.append(finite)

    vmax = float(np.max(np.concatenate(finite_arrays))) if finite_arrays else 1.0
    vmax = max(vmax, 1e-6)
    norm = mpl.colors.PowerNorm(gamma=1.25, vmin=0.0, vmax=vmax)

    method_tags = {
        "Real reference": "real_reference",
        "SpatialZ": "spatialz",
        "AXIS": "axis",
    }

    set_pub_style()
    n_domains = len(combined_domains)

    for record in plot_records:
        method_label = record["method_label"]
        matrix = record["matrix"]

        figure_width = max(4.8, min(9.0, 3.8 + 0.56 * n_domains))
        figure_height = max(4.5, min(8.5, 3.0 + 0.56 * n_domains))


        grid_fontsize = 12
        tick_fontsize = 12

        figure = plt.figure(figsize=(figure_width, figure_height))
        gs = figure.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[1.0, 0.045],
            left=0.35,
            right=0.94,
            bottom=0.26,
            top=0.88,
            wspace=0.14,
        )
        heatmap_ax = figure.add_subplot(gs[0, 0])
        cbar_ax = figure.add_subplot(gs[0, 1])

        draw_condition_heatmap(
            axis=heatmap_ax,
            matrix=matrix,
            method_label=method_label,
            region_title="Cortex + Hippocampus",
            norm=norm,
            cbar_ax=cbar_ax,
            annotate=True,
            grid_fontsize=grid_fontsize,
            tick_fontsize=tick_fontsize,
            title_fontsize=14.5,
            axis_fontsize=12.0,
            partition_index=partition_index,
            partition_color="#9A9A9A",
        )

        save_path = (
            cfg.figure_dir
            / f"anatomical_neighborhood_heatmap_cortex_hippocampus_{method_tags[method_label]}_average.png"
        )
        save_or_show(figure, str(save_path))


# =============================================================================
# PLOTTING WORKFLOW
# =============================================================================

def plot_results(
    agreement_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    config: Mapping[str, object],
    cfg: Config,
) -> None:
    ensure_directory(cfg.figure_dir)
    domain_order = [str(value) for value in config["domain_order"]]

    # print("[Plot 1/2] Anatomical neighborhood component agreement")
    # plot_domain_component_agreement(
    #     agreement_df,
    #     # save_path=None,
    #     save_path=str(
    #         cfg.figure_dir
    #         / "anatomical_neighborhood_component_agreement.png"
    #     ),
    # )

    print("[Plot 2/2] Method-specific average neighborhood heatmaps across all input data")
    plot_average_neighborhood_heatmaps(
        matrix_df=matrix_df,
        domain_order=domain_order,
        cfg=cfg,
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("summarize", "plot", "all"),
        default="plot",
    )
    args = parser.parse_args()

    if args.mode in {"summarize", "all"}:
        agreement_df, matrix_df = save_results(CFG)
        config = json.loads(CFG.config_json.read_text(encoding="utf-8"))
    else:
        agreement_df, matrix_df, config = load_results(CFG)

    if args.mode in {"plot", "all"}:
        plot_results(agreement_df, matrix_df, config, CFG)


if __name__ == "__main__":
    main()
