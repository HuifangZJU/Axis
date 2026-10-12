#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnosis script: should disease-normal subtraction be done before or after log1p?

For each mouse-brain time point (2.5m, 5.7m, 13.4m), the script:
1. Computes spatial delta on the original scale:
       delta_raw = disease - normal
2. Computes spatial delta after log1p:
       delta_log1p = log1p(disease) - log1p(normal)
3. Visualizes the marker gene Arc on both delta maps.
4. Visualizes Arc count distributions on:
       - original scale
       - log1p scale

The disease and normal virtual tissues are first aligned on the full virtual
support, then restricted to the common-valid locations shared by both tissues.
This makes the comparison rigorous and directly comparable across the two delta
definitions.
"""

from __future__ import annotations
from axis.settings import data_path

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from axis.matching.contrast import compute_spatial_delta_from_files


# ============================================================
# CONFIG
# ============================================================

AGE_PAIR_FILES: List[Tuple[str, str, str]] = [
    (
        "2.5m",
        data_path("xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_2.500.h5ad"),
        data_path("xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_102.500.h5ad"),
    ),
    (
        "5.7m",
        data_path("xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_5.700.h5ad"),
        data_path("xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_105.700.h5ad"),
    ),
    (
        "13.4m",
        data_path("xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_13.400.h5ad"),
        data_path("xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_113.400.h5ad"),
    ),
]

GENE = "Slc17a6"
SPATIAL_KEY = "spatial"
VALID_KEY = "valid"
DOMAIN_KEY = "axis_joint_domain"
USE_COMMON_VALID = True
COORDINATE_DECIMALS = 5

# Figure / saving controls
OUTDIR: Optional[str] = None
SHOW = True
FIG_DPI = 300
POINT_SIZE = 10
HIST_BINS = 60
DISTRIBUTION_MAX_POINTS = 200000  # random subsample for hist if very large
RANDOM_SEED = 0


# ============================================================
# STYLING
# ============================================================

def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_dir(path: Optional[str]) -> None:
    if path is not None:
        os.makedirs(path, exist_ok=True)


def save_or_show(fig, save_path: Optional[str], show: bool = True, dpi: int = FIG_DPI):
    if save_path is not None:
        ensure_dir(os.path.dirname(os.path.abspath(save_path)))
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# HELPERS
# ============================================================

def _to_dense(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x)


def extract_gene_vector(adata: AnnData, gene: str) -> np.ndarray:
    if gene not in adata.var_names:
        raise KeyError(f"Gene {gene!r} not found.")
    x = adata[:, [gene]].X
    x = _to_dense(x).reshape(-1)
    return np.asarray(x, dtype=np.float32)


def get_coords(adata: AnnData, spatial_key: str = SPATIAL_KEY) -> np.ndarray:
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm[{spatial_key!r}] not found.")
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coords.shape[1] < 2:
        raise ValueError("Spatial coordinates must have at least two columns.")
    return coords[:, :2]


def symmetric_vlim(values: np.ndarray, percentile: float = 99.0, fallback: float = 1.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    vmax = float(np.nanpercentile(np.abs(values), percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = fallback
    return vmax


def subsample_for_hist(x: np.ndarray, max_points: int = DISTRIBUTION_MAX_POINTS, seed: int = RANDOM_SEED) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= max_points:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=max_points, replace=False)
    return x[idx]


def compute_arc_diagnosis(
    disease_path: str,
    normal_path: str,
    gene: str = GENE,
) -> Dict[str, object]:
    """Compute raw-scale and log1p-scale spatial delta diagnostics for one time point."""

    result_raw = compute_spatial_delta_from_files(
        disease_path,
        normal_path,
        use_log1p=False,
        valid_key=VALID_KEY,
        spatial_key=SPATIAL_KEY,
        domain_key=DOMAIN_KEY,
        genes=[gene],
        use_common_valid=USE_COMMON_VALID,
        coordinate_decimals=COORDINATE_DECIMALS,
        return_result=True,
    )

    result_log = compute_spatial_delta_from_files(
        disease_path,
        normal_path,
        use_log1p=True,
        valid_key=VALID_KEY,
        spatial_key=SPATIAL_KEY,
        domain_key=DOMAIN_KEY,
        genes=[gene],
        use_common_valid=USE_COMMON_VALID,
        coordinate_decimals=COORDINATE_DECIMALS,
        return_result=True,
    )

    ad_disease = sc.read_h5ad(disease_path)
    ad_normal = sc.read_h5ad(normal_path)

    disease_idx = np.asarray(result_raw.disease_indices, dtype=int)
    normal_idx = np.asarray(result_raw.normal_indices, dtype=int)

    disease_arc_raw = extract_gene_vector(ad_disease[disease_idx].copy(), gene)
    normal_arc_raw = extract_gene_vector(ad_normal[normal_idx].copy(), gene)
    disease_arc_log = np.log1p(disease_arc_raw)
    normal_arc_log = np.log1p(normal_arc_raw)

    delta_arc_raw = extract_gene_vector(result_raw.adata, gene)
    delta_arc_log = extract_gene_vector(result_log.adata, gene)
    coords = get_coords(result_raw.adata)

    return {
        "coords": coords,
        "delta_raw": delta_arc_raw,
        "delta_log": delta_arc_log,
        "disease_arc_raw": disease_arc_raw,
        "normal_arc_raw": normal_arc_raw,
        "disease_arc_log": disease_arc_log,
        "normal_arc_log": normal_arc_log,
        "n_common_valid": int(result_raw.common_valid_count),
        "n_matched_before_valid": int(result_raw.matched_count),
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_one_timepoint_diagnosis(
    label: str,
    diag: Dict[str, object],
    gene: str = GENE,
    save_path: Optional[str] = None,
    show: bool = SHOW,
):
    """Create a 1x4 diagnosis figure for one time point."""

    coords = np.asarray(diag["coords"], dtype=float)
    delta_raw = np.asarray(diag["delta_raw"], dtype=float)
    delta_log = np.asarray(diag["delta_log"], dtype=float)
    disease_arc_raw = subsample_for_hist(np.asarray(diag["disease_arc_raw"], dtype=float))
    normal_arc_raw = subsample_for_hist(np.asarray(diag["normal_arc_raw"], dtype=float))
    disease_arc_log = subsample_for_hist(np.asarray(diag["disease_arc_log"], dtype=float))
    normal_arc_log = subsample_for_hist(np.asarray(diag["normal_arc_log"], dtype=float))

    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.3), constrained_layout=False)
    ax1, ax2, ax3, ax4 = axes

    vmax_raw = symmetric_vlim(delta_raw, percentile=99.0)
    vmax_log = symmetric_vlim(delta_log, percentile=99.0)

    sc1 = ax1.scatter(
        coords[:, 0],
        coords[:, 1],
        c=delta_raw,
        s=POINT_SIZE,
        cmap="coolwarm",
        vmin=-vmax_raw,
        vmax=vmax_raw,
        linewidths=0,
        rasterized=True,
    )
    ax1.set_title(f"{gene} delta\nDisease - Normal")
    ax1.set_aspect("equal")
    ax1.invert_yaxis()
    ax1.axis("off")
    cbar1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.set_label("delta", rotation=90)

    sc2 = ax2.scatter(
        coords[:, 0],
        coords[:, 1],
        c=delta_log,
        s=POINT_SIZE,
        cmap="coolwarm",
        vmin=-vmax_log,
        vmax=vmax_log,
        linewidths=0,
        rasterized=True,
    )
    ax2.set_title(f"{gene} delta\nlog1p(Disease) - log1p(Normal)")
    ax2.set_aspect("equal")
    ax2.invert_yaxis()
    ax2.axis("off")
    cbar2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.set_label("delta", rotation=90)

    ax3.hist(
        normal_arc_raw,
        bins=HIST_BINS,
        alpha=0.55,
        density=True,
        label="Normal",
        color="#4C78A8",
    )
    ax3.hist(
        disease_arc_raw,
        bins=HIST_BINS,
        alpha=0.55,
        density=True,
        label="Disease",
        color="#E15759",
    )
    ax3.set_title(f"{gene} counts distribution\nOriginal scale")
    ax3.set_xlabel("counts")
    ax3.set_ylabel("density")
    ax3.legend(frameon=False, loc="upper right")

    ax4.hist(
        normal_arc_log,
        bins=HIST_BINS,
        alpha=0.55,
        density=True,
        label="Normal",
        color="#4C78A8",
    )
    ax4.hist(
        disease_arc_log,
        bins=HIST_BINS,
        alpha=0.55,
        density=True,
        label="Disease",
        color="#E15759",
    )
    ax4.set_title(f"{gene} counts distribution\nAfter log1p")
    ax4.set_xlabel("log1p(counts)")
    ax4.set_ylabel("density")
    ax4.legend(frameon=False, loc="upper right")

    fig.suptitle(
        f"{label} | {gene} disease-normal subtraction diagnosis\n"
        f"matched common-valid locations = {diag['n_common_valid']}"
        f" (from {diag['n_matched_before_valid']} matched locations)",
        y=1.03,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(w_pad=1.5)
    save_or_show(fig, save_path=save_path, show=show)


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(OUTDIR)

    for label, disease_path, normal_path in AGE_PAIR_FILES:
        print(f"[INFO] processing {label}")
        diag = compute_arc_diagnosis(
            disease_path=disease_path,
            normal_path=normal_path,
            gene=GENE,
        )

        save_path = None
        if OUTDIR is not None:
            save_path = os.path.join(
                OUTDIR,
                f"diagnosis_{GENE.lower()}_{label.replace('.', 'p')}.png",
            )

        plot_one_timepoint_diagnosis(
            label=label,
            diag=diag,
            gene=GENE,
            save_path=save_path,
            show=SHOW,
        )

    print("[DONE] Arc subtraction diagnosis finished.")


if __name__ == "__main__":
    main()
