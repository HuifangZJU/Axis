#!/usr/bin/env python3
from __future__ import annotations

import glob
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import wilcoxon


# ============================================================
# Config
# ============================================================

AXIS_ROOT = Path("/media/huifang/data/registration/SCC/huifang/pairwise_output")
ISTAR_ROOT = Path("/media/huifang/data/registration/SCC/istar/single_input")

PATIENT_IDS = [2, 5, 9, 10]
PAIR_LIST = [(0, 1), (0, 2), (1, 2)]

# coarse grid size in original iSTAR pixel units
BIN_SIZE_PX = 50

# keep only bins with at least this many contributing pixels
MIN_PIXELS_PER_BIN = 3

# Pearson stability filter
MIN_VALID_BINS = 20
MIN_STD = 1e-6

USE_LOG1P = False

OUTDIR = Path("/media/huifang/data/registration/SCC/istar_coarse_alignment_eval")
OUT_CSV = OUTDIR / "istar_coarse_grid_pearson_all_genes.csv"


# ============================================================
# Loaders (compatible with your current code layout)
# ============================================================

def load_flow_np(patient_id: int, fixed_id: int, moving_id: int) -> np.ndarray:
    """
    Load AXIS flow for (fixed, moving) pair from:
      AXIS_ROOT/group_{patient}/{fixed_id}_{moving_id}_*flow.npy

    Return shape normalized to (2, H, W), channel order assumed (dy, dx).
    """
    group_dir = AXIS_ROOT / f"group_{patient_id}"
    if not group_dir.is_dir():
        raise FileNotFoundError(f"AXIS group dir not found: {group_dir}")

    pattern = str(group_dir / f"{fixed_id}_{moving_id}_*flow.npy")
    matches = glob.glob(pattern)
    if not matches:
        fallback = group_dir / f"{fixed_id}_{moving_id}_flow.npy"
        if not fallback.is_file():
            raise FileNotFoundError(
                f"No flow file found for {fixed_id}_{moving_id} in {group_dir}"
            )
        flow_path = fallback
    else:
        flow_path = matches[0]

    flow = np.load(flow_path).astype(np.float32)
    if flow.ndim != 3:
        raise ValueError(f"Unexpected flow dims {flow.shape} in {flow_path}")

    if flow.shape[0] == 2:
        pass
    elif flow.shape[-1] == 2:
        flow = np.moveaxis(flow, -1, 0)
    else:
        raise ValueError(f"Flow must have channel dimension 2, got {flow.shape}")

    return flow


def load_gene_map_np(patient_id: int, axis_id: int, sample_id: int, gene_name: str) -> np.ndarray:
    sample_root = (
        ISTAR_ROOT
        / f"group_{patient_id}"
        / f"axis_{axis_id}"
        / f"sample{sample_id}"
    )
    cnts_dir = sample_root / "cnts-super"
    pkl_path = cnts_dir / f"{gene_name}.pickle"

    if not pkl_path.is_file():
        raise FileNotFoundError(f"Gene map missing: {pkl_path}")

    with open(pkl_path, "rb") as f:
        arr = pickle.load(f)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Gene map {pkl_path} must be 2D, got {arr.shape}")

    return arr


def load_mask_np(patient_id: int, axis_id: int, sample_id: int) -> np.ndarray:
    sample_root = (
        ISTAR_ROOT
        / f"group_{patient_id}"
        / f"axis_{axis_id}"
        / f"sample{sample_id}"
    )
    mask_path = sample_root / "mask-small.png"

    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file missing: {mask_path}")

    mask_img = Image.open(mask_path)
    mask_arr = np.array(mask_img)

    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    return (mask_arr > 0).astype(np.uint8)


def list_genes_for_sample(patient_id: int, axis_id: int, sample_id: int) -> set:
    cnts_dir = (
        ISTAR_ROOT
        / f"group_{patient_id}"
        / f"axis_{axis_id}"
        / f"sample{sample_id}"
        / "cnts-super"
    )
    if not cnts_dir.is_dir():
        raise FileNotFoundError(f"Gene directory missing: {cnts_dir}")
    # genelist = {p.stem for p in cnts_dir.glob("*.pickle")}
    genelist={'A2ML1'}

    return genelist


def list_common_genes_for_pair(patient_id: int, fixed_id: int, moving_id: int) -> List[str]:
    fixed_genes = list_genes_for_sample(patient_id, fixed_id, fixed_id)
    moving_genes = list_genes_for_sample(patient_id, moving_id, moving_id)
    return sorted(fixed_genes & moving_genes)


# ============================================================
# Coordinate mapping by flow (no dense gene resampling)
# ============================================================

def valid_pixel_coords_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Return valid pixel centers as xy coordinates with shape (N, 2).
    """
    yy, xx = np.where(mask > 0)
    return np.column_stack([xx, yy]).astype(np.float32)


def map_points_with_flow_nearest(
    coords_xy: np.ndarray,
    flow_np: np.ndarray,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Map high-resolution pixel coordinates with a lower-resolution flow field.

    Parameters
    ----------
    coords_xy : (N,2), in high-resolution image pixel coordinates
    flow_np : (2,Hf,Wf), assumed channel order (dy, dx)
    target_shape : (Ht, Wt), the high-resolution image shape

    Returns
    -------
    mapped_xy : (N,2)
    """
    coords_xy = np.asarray(coords_xy, dtype=np.float32)
    flow = np.asarray(flow_np, dtype=np.float32)

    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"Expected flow shape (2,H,W), got {flow.shape}")

    Hf, Wf = flow.shape[1], flow.shape[2]
    Ht, Wt = target_shape

    # map high-res coords into flow-grid indices
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]

    xf = np.round(x / max(Wt - 1, 1) * max(Wf - 1, 1)).astype(int)
    yf = np.round(y / max(Ht - 1, 1) * max(Hf - 1, 1)).astype(int)

    xf = np.clip(xf, 0, Wf - 1)
    yf = np.clip(yf, 0, Hf - 1)

    dy = flow[0, yf, xf]
    dx = flow[1, yf, xf]

    # scale displacement magnitude back to high-res pixel units
    scale_y = Ht / float(Hf)
    scale_x = Wt / float(Wf)

    mapped_x = x + dx * scale_x
    mapped_y = y + dy * scale_y

    return np.column_stack([mapped_x, mapped_y]).astype(np.float32)


# ============================================================
# Coarse-grid binning
# ============================================================

def compute_canvas_shape(shape_a: Tuple[int, int], shape_b: Tuple[int, int]) -> Tuple[int, int]:
    return max(shape_a[0], shape_b[0]), max(shape_a[1], shape_b[1])


def build_grid_spec(canvas_shape: Tuple[int, int], bin_size_px: int) -> Dict[str, int]:
    H, W = canvas_shape
    n_rows = int(np.ceil(H / float(bin_size_px)))
    n_cols = int(np.ceil(W / float(bin_size_px)))
    n_bins = n_rows * n_cols
    return {
        "H": H,
        "W": W,
        "bin_size_px": int(bin_size_px),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_bins": n_bins,
    }


def coords_to_bin_ids(coords_xy: np.ndarray, grid_spec: Dict[str, int]) -> np.ndarray:
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]

    bx = np.floor(x / grid_spec["bin_size_px"]).astype(int)
    by = np.floor(y / grid_spec["bin_size_px"]).astype(int)

    bx = np.clip(bx, 0, grid_spec["n_cols"] - 1)
    by = np.clip(by, 0, grid_spec["n_rows"] - 1)

    return by * grid_spec["n_cols"] + bx


def bin_mean_from_values(
    values: np.ndarray,
    bin_ids: np.ndarray,
    n_bins: int,
    min_pixels_per_bin: int = MIN_PIXELS_PER_BIN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate values into coarse bins by mean.
    """
    values = np.asarray(values, dtype=np.float32)
    bin_ids = np.asarray(bin_ids, dtype=np.int64)

    finite = np.isfinite(values)
    if finite.sum() == 0:
        means = np.full(n_bins, np.nan, dtype=np.float32)
        counts = np.zeros(n_bins, dtype=np.int32)
        return means, counts

    sums = np.bincount(bin_ids[finite], weights=values[finite], minlength=n_bins).astype(np.float32)
    counts = np.bincount(bin_ids[finite], minlength=n_bins).astype(np.int32)

    means = np.full(n_bins, np.nan, dtype=np.float32)
    valid_bins = counts >= int(min_pixels_per_bin)
    means[valid_bins] = sums[valid_bins] / counts[valid_bins]
    return means, counts


def masked_pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]

    if a.size < MIN_VALID_BINS:
        return np.nan
    if np.std(a) < MIN_STD or np.std(b) < MIN_STD:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


# ============================================================
# Gene-level evaluation
# ============================================================

def preprocess_pair_geometry(
    patient_id: int,
    fixed_id: int,
    moving_id: int,
    bin_size_px: int = BIN_SIZE_PX,
) -> Dict[str, object]:
    """
    Precompute geometry once per patient/pair.
    """
    fixed_mask = load_mask_np(patient_id, fixed_id, fixed_id)
    moving_mask = load_mask_np(patient_id, moving_id, moving_id)
    flow = load_flow_np(patient_id, fixed_id, moving_id)

    fixed_shape = fixed_mask.shape
    moving_shape = moving_mask.shape
    canvas_shape = compute_canvas_shape(fixed_shape, moving_shape)
    grid_spec = build_grid_spec(canvas_shape, bin_size_px=bin_size_px)

    fixed_coords = valid_pixel_coords_from_mask(fixed_mask)
    moving_coords_before = valid_pixel_coords_from_mask(moving_mask)
    moving_coords_after = map_points_with_flow_nearest(
        moving_coords_before,
        flow_np=flow,
        target_shape=moving_shape,
    )

    fixed_bin_ids = coords_to_bin_ids(fixed_coords, grid_spec)
    moving_bin_ids_before = coords_to_bin_ids(moving_coords_before, grid_spec)
    moving_bin_ids_after = coords_to_bin_ids(moving_coords_after, grid_spec)

    fixed_yx = np.where(fixed_mask > 0)
    moving_yx = np.where(moving_mask > 0)

    return {
        "fixed_shape": fixed_shape,
        "moving_shape": moving_shape,
        "canvas_shape": canvas_shape,
        "grid_spec": grid_spec,
        "fixed_mask": fixed_mask,
        "moving_mask": moving_mask,
        "fixed_coords": fixed_coords,
        "moving_coords_before": moving_coords_before,
        "moving_coords_after": moving_coords_after,
        "fixed_bin_ids": fixed_bin_ids,
        "moving_bin_ids_before": moving_bin_ids_before,
        "moving_bin_ids_after": moving_bin_ids_after,
        "fixed_yx": fixed_yx,
        "moving_yx": moving_yx,
    }


def extract_valid_values_from_gene_map(
    gene_map: np.ndarray,
    mask: np.ndarray,
    yx_index: Tuple[np.ndarray, np.ndarray],
    use_log1p: bool = USE_LOG1P,
) -> np.ndarray:
    gene_map = np.asarray(gene_map, dtype=np.float32)
    if gene_map.shape != mask.shape:
        raise ValueError(f"Gene map shape {gene_map.shape} != mask shape {mask.shape}")

    vals = gene_map[yx_index].astype(np.float32)

    # set background already excluded by yx_index, but keep NaN safety
    vals[~np.isfinite(vals)] = np.nan

    if use_log1p:
        vals = np.log1p(np.clip(vals, a_min=0, a_max=None))

    return vals


def evaluate_gene_for_pair(
    patient_id: int,
    fixed_id: int,
    moving_id: int,
    gene_name: str,
    geom: Dict[str, object],
) -> Optional[Dict[str, object]]:
    """
    Evaluate one gene for one patient/pair using coarse-grid Pearson.
    """
    fixed_map = load_gene_map_np(patient_id, fixed_id, fixed_id, gene_name)
    moving_map = load_gene_map_np(patient_id, moving_id, moving_id, gene_name)

    fixed_vals = extract_valid_values_from_gene_map(
        fixed_map,
        geom["fixed_mask"],
        geom["fixed_yx"],
        use_log1p=USE_LOG1P,
    )
    moving_vals = extract_valid_values_from_gene_map(
        moving_map,
        geom["moving_mask"],
        geom["moving_yx"],
        use_log1p=USE_LOG1P,
    )

    n_bins = geom["grid_spec"]["n_bins"]

    fixed_bin_means, fixed_counts = bin_mean_from_values(
        fixed_vals,
        geom["fixed_bin_ids"],
        n_bins=n_bins,
        min_pixels_per_bin=MIN_PIXELS_PER_BIN,
    )
    moving_before_bin_means, moving_before_counts = bin_mean_from_values(
        moving_vals,
        geom["moving_bin_ids_before"],
        n_bins=n_bins,
        min_pixels_per_bin=MIN_PIXELS_PER_BIN,
    )
    moving_after_bin_means, moving_after_counts = bin_mean_from_values(
        moving_vals,
        geom["moving_bin_ids_after"],
        n_bins=n_bins,
        min_pixels_per_bin=MIN_PIXELS_PER_BIN,
    )

    pearson_before = masked_pearson_1d(fixed_bin_means, moving_before_bin_means)
    pearson_after = masked_pearson_1d(fixed_bin_means, moving_after_bin_means)

    if not (np.isfinite(pearson_before) or np.isfinite(pearson_after)):
        return None

    common_before = int(np.sum(np.isfinite(fixed_bin_means) & np.isfinite(moving_before_bin_means)))
    common_after = int(np.sum(np.isfinite(fixed_bin_means) & np.isfinite(moving_after_bin_means)))

    return {
        "patient_id": patient_id,
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "gene": gene_name,
        "bin_size_px": geom["grid_spec"]["bin_size_px"],
        "n_bins_total": int(n_bins),
        "n_common_bins_before": common_before,
        "n_common_bins_after": common_after,
        "pearson_before": pearson_before,
        "pearson_after": pearson_after,
        "delta_pearson": (
            pearson_after - pearson_before
            if np.isfinite(pearson_before) and np.isfinite(pearson_after)
            else np.nan
        ),
    }


def collect_all_gene_metrics(
    patient_ids: Iterable[int],
    pair_list: Iterable[Tuple[int, int]],
    out_csv: Optional[Path] = OUT_CSV,
    bin_size_px: int = BIN_SIZE_PX,
) -> pd.DataFrame:
    rows = []

    for pid in patient_ids:
        for fixed_id, moving_id in pair_list:
            print(f"[INFO] patient={pid}, pair={fixed_id}->{moving_id}")

            try:
                common_genes = list_common_genes_for_pair(pid, fixed_id, moving_id)
                geom = preprocess_pair_geometry(
                    patient_id=pid,
                    fixed_id=fixed_id,
                    moving_id=moving_id,
                    bin_size_px=bin_size_px,
                )
            except FileNotFoundError as e:
                print(f"[WARN] skip patient={pid}, pair={fixed_id}->{moving_id}: {e}")
                continue

            for gene in common_genes:
                try:
                    row = evaluate_gene_for_pair(
                        patient_id=pid,
                        fixed_id=fixed_id,
                        moving_id=moving_id,
                        gene_name=gene,
                        geom=geom,
                    )
                    if row is not None:
                        rows.append(row)
                except Exception as e:
                    print(f"[WARN] gene={gene} failed for patient={pid}, pair={fixed_id}->{moving_id}: {e}")

    if not rows:
        raise RuntimeError("No valid gene-level Pearson results collected.")

    df = pd.DataFrame(rows)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[OK] saved csv: {out_csv}")

    return df


# ============================================================
# Plotting (paper-friendly, Pearson only)
# ============================================================

def summarize_by_gene(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across patient/pair instances to one row per gene.
    """
    gene_df = (
        df.groupby("gene", as_index=False)
        .agg(
            pearson_before=("pearson_before", "mean"),
            pearson_after=("pearson_after", "mean"),
            delta_pearson=("delta_pearson", "mean"),
            n_obs=("delta_pearson", "size"),
        )
        .sort_values("delta_pearson", ascending=False)
        .reset_index(drop=True)
    )
    return gene_df


def plot_gene_level_pearson_summary(
    gene_df: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    """
    2-panel summary:
      A. paired before/after Pearson across genes
      B. ranked gene improvement curve
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Panel A
    ax = axes[0]
    for _, row in gene_df.iterrows():
        ax.plot(
            [0, 1],
            [row["pearson_before"], row["pearson_after"]],
            color="0.82",
            alpha=0.45,
            linewidth=0.8,
            zorder=1,
        )

    ax.scatter(
        np.zeros(len(gene_df)),
        gene_df["pearson_before"].values,
        s=18,
        color="white",
        edgecolor="black",
        zorder=3,
    )
    ax.scatter(
        np.ones(len(gene_df)),
        gene_df["pearson_after"].values,
        s=18,
        color="crimson",
        edgecolor="black",
        zorder=3,
    )

    mean_before = gene_df["pearson_before"].mean()
    mean_after = gene_df["pearson_after"].mean()

    ax.scatter([0, 1], [mean_before, mean_after], marker="D", s=85, color="black", zorder=4)

    paired = gene_df[["pearson_before", "pearson_after"]].dropna()
    if len(paired) > 0:
        try:
            pval = wilcoxon(
                paired["pearson_before"].values,
                paired["pearson_after"].values,
                zero_method="wilcox",
            ).pvalue
        except Exception:
            pval = np.nan
    else:
        pval = np.nan

    ymax = np.nanmax(np.r_[gene_df["pearson_before"].values, gene_df["pearson_after"].values])
    ymin = np.nanmin(np.r_[gene_df["pearson_before"].values, gene_df["pearson_after"].values])
    ypad = 0.08 * (ymax - ymin + 1e-6)

    ax.plot([0, 0, 1, 1], [ymax + ypad, ymax + 2 * ypad, ymax + 2 * ypad, ymax + ypad], color="black", lw=1)
    ax.text(0.5, ymax + 2.2 * ypad, f"p = {pval:.2e}" if np.isfinite(pval) else "p = NA",
            ha="center", va="bottom", fontsize=10)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before", "After"])
    ax.set_ylabel("Pearson correlation")
    ax.set_title("All evaluable genes")
    ax.text(0.02, 0.04, f"n = {len(gene_df)} genes", transform=ax.transAxes, fontsize=9, color="0.35")
    ax.grid(axis="y", alpha=0.25)

    # Panel B
    ax = axes[1]
    ranked = gene_df.sort_values("delta_pearson", ascending=False).reset_index(drop=True)
    ax.plot(
        np.arange(len(ranked)),
        ranked["delta_pearson"].values,
        color="black",
        linewidth=1.8,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Genes ranked by improvement")
    ax.set_ylabel("Δ Pearson (after - before)")
    ax.set_title("Ranked gene improvement")

    topn = min(10, len(ranked))
    for i in range(topn):
        ax.text(
            i,
            ranked.loc[i, "delta_pearson"],
            ranked.loc[i, "gene"],
            fontsize=8,
            rotation=45,
            ha="left",
            va="bottom",
        )

    fig.suptitle("Coarse-grid gene-map alignment improves after AXIS transformation", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved figure: {save_path}")

    plt.show()


def plot_gene_delta_histogram(
    gene_df: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    vals = gene_df["delta_pearson"].dropna().values

    ax.hist(vals, bins=30, color="0.55", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(np.mean(vals), color="crimson", linewidth=2)

    ax.set_xlabel("Δ Pearson (after - before)")
    ax.set_ylabel("Number of genes")
    ax.set_title("Distribution of Pearson improvement")
    ax.grid(axis="y", alpha=0.25)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved figure: {save_path}")

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = collect_all_gene_metrics(
        patient_ids=PATIENT_IDS,
        pair_list=PAIR_LIST,
        out_csv=OUT_CSV,
        bin_size_px=BIN_SIZE_PX,
    )
    # df = pd.read_csv(OUT_CSV)

    gene_df = summarize_by_gene(df)

    plot_gene_level_pearson_summary(
        gene_df,
        # save_path=OUTDIR / "coarse_grid_pearson_summary.png",
        save_path=None
    )

    # optional second figure
    # plot_gene_delta_histogram(
    #     gene_df,
    #     save_path=OUTDIR / "coarse_grid_pearson_delta_histogram.png",
    # )


if __name__ == "__main__":
    main()