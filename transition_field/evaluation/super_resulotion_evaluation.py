#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

import glob
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import wilcoxon

try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False


# ============================================================
# Config
# ============================================================

AXIS_ROOT = Path("/media/huifang/data/registration/SCC/huifang/pairwise_output")
ISTAR_ROOT = Path("/media/huifang/data/registration/SCC/istar/single_input")

PATIENT_IDS = [2, 5, 9, 10]
PAIR_LIST = [(0, 1), (0, 2), (1, 2)]
GENE_NAMES = ["A2ML1","COL17A1","KRT1"]   # add more marker genes here if desired

OUTDIR = Path("/media/huifang/data/registration/SCC/istar_alignment_eval")
OUT_CSV = OUTDIR / "istar_highres_gene_alignment_metrics.csv"

TOP_PERCENTILE = 90.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Current loader / warp logic (kept compatible)
# ============================================================


def list_genes_for_sample(patient_id: int, axis_id: int, sample_id: int, istar_root) -> set:
    cnts_dir = (
        Path(istar_root)
        / f"group_{patient_id}"
        / f"axis_{axis_id}"
        / f"sample{sample_id}"
        / "cnts-super"
    )
    if not cnts_dir.is_dir():
        raise FileNotFoundError(f"cnts-super folder missing: {cnts_dir}")

    genes = {p.stem for p in cnts_dir.glob("*.pickle")}
    return genes


def list_common_genes_for_pair(patient_id: int, fixed_id: int, moving_id: int, istar_root) -> list:
    genes_fixed = list_genes_for_sample(patient_id, fixed_id, fixed_id, istar_root)
    genes_moving = list_genes_for_sample(patient_id, moving_id, moving_id, istar_root)
    common = sorted(genes_fixed & genes_moving)
    return common









def load_flow_np(patient_id: int, fixed_id: int, moving_id: int) -> np.ndarray:
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

def gene_is_evaluable(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    valid_mask: np.ndarray,
    min_valid_pixels: int = 200,
    min_std: float = 1e-4,
) -> bool:
    va = fixed_img[valid_mask]
    vb = moving_img[valid_mask]

    finite = np.isfinite(va) & np.isfinite(vb)
    va = va[finite]
    vb = vb[finite]

    if va.size < min_valid_pixels:
        return False
    if np.std(va) < min_std or np.std(vb) < min_std:
        return False
    return True


def warp_gene_with_axis_flow(gene_map_np: np.ndarray, flow_np: np.ndarray) -> np.ndarray:
    """
    Same SpatialTransformer-style pull warp used in your current script.
    """
    gene_np = np.asarray(gene_map_np, dtype=np.float32)
    src = torch.from_numpy(gene_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    f = np.asarray(flow_np, dtype=np.float32)
    if f.ndim != 3:
        raise ValueError(f"Flow must be 3D, got {f.shape}")

    if f.shape[0] == 2:
        pass
    elif f.shape[-1] == 2:
        f = np.moveaxis(f, -1, 0)
    else:
        raise ValueError(f"Flow must have channel dim 2, got {f.shape}")

    _, Hf, Wf = f.shape
    flow = torch.from_numpy(f).float().unsqueeze(0).to(DEVICE)

    src = F.interpolate(src, size=(Hf, Wf), mode="bilinear", align_corners=False)

    shape = flow.shape[2:]
    vectors = [torch.arange(0, s, device=DEVICE, dtype=torch.float32) for s in shape]
    grids = torch.meshgrid(vectors, indexing="ij")
    grid = torch.stack(grids, dim=0).unsqueeze(0)

    new_locs = grid + flow

    for i, s in enumerate(shape):
        new_locs[:, i, ...] = 2.0 * (new_locs[:, i, ...] / (s - 1.0) - 0.5)

    new_locs = new_locs.permute(0, 2, 3, 1)
    new_locs = new_locs[..., [1, 0]]

    warped = F.grid_sample(
        src,
        new_locs,
        align_corners=False,
        mode="bilinear",
        padding_mode="border",
    )

    return warped.squeeze().detach().cpu().numpy().astype(np.float32)


# ============================================================
# Evaluation helpers
# ============================================================

def warp_value_and_mask_to_fixed(
    gene_map: np.ndarray,
    mask: np.ndarray,
    flow_np: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp a gene map and its tissue mask to the fixed-frame flow grid.

    Important:
    - fill background with zero before warping
    - warp the binary mask separately
    - reset outside-mask pixels to NaN afterward
    """
    gene_map = np.asarray(gene_map, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.uint8)

    gene_filled = np.where(mask > 0, gene_map, 0.0).astype(np.float32)

    warped_val = warp_gene_with_axis_flow(gene_filled, flow_np)
    warped_mask = warp_gene_with_axis_flow(mask.astype(np.float32), flow_np)
    warped_mask = warped_mask > 0.5

    warped_val = warped_val.astype(np.float32)
    warped_val[~warped_mask] = np.nan
    return warped_val, warped_mask


def masked_pearson(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    va = a[valid_mask]
    vb = b[valid_mask]
    finite = np.isfinite(va) & np.isfinite(vb)

    va = va[finite]
    vb = vb[finite]

    if va.size < 3:
        return np.nan
    if np.std(va) < 1e-8 or np.std(vb) < 1e-8:
        return np.nan

    return float(np.corrcoef(va, vb)[0, 1])


def masked_ssim(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    if not HAVE_SKIMAGE:
        return np.nan

    ys, xs = np.where(valid_mask)
    if len(xs) < 25:
        return np.nan

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    a_crop = a[y0:y1, x0:x1].copy()
    b_crop = b[y0:y1, x0:x1].copy()
    m_crop = valid_mask[y0:y1, x0:x1]

    finite = np.isfinite(a_crop) & np.isfinite(b_crop) & m_crop
    if finite.sum() < 25:
        return np.nan

    a_fill = a_crop.copy()
    b_fill = b_crop.copy()

    mean_a = float(np.nanmean(a_crop[finite]))
    mean_b = float(np.nanmean(b_crop[finite]))
    a_fill[~finite] = mean_a
    b_fill[~finite] = mean_b

    data_range = float(max(np.nanmax([a_fill[finite].max(), b_fill[finite].max()]) -
                           np.nanmin([a_fill[finite].min(), b_fill[finite].min()]), 1e-6))

    win_size = min(7, min(a_fill.shape))
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return np.nan

    return float(ssim(a_fill, b_fill, data_range=data_range, win_size=win_size))


def top_percentile_mask(img: np.ndarray, valid_mask: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    vals = img[valid_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(valid_mask, dtype=bool)

    thr = np.percentile(vals, percentile)
    return valid_mask & np.isfinite(img) & (img >= thr)

def evaluate_before_after_metrics_for_pair(
    patient_id: int,
    fixed_id: int,
    moving_id: int,
    gene_name: str,
    top_percentile: float = TOP_PERCENTILE,
) -> Dict[str, float]:
    """
    Evaluate before vs after warping for one patient/pair/gene.
    """
    flow = load_flow_np(patient_id, fixed_id, moving_id)
    _, Hf, Wf = flow.shape
    zero_flow = np.zeros((2, Hf, Wf), dtype=np.float32)

    # load in each slice's own iSTAR axis/sample frame
    fixed_gene = load_gene_map_np(patient_id, fixed_id, fixed_id, gene_name)
    fixed_mask = load_mask_np(patient_id, fixed_id, fixed_id)

    moving_gene = load_gene_map_np(patient_id, moving_id, moving_id, gene_name)
    moving_mask = load_mask_np(patient_id, moving_id, moving_id)

    # fixed slice in fixed frame
    fixed_img, fixed_valid = warp_value_and_mask_to_fixed(fixed_gene, fixed_mask, zero_flow)

    # moving slice baseline (resize only, no warp)
    moving_before_img, moving_before_valid = warp_value_and_mask_to_fixed(moving_gene, moving_mask, zero_flow)

    # moving slice after AXIS warp
    moving_after_img, moving_after_valid = warp_value_and_mask_to_fixed(moving_gene, moving_mask, flow)

    common_before = fixed_valid & moving_before_valid
    common_after = fixed_valid & moving_after_valid

    fixed_top_before = top_percentile_mask(fixed_img, common_before, percentile=top_percentile)
    moving_top_before = top_percentile_mask(moving_before_img, common_before, percentile=top_percentile)

    fixed_top_after = top_percentile_mask(fixed_img, common_after, percentile=top_percentile)
    moving_top_after = top_percentile_mask(moving_after_img, common_after, percentile=top_percentile)

    row = {
        "patient_id": patient_id,
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "gene": gene_name,
        "n_common_before": int(common_before.sum()),
        "n_common_after": int(common_after.sum()),

        "pearson_before": masked_pearson(fixed_img, moving_before_img, common_before),
        "pearson_after": masked_pearson(fixed_img, moving_after_img, common_after),

        "ssim_before": masked_ssim(fixed_img, moving_before_img, common_before),
        "ssim_after": masked_ssim(fixed_img, moving_after_img, common_after),

    }

    return row


def collect_alignment_metrics(
    patient_ids: Iterable[int],
    pair_list: Iterable[Tuple[int, int]],
    gene_names: Iterable[str],
    out_csv: Optional[Path] = OUT_CSV,
    top_percentile: float = TOP_PERCENTILE,
) -> pd.DataFrame:
    rows = []

    for pid in patient_ids:
        for fixed_id, moving_id in pair_list:
            for gene in gene_names:
                print(f"[INFO] patient={pid}, pair={fixed_id}->{moving_id}, gene={gene}")
                try:
                    row = evaluate_before_after_metrics_for_pair(
                        patient_id=pid,
                        fixed_id=fixed_id,
                        moving_id=moving_id,
                        gene_name=gene,
                        top_percentile=top_percentile,
                    )
                    rows.append(row)
                except FileNotFoundError as e:
                    print(f"[WARN] skip patient={pid}, pair={fixed_id}->{moving_id}, gene={gene}: {e}")

    if not rows:
        raise RuntimeError("No alignment metrics collected.")

    df = pd.DataFrame(rows)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[OK] saved metrics csv: {out_csv}")
    return df

def evaluate_gene_alignment_for_pair(
    patient_id: int,
    fixed_id: int,
    moving_id: int,
    gene_name: str,
):
    flow = load_flow_np(patient_id, fixed_id, moving_id)
    _, Hf, Wf = flow.shape
    zero_flow = np.zeros((2, Hf, Wf), dtype=np.float32)

    fixed_gene = load_gene_map_np(patient_id, fixed_id, fixed_id, gene_name)
    fixed_mask = load_mask_np(patient_id, fixed_id, fixed_id)

    moving_gene = load_gene_map_np(patient_id, moving_id, moving_id, gene_name)
    moving_mask = load_mask_np(patient_id, moving_id, moving_id)

    fixed_img, fixed_valid = warp_value_and_mask_to_fixed(fixed_gene, fixed_mask, zero_flow)
    moving_before_img, moving_before_valid = warp_value_and_mask_to_fixed(moving_gene, moving_mask, zero_flow)
    moving_after_img, moving_after_valid = warp_value_and_mask_to_fixed(moving_gene, moving_mask, flow)

    common_before = fixed_valid & moving_before_valid
    common_after = fixed_valid & moving_after_valid

    if not gene_is_evaluable(fixed_img, moving_before_img, common_before):
        return None

    pearson_before = masked_pearson(fixed_img, moving_before_img, common_before)
    pearson_after = masked_pearson(fixed_img, moving_after_img, common_after)

    return {
        "patient_id": patient_id,
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "gene": gene_name,
        "n_common_before": int(common_before.sum()),
        "n_common_after": int(common_after.sum()),
        "pearson_before": pearson_before,
        "pearson_after": pearson_after,
        "delta_pearson": (
            pearson_after - pearson_before
            if np.isfinite(pearson_before) and np.isfinite(pearson_after)
            else np.nan
        ),
    }

def collect_all_gene_alignment_metrics(
    patient_ids,
    pair_list,
    istar_root,
    out_csv,
):
    rows = []

    for pid in patient_ids:
        for fixed_id, moving_id in pair_list:
            print(f"[INFO] patient={pid}, pair={fixed_id}->{moving_id}")

            try:
                genes = list_common_genes_for_pair(
                    patient_id=pid,
                    fixed_id=fixed_id,
                    moving_id=moving_id,
                    istar_root=istar_root,
                )
            except FileNotFoundError as e:
                print(f"[WARN] skip patient={pid}, pair={fixed_id}->{moving_id}: {e}")
                continue

            for gene in genes:
                try:
                    row = evaluate_gene_alignment_for_pair(
                        patient_id=pid,
                        fixed_id=fixed_id,
                        moving_id=moving_id,
                        gene_name=gene,
                    )
                    if row is not None:
                        rows.append(row)
                except Exception as e:
                    print(f"[WARN] failed gene={gene} for patient={pid}, pair={fixed_id}->{moving_id}: {e}")

    if not rows:
        raise RuntimeError("No valid gene-level alignment results collected.")

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] saved gene-level csv: {out_csv}")
    return df
# ============================================================
# Plotting
# ============================================================

def _paired_metric_plot(ax, df: pd.DataFrame, metric_base: str):
    before_col = f"{metric_base}_before"
    after_col = f"{metric_base}_after"

    plot_df = df[[before_col, after_col]].copy().dropna()
    if plot_df.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Before", "After"])
        ax.set_title(metric_base)
        return

    x0, x1 = 0, 1

    # paired lines
    for _, row in plot_df.iterrows():
        ax.plot(
            [x0, x1],
            [row[before_col], row[after_col]],
            color="0.75",
            alpha=0.8,
            linewidth=1.2,
            zorder=1,
        )

    # points
    ax.scatter(
        np.full(len(plot_df), x0),
        plot_df[before_col].values,
        color="white",
        edgecolor="black",
        s=45,
        zorder=3,
    )
    ax.scatter(
        np.full(len(plot_df), x1),
        plot_df[after_col].values,
        color="crimson",
        edgecolor="black",
        s=45,
        zorder=3,
    )

    # mean diamonds
    mean_before = plot_df[before_col].mean()
    mean_after = plot_df[after_col].mean()
    ax.scatter([x0, x1], [mean_before, mean_after], marker="D", s=90, color="black", zorder=4)

    # paired test
    try:
        stat = wilcoxon(plot_df[before_col].values, plot_df[after_col].values, zero_method="wilcox")
        pval = stat.pvalue
    except Exception:
        pval = np.nan

    # annotation
    y_max = np.nanmax(np.concatenate([plot_df[before_col].values, plot_df[after_col].values]))
    y_min = np.nanmin(np.concatenate([plot_df[before_col].values, plot_df[after_col].values]))
    y_pad = 0.08 * (y_max - y_min + 1e-6)

    ax.plot([x0, x0, x1, x1], [y_max + y_pad, y_max + 2*y_pad, y_max + 2*y_pad, y_max + y_pad], color="black", lw=1)
    ptxt = f"p = {pval:.2e}" if np.isfinite(pval) else "p = NA"
    ax.text(0.5, y_max + 2.2*y_pad, ptxt, ha="center", va="bottom", fontsize=10)

    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["Before", "After"])
    ax.grid(axis="y", alpha=0.25)

    title_map = {
        "pearson": "Pearson correlation",
        "ssim": "SSIM",
    }
    ax.set_title(title_map.get(metric_base, metric_base))


def plot_alignment_summary(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 4.2))

    _paired_metric_plot(axes[0], df, "pearson")
    # _paired_metric_plot(axes[1], df, "ssim")


    fig.suptitle("High-resolution iSTAR gene-map alignment improves after AXIS warping", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved figure: {save_path}")

    plt.show()

def plot_gene_level_pearson_summary(gene_df: pd.DataFrame, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # A: paired before/after
    ax = axes[0]
    for _, row in gene_df.iterrows():
        ax.plot([0, 1], [row["pearson_before"], row["pearson_after"]],
                color="0.8", alpha=0.5, linewidth=0.8)
    ax.scatter(np.zeros(len(gene_df)), gene_df["pearson_before"], s=18, color="white", edgecolor="black", zorder=3)
    ax.scatter(np.ones(len(gene_df)), gene_df["pearson_after"], s=18, color="crimson", edgecolor="black", zorder=3)
    ax.scatter([0, 1],
               [gene_df["pearson_before"].mean(), gene_df["pearson_after"].mean()],
               marker="D", s=90, color="black", zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before", "After"])
    ax.set_ylabel("Pearson correlation")
    ax.set_title("All evaluable genes")

    # B: delta distribution
    ax = axes[1]
    ax.hist(gene_df["delta_pearson"].dropna(), bins=30, color="0.5", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(gene_df["delta_pearson"].mean(), color="crimson", linewidth=2)
    ax.set_xlabel("Δ Pearson (after - before)")
    ax.set_ylabel("Number of genes")
    ax.set_title("Improvement distribution")

    # C: ranked gene improvement
    ax = axes[2]
    ranked = gene_df.sort_values("delta_pearson", ascending=False).reset_index(drop=True)
    ax.plot(np.arange(len(ranked)), ranked["delta_pearson"], color="black", linewidth=1.8)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Genes ranked by Δ Pearson")
    ax.set_ylabel("Δ Pearson")
    ax.set_title("Ranked gene improvement")

    # label top genes
    topn = min(8, len(ranked))
    for i in range(topn):
        ax.text(i, ranked.loc[i, "delta_pearson"], ranked.loc[i, "gene"], fontsize=8, rotation=45)

    fig.suptitle("High-resolution gene-map alignment improves after AXIS warping", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def aggregate_pairwise_pearson(
    df,
    before_col="pearson_before",
    after_col="pearson_after",
    pair_keys=("patient_id", "fixed_id", "moving_id"),
):
    """
    Average gene-level Pearson scores into one row per pair.

    Expected input:
        one row per (patient_id, fixed_id, moving_id, gene)

    Required columns:
        before_col, after_col, and pair_keys
    """
    required = list(pair_keys) + [before_col, after_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    pair_df = (
        df.groupby(list(pair_keys), as_index=False)
        .agg(
            pearson_before=(before_col, "mean"),
            pearson_after=(after_col, "mean"),
            n_genes=(before_col, "size"),
        )
    )

    pair_df["delta_pearson"] = pair_df["pearson_after"] - pair_df["pearson_before"]

    if set(pair_keys) == {"patient_id", "fixed_id", "moving_id"}:
        pair_df["pair_label"] = (
            "P" + pair_df["patient_id"].astype(str)
            + ": " + pair_df["fixed_id"].astype(str)
            + "→" + pair_df["moving_id"].astype(str)
        )
    else:
        pair_df["pair_label"] = (
            pair_df[list(pair_keys)]
            .astype(str)
            .agg(" | ".join, axis=1)
        )

    return pair_df


def plot_pairwise_mean_pearson_trend(
    df,
    before_col="pearson_before",
    after_col="pearson_after",
    pair_keys=("patient_id", "fixed_id", "moving_id"),
    sort_by="delta",
    annotate_delta=True,
    annotate_pair_label=True,
    pair_label_dx=0.03,
    delta_dx=0.22,
    figsize=(5.8, 6.2),
    save_path=None,
):
    pair_df = aggregate_pairwise_pearson(
        df,
        before_col=before_col,
        after_col=after_col,
        pair_keys=pair_keys,
    )

    if sort_by == "delta":
        pair_df = pair_df.sort_values("delta_pearson", ascending=True).reset_index(drop=True)
    elif sort_by == "after":
        pair_df = pair_df.sort_values("pearson_after", ascending=True).reset_index(drop=True)
    elif sort_by == "before":
        pair_df = pair_df.sort_values("pearson_before", ascending=True).reset_index(drop=True)
    elif sort_by is None:
        pair_df = pair_df.reset_index(drop=True)
    else:
        raise ValueError("sort_by must be one of {'delta', 'after', 'before', None}")

    fig, ax = plt.subplots(figsize=figsize)
    x = [0, 1]

    for _, row in pair_df.iterrows():
        y0 = row["pearson_before"]
        y1 = row["pearson_after"]
        delta = row["delta_pearson"]

        line_color = "crimson" if delta >= 0 else "steelblue"

        ax.plot(
            x,
            [y0, y1],
            color=line_color,
            alpha=0.45,
            linewidth=1.5,
            zorder=1,
        )

        ax.scatter(
            [0], [y0],
            s=40,
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            [1], [y1],
            s=40,
            facecolor=line_color,
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )

        if annotate_pair_label:
            ax.text(
                1 + pair_label_dx,
                y1,
                row["pair_label"],
                ha="left",
                va="center",
                fontsize=8.3,
                color="black",
            )

        if annotate_delta:
            ax.text(
                1 + delta_dx,
                y1,
                f"{delta:+.3f}",
                ha="left",
                va="center",
                fontsize=8.3,
                color=line_color,
            )

    mean_before = pair_df["pearson_before"].mean()
    mean_after = pair_df["pearson_after"].mean()

    ax.plot(
        x,
        [mean_before, mean_after],
        color="black",
        linewidth=2.4,
        linestyle="--",
        zorder=5,
    )
    ax.scatter(
        x,
        [mean_before, mean_after],
        marker="D",
        s=85,
        color="black",
        zorder=6,
        label="Mean across pairs",
    )

    if HAVE_SCIPY and len(pair_df) > 0:
        try:
            pval = wilcoxon(
                pair_df["pearson_before"].values,
                pair_df["pearson_after"].values,
                zero_method="wilcox",
            ).pvalue
            ptxt = f"Wilcoxon p = {pval:.2e}"
        except Exception:
            ptxt = "Wilcoxon p = NA"
    else:
        ptxt = None

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before", "After"])
    ax.set_ylabel("Mean Pearson correlation across genes")
    ax.set_title("Pair-wise Pearson comparison before and after warping")

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.02, 0.04,
        f"n = {len(pair_df)} pairs",
        transform=ax.transAxes,
        fontsize=9,
        color="0.35",
    )

    if ptxt is not None:
        ax.text(
            0.02, 1.02,
            ptxt,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
        )

    xmax = 1.08
    if annotate_pair_label:
        xmax = max(xmax, 1 + pair_label_dx + 0.14)
    if annotate_delta:
        xmax = max(xmax, 1 + delta_dx + 0.10)
    ax.set_xlim(-0.1, xmax)

    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def aggregate_pairwise_pearson(
    df,
    before_col="pearson_before",
    after_col="pearson_after",
    pair_keys=("patient_id", "fixed_id", "moving_id"),
):
    """
    Average gene-level Pearson scores into one row per pair.
    """
    required = list(pair_keys) + [before_col, after_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    pair_df = (
        df.groupby(list(pair_keys), as_index=False)
        .agg(
            pearson_before=(before_col, "mean"),
            pearson_after=(after_col, "mean"),
            n_genes=(before_col, "size"),
        )
    )

    pair_df["delta_pearson"] = pair_df["pearson_after"] - pair_df["pearson_before"]
    pair_df["pair_label"] = (
        pair_df["fixed_id"].astype(str) + "→" + pair_df["moving_id"].astype(str)
    )
    return pair_df


def _nudge_y_positions(y_values, min_gap=0.012):
    """
    Nudge text y positions slightly to reduce overlap.
    """
    y_values = list(y_values)
    if len(y_values) == 0:
        return []

    order = np.argsort(y_values)
    y_sorted = [y_values[i] for i in order]
    y_adj = y_sorted.copy()

    for i in range(1, len(y_adj)):
        if y_adj[i] - y_adj[i - 1] < min_gap:
            y_adj[i] = y_adj[i - 1] + min_gap

    out = [None] * len(y_values)
    for idx_sorted, idx_orig in enumerate(order):
        out[idx_orig] = y_adj[idx_sorted]
    return out


def _plot_one_patient_subplot(
    ax,
    sub,
    patient_id,
    annotate_pair_label=True,
    annotate_delta=True,
    pair_label_dx=0.03,
    delta_dx=0.18,
    label_min_gap=0.02,
    show_ylabel=False,
):
    x = [0, 1]

    # gray pair lines
    for _, row in sub.iterrows():
        y0 = row["pearson_before"]
        y1 = row["pearson_after"]

        ax.plot(
            x,
            [y0, y1],
            color="0.65",
            alpha=0.85,
            linewidth=1.5,
            zorder=1,
        )

        ax.scatter(
            [0], [y0],
            s=38,
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            [1], [y1],
            s=38,
            facecolor="0.35",
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )

    # overall mean trend
    mean_before = sub["pearson_before"].mean()
    mean_after = sub["pearson_after"].mean()

    ax.plot(
        x,
        [mean_before, mean_after],
        color="black",
        linewidth=2.3,
        linestyle="--",
        zorder=5,
    )
    ax.scatter(
        x,
        [mean_before, mean_after],
        marker="D",
        s=75,
        color="black",
        zorder=6,
    )

    # text labels near the after point
    y_after = sub["pearson_after"].to_numpy()
    y_text = _nudge_y_positions(y_after, min_gap=label_min_gap)


    for i, (_, row) in enumerate(sub.iterrows()):
        y1 = row["pearson_after"]
        ylab = y_text[i]
        delta = row["delta_pearson"]
        delta_color = "crimson" if delta >= 0 else "steelblue"

        if annotate_pair_label:
            ax.text(
                1.1 + pair_label_dx,
                ylab,
                row["pair_label"],
                ha="left",
                va="center",
                fontsize=11,
                color="black",
            )

        if annotate_delta:
            ax.text(
                1 + delta_dx,
                ylab,
                f"{delta:+.3f}",
                ha="left",
                va="center",
                fontsize=13,
                color=delta_color,
            )

    # paired significance
    if HAVE_SCIPY and len(sub) > 0:
        try:
            pval = wilcoxon(
                sub["pearson_before"].values,
                sub["pearson_after"].values,
                zero_method="wilcox",
            ).pvalue
            ptxt = f"p = {pval:.2e}"
        except Exception:
            ptxt = "p = NA"
    else:
        ptxt = None

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["iStar", "iStar+Axis"],fontsize=14)
    ax.set_title(f"Patient {patient_id}", fontsize=14)

    if show_ylabel:
        ax.set_ylabel("Spatial Pearson correlation across genes",fontsize=13)
    else:
        ax.set_ylabel("")

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    # if ptxt is not None:
    #     ax.text(
    #         0.02, 1.02,
    #         ptxt,
    #         transform=ax.transAxes,
    #         ha="left",
    #         va="bottom",
    #         fontsize=9,
    #     )

    xmax = 1.08
    if annotate_pair_label:
        xmax = max(xmax, 1 + pair_label_dx + 0.08)
    if annotate_delta:
        xmax = max(xmax, 1 + delta_dx + 0.08)
    ax.set_xlim(-0.1, xmax)


def plot_pairwise_mean_pearson_trend_4patients(
    df,
    before_col="pearson_before",
    after_col="pearson_after",
    pair_keys=("patient_id", "fixed_id", "moving_id"),
    sort_by="delta",
    patient_ids=None,
    annotate_pair_label=True,
    annotate_delta=True,
    pair_label_dx=0.03,
    delta_dx=0.18,
    label_min_gap=0.03,
    figsize=(11, 8),
    save_path=None,
):
    """
    One 2x2 figure with 4 subplots, one subplot per patient.
    """
    pair_df = aggregate_pairwise_pearson(
        df,
        before_col=before_col,
        after_col=after_col,
        pair_keys=pair_keys,
    )

    if patient_ids is None:
        patient_ids = sorted(pair_df["patient_id"].unique())

    if len(patient_ids) != 4:
        raise ValueError(
            f"This function expects exactly 4 patients for a 2x2 figure, got {len(patient_ids)}: {patient_ids}"
        )

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    axes = axes.flatten()

    # global y-limits for consistency
    y_all = np.r_[pair_df["pearson_before"].values, pair_df["pearson_after"].values]
    y_min = np.nanmin(y_all)
    y_max = np.nanmax(y_all)
    y_pad = 0.08 * (y_max - y_min + 1e-6)

    for idx, pid in enumerate(patient_ids):
        ax = axes[idx]
        sub = pair_df[pair_df["patient_id"] == pid].copy()

        if sort_by == "delta":
            sub = sub.sort_values("delta_pearson", ascending=True).reset_index(drop=True)
        elif sort_by == "after":
            sub = sub.sort_values("pearson_after", ascending=True).reset_index(drop=True)
        elif sort_by == "before":
            sub = sub.sort_values("pearson_before", ascending=True).reset_index(drop=True)
        elif sort_by is None:
            sub = sub.reset_index(drop=True)
        else:
            raise ValueError("sort_by must be one of {'delta', 'after', 'before', None}")

        _plot_one_patient_subplot(
            ax=ax,
            sub=sub,
            patient_id=pid,
            annotate_pair_label=False,
            annotate_delta=True,
            pair_label_dx=pair_label_dx,
            delta_dx=delta_dx,
            label_min_gap=label_min_gap,
            show_ylabel=(idx == 0),
        )

        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    fig.suptitle("Pair-wise Pearson comparison before and after warping", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
# ============================================================
# Main
# ============================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # df = collect_alignment_metrics(
    #     patient_ids=PATIENT_IDS,
    #     pair_list=PAIR_LIST,
    #     gene_names=GENE_NAMES,
    #     out_csv=OUT_CSV,
    #     top_percentile=TOP_PERCENTILE,
    # )
    #
    # plot_alignment_summary(
    #     df,
    #     # save_path=OUTDIR / "istar_highres_gene_alignment_summary.png",
    #     save_path=None,
    # )
    # df = collect_all_gene_alignment_metrics(
    #     patient_ids=PATIENT_IDS,
    #     pair_list=PAIR_LIST,
    #     istar_root=ISTAR_ROOT,
    #     out_csv=OUT_CSV,
    # )
    df = pd.read_csv("/media/huifang/data/registration/SCC/istar_alignment_eval/istar_highres_gene_alignment_metrics_all.csv")

    plot_pairwise_mean_pearson_trend_4patients(
        df,
        before_col="pearson_before",
        after_col="pearson_after",
        pair_keys=("patient_id", "fixed_id", "moving_id"),
        patient_ids=[2, 5, 9, 10],  # your 4 SCC patients
        sort_by="delta",
        annotate_pair_label=True,
        annotate_delta=True,
        pair_label_dx=0.03,
        delta_dx=0.17,
        figsize=(10, 5),
        save_path=None,
    )
    # plot_pairwise_mean_pearson_trend(
    #     df,
    #     before_col="pearson_before",
    #     after_col="pearson_after",
    #     pair_keys=("patient_id", "fixed_id", "moving_id"),
    #     sort_by="delta",
    #     annotate_delta=True,
    #     figsize=(5.2, 6.0),
    #     save_path=None,
    # )
    # plot_gene_level_pearson_summary(df,save_path=None)



if __name__ == "__main__":
    main()