#!/usr/bin/env python3
from __future__ import annotations
from axis.settings import data_path

import glob
import pickle
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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

AXIS_ROOT = Path(data_path("SCC/huifang/pairwise_output"))
ISTAR_ROOT = Path(data_path("SCC/istar/single_input"))

PATIENT_IDS = [2, 5, 9, 10]
PAIR_LIST = [(0, 1), (0, 2), (1, 2)]
GENE_NAMES = ["A2ML1", "COL17A1", "KRT1"]

OUTDIR = Path(data_path("SCC/istar_alignment_eval"))
OUT_CSV = OUTDIR / "istar_highres_gene_alignment_metrics.csv"
ALL_GENE_CSV = OUTDIR / "istar_highres_gene_alignment_metrics_all.csv"
FIGURE_PATH = Path(
    data_path("paper_analysis/"
    "Figure2_transition_evaluation/m_istar_metric.png")
)

TOP_PERCENTILE = 90.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Final 4-patient figure controls.
PATIENT_FIGSIZE = (10, 3.7)
PATIENT_SUBPLOT_WSPACE = 0.75   # horizontal gap between neighboring patient panels
PATIENT_DELTA_DX = 0.17
PATIENT_LABEL_MIN_GAP = 0.04
PATIENT_FIG_DPI = 400


# ============================================================
# Data loading and warping
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
    return {p.stem for p in cnts_dir.glob("*.pickle")}


def list_common_genes_for_pair(patient_id: int, fixed_id: int, moving_id: int, istar_root) -> list:
    genes_fixed = list_genes_for_sample(patient_id, fixed_id, fixed_id, istar_root)
    genes_moving = list_genes_for_sample(patient_id, moving_id, moving_id, istar_root)
    return sorted(genes_fixed & genes_moving)


def load_flow_np(patient_id: int, fixed_id: int, moving_id: int) -> np.ndarray:
    group_dir = AXIS_ROOT / f"group_{patient_id}"
    if not group_dir.is_dir():
        raise FileNotFoundError(f"AXIS group dir not found: {group_dir}")

    matches = glob.glob(str(group_dir / f"{fixed_id}_{moving_id}_*flow.npy"))
    if matches:
        flow_path = matches[0]
    else:
        flow_path = group_dir / f"{fixed_id}_{moving_id}_flow.npy"
        if not flow_path.is_file():
            raise FileNotFoundError(
                f"No flow file found for {fixed_id}_{moving_id} in {group_dir}"
            )

    flow = np.load(flow_path).astype(np.float32)
    if flow.ndim != 3:
        raise ValueError(f"Unexpected flow dims {flow.shape} in {flow_path}")

    if flow.shape[0] == 2:
        return flow
    if flow.shape[-1] == 2:
        return np.moveaxis(flow, -1, 0)
    raise ValueError(f"Flow must have channel dimension 2, got {flow.shape}")


def load_gene_map_np(patient_id: int, axis_id: int, sample_id: int, gene_name: str) -> np.ndarray:
    pkl_path = (
        ISTAR_ROOT
        / f"group_{patient_id}"
        / f"axis_{axis_id}"
        / f"sample{sample_id}"
        / "cnts-super"
        / f"{gene_name}.pickle"
    )
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Gene map missing: {pkl_path}")

    with open(pkl_path, "rb") as f:
        arr = np.asarray(pickle.load(f), dtype=np.float32)

    if arr.ndim == 3:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Gene map {pkl_path} must be 2D, got {arr.shape}")
    return arr


def load_mask_np(patient_id: int, axis_id: int, sample_id: int) -> np.ndarray:
    mask_path = (
        ISTAR_ROOT
        / f"group_{patient_id}"
        / f"axis_{axis_id}"
        / f"sample{sample_id}"
        / "mask-small.png"
    )
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file missing: {mask_path}")

    mask_arr = np.asarray(Image.open(mask_path))
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    return (mask_arr > 0).astype(np.uint8)


def warp_gene_with_axis_flow(gene_map_np: np.ndarray, flow_np: np.ndarray) -> np.ndarray:
    """SpatialTransformer-style pull warp used by the AXIS evaluation."""
    src = torch.from_numpy(np.asarray(gene_map_np, dtype=np.float32))
    src = src.unsqueeze(0).unsqueeze(0).to(DEVICE)

    flow_arr = np.asarray(flow_np, dtype=np.float32)
    if flow_arr.ndim != 3:
        raise ValueError(f"Flow must be 3D, got {flow_arr.shape}")
    if flow_arr.shape[0] != 2:
        if flow_arr.shape[-1] == 2:
            flow_arr = np.moveaxis(flow_arr, -1, 0)
        else:
            raise ValueError(f"Flow must have channel dim 2, got {flow_arr.shape}")

    _, height, width = flow_arr.shape
    flow = torch.from_numpy(flow_arr).float().unsqueeze(0).to(DEVICE)
    src = F.interpolate(src, size=(height, width), mode="bilinear", align_corners=False)

    vectors = [
        torch.arange(0, s, device=DEVICE, dtype=torch.float32)
        for s in flow.shape[2:]
    ]
    grid = torch.stack(torch.meshgrid(vectors, indexing="ij"), dim=0).unsqueeze(0)
    new_locs = grid + flow

    for i, size in enumerate(flow.shape[2:]):
        new_locs[:, i, ...] = 2.0 * (new_locs[:, i, ...] / (size - 1.0) - 0.5)

    new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
    warped = F.grid_sample(
        src,
        new_locs,
        align_corners=False,
        mode="bilinear",
        padding_mode="border",
    )
    return warped.squeeze().detach().cpu().numpy().astype(np.float32)


def warp_value_and_mask_to_fixed(
    gene_map: np.ndarray,
    mask: np.ndarray,
    flow_np: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp gene values and tissue mask into the fixed-frame flow grid."""
    gene_filled = np.where(mask > 0, gene_map, 0.0).astype(np.float32)
    warped_val = warp_gene_with_axis_flow(gene_filled, flow_np)
    warped_mask = warp_gene_with_axis_flow(mask.astype(np.float32), flow_np) > 0.5
    warped_val = warped_val.astype(np.float32)
    warped_val[~warped_mask] = np.nan
    return warped_val, warped_mask


# ============================================================
# Metrics
# ============================================================

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
    va, vb = va[finite], vb[finite]
    return (
        va.size >= min_valid_pixels
        and np.std(va) >= min_std
        and np.std(vb) >= min_std
    )


def masked_pearson(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    va = a[valid_mask]
    vb = b[valid_mask]
    finite = np.isfinite(va) & np.isfinite(vb)
    va, vb = va[finite], vb[finite]
    if va.size < 3 or np.std(va) < 1e-8 or np.std(vb) < 1e-8:
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
    a_fill[~finite] = float(np.nanmean(a_crop[finite]))
    b_fill[~finite] = float(np.nanmean(b_crop[finite]))

    vals = np.r_[a_fill[finite], b_fill[finite]]
    data_range = max(float(np.nanmax(vals) - np.nanmin(vals)), 1e-6)

    win_size = min(7, min(a_fill.shape))
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return np.nan

    return float(ssim(a_fill, b_fill, data_range=data_range, win_size=win_size))


def top_percentile_mask(
    img: np.ndarray,
    valid_mask: np.ndarray,
    percentile: float = TOP_PERCENTILE,
) -> np.ndarray:
    vals = img[valid_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(valid_mask, dtype=bool)
    threshold = np.percentile(vals, percentile)
    return valid_mask & np.isfinite(img) & (img >= threshold)


def evaluate_before_after_metrics_for_pair(
    patient_id: int,
    fixed_id: int,
    moving_id: int,
    gene_name: str,
    top_percentile: float = TOP_PERCENTILE,
) -> dict:
    flow = load_flow_np(patient_id, fixed_id, moving_id)
    _, height, width = flow.shape
    zero_flow = np.zeros((2, height, width), dtype=np.float32)

    fixed_gene = load_gene_map_np(patient_id, fixed_id, fixed_id, gene_name)
    fixed_mask = load_mask_np(patient_id, fixed_id, fixed_id)
    moving_gene = load_gene_map_np(patient_id, moving_id, moving_id, gene_name)
    moving_mask = load_mask_np(patient_id, moving_id, moving_id)

    fixed_img, fixed_valid = warp_value_and_mask_to_fixed(fixed_gene, fixed_mask, zero_flow)
    moving_before_img, moving_before_valid = warp_value_and_mask_to_fixed(
        moving_gene, moving_mask, zero_flow
    )
    moving_after_img, moving_after_valid = warp_value_and_mask_to_fixed(
        moving_gene, moving_mask, flow
    )

    common_before = fixed_valid & moving_before_valid
    common_after = fixed_valid & moving_after_valid

    # Kept for compatibility with the original evaluation pipeline.
    _ = top_percentile_mask(fixed_img, common_before, percentile=top_percentile)
    _ = top_percentile_mask(moving_before_img, common_before, percentile=top_percentile)
    _ = top_percentile_mask(fixed_img, common_after, percentile=top_percentile)
    _ = top_percentile_mask(moving_after_img, common_after, percentile=top_percentile)

    return {
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


def evaluate_gene_alignment_for_pair(
    patient_id: int,
    fixed_id: int,
    moving_id: int,
    gene_name: str,
):
    flow = load_flow_np(patient_id, fixed_id, moving_id)
    _, height, width = flow.shape
    zero_flow = np.zeros((2, height, width), dtype=np.float32)

    fixed_gene = load_gene_map_np(patient_id, fixed_id, fixed_id, gene_name)
    fixed_mask = load_mask_np(patient_id, fixed_id, fixed_id)
    moving_gene = load_gene_map_np(patient_id, moving_id, moving_id, gene_name)
    moving_mask = load_mask_np(patient_id, moving_id, moving_id)

    fixed_img, fixed_valid = warp_value_and_mask_to_fixed(fixed_gene, fixed_mask, zero_flow)
    moving_before_img, moving_before_valid = warp_value_and_mask_to_fixed(
        moving_gene, moving_mask, zero_flow
    )
    moving_after_img, moving_after_valid = warp_value_and_mask_to_fixed(
        moving_gene, moving_mask, flow
    )

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


def collect_alignment_metrics(
    patient_ids: Iterable[int],
    pair_list: Iterable[Tuple[int, int]],
    gene_names: Iterable[str],
    out_csv: Optional[Path] = OUT_CSV,
    top_percentile: float = TOP_PERCENTILE,
) -> pd.DataFrame:
    rows = []
    for patient_id in patient_ids:
        for fixed_id, moving_id in pair_list:
            for gene_name in gene_names:
                print(
                    f"[INFO] patient={patient_id}, pair={fixed_id}->{moving_id}, "
                    f"gene={gene_name}"
                )
                try:
                    rows.append(
                        evaluate_before_after_metrics_for_pair(
                            patient_id,
                            fixed_id,
                            moving_id,
                            gene_name,
                            top_percentile=top_percentile,
                        )
                    )
                except FileNotFoundError as exc:
                    print(
                        f"[WARN] skip patient={patient_id}, pair={fixed_id}->{moving_id}, "
                        f"gene={gene_name}: {exc}"
                    )

    if not rows:
        raise RuntimeError("No alignment metrics collected.")

    df = pd.DataFrame(rows)
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[OK] saved metrics csv: {out_csv}")
    return df


def collect_all_gene_alignment_metrics(
    patient_ids: Iterable[int],
    pair_list: Iterable[Tuple[int, int]],
    istar_root,
    out_csv: Path,
) -> pd.DataFrame:
    rows = []
    for patient_id in patient_ids:
        for fixed_id, moving_id in pair_list:
            print(f"[INFO] patient={patient_id}, pair={fixed_id}->{moving_id}")
            try:
                genes = list_common_genes_for_pair(
                    patient_id,
                    fixed_id,
                    moving_id,
                    istar_root,
                )
            except FileNotFoundError as exc:
                print(f"[WARN] skip patient={patient_id}, pair={fixed_id}->{moving_id}: {exc}")
                continue

            for gene_name in genes:
                try:
                    row = evaluate_gene_alignment_for_pair(
                        patient_id,
                        fixed_id,
                        moving_id,
                        gene_name,
                    )
                    if row is not None:
                        rows.append(row)
                except Exception as exc:
                    print(
                        f"[WARN] failed gene={gene_name} for patient={patient_id}, "
                        f"pair={fixed_id}->{moving_id}: {exc}"
                    )

    if not rows:
        raise RuntimeError("No valid gene-level alignment results collected.")

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] saved gene-level csv: {out_csv}")
    return df


# ============================================================
# Final patient-level plotting
# ============================================================

def aggregate_pairwise_pearson(
    df: pd.DataFrame,
    before_col: str = "pearson_before",
    after_col: str = "pearson_after",
    pair_keys=("patient_id", "fixed_id", "moving_id"),
) -> pd.DataFrame:
    """Average gene-level Pearson scores into one row per patient/slice pair."""
    required = list(pair_keys) + [before_col, after_col]
    missing = [col for col in required if col not in df.columns]
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


def _nudge_y_positions(y_values, min_gap: float = 0.012):
    """Nudge annotation y-positions to reduce text overlap."""
    y_values = list(y_values)
    if not y_values:
        return []

    order = np.argsort(y_values)
    y_sorted = [y_values[i] for i in order]
    y_adjusted = y_sorted.copy()

    for i in range(1, len(y_adjusted)):
        if y_adjusted[i] - y_adjusted[i - 1] < min_gap:
            y_adjusted[i] = y_adjusted[i - 1] + min_gap

    out = [None] * len(y_values)
    for idx_sorted, idx_original in enumerate(order):
        out[idx_original] = y_adjusted[idx_sorted]
    return out


def _plot_one_patient_subplot(
    ax,
    sub: pd.DataFrame,
    patient_id: int,
    annotate_delta: bool = True,
    delta_dx: float = PATIENT_DELTA_DX,
    label_min_gap: float = PATIENT_LABEL_MIN_GAP,
    show_ylabel: bool = False,
):
    x = [0, 1]

    for _, row in sub.iterrows():
        y0 = row["pearson_before"]
        y1 = row["pearson_after"]
        ax.plot(x, [y0, y1], color="0.65", alpha=0.85, linewidth=1.5, zorder=1)
        ax.scatter(
            [0], [y0], s=38, facecolor="white", edgecolor="black",
            linewidth=0.9, zorder=3,
        )
        ax.scatter(
            [1], [y1], s=38, facecolor="0.35", edgecolor="black",
            linewidth=0.7, zorder=4,
        )

    y_text = _nudge_y_positions(sub["pearson_after"].to_numpy(), min_gap=label_min_gap)
    for i, (_, row) in enumerate(sub.iterrows()):
        if not annotate_delta:
            continue
        delta = row["delta_pearson"]
        delta_color = "crimson" if delta >= 0 else "steelblue"
        ax.text(
            1 + delta_dx,
            y_text[i],
            f"{delta:+.3f}",
            ha="left",
            va="center",
            fontsize=13,
            color=delta_color,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before\nAXIS", "After\nAXIS"], fontsize=14)
    ax.set_title(f"Patient {patient_id}", fontsize=16)
    ax.set_ylabel("Spatial Pearson correlation" if show_ylabel else "", fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xmax = 1.08
    if annotate_delta:
        xmax = max(xmax, 1 + delta_dx + 0.08)
    ax.set_xlim(-0.1, xmax)


def plot_pairwise_mean_pearson_trend_4patients(
    df: pd.DataFrame,
    before_col: str = "pearson_before",
    after_col: str = "pearson_after",
    pair_keys=("patient_id", "fixed_id", "moving_id"),
    sort_by: Optional[str] = "delta",
    patient_ids=None,
    annotate_delta: bool = True,
    delta_dx: float = PATIENT_DELTA_DX,
    label_min_gap: float = PATIENT_LABEL_MIN_GAP,
    figsize=PATIENT_FIGSIZE,
    subplot_wspace: float = PATIENT_SUBPLOT_WSPACE,
    save_path: Optional[Path] = None,
):
    """Plot four patient panels in one row with directly controllable wspace."""
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
            f"This function expects exactly 4 patients, got {len(patient_ids)}: "
            f"{patient_ids}"
        )

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    axes = np.asarray(axes).ravel()

    y_all = np.r_[pair_df["pearson_before"].values, pair_df["pearson_after"].values]
    y_min = np.nanmin(y_all)
    y_max = np.nanmax(y_all)
    y_pad = 0.08 * (y_max - y_min + 1e-6)

    for idx, patient_id in enumerate(patient_ids):
        sub = pair_df[pair_df["patient_id"] == patient_id].copy()

        if sort_by == "delta":
            sub = sub.sort_values("delta_pearson", ascending=True)
        elif sort_by == "after":
            sub = sub.sort_values("pearson_after", ascending=True)
        elif sort_by == "before":
            sub = sub.sort_values("pearson_before", ascending=True)
        elif sort_by is not None:
            raise ValueError("sort_by must be one of {'delta', 'after', 'before', None}")
        sub = sub.reset_index(drop=True)

        _plot_one_patient_subplot(
            ax=axes[idx],
            sub=sub,
            patient_id=patient_id,
            annotate_delta=annotate_delta,
            delta_dx=delta_dx,
            label_min_gap=label_min_gap,
            show_ylabel=(idx == 0),
        )
        axes[idx].set_ylim(y_min - y_pad, y_max + y_pad)

    # Primary control requested: horizontal spacing between the four subplots.
    fig.subplots_adjust(wspace=float(subplot_wspace))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PATIENT_FIG_DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


# ============================================================
# Main
# ============================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Recompute if needed:
    # df = collect_all_gene_alignment_metrics(
    #     patient_ids=PATIENT_IDS,
    #     pair_list=PAIR_LIST,
    #     istar_root=ISTAR_ROOT,
    #     out_csv=ALL_GENE_CSV,
    # )

    df = pd.read_csv(ALL_GENE_CSV)

    plot_pairwise_mean_pearson_trend_4patients(
        df,
        before_col="pearson_before",
        after_col="pearson_after",
        pair_keys=("patient_id", "fixed_id", "moving_id"),
        patient_ids=PATIENT_IDS,
        sort_by="delta",
        annotate_delta=True,
        delta_dx=PATIENT_DELTA_DX,
        label_min_gap=PATIENT_LABEL_MIN_GAP,
        figsize=PATIENT_FIGSIZE,
        subplot_wspace=PATIENT_SUBPLOT_WSPACE,
        save_path=FIGURE_PATH,
    )


if __name__ == "__main__":
    main()
