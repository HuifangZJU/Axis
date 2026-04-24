#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import pyreadr



import numpy as np
import pandas as pd

# ---- desired markers ----
protein_markers = ["CD4","CD8","CD20","CD68","Ki67","a.SMA","IDO1",'CD44','PD.1']
# gene_markers = ["NAPSA","DUOX1","IGKC","IGHM","SPINK1","MUC5AC","SLC2A1","IFITM1","NKX2-1",
#                 "HNF4A","IFI6","RHOV","IDO1","ACTA2","MMP7","CXCL14"]


def canon(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(".", "").replace("-", "").replace("_", "").replace(" ", "")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _iqr(x: np.ndarray) -> float:
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)

def rank_markers_by_variation(
    df: pd.DataFrame,
    fdr_thresh: float = 0.05,
    min_pairs: int = 50,
    use_abs: bool = False,
    weight_by_fdr: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: marker, score, n_pairs.
    Score = IQR(logFC or |logFC|) * mean(-log10(FDR)) (optional weight).
    Falls back to all pairs for markers with too few significant pairs.
    """
    d = df.copy()
    d["logFC"] = pd.to_numeric(d["logFC"], errors="coerce")
    d["adj.P.Val"] = pd.to_numeric(d["adj.P.Val"], errors="coerce")
    d = d.dropna(subset=["marker", "logFC", "adj.P.Val"])

    d_sig = d[d["adj.P.Val"] <= fdr_thresh].copy()

    def score_block(g):
        vals = g["logFC"].to_numpy()
        if use_abs:
            vals = np.abs(vals)
        var = _iqr(vals)
        if weight_by_fdr:
            w = (-np.log10(g["adj.P.Val"].to_numpy() + 1e-300)).mean()
            return var * float(w)
        return var

    # compute score from significant pairs when enough exist; else fall back to all pairs
    out_rows = []
    for marker, g_all in d.groupby("marker"):
        g_sig = d_sig[d_sig["marker"] == marker]
        if len(g_sig) >= min_pairs:
            g_use = g_sig
        else:
            g_use = g_all  # fallback
        out_rows.append({
            "marker": marker,
            "score": score_block(g_use),
            "n_pairs": int(len(g_use)),
            "n_sig": int(len(g_sig)),
        })

    out = pd.DataFrame(out_rows).sort_values("score", ascending=False).reset_index(drop=True)
    return out

def build_gene_order_by_best_marker(df: pd.DataFrame, marker_order: list, genes: list) -> list:
    """
    For each gene, find its best marker (min FDR), then order genes by the marker_order,
    and within each marker by FDR ascending then |logFC| descending.
    """
    d = df[df["gene"].isin(genes)].copy()
    d["logFC"] = pd.to_numeric(d["logFC"], errors="coerce")
    d["adj.P.Val"] = pd.to_numeric(d["adj.P.Val"], errors="coerce")
    d = d.dropna(subset=["gene", "marker", "adj.P.Val", "logFC"])
    d["abs_logFC"] = d["logFC"].abs()

    # best marker per gene
    idx = d.groupby("gene")["adj.P.Val"].idxmin()
    best = d.loc[idx, ["gene", "marker", "adj.P.Val", "abs_logFC"]].copy()

    rank_map = {m: i for i, m in enumerate(marker_order)}
    best["marker_rank"] = best["marker"].map(rank_map).fillna(1e9)

    best = best.sort_values(["marker_rank", "adj.P.Val", "abs_logFC"], ascending=[True, True, False])
    return best["gene"].tolist()

def rank_genes_by_mean_aveexpr(df: pd.DataFrame, genes_subset: list = None) -> list:
    """
    Rank genes by mean(AveExpr) across all markers (descending).
    If genes_subset is provided, only rank those genes.
    """
    d = df.copy()
    d["AveExpr"] = pd.to_numeric(d["AveExpr"], errors="coerce")
    d = d.dropna(subset=["gene", "AveExpr"])

    if genes_subset is not None:
        d = d[d["gene"].isin(genes_subset)]

    gene_mean = d.groupby("gene")["AveExpr"].mean().sort_values(ascending=False)
    return gene_mean.index.tolist()




# -------------------------
# IO
# -------------------------
def read_rds_df(path: str) -> pd.DataFrame:
    """Read an RDS data.frame into pandas. Try pyreadr, then pyreadstat."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    res = pyreadr.read_r(path)
    if len(res) == 0:
        raise ValueError(f"No objects in RDS: {path}")
    obj = next(iter(res.values()))
    if isinstance(obj, pd.DataFrame):
        return obj



def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))


# -------------------------
# Selection + ordering
# -------------------------
def validate_columns(df: pd.DataFrame):
    required = ["logFC", "AveExpr", "adj.P.Val", "gene", "marker"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")


def to_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ["logFC", "AveExpr", "adj.P.Val"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["gene"] = d["gene"].astype(str)
    d["marker"] = d["marker"].astype(str)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["logFC", "AveExpr", "adj.P.Val", "gene", "marker"])
    return d


def marker_order_from_df(df: pd.DataFrame):
    """Preserve first-appearance order for markers."""
    return list(pd.unique(df["marker"]))


def pick_topk_genes_per_marker(df: pd.DataFrame, topk: int = 5) -> pd.DataFrame:
    """
    Pick topk genes per marker by adj.P.Val asc, tie-break by |logFC| desc.
    Returns rows only for those top genes (still one row per gene-marker pair in that selection).
    """
    d = df.copy()
    d["abs_logFC"] = d["logFC"].abs()
    d = d.sort_values(["marker", "adj.P.Val", "abs_logFC"], ascending=[True, True, False])
    top = d.groupby("marker", group_keys=False).head(topk).copy()
    return top


def build_gene_order(df_all: pd.DataFrame, genes_selected: list, marker_order: list):
    """
    Order genes by the marker where they are most significant (min adj.P.Val),
    grouped by marker_order (so you get “blocks” like the example figure).
    """
    d = df_all[df_all["gene"].isin(genes_selected)].copy()
    # for each gene, find best marker (min adj.P.Val)
    idx = d.groupby("gene")["adj.P.Val"].idxmin()
    best = d.loc[idx, ["gene", "marker", "adj.P.Val"]].copy()
    best["marker_rank"] = best["marker"].map({m: i for i, m in enumerate(marker_order)}).fillna(1e9)
    best = best.sort_values(["marker_rank", "adj.P.Val"], ascending=[True, True])
    return best["gene"].tolist()


# -------------------------
# Scaling for bubble plot
# -------------------------
def map_sizes(values, smin=30, smax=200):
    """Map AveExpr -> dot sizes using robust percentiles (denser bubbles)."""
    v = np.asarray(values, dtype=float)
    lo = np.nanpercentile(v, 5)
    hi = np.nanpercentile(v, 95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(v)
        hi = np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(v, (smin + smax) / 2.0)

    vv = np.clip(v, lo, hi)
    return smin + (vv - lo) * (smax - smin) / (hi - lo)


# -------------------------
# Plot
# -------------------------
def dotplot_marker_gene_matrix(
    df_plot: pd.DataFrame,
    marker_order: list,
    gene_order: list,
    out_png: str,
    out_pdf: str,
    cmap="RdBu_r",           # higher-contrast diverging map
    vmax_percentile=95,      # lower -> stronger contrast
):
    # categorical positions
    x_map = {m: i for i, m in enumerate(marker_order)}
    y_map = {g: i for i, g in enumerate(gene_order[::-1])}  # top at top

    d = df_plot.copy()
    d = d[d["marker"].isin(marker_order) & d["gene"].isin(gene_order)]
    d["x"] = d["marker"].map(x_map)
    d["y"] = d["gene"].map(y_map)
    d = d.dropna(subset=["x", "y", "logFC", "AveExpr"]).copy()

    x = d["x"].to_numpy(dtype=int)
    y = d["y"].to_numpy(dtype=int)
    logfc = d["logFC"].to_numpy(dtype=float)
    ave = d["AveExpr"].to_numpy(dtype=float)

    # stronger color contrast: clip to p95(|logFC|) and center at 0
    vmax = np.nanpercentile(np.abs(logfc), vmax_percentile)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(np.nanmax(np.abs(logfc))) if np.isfinite(np.nanmax(np.abs(logfc))) else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # denser bubbles
    sizes = map_sizes(ave, smin=30, smax=250)

    # compress figure size to increase density (tighter spacing)
    W = max(5, 0.22 * len(marker_order) + 3)
    H = max(6,  0.13 * len(gene_order) + 3)

    fig, ax = plt.subplots(figsize=(W, H), dpi=250)

    sc = ax.scatter(
        x, y,
        c=np.clip(logfc, -vmax, vmax),
        s=sizes,
        cmap=cmap,
        norm=norm,
        linewidths=0.0,
        alpha=1.0
    )

    ax.set_xticks(np.arange(len(marker_order)))
    ax.set_xticklabels(marker_order, rotation=45, ha="right", fontsize=9)

    ax.set_yticks(np.arange(len(gene_order)))
    ax.set_yticklabels(gene_order[::-1], fontsize=9)

    ax.set_xlim(-0.5, len(marker_order) - 0.5)
    ax.set_ylim(-0.5, len(gene_order) - 0.5)

    ax.set_axisbelow(True)
    ax.grid(which="major", axis="both", linestyle=":", alpha=0.18)

    ax.set_xlabel("Protein markers (marker)")
    ax.set_ylabel("Genes (union of top-5 per marker)")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("logFC (slope)")

    # size legend
    ref = np.nanpercentile(ave, [20, 50, 80])
    ref_sizes = map_sizes(ref, smin=30, smax=250)
    handles = [ax.scatter([], [], s=ref_sizes[i], c="gray", alpha=0.6, label=f"AveExpr≈{ref[i]:.2g}")
               for i in range(len(ref))]
    ax.legend(handles=handles, title="Dot size", loc="upper left",
              bbox_to_anchor=(1.02, 1.0), frameon=False)

    plt.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_folder", required=True)
    ap.add_argument("--in_rds", default="xenium_DE.rds")
    ap.add_argument("--topk", type=int, default=5, help="Top-k genes per protein marker")
    ap.add_argument("--out_dir", default="de_dotplot_matrix")
    args = ap.parse_args()

    in_path = os.path.join(args.root_folder, args.in_rds)
    out_dir = os.path.join(args.root_folder, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("[LOAD]", in_path)
    df = read_rds_df(in_path)
    validate_columns(df)
    df = to_numeric_cols(df)

    prot_set = set(canon(x) for x in protein_markers)
    df2 = df.copy()
    df2["marker_c"] = df2["marker"].map(canon)
    df2["gene_c"] = df2["gene"].map(canon)

    # Filter df to your selected gene/protein markers
    df2 = df2[df2["marker_c"].isin(prot_set)].copy()
    df = df2

    # Marker order (all proteins)
    marker_scores = rank_markers_by_variation(
        df,
        fdr_thresh=0.001,
        min_pairs=50,
        use_abs=True,  # True if you only care about magnitude
        weight_by_fdr=True
    )
    markers = marker_scores["marker"].tolist()  # reordered markers
    # Select union genes: topK per marker
    top = pick_topk_genes_per_marker(df, topk=args.topk)
    genes_selected = sorted(pd.unique(top["gene"]))
    genes_order = rank_genes_by_mean_aveexpr(df, genes_subset=genes_selected)
    # genes_order = build_gene_order_by_best_marker(df, markers, genes_selected)

    # Plot dataset: all marker–gene pairs for selected genes
    df_plot = df[df["gene"].isin(genes_order) & df["marker"].isin(markers)].copy()

    # Save selection table (for reproducibility)
    sel_csv = os.path.join(out_dir, f"selected_genes_union_top{args.topk}_per_marker.csv")
    top.sort_values(["marker", "adj.P.Val", "logFC"], ascending=[True, True, False]).to_csv(sel_csv, index=False)
    print("[SAVED]", sel_csv)

    out_png = os.path.join(out_dir, f"dotplot_union_top{args.topk}_genes_ALL_markers.png")
    out_pdf = os.path.join(out_dir, f"dotplot_union_top{args.topk}_genes_ALL_markers.pdf")

    dotplot_marker_gene_matrix(
        df_plot=df_plot,
        marker_order=markers,
        gene_order=genes_order,
        out_png=out_png,
        out_pdf=out_pdf,
        cmap="coolwarm",
    )
    print("[SAVED]", out_png)
    print("[SAVED]", out_pdf)


if __name__ == "__main__":
    main()
