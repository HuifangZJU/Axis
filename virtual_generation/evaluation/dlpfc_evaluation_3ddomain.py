import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


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

# Which real slice to use as the reference geometry for scaling
REFERENCE_REAL_Z = 10

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
# Fair comparison: use the common subset available for both methods.
# If both methods are available at every 10 um, replace with:
# virtual_depth = np.arange(20, 310, 10)
virtual_depth = np.arange(40, 310, 30)
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
# Domain key
# ------------------------------------------------------------
DOMAIN_KEY = "layer_guess_reordered"
# If needed, change to:
# DOMAIN_KEY = "axis_joint_domain"

# ------------------------------------------------------------
# Physical scaling
# ------------------------------------------------------------
# Current XY coordinates were rescaled for neural network use.
# Re-scale them using the real Visium nearest-neighbor spacing (~100 um).
VISIUM_NEAREST_SPOT_DISTANCE_UM = 100.0

# ------------------------------------------------------------
# Neighborhood continuity scoring
# ------------------------------------------------------------
# Main method comparison should score only the virtual slices, not the shared real anchors.
USE_REAL_ANCHORS_IN_SCORE = False
Z_ORDER_SCORE = Z_ORDER if USE_REAL_ANCHORS_IN_SCORE else virtual_depth.tolist()

# Cross-slice only: neighbors from different slices only.
INCLUDE_SAME_SLICE_NEIGHBORS = False

# Only compare to nearby slices in z.
# By default, use immediately adjacent scored slices.
MAX_Z_DIFF_FACTOR = 1.05  # multiplied by median z step

# Real physical 3D radius for neighboring search.
# We set it from physical XY spacing and z step:
# radius_um = RADIUS_FACTOR * sqrt((100 um)^2 + (median_dz)^2)
RADIUS_FACTOR = 1.05

# Ignore tiny domains in per-domain summaries
MIN_POINTS_PER_DOMAIN = 80

# Invalid labels to drop
INVALID_DOMAIN_LABELS = {"nan", "NaN", "None", "", "unknown"}


# ============================================================
# BASIC HELPERS
# ============================================================
def load_slice_dict(path_dict: Dict[int, str]) -> Dict[int, sc.AnnData]:
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
        out[z] = ad
    return out


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


def infer_network_nn_distance(
    adata_ref: sc.AnnData,
    coord_key: str = COORD_KEY,
    k: int = 2,
) -> float:
    """
    Infer current median nearest-neighbor spacing in network-rescaled XY coordinates.
    """
    coords = get_coords(adata_ref, coord_key)
    if coords.shape[0] < 3:
        raise ValueError("Too few spots to infer nearest-neighbor spacing.")

    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k)
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if len(nn) == 0:
        raise ValueError("Failed to infer nearest-neighbor distance.")
    return float(np.median(nn))


def collect_all_domain_labels(
    real_ads: Dict[int, sc.AnnData],
    spatialz_ads: Dict[int, sc.AnnData],
    axis_ads: Dict[int, sc.AnnData],
    domain_key: str = DOMAIN_KEY,
) -> List[str]:
    vals = []
    for ad in list(real_ads.values()) + list(spatialz_ads.values()) + list(axis_ads.values()):
        vals.extend(get_labels(ad, domain_key))

    vals = [str(x) for x in vals if str(x) not in INVALID_DOMAIN_LABELS]
    uniq = pd.unique(pd.Series(vals)).tolist()

    def sort_key(x):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)

    return sorted(uniq, key=sort_key)


def compute_radius_and_z_threshold(
    z_order_score: List[int],
    visium_nn_um: float = VISIUM_NEAREST_SPOT_DISTANCE_UM,
    radius_factor: float = RADIUS_FACTOR,
    max_z_diff_factor: float = MAX_Z_DIFF_FACTOR,
) -> Tuple[float, float]:
    z_sorted = np.sort(np.asarray(z_order_score, dtype=np.float32))
    if len(z_sorted) >= 2:
        dz_med = float(np.median(np.diff(z_sorted)))
    else:
        dz_med = 0.0

    radius_um = float(radius_factor * np.sqrt(visium_nn_um ** 2 + dz_med ** 2))
    max_z_diff_um = float(max_z_diff_factor * dz_med) if dz_med > 0 else np.inf
    return radius_um, max_z_diff_um


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


def get_domain_tab10_palette(domain_order):
    """
    Use the exact first 5 colors of matplotlib tab10, matching your 3D visualization.
    """
    cmap = plt.get_cmap("tab10")
    base_colors = [cmap(i) for i in range(5)]
    return {dom: base_colors[i % 5] for i, dom in enumerate(domain_order)}


def plot_domain_continuity_combined(
    per_domain_df: pd.DataFrame,
    overall_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
):
    """
    One combined figure:
      - boxplot over per-domain continuity scores
      - paired lines across methods for each domain
      - domain-colored points using first 5 colors of tab10
      - optional black diamond for the weighted overall score

    Required columns in per_domain_df:
      - domain
      - method
      - continuity_score

    Optional columns in overall_df:
      - method
      - overall_continuity_score
    """
    sns.set_style("whitegrid")

    method_order = ["SpatialZ", "AXIS"]
    x_map = {m: i for i, m in enumerate(method_order)}

    # fixed method fill colors
    method_palette = {
        "SpatialZ": "#AEBBD0",
        "AXIS": "#D98A7D",
    }

    # domain colors from first 5 colors of tab10
    domain_order = sorted(pd.unique(per_domain_df["domain"].astype(str)))
    domain_palette = get_domain_tab10_palette(domain_order)


    fig, ax = plt.subplots(figsize=(4.8, 6.))

    # boxplot
    sns.boxplot(
        data=per_domain_df,
        x="method",
        y="continuity_score",
        order=method_order,
        palette=method_palette,
        showfliers=False,
        width=0.52,
        linewidth=1.1,
        ax=ax,
        zorder=1,
    )

    # paired lines by domain
    wide = per_domain_df.pivot(index="domain", columns="method", values="continuity_score")
    for dom in wide.index:
        row = wide.loc[dom]
        if pd.notnull(row.get("SpatialZ", np.nan)) and pd.notnull(row.get("AXIS", np.nan)):
            ax.plot(
                [x_map["SpatialZ"], x_map["AXIS"]],
                [row["SpatialZ"], row["AXIS"]],
                color="0.83",
                lw=1.0,
                alpha=0.95,
                zorder=2,
            )

    # domain-colored points with slight jitter
    rng = np.random.default_rng(0)
    for _, r in per_domain_df.iterrows():
        x = x_map[r["method"]] + rng.uniform(-0.08, 0.08)
        ax.scatter(
            x,
            r["continuity_score"],
            s=95,
            color=domain_palette[str(r["domain"])],
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )

    # # optional weighted overall score as black diamond
    # if overall_df is not None and "overall_continuity_score" in overall_df.columns:
    #     for _, r in overall_df.iterrows():
    #         if r["method"] not in x_map:
    #             continue
    #         ax.scatter(
    #             x_map[r["method"]],
    #             r["overall_continuity_score"],
    #             s=120,
    #             marker="D",
    #             color="black",
    #             edgecolor="white",
    #             linewidth=0.8,
    #             zorder=4,
    #         )

    ax.set_xlabel("")
    ax.set_ylabel(" ")
    ax.set_title("3D domain continuity score", fontsize=16,weight="bold")
    ax.grid(axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=12)

    # legend for domains + overall diamond
    domain_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=domain_palette[d],
            markeredgecolor="black",
            label=str(d),
        )
        for d in domain_order
    ]


    ax.legend(
        handles=domain_handles,
        title="Domain",
        frameon=False,
        fontsize=10,
        title_fontsize=13,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()

# ============================================================
# BUILD PHYSICAL SLICE DATA
# ============================================================
def build_physical_slice_data(
    real_ads: Dict[int, sc.AnnData],
    method_ads: Dict[int, sc.AnnData],
    z_order_score: List[int],
    domain_key: str,
    coord_key: str,
    xy_scale_to_um: float,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Returns
    -------
    slice_data[z] = {
        "xy_um": (N,2),
        "labels": (N,),
        "z_um": scalar
    }
    """
    slice_data = {}

    for z in z_order_score:
        ad, is_real = get_display_slice(real_ads, method_ads, z)
        if ad is None:
            print(f"[WARN] missing scored slice at z={z}")
            continue

        coords_xy = get_coords(ad, coord_key=coord_key).astype(np.float32)
        labels = get_labels(ad, domain_key=domain_key).astype(str)

        valid = np.array([lab not in INVALID_DOMAIN_LABELS for lab in labels], dtype=bool)
        coords_xy = coords_xy[valid]
        labels = labels[valid]

        xy_um = coords_xy * float(xy_scale_to_um)

        slice_data[int(z)] = {
            "xy_um": xy_um.astype(np.float32),
            "labels": labels,
            "z_um": float(z),
            "is_real": bool(is_real),
        }

    return slice_data


# ============================================================
# NEIGHBORHOOD CONTINUITY SCORE
# ============================================================
def compute_cross_slice_neighbor_purity(
    slice_data: Dict[int, Dict[str, np.ndarray]],
    z_order_score: List[int],
    radius_um: float,
    max_z_diff_um: float,
    include_same_slice_neighbors: bool = False,
) -> pd.DataFrame:
    """
    For each spot i, gather neighbors from nearby slices within physical radius.
    Compute:
      purity_i = same_domain_neighbors / all_neighbors

    Returns per-spot dataframe.
    """
    z_available = [z for z in z_order_score if z in slice_data]
    if len(z_available) == 0:
        raise ValueError("No scored slices available.")

    # pre-build KDTree for each slice
    trees = {}
    for z in z_available:
        xy = slice_data[z]["xy_um"]
        trees[z] = cKDTree(xy)

    rows = []

    for z_src in z_available:
        xy_src = slice_data[z_src]["xy_um"]
        lab_src = slice_data[z_src]["labels"]
        n_src = xy_src.shape[0]

        total_neighbors = np.zeros(n_src, dtype=np.int32)
        same_neighbors = np.zeros(n_src, dtype=np.int32)

        for z_tgt in z_available:
            dz = abs(float(z_tgt) - float(z_src))

            if (not include_same_slice_neighbors) and (z_tgt == z_src):
                continue

            if dz > max_z_diff_um + 1e-8:
                continue

            if dz >= radius_um:
                continue

            xy_tgt = slice_data[z_tgt]["xy_um"]
            lab_tgt = slice_data[z_tgt]["labels"]

            # XY search radius after accounting for z separation
            r_xy = np.sqrt(max(radius_um ** 2 - dz ** 2, 0.0))
            if r_xy <= 0:
                continue

            nbr_lists = trees[z_tgt].query_ball_point(xy_src, r=r_xy)

            for i, idxs in enumerate(nbr_lists):
                if len(idxs) == 0:
                    continue

                total_neighbors[i] += len(idxs)
                same_neighbors[i] += int(np.sum(lab_tgt[np.asarray(idxs, dtype=int)] == lab_src[i]))

        purity = np.full(n_src, np.nan, dtype=np.float32)
        valid = total_neighbors > 0
        purity[valid] = same_neighbors[valid] / total_neighbors[valid]

        for i in range(n_src):
            rows.append({
                "slice_z": float(z_src),
                "domain": str(lab_src[i]),
                "n_neighbors": int(total_neighbors[i]),
                "n_same_domain_neighbors": int(same_neighbors[i]),
                "neighbor_purity": float(purity[i]) if np.isfinite(purity[i]) else np.nan,
                "has_neighbors": bool(valid[i]),
            })

    return pd.DataFrame(rows)


def summarize_neighbor_continuity(
    per_spot_df: pd.DataFrame,
    min_points_per_domain: int = MIN_POINTS_PER_DOMAIN,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Summarize per-domain and overall neighborhood continuity.
    """
    df_valid = per_spot_df[per_spot_df["has_neighbors"]].copy()

    overall_score = float(df_valid["neighbor_purity"].mean()) if len(df_valid) > 0 else np.nan
    overall_coverage = float(per_spot_df["has_neighbors"].mean()) if len(per_spot_df) > 0 else np.nan

    rows = []
    for dom, sub in per_spot_df.groupby("domain"):
        n_total = len(sub)
        n_valid = int(sub["has_neighbors"].sum())
        if n_total < min_points_per_domain:
            continue

        sub_valid = sub[sub["has_neighbors"]]
        dom_score = float(sub_valid["neighbor_purity"].mean()) if len(sub_valid) > 0 else np.nan
        dom_cov = n_valid / max(n_total, 1)

        rows.append({
            "domain": dom,
            "n_points": n_total,
            "n_points_with_neighbors": n_valid,
            "neighbor_coverage": dom_cov,
            "continuity_score": dom_score,
        })

    per_domain_df = pd.DataFrame(rows)

    overall = {
        "overall_continuity_score": overall_score,
        "overall_neighbor_coverage": overall_coverage,
        "n_total_points": int(len(per_spot_df)),
        "n_points_with_neighbors": int(per_spot_df["has_neighbors"].sum()),
    }
    return per_domain_df, overall


# ============================================================
# PLOTTING
# ============================================================
def method_palette():
    return {
        "SpatialZ": "#7C8DA6",
        "AXIS": "#C4473A",
    }


def plot_overall_continuity_bar(
    overall_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    palette = method_palette()

    methods = overall_df["method"].tolist()
    vals = overall_df["overall_continuity_score"].tolist()
    covs = overall_df["overall_neighbor_coverage"].tolist()

    fig, ax = plt.subplots(figsize=(4.0, 4.3))
    bars = ax.bar(
        methods,
        vals,
        color=[palette.get(m, "#999999") for m in methods],
        edgecolor="black",
        linewidth=1.0,
        width=0.55,
    )

    for b, v, c in zip(bars, vals, covs):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.01,
            f"{v:.3f}\n(cov={c:.2f})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Cross-slice neighborhood continuity", fontsize=12)
    ax.set_title("Higher is better", fontsize=11, color="0.35")
    ax.grid(axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, min(1.04, max(vals) + 0.12))

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()


def plot_per_domain_continuity(
    per_domain_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    palette = method_palette()

    wide = per_domain_df.pivot(index="domain", columns="method", values="continuity_score")
    domains = wide.index.tolist()

    x = np.arange(len(domains), dtype=float)
    offset = 0.14

    fig, ax = plt.subplots(figsize=(max(4.8, 1.0 * len(domains) + 2.0), 4.2))

    # paired lines by domain
    for i, dom in enumerate(domains):
        row = wide.loc[dom]
        if pd.notnull(row.get("SpatialZ", np.nan)) and pd.notnull(row.get("AXIS", np.nan)):
            ax.plot(
                [x[i] - offset, x[i] + offset],
                [row["SpatialZ"], row["AXIS"]],
                color="0.82",
                lw=1.1,
                alpha=0.95,
                zorder=1,
            )

    # points
    for i, dom in enumerate(domains):
        row = wide.loc[dom]
        if pd.notnull(row.get("SpatialZ", np.nan)):
            ax.scatter(
                x[i] - offset,
                row["SpatialZ"],
                s=95,
                color=palette["SpatialZ"],
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
            )
        if pd.notnull(row.get("AXIS", np.nan)):
            ax.scatter(
                x[i] + offset,
                row["AXIS"],
                s=95,
                color=palette["AXIS"],
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=11)
    ax.set_xlabel("Domain", fontsize=12)
    ax.set_ylabel("Per-domain continuity score", fontsize=12)
    ax.set_title("Higher is better", fontsize=11, color="0.35")
    ax.grid(axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                   markerfacecolor=palette["SpatialZ"], markeredgecolor="black", label="SpatialZ"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                   markerfacecolor=palette["AXIS"], markeredgecolor="black", label="AXIS"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=11, loc="best")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # # --------------------------------------------------------
    # # 1) Load slices
    # # --------------------------------------------------------
    # real_ads = load_slice_dict(REAL_SLICE_PATHS)
    # spatialz_ads = load_slice_dict(SPATIALZ_SLICE_PATHS)
    # axis_ads = load_slice_dict(AXIS_SLICE_PATHS)
    #
    # if REFERENCE_REAL_Z not in real_ads:
    #     raise ValueError(
    #         f"REFERENCE_REAL_Z={REFERENCE_REAL_Z} not found in REAL_SLICE_PATHS"
    #     )
    #
    # # --------------------------------------------------------
    # # 2) Infer XY scaling back to physical microns
    # # --------------------------------------------------------
    # nn_current = infer_network_nn_distance(real_ads[REFERENCE_REAL_Z], coord_key=COORD_KEY)
    # xy_scale_to_um = VISIUM_NEAREST_SPOT_DISTANCE_UM / nn_current
    #
    # print(f"[INFO] current network-space NN distance: {nn_current:.4f}")
    # print(f"[INFO] XY scale factor to physical um: {xy_scale_to_um:.6f}")
    # print(f"[INFO] target physical NN distance: {VISIUM_NEAREST_SPOT_DISTANCE_UM:.1f} um")
    #
    # # --------------------------------------------------------
    # # 3) Physical neighborhood thresholds
    # # --------------------------------------------------------
    # radius_um, max_z_diff_um = compute_radius_and_z_threshold(
    #     z_order_score=Z_ORDER_SCORE,
    #     visium_nn_um=VISIUM_NEAREST_SPOT_DISTANCE_UM,
    #     radius_factor=RADIUS_FACTOR,
    #     max_z_diff_factor=MAX_Z_DIFF_FACTOR,
    # )
    #
    # print(f"[INFO] scoring slices: {Z_ORDER_SCORE}")
    # print(f"[INFO] 3D neighborhood radius: {radius_um:.2f} um")
    # print(f"[INFO] max z difference: {max_z_diff_um:.2f} um")
    # print(f"[INFO] include same-slice neighbors: {INCLUDE_SAME_SLICE_NEIGHBORS}")
    #
    # # --------------------------------------------------------
    # # 4) Build physical slice data
    # # --------------------------------------------------------
    # spatialz_slice_data = build_physical_slice_data(
    #     real_ads=real_ads,
    #     method_ads=spatialz_ads,
    #     z_order_score=Z_ORDER_SCORE,
    #     domain_key=DOMAIN_KEY,
    #     coord_key=COORD_KEY,
    #     xy_scale_to_um=xy_scale_to_um,
    # )
    #
    # axis_slice_data = build_physical_slice_data(
    #     real_ads=real_ads,
    #     method_ads=axis_ads,
    #     z_order_score=Z_ORDER_SCORE,
    #     domain_key=DOMAIN_KEY,
    #     coord_key=COORD_KEY,
    #     xy_scale_to_um=xy_scale_to_um,
    # )
    #
    # # --------------------------------------------------------
    # # 5) Per-spot neighborhood continuity
    # # --------------------------------------------------------
    # spatialz_per_spot = compute_cross_slice_neighbor_purity(
    #     slice_data=spatialz_slice_data,
    #     z_order_score=Z_ORDER_SCORE,
    #     radius_um=radius_um,
    #     max_z_diff_um=max_z_diff_um,
    #     include_same_slice_neighbors=INCLUDE_SAME_SLICE_NEIGHBORS,
    # )
    # spatialz_per_spot["method"] = "SpatialZ"
    #
    # axis_per_spot = compute_cross_slice_neighbor_purity(
    #     slice_data=axis_slice_data,
    #     z_order_score=Z_ORDER_SCORE,
    #     radius_um=radius_um,
    #     max_z_diff_um=max_z_diff_um,
    #     include_same_slice_neighbors=INCLUDE_SAME_SLICE_NEIGHBORS,
    # )
    # axis_per_spot["method"] = "AXIS"
    #
    # # --------------------------------------------------------
    # # 6) Summaries
    # # --------------------------------------------------------
    # spatialz_per_domain, spatialz_overall = summarize_neighbor_continuity(
    #     spatialz_per_spot,
    #     min_points_per_domain=MIN_POINTS_PER_DOMAIN,
    # )
    # spatialz_per_domain["method"] = "SpatialZ"
    #
    # axis_per_domain, axis_overall = summarize_neighbor_continuity(
    #     axis_per_spot,
    #     min_points_per_domain=MIN_POINTS_PER_DOMAIN,
    # )
    # axis_per_domain["method"] = "AXIS"
    #
    # overall_df = pd.DataFrame([
    #     {"method": "SpatialZ", **spatialz_overall},
    #     {"method": "AXIS", **axis_overall},
    # ])
    #
    # per_domain_df = pd.concat([spatialz_per_domain, axis_per_domain], axis=0, ignore_index=True)
    # per_spot_df = pd.concat([spatialz_per_spot, axis_per_spot], axis=0, ignore_index=True)
    #
    # # --------------------------------------------------------
    # # 7) Save tables
    # # --------------------------------------------------------
    # overall_csv = os.path.join(OUTDIR, "overall_cross_slice_domain_continuity.csv")
    # per_domain_csv = os.path.join(OUTDIR, "per_domain_cross_slice_domain_continuity.csv")
    # per_spot_csv = os.path.join(OUTDIR, "per_spot_cross_slice_domain_continuity.csv")
    #
    # overall_df["xy_scale_to_um"] = xy_scale_to_um
    # overall_df["radius_um"] = radius_um
    # overall_df["max_z_diff_um"] = max_z_diff_um
    # overall_df["z_order_score"] = ",".join(map(str, Z_ORDER_SCORE))
    #
    # overall_df.to_csv(overall_csv, index=False)
    # per_domain_df.to_csv(per_domain_csv, index=False)
    # per_spot_df.to_csv(per_spot_csv, index=False)
    #
    # print(f"[OK] saved: {overall_csv}")
    # print(f"[OK] saved: {per_domain_csv}")
    # print(f"[OK] saved: {per_spot_csv}")
    #
    # print("[RESULT] overall continuity:")
    # print(overall_df[["method", "overall_continuity_score", "overall_neighbor_coverage"]])
    #


    overall_df = pd.read_csv(os.path.join(OUTDIR, "overall_cross_slice_domain_continuity.csv"))
    per_domain_df = pd.read_csv(os.path.join(OUTDIR, "per_domain_cross_slice_domain_continuity.csv"))
    per_spot_df = pd.read_csv(os.path.join(OUTDIR, "per_spot_cross_slice_domain_continuity.csv"))

    # --------------------------------------------------------
    # 8) Plots
    # --------------------------------------------------------



    plot_domain_continuity_combined(
        per_domain_df=per_domain_df,
        overall_df=overall_df,
        save_path=os.path.join(OUTDIR, "combined_3d_domain_continuity.png"),
    )