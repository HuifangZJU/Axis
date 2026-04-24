#!/usr/bin/env python
"""
Hero sample visualization for iStar + AXIS.

Outputs:
  hero_overlays.png     – H&E with luminal + cancer domains overlaid (single vs joint)
  hero_heatmaps.png     – Luminal & cancer score heatmaps on H&E (single vs joint)
  hero_specificity.png  – Specificity barplots (single vs joint)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
Image.MAX_IMAGE_PIXELS = None


# ============================================================
# CONFIG – EDIT THESE FOR YOUR HERO SAMPLE
# ============================================================

# iStar result roots
SINGLE_ROOT = Path(
    "/media/huifang/data/registration/SCC/istar/single_input/group_10/axis_1/sample1/"
)
JOINT_ROOT = Path(
    "/media/huifang/data/registration/SCC/istar/joint/group_10/sample1/"
)

# Which cluster IDs correspond to luminal / cancer domains
# Single-slice cluster IDs (Sx) – you can adjust these based on your violin plots
SINGLE_LUMINAL_DOMAINS = [9]   # e.g. S9
SINGLE_CANCER_DOMAINS  = [7]   # e.g. S7

# AXIS joint cluster IDs (Jx)
JOINT_LUMINAL_DOMAINS = [9]    # J9
JOINT_CANCER_DOMAINS  = [8]    # J8

# Marker sets (keep only the two you care about here)
NORMAL_MARKERS = [
    "EPCAM",
    "KRT8",
    "KRT18",
    "KRT19",
    "KRT5",
    "KRT14",
    "MUC1",
    "KRT15",
    "KRT4",
]

CANCER_MARKERS = [
     "EPCAM",
        "KRT5",
        "KRT6A",
        "KRT6B",
        "KRT14",
        "KRT17",
        "S100A2",
        "COL17A1",
        "MKI67",    # proliferation
        "MMP1",
        "MMP10",
]

# Transparency for overlays
OVERLAY_ALPHA = 0.45
HEATMAP_ALPHA = 0.5


# ============================================================
# I/O UTILITIES
# ============================================================

def load_labels(run_root: Path):
    """Load domain labels (clusters-gene/labels.pickle)."""
    labels_path = run_root / "clusters-gene" / "labels.pickle"
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
    labels = np.asarray(labels, dtype=int)

    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels, got shape {labels.shape}")
    return labels


def load_he_image(run_root: Path, target_shape=None):
    """
    Load he.jpg and optionally resize to target_shape (H, W).
    Returns float array in [0,1], shape (H,W,3).
    """
    img_path = run_root / "he.jpg"
    img = Image.open(img_path).convert("RGB")
    if target_shape is not None:
        H, W = target_shape
        img = img.resize((W, H), resample=Image.BILINEAR)
    img = np.asarray(img, dtype=np.float32) / 255.0
    return img


def load_gene_map(run_root: Path, gene: str, shape):
    """
    Load cnts-super/<gene>.pickle and ensure shape matches labels.
    """
    path = run_root / "cnts-super" / f"{gene}.pickle"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found.")
    with open(path, "rb") as f:
        arr = pickle.load(f)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape == shape:
        return arr
    if arr.ndim == 3:
        arr = np.squeeze(arr)
    if arr.shape != shape:
        raise ValueError(f"Gene {gene} map shape {arr.shape} != labels shape {shape}")
    return arr


# ============================================================
# SCORE MAPS & SPECIFICITY
# ============================================================

def compute_score_map(run_root: Path, labels, genes, log1p=True):
    """
    Compute average score map for a marker set:
      score[pixel] = mean_g log1p(expr_g[pixel]).
    Returns (score_map, used_genes).
    """
    H, W = labels.shape
    shape = (H, W)
    maps = []
    used = []
    for g in genes:
        try:
            gm = load_gene_map(run_root, g, shape)
        except FileNotFoundError:
            continue
        if log1p:
            gm = np.log1p(gm)
        maps.append(gm)
        used.append(g)
    if not maps:
        raise RuntimeError(f"No marker genes found for {genes}")
    stack = np.stack(maps, axis=0)  # G × H × W
    score = stack.mean(axis=0)
    return score, used


def compute_specificity(score_map, labels, domain_ids):
    """
    Specificity = median(score in domain_ids) - median(score outside).
    """
    labels = np.asarray(labels)
    mask_dom = np.isin(labels, domain_ids)
    vals_dom = score_map[mask_dom]
    vals_other = score_map[~mask_dom]
    med_dom = float(np.median(vals_dom)) if vals_dom.size > 0 else np.nan
    med_other = float(np.median(vals_other)) if vals_other.size > 0 else np.nan
    return med_dom, med_other, med_dom - med_other


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def make_overlap_figure(
    single_root, joint_root,
    single_lum_ids, single_cancer_ids,
    joint_lum_ids, joint_cancer_ids,
    out_png="hero_overlays.png",
):
    """
    Panel A: H&E with luminal + cancer domains overlaid, single vs joint.
    """

    # ---- single ----
    labels_s = load_labels(single_root)
    img_s = load_he_image(single_root, target_shape=labels_s.shape)

    mask_s_lum = np.isin(labels_s, single_lum_ids)
    mask_s_can = np.isin(labels_s, single_cancer_ids)

    # ---- joint ----
    labels_j = load_labels(joint_root)
    img_j = load_he_image(joint_root, target_shape=labels_j.shape)

    mask_j_lum = np.isin(labels_j, joint_lum_ids)
    mask_j_can = np.isin(labels_j, joint_cancer_ids)

    # colors: luminal = blue, cancer = red
    blue = np.array([0.0, 0.3, 1.0])
    red  = np.array([1.0, 0.0, 0.0])

    def overlay(img, mask_lum, mask_can):
        ov = img.copy()
        # luminal first
        ov[mask_lum] = (1 - OVERLAY_ALPHA) * ov[mask_lum] + OVERLAY_ALPHA * blue
        # cancer overlay on top
        ov[mask_can] = (1 - OVERLAY_ALPHA) * ov[mask_can] + OVERLAY_ALPHA * red
        return ov

    ov_s = overlay(img_s, mask_s_lum, mask_s_can)
    ov_j = overlay(img_j, mask_j_lum, mask_j_can)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(ov_s)
    axes[0].set_title(
        f"Single slice\nNormal: S{single_lum_ids}, Cancer: S{single_cancer_ids}"
    )
    axes[0].axis("off")

    axes[1].imshow(ov_j)
    axes[1].set_title(
        f"AXIS joint\nNormal: J{joint_lum_ids}, Cancer: J{joint_cancer_ids}"
    )
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"Saved: {out_png}")
    plt.close(fig)


def make_heatmap_figure(
    single_root, joint_root,
    lum_genes, cancer_genes,
    out_png="hero_heatmaps.png",
):
    """
    Panel B/C: luminal & cancer score heatmaps on H&E, single vs joint.
    """

    # single
    labels_s = load_labels(single_root)
    img_s = load_he_image(single_root, target_shape=labels_s.shape)
    lum_s, used_lum_s = compute_score_map(single_root, labels_s, lum_genes)
    can_s, used_can_s = compute_score_map(single_root, labels_s, cancer_genes)

    # joint
    labels_j = load_labels(joint_root)
    img_j = load_he_image(joint_root, target_shape=labels_j.shape)
    lum_j, used_lum_j = compute_score_map(joint_root, labels_j, lum_genes)
    can_j, used_can_j = compute_score_map(joint_root, labels_j, cancer_genes)

    # normalization per cell type across both runs
    # def norm_two(a, b):
    #     all_vals = np.concatenate([a.ravel(), b.ravel()])
    #     vmin, vmax = np.percentile(all_vals, [1, 99])
    #     vmax = max(vmax, vmin + 1e-6)
    #     a_n = np.clip((a - vmin) / (vmax - vmin), 0, 1)
    #     b_n = np.clip((b - vmin) / (vmax - vmin), 0, 1)
    #     return a_n, b_n
    #
    # lum_s_n, lum_j_n = norm_two(lum_s, lum_j)
    # can_s_n, can_j_n = norm_two(can_s, can_j)
    lum_s_n, lum_j_n = lum_s, lum_j
    can_s_n, can_j_n = can_s, can_j

    cmap = plt.get_cmap("viridis")

    def overlay_heat(img, norm_map):
        cm = cmap(norm_map)[..., :3]
        return (1 - HEATMAP_ALPHA) * img + HEATMAP_ALPHA * cm

    ov_lum_s = overlay_heat(img_s, lum_s_n)
    ov_lum_j = overlay_heat(img_j, lum_j_n)
    ov_can_s = overlay_heat(img_s, can_s_n)
    ov_can_j = overlay_heat(img_j, can_j_n)

    # plot 2x2: luminal row, cancer row; single vs joint columns
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(ov_lum_s)
    axes[0, 0].set_title(
        f"Normal_epithelial – Single\nmarkers: {', '.join(used_lum_s)}"
    )
    axes[0, 0].axis("off")

    axes[0, 1].imshow(ov_lum_j)
    axes[0, 1].set_title(
        f"Normal_epithelial – AXIS joint\nmarkers: {', '.join(used_lum_j)}"
    )
    axes[0, 1].axis("off")

    axes[1, 0].imshow(ov_can_s)
    axes[1, 0].set_title(
        f"Cancer_epithelial – Single\nmarkers: {', '.join(used_can_s)}"
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(ov_can_j)
    axes[1, 1].set_title(
        f"Cancer_epithelial – AXIS joint\nmarkers: {', '.join(used_can_j)}"
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"Saved: {out_png}")
    plt.close(fig)


def make_specificity_figure(
    single_root, joint_root,
    lum_genes, cancer_genes,
    single_lum_ids, joint_lum_ids,
    single_cancer_ids, joint_cancer_ids,
    out_png="hero_specificity.png",
):
    """
    Panel D: barplots of specificity (single vs joint) for luminal & cancer.
    """

    # single scores
    labels_s = load_labels(single_root)
    lum_s, _ = compute_score_map(single_root, labels_s, lum_genes)
    can_s, _ = compute_score_map(single_root, labels_s, cancer_genes)

    # joint scores
    labels_j = load_labels(joint_root)
    lum_j, _ = compute_score_map(joint_root, labels_j, lum_genes)
    can_j, _ = compute_score_map(joint_root, labels_j, cancer_genes)

    lum_s_dom, lum_s_other, lum_s_spec = compute_specificity(
        lum_s, labels_s, single_lum_ids
    )
    lum_j_dom, lum_j_other, lum_j_spec = compute_specificity(
        lum_j, labels_j, joint_lum_ids
    )
    can_s_dom, can_s_other, can_s_spec = compute_specificity(
        can_s, labels_s, single_cancer_ids
    )
    can_j_dom, can_j_other, can_j_spec = compute_specificity(
        can_j, labels_j, joint_cancer_ids
    )

    # plotting
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Luminal
    axes[0].bar(
        [0, 1],
        [lum_s_spec, lum_j_spec],
        tick_label=["Single", "AXIS"],
    )
    axes[0].set_ylabel("Specificity\n(median in domain − outside)")
    axes[0].set_title("Normal_epithelial")
    axes[0].grid(axis="y", alpha=0.3)

    # Cancer
    axes[1].bar(
        [0, 1],
        [can_s_spec, can_j_spec],
        tick_label=["Single", "AXIS"],
    )
    axes[1].set_title("Cancer_epithelial")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"Saved: {out_png}")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    # Panel A: overlays
    make_overlap_figure(
        SINGLE_ROOT,
        JOINT_ROOT,
        SINGLE_LUMINAL_DOMAINS,
        SINGLE_CANCER_DOMAINS,
        JOINT_LUMINAL_DOMAINS,
        JOINT_CANCER_DOMAINS,
        out_png="hero_overlays.png",
    )

    # Panel B/C: heatmaps
    make_heatmap_figure(
        SINGLE_ROOT,
        JOINT_ROOT,
        NORMAL_MARKERS,
        CANCER_MARKERS,
        out_png="hero_heatmaps.png",
    )

    # Panel D: specificity barplots
    make_specificity_figure(
        SINGLE_ROOT,
        JOINT_ROOT,
        NORMAL_MARKERS,
        CANCER_MARKERS,
        SINGLE_LUMINAL_DOMAINS,
        JOINT_LUMINAL_DOMAINS,
        SINGLE_CANCER_DOMAINS,
        JOINT_CANCER_DOMAINS,
        out_png="hero_specificity.png",
    )


if __name__ == "__main__":
    main()
