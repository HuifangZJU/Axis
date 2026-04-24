import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image


# -------------------------------
# CONFIG
# -------------------------------

# Roots you gave
ROOT_JOINT = Path("/media/huifang/data/registration/SCC/istar/joint")
ROOT_SINGLE = Path("/media/huifang/data/registration/SCC/istar/single_input")

# Example marker gene list (replace with your own)
MARKER_GENES = [
    # "KRT5", "KRT14", "CD3D", "CD8A", ...
]

# Threshold for "marker is present" based on mean expression
MARKER_EXPR_THRESHOLD = 0.0  # e.g. 0 or a small positive value


# -------------------------------
# I/O helpers
# -------------------------------

def load_cnts(cnts_path: Path) -> pd.DataFrame:
    """
    Load cnts.tsv: first column is 'spot_id', remaining columns are genes.
    Returns DataFrame with index=spot_id, columns=genes.
    """
    df = pd.read_csv(cnts_path, sep="\t")
    if df.columns[0].lower() != "spot_id":
        df = df.rename(columns={df.columns[0]: "spot_id"})
    df = df.set_index("spot_id")
    return df


def load_locs(locs_path: Path) -> pd.DataFrame:
    """
    Load locs-raw.tsv: columns [spot_id, x, y].
    Returns DataFrame with index=spot_id, columns=['x','y'].
    """
    df = pd.read_csv(locs_path, sep="\t")
    if df.columns[0].lower() != "spot_id":
        df = df.rename(columns={df.columns[0]: "spot_id"})
    df = df.set_index("spot_id")
    # ensure numeric and integer-ish
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    return df


def load_labels_png(labels_png_path: Path) -> np.ndarray:
    """
    Load clusters-gene/labels.png as a 2D array of integer labels.
    Assumes labels are encoded as palette or grayscale.
    """
    img = Image.open(labels_png_path)
    arr = np.asarray(img)
    return arr


def assign_domains_to_spots(locs_df: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    """
    Given spot coordinates (locs_df with columns ['x','y']) and a label mask
    (2D array labels[h,w]), assign each spot to a domain based on label[y, x].
    """
    h, w = labels.shape
    # indices are y, x in image coordinates
    xs = np.clip(locs_df["x"].round().astype(int), 0, w - 1)
    ys = np.clip(locs_df["y"].round().astype(int), 0, h - 1)
    domain_labels = labels[ys.to_numpy(), xs.to_numpy()]
    return pd.Series(domain_labels, index=locs_df.index, name="domain")


# -------------------------------
# Marker gene richness
# -------------------------------

def compute_marker_richness_per_domain(
    cnts: pd.DataFrame,
    spot_domains: pd.Series,
    marker_genes,
    expr_threshold=0.0,
) -> pd.DataFrame:
    """
    For each domain, compute marker richness:
      - For each marker gene present in cnts, compute mean expression in that domain.
      - Richness = number of marker genes with mean expression > expr_threshold.

    Returns DataFrame with index=domain, columns=['richness', 'n_spots'].
    """
    # Restrict to genes we have
    marker_genes_present = [g for g in marker_genes if g in cnts.columns]
    if not marker_genes_present:
        raise ValueError("None of the marker genes are present in cnts.tsv columns.")

    domains = np.sort(spot_domains.unique())
    rows = []
    for d in domains:
        spot_ids = spot_domains[spot_domains == d].index
        if len(spot_ids) == 0:
            continue
        sub = cnts.loc[spot_ids, marker_genes_present]
        mean_expr = sub.mean(axis=0)
        richness = (mean_expr > expr_threshold).sum()
        rows.append(
            {
                "domain": d,
                "richness": int(richness),
                "n_spots": int(len(spot_ids)),
            }
        )
    df = pd.DataFrame(rows).set_index("domain")
    return df


# -------------------------------
# Visualization helpers
# -------------------------------

def overlay_domains_on_he(
    he_path: Path,
    labels_png_path: Path,
    title: str,
    alpha: float = 0.5,
):
    """
    Show H&E with domain labels overlaid.
    """
    he = Image.open(he_path)
    labels = load_labels_png(labels_png_path)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(he)
    ax.imshow(labels, cmap="tab20", alpha=alpha)
    ax.set_title(title)
    ax.axis("off")
    return fig, ax


def plot_richness_barplot(df_joint: pd.DataFrame, df_single: pd.DataFrame, title: str):
    """
    Barplot comparing domain-level marker richness between joint and single.
    """
    # Align domains
    df = pd.concat(
        {
            "joint": df_joint["richness"],
            "single": df_single["richness"],
        },
        axis=1,
    ).fillna(0)

    domains = df.index.to_numpy()
    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, df["joint"], width, label="joint")
    ax.bar(x + width / 2, df["single"], width, label="single")

    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_xlabel("Domain label")
    ax.set_ylabel("Marker richness (#markers above threshold)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


# -------------------------------
# Main analysis for one sample
# -------------------------------

def analyze_patient_sample(patient_id: int, sample_id: int):
    """
    Compare joint vs single-slice iStar outputs for one (patient, sample).

    Uses:
      joint:  ROOT_JOINT / f"group_{patient_id}" / f"sample{sample_id}"
      single: ROOT_SINGLE / f"group_{patient_id}" / f"axis_{sample_id}" / f"sample{sample_id}"
    """
    joint_dir = ROOT_JOINT / f"group_{patient_id}" / f"sample{sample_id}"
    single_dir = ROOT_SINGLE / f"group_{patient_id}" / f"axis_{sample_id}" / f"sample{sample_id}"

    print(f"\n=== Analyzing patient {patient_id}, sample {sample_id} ===")
    print(" joint dir :", joint_dir)
    print(" single dir:", single_dir)

    # --- paths ---
    joint_cnts = joint_dir / "cnts.tsv"
    joint_locs = joint_dir / "locs-raw.tsv"
    joint_he = joint_dir / "he-scaled.jpg"
    joint_labels_png = joint_dir / "clusters-gene" / "labels.png"

    single_cnts = single_dir / "cnts.tsv"
    single_locs = single_dir / "locs-raw.tsv"
    single_he = single_dir / "he-scaled.jpg"
    single_labels_png = single_dir / "clusters-gene" / "labels.png"
    single_lowres = f"/media/huifang/data/registration/SCC/huifang/network_input/{patient_id}_{sample_id}_image.png"

    # # --- load data ---
    cnts_joint = load_cnts(joint_cnts)
    cnts_single = load_cnts(single_cnts)

    locs_joint = load_locs(joint_locs)
    locs_single = load_locs(single_locs)

    labels_joint = load_labels_png(joint_labels_png)
    labels_single = load_labels_png(single_labels_png)

    # # align spot ordering between cnts and locs (just in case)
    common_joint = cnts_joint.index.intersection(locs_joint.index)
    cnts_joint = cnts_joint.loc[common_joint]
    locs_joint = locs_joint.loc[common_joint]

    common_single = cnts_single.index.intersection(locs_single.index)
    cnts_single = cnts_single.loc[common_single]
    locs_single = locs_single.loc[common_single]

    # --- assign domain per spot ---
    domains_joint = assign_domains_to_spots(locs_joint, labels_joint)
    domains_single = assign_domains_to_spots(locs_single, labels_single)

    # --- compute marker richness per domain ---
    df_rich_joint = compute_marker_richness_per_domain(
        cnts_joint, domains_joint, MARKER_GENES, expr_threshold=MARKER_EXPR_THRESHOLD
    )
    df_rich_single = compute_marker_richness_per_domain(
        cnts_single, domains_single, MARKER_GENES, expr_threshold=MARKER_EXPR_THRESHOLD
    )

    # -------------------------
    # 1) Visualization: domain maps
    # -------------------------
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # # joint
    # # he_j = Image.open(joint_he)
    # # axes[0].imshow(he_j)
    # low_res = Image.open(single_lowres)
    # axes[0].imshow(low_res)
    #
    # axes[1].imshow(labels_joint)
    # axes[1].set_title(f"Joint (patient {patient_id}, sample {sample_id})")
    # axes[1].axis("off")
    # # single
    # # he_s = Image.open(single_he)
    # # axes[1].imshow(he_s)
    # axes[2].imshow(labels_single)
    # axes[2].set_title(f"Single-slice (patient {patient_id}, sample {sample_id})")
    # axes[2].axis("off")
    # fig.suptitle("Gene-based tissue domains: joint vs single")
    # fig.tight_layout()
    # plt.show()

    # -------------------------
    # 2) Visualization: marker richness per domain
    # -------------------------
    title = (
        f"Marker richness per domain\n"
        f"patient {patient_id}, sample {sample_id} "
        f"(threshold={MARKER_EXPR_THRESHOLD})"
    )
    plot_richness_barplot(df_rich_joint, df_rich_single, title)
    plt.show()

    # Optionally return stats for further analysis
    return {
        "richness_joint": df_rich_joint,
        "richness_single": df_rich_single,
        "domains_joint": domains_joint,
        "domains_single": domains_single,
    }


# -------------------------------
# Example: run for a list of samples
# -------------------------------

if __name__ == "__main__":
    # Fill this with the patient/sample pairs you want to analyze
    patient_sample_pairs = [
        (2, 0),
        (2, 1),
        (2, 2),
        (5, 0),
        (5, 1),
        (5, 2),
        (9, 0),
        (9, 1),
        (9, 2),
        (10, 0),
        (10, 1),
        (10, 2)
    ]


    for pid, sid in patient_sample_pairs:
        results = analyze_patient_sample(pid, sid)
