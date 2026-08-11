import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


# ==========================================
# 1. Load iStar cluster results and data
# ==========================================

def load_cluster_results(run_root: Path):
    """
    Load all cluster-*.tsv for one iStar run.

    Expects:
      run_root/cnts-clustered/by-clusters/contrast/by-clusters/cluster-<id>.tsv

    Returns:
      clusters: list of cluster names (strings)
      cluster_tables: dict {cluster_name: DataFrame indexed by gene}
    """
    cluster_dir = (
        run_root
        / "cnts-clustered"
        / "by-clusters"
        / "contrast"
        / "by-clusters"
    )
    cluster_files = sorted(cluster_dir.glob("cluster-*.tsv"))
    if not cluster_files:
        raise FileNotFoundError(f"No cluster-*.tsv found in {cluster_dir}")

    clusters = []
    cluster_tables = {}
    for path in cluster_files:
        df = pd.read_csv(path, sep="\t")
        if "gene" not in df.columns:
            raise ValueError(f"'gene' column not found in {path}")
        df = df.set_index("gene")
        cname = path.stem.split("cluster-")[-1]  # e.g. "0", "1", ...
        clusters.append(cname)
        cluster_tables[cname] = df

    return clusters, cluster_tables


def load_labels_pickle(run_root: Path):
    """
    Load full-resolution domain labels: clusters-gene/labels.pickle.
    Returns 2D integer array of labels.
    """
    path = run_root / "clusters-gene" / "labels.pickle"
    with open(path, "rb") as f:
        labels = pickle.load(f)
    labels = np.array(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels array, got shape {labels.shape}")
    return labels.astype(int)


def load_gene_map(run_root: Path, gene: str, labels_shape):
    """
    Load super-resolution gene expression map for given gene:
      run_root/cnts-super/<gene>.pickle

    Returns 2D numpy array (same shape as labels) of expression values.
    """
    path = run_root / "cnts-super" / f"{gene}.pickle"
    if not path.is_file():
        raise FileNotFoundError(f"cnts-super file not found for gene {gene}: {path}")

    with open(path, "rb") as f:
        arr = pickle.load(f)

    arr = np.array(arr)
    # Try to match labels shape
    if arr.shape == labels_shape:
        return arr
    if arr.ndim == 3:
        arr = np.squeeze(arr)
    arr = np.array(arr)
    if arr.shape != labels_shape:
        raise ValueError(
            f"Gene map shape {arr.shape} does not match labels shape {labels_shape}"
        )
    return arr


# ==========================================
# 2. Select top 5 genes per cluster
# ==========================================

def get_top_genes_for_cluster(
    cluster_df: pd.DataFrame,
    n_top: int = 10,
    fc_col: str = "fold_change",
    pval_col: str = "pvalue_positive_sided",
    min_fc: float = 1.2,
    max_pval: float = 0.05,
):
    """
    Given one cluster table (one cluster's differential result),
    returns list of top genes (up to n_top) sorted by fold_change,
    filtered by fold_change >= min_fc and pvalue <= max_pval.
    """
    df = cluster_df.copy()
    if fc_col not in df.columns or pval_col not in df.columns:
        raise ValueError(f"Need columns {fc_col} and {pval_col} in cluster table.")

    df[fc_col] = df[fc_col].astype(float)
    df[pval_col] = df[pval_col].astype(float)

    mask = (df[fc_col] >= min_fc) & (df[pval_col] <= max_pval)
    df = df[mask].sort_values(fc_col, ascending=False)
    df = df.head(n_top)

    return df.index.to_list()


def collect_top_genes_all_clusters(
    clusters,
    cluster_tables,
    n_top: int = 10,
    min_fc: float = 1.2,
    max_pval: float = 0.05,
):
    """
    For all clusters in a run, get top genes per cluster.
    Returns:
      top_genes_per_cluster: dict {cluster_name: [gene1, ...]}
      all_genes: sorted list of unique genes appearing in any top list.
    """
    top_genes_per_cluster = {}
    all_genes = set()
    for cname in clusters:
        df = cluster_tables[cname]
        genes = get_top_genes_for_cluster(
            df, n_top=n_top, min_fc=min_fc, max_pval=max_pval
        )
        top_genes_per_cluster[cname] = genes
        all_genes.update(genes)
    return top_genes_per_cluster, sorted(all_genes)


# ==========================================
# 3. Gene-set activity maps per cluster
# ==========================================

def compute_gene_set_activity_maps(
    run_root: Path,
    top_genes_per_cluster: dict,
    labels: np.ndarray,
    log1p: bool = True,
):
    """
    For each cluster (domain) C, compute a gene-set activity map:

      gene_set_map_C[pixel] = mean_g( log1p(expr_g[pixel]) )

    where g runs over the top genes for cluster C.

    Returns:
      gene_set_maps: dict {cluster_name: 2D array (same shape as labels)}
    """
    labels = np.array(labels)
    labels_shape = labels.shape

    # Preload all needed gene maps
    all_genes = sorted({g for genes in top_genes_per_cluster.values() for g in genes})
    gene_maps = {}
    for gene in all_genes:
        try:
            gm = load_gene_map(run_root, gene, labels_shape)
        except FileNotFoundError:
            # Skip missing genes (can happen if gene was filtered for imputation)
            continue
        if log1p:
            gm = np.log1p(gm)
        gene_maps[gene] = gm

    gene_set_maps = {}

    for cname, genes in top_genes_per_cluster.items():
        valid_genes = [g for g in genes if g in gene_maps]
        if len(valid_genes) == 0:
            # No gene maps available for this cluster's top genes
            continue
        stack = np.stack([gene_maps[g] for g in valid_genes], axis=0)  # G × H × W
        gene_set_map = stack.mean(axis=0)  # H × W
        gene_set_maps[cname] = gene_set_map

    return gene_set_maps


def compute_gene_set_distributions(gene_set_maps: dict, labels: np.ndarray):
    """
    For each cluster C (source cluster defining top-gene set),
    and each cluster ID K in labels, collect:

      values_C[K] = gene-set activity values of pixels with label==K

    Returns:
      gene_set_expr: dict {cluster_name: {cluster_id: 1D array}}
      cluster_ids: sorted list of cluster IDs in labels
    """
    labels = np.array(labels)
    cluster_ids = np.unique(labels)
    cluster_ids = cluster_ids[cluster_ids >= 0]  # ignore negative labels if any
    cluster_ids = [int(c) for c in cluster_ids]

    gene_set_expr = {}
    for cname, gmap in gene_set_maps.items():
        expr_per_cluster = {}
        for cid in cluster_ids:
            mask = labels == cid
            vals = gmap[mask]
            expr_per_cluster[cid] = vals
        gene_set_expr[cname] = expr_per_cluster

    return gene_set_expr, cluster_ids


# ==========================================
# 4. Plot: one subplot per cluster's gene set
# ==========================================

def plot_all_clusters_gene_set_violins(
    gene_set_expr: dict,
    cluster_ids,
    run_label: str,
    top_genes_per_cluster: dict,
):
    """
    One multi-panel figure:
      - Each subplot corresponds to a source cluster C.
      - Within subplot: violin plot of gene-set activity across all clusters K.
      - Subplot title includes the top 5 gene names for C.
    """
    source_clusters = sorted(gene_set_expr.keys(), key=lambda x: int(x))
    n_clusters = len(source_clusters)
    if n_clusters == 0:
        print(f"No gene-set maps to plot for run {run_label}.")
        return

    n_cols = int(np.ceil(np.sqrt(n_clusters)))
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False,
    )

    x_positions = np.arange(len(cluster_ids))

    for idx, cname in enumerate(source_clusters):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]

        expr_per_cluster = gene_set_expr[cname]
        data = [expr_per_cluster[cid] for cid in cluster_ids]

        ax.violinplot(
            data,
            positions=x_positions,
            showmeans=True,
            showmedians=False,
            showextrema=False,
        )

        # Highlight the originating cluster
        target_cid = int(cname)
        if target_cid in cluster_ids:
            tidx = cluster_ids.index(target_cid)
            ax.scatter(
                x_positions[tidx],
                np.median(expr_per_cluster[target_cid]),
                color="red",
                zorder=3,
                label=f"Cluster {target_cid} median",
            )

        # Build a short gene list string for the title
        genes = top_genes_per_cluster.get(cname, [])
        gene_str = ", ".join(genes)
        # Truncate if too long
        if len(gene_str) > 60:
            gene_str = gene_str[:57] + "..."

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(cid) for cid in cluster_ids], rotation=0)
        ax.set_ylabel("gene-set activity (mean log1p)")
        ax.set_title(f"{run_label} – cluster {cname}\nTop genes: {gene_str}", fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    # Hide unused axes
    for idx in range(n_clusters, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis("off")

    fig.suptitle(
        f"{run_label} – top-5 gene-set activity per cluster\n"
        f"(titles show top genes for each cluster)",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


# ---------- wrapper: unchanged except we pass top_genes_per_cluster into plot ----------

def analyze_run_gene_set_violin(
    run_root: Path,
    run_label: str,
    n_top_genes: int = 5,
    min_fc: float = 1.2,
    max_pval: float = 0.05,
    log1p: bool = True,
):
    print(f"\n=== Running gene-set violin analysis for {run_label} ===")
    clusters, cluster_tables = load_cluster_results(run_root)
    labels = load_labels_pickle(run_root)

    top_genes_per_cluster, _ = collect_top_genes_all_clusters(
        clusters,
        cluster_tables,
        n_top=n_top_genes,
        min_fc=min_fc,
        max_pval=max_pval,
    )

    # Optional: print the genes clearly to console as well
    print("Top genes per cluster:")
    for cname in sorted(top_genes_per_cluster.keys(), key=lambda x: int(x)):
        print(f"  Cluster {cname}: {', '.join(top_genes_per_cluster[cname])}")

    gene_set_maps = compute_gene_set_activity_maps(
        run_root,
        top_genes_per_cluster,
        labels,
        log1p=log1p,
    )
    if not gene_set_maps:
        print(f"No gene-set maps could be computed for {run_label}.")
        return

    gene_set_expr, cluster_ids = compute_gene_set_distributions(gene_set_maps, labels)

    plot_all_clusters_gene_set_violins(
        gene_set_expr,
        cluster_ids,
        run_label,
        top_genes_per_cluster=top_genes_per_cluster,
    )



# ==========================================
# 6. Example usage: single vs joint
# ==========================================

if __name__ == "__main__":
    # Replace with your actual paths
    single_root = Path(
        "/media/huifang/data/registration/SCC/istar/single_input/group_10/axis_1/sample1/"
    )
    joint_root = Path(
        "/media/huifang/data/registration/SCC/istar/joint/group_10/sample1/"
    )

    # Single-slice run
    analyze_run_gene_set_violin(
        single_root,
        run_label="Single slice",
        n_top_genes=10,
        min_fc=1.5,
        max_pval=0.01,
        log1p=True,
    )

    # AXIS joint run
    analyze_run_gene_set_violin(
        joint_root,
        run_label="AXIS joint",
        n_top_genes=10,
        min_fc=1.5,
        max_pval=0.01,
        log1p=True,
    )
