from sklearn.decomposition import NMF
import scipy.sparse
import sklearn
from matplotlib import pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.spatial import cKDTree
from anndata import AnnData
import networkx as nx
from sklearn.neighbors import NearestNeighbors
### ----------------------- FILE LOADING FUNCTION -----------------------

def load_registration_result(folder_path):
    """Load image and spot data from a given folder."""
    data = np.load(folder_path)
    image = data["img_fix"]
    if image.shape[0]>512 and image.shape[0]<1500:
        pts1 = data["pts_fix"]/2
        pts2 = data["pts_warp"]/2
        pts3 = data["pts_moving"] / 2
        return pts1,pts2,pts3,data["fixed_label"],data['moving_label']
    elif image.shape[0] >1500 and image.shape[0]<2500:
        pts1 = data["pts_fix"]/4
        pts2 = data["pts_warp"] / 4
        pts3 = data["pts_moving"] / 4
        return pts1,pts2,pts3,data["fixed_label"],data['moving_label']
    else:
        return data["pts_fix"], data["pts_warp"], data["pts_moving"],data["fixed_label"],data['moving_label']


def load_gene_expression(adata, genes):
    gene_expr = adata[:, genes].X
    if hasattr(gene_expr, 'toarray'):
        gene_expr = gene_expr.toarray()

    # Double log1p transform
    gene_expr = np.log1p(np.log1p(gene_expr))

    # Min-max normalization per gene
    gene_expr_norm = (gene_expr - gene_expr.min(axis=0)) / (gene_expr.max(axis=0) - gene_expr.min(axis=0) + 1e-6)
    return gene_expr_norm

def integrate_slices(fix_adata_path, moving_adata_paths, registration_paths):
    fix_data = sc.read_h5ad(fix_adata_path)
    slices=[]
    for idx, (adata_path, reg_path) in enumerate(zip(moving_adata_paths, registration_paths)):
        coords_fixed, coords_warp, _, _, _ = load_registration_result(reg_path)
        if idx==0:
            fix_data.obsm['spatial'] = coords_fixed
            slices.append(fix_data)

        moving_data = sc.read_h5ad(adata_path)
        moving_data.obsm['spatial'] = coords_warp
        slices.append(moving_data)

    adata_combined = sc.concat(slices, join='outer', label='batch', fill_value=0)
    return adata_combined

def spatial_resample_to_adata(adata, grid_size=50):
    coords = adata.obsm["spatial"]
    expr = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    # Compute spatial bounds
    x_min, y_min = coords.min(axis=0)
    # Map each spot to grid cell
    col_idx = ((coords[:, 0] - x_min) / grid_size).astype(int)
    row_idx = ((coords[:, 1] - y_min) / grid_size).astype(int)

    # Dictionary to accumulate gene reads
    grid_dict = {}
    for i in range(expr.shape[0]):
        key = (row_idx[i], col_idx[i])
        if key not in grid_dict:
            grid_dict[key] = expr[i].copy()
        else:
            grid_dict[key] += expr[i]

    # Build new AnnData
    new_X = []
    new_coords = []
    for (r, c), vals in sorted(grid_dict.items()):
        new_X.append(vals)
        new_coords.append([x_min + c * grid_size, y_min + r * grid_size])

    new_X = np.array(new_X)
    new_coords = np.array(new_coords)

    new_adata = AnnData(X=new_X, var=adata.var.copy())
    new_adata.obsm["spatial"] = new_coords
    new_adata.obs_names = [f"grid_{i}" for i in range(new_adata.n_obs)]

    return new_adata

def spatial_resample_nmf(adata: AnnData, grid_size: int = 64, n_components: int = 20) -> AnnData:
    from sklearn.decomposition import NMF

    coords = adata.obsm['spatial']
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    # Compute grid boundaries
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)

    # Create regular grid
    x_edges = np.linspace(min_x, max_x, grid_size + 1)
    y_edges = np.linspace(min_y, max_y, grid_size + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Assign spots to grid
    x_bin = np.digitize(coords[:, 0], x_edges) - 1
    y_bin = np.digitize(coords[:, 1], y_edges) - 1
    x_bin = np.clip(x_bin, 0, grid_size - 1)
    y_bin = np.clip(y_bin, 0, grid_size - 1)

    grid = np.zeros((grid_size, grid_size, X.shape[1]))
    counts = np.zeros((grid_size, grid_size))

    for i in range(coords.shape[0]):
        grid[y_bin[i], x_bin[i]] += X[i]
        counts[y_bin[i], x_bin[i]] += 1

    # Fill empty grids with nearest non-empty grid values
    filled_grid = grid.copy()
    non_empty_idx = np.argwhere(counts > 0)
    empty_idx = np.argwhere(counts == 0)

    if len(empty_idx) > 0:
        tree = cKDTree(non_empty_idx)
        _, nearest_indices = tree.query(empty_idx)
        for i, empty in enumerate(empty_idx):
            filled_grid[tuple(empty)] = grid[tuple(non_empty_idx[nearest_indices[i]])]

    # Flatten and apply NMF
    flat_grid = filled_grid.reshape(-1, X.shape[1])
    model = NMF(n_components=n_components, solver = 'mu', beta_loss = 'kullback-leibler', init='random', random_state = None, verbose = False)
    W = model.fit_transform(flat_grid)
    H = model.components_
    smoothed_flat = W @ H
    smoothed_grid = smoothed_flat.reshape((grid_size, grid_size, X.shape[1]))

    # Construct new AnnData
    new_coords = grid_coords
    new_expr = smoothed_grid.reshape(-1, X.shape[1])
    new_adata = AnnData(X=new_expr)
    new_adata.obsm['spatial'] = new_coords

    return new_adata

def match_cluster_labels(true_labels, est_labels):
    true_labels_arr = np.array(list(true_labels))
    est_labels_arr = np.array(list(est_labels))
    org_cat = list(np.sort(list(pd.unique(true_labels))))
    est_cat = list(np.sort(list(pd.unique(est_labels))))
    B = nx.Graph()
    B.add_nodes_from([i + 1 for i in range(len(org_cat))], bipartite=0)
    B.add_nodes_from([-j - 1 for j in range(len(est_cat))], bipartite=1)
    for i in range(len(org_cat)):
        for j in range(len(est_cat)):
            weight = np.sum((true_labels_arr == org_cat[i]) * (est_labels_arr == est_cat[j]))
            B.add_edge(i + 1, -j - 1, weight=-weight)
    match = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)
    #     match = minimum_weight_full_matching(B)
    if len(org_cat) >= len(est_cat):
        return np.array([match[-est_cat.index(c) - 1] - 1 for c in est_labels_arr])
    else:
        unmatched = [c for c in est_cat if not (-est_cat.index(c) - 1) in match.keys()]
        l = []
        for c in est_labels_arr:
            if (-est_cat.index(c) - 1) in match:
                l.append(match[-est_cat.index(c) - 1] - 1)
            else:
                l.append(len(org_cat) + unmatched.index(c))
        return np.array(l)


def cluster_adata(adata, n_clusters=7, sample_name='', use_nmf=False):
    from sklearn.cluster import KMeans
    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy, inplace=True)
    sc.pp.log1p(adata_copy)
    sc.pp.highly_variable_genes(adata_copy, flavor="seurat", n_top_genes=1000)
    sc.pp.pca(adata_copy)

    if use_nmf:
        model = sklearn.decomposition.NMF(n_components=50)
        adata_copy.obsm['X_pca'] = model.fit_transform(adata_copy.X)

    cluster_labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=500).fit_predict(adata_copy.obsm['X_pca'])

    adata_copy.obs['my_clusters'] = pd.Series(
        1 + match_cluster_labels(adata_copy.obs['layer_guess_reordered'], cluster_labels), index=adata_copy.obs.index,
        dtype='category')

    ari = sklearn.metrics.adjusted_rand_score(adata_copy.obs['layer_guess_reordered'], adata_copy.obs['my_clusters'])
    print('ARI', ari)
    adata.obs['my_clusters'] = adata_copy.obs['my_clusters'].copy()
    return

def get_scatter_contours(layer, labels, interest=['WM']):
    idx = np.array(range(len(labels)))[(labels.isin(interest)).to_numpy()]
    idx_not = np.array(range(len(labels)))[(labels.isin(set(labels.cat.categories).difference(interest))).to_numpy()]
    dist = scipy.spatial.distance_matrix(layer.obsm['spatial'], layer.obsm['spatial'])
    min_dist = np.min(dist[dist > 0])
    eps = 0.01
    edges = np.zeros(dist.shape)
    edges[dist > 0] = (dist[dist > 0] - min_dist) ** 2 < eps
    border = list(filter(lambda x: np.sum(edges[x, idx_not] > 0), idx))
    # Early return if border is empty
    if len(border) == 0:
        return []  # or return an empty list
    j = np.argmin(layer.obsm['spatial'][border, 0])
    contours, left = [[border[j]]], set(border).difference(set([border[j]]))
    for i in range(1, len(border)):
        last = contours[-1][-1]
        neighbors = set(left).intersection(np.where((dist[last, :] - min_dist) ** 2 < eps)[0])
        if len(neighbors) > 0:
            j = neighbors.pop()
            contours[-1].append(j)
        else:
            l = list(left)
            j = l[np.argmin(layer.obsm['spatial'][l, 0])]
            contours.append([j])
        left = left.difference(set([j]))
    return contours

def draw_spatial(adata, clusters='my_clusters', sample_name='', draw_contours=False):
    fig = plt.figure(figsize=(12, 10))
    ax = sc.pl.spatial(adata, color=clusters, spot_size=5, show=False,
                       palette=['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d'],
                       ax=plt.gca())
    ax[0].axis('off')
    ari = sklearn.metrics.adjusted_rand_score(adata.obs['layer_guess_reordered'], adata.obs[clusters])
    ax[0].legend(title='Cluster', bbox_to_anchor=(0.9, 0.9), fontsize=20, title_fontsize=20)
    ax[0].set_title('{}: ARI={:.4f}'.format(sample_name, ari), fontsize=26)
    if draw_contours:
        for l in ['Layer{}'.format(i) for i in [1, 3, 5]] + ['WM']:
            contours = get_scatter_contours(adata, adata.obs['layer_guess_reordered'], [l])
            for k in range(len(contours)):
                plt.plot(adata.obsm['spatial'][contours[k], 0], adata.obsm['spatial'][contours[k], 1], 'lime',
                         # dict(zip(['Layer{}'.format(i) for i in range(1,7)]+['WM'],adata.uns['layer_guess_reordered_colors']))[l],
                         lw=4, alpha=0.6)
    plt.gca().text(105, 150, 'L1')
    plt.gca().text(105, 220, 'L2')
    plt.gca().text(105, 260, 'L3')
    plt.gca().text(105, 305, 'L4')
    plt.gca().text(105, 340, 'L5')
    plt.gca().text(105, 380, 'L6')
    plt.gca().text(105, 425, 'WM')
    plt.show()


def plot_marker_genes(adata, genes, gene_expr_norm):
    n = len(genes)
    # fig, axes = plt.subplots(1, n, figsize=(8 * n, 10), constrained_layout=True)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)

    for i, gene in enumerate(genes):
        ax = axes[i] if n > 1 else axes
        expr = gene_expr_norm[:, i]
        cmap = plt.cm.Reds
        colors = cmap(expr)
        colors[:, -1] = expr  # set alpha as expression level

        ax.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1],
                   color=colors, s=10)
        ax.set_title(gene,fontsize=26)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

    plt.show()






root_folder = "/media/huifang/data/registration/result/center_align/DLPFC/attention_fusion/"
gene_folder = "/home/huifang/workspace/code/registration/data/DLPFC"
sampleids=[['151507','151508','151509','151510'],['151669','151670','151671','151672'],['151673','151674','151675','151676']]
for i in range(2,3):

    fix_adata_path = f"{gene_folder}/{sampleids[i][0]}_preprocessed.h5"
    moving_adata_paths = [
        f"{gene_folder}/{sampleids[i][1]}_preprocessed.h5",
        f"{gene_folder}/{sampleids[i][2]}_preprocessed.h5",
        f"{gene_folder}/{sampleids[i][3]}_preprocessed.h5"
    ]
    registration_paths = [
        f"{root_folder}/{i}_0_result.npz",
        f"{root_folder}/{i}_1_result.npz",
        f"{root_folder}/{i}_2_result.npz"
    ]
    genes = ['MFGE8','MOBP','PCP4','TRABD2A']
    # adata_single = sc.read_h5ad(fix_adata_path)
    adata_single = sc.read_h5ad('/home/huifang/workspace/code/registration/data/DLPFC/saved_results/center2_a0.1_KL_seed2.h5ad')
    print(adata_single)
    # # print(adata_single)
    # # test = input()
    exprs = load_gene_expression(adata_single, genes)
    plot_marker_genes(adata_single, genes, exprs)
    # cluster_adata(adata_single, 7, sample_name="151671")
    # draw_spatial(adata_single, 'my_clusters',
    #              {"151671": 'Slice A', "151672": 'Slice B'}["151671"],
    #              draw_contours=True)



    adata_combined = integrate_slices(fix_adata_path, moving_adata_paths, registration_paths)


    adata_combined = spatial_resample_to_adata(adata_combined,grid_size=3)
    print(adata_combined)
    adata_combined = spatial_resample_nmf(adata_combined,grid_size=100)
    # print(adata_combined.shape)

    # print(adata_combined)
    # print(new_adata)
    # test = input()
    exprs = load_gene_expression(adata_combined, genes)
    plot_marker_genes(adata_combined, genes, exprs)

    # cmap = plt.cm.Reds
    # colors = cmap(norm_expr)
    # colors[:, -1] = norm_expr  # transparency reflects expression level
    # plt.scatter(adata_combined.obsm['spatial'][:, 0], adata_combined.obsm['spatial'][:, 1], color=colors, s=10)
    # plt.gca().invert_yaxis()
    # plt.gca().set_aspect('equal')
    # plt.show()
