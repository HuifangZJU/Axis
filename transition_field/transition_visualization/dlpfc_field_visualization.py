import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from axis.coordinate_utils import find_inverse_warp_coords
from visualization_utils import *
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
# ============================================================
# Basic path helpers
# ============================================================

def parse_h5ad_path(root, group_id, sample_id):
    return os.path.join(root, "h5ad", f"{group_id}_{sample_id}.h5ad")


def parse_network_npz_path(root, group_id, sample_id):
    return os.path.join(root, "network_input", f"{group_id}_{sample_id}_validation.npz")


def find_group_dir(root, group_id, pairwise_subdir="pairwise_output"):
    p1 = os.path.join(root, pairwise_subdir, f"group_{group_id}")
    p2 = os.path.join(root, f"group_{group_id}")

    if os.path.isdir(p1):
        return p1
    if os.path.isdir(p2):
        return p2

    raise FileNotFoundError(
        f"Cannot find pairwise folder for group {group_id}.\n"
        f"Tried:\n  {p1}\n  {p2}"
    )


def parse_flow_path(root, group_id, fixed_id, moving_id, pairwise_subdir="pairwise_output"):
    group_dir = find_group_dir(root, group_id, pairwise_subdir=pairwise_subdir)
    return os.path.join(group_dir, f"{fixed_id}_{moving_id}_flow.npy")


# ============================================================
# Data helpers
# ============================================================

def load_npz_coord(npz_path):
    data = np.load(npz_path)
    if "coord" not in data:
        raise KeyError(f"'coord' not found in {npz_path}")
    coord = np.asarray(data["coord"], dtype=np.float32)
    if coord.ndim != 2 or coord.shape[1] < 2:
        raise ValueError(f"coord in {npz_path} must have shape [N,2+], got {coord.shape}")
    return coord[:, :2]




def extract_gene_vals(adata, gene, use_log1p=True):
    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names")

    j = int(np.where(adata.var_names == gene)[0][0])
    v = adata.X[:, j]

    if sp.issparse(v):
        v = np.asarray(v.toarray()).ravel().astype(np.float32)
    else:
        v = np.asarray(v).ravel().astype(np.float32)

    if use_log1p:
        v = np.log1p(v)

    return v

def extract_obs_vals(
    adata,
    obs_key,
    use_log1p_continuous=False,
    categorical_as_codes=True,
):
    """
    Extract obs values in a visualization-friendly numeric format.

    Returns
    -------
    vals : np.ndarray, shape (n_obs,), float32
        Numeric values ready for scatter(c=vals, ...).
        For categorical attributes, this is category code as float32.
        Missing values are returned as np.nan.
    meta : dict
        {
            "kind": "continuous" | "categorical" | "boolean",
            "categories": list[str] or None,
            "is_discrete": bool,
        }
    """

    if obs_key not in adata.obs.columns:
        raise ValueError(f"obs key '{obs_key}' not found in adata.obs.")

    s = adata.obs[obs_key]

    # ---- boolean ----
    if pd.api.types.is_bool_dtype(s):
        vals = s.astype(np.float32).to_numpy()
        meta = {
            "kind": "boolean",
            "categories": ["False", "True"],
            "is_discrete": True,
        }
        return vals, meta

    # ---- numeric continuous ----
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)
        if use_log1p_continuous:
            valid = np.isfinite(vals)
            if np.any(vals[valid] < 0):
                raise ValueError(f"obs['{obs_key}'] contains negative values; cannot apply log1p safely.")
            vals[valid] = np.log1p(vals[valid]).astype(np.float32)
        meta = {
            "kind": "continuous",
            "categories": None,
            "is_discrete": False,
        }
        return vals, meta

    # ---- categorical / object / string ----
    cat = pd.Categorical(s)
    codes = cat.codes.astype(np.float32)
    codes[codes < 0] = np.nan   # missing values become NaN

    meta = {
        "kind": "categorical",
        "categories": [str(x) for x in cat.categories.tolist()],
        "is_discrete": True,
    }
    return codes, meta

def extract_obs_vals(
    adata,
    obs_key,
    use_log1p_continuous=False,
    categorical_as_codes=True,
):
    """
    Extract obs values in a visualization-friendly numeric format.

    Returns
    -------
    vals : np.ndarray, shape (n_obs,), float32
        Numeric values ready for scatter(c=vals, ...).
        For categorical attributes, this is category code as float32.
        Missing values are returned as np.nan.
    meta : dict
        {
            "kind": "continuous" | "categorical" | "boolean",
            "categories": list[str] or None,
            "is_discrete": bool,
        }
    """

    if obs_key not in adata.obs.columns:
        raise ValueError(f"obs key '{obs_key}' not found in adata.obs.")

    s = adata.obs[obs_key]

    # ---- boolean ----
    if pd.api.types.is_bool_dtype(s):
        vals = s.astype(np.float32).to_numpy()
        meta = {
            "kind": "boolean",
            "categories": ["False", "True"],
            "is_discrete": True,
        }
        return vals, meta

    # ---- numeric continuous ----
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)
        if use_log1p_continuous:
            valid = np.isfinite(vals)
            if np.any(vals[valid] < 0):
                raise ValueError(f"obs['{obs_key}'] contains negative values; cannot apply log1p safely.")
            vals[valid] = np.log1p(vals[valid]).astype(np.float32)
        meta = {
            "kind": "continuous",
            "categories": None,
            "is_discrete": False,
        }
        return vals, meta

    # ---- categorical / object / string ----
    cat = pd.Categorical(s)
    codes = cat.codes.astype(np.float32)
    codes[codes < 0] = np.nan   # missing values become NaN

    meta = {
        "kind": "categorical",
        "categories": [str(x) for x in cat.categories.tolist()],
        "is_discrete": True,
    }
    return codes, meta




def robust_vmax(vals_list, q=0.995):
    if len(vals_list) == 0:
        return 1.0
    x = np.concatenate(vals_list, axis=0)
    if x.size == 0:
        return 1.0
    return max(float(np.quantile(x, q)), 1e-6)



def _build_label_color_map(labels, label_cmap="tab10"):
    all_labels = pd.Index(labels).unique().tolist()
    label_to_code = {lab: i for i, lab in enumerate(all_labels)}
    cmap = plt.get_cmap(label_cmap, max(len(all_labels), 1))

    codes = np.array([label_to_code[x] for x in labels], dtype=int)

    return cmap,codes






def make_domain_mask(adata, include_domains=None, domain_key="layer_guess_reordered", n_keep=None):
    """
    Boolean mask over obs for selected domains.
    """
    if n_keep is None:
        labels = adata.obs[domain_key]
    else:
        labels = adata.obs[domain_key].iloc[:n_keep]

    if include_domains is None:
        return np.ones(len(labels), dtype=bool)
    labels = np.asarray(labels.astype(str))
    include_domains = set(include_domains)
    return np.array([x in include_domains for x in labels], dtype=bool)


def choose_spatially_downsampled_points(coords, max_keep=800, seed=0):
    """
    Simple random downsample for seed trajectories.
    """
    n = coords.shape[0]
    if n <= max_keep:
        return np.arange(n, dtype=int)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_keep, replace=False)
    idx.sort()
    return idx


# ============================================================
# Coordinate loading
# ============================================================

def load_plot_slice_coords_aligned(
    root,
    group_id,
    sample_id,
    reference_id=0,
    coord_key="spatial",
    pairwise_subdir="pairwise_output",
):
    """
    Coordinates used for 3D plotting should be in the SAME coordinate system
    as the pairwise flow / inverse-warp propagation.

    Priority:
      1) adata.obsm[coord_key]
      2) reference slice -> network_input coord
      3) moving slice -> warped coord from reference
    """
    h5ad_path = parse_h5ad_path(root, group_id, sample_id)
    adata = sc.read_h5ad(h5ad_path)

    if coord_key in adata.obsm:
        coord = np.asarray(adata.obsm[coord_key], dtype=np.float32)
        return adata, coord[:, :2]

    if sample_id == reference_id:
        npz_path = parse_network_npz_path(root, group_id, sample_id)
        coord = load_npz_coord(npz_path)
        return adata, coord[:, :2]

    group_dir = find_group_dir(root, group_id, pairwise_subdir=pairwise_subdir)
    warped_path = os.path.join(group_dir, f"{reference_id}_{sample_id}_warped_coor.npy")
    if os.path.exists(warped_path):
        coord = np.asarray(np.load(warped_path), dtype=np.float32)
        return adata, coord[:, :2]

    raise ValueError(
        f"Could not find aligned plotting coordinates for group={group_id}, sample={sample_id}. "
        f"Please assign obsm['{coord_key}'] first, or provide warped coords."
    )

# ============================================================
# Inverse warp wrapper
# ============================================================

def inverse_warp_points(points_xy, flow_hw2):
    """
    Wrapper around axis.coordinate_utils.find_inverse_warp_coords.

    This helper tries a few common calling conventions because the exact
    signature may differ across local versions of the utility.
    """
    trials = [
        lambda: find_inverse_warp_coords(points_xy, flow_hw2),
    ]

    last_err = None
    for fn in trials:
        try:
            out = fn()

            # handle tuple/list returns like (coords, valid_mask)
            if isinstance(out, (tuple, list)):
                out = out[0]

            out = np.asarray(out, dtype=np.float32)
            if out.ndim != 2 or out.shape[1] < 2:
                raise ValueError(f"Unexpected inverse-warp output shape: {out.shape}")

            return out[:, :2]

        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Failed to call find_inverse_warp_coords with the tried signatures. "
        "Please adapt inverse_warp_points() to your local function signature.\n"
        f"Last error: {last_err}"
    )




# ============================================================
# Main plotting function
# ============================================================
def make_seed_mask(
    adata,
    include_domains=None,
    domain_key="layer_guess_reordered",
    gene=None,
    gene_percentile=90.0,
    gene_min_value=None,
    gene_use_log1p=True,
    percentile_on_nonzero=True,
    combine_mode="and",
    n_keep=None,
):
    """
    Build a seed mask for coherent trajectory visualization.

    Supports:
      1) domain-based selection
      2) gene-based high-expression selection
      3) combining both

    Parameters
    ----------
    adata : AnnData
    include_domains : list[str] or None
        Domain labels to keep, e.g. ["Layer3", "Layer4"].
    domain_key : str
        obs column storing domain labels.
    gene : str or None
        Marker gene name for seed selection.
    gene_percentile : float
        Percentile threshold for gene-based selection if gene_min_value is None.
    gene_min_value : float or None
        Absolute threshold on gene value. If provided, overrides percentile.
    gene_use_log1p : bool
        Whether to log1p-transform gene values before thresholding.
    percentile_on_nonzero : bool
        If True, compute percentile only on nonzero gene values.
    combine_mode : {"and", "or"}
        How to combine domain mask and gene mask when both are provided.
    n_keep : int or None
        Optional truncation length to match coordinate length.

    Returns
    -------
    seed_mask : np.ndarray of bool
    info : dict
        Useful metadata for debugging/printing.
    """
    if n_keep is None:
        adata_sub = adata
    else:
        adata_sub = adata[:n_keep].copy()

    n = adata_sub.n_obs
    masks = []
    info = {}

    # --------------------
    # Domain mask
    # --------------------
    if include_domains is not None:
        if domain_key not in adata_sub.obs:
            raise KeyError(f"'{domain_key}' not found in adata.obs")

        labels = np.asarray(adata_sub.obs[domain_key].astype(str))
        keep_set = set(include_domains)
        domain_mask = np.array([x in keep_set for x in labels], dtype=bool)

        masks.append(domain_mask)
        info["include_domains"] = list(include_domains)
        info["n_domain_kept"] = int(domain_mask.sum())

    # --------------------
    # Gene mask
    # --------------------
    if gene is not None:
        gene_vals =extract_gene_vals(adata_sub, gene)

        finite_mask = np.isfinite(gene_vals)
        valid_vals = gene_vals[finite_mask]

        if valid_vals.size == 0:
            raise ValueError(f"No finite values found for gene '{gene}'")

        if gene_min_value is not None:
            threshold = float(gene_min_value)
        else:
            if percentile_on_nonzero:
                ref_vals = valid_vals[valid_vals > 0]
                if ref_vals.size == 0:
                    raise ValueError(
                        f"Gene '{gene}' has no nonzero values, cannot compute nonzero percentile."
                    )
            else:
                ref_vals = valid_vals

            threshold = float(np.percentile(ref_vals, gene_percentile))

        gene_mask = finite_mask & (gene_vals >= threshold)

        masks.append(gene_mask)
        info["gene"] = gene
        info["gene_threshold"] = threshold
        info["gene_percentile"] = gene_percentile if gene_min_value is None else None
        info["gene_use_log1p"] = bool(gene_use_log1p)
        info["percentile_on_nonzero"] = bool(percentile_on_nonzero)
        info["n_gene_kept"] = int(gene_mask.sum())

    # --------------------
    # Combine masks
    # --------------------
    if len(masks) == 0:
        seed_mask = np.ones(n, dtype=bool)
        info["mode"] = "all_points"
    elif len(masks) == 1:
        seed_mask = masks[0]
        info["mode"] = "single_criterion"
    else:
        if combine_mode == "and":
            seed_mask = masks[0].copy()
            for m in masks[1:]:
                seed_mask &= m
        elif combine_mode == "or":
            seed_mask = masks[0].copy()
            for m in masks[1:]:
                seed_mask |= m
        else:
            raise ValueError("combine_mode must be 'and' or 'or'")

        info["combine_mode"] = combine_mode

    info["n_total"] = int(n)
    info["n_seed"] = int(seed_mask.sum())

    return seed_mask, info

def plot_true_slices_with_coherent_trajectories_3d(
    root,
    group_id,
    sample_ids,
    gene=None,
    reference_id=0,
    coord_key="spatial",
    pairwise_subdir="pairwise_output",
    axis_positions=None,
    depth_gap=220.0,
    physical_spacing=False,
    use_log1p=True,
    slice_point_size=0.8,
    gene_cmap="viridis",
    alpha_min=0.02,
    alpha_max=0.70,   # lower than before so trajectories remain visible
    alpha_gamma=1.8,
    trajectory_domains=None,
    domain_key="layer_guess_reordered",
    max_seed_points=500,
    trajectory_alpha=0.42,
    trajectory_linewidth=1.0,
    trajectory_color="crimson",
    elev=16,
    azim=-82,
    fig_width=15,
    fig_height=10,
    xlim=None,
    ylim=None,
    zlim=None,
    show_colorbar=True,
    save_path=None,
    show=True,
    random_seed=0,
):
    if len(sample_ids) < 2:
        raise ValueError("sample_ids must contain at least two slices")

    # --------------------------------------------------------
    # Plane positions
    # --------------------------------------------------------
    if axis_positions is not None:
        if len(axis_positions) != len(sample_ids):
            raise ValueError("axis_positions must have the same length as sample_ids")
        plane_pos = {sid: float(pos) for sid, pos in zip(sample_ids, axis_positions)}
    else:
        if physical_spacing:
            s0 = float(sample_ids[0])
            plane_pos = {sid: (float(sid) - s0) * float(depth_gap) for sid in sample_ids}
        else:
            plane_pos = {sid: i * float(depth_gap) for i, sid in enumerate(sample_ids)}

    # --------------------------------------------------------
    # Load aligned slice coordinates for plotting + genes
    # --------------------------------------------------------
    slice_cache = {}
    gene_vals_list = []

    all_y = []
    all_z = []

    for sid in sample_ids:
        adata, coord = load_plot_slice_coords_aligned(
            root=root,
            group_id=group_id,
            sample_id=sid,
            reference_id=reference_id,
            coord_key=coord_key,
            pairwise_subdir=pairwise_subdir,
        )


        n = min(adata.n_obs, coord.shape[0])
        if adata.n_obs != coord.shape[0]:
            print(
                f"[WARN] group={group_id}, sample={sid}: "
                f"adata.n_obs={adata.n_obs}, coord.shape[0]={coord.shape[0]}. Using first {n}."
            )

        adata_sub = adata[:n].copy()
        coord = coord[:n, :2]

        if gene is not None:
            try:
                vals = extract_gene_vals(adata_sub, gene)
            except ValueError:
                    vals, meta = extract_obs_vals(adata_sub, gene)
            gene_vals_list.append(vals)
        else:
            vals = np.ones(n, dtype=np.float32)

        slice_cache[sid] = {
            "adata": adata_sub,
            "coord": coord,
            "gene": vals,
            "plane": plane_pos[sid],
        }

        all_y.append(coord[:, 0])
        all_z.append(coord[:, 1])

    all_y = np.concatenate(all_y, axis=0)
    all_z = np.concatenate(all_z, axis=0)

    # --------------------------------------------------------
    # Color norm
    # --------------------------------------------------------
    if gene is not None:
        vmax = robust_vmax(gene_vals_list, q=0.995)
        norm = Normalize(vmin=0.0, vmax=vmax)
        cbar_label = f"{gene} log1p(counts)" if use_log1p else gene
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)
        cbar_label = None

    # --------------------------------------------------------
    # Build coherent seed trajectories from slice 0 domain points
    # --------------------------------------------------------
    first_sid = sample_ids[0]
    first_adata = slice_cache[first_sid]["adata"]
    first_coord = slice_cache[first_sid]["coord"]

    if domain_key not in first_adata.obs:
        raise KeyError(f"'{domain_key}' not found in first slice obs")

    seed_mask = make_domain_mask(
        first_adata,
        include_domains=trajectory_domains,
        domain_key=domain_key,
        n_keep=first_coord.shape[0],
    )

    seed_coord = first_coord[seed_mask]
    if seed_coord.shape[0] == 0:
        raise ValueError(f"No seed points found for domains={trajectory_domains}")

    keep_idx = choose_spatially_downsampled_points(
        seed_coord, max_keep=max_seed_points, seed=random_seed
    )
    seed_coord = seed_coord[keep_idx]

    trajectories = [seed_coord.copy()]
    active = np.ones(seed_coord.shape[0], dtype=bool)
    current = seed_coord.copy()

    for i in range(len(sample_ids) - 1):
        fixed_id = sample_ids[i]
        moving_id = sample_ids[i + 1]

        flow_path = parse_flow_path(
            root=root,
            group_id=group_id,
            fixed_id=moving_id,
            moving_id=fixed_id,
            pairwise_subdir=pairwise_subdir,
        )
        if not os.path.exists(flow_path):
            raise FileNotFoundError(f"Missing flow file: {flow_path}")

        flow = ensure_flow_hw2(np.load(flow_path))

        next_all = np.full((seed_coord.shape[0], 2), np.nan, dtype=np.float32)

        if np.any(active):
            next_pts = inverse_warp_points(current[active], flow)

            valid_now = np.all(np.isfinite(next_pts), axis=1)
            H, W, _ = flow.shape
            valid_now &= (
                (next_pts[:, 0] >= 0) & (next_pts[:, 0] < W) &
                (next_pts[:, 1] >= 0) & (next_pts[:, 1] < H)
            )

            idx_active = np.where(active)[0]
            idx_keep = idx_active[valid_now]

            next_all[idx_keep] = next_pts[valid_now]

            new_active = np.zeros_like(active)
            new_active[idx_keep] = True
            active = new_active
            current = next_all.copy()

        trajectories.append(next_all)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.03, 0.06, 0.80, 0.88])
    ax.set_box_aspect((2, 1.3, 1.3))

    ax.set_proj_type("ortho")
    # global slice plane extent
    y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    z_min_, z_max_ = float(np.nanmin(all_z)), float(np.nanmax(all_z))
    y_pad = max(10.0, 0.03 * (y_max - y_min + 1e-6))
    z_pad = max(10.0, 0.03 * (z_max_ - z_min_ + 1e-6))

    # 1) draw slice planes first
    for sid in sample_ids:
        draw_slice_plane(
            ax=ax,
            x_plane=plane_pos[sid],
            y_min=y_min - y_pad,
            y_max=y_max + y_pad,
            z_min=z_min_ - z_pad,
            z_max=z_max_ + z_pad,
            plane_alpha=0.04,
            edge_alpha=0.20,
        )

    # 2) draw coherent trajectories next
    draw_trajectory_segments_pairwise(
        ax=ax,
        sample_ids=sample_ids,
        plane_pos=plane_pos,
        trajectories=trajectories,
        color=trajectory_color,
        alpha=trajectory_alpha,
        linewidth=trajectory_linewidth,
    )

    # 3) draw each slice point cloud separately last
    for sid in sample_ids:
        coord = slice_cache[sid]["coord"]
        vals = slice_cache[sid]["gene"]
        x_plane = slice_cache[sid]["plane"]

        if gene is not None:
            rgba_slice = make_rgba_with_value_alpha_abs(
                vals=vals,
                norm=norm,
                cmap_name=gene_cmap,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                gamma=alpha_gamma,
            )
        else:
            rgba_slice = get_cmap("Greys")(np.full(coord.shape[0], 0.55))
            rgba_slice[:, 3] = 0.20

        ax.scatter(
            np.full(coord.shape[0], x_plane, dtype=np.float32),
            coord[:, 0],
            coord[:, 1],
            c=rgba_slice,
            s=slice_point_size,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )

    # limits
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        dmin = min(plane_pos.values())
        dmax = max(plane_pos.values())
        dpad = max(20.0, 0.05 * (dmax - dmin + 1e-6))
        ax.set_xlim(dmin - dpad, dmax + dpad)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    if zlim is not None:
        ax.set_zlim(zlim)
    else:
        ax.set_zlim(z_min_ - z_pad, z_max_ + z_pad)



    if ylim is not None:
        ax.set_zlim(ylim[::-1])
        ax.set_ylim(xlim[::-1])
    # ax.set_xlabel("Slice order")
    # ax.set_ylabel("Tissue x")
    # ax.set_zlabel("Tissue y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    # ax.axis("off")

    # try:
    #     ax.set_box_aspect((4.6, 1.5, 1.5))
    # except Exception:
    #     pass

    ax.grid(False)
    ax.set_title(
        f"Group {group_id} | samples {sample_ids} | all gene points + coherent trajectories from {trajectory_domains}",
        pad=18,
    )

    # slice labels
    for sid in sample_ids:
        ax.text(
            plane_pos[sid],
            y_min + y_pad * 3,
            z_min_ - z_pad * 2,
            f"Sample{sid}",
            fontsize=14,
            color="black",
        )

    if gene is not None and show_colorbar:
        layer_label=["Layer3","Layer4","Layer5","Layer6","WM"]
        cluster_labels = sorted(np.unique(vals).astype(int).tolist())
        cmap_obj = plt.get_cmap(gene_cmap)
        handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=cmap_obj(lab),
                markeredgecolor="none",
                markersize=8,
                label=layer_label[lab],
            )
            for lab in cluster_labels
        ]

        fig.legend(
            handles=handles,
            title="Domains",
            loc="center left",
            bbox_to_anchor=(0.65, 0.6),
            frameon=False,
            ncol=1,
            handlelength=0.8,
            handletextpad=0.4,
            labelspacing=0.38,
            borderaxespad=0.0,
            fontsize=13,
            title_fontsize=15,
        )

        # sm = ScalarMappable(norm=norm, cmap=gene_cmap)
        # sm.set_array([])
        # cax = fig.add_axes([0.86, 0.20, 0.016, 0.26])
        # cbar = fig.colorbar(sm, cax=cax)
        # cbar.set_label(cbar_label)

    if save_path is not None:
        # plt.savefig(save_path, dpi=400, bbox_inches="tight")
        # print(f"[OK] Saved: {save_path}")
        fig = plt.gcf()
        w, h = fig.get_size_inches()

        # crop middle region: x0, y0, width, height as fractions
        x0, y0, ww, hh = 0.2, 0.22, 0.45, 0.58

        from matplotlib.transforms import Bbox
        bbox = Bbox.from_bounds(x0 * w, y0 * h, ww * w, hh * h)

        fig.savefig(save_path, dpi=400, bbox_inches=bbox)
        print(f"[OK] Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    root = "/media/huifang/data/registration/DLPFC/huifang"
    group_id = 1
    sample_ids = [0,1,2,3]

    plot_true_slices_with_coherent_trajectories_3d(
        root=root,
        group_id=group_id,
        sample_ids=sample_ids,
        gene="layer_guess_reordered",
        pairwise_subdir="pairwise_output",  # or adjust if your flow files are directly under root/group_[id]
        axis_positions=None,
        depth_gap=160.0,
        physical_spacing=False,
        use_log1p=True,
        slice_point_size=2,
        gene_cmap="tab10",
        alpha_min=0.5,
        alpha_max=0.5,
        alpha_gamma=1.6,
        trajectory_domains=["Layer4"],  # only for seed trajectories
        # trajectory_domains=None,
        domain_key="layer_guess_reordered",
        max_seed_points=1200,
        trajectory_alpha=0.2,
        trajectory_linewidth=1,
        trajectory_color="crimson",
        elev=10,
        azim=-135,
        fig_width=18,
        fig_height=10,
        # save_path="/media/huifang/data/registration/DLPFC/evaluation/transition_3d.png",  # e.g. os.path.join(root, "coherent_transition_3d.png")
        save_path=None,
        show=True,
        random_seed=0,
        xlim=[0, 512],
        ylim=[0, 512],
    )
