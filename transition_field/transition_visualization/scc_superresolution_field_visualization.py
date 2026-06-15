from istar_utils import *
from axis.coordinate_utils import find_inverse_warp_coords
from visualization_utils import *


def load_flow_np(root,group_id,fixed_id,moving_id):
    flow_path = parse_flow_path(
        root=root,
        group_id=group_id,
        fixed_id=moving_id,
        moving_id=fixed_id,
        pairwise_subdir="pairwise_output",
    )
    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"Missing flow file: {flow_path}")

    flow = np.load(flow_path)
    return flow


def precompute_all_hr_flows(patient_id, sample_ids, axis_root, istar_root, out_root):
    for fixed_sid, moving_sid in zip(sample_ids[:-1], sample_ids[1:]):
        low_flow = load_flow_np(axis_root,patient_id, fixed_sid, moving_sid)

        fixed_mask = load_istar_mask(
            patient_id=patient_id,
            axis_id=fixed_sid,
            sample_id=fixed_sid,
            istar_root=istar_root,
        )
        target_shape = fixed_mask.shape

        flow_hr = upsample_flow_to_target_shape(low_flow, target_shape=target_shape)


        save_path = (
            Path(out_root)
            / f"group_{patient_id}"
            / f"{fixed_sid}_{moving_sid}_flow_hr.npz"
        )
        save_istar_hr_flow(
            flow_np=flow_hr,
            save_path=save_path,
            fixed_id=fixed_sid,
            moving_id=moving_sid,
            fixed_shape=target_shape,
        )


def make_istar_gene_seed_mask(
    vals,
    percentile=95.0,
    min_value=None,
    percentile_on_nonzero=True,
):


    finite = np.isfinite(vals)
    ref = vals[finite]

    if ref.size == 0:
        raise ValueError("No finite gene values found in slice payload")

    if min_value is None:
        if percentile_on_nonzero:
            ref2 = ref[ref > 0]
            if ref2.size == 0:
                raise ValueError("No nonzero values found for percentile thresholding")
            thr = float(np.percentile(ref2, percentile))
        else:
            thr = float(np.percentile(ref, percentile))
    else:
        thr = float(min_value)

    mask = finite & (vals >= thr)
    return mask, {"threshold": thr, "n_seed": int(mask.sum())}


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
    fig_width=18,
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
        temp = load_istar_gene_slice(
            patient_id=group_id,
            sample_id=sid,
            gene=gene,
            istar_root=ISTAR_ROOT,
            axis_id=sid,
            use_log1p=use_log1p,
        )
        vals = temp["values"]
        coord = temp["coords"]

        gene_vals_list.append(vals)

        slice_cache[sid] = {
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
    first_vals = slice_cache[first_sid]["gene"]
    first_coord = slice_cache[first_sid]["coord"]


    seed_mask, seed_info = make_istar_gene_seed_mask(
        first_vals,
        percentile=95,
        min_value=None,
        percentile_on_nonzero=True,
    )

    seed_coord = first_coord[seed_mask]

    if seed_coord.shape[0] == 0:
        raise ValueError(f"No seed points found for domains={trajectory_domains}")

    trajectories = [seed_coord.copy()]
    active = np.ones(seed_coord.shape[0], dtype=bool)
    current = seed_coord.copy()

    for i in range(len(sample_ids) - 1):
        fixed_id = sample_ids[i]
        moving_id = sample_ids[i + 1]

        flow_path = (
                Path(HR_FLOW_ROOT)
                / f"group_{group_id}"
                / f"{fixed_id}_{moving_id}_flow_hr.npz"
        )
        flow, _ = load_istar_hr_flow(flow_path)
        flow = ensure_flow_hw2(flow)

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
    # for sid in sample_ids:
        # ax.text(
        #     plane_pos[sid],
        #     y_min - y_pad * 0.7,
        #     z_min_ - z_pad * 0.7,
        #     # f"Group{group_id}_Sample{sid}",
        #     f"Patient{group_id}_Sample{sid}",
        #     fontsize=16,
        #     color="black",
        # )

    if gene is not None and show_colorbar:
        sm = ScalarMappable(norm=norm, cmap=gene_cmap)
        sm.set_array([])
        cax = fig.add_axes([0.86, 0.20, 0.016, 0.26])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label)

    if save_path is not None:
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"[OK] Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

PATIENT_ID=2
SAMPLE_IDS=[0,1,2]
AXIS_ROOT="/media/huifang/data/registration/SCC/huifang/"
ISTAR_ROOT="/media/huifang/data/registration/SCC/istar/single_input"
HR_FLOW_ROOT="/media/huifang/data/registration/SCC/huifang/istar_hr_flow"

# precompute_all_hr_flows(patient_id=PATIENT_ID, sample_ids=SAMPLE_IDS, axis_root=AXIS_ROOT, istar_root=ISTAR_ROOT, out_root=HR_FLOW_ROOT)


plot_true_slices_with_coherent_trajectories_3d(
        root=AXIS_ROOT,
        group_id=PATIENT_ID,
        sample_ids=SAMPLE_IDS,
        gene="COL17A1",
        pairwise_subdir="pairwise_output",     # or adjust if your flow files are directly under root/group_[id]
        axis_positions=None,                   # or e.g. [2.5, 5.7, 17.9]
        depth_gap=400.0,
        physical_spacing=False,
        use_log1p=True,
        slice_point_size=0.1,
        # gene_cmap="tab10",
        gene_cmap="coolwarm",
        alpha_min=0.9,
        alpha_max=1.0,
        alpha_gamma=1.6,
        trajectory_domains=None,  # only for seed trajectories
        max_seed_points=1200,
        trajectory_alpha=1.0,
        trajectory_linewidth=0.01,
        trajectory_color="crimson",
        elev=15,
        azim=-145,
        fig_width=10,
        fig_height=10,
        save_path=None,   # e.g. os.path.join(root, "coherent_transition_3d.png")
        show=True,
        random_seed=0,
        xlim=[0, 800],
        ylim=[0, 800],

    )