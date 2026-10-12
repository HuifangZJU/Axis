"""Diagnostic plots used by the inference scripts (``visualization = True``)."""

import itertools
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt


def visualize_flow(flow, step=50, title="Flow field"):
    """
    Visualize a 2D flow field using a quiver plot.

    Args:
        flow: torch.Tensor or np.ndarray
              Shape [2, H, W] or [1, 2, H, W]
        step: int, subsampling step for arrows
        title: str, plot title
    """
    # Ensure torch tensor on CPU
    if isinstance(flow, np.ndarray):
        flow = torch.from_numpy(flow)
    elif not isinstance(flow, torch.Tensor):
        raise TypeError("flow must be a torch.Tensor or np.ndarray")

    flow = flow.detach().cpu()

    # Remove batch dim if present
    if flow.ndim == 4:
        flow = flow[0]  # [2, H, W]

    assert flow.shape[0] == 2, f"Expected flow shape [2, H, W], got {flow.shape}"

    dy, dx = flow[0], flow[1]  # NOTE: dy = vertical, dx = horizontal
    H, W = dy.shape

    # Subsample grid
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    X, Y = np.meshgrid(xs, ys)  # shapes [H_s, W_s]

    # Subsample flow vectors to match grid
    u = dx[::step, ::step].numpy()  # horizontal
    v = dy[::step, ::step].numpy()  # vertical

    # Optional: auto-scale arrows to avoid looking too tiny/too large
    mag = np.sqrt(u**2 + v**2)
    max_mag = np.max(mag) if mag.size > 0 else 1.0
    if max_mag > 0:
        # normalize to a reasonable range
        u_plot = u / max_mag
        v_plot = v / max_mag
        scale = 0.1  # or tune this
    else:
        u_plot, v_plot, scale = u, v, 1.0

    plt.figure(figsize=(8, 8))
    # If your flow is defined in image coordinates (y downwards),
    # using -v and invert_yaxis keeps directions visually consistent:
    plt.quiver(
        X, Y,
        -u_plot, -v_plot,            # negate to match image coord + invert_yaxis
        angles="xy",
        scale_units="xy",
        scale=scale,
        color="purple",
    )
    plt.gca().invert_yaxis()         # (0,0) at top-left like images
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.show()


def visualize_pair_data(pair):
    fixed_img = pair["fixed_image"]
    moving_img = pair["moving_image"]
    warped_img = pair["warped_image"]

    fixed_coor = pair["fixed_coor"]
    moving_coor = pair["moving_coor"]
    warped_coor = pair["warped_coor"]

    fixed_label = pair["fixed_label"]
    moving_label = pair["moving_label"]

    flow = pair["flow"]
    # visualize_flow(flow,step=5)
    # visualize_flow(dense_flow,step=50)

    f, a = plt.subplots(2, 3, figsize=(12, 8))
    # plt.imshow(image_ovelayed)
    a[0, 0].imshow(0.5 * fixed_img + 0.5 * moving_img)
    a[0, 1].imshow(moving_img)
    a[0, 1].scatter(moving_coor[:, 0], moving_coor[:, 1], c=moving_label, cmap='tab10', s=5)
    a[0, 2].imshow(warped_img)
    a[0, 2].scatter(warped_coor[:, 0], warped_coor[:, 1], c=moving_label, cmap='tab10', s=5)

    a[1, 0].imshow(0.5 * fixed_img + 0.5 * warped_img)
    a[1, 1].scatter(fixed_coor[:, 0], fixed_coor[:, 1], c=fixed_label, cmap='tab10', s=5)
    a[1, 1].scatter(moving_coor[:, 0], moving_coor[:, 1], c=moving_label, cmap='tab10', s=5)
    # a[1, 1].scatter(fixed_coor[:, 0], fixed_coor[:, 1],  s=10)
    # a[1, 1].scatter(moving_coor[:, 0], moving_coor[:, 1],  s=10)
    a[1, 1].invert_yaxis()
    a[1, 1].axis('equal')

    a[1, 2].scatter(fixed_coor[:, 0], fixed_coor[:, 1], c=fixed_label, cmap='tab10', s=5)
    a[1, 2].scatter(warped_coor[:, 0], warped_coor[:, 1], c=moving_label, cmap='tab10', s=5)
    # a[1, 2].scatter(fixed_coor[:, 0], fixed_coor[:, 1], s=10)
    # a[1, 2].scatter(warped_coor[:, 0], warped_coor[:, 1], s=10)
    a[1, 2].invert_yaxis()
    a[1, 2].axis('equal')
    plt.show()


def visualize_unified_coords(
    group_dir: str,
    ref_coords: Optional[np.ndarray] = None,
    point_size: float = 5.0,
):
    """
    Overlay all unified coordinates in the reference frame to visually
    inspect registration quality.

    Parameters
    ----------
    group_dir : str
        Path to the group folder, e.g. ".../output/group_2".
        The function expects:
            group_dir/unified/meta.json
            group_dir/unified/coords_slice{sid}_to_{ref_id}.npy
        created previously by unify_group_to_reference(...).

    ref_coords : np.ndarray or None
        Optional coordinates of the reference slice in its own frame,
        shape (N_ref, 2). If provided, they will be plotted as well.
        Typically you would obtain this from the corresponding h5ad:
            ref_coords = adata_ref.obsm["spatial_image_coor"]
        or whatever original coordinate key you use.

    point_size : float
        Marker size for scatter plot.

    Returns
    -------
    None
    """
    group_dir = Path(group_dir)
    unified_dir = group_dir / "unified"
    meta_path = unified_dir / "meta.json"

    if not unified_dir.is_dir():
        raise NotADirectoryError(f"[ERROR] Unified dir not found: {unified_dir}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"[ERROR] meta.json not found in {unified_dir}")

    # Load meta information
    with open(meta_path, "r") as f:
        meta = json.load(f)

    ref_id = int(meta["reference_slice"])
    sample_ids = [int(s) for s in meta.get("sample_ids", [])]
    if not sample_ids:
        # Fallback: infer from existing coord files
        sample_ids = []
        for npy in unified_dir.glob("coords_slice*_to_*.npy"):
            stem = npy.stem  # coords_slice{sid}_to_{ref}
            parts = stem.split("_")
            sid_str = parts[1].replace("slice", "")  # "slice0" -> "0"
            sample_ids.append(int(sid_str))
        sample_ids = sorted(set(sample_ids))

    print(f"[INFO] Visualizing unified coords in {group_dir.name}")
    print(f"       Reference slice: {ref_id}")
    print(f"       Sample IDs: {sample_ids}")

    # Color palette
    import itertools
    color_cycle = itertools.cycle(plt.cm.tab10.colors)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot reference slice if coords provided
    if ref_coords is not None:
        ref_coords = np.asarray(ref_coords, dtype=np.float32)
        ax.scatter(
            ref_coords[:, 0],
            ref_coords[:, 1],
            s=point_size,
            c="black",
            alpha=0.7,
            label=f"slice {ref_id} (ref)",
        )

    # Plot each non-reference slice in reference frame
    for sid in sample_ids:
        if sid == ref_id:
            continue

        coord_path = unified_dir / f"coords_slice{sid}_to_{ref_id}.npy"
        if not coord_path.is_file():
            print(f"[WARN] Missing unified coords for slice {sid}: {coord_path}, skipping.")
            continue

        coords = np.load(coord_path).astype(np.float32)
        color = next(color_cycle)

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=point_size,
            c=[color],
            alpha=0.6,
            label=f"slice {sid} → {ref_id}",
        )

    ax.set_aspect("equal")
    ax.invert_yaxis()  # match image coordinate convention
    ax.axis("off")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.set_title(f"{group_dir.name} – unified coordinates in frame {ref_id}")
    plt.tight_layout()
    plt.show()


def visualize_unified_coords_with_domains(
    root_dir: str,
    group_id: int,
    sample_ids: List[int],
    unified_subdir: str = "unified",
    label_npz_suffix: str = "validation_basesize.npz",
    point_size: float = 5.0,
):
    """
    Overlay unified coordinates in the AXIS reference frame, colored by domain labels.

    Coordinates are read from:
        root_dir/output/group_{group_id}/{unified_subdir}/coords_slice{sid}_to_{ref_id}.npy

    Domain labels are read from:
        root_dir/network_input/{group_token}_{sid}_{label_npz_suffix}
        where `group_token` is the numeric part of group_id (e.g. "0" for group_0)

    Assumptions
    -----------
    - unify_group_to_reference(...) has already been run with the same
      `unified_subdir`, and (optionally) `ref_coords`, so that
      coords_slice{ref_id}_to_{ref_id}.npy exists.
    - The npz label file for each slice has:
        "label": (N_spots,)

    Parameters
    ----------
    root_dir : str
        Base directory containing `output/` and `network_input/`.
    group_id : int
        Group identifier; folders on disk are "group_{group_id}".
    sample_ids : list[int]
        Slice/sample IDs used in this group, e.g. [0, 1, 2, 3].
    unified_subdir : str
        Subfolder under `output/group_{group_id}` where unified coords are stored.
        Example: "unified", "unified_axis0", ...
    label_npz_suffix : str
        Suffix for label npz files in `network_input/`.
        Default: "validation_basesize.npz"
        File pattern: f"{group_token}_{sid}_{label_npz_suffix}"
    point_size : float
        Marker size in the scatter plot.

    Returns
    -------
    None (shows a matplotlib figure)
    """

    root_dir = Path(root_dir)

    # Figure out group folder and numeric token
    group_folder = f"group_{group_id}"
    group_token = str(group_id)  # for network_input filenames

    group_dir = root_dir / "pairwise_output" / group_folder
    unified_dir = group_dir / unified_subdir
    meta_path = unified_dir / "meta.json"

    if not unified_dir.is_dir():
        raise NotADirectoryError(f"[ERROR] Unified dir not found: {unified_dir}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"[ERROR] meta.json not found in {unified_dir}")

    # --- Load meta.json to get reference slice --- #
    with open(meta_path, "r") as f:
        meta = json.load(f)

    ref_id = int(meta["reference_slice"])
    meta_sample_ids = [int(s) for s in meta.get("sample_ids", [])]

    if meta_sample_ids:
        missing = set(sample_ids) - set(meta_sample_ids)
        if missing:
            print(f"[WARN] Some sample_ids not in meta.sample_ids: {missing}")

    print(f"[INFO] Visualizing domains for group {group_id}")
    print(f"       Reference slice: {ref_id}")
    print(f"       Sample IDs: {sample_ids}")
    print(f"       Unified subdir: {unified_subdir}")

    all_coords = []
    all_labels = []
    all_slice_ids = []

    for sid in sample_ids:
        sid = int(sid)

        # --- Load unified coordinates for this slice (in reference frame) --- #
        coord_path = unified_dir / f"coords_slice{sid}_to_ref.npy"
        if not coord_path.is_file():
            print(f"[WARN] Unified coords not found for slice {sid}: {coord_path}, skipping slice.")
            continue

        coords_ref_frame = np.load(coord_path).astype(np.float32)

        # --- Load domain labels from network_input --- #
        npz_path = root_dir / "network_input" / f"{group_token}_{sid}_{label_npz_suffix}"
        if not npz_path.is_file():
            print(f"[WARN] Label npz not found for slice {sid}: {npz_path}, skipping slice.")
            continue

        data = np.load(npz_path)
        if "label" not in data:
            print(f"[WARN] 'label' not found in {npz_path}, skipping slice {sid}.")
            continue

        labels = np.asarray(data["label"]).ravel()

        # Sanity check / trim to min length
        if coords_ref_frame.shape[0] != labels.shape[0]:
            print(
                f"[WARN] coords ({coords_ref_frame.shape[0]}) and labels ({labels.shape[0]}) "
                f"length mismatch for slice {sid}; truncating to min length."
            )
            n = min(coords_ref_frame.shape[0], labels.shape[0])
            coords_ref_frame = coords_ref_frame[:n]
            labels = labels[:n]

        all_coords.append(coords_ref_frame)
        all_labels.append(labels)
        all_slice_ids.append(np.full(labels.shape, sid, dtype=int))

    if not all_coords:
        print("[ERROR] No coords+labels collected; nothing to visualize.")
        return

    all_coords = np.vstack(all_coords)          # (N_total, 2)
    all_labels = np.concatenate(all_labels)     # (N_total,)
    all_slice_ids = np.concatenate(all_slice_ids)  # (N_total,)

    # --- Prepare colormap by domain label --- #
    unique_labels = np.unique(all_labels)
    has_bg = -1 in unique_labels
    if has_bg:
        fg_labels = [lab for lab in unique_labels if lab != -1]
    else:
        fg_labels = list(unique_labels)

    n_fg = len(fg_labels)
    cmap = plt.get_cmap("tab20", max(n_fg, 1))

    label_to_color = {}
    for i, lab in enumerate(sorted(fg_labels)):
        label_to_color[lab] = cmap(i)
    if has_bg:
        label_to_color[-1] = (0.8, 0.8, 0.8, 0.25)  # background

    # --- Plot --- #
    fig, ax = plt.subplots(figsize=(6, 6))

    for lab in sorted(unique_labels):
        mask = (all_labels == lab)
        coords_lab = all_coords[mask]
        if coords_lab.size == 0:
            continue

        color = label_to_color.get(lab, (0.5, 0.5, 0.5, 0.5))
        lab_name = f"domain {lab}" if lab != -1 else "background"

        ax.scatter(
            coords_lab[:, 0],
            coords_lab[:, 1],
            s=point_size,
            c=[color],
            alpha=color[3] if len(color) == 4 else 0.6,
            label=lab_name,
        )

    ax.set_aspect("equal")
    # ax.invert_xaxis()
    ax.axis("off")
    ax.legend(loc="upper right", frameon=True, fontsize=7, title="Domains")
    ax.set_title(
        f"group_{group_id} – domains in AXIS ref frame (ref slice {ref_id})\n"
        f"unified_subdir='{unified_subdir}'"
    )
    plt.tight_layout()
    plt.show()
