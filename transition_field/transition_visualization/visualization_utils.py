
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd


def robust_vmax(vals_list, q=0.995):
    if len(vals_list) == 0:
        return 1.0
    x = np.concatenate(vals_list, axis=0)
    if x.size == 0:
        return 1.0
    return max(float(np.quantile(x, q)), 1e-6)

def parse_flow_path(root, group_id, fixed_id, moving_id, pairwise_subdir="pairwise_output"):
    group_dir = os.path.join(root, pairwise_subdir, f"group_{group_id}")
    return os.path.join(group_dir, f"{fixed_id}_{moving_id}_flow.npy")


def draw_slice_plane(
    ax,
    x_plane,
    y_min,
    y_max,
    z_min,
    z_max,
    plane_color="lightgray",
    plane_alpha=0.045,
    edge_color="gray",
    edge_alpha=0.22,
    n_grid=2,
):
    yy = np.linspace(y_min, y_max, n_grid)
    zz = np.linspace(z_min, z_max, n_grid)
    YY, ZZ = np.meshgrid(yy, zz)
    XX = np.full_like(YY, x_plane, dtype=np.float32)

    ax.plot_surface(
        XX, YY, ZZ,
        color=plane_color,
        alpha=plane_alpha,
        linewidth=0,
        shade=False,
        antialiased=False,
        zorder=0,
    )

    # rectangle outline
    corners = np.array([
        [x_plane, y_min, z_min],
        [x_plane, y_max, z_min],
        [x_plane, y_max, z_max],
        [x_plane, y_min, z_max],
        [x_plane, y_min, z_min],
    ], dtype=np.float32)

    ax.plot(
        corners[:, 0], corners[:, 1], corners[:, 2],
        color=edge_color,
        alpha=edge_alpha,
        linewidth=0.7,
    )


def draw_trajectory_segments_pairwise(
    ax,
    sample_ids,
    plane_pos,
    trajectories,
    color="crimson",
    alpha=0.35,
    linewidth=0.8,
):
    """
    trajectories[k] is [N_seed, 2] for slice k, NaN if lost.
    Draw pairwise line segments with ax.plot(), which usually renders
    more reliably than one big Line3DCollection in mplot3d.
    """
    for i in range(len(sample_ids) - 1):
        sid0 = sample_ids[i]
        sid1 = sample_ids[i + 1]
        x0 = plane_pos[sid0]
        x1 = plane_pos[sid1]

        p0 = trajectories[i]
        p1 = trajectories[i + 1]

        valid = np.all(np.isfinite(p0), axis=1) & np.all(np.isfinite(p1), axis=1)
        idx = np.where(valid)[0]

        for j in idx:
            ax.plot(
                [x0, x1],
                [p0[j, 0], p1[j, 0]],
                [p0[j, 1], p1[j, 1]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )


def make_rgba_with_value_alpha(
    vals,
    norm,
    cmap_name="viridis",
    alpha_min=0.03,
    alpha_max=0.90,
    gamma=1.6,
):
    frac = np.clip(norm(vals), 0.0, 1.0)
    cmap = get_cmap(cmap_name)
    rgba = cmap(frac)
    alpha = alpha_min + (frac ** gamma) * (alpha_max - alpha_min)
    rgba[:, 3] = np.clip(alpha, 0.0, 1.0)
    return rgba

def ensure_flow_hw2(flow):
    """
    Accept [H, W, 2] or [2, H, W], return [H, W, 2].
    """
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3:
        raise ValueError(f"flow must be 3D, got shape {flow.shape}")

    if flow.shape[-1] == 2:
        return flow
    if flow.shape[0] == 2:
        return np.moveaxis(flow, 0, -1)

    raise ValueError(f"Cannot interpret flow shape {flow.shape}")
