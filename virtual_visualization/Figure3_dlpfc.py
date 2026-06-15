
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from visualization_utils import *
import seaborn as sns
# --------------------
# Config
# --------------------
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
import glob
import os
import scanpy as sc
import matplotlib.pyplot as plt
from anndata import AnnData
from scipy.spatial import cKDTree
from sklearn.decomposition import NMF


import os
import glob
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection













# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.cm import ScalarMappable, get_cmap
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#
#
# # ============================================================
# # Basic helpers
# # ============================================================
# def _rgba_from_vals(vals, norm, cmap_name, alpha):
#     vals = np.asarray(vals, dtype=np.float32)
#     rgba = np.asarray(get_cmap(cmap_name)(norm(vals)), dtype=np.float32)
#     if rgba.ndim == 1:
#         rgba = rgba[None, :]
#     if rgba.shape[1] == 3:
#         alpha_col = np.full((rgba.shape[0], 1), float(alpha), dtype=np.float32)
#         rgba = np.concatenate([rgba, alpha_col], axis=1)
#     else:
#         rgba[:, 3] = float(alpha)
#     return rgba
#
#
# def _collect_xyz_limits(real_panels, virtual_panels, depth_scale, pad_ratio=0.04):
#     """
#     3D convention used here:
#       X axis = depth = z_um * depth_scale
#       Y axis = coords[:, 0]
#       Z axis = coords[:, 1]
#     """
#     all_panels = list(real_panels) + list(virtual_panels)
#
#     xs = []
#     ys = []
#     zs = []
#     for p in all_panels:
#         coords = np.asarray(p["coords"], dtype=np.float32)
#         xs.append(np.full(coords.shape[0], float(p["z"]) * float(depth_scale), dtype=np.float32))
#         ys.append(coords[:, 0])
#         zs.append(coords[:, 1])
#
#     x = np.concatenate(xs)
#     y = np.concatenate(ys)
#     z = np.concatenate(zs)
#
#     def _pad(a):
#         amin = float(np.min(a))
#         amax = float(np.max(a))
#         span = max(amax - amin, 1e-6)
#         pad = span * pad_ratio
#         return amin - pad, amax + pad
#
#     return _pad(x), _pad(y), _pad(z)
#
#
# # ============================================================
# # New cut mode: vertical plane spinning around a vertical line
# # ============================================================
# def split_points_by_vertical_axis_spin(
#     coords,
#     z_um,
#     depth_scale,
#     cut_anchor_depth,
#     cut_anchor_coord0,
#     cut_angle_deg,
#     slab_half_width=8.0,
#     keep_positive_side=True,
# ):
#     """
#     Vertical revolving-door cut.
#
#     3D convention:
#       x = depth = z_um * depth_scale
#       y = coords[:, 0]
#       z = coords[:, 1]  (vertical)
#
#     The plane contains the vertical z direction and rotates in the
#     ground plane spanned by (depth, coords[:,0]).
#     """
#     coords = np.asarray(coords, dtype=np.float32)
#
#     x = np.full(coords.shape[0], float(z_um) * float(depth_scale), dtype=np.float32)
#     y = coords[:, 0]
#
#     dx = x - float(cut_anchor_depth)
#     dy = y - float(cut_anchor_coord0)
#
#     phi = np.deg2rad(cut_angle_deg)
#
#     # Horizontal in-plane direction
#     u = np.array([np.cos(phi), np.sin(phi)], dtype=np.float32)
#
#     # Horizontal normal to the cut plane
#     n = np.array([-u[1], u[0]], dtype=np.float32)
#
#     signed = dx * n[0] + dy * n[1]
#
#     mask_plane = np.abs(signed) <= slab_half_width
#
#     if keep_positive_side:
#         mask_keep = signed > slab_half_width
#         mask_remove = signed < -slab_half_width
#     else:
#         mask_keep = signed < -slab_half_width
#         mask_remove = signed > slab_half_width
#
#     return mask_keep, mask_plane, mask_remove
#
#
# def scatter_resection_slice_vertical_spin(
#     ax,
#     coords,
#     vals,
#     z_um,
#     norm,
#     cmap_name,
#     depth_scale,
#     cut_anchor_depth,
#     cut_anchor_coord0,
#     cut_angle_deg,
#     slab_half_width=8.0,
#     keep_positive_side=True,
#     s_volume=5,
#     s_plane=11,
#     alpha_volume=0.12,
#     alpha_plane=0.95,
#     show_removed_side=False,
#     alpha_removed=0.03,
# ):
#     """
#     Render a slice with revolving-door style vertical resection plane.
#     """
#     coords = np.asarray(coords, dtype=np.float32)
#     vals = np.asarray(vals, dtype=np.float32)
#
#     mask_keep, mask_plane, mask_remove = split_points_by_vertical_axis_spin(
#         coords=coords,
#         z_um=z_um,
#         depth_scale=depth_scale,
#         cut_anchor_depth=cut_anchor_depth,
#         cut_anchor_coord0=cut_anchor_coord0,
#         cut_angle_deg=cut_angle_deg,
#         slab_half_width=slab_half_width,
#         keep_positive_side=keep_positive_side,
#     )
#
#     x = np.full(coords.shape[0], float(z_um) * float(depth_scale), dtype=np.float32)
#     y = coords[:, 0]
#     z = coords[:, 1]
#
#     if show_removed_side and np.any(mask_remove):
#         rgba = _rgba_from_vals(vals[mask_remove], norm, cmap_name, alpha_removed)
#         ax.scatter(
#             x[mask_remove], y[mask_remove], z[mask_remove],
#             s=s_volume, c=rgba, linewidths=0, depthshade=False, rasterized=True
#         )
#
#     if np.any(mask_keep):
#         rgba = _rgba_from_vals(vals[mask_keep], norm, cmap_name, alpha_volume)
#         ax.scatter(
#             x[mask_keep], y[mask_keep], z[mask_keep],
#             s=s_volume, c=rgba, linewidths=0, depthshade=False, rasterized=True
#         )
#
#     if np.any(mask_plane):
#         rgba = _rgba_from_vals(vals[mask_plane], norm, cmap_name, alpha_plane)
#         ax.scatter(
#             x[mask_plane], y[mask_plane], z[mask_plane],
#             s=s_plane, c=rgba, linewidths=0, depthshade=False, rasterized=True
#         )
#
#
# def draw_cut_plane_frame_vertical_spin(
#     ax,
#     cut_anchor_depth,
#     cut_anchor_coord0,
#     cut_angle_deg,
#     depth_lim,
#     coord0_lim,
#     coord1_lim,
#     frame_color="crimson",
#     frame_width=2.2,
#     frame_alpha=0.95,
#     face_alpha=0.05,
#     draw_face=True,
#     draw_label=True,
#     label="Cut plane",
#     label_fontsize=12,
# ):
#     """
#     Draw the vertical spinning cut plane as a rectangular frame.
#     """
#     x_min, x_max = depth_lim
#     y_min, y_max = coord0_lim
#     z_min, z_max = coord1_lim
#
#     phi = np.deg2rad(cut_angle_deg)
#
#     # In-plane horizontal direction in (depth, coord0)
#     u = np.array([np.cos(phi), np.sin(phi), 0.0], dtype=np.float32)
#     # Vertical direction
#     v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
#
#     center = np.array([
#         float(cut_anchor_depth),
#         float(cut_anchor_coord0),
#         0.5 * (float(z_min) + float(z_max)),
#     ], dtype=np.float32)
#
#     half_h = 0.50 * np.hypot(float(x_max) - float(x_min), float(y_max) - float(y_min))
#     half_v = 0.55 * (float(z_max) - float(z_min))
#
#     A = center - half_h * u - half_v * v
#     B = center + half_h * u - half_v * v
#     C = center + half_h * u + half_v * v
#     D = center - half_h * u + half_v * v
#
#     corners = np.vstack([A, B, C, D, A])
#
#     ax.plot(
#         corners[:, 0], corners[:, 1], corners[:, 2],
#         color=frame_color, linewidth=frame_width, alpha=frame_alpha, zorder=10
#     )
#
#     if draw_face:
#         poly = Poly3DCollection([[A, B, C, D]])
#         poly.set_facecolor(frame_color)
#         poly.set_alpha(face_alpha)
#         poly.set_edgecolor("none")
#         ax.add_collection3d(poly)
#
#     if draw_label:
#         ax.text(
#             center[0], center[1], z_max,
#             label,
#             color=frame_color,
#             fontsize=label_fontsize,
#             ha="center",
#             va="bottom",
#             zorder=11,
#         )
#
#
# # ============================================================
# # Old mode support
# # ============================================================
# # This mode assumes you already have your old function:
# #   scatter_resection_slice_horizontal_depth(...)
# # and optionally a frame-drawing helper.
# #
# # If you no longer need the old mode, you can remove this branch.
#
#
# # ============================================================
# # Main rewritten plotting function
# # ============================================================
# def plot_insilico_resectioning_fancy(
#     GENE,
#     CMAP="tab10",
#     DEPTH_SCALE=1.0,
#     CUT_STYLE="vertical_spin",   # "vertical_spin" or "depth_extruded"
#     CUT_ANCHOR_2D=(0.0, 0.0),    # used by old mode if needed
#     CUT_ANCHOR_DEPTH=None,       # used by vertical_spin
#     CUT_ANCHOR_COORD0=None,      # used by vertical_spin
#     CUT_ANGLE_DEG=35.0,
#     CUT_SLAB_HALF_WIDTH=10.0,
#     CUT_KEEP_POSITIVE_SIDE=True,
#     SHOW_REMOVED_SIDE=False,
#     REAL_S_VOLUME=5,
#     REAL_S_PLANE=10,
#     REAL_ALPHA_VOLUME=0.10,
#     REAL_ALPHA_PLANE=0.95,
#     REAL_ALPHA_REMOVED=0.02,
#     VIRTUAL_S_VOLUME=4,
#     VIRTUAL_S_PLANE=9,
#     VIRTUAL_ALPHA_VOLUME=0.08,
#     VIRTUAL_ALPHA_PLANE=0.90,
#     VIRTUAL_ALPHA_REMOVED=0.02,
#     DRAW_CUT_FRAME=True,
#     FRAME_COLOR="crimson",
#     FRAME_LABEL=None,
#     SHOW_COLORBAR=False,
#     COLORBAR_LABEL=None,
#     TITLE=None,
#     FIGSIZE=(8, 8),
#     ELEV=18,
#     AZIM=-58,
#     DEPTH_LIM=None,
#     BOX_ASPECT=None,
#     HIDE_AXES=False,
#     save_path=None,
# ):
#     """
#     Full rewritten resection plot.
#
#     Expected panel format for each entry in real_panels / virtual_panels:
#       {
#         "coords": ndarray of shape (N, 2),
#         "vals":   ndarray of shape (N,),
#         "z":      scalar depth in um,
#       }
#
#     3D plotting convention:
#       ax.scatter(depth, coords[:,0], coords[:,1])
#       => x-axis: depth
#          y-axis: in-slice x
#          z-axis: in-slice y (vertical in your figure)
#     """
#
#     # --------------------
#     # Load real slices
#     # --------------------
#     real_panels = []
#     real_vals_for_scale = []
#
#     for sid in ORIG_SAMPLES:
#         z = ORIG_Z_UM[sid]
#         ad = sc.read_h5ad(os.path.join(H5AD_DIR, f"{GROUP_ID}_{sid}.h5ad"))
#         coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)
#
#         try:
#             vals = extract_gene_vals(ad, GENE)
#         except ValueError:
#             vals, _ = extract_obs_vals(ad, GENE)
#
#         real_panels.append({
#             "title": f"Real ({z}um)",
#             "coords": coords,
#             "vals": vals,
#             "z": float(z),
#         })
#         real_vals_for_scale.append(vals)
#
#     # --------------------
#     # Load virtual slices
#     # --------------------
#     gen_pattern = os.path.join(GEN_DIR, "*.h5ad")
#     gen_files = sorted(glob.glob(gen_pattern), key=parse_z_from_filename)
#
#     virtual_panels = []
#     virtual_vals_for_scale = []
#
#     for f in gen_files:
#         z = int(parse_z_from_filename(f))
#
#         ad = sc.read_h5ad(f)
#         if "valid" in ad.obs.columns:
#             ad = ad[ad.obs["valid"]].copy()
#
#         coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)
#
#         if GENE is None:
#             vals = np.full(coords.shape[0], 1.0, dtype=np.float32)
#         else:
#             try:
#                 vals = extract_gene_vals(ad, GENE)
#             except ValueError:
#                 vals, _ = extract_obs_vals(ad, GENE)
#
#         virtual_panels.append({
#             "title": f"Virtual ({z}um)",
#             "coords": coords,
#             "vals": vals,
#             "z": float(z),
#         })
#         virtual_vals_for_scale.append(vals)
#
#     # --------------------
#     # Color scales
#     # --------------------
#     vmin_real, vmax_real = robust_vrange(
#         real_vals_for_scale,
#         1 - VMAX_PCT_REAL,
#         VMAX_PCT_REAL,
#     )
#     vmin_virtual, vmax_virtual = robust_vrange(
#         virtual_vals_for_scale,
#         1 - VMAX_PCT_VIRTUAL,
#         VMAX_PCT_VIRTUAL,
#     )
#
#     norm_real = Normalize(vmin=float(vmin_real), vmax=float(vmax_real))
#     norm_virtual = Normalize(vmin=float(vmin_virtual), vmax=float(vmax_virtual))
#
#     print(f"[SCALE] Real:    vmin={vmin_real:.4f}, vmax={vmax_real:.4f}")
#     print(f"[SCALE] Virtual: vmin={vmin_virtual:.4f}, vmax={vmax_virtual:.4f}")
#
#     if norm_virtual is None:
#         norm_virtual = norm_real
#
#     fig = plt.figure(figsize=FIGSIZE)
#     ax = fig.add_subplot(111, projection="3d")
#
#     # --------------------------------------------------------
#     # Global limits
#     # --------------------------------------------------------
#     depth_lim_auto, coord0_lim_auto, coord1_lim_auto = _collect_xyz_limits(
#         real_panels=real_panels,
#         virtual_panels=virtual_panels,
#         depth_scale=DEPTH_SCALE,
#         pad_ratio=0.04,
#     )
#
#     depth_lim = DEPTH_LIM if DEPTH_LIM is not None else depth_lim_auto
#     coord0_lim = X_LIM if X_LIM is not None else coord0_lim_auto
#     coord1_lim = Y_LIM if Y_LIM is not None else coord1_lim_auto
#
#     if CUT_ANCHOR_DEPTH is None:
#         CUT_ANCHOR_DEPTH = 0.5 * (depth_lim[0] + depth_lim[1])
#     if CUT_ANCHOR_COORD0 is None:
#         CUT_ANCHOR_COORD0 = 0.5 * (coord0_lim[0] + coord0_lim[1])
#
#     # --------------------------------------------------------
#     # Render real panels
#     # --------------------------------------------------------
#     if CUT_STYLE == "vertical_spin":
#         for p in real_panels:
#             scatter_resection_slice_vertical_spin(
#                 ax=ax,
#                 coords=p["coords"],
#                 vals=p["vals"],
#                 z_um=p["z"],
#                 norm=norm_real,
#                 cmap_name=CMAP,
#                 depth_scale=DEPTH_SCALE,
#                 cut_anchor_depth=CUT_ANCHOR_DEPTH,
#                 cut_anchor_coord0=CUT_ANCHOR_COORD0,
#                 cut_angle_deg=CUT_ANGLE_DEG,
#                 slab_half_width=CUT_SLAB_HALF_WIDTH,
#                 keep_positive_side=CUT_KEEP_POSITIVE_SIDE,
#                 s_volume=REAL_S_VOLUME,
#                 s_plane=REAL_S_PLANE,
#                 alpha_volume=REAL_ALPHA_VOLUME,
#                 alpha_plane=REAL_ALPHA_PLANE,
#                 show_removed_side=SHOW_REMOVED_SIDE,
#                 alpha_removed=REAL_ALPHA_REMOVED,
#             )
#
#         for p in virtual_panels:
#             scatter_resection_slice_vertical_spin(
#                 ax=ax,
#                 coords=p["coords"],
#                 vals=p["vals"],
#                 z_um=p["z"],
#                 norm=norm_virtual,
#                 cmap_name=CMAP,
#                 depth_scale=DEPTH_SCALE,
#                 cut_anchor_depth=CUT_ANCHOR_DEPTH,
#                 cut_anchor_coord0=CUT_ANCHOR_COORD0,
#                 cut_angle_deg=CUT_ANGLE_DEG,
#                 slab_half_width=CUT_SLAB_HALF_WIDTH,
#                 keep_positive_side=CUT_KEEP_POSITIVE_SIDE,
#                 s_volume=VIRTUAL_S_VOLUME,
#                 s_plane=VIRTUAL_S_PLANE,
#                 alpha_volume=VIRTUAL_ALPHA_VOLUME,
#                 alpha_plane=VIRTUAL_ALPHA_PLANE,
#                 show_removed_side=SHOW_REMOVED_SIDE,
#                 alpha_removed=VIRTUAL_ALPHA_REMOVED,
#             )
#
#     elif CUT_STYLE == "depth_extruded":
#         # This branch uses your existing old helper
#         if "scatter_resection_slice_horizontal_depth" not in globals():
#             raise RuntimeError(
#                 "CUT_STYLE='depth_extruded' requires your existing "
#                 "scatter_resection_slice_horizontal_depth(...) function."
#             )
#
#         for p in real_panels:
#             scatter_resection_slice_horizontal_depth(
#                 ax=ax,
#                 coords=p["coords"],
#                 vals=p["vals"],
#                 z_um=p["z"],
#                 norm=norm_real,
#                 cmap_name=CMAP,
#                 depth_scale=DEPTH_SCALE,
#                 cut_anchor_2d=CUT_ANCHOR_2D,
#                 cut_angle_deg=CUT_ANGLE_DEG,
#                 slab_half_width=CUT_SLAB_HALF_WIDTH,
#                 keep_positive_side=CUT_KEEP_POSITIVE_SIDE,
#                 s_volume=REAL_S_VOLUME,
#                 s_plane=REAL_S_PLANE,
#                 alpha_volume=REAL_ALPHA_VOLUME,
#                 alpha_plane=REAL_ALPHA_PLANE,
#                 show_removed_side=SHOW_REMOVED_SIDE,
#                 alpha_removed=REAL_ALPHA_REMOVED,
#             )
#
#         for p in virtual_panels:
#             scatter_resection_slice_horizontal_depth(
#                 ax=ax,
#                 coords=p["coords"],
#                 vals=p["vals"],
#                 z_um=p["z"],
#                 norm=norm_virtual,
#                 cmap_name=CMAP,
#                 depth_scale=DEPTH_SCALE,
#                 cut_anchor_2d=CUT_ANCHOR_2D,
#                 cut_angle_deg=CUT_ANGLE_DEG,
#                 slab_half_width=CUT_SLAB_HALF_WIDTH,
#                 keep_positive_side=CUT_KEEP_POSITIVE_SIDE,
#                 s_volume=VIRTUAL_S_VOLUME,
#                 s_plane=VIRTUAL_S_PLANE,
#                 alpha_volume=VIRTUAL_ALPHA_VOLUME,
#                 alpha_plane=VIRTUAL_ALPHA_PLANE,
#                 show_removed_side=SHOW_REMOVED_SIDE,
#                 alpha_removed=VIRTUAL_ALPHA_REMOVED,
#             )
#     else:
#         raise ValueError(f"Unknown CUT_STYLE: {CUT_STYLE}")
#
#     # --------------------------------------------------------
#     # Draw cut frame
#     # --------------------------------------------------------
#     if DRAW_CUT_FRAME:
#         if FRAME_LABEL is None:
#             if CUT_STYLE == "vertical_spin":
#                 FRAME_LABEL = f"Cut plane ({CUT_ANGLE_DEG:.0f}°)"
#             else:
#                 FRAME_LABEL = "Cut plane"
#
#         if CUT_STYLE == "vertical_spin":
#             draw_cut_plane_frame_vertical_spin(
#                 ax=ax,
#                 cut_anchor_depth=CUT_ANCHOR_DEPTH,
#                 cut_anchor_coord0=CUT_ANCHOR_COORD0,
#                 cut_angle_deg=CUT_ANGLE_DEG,
#                 depth_lim=depth_lim,
#                 coord0_lim=coord0_lim,
#                 coord1_lim=coord1_lim,
#                 frame_color=FRAME_COLOR,
#                 frame_width=2.2,
#                 frame_alpha=0.95,
#                 face_alpha=0.05,
#                 draw_face=True,
#                 draw_label=True,
#                 label=FRAME_LABEL,
#                 label_fontsize=12,
#             )
#
#     # --------------------------------------------------------
#     # Axes / view
#     # --------------------------------------------------------
#     ax.set_xlim(depth_lim)
#     ax.set_ylim(coord0_lim)
#     ax.set_zlim(coord1_lim)
#
#     # --------------------
#     # Axes direction
#     # --------------------
#     if INVERT_Y:
#         ax.set_zlim(Y_LIM[::-1])
#     else:
#         ax.set_zlim(Y_LIM)
#
#     if INVERT_X:
#         ax.set_ylim(X_LIM[::-1])
#     else:
#         ax.set_ylim(X_LIM)
#
#     if BOX_ASPECT is None:
#         depth_span = max(depth_lim[1] - depth_lim[0], 1e-6)
#         coord0_span = max(coord0_lim[1] - coord0_lim[0], 1e-6)
#         coord1_span = max(coord1_lim[1] - coord1_lim[0], 1e-6)
#         ax.set_box_aspect((depth_span, coord0_span, coord1_span))
#     else:
#         ax.set_box_aspect(BOX_ASPECT)
#
#     ax.view_init(elev=ELEV, azim=AZIM)
#
#     if HIDE_AXES:
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_zticks([])
#         ax.set_xlabel("")
#         ax.set_ylabel("")
#         ax.set_zlabel("")
#     else:
#         ax.set_xlabel("Depth", fontsize=11)
#         ax.set_ylabel("In-slice x", fontsize=11)
#         ax.set_zlabel("In-slice y", fontsize=11)
#
#     # lighter panes/grid
#     try:
#         ax.xaxis.pane.set_alpha(0.06)
#         ax.yaxis.pane.set_alpha(0.06)
#         ax.zaxis.pane.set_alpha(0.06)
#     except Exception:
#         pass
#
#     if TITLE is not None:
#         ax.set_title(TITLE, fontsize=14, pad=16)
#
#     # --------------------------------------------------------
#     # Optional colorbar
#     # --------------------------------------------------------
#     if SHOW_COLORBAR:
#         sm = ScalarMappable(norm=norm_real, cmap=get_cmap(CMAP))
#         sm.set_array([])
#         cbar = fig.colorbar(sm, ax=ax, shrink=0.68, pad=0.04)
#         if COLORBAR_LABEL is not None:
#             cbar.set_label(COLORBAR_LABEL, fontsize=11)
#
#     fig.tight_layout()
#
#     if save_path is not None:
#         fig.savefig(save_path, dpi=300, bbox_inches="tight")
#         print(f"[OK] saved: {save_path}")
#     else:
#         plt.show()
#
#     return fig, ax


















# ============================================================
# Geometry helpers for angled resection plane
# ============================================================

def _normalize_vec(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector.")
    return v / n


def _cut_line_direction_from_angle(angle_deg):
    """
    Angle of the cutting line inside each 2D slice.
    Measured counterclockwise from +x_in_slice = coords[:,0].

    Returns unit tangent vector t and unit normal vector n.
    Signed distance to the cut is computed with n.
    """
    theta = np.deg2rad(angle_deg)
    t = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    t = _normalize_vec(t)
    n = np.array([-t[1], t[0]], dtype=np.float32)  # 90-degree rotation
    return t, n


def signed_distance_to_cut(coords_2d, anchor_2d, cut_angle_deg):
    """
    Signed distance from each 2D point to the cut line.
    Positive side = 'kept' side if keep_positive_side=True.
    """
    coords_2d = np.asarray(coords_2d, dtype=np.float32)
    anchor_2d = np.asarray(anchor_2d, dtype=np.float32)
    _, n = _cut_line_direction_from_angle(cut_angle_deg)
    return (coords_2d - anchor_2d[None, :]) @ n


def clip_infinite_line_to_box(anchor_2d, cut_angle_deg, xlim_2d, ylim_2d, eps=1e-8):
    """
    Intersect the 2D cut line with a rectangle box.

    Parameters
    ----------
    anchor_2d : (2,)
        A point that the line passes through.
    cut_angle_deg : float
        Orientation of the line within the slice plane.
    xlim_2d, ylim_2d : tuple
        Bounds in the same coordinate system as coords[:,0], coords[:,1].

    Returns
    -------
    p0, p1 : endpoints of the clipped segment, or (None, None) if no intersection.
    """
    p = np.asarray(anchor_2d, dtype=np.float32)
    t, _ = _cut_line_direction_from_angle(cut_angle_deg)

    xmin, xmax = sorted([float(xlim_2d[0]), float(xlim_2d[1])])
    ymin, ymax = sorted([float(ylim_2d[0]), float(ylim_2d[1])])

    pts = []

    # Intersect with x = xmin / xmax
    if abs(t[0]) > eps:
        for xb in [xmin, xmax]:
            u = (xb - p[0]) / t[0]
            y = p[1] + u * t[1]
            if ymin - 1e-6 <= y <= ymax + 1e-6:
                pts.append(np.array([xb, y], dtype=np.float32))

    # Intersect with y = ymin / ymax
    if abs(t[1]) > eps:
        for yb in [ymin, ymax]:
            u = (yb - p[1]) / t[1]
            x = p[0] + u * t[0]
            if xmin - 1e-6 <= x <= xmax + 1e-6:
                pts.append(np.array([x, yb], dtype=np.float32))

    if len(pts) < 2:
        return None, None

    # Deduplicate near-identical intersections
    uniq = []
    for q in pts:
        if not any(np.linalg.norm(q - qq) < 1e-5 for qq in uniq):
            uniq.append(q)

    if len(uniq) < 2:
        return None, None

    # choose the farthest pair
    best_i, best_j, best_d = 0, 1, -1
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            d = np.linalg.norm(uniq[i] - uniq[j])
            if d > best_d:
                best_d = d
                best_i, best_j = i, j

    return uniq[best_i], uniq[best_j]


def draw_cut_plane_frame(
    ax,
    anchor_2d,
    cut_angle_deg,
    depth_min,
    depth_max,
    xlim_2d,
    ylim_2d,
    frame_color="crimson",
    frame_width=2.2,
    frame_alpha=0.95,
    face_alpha=0.06,
    draw_face=True,
    draw_label=True,
    label="Cut plane",
    label_fontsize=12,
):
    """
    Draw the resection plane as a rectangular frame across the full depth axis.

    3D axes convention in your figure:
      depth -> x axis
      coords[:,0] -> y axis
      coords[:,1] -> z axis
    """
    p0, p1 = clip_infinite_line_to_box(anchor_2d, cut_angle_deg, xlim_2d, ylim_2d)
    if p0 is None or p1 is None:
        return None

    # rectangle corners in 3D
    A0 = [depth_min, p0[0], p0[1]]
    A1 = [depth_min, p1[0], p1[1]]
    B0 = [depth_max, p0[0], p0[1]]
    B1 = [depth_max, p1[0], p1[1]]

    # frame edges
    edges = [
        (A0, A1),
        (B0, B1),
        (A0, B0),
        (A1, B1),
    ]
    for u, v in edges:
        ax.plot(
            [u[0], v[0]],
            [u[1], v[1]],
            [u[2], v[2]],
            color=frame_color,
            linewidth=frame_width,
            alpha=frame_alpha,
            solid_capstyle="round",
            zorder=10,
        )

    # optional faint plane face
    if draw_face:
        poly = Poly3DCollection([[A0, A1, B1, B0]])
        poly.set_facecolor(frame_color)
        poly.set_alpha(face_alpha)
        poly.set_edgecolor("none")
        ax.add_collection3d(poly)

    if draw_label:
        mid = np.mean(np.array([A0, A1, B0, B1]), axis=0)
        ax.text(
            mid[0], mid[1], mid[2],
            label,
            color=frame_color,
            fontsize=label_fontsize,
            ha="center",
            va="bottom",
            zorder=11,
        )

    return (p0, p1)


# ============================================================
# Rendering helpers
# ============================================================

# def rgba_from_vals(vals, norm, cmap_name, alpha):
#     vals = np.asarray(vals, dtype=np.float32)
#     rgba = np.asarray(get_cmap(cmap_name)(norm(vals)), dtype=np.float32)
#     rgba[:, 3] = float(alpha)
#     return rgba
def rgba_from_vals(vals, norm, cmap_name, alpha):
    cmap = get_cmap(cmap_name)
    vals = np.asarray(vals, dtype=np.float32)

    if hasattr(cmap, "colors"):
        colors = np.asarray(cmap.colors, dtype=np.float32)
        idx = np.clip(vals.astype(int), 0, len(colors) - 1)
        base = colors[idx]

        if base.shape[1] == 3:
            rgba = np.concatenate(
                [base, np.ones((base.shape[0], 1), dtype=np.float32)],
                axis=1
            )
        else:
            rgba = base.copy()

        # for categorical labels, use constant alpha
        rgba[:, 3] = alpha
    else:
        vals = np.asarray(vals, dtype=np.float32)
        rgba = np.asarray(get_cmap(cmap_name)(norm(vals)), dtype=np.float32)
        rgba[:, 3] = float(alpha)
    return rgba

def scatter_resection_slice_horizontal_depth(
    ax,
    coords,
    vals,
    z_um,
    norm,
    cmap_name,
    depth_scale,
    cut_anchor_2d,
    cut_angle_deg,
    slab_half_width=8.0,
    keep_positive_side=True,
    s_volume=5,
    s_plane=11,
    alpha_volume=0.12,
    alpha_plane=0.95,
    show_removed_side=False,
    alpha_removed=0.03,
):
    """
    Render one slice with:
      - remaining volume on one side of the cut plane (transparent)
      - near-plane slab (solid / emphasized)
      - optional removed side (very faint)
    """
    coords = np.asarray(coords, dtype=np.float32)
    vals = np.asarray(vals, dtype=np.float32)

    d = signed_distance_to_cut(coords, cut_anchor_2d, cut_angle_deg)

    if keep_positive_side:
        keep_mask = d >= 0
        remove_mask = d < 0
    else:
        keep_mask = d <= 0
        remove_mask = d > 0

    plane_mask = keep_mask & (np.abs(d) <= slab_half_width)
    volume_mask = keep_mask & (~plane_mask)

    x_depth = np.full(coords.shape[0], float(z_um) * depth_scale, dtype=np.float32)
    y = coords[:, 0]
    z = coords[:, 1]

    # Removed side first (optional, almost invisible)
    if show_removed_side and np.any(remove_mask):
        rgba = rgba_from_vals(vals[remove_mask], norm, cmap_name, alpha_removed)
        ax.scatter(
            x_depth[remove_mask], y[remove_mask], z[remove_mask],
            s=s_volume, c=rgba, linewidths=0, depthshade=False, rasterized=True
        )

    # Kept volume
    if np.any(volume_mask):
        rgba = rgba_from_vals(vals[volume_mask], norm, cmap_name, alpha_volume)
        ax.scatter(
            x_depth[volume_mask], y[volume_mask], z[volume_mask],
            s=s_volume, c=rgba, linewidths=0, depthshade=False, rasterized=True
        )

    # Near cut plane, draw last and more opaque
    if np.any(plane_mask):
        rgba = rgba_from_vals(vals[plane_mask], norm, cmap_name, alpha_plane)
        ax.scatter(
            x_depth[plane_mask], y[plane_mask], z[plane_mask],
            s=s_plane, c=rgba, linewidths=0, depthshade=False, rasterized=True
        )


def choose_default_cut_anchor(real_panels, xlim_2d=None, ylim_2d=None):
    """
    Choose a sensible anchor point near the center of the volume in slice coordinates.
    """
    all_coords = np.vstack([p["coords"] for p in real_panels if p["coords"].shape[0] > 0])
    if xlim_2d is None:
        x0 = 0.5 * (float(np.min(all_coords[:, 0])) + float(np.max(all_coords[:, 0])))
    else:
        x0 = 0.5 * (float(xlim_2d[0]) + float(xlim_2d[1]))

    if ylim_2d is None:
        y0 = 0.5 * (float(np.min(all_coords[:, 1])) + float(np.max(all_coords[:, 1])))
    else:
        y0 = 0.5 * (float(ylim_2d[0]) + float(ylim_2d[1]))

    return np.array([x0, y0], dtype=np.float32)


# ============================================================
# Main upgraded plotting function
# ============================================================

def plot_insilico_resectioning_fancy(
    GENE,
    CMAP="viridis",
    REAL_FRAME_COLOR="black",
    VIRTUAL_FRAME_COLOR="0.35",
    REAL_FRAME_STYLE="-",
    VIRTUAL_FRAME_STYLE="--",
    REAL_FRAME_WIDTH=1.4,
    VIRTUAL_FRAME_WIDTH=1.4,
    SHOW_SLICE_FRAME=True,
    SHOW_SLICE_LABEL=True,
    VIRTUAL_UNIT_LABEL="um",

    # new resection parameters
    CUT_ANGLE_DEG=25.0,
    CUT_ANCHOR_2D=None,         # in slice coordinates (coords[:,0], coords[:,1]); None -> auto center
    CUT_KEEP_POSITIVE_SIDE=True,
    CUT_SLAB_HALF_WIDTH=10.0,

    # rendering controls
    SHOW_REMOVED_SIDE=False,
    REAL_ALPHA_VOLUME=0.10,
    REAL_ALPHA_PLANE=0.95,
    REAL_ALPHA_REMOVED=0.03,
    VIRTUAL_ALPHA_VOLUME=0.6,
    VIRTUAL_ALPHA_PLANE=0.70,
    VIRTUAL_ALPHA_REMOVED=0.02,
    REAL_S_VOLUME=None,
    REAL_S_PLANE=None,
    VIRTUAL_S_VOLUME=None,
    VIRTUAL_S_PLANE=None,

    # cut plane/frame styling
    CUT_FRAME_COLOR="crimson",
    CUT_FRAME_WIDTH=2.2,
    CUT_FRAME_ALPHA=0.95,
    CUT_FACE_ALPHA=0.05,
    CUT_LABEL="In silico resection plane",
    SHOW_CUT_LABEL=True,
):
    # --------------------
    # Load real slices
    # --------------------
    real_panels = []
    real_vals_for_scale = []

    for sid in ORIG_SAMPLES:
        z = ORIG_Z_UM[sid]
        ad = sc.read_h5ad(os.path.join(H5AD_DIR, f"{GROUP_ID}_{sid}.h5ad"))
        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)

        try:
            vals = extract_gene_vals(ad, GENE)
        except ValueError:
            vals, _ = extract_obs_vals(ad, GENE)

        real_panels.append({
            "title": f"Real ({z}um)",
            "coords": coords,
            "vals": vals,
            "z": float(z),
        })
        real_vals_for_scale.append(vals)

    # --------------------
    # Load virtual slices
    # --------------------
    gen_pattern = os.path.join(GEN_DIR, "*.h5ad")
    gen_files = sorted(glob.glob(gen_pattern), key=parse_z_from_filename)

    virtual_panels = []
    virtual_vals_for_scale = []

    for f in gen_files:
        z = int(parse_z_from_filename(f))

        ad = sc.read_h5ad(f)
        if "valid" in ad.obs.columns:
            ad = ad[ad.obs["valid"]].copy()

        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)

        if GENE is None:
            vals = np.full(coords.shape[0], 1.0, dtype=np.float32)
        else:
            try:
                vals = extract_gene_vals(ad, GENE)
            except ValueError:
                vals, _ = extract_obs_vals(ad, GENE)

        virtual_panels.append({
            "title": f"Virtual ({z}{VIRTUAL_UNIT_LABEL})",
            "coords": coords,
            "vals": vals,
            "z": float(z),
        })
        virtual_vals_for_scale.append(vals)

    # --------------------
    # Color scales
    # --------------------
    vmin_real, vmax_real = robust_vrange(
        real_vals_for_scale,
        1 - VMAX_PCT_REAL,
        VMAX_PCT_REAL,
    )
    vmin_virtual, vmax_virtual = robust_vrange(
        virtual_vals_for_scale,
        1 - VMAX_PCT_VIRTUAL,
        VMAX_PCT_VIRTUAL,
    )

    norm_real = Normalize(vmin=float(vmin_real), vmax=float(vmax_real))
    norm_virtual = Normalize(vmin=float(vmin_virtual), vmax=float(vmax_virtual))

    print(f"[SCALE] Real:    vmin={vmin_real:.4f}, vmax={vmax_real:.4f}")
    print(f"[SCALE] Virtual: vmin={vmin_virtual:.4f}, vmax={vmax_virtual:.4f}")

    # --------------------
    # Figure / axes
    # --------------------
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    all_depths = [p["z"] * DEPTH_SCALE for p in real_panels + virtual_panels]
    depth_min = min(all_depths)
    depth_max = max(all_depths)
    x_pad = 0.02 * max(depth_max - depth_min, 1.0)

    x_range = (depth_max + x_pad) - (depth_min - x_pad)
    y_range = abs(X_LIM[1] - X_LIM[0])
    z_range = abs(Y_LIM[1] - Y_LIM[0])

    depth_exaggeration = 1.5
    ax.set_box_aspect((x_range * depth_exaggeration, y_range, z_range))
    ax.set_position([0.03, 0.08, 0.80, 0.84])
    ax.set_proj_type("ortho")

    ax.set_xlim(depth_min - x_pad, depth_max + x_pad)
    ax.set_ylim(X_LIM)
    ax.set_zlim(Y_LIM)

    # --------------------
    # Choose cut anchor automatically if needed
    # --------------------
    if CUT_ANCHOR_2D is None:
        CUT_ANCHOR_2D = choose_default_cut_anchor(real_panels, xlim_2d=X_LIM, ylim_2d=Y_LIM)
    CUT_ANCHOR_2D = np.asarray(CUT_ANCHOR_2D, dtype=np.float32)


    # default sizes fall back to your existing globals
    if REAL_S_VOLUME is None:
        REAL_S_VOLUME = max(1, int(round(S_REAL * 0.75)))
    if REAL_S_PLANE is None:
        REAL_S_PLANE = max(REAL_S_VOLUME + 1, int(round(S_REAL * 1.35)))

    if VIRTUAL_S_VOLUME is None:
        VIRTUAL_S_VOLUME = max(1, int(round(S_VIRTUAL * 0.75)))
    if VIRTUAL_S_PLANE is None:
        VIRTUAL_S_PLANE = max(VIRTUAL_S_VOLUME + 1, int(round(S_VIRTUAL * 1.35)))

    # --------------------
    # Scatter real slices with resection rendering
    # --------------------
    for p in real_panels:
        scatter_resection_slice_horizontal_depth(
            ax=ax,
            coords=p["coords"],
            vals=p["vals"],
            z_um=p["z"],
            norm=norm_real,
            cmap_name=CMAP,
            depth_scale=DEPTH_SCALE,
            cut_anchor_2d=CUT_ANCHOR_2D,
            cut_angle_deg=CUT_ANGLE_DEG,
            slab_half_width=CUT_SLAB_HALF_WIDTH,
            keep_positive_side=CUT_KEEP_POSITIVE_SIDE,
            s_volume=REAL_S_VOLUME,
            s_plane=REAL_S_PLANE,
            alpha_volume=REAL_ALPHA_VOLUME,
            alpha_plane=REAL_ALPHA_PLANE,
            show_removed_side=SHOW_REMOVED_SIDE,
            alpha_removed=REAL_ALPHA_REMOVED,
        )

    # --------------------
    # Scatter virtual slices with resection rendering
    # --------------------
    for p in virtual_panels:
        scatter_resection_slice_horizontal_depth(
            ax=ax,
            coords=p["coords"],
            vals=p["vals"],
            z_um=p["z"],
            norm=norm_virtual,
            cmap_name=CMAP,
            depth_scale=DEPTH_SCALE,
            cut_anchor_2d=CUT_ANCHOR_2D,
            cut_angle_deg=CUT_ANGLE_DEG,
            slab_half_width=CUT_SLAB_HALF_WIDTH,
            keep_positive_side=CUT_KEEP_POSITIVE_SIDE,
            s_volume=VIRTUAL_S_VOLUME,
            s_plane=VIRTUAL_S_PLANE,
            alpha_volume=VIRTUAL_ALPHA_VOLUME,
            alpha_plane=VIRTUAL_ALPHA_PLANE,
            show_removed_side=SHOW_REMOVED_SIDE,
            alpha_removed=VIRTUAL_ALPHA_REMOVED,
        )

    # --------------------
    # Draw slice frames
    # --------------------
    if SHOW_SLICE_FRAME:
        for p in real_panels:
            draw_slice_plane_frame(
                ax=ax,
                z_um=p["z"],
                x_lim_2d=X_LIM,
                y_lim_2d=Y_LIM,
                depth_scale=DEPTH_SCALE,
                linestyle=REAL_FRAME_STYLE,
                color=REAL_FRAME_COLOR,
                linewidth=REAL_FRAME_WIDTH,
                alpha=1.0,
                label=p["title"] if SHOW_SLICE_LABEL else None,
                label_color=REAL_FRAME_COLOR,
                label_fontsize=14,
            )

        for p in virtual_panels:
            draw_slice_plane_frame(
                ax=ax,
                z_um=p["z"],
                x_lim_2d=X_LIM,
                y_lim_2d=Y_LIM,
                depth_scale=DEPTH_SCALE,
                linestyle=VIRTUAL_FRAME_STYLE,
                color=VIRTUAL_FRAME_COLOR,
                linewidth=VIRTUAL_FRAME_WIDTH,
                alpha=0.55,
                label=None,
                label_color=VIRTUAL_FRAME_COLOR,
                label_fontsize=12,
            )

    # --------------------
    # Draw cut plane frame across volume
    # --------------------
    draw_cut_plane_frame(
        ax=ax,
        anchor_2d=CUT_ANCHOR_2D,
        cut_angle_deg=CUT_ANGLE_DEG,
        depth_min=depth_min,
        depth_max=depth_max,
        xlim_2d=X_LIM,
        ylim_2d=Y_LIM,
        frame_color=CUT_FRAME_COLOR,
        frame_width=CUT_FRAME_WIDTH,
        frame_alpha=CUT_FRAME_ALPHA,
        face_alpha=CUT_FACE_ALPHA,
        draw_face=True,
        draw_label=SHOW_CUT_LABEL,
        label=f"{CUT_LABEL}\n{CUT_ANGLE_DEG:.1f}°",
        label_fontsize=12,
    )

    # --------------------
    # Axes direction
    # --------------------
    if INVERT_Y:
        ax.set_zlim(Y_LIM[::-1])
    else:
        ax.set_zlim(Y_LIM)

    if INVERT_X:
        ax.set_ylim(X_LIM[::-1])
    else:
        ax.set_ylim(X_LIM)

    # --------------------
    # Cosmetics
    # --------------------
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=ELEV, azim=AZIM)

    fig.suptitle(
        f"{GENE}: angled in silico resection through dense real + virtual stack",
        y=0.97,
        fontsize=14,
    )

    # --------------------
    # Legend
    # --------------------
    legend_handles = [
        Line2D(
            [0], [0],
            color=REAL_FRAME_COLOR,
            linestyle=REAL_FRAME_STYLE,
            linewidth=REAL_FRAME_WIDTH,
            label="Real slices (solid frame)",
        ),
        Line2D(
            [0], [0],
            color=VIRTUAL_FRAME_COLOR,
            linestyle=VIRTUAL_FRAME_STYLE,
            linewidth=VIRTUAL_FRAME_WIDTH,
            label="Virtual slices (dashed frame)",
        ),
        Line2D(
            [0], [0],
            color=CUT_FRAME_COLOR,
            linestyle="-",
            linewidth=CUT_FRAME_WIDTH,
            label="Resection plane",
        ),
    ]
    # ax.legend(
    #     handles=legend_handles,
    #     loc="upper left",
    #     bbox_to_anchor=(0.02, 0.98),
    #     frameon=True,
    # )

    # --------------------
    # Colorbars
    # --------------------
    sm_real = ScalarMappable(norm=norm_real, cmap=CMAP)
    sm_real.set_array([])

    sm_virtual = ScalarMappable(norm=norm_virtual, cmap=CMAP)
    sm_virtual.set_array([])

    fig.subplots_adjust(left=0.02, right=0.84, top=0.94, bottom=0.04)

    cax_real = fig.add_axes([0.87, 0.58, 0.02, 0.21])
    cbar_real = fig.colorbar(sm_real, cax=cax_real)
    cbar_real.set_label(
        f"{GENE} log1p(counts)" if USE_LOG1P else f"{GENE}",
        rotation=90,
    )

    cax_virtual = fig.add_axes([0.87, 0.18, 0.02, 0.25])
    cbar_virtual = fig.colorbar(sm_virtual, cax=cax_virtual)
    cbar_virtual.set_label(
        f"{GENE} log1p(counts)" if USE_LOG1P else f"{GENE}",
        rotation=90,
    )

    plt.show()



def draw_slice_plane_frame(
    ax,
    z_um,
    x_lim_2d,
    y_lim_2d,
    depth_scale=1.0,
    linestyle="-",
    color="0.55",
    linewidth=1.2,
    alpha=0.8,
    label=None,
    label_color=None,
    label_fontsize=10,
    label_y_frac=0.01,
    label_z_frac=0.08,
):
    """
    Draw a rectangular slice frame at constant depth in the 3D stack.

    Parameters
    ----------
    ax : mpl 3d axis
    z_um : float
        Slice depth before scaling.
    x_lim_2d : tuple
        2D x-coordinate limits for the slice plane (mapped to ax.set_ylabel("x")).
    y_lim_2d : tuple
        2D y-coordinate limits for the slice plane (mapped to ax.set_zlabel("y")).
    depth_scale : float
        Scale applied to z_um before plotting on the 3D depth axis.
    linestyle : str
        "-" for real, "--" for virtual.
    color : str
        Frame line color.
    linewidth : float
        Frame line width.
    alpha : float
        Frame transparency.
    label : str or None
        Optional text shown near the upper-left corner of the frame.
    """

    x_depth = float(z_um) * float(depth_scale)

    y0, y1 = float(min(x_lim_2d)), float(max(x_lim_2d))
    z0, z1 = float(min(y_lim_2d)), float(max(y_lim_2d))

    xs = np.full(5, x_depth, dtype=np.float32)
    ys = np.array([y0, y1, y1, y0, y0], dtype=np.float32)
    zs = np.array([z0, z0, z1, z1, z0], dtype=np.float32)

    ax.plot(
        xs,
        ys,
        zs,
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=0,
    )

    if label is not None:
        if label_color is None:
            label_color = color

        y_text = y0 + label_y_frac * (y1 - y0)
        z_text = z0 - label_z_frac * (z1 - z0)

        ax.text(
            x_depth,
            y_text,
            z_text,
            label,
            color=label_color,
            fontsize=label_fontsize,
            ha="left",
            va="top",
            zorder=10,
        )




def plot_virtual_3d(
    GENE,
    CMAP="viridis",
    REAL_FRAME_COLOR="black",
    VIRTUAL_FRAME_COLOR="0.35",
    REAL_FRAME_STYLE="-",
    VIRTUAL_FRAME_STYLE="--",
    REAL_FRAME_WIDTH=1.4,
    VIRTUAL_FRAME_WIDTH=1.4,
    SHOW_SLICE_FRAME=True,
    SHOW_SLICE_LABEL=True,
    VIRTUAL_UNIT_LABEL="um",   # change if your virtual axis is not in microns
):
    # --------------------
    # Load Real
    # --------------------
    real_panels = []
    real_vals_for_scale = []
    # cutting_threshold = 0
    for sid in ORIG_SAMPLES:
        z = ORIG_Z_UM[sid]
        ad = sc.read_h5ad(os.path.join(H5AD_DIR, f"{GROUP_ID}_{sid}.h5ad"))
        # coords = load_coords_npz(NET_IN, GROUP_ID, sid)
        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)

        try:
            vals = extract_gene_vals(ad, GENE)
        except ValueError:
            vals, _ = extract_obs_vals(ad, GENE)

        real_panels.append({
            "title": f"Real ({z}um)",
            "coords": coords,
            "vals": vals,
            "z": float(z),
        })
        real_vals_for_scale.append(vals)


    # --------------------
    # Load Virtual
    # --------------------
    GEN_PATTERN = os.path.join(GEN_DIR, "*.h5ad")

    gen_files = sorted(glob.glob(GEN_PATTERN), key=parse_z_from_filename)

    virtual_panels = []
    virtual_vals_for_scale = []

    for f in gen_files:
        z = parse_z_from_filename(f)
        z = int(z)

        ad = sc.read_h5ad(f)
        if "valid" in ad.obs.columns:
            ad = ad[ad.obs["valid"]].copy()

        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)

        if GENE is None:
            vals = np.full(coords.shape[0], 1.0, dtype=np.float32)
        else:
            try:
                vals = extract_gene_vals(ad, GENE)
            except ValueError:
                vals, _ = extract_obs_vals(ad, GENE)
        del ad

        virtual_panels.append({
            "title": f"Virtual ({z}{VIRTUAL_UNIT_LABEL})",
            "coords": coords,
            "vals": vals,
            "z": int(z),
        })
        virtual_vals_for_scale.append(vals)

    # --------------------
    # Norms
    # --------------------
    vmin_real, vmax_real = robust_vrange(
        real_vals_for_scale,
        1 - VMAX_PCT_REAL,
        VMAX_PCT_REAL,
    )
    vmin_virtual, vmax_virtual = robust_vrange(
        virtual_vals_for_scale,
        1 - VMAX_PCT_VIRTUAL,
        VMAX_PCT_VIRTUAL,
    )

    norm_real = Normalize(vmin=float(vmin_real), vmax=float(vmax_real))
    norm_virtual = Normalize(vmin=float(vmin_virtual), vmax=float(vmax_virtual))

    print(f"[SCALE] Real:    vmin={vmin_real:.4f}, vmax={vmax_real:.4f}")
    print(f"[SCALE] Virtual: vmin={vmin_virtual:.4f}, vmax={vmax_virtual:.4f}")

    # --------------------
    # 3D Figure
    # --------------------
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection="3d")
    all_depths = [p["z"] * DEPTH_SCALE for p in real_panels + virtual_panels]
    x_min = min(all_depths)
    x_max = max(all_depths)
    x_pad = 0.02 * max(x_max - x_min, 1.0)
    # visual aspect, not data aspect
    # ax.set_box_aspect((10, 1.3, 1.3))
    x_range = (x_max + x_pad) - (x_min - x_pad)
    y_range = abs(X_LIM[1] - X_LIM[0])
    z_range = abs(Y_LIM[1] - Y_LIM[0])

    # optional exaggeration of depth for visual clarity
    depth_exaggeration = 2.5
    ax.set_box_aspect((x_range * depth_exaggeration, y_range, z_range))
    ax.set_position([0.03, 0.08, 0.80, 0.84])

    ax.set_proj_type("ortho")

    # 2D coordinate limits on the slice planes


    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(X_LIM)
    ax.set_zlim(Y_LIM)


    # --------------------
    # Scatter real slices
    # --------------------
    for p in real_panels:
        scatter_slice_horizontal_depth(
            ax=ax,
            coords=p["coords"],
            vals=p["vals"],
            z_um=p["z"],
            norm=norm_real,
            cmap_name=CMAP,
            s=S_REAL,
            alpha_max=ALPHA_MAX_REAL,
            label_prefix="Real",
            DEPTH_SCALE=DEPTH_SCALE,
            ALPHA_MIN=ALPHA_MIN_REAL,
            ALPHA_GAMMA=ALPHA_GAMMA,
        )

    # --------------------
    # Scatter virtual slices
    # --------------------
    for p in virtual_panels:
        scatter_slice_horizontal_depth(
            ax=ax,
            coords=p["coords"],
            vals=p["vals"],
            z_um=p["z"],
            norm=norm_virtual,
            cmap_name=CMAP,
            s=S_VIRTUAL,
            alpha_max=ALPHA_MAX_VIRTUAL,
            label_prefix="Virtual",
            DEPTH_SCALE=DEPTH_SCALE,
            ALPHA_MIN=ALPHA_MIN_VIRTUAL,
            ALPHA_GAMMA=ALPHA_GAMMA,
        )

        # --------------------
        # Draw slice frames first
        # --------------------
    if SHOW_SLICE_FRAME:
        for p in real_panels:
            draw_slice_plane_frame(
                ax=ax,
                z_um=p["z"],
                x_lim_2d=X_LIM,
                y_lim_2d=Y_LIM,
                depth_scale=DEPTH_SCALE,
                linestyle=REAL_FRAME_STYLE,
                color=REAL_FRAME_COLOR,
                linewidth=REAL_FRAME_WIDTH,
                alpha=1.,
                label=p["title"] if SHOW_SLICE_LABEL else None,
                label_color=REAL_FRAME_COLOR,
                label_fontsize=20,
            )

        for p in virtual_panels:
            if p["z"]==190:
                title="Virtual slices"
            else:
                title=None
            draw_slice_plane_frame(
                ax=ax,
                z_um=p["z"],
                x_lim_2d=X_LIM,
                y_lim_2d=Y_LIM,
                depth_scale=DEPTH_SCALE,
                linestyle=VIRTUAL_FRAME_STYLE,
                color=VIRTUAL_FRAME_COLOR,
                linewidth=VIRTUAL_FRAME_WIDTH,
                alpha=0.70,
                # label=p["title"] if SHOW_SLICE_LABEL else None,
                label=title,
                label_color=VIRTUAL_FRAME_COLOR,
                label_fontsize=20,
                label_y_frac=-0.05
            )

    # --------------------
    # Axes direction
    # --------------------
    if INVERT_Y:
        ax.set_zlim(Y_LIM[::-1])
    else:
        ax.set_zlim(Y_LIM)

    if INVERT_X:
        ax.set_ylim(X_LIM[::-1])
    else:
        ax.set_ylim(X_LIM)

    # --------------------
    # Labels and title
    # --------------------
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=ELEV, azim=AZIM)
    fig.suptitle(
        f"{GENE}: 3D stack of real and virtual slices",
        y=0.97,
        fontsize=14,
    )

    # --------------------
    # Legend
    # --------------------
    legend_handles = [
        Line2D(
            [0], [0],
            color=REAL_FRAME_COLOR,
            linestyle=REAL_FRAME_STYLE,
            linewidth=REAL_FRAME_WIDTH,
            label="Real slices (solid frame)",
        ),
        Line2D(
            [0], [0],
            color=VIRTUAL_FRAME_COLOR,
            linestyle=VIRTUAL_FRAME_STYLE,
            linewidth=VIRTUAL_FRAME_WIDTH,
            label="Virtual slices (dashed frame)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
    )
    #
    # --------------------
    # Colorbars
    # --------------------
    sm_real = ScalarMappable(norm=norm_real, cmap=CMAP)
    sm_real.set_array([])

    sm_virtual = ScalarMappable(norm=norm_virtual, cmap=CMAP)
    sm_virtual.set_array([])

    fig.subplots_adjust(left=0.02, right=0.84, top=0.94, bottom=0.04)

    cax_real = fig.add_axes([0.87, 0.58, 0.02, 0.21])
    cbar_real = fig.colorbar(sm_real, cax=cax_real)
    cbar_real.set_label(
        f"{GENE} log1p(counts) " if USE_LOG1P else f"{GENE} ",
        rotation=90,
    )

    cax_virtual = fig.add_axes([0.87, 0.18, 0.02, 0.25])
    cbar_virtual = fig.colorbar(sm_virtual, cax=cax_virtual)
    cbar_virtual.set_label(
        f"{GENE} log1p(counts)" if USE_LOG1P else f"{GENE}",
        rotation=90,
    )

    plt.show()

def add_domain_offset(coords,vals):
    scale=10
    mask = vals == 0
    coords[mask, 1] = coords[mask, 1] - 50*scale

    mask = vals == 1
    coords[mask, 1] = coords[mask, 1] - 40*scale

    mask = vals == 2
    coords[mask, 1] = coords[mask, 1] - 30*scale
    mask = vals == 3
    coords[mask, 1] = coords[mask, 1] - 20*scale
    mask = vals == 4
    coords[mask, 1] = coords[mask, 1] - 10*scale

    return coords,vals


def plot_insilico_resectioning(
    GENE,
    cutting_threshold=0,
    CMAP="viridis",
    REAL_FRAME_COLOR="black",
    VIRTUAL_FRAME_COLOR="0.35",
    REAL_FRAME_STYLE="-",
    VIRTUAL_FRAME_STYLE="--",
    REAL_FRAME_WIDTH=1.4,
    VIRTUAL_FRAME_WIDTH=1.4,
    SHOW_SLICE_FRAME=True,
    SHOW_SLICE_LABEL=True,
    VIRTUAL_UNIT_LABEL="um",   # change if your virtual axis is not in microns
):
    # --------------------
    # Load Real
    # --------------------
    real_panels = []
    real_vals_for_scale = []

    # cutting_threshold = [0,120,200,320]
    for sid in ORIG_SAMPLES:
        z = ORIG_Z_UM[sid]
        ad = sc.read_h5ad(os.path.join(H5AD_DIR, f"{GROUP_ID}_{sid}.h5ad"))
        # coords = load_coords_npz(NET_IN, GROUP_ID, sid)
        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)

        try:
            vals = extract_gene_vals(ad, GENE)
        except ValueError:
            vals, _ = extract_obs_vals(ad, GENE)

        coords,vals = add_domain_offset(coords,vals)

        real_panels.append({
            "title": f"Real ({z}um)",
            "coords": coords,
            "vals": vals,
            "z": float(z),
        })
        real_vals_for_scale.append(vals)


    # --------------------
    # Load Virtual
    # --------------------
    GEN_PATTERN = os.path.join(GEN_DIR, "*.h5ad")

    gen_files = sorted(glob.glob(GEN_PATTERN), key=parse_z_from_filename)

    virtual_panels = []
    virtual_vals_for_scale = []
    virtual_depth = np.arange(40, 310, 30)

    for f in gen_files:
        z = parse_z_from_filename(f)
        z = int(z)
        if z not in virtual_depth:
            continue

        ad = sc.read_h5ad(f)
        if "valid" in ad.obs.columns:
            ad = ad[ad.obs["valid"]].copy()

        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)

        if GENE is None:
            vals = np.full(coords.shape[0], 1.0, dtype=np.float32)
        else:
            try:
                vals = extract_gene_vals(ad, GENE)
            except ValueError:
                vals, _ = extract_obs_vals(ad, GENE)
        del ad

        coords,vals = add_domain_offset(coords,vals)

        virtual_panels.append({
            "title": f"Virtual ({z}{VIRTUAL_UNIT_LABEL})",
            "coords": coords,
            "vals": vals,
            "z": int(z),
        })
        virtual_vals_for_scale.append(vals)

    # --------------------
    # Norms
    # --------------------
    vmin_real, vmax_real = robust_vrange(
        real_vals_for_scale,
        1 - VMAX_PCT_REAL,
        VMAX_PCT_REAL,
    )
    vmin_virtual, vmax_virtual = robust_vrange(
        virtual_vals_for_scale,
        1 - VMAX_PCT_VIRTUAL,
        VMAX_PCT_VIRTUAL,
    )

    norm_real = Normalize(vmin=float(vmin_real), vmax=float(vmax_real))
    norm_virtual = Normalize(vmin=float(vmin_virtual), vmax=float(vmax_virtual))

    print(f"[SCALE] Real:    vmin={vmin_real:.4f}, vmax={vmax_real:.4f}")
    print(f"[SCALE] Virtual: vmin={vmin_virtual:.4f}, vmax={vmax_virtual:.4f}")

    # --------------------
    # 3D Figure
    # --------------------
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection="3d")
    all_depths = [p["z"] * DEPTH_SCALE for p in real_panels + virtual_panels]
    x_min = min(all_depths)
    x_max = max(all_depths)
    x_pad = 0.02 * max(x_max - x_min, 1.0)
    # visual aspect, not data aspect
    # ax.set_box_aspect((10, 1.3, 1.3))
    x_range = (x_max + x_pad) - (x_min - x_pad)
    y_range = abs(X_LIM[1] - X_LIM[0])
    z_range = abs(Y_LIM[1] - Y_LIM[0])

    # optional exaggeration of depth for visual clarity
    depth_exaggeration = 1.5
    ax.set_box_aspect((x_range * depth_exaggeration, y_range, z_range))
    ax.set_position([0.03, 0.08, 0.80, 0.84])

    ax.set_proj_type("ortho")

    # 2D coordinate limits on the slice planes


    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(X_LIM)
    ax.set_zlim(Y_LIM)


    # --------------------
    # Scatter real slices
    # --------------------
    for p in real_panels:
        scatter_slice_horizontal_depth(
            ax=ax,
            coords=p["coords"],
            vals=p["vals"],
            z_um=p["z"],
            norm=norm_real,
            cmap_name=CMAP,
            s=S_REAL,
            alpha_max=ALPHA_MAX_REAL,
            label_prefix="Real",
            DEPTH_SCALE=DEPTH_SCALE,
            ALPHA_MIN=ALPHA_MIN_REAL,
            ALPHA_GAMMA=ALPHA_GAMMA,
        )

    # --------------------
    # Scatter virtual slices
    # --------------------
    for p in virtual_panels:
        scatter_slice_horizontal_depth(
            ax=ax,
            coords=p["coords"],
            vals=p["vals"],
            z_um=p["z"],
            norm=norm_virtual,
            cmap_name=CMAP,
            s=S_VIRTUAL,
            alpha_max=ALPHA_MAX_VIRTUAL,
            label_prefix="Virtual",
            DEPTH_SCALE=DEPTH_SCALE,
            ALPHA_MIN=ALPHA_MIN_VIRTUAL,
            ALPHA_GAMMA=ALPHA_GAMMA,
        )

        # --------------------
        # Draw slice frames first
        # --------------------
    if SHOW_SLICE_FRAME:
        for p in real_panels:
            draw_slice_plane_frame(
                ax=ax,
                z_um=p["z"],
                x_lim_2d=X_LIM,
                y_lim_2d=Y_LIM,
                depth_scale=DEPTH_SCALE,
                linestyle=REAL_FRAME_STYLE,
                color=REAL_FRAME_COLOR,
                linewidth=REAL_FRAME_WIDTH,
                alpha=1.,
                label=p["title"] if SHOW_SLICE_LABEL else None,
                label_color=REAL_FRAME_COLOR,
                label_fontsize=14,
            )

        # for p in virtual_panels:
        #     draw_slice_plane_frame(
        #         ax=ax,
        #         z_um=p["z"],
        #         x_lim_2d=X_LIM,
        #         y_lim_2d=Y_LIM,
        #         depth_scale=DEPTH_SCALE,
        #         linestyle=VIRTUAL_FRAME_STYLE,
        #         color=VIRTUAL_FRAME_COLOR,
        #         linewidth=VIRTUAL_FRAME_WIDTH,
        #         alpha=0.70,
        #         label=p["title"] if SHOW_SLICE_LABEL else None,
        #         label_color=VIRTUAL_FRAME_COLOR,
        #         label_fontsize=14,
        #     )

    # --------------------
    # Axes direction
    # --------------------
    if INVERT_Y:
        ax.set_zlim(Y_LIM[::-1])
    else:
        ax.set_zlim(Y_LIM)

    if INVERT_X:
        ax.set_ylim(X_LIM[::-1])
    else:
        ax.set_ylim(X_LIM)

    # --------------------
    # Labels and title
    # --------------------
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=ELEV, azim=AZIM)
    fig.suptitle(
        f"{GENE}: 3D stack of real and virtual slices",
        y=0.97,
        fontsize=14,
    )

    # --------------------
    # Legend
    # --------------------
    legend_handles = [
        Line2D(
            [0], [0],
            color=REAL_FRAME_COLOR,
            linestyle=REAL_FRAME_STYLE,
            linewidth=REAL_FRAME_WIDTH,
            label="Real slices (solid frame)",
        )
    ]
    # ax.legend(
    #     handles=legend_handles,
    #     loc="upper left",
    #     bbox_to_anchor=(0.02, 0.98),
    #     frameon=True,
    # )
    #
    # --------------------
    # Colorbars
    # --------------------
    sm_real = ScalarMappable(norm=norm_real, cmap=CMAP)
    sm_real.set_array([])

    sm_virtual = ScalarMappable(norm=norm_virtual, cmap=CMAP)
    sm_virtual.set_array([])

    fig.subplots_adjust(left=0.02, right=0.84, top=0.94, bottom=0.04)

    cax_real = fig.add_axes([0.87, 0.58, 0.02, 0.21])
    cbar_real = fig.colorbar(sm_real, cax=cax_real)
    cbar_real.set_label(
        f"{GENE} log1p(counts) " if USE_LOG1P else f"{GENE} ",
        rotation=90,
    )

    cax_virtual = fig.add_axes([0.87, 0.18, 0.02, 0.25])
    cbar_virtual = fig.colorbar(sm_virtual, cax=cax_virtual)
    cbar_virtual.set_label(
        f"{GENE} log1p(counts)" if USE_LOG1P else f"{GENE}",
        rotation=90,
    )

    plt.show()


def plot_real_3d(
    GENE,
    CMAP="viridis",
    REAL_FRAME_COLOR="black",
    VIRTUAL_FRAME_COLOR="0.35",
    REAL_FRAME_STYLE="-",
    VIRTUAL_FRAME_STYLE="--",
    REAL_FRAME_WIDTH=1.4,
    VIRTUAL_FRAME_WIDTH=1.4,
    SHOW_SLICE_FRAME=True,
    SHOW_SLICE_LABEL=True,
):
    # --------------------
    # Load Real
    # --------------------
    real_panels = []
    real_vals_for_scale = []
    z_locations=[0,110,220,320]
    for sid in ORIG_SAMPLES:
        z = ORIG_Z_UM[sid]
        ad = sc.read_h5ad(os.path.join(H5AD_DIR, f"{GROUP_ID}_{sid}.h5ad"))
        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)

        try:
            vals = extract_gene_vals(ad, GENE)
        except ValueError:
            vals, _ = extract_obs_vals(ad, GENE)

        n = min(coords.shape[0], vals.shape[0])
        coords = coords[:n]
        vals = vals[:n]

        real_panels.append({
            "title": f"Real ({z} um)",
            "coords": coords,
            "vals": vals,
            # "z": float(z),
            "z":z_locations[sid]
        })
        real_vals_for_scale.append(vals)

    # --------------------
    # Norms
    # --------------------
    vmin_real, vmax_real = robust_vrange(
        real_vals_for_scale,
        1 - VMAX_PCT_REAL,
        VMAX_PCT_REAL,
    )


    norm_real = Normalize(vmin=float(vmin_real), vmax=float(vmax_real))
    print(f"[SCALE] Real:    vmin={vmin_real:.4f}, vmax={vmax_real:.4f}")
    # --------------------
    # 3D Figure
    # --------------------

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    all_depths = [p["z"] * DEPTH_SCALE for p in real_panels]
    x_min = min(all_depths)
    x_max = max(all_depths)
    x_pad = 0.02 * max(x_max - x_min, 1.0)
    # visual aspect, not data aspect
    # ax.set_box_aspect((10, 1.3, 1.3))
    x_range = (x_max + x_pad) - (x_min - x_pad)
    y_range = abs(X_LIM[1] - X_LIM[0])
    z_range = abs(Y_LIM[1] - Y_LIM[0])

    # optional exaggeration of depth for visual clarity
    depth_exaggeration = 2.
    ax.set_box_aspect((x_range * depth_exaggeration, y_range, z_range))
    ax.set_position([0.03, 0.08, 0.80, 0.84])

    ax.set_proj_type("ortho")

    # 2D coordinate limits on the slice planes
    ax.set_ylim(X_LIM)
    ax.set_zlim(Y_LIM)

    # --------------------
    # Draw slice frames first
    # --------------------
    if SHOW_SLICE_FRAME:
        for p in real_panels:

            draw_slice_plane_frame(
                ax=ax,
                z_um=p["z"],
                x_lim_2d=X_LIM,
                y_lim_2d=Y_LIM,
                depth_scale=DEPTH_SCALE,
                linestyle=REAL_FRAME_STYLE,
                color=REAL_FRAME_COLOR,
                linewidth=REAL_FRAME_WIDTH,
                alpha=0.75,
                label=p["title"] if SHOW_SLICE_LABEL else None,
                label_color=REAL_FRAME_COLOR,
                label_fontsize=20,
                label_z_frac=0.05
            )
        # draw_slice_plane_frame(
        #     ax=ax,
        #     z_um=160,
        #     x_lim_2d=X_LIM,
        #     y_lim_2d=Y_LIM,
        #     depth_scale=DEPTH_SCALE,
        #     linestyle="--",
        #     color=REAL_FRAME_COLOR,
        #     linewidth=REAL_FRAME_WIDTH,
        #     alpha=0.75,
        #     label="Unmeasured",
        #     label_color=REAL_FRAME_COLOR,
        #     label_fontsize=20,
        #     label_y_frac=-0.0,
        #     label_z_frac=0.05
        # )
    # --------------------
    # Scatter real slices
    # --------------------
    for p in real_panels:
        scatter_slice_horizontal_depth(
            ax=ax,
            coords=p["coords"],
            vals=p["vals"],
            z_um=p["z"],
            norm=norm_real,
            cmap_name=CMAP,
            s=S_REAL,
            alpha_max=ALPHA_MAX_REAL,
            label_prefix="Real",
            DEPTH_SCALE=DEPTH_SCALE,
            ALPHA_MIN=ALPHA_MIN_REAL,
            ALPHA_GAMMA=ALPHA_GAMMA,
        )

    # --------------------
    # Axes direction
    # --------------------
    if INVERT_Y:
        ax.set_zlim(Y_LIM[::-1])
    else:
        ax.set_zlim(Y_LIM)

    if INVERT_X:
        ax.set_ylim(X_LIM[::-1])
    else:
        ax.set_ylim(X_LIM)

    # --------------------
    # Labels and title
    # --------------------
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=ELEV, azim=AZIM)
    fig.suptitle(
        f"{GENE}: 3D stack of real and virtual slices",
        y=0.97,
        fontsize=14,
    )

    # --------------------
    # Legend
    # --------------------
    legend_handles = [
        Line2D(
            [0], [0],
            color=REAL_FRAME_COLOR,
            linestyle=REAL_FRAME_STYLE,
            linewidth=REAL_FRAME_WIDTH,
            label="Real slices (solid frame)",
        ),
        Line2D(
            [0], [0],
            color=VIRTUAL_FRAME_COLOR,
            linestyle=VIRTUAL_FRAME_STYLE,
            linewidth=VIRTUAL_FRAME_WIDTH,
            label="Unmeasured tissue states (dashed frame)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
    )
    #
    # --------------------
    # Colorbars
    # --------------------
    sm_real = ScalarMappable(norm=norm_real, cmap=CMAP)
    sm_real.set_array([])


    fig.subplots_adjust(left=0.02, right=0.84, top=0.94, bottom=0.04)

    cax_real = fig.add_axes([0.87, 0.58, 0.02, 0.21])
    cbar_real = fig.colorbar(sm_real, cax=cax_real)
    cbar_real.set_label(
        f"{GENE} log1p(counts) " if USE_LOG1P else f"{GENE} ",
        rotation=90,
    )


    plt.show()

import scipy
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

def plot_counterfactual_3d(
    GENE,
    CMAP="viridis",
    VIRTUAL_FRAME_COLOR="0.35",
    VIRTUAL_FRAME_STYLE="--",
    VIRTUAL_FRAME_WIDTH=1.4,
    SHOW_SLICE_FRAME=True,
    SHOW_SLICE_LABEL=True,
    VIRTUAL_UNIT_LABEL="um",   # change if your virtual axis is not in microns
):
    # --------------------
    # Load Virtual
    # --------------------
    GEN_PATTERN = os.path.join(GEN_DIR, "*.h5ad")
    gen_files = sorted(glob.glob(GEN_PATTERN), key=parse_z_from_filename)
    virtual_panels = []
    virtual_vals_for_scale = []


    for id,f in enumerate(gen_files):
        z = parse_z_from_filename(f)
        ad = sc.read_h5ad(f)
        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)

        if GENE is None:
            vals = np.full(coords.shape[0], 1.0, dtype=np.float32)
        else:
            try:
                vals = extract_gene_vals(ad, GENE)
            except ValueError:
                vals, _ = extract_obs_vals(ad, GENE)

        mask = get_valid_mask(ad)
        coords = coords[mask]
        vals = vals[mask]

        # virtual_panels.append({
        #     # "title": f"Counterfactual \n ({z}{VIRTUAL_UNIT_LABEL})",
        #     "title": f"Counterfactual slices",
        #     "coords": coords,
        #     "vals": vals,
        #     "z": int(id),
        # })
        virtual_panels.append({
            "title": f"Aggregated gene reads",
            "coords": coords,
            "vals": vals,
            "z": int(1),
        })
        virtual_vals_for_scale.append(vals)

    # --------------------
    # Norms
    # --------------------
    vmin_virtual, vmax_virtual = robust_vrange(
        virtual_vals_for_scale,
        1 - VMAX_PCT_VIRTUAL,
        VMAX_PCT_VIRTUAL,
    )

    norm_virtual = Normalize(vmin=float(vmin_virtual), vmax=float(vmax_virtual))
    print(f"[SCALE] Virtual: vmin={vmin_virtual:.4f}, vmax={vmax_virtual:.4f}")

    # --------------------
    # 3D Figure
    # --------------------
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    # all_depths = [p["z"] * DEPTH_SCALE for p in virtual_panels]
    # x_min = min(all_depths)
    # x_max = max(all_depths)
    # x_pad = 0.02 * max(x_max - x_min, 1.0)
    # # visual aspect, not data aspect
    # # ax.set_box_aspect((10, 1.3, 1.3))
    # x_range = (x_max + x_pad) - (x_min - x_pad)
    y_range = abs(X_LIM[1] - X_LIM[0])
    z_range = abs(Y_LIM[1] - Y_LIM[0])

    # optional exaggeration of depth for visual clarity
    # ax.set_box_aspect((1.5*y_range, y_range, z_range), zoom=1.2)
    ax.set_box_aspect((1. * y_range, y_range, z_range), zoom=1.2)
    ax.set_position([0.03, 0.08, 0.80, 0.84])

    ax.set_proj_type("ortho")

    # 2D coordinate limits on the slice planes


    # ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(X_LIM)
    ax.set_zlim(Y_LIM)

    # --------------------
    # Draw slice frames first
    # --------------------
    if SHOW_SLICE_FRAME:
        for i,p in enumerate(virtual_panels):
            if i==1:
                SHOW_SLICE_LABEL=True
            else:
                SHOW_SLICE_LABEL=False
            draw_slice_plane_frame(
                ax=ax,
                z_um=p["z"],
                x_lim_2d=X_LIM,
                y_lim_2d=Y_LIM,
                depth_scale=DEPTH_SCALE,
                linestyle=VIRTUAL_FRAME_STYLE,
                color=VIRTUAL_FRAME_COLOR,
                linewidth=VIRTUAL_FRAME_WIDTH,
                alpha=0.70,
                label=p["title"] if SHOW_SLICE_LABEL else None,
                label_color=VIRTUAL_FRAME_COLOR,
                label_fontsize=20,
                label_y_frac=0.2,
                label_z_frac=0.15,
            )

    # --------------------
    # Scatter virtual slices
    # --------------------
    for p in virtual_panels:
        scatter_slice_horizontal_depth(
            ax=ax,
            coords=p["coords"],
            vals=p["vals"],
            z_um=p["z"],
            norm=norm_virtual,
            cmap_name=CMAP,
            s=S_VIRTUAL,
            alpha_max=ALPHA_MAX_VIRTUAL,
            label_prefix="Virtual",
            DEPTH_SCALE=DEPTH_SCALE,
            ALPHA_MIN=ALPHA_MIN_VIRTUAL,
            ALPHA_GAMMA=ALPHA_GAMMA,
        )

    # --------------------
    # Axes direction
    # --------------------
    if INVERT_Y:
        ax.set_zlim(Y_LIM[::-1])
    else:
        ax.set_zlim(Y_LIM)

    if INVERT_X:
        ax.set_ylim(X_LIM[::-1])
    else:
        ax.set_ylim(X_LIM)

    # --------------------
    # Labels and title
    # --------------------
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=ELEV, azim=AZIM)
    fig.suptitle(
        f"{GENE}: 3D stack of real and virtual slices",
        y=0.97,
        fontsize=14,
    )

    # --------------------
    # Colorbars


    sm_virtual = ScalarMappable(norm=norm_virtual, cmap=CMAP)
    sm_virtual.set_array([])

    fig.subplots_adjust(left=0.02, right=0.84, top=0.94, bottom=0.04)


    cax_virtual = fig.add_axes([0.87, 0.18, 0.02, 0.25])
    cbar_virtual = fig.colorbar(sm_virtual, cax=cax_virtual)
    cbar_virtual.set_label(
        f"{GENE} log1p(counts)" if USE_LOG1P else f"{GENE}",
        rotation=90,
    )

    plt.show()





def plot_channel_3d(GEN_FILES,GENE_set,CMAPS,DEPTH_SCALE,X_LIM,Y_LIM,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA):
    virtual_panels = []
    virtual_vals_for_scale = []
    channel_z=0
    for f,GENE,CMAP in zip (GEN_FILES,GENE_set,CMAPS):
        ad = sc.read_h5ad(f)
        if "valid" in ad.obs.columns:
            ad = ad[ad.obs["valid"]].copy()
        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)
        for gene,cmap in zip(GENE,CMAP):

            try:
                vals = extract_gene_vals(ad, gene)
            except ValueError:
                vals, meta = extract_obs_vals(ad, gene)
            virtual_panels.append({
                "title": f"Virtual ({gene})",
                "coords": coords,
                "vals": vals,
                "z": float(channel_z),
                'cmap':cmap

            })
            virtual_vals_for_scale.append(vals)
            channel_z = channel_z+1
    # --------------------
    # Norms
    # --------------------
    vmax_virtual = robust_vmax(virtual_vals_for_scale, VMAX_PCT_VIRTUAL, VMAX_HARD_CAP_VIRTUAL)
    norm_virtual = Normalize(vmin=float(VMIN), vmax=float(vmax_virtual))
    print(f"[SCALE] Virtual: vmin={VMIN}, vmax={vmax_virtual:.4f}")

    # --------------------
    # 3D Figure
    # --------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # make the main 3D axes actually large
    ax.set_position([0.03, 0.08, 0.80, 0.84])

    # scientific-looking projection
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.3, 1.3, 1.3))
    # ax.set_box_aspect((1.3, 1.3, 1.3))

    # virtual
    for p in virtual_panels:
        scatter_slice_horizontal_depth(
            ax=ax,
            coords=p["coords"],
            vals=p["vals"],
            z_um=p["z"],
            norm=norm_virtual,
            cmap_name=p['cmap'],
            s=S_VIRTUAL,
            alpha_max=ALPHA_MAX_VIRTUAL,
            label_prefix="Virtual",
            DEPTH_SCALE=DEPTH_SCALE,
            ALPHA_MIN=ALPHA_MIN,
            ALPHA_GAMMA=ALPHA_GAMMA
        )


    if INVERT_Y:
        ax.set_zlim(Y_LIM[::-1])
    else:
        ax.set_zlim(Y_LIM)

    if INVERT_X:
        ax.set_ylim(X_LIM[::-1])
    else:
        ax.set_ylim(X_LIM)

    # ax.set_xlim(-10,25)

    ax.set_xlabel(f"slice depth (original z * {DEPTH_SCALE:g})")
    ax.set_ylabel("x")
    ax.set_zlabel("y")

    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_aspect("auto")
    ax.grid(False)

    fig.suptitle(
        f"{GENE}: 3D stack with horizontal slice depth and value-dependent transparency",
        y=0.97,
        fontsize=14,
    )

    # legend for slice type
    legend_handles = [

        Line2D([0], [0], marker='o', color='w', label='Virtual slices',
               markerfacecolor='gray', markersize=8, alpha=0.6),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98))
    ax.set_proj_type("ortho")
    sm_virtual = ScalarMappable(norm=norm_virtual, cmap="plasma")
    sm_virtual.set_array([])

    fig.subplots_adjust(left=0.02, right=0.84, top=0.94, bottom=0.04)

    cax_virtual = fig.add_axes([0.87, 0.18, 0.02, 0.25])
    cbar_virtual = fig.colorbar(sm_virtual, cax=cax_virtual)
    cbar_virtual.set_label(
        f"{GENE} log1p(counts) (Virtual)" if USE_LOG1P else f"{GENE} (Virtual)",
        rotation=90
    )
    plt.show()










GROUP_ID = 1

ORIG_Z_UM = {0: 0, 1: 10, 2: 310, 3: 320}

ROOT = "/media/huifang/data/registration/DLPFC/huifang"
NET_IN = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ad")
# H5AD_DIR = "/media/huifang/data/registration/DLPFC/original_h5adfile/"
GEN_USE_VALID_MASK = True
USE_LOG1P = True



VMIN = 0.0
VMAX_PCT_REAL = 0.995
VMAX_PCT_VIRTUAL = 0.995
VMAX_HARD_CAP_REAL = None
VMAX_HARD_CAP_VIRTUAL = None

# tissue coordinate display range
X_LIM = (0, 512)
Y_LIM = (0, 512)
# Y_LIM = (-500,512)
# X_LIM = (0, 12000)
# Y_LIM = (0, 12000)

# Magnify slice spacing while preserving actual z ordering/ratios
DEPTH_SCALE = 1

# view

INVERT_Y=True
INVERT_X =True


def get_3a_paras():
    # ORIG_SAMPLES = [1, 2]
    ORIG_SAMPLES = [0,1, 2,3]
    dir = os.path.join(ROOT, "virtual_tissue", f"dlpfc_z_group{GROUP_ID}_grid80_spatial1/query_discrete")
    ALPHA_MIN_REAL = 0.2  # almost invisible near zero
    ALPHA_MIN_VIRTUAL=0.2
    ALPHA_MAX_REAL =0.9
    ALPHA_MAX_VIRTUAL = 0.9
    ALPHA_GAMMA = 1.6  # >1 makes low values fade more aggressively
    # ELEV = 15
    # AZIM = -145
    ELEV = 0
    AZIM = -180
    S_REAL = 12
    S_VIRTUAL = 12
    return ORIG_SAMPLES,dir,ALPHA_MIN_REAL,ALPHA_MIN_VIRTUAL,ALPHA_MAX_REAL,ALPHA_MAX_VIRTUAL,ALPHA_GAMMA,ELEV,AZIM,S_REAL,S_VIRTUAL

def get_3a2_paras():
    ORIG_SAMPLES = [1, 2]
    # dir = os.path.join(ROOT, "virtual_tissue", f"dlpfc_z_group{GROUP_ID}_grid80_spatial1/query")
    dir = "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/spatialz/dense_reconstruction/group1/"
    ALPHA_MIN_REAL = 0.0  # almost invisible near zero
    ALPHA_MIN_VIRTUAL=0.0
    ALPHA_MAX_REAL =1
    ALPHA_MAX_VIRTUAL = 1
    ALPHA_GAMMA = 1.6  # >1 makes low values fade more aggressively
    # ELEV = 15
    # AZIM = 145
    ELEV = 15
    AZIM = 155
    S_REAL = 5
    S_VIRTUAL = 5
    return ORIG_SAMPLES,dir,ALPHA_MIN_REAL,ALPHA_MIN_VIRTUAL,ALPHA_MAX_REAL,ALPHA_MAX_VIRTUAL,ALPHA_GAMMA,ELEV,AZIM,S_REAL,S_VIRTUAL

def get_3a3_paras():
    ORIG_SAMPLES = [1, 2]
    dir = os.path.join(ROOT, "virtual_tissue", f"dlpfc_z_group{GROUP_ID}_grid80_spatial1/analysis/aggregation")
    GEN_FILES=[os.path.join(dir, "aggregate_max_ref_0.000.h5ad")]
    GENE_SET=[["PCP4","layer_guess_reordered"]]
    CMAPS=[["Reds",'tab10']]
    ALPHA_MIN=0.9
    ALPHA_MAX_VIRTUAL = 1
    ALPHA_GAMMA = 1.6  # >1 makes low values fade more aggressively
    ELEV = 15
    AZIM = -145
    S_REAL = 9
    S_VIRTUAL = 9
    return GEN_FILES,GENE_SET,CMAPS,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA



def get_3b_paras():
    ORIG_SAMPLES = [1,2]
    dir = os.path.join(ROOT, "virtual_tissue", f"dlpfc_z_group{GROUP_ID}_grid80_spatial1/query_discrete")
    # dir = "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/spatialz/dense_reconstruction/group1/backup"
    ALPHA_MIN_REAL = 0.  # almost invisible near zero
    ALPHA_MIN_VIRTUAL = 0.
    ALPHA_MAX_REAL = 1.0
    ALPHA_MAX_VIRTUAL = 1
    ALPHA_GAMMA = 1.6  # >1 makes low values fade more aggressively
    ELEV = 15
    AZIM = -145
    S_REAL = 9
    S_VIRTUAL = 9
    return ORIG_SAMPLES,dir,ALPHA_MIN_REAL,ALPHA_MIN_VIRTUAL,ALPHA_MAX_REAL,ALPHA_MAX_VIRTUAL,ALPHA_GAMMA,ELEV,AZIM,S_REAL,S_VIRTUAL


def get_3d_paras():
    ORIG_SAMPLES = [0,1,2,3]
    dir = os.path.join(ROOT, "virtual_tissue", f"dlpfc_z_group{GROUP_ID}_grid80_spatial1/analysis")
    ALPHA_MIN_REAL = 0.2  # almost invisible near zero
    ALPHA_MIN_VIRTUAL=0.2
    ALPHA_MAX_REAL = 1.0
    ALPHA_MAX_VIRTUAL = 1.0
    ALPHA_GAMMA = 1.6  # >1 makes low values fade more aggressively
    ELEV = 15
    AZIM = -145
    S_REAL = 12
    S_VIRTUAL = 12
    return ORIG_SAMPLES,dir,ALPHA_MIN_REAL,ALPHA_MIN_VIRTUAL,ALPHA_MAX_REAL,ALPHA_MAX_VIRTUAL,ALPHA_GAMMA,ELEV,AZIM,S_REAL,S_VIRTUAL

ORIG_SAMPLES,GEN_DIR, ALPHA_MIN_REAL,ALPHA_MIN_VIRTUAL,ALPHA_MAX_REAL,ALPHA_MAX_VIRTUAL,ALPHA_GAMMA,ELEV,AZIM,S_REAL,S_VIRTUAL= get_3a_paras()
plot_real_3d(GENE = "layer_guess_reordered", CMAP='tab10')
# plot_real_3d(GENE = "PCP4", CMAP="Reds")
#
ORIG_SAMPLES,GEN_DIR, ALPHA_MIN_REAL,ALPHA_MIN_VIRTUAL,ALPHA_MAX_REAL,ALPHA_MAX_VIRTUAL,ALPHA_GAMMA,ELEV,AZIM,S_REAL,S_VIRTUAL = get_3a2_paras()
plot_insilico_resectioning(GENE = "layer_guess_reordered", CMAP='tab10',SHOW_SLICE_FRAME=False)
# # plot_insilico_resectioning(GENE = "PCP4", CMAP="Reds",cutting_threshold=0,SHOW_SLICE_FRAME=False)
plot_insilico_resectioning_fancy(
    GENE="layer_guess_reordered",
    CUT_ANGLE_DEG=-90,
    CUT_ANCHOR_2D=[120,120],
    CMAP='tab10',
    SHOW_SLICE_FRAME=False,
    SHOW_CUT_LABEL=False
)
# [200,360]


# plot_insilico_resectioning_fancy(
#     GENE="layer_guess_reordered",
#     CMAP="tab10",
#     DEPTH_SCALE=DEPTH_SCALE,
#
#     CUT_STYLE="vertical_spin",
#
#     # vertical hinge line location
#     CUT_ANCHOR_DEPTH=160 * DEPTH_SCALE,
#     CUT_ANCHOR_COORD0=260.0,
#
#     # rotate in the ground plane
#     CUT_ANGLE_DEG=0.0,
#     CUT_SLAB_HALF_WIDTH=10.0,
#     CUT_KEEP_POSITIVE_SIDE=True,
#
#     SHOW_REMOVED_SIDE=False,
#
#     REAL_S_VOLUME=5,
#     REAL_S_PLANE=10,
#     REAL_ALPHA_VOLUME=0.10,
#     REAL_ALPHA_PLANE=0.95,
#
#     VIRTUAL_S_VOLUME=15,
#     VIRTUAL_S_PLANE=15,
#     VIRTUAL_ALPHA_VOLUME=0.2,
#     VIRTUAL_ALPHA_PLANE=0.90,
#
#     DRAW_CUT_FRAME=True,
#     FRAME_COLOR="crimson",
#     FRAME_LABEL="In silico resection",
#
#     SHOW_COLORBAR=False,
#     TITLE=None,
#     FIGSIZE=(7.5, 7.5),
#     ELEV=15,
#     AZIM=145,
#     HIDE_AXES=False,
#
# )

# ORIG_SAMPLES,GEN_DIR, ALPHA_MIN_REAL,ALPHA_MIN_VIRTUAL,ALPHA_MAX_REAL,ALPHA_MAX_VIRTUAL,ALPHA_GAMMA,ELEV,AZIM,S_REAL,S_VIRTUAL = get_3b_paras()
# plot_virtual_3d(GENE = "layer_guess_reordered", CMAP='tab10',SHOW_SLICE_FRAME=False)
# plot_virtual_3d(GENE = "PCP4", CMAP="Reds")


# ORIG_SAMPLES,GEN_DIR, ALPHA_MIN_REAL,ALPHA_MIN_VIRTUAL,ALPHA_MAX_REAL,ALPHA_MAX_VIRTUAL,ALPHA_GAMMA,ELEV,AZIM,S_REAL,S_VIRTUAL= get_3d_paras()
# plot_counterfactual_3d(GENE = "layer_guess_reordered", CMAP='tab10')
# plot_counterfactual_3d(GENE = "PCP4", CMAP='Reds')

#
# GEN_FILES,GENE_SET,CMAPS,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA,= get_3a3_paras()
# plot_channel_3d(GEN_FILES,GENE_SET,CMAPS,DEPTH_SCALE,X_LIM,Y_LIM,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA)