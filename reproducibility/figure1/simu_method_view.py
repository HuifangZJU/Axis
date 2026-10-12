from __future__ import annotations
from axis.settings import data_path

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.pyplot import get_cmap  # matplotlib.cm.get_cmap was removed in matplotlib 3.9
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# Global tissue orientation controls.
# Figure 3 (top-to-bottom stacked view) uses its own orientation pair.
FIG3_INVERT_X = False
FIG3_INVERT_Y = True

# Figures 4-5 (SCC-like depth view) use a separate orientation pair.
FLOW_INVERT_X = False
FLOW_INVERT_Y = False


@dataclass
class StyleConfig:
    # ----------------------------- 2D stacks -----------------------------
    stack_dx: float = -0.12
    stack_dy: float = 0.07
    frame_color: str = "#8c8c8c"
    frame_lw: float = 1.0
    virtual_frame_color: str = "#666666"
    virtual_frame_lw: float = 1.15

    # -------------------------- SCC-like 3D view -------------------------
    # Depth is the FIRST 3D axis in Figures 4-5:
    #   (biological depth, spatial x, spatial y)
    view_elev: float = 12.0
    view_azim: float = 130.0

    # Figure 3 follows the attached vertical-stack 3D control:
    # spatial coordinates are x/y and biological progression is z.
    fig3_view_elev: float = 25.0
    fig3_view_azim: float = -60.0
    fig3_stack_gap: float = 300.0
    fig3_spatial_display_scale: float = 0.25
    fig3_xy_pad: float = 70.0
    fig3_z_pad_low_ratio: float = 0.12
    fig3_z_pad_high_ratio: float = 0.10
    fig3_curve_end_lift: float = 5.0

    depth_a: float = 0.0
    depth_query: float = 0.50
    depth_b: float = 1.0


    real_frame_color: str = "#4A4A4A"
    real_frame_alpha: float = 0.72
    real_frame_lw: float = 1.05
    virtual_frame_alpha: float = 0.88

    # Real anchor rendering
    plane_alpha: float = 0.96
    overlay_spot_radius_px: int = 6
    overlay_spot_alpha: float = 0.94

    # Figure 3: pixel-level transition field (top-to-bottom view)
    field_grid_stride_px: int = 22
    field_erode_px: int = 14
    field_max_curves: int = 900
    field_line_alpha: float = 0.075
    field_line_lw: float = 1.95
    field_control_scale: float = 0.42
    field_global_bow_px: float = 95.0
    field_secondary_bow_px: float = 28.0
    field_curve_steps: int = 64
    transition_straight_frac: float = 0.15
    transition_mid1_frac: float = 0.38
    transition_mid2_frac: float = 0.68
    transition_end_straight_frac: float = 0.88
    transition_double_turn_ratio: float = 0.40
    fig3_bend_axis_x: float = 1.0
    fig3_bend_axis_y: float = -0.28
    fig3_bend_strength_px: float = 70.0
    fig3_bend_variation_px: float = 14.0
    field_front_offset: float = 0.010

    # Figures 4-5: spot-level evidence transport
    transport_max_curves: int = 320
    transport_line_alpha: float = 0.180
    transport_line_lw: float = 1.95
    transport_control_scale: float = 0.42

    anchor_point_size: float = 13.0
    anchor_point_alpha: float = 0.90
    evidence_point_scale: float = 2.20
    evidence_point_alpha: float = 0.20
    final_point_size: float = 13.0
    final_point_alpha: float = 0.35

    curve_color_a: str = "#4E79A7"
    curve_color_b: str = "#D3725B"

    # Figures 4 and 5 each use one internally consistent colormap across
    # all planes, but these do not need to match Figure 3.
    fig4_common_cmap: str = "viridis"
    fig5_common_cmap: str = "viridis"

    # Real end states in Figures 4 and 5 are rendered with transparency to
    # indicate that their evidence has been transported toward the middle plane.
    fig4_end_state_alpha: float = 0.42
    fig5_end_state_alpha: float = 0.42

    # Figure 5 vacant-template controls
    # One template is placed between real anchor A and matched state A,
    # and the other between matched state B and real anchor B.
    depth_matched_a: float = 0.3
    depth_matched_b: float = 0.7
    depth_template_a: float = 0.25
    depth_template_b: float = 0.75
    template_point_size: float = 20.0
    template_point_alpha: float = 0.62
    template_point_edge_color: str = "#777777"
    template_point_linewidth: float = 0.85
    template_frame_color: str = "#777777"
    template_frame_alpha: float = 0.72
    template_frame_lw: float = 1.10

    output_dx: float = -0.09
    output_dy: float = 0.05
    dpi: int = 300


def load_rgba(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
    elif arr.shape[-1] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=-1)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    arr = arr.astype(np.float32) / 255.0
    return np.clip(
        np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    ).astype(np.float32)


def handle_save_or_show(fig, out_path: Path, save_or_show="show", dpi=300):
    if save_or_show in ("save", "both"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=True)
    if save_or_show in ("show", "both"):
        plt.show()
    plt.close(fig)


def add_2d_image(ax, rgba, dx, dy, draw_frame=False, frame_kwargs=None):
    ax.imshow(rgba, extent=(dx, 1 + dx, dy, 1 + dy), interpolation="antialiased")
    if draw_frame:
        kwargs = dict(color="#8c8c8c", lw=1.0, ls="-")
        if frame_kwargs:
            kwargs.update(frame_kwargs)
        rect = Rectangle((dx, dy), 1, 1, fill=False, **kwargs)
        ax.add_patch(rect)


def add_ellipsis(ax, x, y, dx=0.035, color="black", radius=0.010):
    for k in range(3):
        circ = Circle((x + k * dx, y), radius=radius, facecolor=color, edgecolor="none")
        ax.add_patch(circ)



def sanitize_rgba(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB/RGBA arrays to finite float32 values in [0, 1].
    This avoids old Matplotlib 3D facecolor errors caused by tiny
    interpolation overshoots or uint8-like values.
    """
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(
            "Expected RGB/RGBA image with shape (H, W, 3/4), got {}".format(arr.shape)
        )

    arr = arr.astype(np.float32, copy=False)

    finite_max = np.nanmax(arr) if arr.size else 1.0
    if finite_max > 1.5:
        arr = arr / 255.0

    if arr.shape[-1] == 3:
        alpha = np.ones(arr.shape[:2] + (1,), dtype=np.float32)
        arr = np.concatenate([arr, alpha], axis=-1)

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32, copy=False)


def downsample_rgba(img: np.ndarray, target=180):
    img = sanitize_rgba(img)
    h, w = img.shape[:2]
    scale = target / float(max(h, w))
    nh = max(2, int(round(h * scale)))
    nw = max(2, int(round(w * scale)))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return sanitize_rgba(out)


def rgba_to_gray_u8(rgba: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgba[..., :3] * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def unit_xy(coords: np.ndarray, canvas: int, invert_x=None, invert_y=None):
    """
    Convert image-space coordinates to unit coordinates for Figures 4-5.

    Set FLOW_INVERT_X / FLOW_INVERT_Y near the top of this script to control
    the overall tissue direction in the SCC-like depth views.
    """
    if invert_x is None:
        invert_x = FLOW_INVERT_X
    if invert_y is None:
        invert_y = FLOW_INVERT_Y

    x = coords[:, 0] / float(canvas)
    y = coords[:, 1] / float(canvas)

    if invert_x:
        x = 1.0 - x
    if invert_y:
        y = 1.0 - y

    return np.column_stack([x, y])


def tissue_outline_from_mask(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(cnts, key=cv2.contourArea)
    contour = contour.reshape(-1, 2).astype(np.float32)
    contour[:, 1] = mask.shape[0] - 1 - contour[:, 1]
    return contour / float(mask.shape[0])


def style_3d_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
    except Exception:
        pass
    ax.set_axis_off()



def style_topdown_transition_axis(ax, style, z_top=1.0, z_bottom=0.0):
    """
    Styling for Figure 3 only: vertical stack (top slice above bottom slice)
    with a clean top-to-bottom 3D view.
    """
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_zlim(float(z_bottom) - 0.08, float(z_top) + 0.08)

    ax.set_box_aspect((1.0, 1.0, 1.15))
    ax.view_init(elev=style.fig3_view_elev, azim=style.fig3_view_azim)
    style_3d_axes(ax)

def add_rgba_plane(
    ax,
    rgba,
    z,
    alpha_scale=1.0,
    draw_frame=True,
    frame_color="#8c8c8c",
    frame_lw=1.0,
    frame_ls="-",
):
    """
    Draw an RGBA image as a horizontal 3D plane.

    Older Matplotlib releases require every RGBA component supplied as
    facecolors to lie strictly inside [0, 1]. We therefore sanitize the
    texture immediately before plot_surface().
    """
    img = sanitize_rgba(downsample_rgba(rgba, target=220))

    h, w = img.shape[:2]
    X, Y = np.meshgrid(
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        np.linspace(0.0, 1.0, h, dtype=np.float32),
    )
    Z = np.full_like(X, float(z), dtype=np.float32)

    face = img.copy()
    alpha_scale = float(np.clip(alpha_scale, 0.0, 1.0))
    face[..., 3] = np.clip(face[..., 3] * alpha_scale, 0.0, 1.0)
    face = sanitize_rgba(face)

    if not np.all(np.isfinite(face)):
        raise ValueError("Non-finite RGBA value detected before plot_surface().")

    fmin = float(face.min())
    fmax = float(face.max())
    if fmin < 0.0 or fmax > 1.0:
        raise ValueError(
            "RGBA texture outside [0,1]: min={}, max={}".format(fmin, fmax)
        )

    ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        facecolors=face,
        shade=False,
        linewidth=0,
        antialiased=False,
    )

    if draw_frame:
        xs = [0, 1, 1, 0, 0]
        ys = [0, 0, 1, 1, 0]
        zs = [z] * 5
        ax.plot(
            xs,
            ys,
            zs,
            color=frame_color,
            lw=frame_lw,
            ls=frame_ls,
            alpha=0.90,
        )


def add_blank_frame(ax, z, frame_color="#8c8c8c", frame_lw=1.0, frame_ls="-"):
    xs = [0, 1, 1, 0, 0]
    ys = [0, 0, 1, 1, 0]
    zs = [z] * 5
    ax.plot(xs, ys, zs, color=frame_color, lw=frame_lw, ls=frame_ls, alpha=0.90)


def add_outline(ax, outline_unit, z, color="#6a6a6a", lw=0.9, alpha=0.55):
    ax.plot(outline_unit[:, 0], outline_unit[:, 1], np.full(len(outline_unit), z),
            color=color, lw=lw, alpha=alpha)


def add_spots_3d(ax, xy_unit, values, z, size=8, alpha=0.8, cmap_name="viridis"):
    cmap = get_cmap(cmap_name)
    colors = cmap(values)
    colors[:, 3] = alpha
    ax.scatter(
        xy_unit[:, 0], xy_unit[:, 1], np.full(len(xy_unit), z),
        s=size, c=colors, depthshade=False, linewidths=0
    )


def overlay_spots_on_rgba(
    base_rgba: np.ndarray,
    spot_xy: np.ndarray,
    values: np.ndarray,
    radius_px: int = 6,
    alpha: float = 0.95,
    cmap_name: str = "viridis",
) -> np.ndarray:
    out = sanitize_rgba(base_rgba)
    canvas = np.rint(out * 255.0).astype(np.uint8)
    cmap = get_cmap(cmap_name)
    blend_alpha = float(np.clip(alpha, 0.0, 1.0))

    for (x, y), v in zip(spot_xy, values):
        color_rgba = cmap(float(np.clip(v, 0.0, 1.0)))
        color_rgb = tuple(int(round(float(c) * 255.0)) for c in color_rgba[:3])
        overlay = canvas.copy()
        cv2.circle(
            overlay,
            (int(round(float(x))), int(round(float(y)))),
            int(radius_px),
            color_rgb + (255,),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        canvas = cv2.addWeighted(
            overlay,
            blend_alpha,
            canvas,
            1.0 - blend_alpha,
            0.0,
        )

    return sanitize_rgba(canvas.astype(np.float32) / 255.0)


def sample_mask_points(mask: np.ndarray, stride_px: int = 30, erode_px: int = 18) -> np.ndarray:
    work = (mask > 0).astype(np.uint8)
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        work = cv2.erode(work, kernel, iterations=1)
    yy, xx = np.mgrid[0:mask.shape[0]:stride_px, 0:mask.shape[1]:stride_px]
    yy = yy.ravel()
    xx = xx.ravel()
    inside = work[yy, xx] > 0
    pts = np.column_stack([xx[inside], yy[inside]]).astype(np.float32)
    return pts


def interpolate_displacement_from_spots(src_spots: np.ndarray, dst_spots: np.ndarray,
                                        query_pts: np.ndarray, k: int = 6) -> np.ndarray:
    disp = dst_spots - src_spots
    out = np.zeros_like(query_pts, dtype=np.float32)
    for i, q in enumerate(query_pts):
        d2 = np.sum((src_spots - q[None, :]) ** 2, axis=1)
        idx = np.argsort(d2)[:k]
        w = 1.0 / (np.sqrt(d2[idx]) + 1e-6)
        w = w / (w.sum() + 1e-12)
        out[i] = q + np.sum(disp[idx] * w[:, None], axis=0)
    return out


def project_inside_mask(points: np.ndarray, mask: np.ndarray, fallback_pts: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = points.copy()
    xi = np.clip(np.rint(out[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.rint(out[:, 1]).astype(int), 0, h - 1)
    bad = mask[yi, xi] == 0
    if np.any(bad):
        fb = fallback_pts.astype(np.float32)
        for i in np.where(bad)[0]:
            d2 = np.sum((fb - out[i][None, :]) ** 2, axis=1)
            out[i] = fb[np.argmin(d2)]
    return out


def add_fancy_transition_curves(ax, pts_start_img, pts_end_img, z_start, z_end,
                                canvas: int, cmap_name="turbo",
                                line_alpha=0.20, line_lw=2.3, curve_amp_px=22.0):
    start = unit_xy(pts_start_img, canvas)
    end = unit_xy(pts_end_img, canvas)
    n = len(start)
    colors = get_cmap(cmap_name)(np.linspace(0.08, 0.92, max(n, 2)))

    t = np.linspace(0, 1, 60)
    for i in range(n):
        p = start[i]
        q = end[i]
        x0, y0 = p
        x1, y1 = q

        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)

        phase = (0.17 * i / max(1, n - 1)) + 0.3 * x0 + 0.15 * y0
        amp = (curve_amp_px / float(canvas)) * (0.85 + 0.4 * np.sin(2 * np.pi * phase))

        sign = -1.0 if x0 < 0.5 else 1.0
        z = z_start * (1 - t) + z_end * t

        bx = ((1 - t) ** 3 * x0
              + 3 * (1 - t) ** 2 * t * (xm + sign * amp * np.sin(np.pi * t))
              + 3 * (1 - t) * t ** 2 * (xm - 0.55 * sign * amp * np.sin(np.pi * t))
              + t ** 3 * x1)

        by = ((1 - t) ** 3 * y0
              + 3 * (1 - t) ** 2 * t * (ym + 0.65 * amp * np.cos(np.pi * t))
              + 3 * (1 - t) * t ** 2 * (ym - 0.35 * amp * np.cos(np.pi * t))
              + t ** 3 * y1)

        ax.plot(bx, by, z, color=colors[i], linewidth=line_lw, alpha=line_alpha)


# ============================================================================
# SCC-style 3D helpers
# ============================================================================
# The reference SCC plotting code uses biological depth as the FIRST 3D axis,
# while the two spatial coordinates occupy the second and third axes.  This
# produces much more natural plane/curve occlusion than treating depth as z.


def draw_depth_plane(
    ax,
    depth,
    *,
    edgecolor="#4A4A4A",
    linewidth=1.0,
    alpha=0.75,
    face_alpha=0.025,
    linestyle="-",
):
    x0, x1 = 0.0, 1.0
    y0, y1 = 0.0, 1.0

    verts = [[
        (depth, x0, y0),
        (depth, x1, y0),
        (depth, x1, y1),
        (depth, x0, y1),
    ]]

    poly = Poly3DCollection(
        verts,
        facecolors=(0.94, 0.94, 0.94, face_alpha),
        edgecolors=(0, 0, 0, 0),
        linewidths=0,
    )
    ax.add_collection3d(poly)

    frame = np.asarray([
        [depth, x0, y0],
        [depth, x1, y0],
        [depth, x1, y1],
        [depth, x0, y1],
        [depth, x0, y0],
    ], dtype=float)

    line, = ax.plot(
        frame[:, 0],
        frame[:, 1],
        frame[:, 2],
        color=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        zorder=1,
    )
    if linestyle == "--":
        line.set_dashes((4.0, 3.0))


def add_rgba_depth_plane(
    ax,
    rgba,
    depth,
    *,
    alpha_scale=1.0,
    draw_frame=True,
    frame_color="#4A4A4A",
    frame_lw=1.0,
    frame_alpha=0.75,
    frame_ls="-",
):
    """
    Render an RGBA tissue texture on a plane of constant biological depth.

    Coordinates follow the SCC convention:
        axis-0 = biological depth
        axis-1 = spatial x
        axis-2 = spatial y
    """
    img = sanitize_rgba(downsample_rgba(rgba, target=220))
    h, w = img.shape[:2]

    spatial_x, spatial_y = np.meshgrid(
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        np.linspace(1.0, 0.0, h, dtype=np.float32),
    )
    D = np.full_like(spatial_x, float(depth), dtype=np.float32)

    face = img.copy()
    face[..., 3] = np.clip(
        face[..., 3] * float(np.clip(alpha_scale, 0.0, 1.0)),
        0.0,
        1.0,
    )
    face = sanitize_rgba(face)

    ax.plot_surface(
        D,
        spatial_x,
        spatial_y,
        rstride=1,
        cstride=1,
        facecolors=face,
        shade=False,
        linewidth=0,
        antialiased=False,
        zorder=2,
    )

    if draw_frame:
        draw_depth_plane(
            ax,
            depth,
            edgecolor=frame_color,
            linewidth=frame_lw,
            alpha=frame_alpha,
            face_alpha=0.0,
            linestyle=frame_ls,
        )



def style_scc_topdown_axis(ax, depths, style):
    """
    Same SCC-style biological-depth geometry as v6, but viewed from a
    top-to-bottom direction for Figure 3. This preserves the correct 3D
    occlusion ordering while only changing the viewing direction.
    """
    d0 = float(min(depths))
    d1 = float(max(depths))
    dspan = max(d1 - d0, 1e-3)

    ax.set_xlim(d0 - 0.09 * dspan, d1 + 0.09 * dspan)
    ax.set_ylim(1.055, -0.055)
    ax.set_zlim(1.055, -0.055)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)

    ax.set_proj_type("ortho")
    ax.view_init(elev=style.fig3_view_elev, azim=style.fig3_view_azim)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor((0.82, 0.82, 0.82, 0.18))
        except Exception:
            pass

    ax.set_box_aspect((1.25 * dspan, 1.0, 1.0))
    ax.set_axis_off()


def style_scc_3d_axis(ax, depths, style):
    d0 = float(min(depths))
    d1 = float(max(depths))
    dspan = max(d1 - d0, 1e-3)

    ax.set_xlim(d0 - 0.09 * dspan, d1 + 0.09 * dspan)
    ax.set_ylim(1.055, -0.055)
    ax.set_zlim(1.055, -0.055)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)

    ax.set_proj_type("ortho")
    ax.view_init(elev=style.view_elev, azim=style.view_azim)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor((0.82, 0.82, 0.82, 0.18))
        except Exception:
            pass

    ax.set_box_aspect((2.2 * dspan, 1.0, 1.0))
    ax.set_axis_off()


def scatter_values_depth(
    ax,
    coords_img,
    values,
    *,
    depth,
    canvas,
    point_size,
    alpha,
    cmap_name="viridis",
    zorder=5,
):
    xy = unit_xy(np.asarray(coords_img), canvas)
    vals = np.asarray(values, dtype=float)
    vals = np.clip(vals, 0.0, 1.0)

    rgba = get_cmap(cmap_name)(vals)
    rgba[:, 3] = float(alpha)

    order = np.argsort(vals)

    ax.scatter(
        np.full(len(xy), float(depth))[order],
        xy[order, 0],
        xy[order, 1],
        c=rgba[order],
        s=float(point_size),
        linewidths=0,
        depthshade=False,
        rasterized=True,
        zorder=zorder,
    )



def scatter_hollow_template_depth(
    ax,
    coords_img,
    *,
    depth,
    canvas,
    point_size,
    edge_color="#777777",
    linewidth=0.85,
    alpha=0.62,
    zorder=10,
):
    """
    Draw a vacant spatial template using hollow circles only.

    The template carries geometry/preset anatomy but no molecular intensity.
    Coordinates use the same FLOW_INVERT_X / FLOW_INVERT_Y convention as
    Figures 4-5.
    """
    xy = unit_xy(
        np.asarray(coords_img),
        canvas,
    )

    ax.scatter(
        np.full(len(xy), float(depth)),
        xy[:, 0],
        xy[:, 1],
        s=float(point_size),
        marker="o",
        facecolors="none",
        edgecolors=edge_color,
        linewidths=float(linewidth),
        alpha=float(alpha),
        depthshade=False,
        rasterized=True,
        zorder=zorder,
    )


def draw_template_to_matched_arrows(
    ax,
    *,
    depth_template,
    depth_left,
    depth_right,
    spatial_x,
    spatial_y,
    color="#666666",
    linewidth=1.45,
    alpha=0.82,
    arrow_head_ratio=0.16,
):
    """
    Draw two depth-axis arrows from the vacant middle template toward
    the two spatially matched states.

    Figures 4-5 use coordinate order:
        (biological depth, spatial x, spatial y)
    """
    # Template -> left matched state
    d_left = float(depth_left) - float(depth_template)
    ax.quiver(
        float(depth_template),
        float(spatial_x),
        float(spatial_y),
        d_left,
        0.0,
        0.0,
        color=color,
        linewidth=float(linewidth),
        alpha=float(alpha),
        arrow_length_ratio=float(arrow_head_ratio),
        normalize=False,
        zorder=30,
    )

    # Template -> right matched state
    d_right = float(depth_right) - float(depth_template)
    ax.quiver(
        float(depth_template),
        float(spatial_x),
        float(spatial_y),
        d_right,
        0.0,
        0.0,
        color=color,
        linewidth=float(linewidth),
        alpha=float(alpha),
        arrow_length_ratio=float(arrow_head_ratio),
        normalize=False,
        zorder=30,
    )


def scatter_fixed_depth(
    ax,
    coords_img,
    *,
    depth,
    canvas,
    color,
    point_size,
    alpha,
    zorder=4,
):
    xy = unit_xy(np.asarray(coords_img), canvas)
    ax.scatter(
        np.full(len(xy), float(depth)),
        xy[:, 0],
        xy[:, 1],
        c=color,
        s=float(point_size),
        alpha=float(alpha),
        linewidths=0,
        depthshade=False,
        rasterized=True,
        zorder=zorder,
    )




def chaikin_smooth_polyline(points: np.ndarray, n_iter: int = 3) -> np.ndarray:
    """Smooth a polyline using Chaikin corner cutting."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3:
        return pts
    for _ in range(int(n_iter)):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p = pts[i]
            q = pts[i + 1]
            Q = 0.75 * p + 0.25 * q
            R = 0.25 * p + 0.75 * q
            new_pts.extend([Q, R])
        new_pts.append(pts[-1])
        pts = np.asarray(new_pts, dtype=np.float32)
    return pts




def add_fading_curve_collection(
    ax,
    curves_xyz,
    curve_colors,
    *,
    linewidth,
    alpha_start,
    alpha_end,
    zorder=10,
):
    """
    Efficiently draw MANY fading 3D curves as one Line3DCollection.

    This replaces the very slow implementation that called ax.plot() for
    every short segment of every curve. Per-segment alpha fading is preserved,
    but Matplotlib only manages one collection per curve family.
    """
    all_segments = []
    all_colors = []

    for curve, base_color in zip(curves_xyz, curve_colors):
        curve = np.asarray(curve, dtype=np.float32)

        if len(curve) < 2:
            continue

        segments = np.stack(
            [curve[:-1], curve[1:]],
            axis=1,
        )

        n_seg = len(segments)

        rgba = np.tile(
            np.asarray(mcolors.to_rgba(base_color), dtype=np.float32),
            (n_seg, 1),
        )

        rgba[:, 3] = np.linspace(
            float(alpha_start),
            float(alpha_end),
            n_seg,
            dtype=np.float32,
        )

        all_segments.append(segments)
        all_colors.append(rgba)

    if not all_segments:
        return None

    all_segments = np.concatenate(
        all_segments,
        axis=0,
    )

    all_colors = np.concatenate(
        all_colors,
        axis=0,
    )

    collection = Line3DCollection(
        all_segments,
        colors=all_colors,
        linewidths=float(linewidth),
        zorder=zorder,
    )

    # Rasterization is useful when exporting dense fields.
    try:
        collection.set_rasterized(True)
    except Exception:
        pass

    ax.add_collection3d(collection)

    return collection


def plot_fading_curve_3d(
    ax,
    xs,
    ys,
    zs,
    *,
    color,
    linewidth,
    alpha_start,
    alpha_end,
    zorder=10,
):
    """
    Backward-compatible wrapper for a single fading curve.
    """
    curve = np.column_stack(
        [
            np.asarray(xs, dtype=float),
            np.asarray(ys, dtype=float),
            np.asarray(zs, dtype=float),
        ]
    )

    return add_fading_curve_collection(
        ax,
        [curve],
        [color],
        linewidth=linewidth,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        zorder=zorder,
    )

def choose_subset_indices(n_total, n_keep, seed=0):
    if n_total <= n_keep:
        return np.arange(n_total)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_total, size=int(n_keep), replace=False))
def build_consistent_transition_curve_xy(
    p0,
    p3,
    *,
    family_sign,
    family_strength,
    straight_frac,
    mid1_frac,
    mid2_frac,
    end_straight_frac,
    double_turn_ratio,
    lateral_offset=0.0,
):
    """
    Build a smooth, shared family of 2D transition curves.

    Shape logic:
      1) leave the tissue plane almost straight,
      2) make a first coordinated turn in the middle,
      3) make a second coordinated compensatory turn,
      4) approach the destination nearly straight.

    All curves in one family share the same turn direction and the same turn
    schedule. Only the amplitude changes gradually across the bundle.
    """
    p0 = np.asarray(p0, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    disp = p3 - p0
    dist = float(np.linalg.norm(disp))
    if dist <= 1e-8:
        return np.repeat(p0[None, :], 12, axis=0).astype(np.float32)

    unit = disp / dist
    perp = np.array([-unit[1], unit[0]], dtype=float)

    # Keep the beginning and end nearly straight.
    n1 = p0 + straight_frac * disp
    n4 = p0 + end_straight_frac * disp

    # Shared family bow. The sign is fixed for the whole family so the curves
    # bend together in the same way.
    bow = float(family_sign) * float(family_strength + lateral_offset)

    # Two coordinated turns.
    n2 = p0 + mid1_frac * disp + bow * perp
    n3 = p0 + mid2_frac * disp - (double_turn_ratio * bow) * perp

    # Small shared sweep so the bundle looks like a smooth transport sheet.
    sweep = np.array([0.02 * bow, -0.08 * abs(bow)], dtype=float)
    n2 = n2 + 0.85 * sweep
    n3 = n3 + 0.35 * sweep

    poly = np.vstack([p0, n1, n2, n3, n4, p3]).astype(np.float32)
    curve = chaikin_smooth_polyline(poly, n_iter=3)
    return curve


def add_transport_curves_depth(
    ax,
    start_coords_img,
    end_coords_img,
    *,
    depth_start,
    depth_end,
    canvas,
    max_curves,
    seed,
    color=None,
    cmap=None,
    linewidth=1.8,
    alpha=0.11,
    control_scale=0.34,
    front_offset_end=0.0,
    family_sign=-1.0,
    fade_to_end=True,
    zorder=9,
):
    """
    Efficient SCC-style transport curves using a shared double-turn family.

    All curves from this call are assembled first and rendered as ONE
    Line3DCollection.
    """
    start = unit_xy(
        np.asarray(start_coords_img),
        canvas,
    )
    end = unit_xy(
        np.asarray(end_coords_img),
        canvas,
    )

    n = min(
        len(start),
        len(end),
    )

    start = start[:n]
    end = end[:n]

    keep = choose_subset_indices(
        n,
        max_curves,
        seed=seed,
    )

    start = start[keep]
    end = end[keep]

    d_end = (
        float(depth_end)
        + float(front_offset_end)
    )

    if cmap is not None:
        cm = (
            get_cmap(cmap)
            if isinstance(cmap, str)
            else cmap
        )

        colors = cm(
            np.linspace(
                0.06,
                0.94,
                max(len(start), 2),
            )
        )
    else:
        rgba = mcolors.to_rgba(
            color
            if color is not None
            else "#7F7F7F"
        )

        colors = [
            rgba
        ] * len(start)

    center_x = 0.50

    curves_xyz = []
    curve_colors = []

    for i, (p0, p1) in enumerate(
        zip(start, end)
    ):
        lateral = (
            0.5
            * (p0[0] + p1[0])
            - center_x
        ) / max(
            center_x,
            1e-8,
        )

        dist = float(
            np.linalg.norm(
                p1 - p0
            )
        )

        # Gradual amplitude variation only.
        base_amp = (
            0.050
            + 0.11
            * control_scale
            * min(
                dist / 0.8,
                1.0,
            )
        )

        lateral_offset = (
            0.030
            * lateral
        )

        curve_xy = build_consistent_transition_curve_xy(
            p0,
            p1,
            family_sign=family_sign,
            family_strength=base_amp,
            straight_frac=0.14,
            mid1_frac=0.38,
            mid2_frac=0.70,
            end_straight_frac=0.88,
            double_turn_ratio=0.40,
            lateral_offset=lateral_offset,
        )

        depths = np.linspace(
            float(depth_start),
            d_end,
            len(curve_xy),
            dtype=np.float32,
        )

        # SCC coordinate order:
        # biological depth, spatial x, spatial y
        curve_xyz = np.column_stack(
            [
                depths,
                curve_xy[:, 0],
                curve_xy[:, 1],
            ]
        )

        curves_xyz.append(
            curve_xyz
        )
        curve_colors.append(
            colors[i]
        )

    if fade_to_end:
        alpha_start = float(alpha)
        alpha_end = (
            0.18
            * float(alpha)
        )
    else:
        alpha_start = (
            0.18
            * float(alpha)
        )
        alpha_end = float(alpha)

    add_fading_curve_collection(
        ax,
        curves_xyz,
        curve_colors,
        linewidth=float(linewidth),
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        zorder=zorder,
    )




def eroded_mask(mask, erode_px):
    m = (mask > 0).astype(np.uint8)
    if int(erode_px) <= 0:
        return m
    k = 2 * int(erode_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(m, kernel, iterations=1)


def sample_dense_correspondence(
    src_mask,
    dst_mask,
    src_spots,
    dst_spots,
    *,
    stride_px,
    erode_px,
    max_points,
):
    """
    Pixel-grid transition samples whose start AND end lie inside their
    respective tissue supports.  Invalid mapped endpoints are discarded
    rather than projected to a boundary, preventing the artificial contour
    seen in earlier versions.
    """
    src_inner = eroded_mask(src_mask, erode_px)
    dst_inner = eroded_mask(dst_mask, erode_px)

    yy, xx = np.mgrid[
        0:src_mask.shape[0]:int(stride_px),
        0:src_mask.shape[1]:int(stride_px),
    ]
    yy = yy.ravel()
    xx = xx.ravel()

    inside = src_inner[yy, xx] > 0
    q = np.column_stack([xx[inside], yy[inside]]).astype(np.float32)

    if len(q) == 0:
        return q, q.copy()

    mapped = interpolate_displacement_from_spots(
        np.asarray(src_spots, dtype=np.float32),
        np.asarray(dst_spots, dtype=np.float32),
        q,
        k=7,
    )

    h, w = dst_mask.shape
    xi = np.clip(np.rint(mapped[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.rint(mapped[:, 1]).astype(int), 0, h - 1)

    valid = (
        (mapped[:, 0] >= 0)
        & (mapped[:, 0] < w)
        & (mapped[:, 1] >= 0)
        & (mapped[:, 1] < h)
        & (dst_inner[yi, xi] > 0)
    )

    q = q[valid]
    mapped = mapped[valid]

    if len(q) > int(max_points):
        keep = choose_subset_indices(len(q), int(max_points), seed=11)
        q = q[keep]
        mapped = mapped[keep]

    return q, mapped


def fancy_axis_field_cmap():
    # Deliberately restrained blue -> teal -> violet -> coral gradient.
    return LinearSegmentedColormap.from_list(
        "axis_transition",
        ["#4E79A7", "#76B7B2", "#9C8AC7", "#D98B78"],
        N=256,
    )


# ============================================================================
# Figure-3 helpers: vertical stack in (spatial x, spatial y, progression z)
# ============================================================================

def orient_coords_pixels(coords: np.ndarray, canvas: int) -> np.ndarray:
    """Apply the Figure-3-specific x/y inversion controls in pixel coordinates."""
    out = np.asarray(coords, dtype=np.float32).copy()
    if FIG3_INVERT_X:
        out[:, 0] = float(canvas) - out[:, 0]
    if FIG3_INVERT_Y:
        out[:, 1] = float(canvas) - out[:, 1]
    return out


def orient_rgba_texture(rgba: np.ndarray) -> np.ndarray:
    """Apply the Figure-3-specific orientation transform to an image texture."""
    out = sanitize_rgba(rgba)
    if FIG3_INVERT_X:
        out = np.flip(out, axis=1)
    if FIG3_INVERT_Y:
        out = np.flip(out, axis=0)
    return np.ascontiguousarray(out)


def add_rgba_z_plane_pixels(
    ax,
    rgba,
    z_value,
    *,
    canvas,
    alpha_scale=1.0,
    frame_color="#4A4A4A",
    frame_lw=1.05,
    frame_alpha=0.72,
):
    """
    Draw an image plane at constant z using the SAME coordinate convention as
    the attached vertical-stack code: x/y are spatial axes; z is progression.
    """
    img = orient_rgba_texture(rgba)
    img = sanitize_rgba(downsample_rgba(img, target=230))

    h, w = img.shape[:2]
    X, Y = np.meshgrid(
        np.linspace(0.0, float(canvas), w, dtype=np.float32),
        np.linspace(0.0, float(canvas), h, dtype=np.float32),
    )
    Z = np.full_like(X, float(z_value), dtype=np.float32)

    face = img.copy()
    face[..., 3] = np.clip(
        face[..., 3] * float(np.clip(alpha_scale, 0.0, 1.0)),
        0.0,
        1.0,
    )
    face = sanitize_rgba(face)

    ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        facecolors=face,
        shade=False,
        linewidth=0,
        antialiased=False,
        zorder=3,
    )

    frame_x = np.asarray([0, canvas, canvas, 0, 0], dtype=float)
    frame_y = np.asarray([0, 0, canvas, canvas, 0], dtype=float)
    frame_z = np.full(5, float(z_value), dtype=float)
    ax.plot(
        frame_x,
        frame_y,
        frame_z,
        color=frame_color,
        linewidth=frame_lw,
        alpha=frame_alpha,
        zorder=20,
    )


def style_figure3_vertical_stack(ax, z_positions, canvas, style):
    """
    Direct adaptation of the attached code's style_3d_axis():
      - horizontal tissue planes in x/y,
      - progression on z,
      - orthographic projection,
      - explicit spatial-display scaling,
      - camera controlled only by elevation/azimuth.
    """
    zs = np.asarray(z_positions, dtype=float)
    z_span = max(float(np.ptp(zs)), 1.0)

    x_pad = float(style.fig3_xy_pad)
    y_pad = float(style.fig3_xy_pad)
    z_pad_low = float(style.fig3_z_pad_low_ratio) * z_span
    z_pad_high = float(style.fig3_z_pad_high_ratio) * z_span

    x0, x1 = -x_pad, float(canvas) + x_pad
    y0, y1 = -y_pad, float(canvas) + y_pad
    z0 = float(np.min(zs) - z_pad_low)
    z1 = float(np.max(zs) + z_pad_high)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)

    display_span = np.asarray(
        [
            (x1 - x0) * float(style.fig3_spatial_display_scale),
            (y1 - y0) * float(style.fig3_spatial_display_scale),
            (z1 - z0),
        ],
        dtype=float,
    )
    display_span /= max(float(np.max(display_span)), 1e-8)

    ax.view_init(
        elev=float(style.fig3_view_elev),
        azim=float(style.fig3_view_azim),
    )
    ax.set_proj_type("ortho")
    ax.set_box_aspect(tuple(display_span))
    ax.set_axis_off()


def add_figure3_transition_curves(
    ax,
    start_coords,
    end_coords,
    *,
    z_start,
    z_end,
    canvas,
    style,
    seed=3,
):
    """
    Efficient bidirectional pixel-level transition field for Figure 3.

    Key design update:
    all curves in one directional family bend together along the SAME global
    2D axis, so the whole field looks like a coherent flowing sheet rather
    than a set of independently bending trajectories.

    Two directional sets:
      - top -> bottom: green family
      - bottom -> top: purple family
    """
    start = orient_coords_pixels(
        start_coords,
        canvas,
    )
    end = orient_coords_pixels(
        end_coords,
        canvas,
    )

    n = min(
        len(start),
        len(end),
    )

    if n == 0:
        return

    start = start[:n]
    end = end[:n]

    if n > int(style.field_max_curves):
        keep = choose_subset_indices(
            n,
            int(style.field_max_curves),
            seed=seed,
        )
        start = start[keep]
        end = end[keep]

    cmap_top = LinearSegmentedColormap.from_list(
        "axis_topdown_green",
        [
            "#A7D9B0",
            "#81CC86",
            "#67BF63",
            "#8CCBB5",
        ],
        N=256,
    )

    cmap_bottom = LinearSegmentedColormap.from_list(
        "axis_bottomup_purple",
        [
            "#CBB8EC",
            "#A790DE",
            "#886FCF",
            "#B89FDF",
        ],
        N=256,
    )

    # Smooth color progression across x.
    order = np.argsort(start[:, 0])
    vals = np.linspace(0.15, 0.95, max(len(order), 2))

    colors_top = [None] * len(start)
    colors_bottom = [None] * len(start)

    for k, ii in enumerate(order):
        colors_top[ii] = cmap_top(vals[k])
        colors_bottom[ii] = cmap_bottom(vals[k])

    # Global shared bend axis in the image plane.
    bend_axis = np.asarray(
        [float(style.fig3_bend_axis_x), float(style.fig3_bend_axis_y)],
        dtype=np.float32,
    )
    bend_norm = np.linalg.norm(bend_axis)
    if bend_norm < 1e-8:
        bend_axis = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        bend_axis = bend_axis / bend_norm

    center_x = 0.5 * float(canvas)

    def build_family_curves(src_pts, dst_pts, family_sign, curve_colors):
        curves = []
        out_colors = []

        for i, (p0, p3) in enumerate(zip(src_pts, dst_pts)):
            p0 = p0.astype(np.float32)
            p3 = p3.astype(np.float32)

            t = np.linspace(0.0, 1.0, int(style.field_curve_steps), dtype=np.float32)

            # Base straight transport between endpoints.
            base = (1.0 - t)[:, None] * p0[None, :] + t[:, None] * p3[None, :]

            # Keep the ends nearly straight and concentrate bending in the middle.
            gate = np.sin(np.pi * t) ** 1.20

            # Two coordinated turns, but both aligned to the SAME global axis.
            bump1 = np.exp(-0.5 * ((t - 0.42) / 0.12) ** 2)
            bump2 = np.exp(-0.5 * ((t - 0.70) / 0.12) ** 2)

            # Mild gradual variation only, so all curves still read as one family.
            lateral = (0.5 * (p0[0] + p3[0]) - center_x) / max(center_x, 1.0)
            amp = float(style.fig3_bend_strength_px) + float(style.fig3_bend_variation_px) * lateral

            disp = (family_sign * amp) * (bump1 - float(style.transition_double_turn_ratio) * bump2) * gate
            curve_xy = base + disp[:, None] * bend_axis[None, :]

            # Cosine easing in z keeps a smooth suspended look.
            z_curve = float(z_start) + (float(z_end) - float(z_start)) * (0.5 - 0.5 * np.cos(np.pi * t))
            if family_sign > 0:
                z_curve = float(z_end) + (float(z_start) - float(z_end)) * (0.5 - 0.5 * np.cos(np.pi * t))

            curves.append(
                np.column_stack(
                    [
                        curve_xy[:, 0],
                        curve_xy[:, 1],
                        z_curve,
                    ]
                )
            )
            out_colors.append(curve_colors[i])

        return curves, out_colors

    # Family 1: top -> bottom. All curves bend together in one shared direction.
    top_curves, top_colors = build_family_curves(
        start,
        end,
        family_sign=-1.0,
        curve_colors=colors_top,
    )

    add_fading_curve_collection(
        ax,
        top_curves,
        top_colors,
        linewidth=float(style.field_line_lw),
        alpha_start=float(style.field_line_alpha),
        alpha_end=0.12 * float(style.field_line_alpha),
        zorder=12,
    )

    # Family 2: bottom -> top. Same geometric family, opposite direction set.
    bottom_curves, bottom_colors = build_family_curves(
        end,
        start,
        family_sign=+1.0,
        curve_colors=colors_bottom,
    )

    add_fading_curve_collection(
        ax,
        bottom_curves,
        bottom_colors,
        linewidth=float(style.field_line_lw),
        alpha_start=float(style.field_line_alpha),
        alpha_end=0.12 * float(style.field_line_alpha),
        zorder=11,
    )




class AxisSimData:
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        if self.data_root.name != "simu_data" and (self.data_root / "simu_data").exists():
            self.data_root = self.data_root / "simu_data"

        self.native_dirs = [self.data_root / f"tau_{i:02d}" for i in range(1, 6)]
        self.matched_root = self.data_root / "matched_states"
        self.canvas = 1000

        self.native = [self._load_native(d) for d in self.native_dirs]
        self.matched = [self._load_matched(self.matched_root / f"tau_{i:02d}") for i in range(1, 6)]
        self.shared_domain = load_rgba(self.matched_root / "shared_domain.png")
        self.shared_domain_boundaries = load_rgba(self.matched_root / "shared_domain_boundaries.png")

        shared_npz = np.load(self.matched_root / "shared_domain_data.npz")
        self.shared_xy = shared_npz["matched_spot_xy"]
        self.shared_labels = shared_npz["domain_labels_at_spots"]

    def _load_native(self, d: Path):
        rec = {}
        rec["anatomy_png"] = load_rgba(d / "anatomy.png")
        rec["molecular_png"] = load_rgba(d / "molecular.png")
        rec["pseudo_png"] = load_rgba(d / "pseudo_omics.png")
        rec["domain_png"] = load_rgba(d / "domain.png")
        rec["boundary_png"] = load_rgba(d / "domain_boundaries.png")
        npz = np.load(d / "data.npz")
        for k in npz.files:
            rec[k] = npz[k]
        return rec

    def _load_matched(self, d: Path):
        rec = {}
        rec["molecular_png"] = load_rgba(d / "molecular.png")
        npz = np.load(d / "data.npz")
        for k in npz.files:
            rec[k] = npz[k]
        return rec


class AxisMethodIllustrator:
    def __init__(self, data_root: Path, out_dir: Path, style: StyleConfig | None = None):
        self.data = AxisSimData(data_root)
        self.out_dir = Path(out_dir)
        self.style = style or StyleConfig()

    def figure1_native_discrete_stack(self, save_or_show="show"):
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.2))
        states = [self.data.native[0], self.data.native[4]]
        titles = ["Pseudo time 01", "Pseudo time 05"]
        for ax, st, title in zip(axes, states, titles):
            ax.set_aspect("equal")
            ax.axis("off")
            add_2d_image(ax, st["molecular_png"], 2*self.style.stack_dx, 2*self.style.stack_dy)
            add_2d_image(ax, st["molecular_png"], 1*self.style.stack_dx, 1*self.style.stack_dy)
            add_2d_image(ax, st["anatomy_png"], 0, 0)
            add_ellipsis(ax, -0.04, 0.08, dx=0.040, color="black", radius=0.012)
            ax.set_xlim(-0.35, 1.05)
            ax.set_ylim(-0.05, 1.25)
            ax.set_title(title, fontsize=12)
        plt.tight_layout()
        handle_save_or_show(fig, self.out_dir / "figure1_native_discrete_stack.png", save_or_show, self.style.dpi)

    def figure2_native_pseudo_stack(self, save_or_show="show"):
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.2))
        states = [self.data.native[0], self.data.native[4]]
        titles = ["Pseudo time 01", "Pseudo time 05"]
        for ax, st, title in zip(axes, states, titles):
            ax.set_aspect("equal")
            ax.axis("off")
            add_2d_image(ax, st["pseudo_png"], 2*self.style.stack_dx, 2*self.style.stack_dy, True,
                         dict(color=self.style.frame_color, lw=self.style.frame_lw, ls="-"))
            add_2d_image(ax, st["pseudo_png"], 1*self.style.stack_dx, 1*self.style.stack_dy, True,
                         dict(color=self.style.frame_color, lw=self.style.frame_lw, ls="-"))
            add_2d_image(ax, st["anatomy_png"], 0, 0, True,
                         dict(color=self.style.frame_color, lw=self.style.frame_lw, ls="-"))
            add_ellipsis(ax, -0.04, 0.08, dx=0.040, color="black", radius=0.012)
            ax.set_xlim(-0.35, 1.05)
            ax.set_ylim(-0.05, 1.25)
            ax.set_title(title, fontsize=12)
        plt.tight_layout()
        handle_save_or_show(fig, self.out_dir / "figure2_native_pseudo_stack.png", save_or_show, self.style.dpi)

    def figure3_transition_field_3d(self, save_or_show="show"):
        """
        Pixel-level transition field between pseudo time 01 and 05.

        Figure 3 uses the SAME 3D control as the attached vertical-stack
        script: spatial x/y stay horizontal and pseudo-time progression is the
        z axis. Only the simulated AXIS transition field is added between the
        two horizontal tissue planes.
        """
        a = self.data.native[0]
        b = self.data.native[4]

        fig = plt.figure(figsize=(8.2, 8.2))
        ax = fig.add_subplot(111, projection="3d")

        z_top = 0.0
        z_bottom = -float(self.style.fig3_stack_gap)

        tex_a = overlay_spots_on_rgba(
            a["anatomy_png"],
            a["native_spot_xy"],
            a["molecular_read"],
            radius_px=self.style.overlay_spot_radius_px,
            alpha=self.style.overlay_spot_alpha,
        )
        tex_b = overlay_spots_on_rgba(
            b["anatomy_png"],
            b["native_spot_xy"],
            b["molecular_read"],
            radius_px=self.style.overlay_spot_radius_px,
            alpha=self.style.overlay_spot_alpha,
        )

        add_rgba_z_plane_pixels(
            ax,
            tex_a,
            z_top,
            canvas=self.data.canvas,
            alpha_scale=self.style.plane_alpha,
            frame_color=self.style.real_frame_color,
            frame_lw=self.style.real_frame_lw,
            frame_alpha=self.style.real_frame_alpha,
        )
        add_rgba_z_plane_pixels(
            ax,
            tex_b,
            z_bottom,
            canvas=self.data.canvas,
            alpha_scale=self.style.plane_alpha,
            frame_color=self.style.real_frame_color,
            frame_lw=self.style.real_frame_lw,
            frame_alpha=self.style.real_frame_alpha,
        )

        q0, q1 = sample_dense_correspondence(
            a["tissue_mask"],
            b["tissue_mask"],
            a["native_spot_xy"],
            b["native_spot_xy"],
            stride_px=self.style.field_grid_stride_px,
            erode_px=self.style.field_erode_px,
            max_points=self.style.field_max_curves,
        )

        add_figure3_transition_curves(
            ax,
            q0,
            q1,
            z_start=z_top - 2.0,
            z_end=z_bottom + float(self.style.fig3_curve_end_lift),
            canvas=self.data.canvas,
            style=self.style,
            seed=3,
        )

        style_figure3_vertical_stack(
            ax,
            [z_top, z_bottom],
            self.data.canvas,
            self.style,
        )

        handle_save_or_show(
            fig,
            self.out_dir / "figure3_transition_field_3d.png",
            save_or_show,
            self.style.dpi,
        )

    def figure4_virtual_generation_3d(self, save_or_show="show"):
        """
        Virtual generation at pseudo time 03.

        The middle query plane contains:
            - transported evidence from time 01 and time 05:
              larger, highly transparent spots;
            - final time-03 reads:
              normal-size opaque spots overlaid on the evidence.

        Transport curves follow the SCC evidence-transport style.
        """
        a = self.data.native[0]
        q = self.data.native[2]
        b = self.data.native[4]

        fig = plt.figure(figsize=(8.0, 7.8))
        ax = fig.add_subplot(111, projection="3d")

        da = self.style.depth_a
        dq = self.style.depth_query
        db = self.style.depth_b

        # Slice planes.
        draw_depth_plane(
            ax, da,
            edgecolor=self.style.real_frame_color,
            linewidth=self.style.real_frame_lw,
            alpha=self.style.real_frame_alpha,
            face_alpha=0.018,
            linestyle="-",
        )
        draw_depth_plane(
            ax, dq,
            edgecolor=self.style.virtual_frame_color,
            linewidth=self.style.virtual_frame_lw,
            alpha=self.style.virtual_frame_alpha,
            face_alpha=0.010,
            linestyle="--",
        )
        draw_depth_plane(
            ax, db,
            edgecolor=self.style.real_frame_color,
            linewidth=self.style.real_frame_lw,
            alpha=self.style.real_frame_alpha,
            face_alpha=0.018,
            linestyle="-",
        )

        # Real anchor reads.
        scatter_values_depth(
            ax, a["native_spot_xy"], a["molecular_read"],
            depth=da, canvas=self.data.canvas,
            point_size=self.style.anchor_point_size,
            alpha=self.style.fig4_end_state_alpha,
            cmap_name=self.style.fig4_common_cmap,
            zorder=5,
        )
        scatter_values_depth(
            ax, b["native_spot_xy"], b["molecular_read"],
            depth=db, canvas=self.data.canvas,
            point_size=self.style.anchor_point_size,
            alpha=self.style.fig4_end_state_alpha,
            cmap_name=self.style.fig4_common_cmap,
            zorder=5,
        )

        # Curves first, so the evidence/final spots form the visible landing plane.
        add_transport_curves_depth(
            ax,
            a["native_spot_xy"],
            q["native_spot_xy"],
            depth_start=da,
            depth_end=dq,
            canvas=self.data.canvas,
            max_curves=self.style.transport_max_curves,
            seed=7,
            color=self.style.curve_color_a,
            linewidth=self.style.transport_line_lw,
            alpha=self.style.transport_line_alpha,
            control_scale=self.style.transport_control_scale,
            front_offset_end=-0.004,
            family_sign=-1.0,
            fade_to_end=True,
            zorder=7,
        )
        add_transport_curves_depth(
            ax,
            b["native_spot_xy"],
            q["native_spot_xy"],
            depth_start=db,
            depth_end=dq,
            canvas=self.data.canvas,
            max_curves=self.style.transport_max_curves,
            seed=8,
            color=self.style.curve_color_b,
            linewidth=self.style.transport_line_lw,
            alpha=self.style.transport_line_alpha,
            control_scale=self.style.transport_control_scale,
            front_offset_end=0.004,
            family_sign=+1.0,
            fade_to_end=True,
            zorder=7,
        )

        # Evidence: two large transparent projected measurements.
        scatter_values_depth(
            ax,
            q["native_spot_xy"],
            a["molecular_read"],
            depth=dq - 0.002,
            canvas=self.data.canvas,
            point_size=self.style.anchor_point_size * self.style.evidence_point_scale,
            alpha=self.style.evidence_point_alpha,
            cmap_name=self.style.fig4_common_cmap,
            zorder=8,
        )
        scatter_values_depth(
            ax,
            q["native_spot_xy"],
            b["molecular_read"],
            depth=dq + 0.002,
            canvas=self.data.canvas,
            point_size=self.style.anchor_point_size * self.style.evidence_point_scale,
            alpha=self.style.evidence_point_alpha,
            cmap_name=self.style.fig4_common_cmap,
            zorder=8,
        )

        # Final virtual tissue overlays the evidence plane.
        scatter_values_depth(
            ax,
            q["native_spot_xy"],
            q["molecular_read"],
            depth=dq - 0.006,
            canvas=self.data.canvas,
            point_size=self.style.final_point_size,
            alpha=self.style.final_point_alpha,
            cmap_name=self.style.fig4_common_cmap,
            zorder=12,
        )

        style_scc_3d_axis(ax, [da, dq, db], self.style)

        handle_save_or_show(
            fig,
            self.out_dir / "figure4_virtual_generation_3d.png",
            save_or_show,
            self.style.dpi,
        )

    def figure5_matched_analysis_3d(self, save_or_show="show"):
        """
        Spatially matched analysis.

        Updated layout:
          - the central-arrow design is removed;
          - two vacant templates are shown;
          - one template is placed between real anchor A and matched state A;
          - the other template is placed between matched state B and real anchor B.

        Each template uses the shared/reference spatial geometry and is rendered
        as hollow circles only, with no molecular fill.
        """
        a = self.data.native[0]
        b = self.data.native[4]
        ma = self.data.matched[0]
        mb = self.data.matched[4]

        fig = plt.figure(figsize=(8.0, 7.8))
        ax = fig.add_subplot(111, projection="3d")

        da = self.style.depth_a
        dta = self.style.depth_template_a
        dma = self.style.depth_matched_a
        dmb = self.style.depth_matched_b
        dtb = self.style.depth_template_b
        db = self.style.depth_b

        # Real + template + matched planes.
        draw_depth_plane(
            ax, da,
            edgecolor=self.style.real_frame_color,
            linewidth=self.style.real_frame_lw,
            alpha=self.style.real_frame_alpha,
            face_alpha=0.018,
            linestyle="-",
        )
        draw_depth_plane(
            ax, dta,
            edgecolor=self.style.template_frame_color,
            linewidth=self.style.template_frame_lw,
            alpha=self.style.template_frame_alpha,
            face_alpha=0.0,
            linestyle="--",
        )
        draw_depth_plane(
            ax, dma,
            edgecolor=self.style.virtual_frame_color,
            linewidth=self.style.virtual_frame_lw,
            alpha=self.style.virtual_frame_alpha,
            face_alpha=0.008,
            linestyle="--",
        )
        draw_depth_plane(
            ax, dmb,
            edgecolor=self.style.virtual_frame_color,
            linewidth=self.style.virtual_frame_lw,
            alpha=self.style.virtual_frame_alpha,
            face_alpha=0.008,
            linestyle="--",
        )
        draw_depth_plane(
            ax, dtb,
            edgecolor=self.style.template_frame_color,
            linewidth=self.style.template_frame_lw,
            alpha=self.style.template_frame_alpha,
            face_alpha=0.0,
            linestyle="--",
        )
        draw_depth_plane(
            ax, db,
            edgecolor=self.style.real_frame_color,
            linewidth=self.style.real_frame_lw,
            alpha=self.style.real_frame_alpha,
            face_alpha=0.018,
            linestyle="-",
        )

        # Real anchor observations.
        scatter_values_depth(
            ax, a["native_spot_xy"], a["molecular_read"],
            depth=da, canvas=self.data.canvas,
            point_size=self.style.anchor_point_size,
            alpha=self.style.fig5_end_state_alpha,
            cmap_name=self.style.fig5_common_cmap,
            zorder=5,
        )
        scatter_values_depth(
            ax, b["native_spot_xy"], b["molecular_read"],
            depth=db, canvas=self.data.canvas,
            point_size=self.style.anchor_point_size,
            alpha=self.style.fig5_end_state_alpha,
            cmap_name=self.style.fig5_common_cmap,
            zorder=5,
        )

        # Separate transport to the two matched states.
        add_transport_curves_depth(
            ax,
            a["native_spot_xy"],
            self.data.shared_xy,
            depth_start=da,
            depth_end=dma,
            canvas=self.data.canvas,
            max_curves=self.style.transport_max_curves,
            seed=17,
            color=self.style.curve_color_a,
            linewidth=self.style.transport_line_lw,
            alpha=self.style.transport_line_alpha,
            control_scale=self.style.transport_control_scale,
            front_offset_end=-0.004,
            zorder=7,
        )
        add_transport_curves_depth(
            ax,
            b["native_spot_xy"],
            self.data.shared_xy,
            depth_start=db,
            depth_end=dmb,
            canvas=self.data.canvas,
            max_curves=self.style.transport_max_curves,
            seed=18,
            color=self.style.curve_color_b,
            linewidth=self.style.transport_line_lw,
            alpha=self.style.transport_line_alpha,
            control_scale=self.style.transport_control_scale,
            front_offset_end=0.004,
            family_sign=+1.0,
            fade_to_end=True,
            zorder=7,
        )

        # Two vacant templates: same shared geometry, hollow circles only.
        scatter_hollow_template_depth(
            ax,
            self.data.shared_xy,
            depth=dta,
            canvas=self.data.canvas,
            point_size=self.style.template_point_size,
            edge_color=self.style.template_point_edge_color,
            linewidth=self.style.template_point_linewidth,
            alpha=self.style.template_point_alpha,
            zorder=10,
        )
        scatter_hollow_template_depth(
            ax,
            self.data.shared_xy,
            depth=dtb,
            canvas=self.data.canvas,
            point_size=self.style.template_point_size,
            edge_color=self.style.template_point_edge_color,
            linewidth=self.style.template_point_linewidth,
            alpha=self.style.template_point_alpha,
            zorder=10,
        )

        # Evidence planes: large transparent transported spots.
        scatter_values_depth(
            ax,
            self.data.shared_xy,
            a["molecular_read"],
            depth=dma - 0.002,
            canvas=self.data.canvas,
            point_size=self.style.anchor_point_size * self.style.evidence_point_scale,
            alpha=self.style.evidence_point_alpha,
            cmap_name=self.style.fig5_common_cmap,
            zorder=8,
        )
        scatter_values_depth(
            ax,
            self.data.shared_xy,
            b["molecular_read"],
            depth=dmb + 0.002,
            canvas=self.data.canvas,
            point_size=self.style.anchor_point_size * self.style.evidence_point_scale,
            alpha=self.style.evidence_point_alpha,
            cmap_name=self.style.fig5_common_cmap,
            zorder=8,
        )

        # Final matched states overlaid on evidence.
        scatter_values_depth(
            ax,
            self.data.shared_xy,
            ma["molecular_read"],
            depth=dma - 0.006,
            canvas=self.data.canvas,
            point_size=self.style.final_point_size,
            alpha=self.style.final_point_alpha,
            cmap_name=self.style.fig5_common_cmap,
            zorder=12,
        )
        scatter_values_depth(
            ax,
            self.data.shared_xy,
            mb["molecular_read"],
            depth=dmb - 0.006,
            canvas=self.data.canvas,
            point_size=self.style.final_point_size,
            alpha=self.style.final_point_alpha,
            cmap_name=self.style.fig5_common_cmap,
            zorder=12,
        )

        style_scc_3d_axis(ax, [da, dta, dma, dmb, dtb, db], self.style)

        handle_save_or_show(
            fig,
            self.out_dir / "figure5_matched_analysis_3d.png",
            save_or_show,
            self.style.dpi,
        )

    def figure6_output_summary(self, save_or_show="show"):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6))
        ax1, ax2 = axes

        ax1.set_aspect("equal")
        ax1.axis("off")
        for idx, st in enumerate(self.data.native):
            dx, dy = idx*self.style.output_dx, idx*self.style.output_dy
            add_2d_image(ax1, st["pseudo_png"], dx, dy, True,
                         dict(color=self.style.frame_color, lw=1.0, ls="-" if idx in (0,4) else "--"))
            add_2d_image(ax1, st["anatomy_png"], dx, dy)
        ax1.set_xlim(-0.05, 1.45)
        ax1.set_ylim(-0.05, 1.30)
        ax1.set_title("Virtual generation", fontsize=12)

        ax2.set_aspect("equal")
        ax2.axis("off")
        for idx, st in enumerate(self.data.matched):
            dx, dy = idx*self.style.output_dx, idx*self.style.output_dy
            add_2d_image(ax2, st["molecular_png"], dx, dy, False)
            add_2d_image(ax2, self.data.shared_domain_boundaries, dx, dy, True,
                         dict(color=self.style.frame_color, lw=1.0, ls="-" if idx in (0,4) else "--"))
        ax2.set_xlim(-0.05, 1.45)
        ax2.set_ylim(-0.05, 1.30)
        ax2.set_title("Spatially matched analysis", fontsize=12)

        plt.tight_layout()
        handle_save_or_show(fig, self.out_dir / "figure6_output_summary.png", save_or_show, self.style.dpi)

    def run(self, which="all", save_or_show="show"):
        funcs = {
            # "figure1": self.figure1_native_discrete_stack,
            # "figure2": self.figure2_native_pseudo_stack,
            # "figure3": self.figure3_transition_field_3d,
            "figure4": self.figure4_virtual_generation_3d,
            "figure5": self.figure5_matched_analysis_3d,
            "figure6": self.figure6_output_summary,
        }
        if which == "all":
            for fn in funcs.values():
                fn(save_or_show=save_or_show)
        else:
            funcs[which](save_or_show=save_or_show)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path,
                   default=Path(data_path("paper_analysis/Figure1/c_method/")))
    p.add_argument("--output-dir", type=Path,
                   default=Path(data_path("paper_analysis/Figure1/c_method/")))
    p.add_argument("--figure", type=str, default="all",
                   choices=["all", "figure1", "figure2", "figure3", "figure4", "figure5", "figure6"])
    p.add_argument("--mode", type=str, default="show", choices=["show", "save", "both"])
    return p


def main():
    args = build_parser().parse_args()
    illustrator = AxisMethodIllustrator(args.data_root, args.output_dir, StyleConfig())
    illustrator.run(which=args.figure, save_or_show=args.mode)


if __name__ == "__main__":
    main()
