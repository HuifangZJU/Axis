from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


DEFAULT_SIMU_DIR = Path(
    "/media/huifang/data/registration/paper_analysis/Figure1/c_method/simu_data"
)
DEFAULT_OUTPUT = Path(
    "/media/huifang/data/registration/paper_analysis/Figure1/c_method/high_dimensional_view"
)

# Keep the same overall orientation convention as the method illustration.
INVERT_X = False
INVERT_Y = True
ANATOMY_OUTLINE_RGB = (43, 125, 188)


@dataclass
class ViewStyle:
    figsize: tuple = (7.8, 4.0)
    dpi: int = 260
    plane_gap: float = 10.0
    plane_alpha: float = 0.98
    anatomy_alpha: float = 0.98
    molecular_alpha: float = 0.95
    spot_radius_px: int = 12
    frame_color: str = "#5A5A5A"
    frame_alpha: float = 0.72
    frame_lw: float = 1.0
    xy_pad: float = 70.0
    z_pad_low_ratio: float = 0.05
    z_pad_high_ratio: float = 0.08
    spatial_display_scale: float = 0.8
    view_elev: float = 25.0
    view_azim: float = -60.0
    downsample_target: int = 220
    default_cmap: str = "viridis"
    # scatter scale used by mpl; independent from simulation spot radius.
    scatter_size: float = 26.0


CMAP_CYCLE = [
    "viridis",
    "plasma",
    "cividis",
    "magma",
    "inferno",
    "turbo",
    "viridis",
    "plasma",
    "cividis",
    "magma",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_rgba(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(str(path))
    if arr.ndim == 2:
        rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
        rgba[..., 0] = arr
        rgba[..., 1] = arr
        rgba[..., 2] = arr
        rgba[..., 3] = 255
        return rgba
    if arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
    return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)


def downsample_rgba(rgba: np.ndarray, target: int = 220) -> np.ndarray:
    h, w = rgba.shape[:2]
    scale = min(1.0, float(target) / float(max(h, w)))
    if scale >= 0.999:
        return rgba
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    return cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sanitize_rgba(rgba: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgba)
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32) / 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def orient_rgba_texture(rgba: np.ndarray) -> np.ndarray:
    img = np.asarray(rgba)
    if INVERT_X:
        img = img[:, ::-1].copy()
    if INVERT_Y:
        img = img[::-1, :].copy()
    return img


def orient_coords_pixels(coords: np.ndarray, canvas: int) -> np.ndarray:
    out = np.asarray(coords, dtype=np.float32).copy()
    if INVERT_X:
        out[:, 0] = float(canvas) - out[:, 0]
    if INVERT_Y:
        out[:, 1] = float(canvas) - out[:, 1]
    return out


def normalized_xy(xy: np.ndarray, canvas: int) -> tuple[np.ndarray, np.ndarray]:
    cx = cy = (canvas - 1) / 2.0
    half = canvas / 2.0
    x = (xy[:, 0] - cx) / half
    y = (xy[:, 1] - cy) / half
    return x.astype(np.float32), y.astype(np.float32)


def gaussian2d(x: np.ndarray, y: np.ndarray, cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
    return np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))


def generate_molecular_channel(spot_xy: np.ndarray, canvas: int, channel_idx: int, n_channels: int) -> np.ndarray:
    """
    Generate a family of smooth but distinct spatial molecular patterns.
    This follows the spirit of the simulation generator's molecular_values()
    but extends it to multiple channels by rotating / shifting Gaussian fields
    and adding mild directional trends.
    """
    x, y = normalized_xy(spot_xy, canvas)
    t = float(channel_idx) / max(float(n_channels - 1), 1.0)
    ang = 2.0 * np.pi * t

    p1 = (0.75 + 0.25 * np.cos(ang)) * gaussian2d(
        x, y,
        cx=-0.34 * np.cos(0.65 * ang) + 0.12 * np.sin(ang),
        cy=-0.12 + 0.22 * np.sin(0.9 * ang),
        sx=0.20 + 0.04 * np.cos(1.7 * ang),
        sy=0.17 + 0.03 * np.sin(1.3 * ang),
    )
    p2 = (0.60 + 0.35 * np.sin(ang + 0.6)) * gaussian2d(
        x, y,
        cx=0.30 * np.cos(ang + 0.5),
        cy=0.22 * np.sin(1.2 * ang + 0.4),
        sx=0.18 + 0.05 * np.sin(1.4 * ang + 0.2),
        sy=0.20 + 0.03 * np.cos(1.1 * ang),
    )
    p3 = (0.25 + 0.25 * np.cos(2.0 * ang)) * gaussian2d(
        x, y,
        cx=0.06 * np.sin(1.7 * ang),
        cy=0.05 * np.cos(1.5 * ang),
        sx=0.24,
        sy=0.21,
    )
    direction = x * np.cos(ang + 0.8) + y * np.sin(ang + 0.8)
    p4 = 0.18 * (direction + 1.0)
    p5 = 0.08 * (np.sin(4.2 * x + 2.8 * y + 1.6 * ang) + 1.0)

    z = p1 + p2 + p3 + p4 + p5
    z -= z.min()
    z /= (z.max() + 1e-12)
    return z.astype(np.float32)


def add_rgba_z_plane_pixels(
    ax,
    rgba: np.ndarray,
    z_value: float,
    *,
    canvas: int,
    target: int,
    alpha_scale: float = 1.0,
    frame_color: str = "#4A4A4A",
    frame_lw: float = 1.0,
    frame_alpha: float = 0.72,
):
    img = orient_rgba_texture(rgba)
    img = sanitize_rgba(downsample_rgba(img, target=target))

    h, w = img.shape[:2]
    X, Y = np.meshgrid(
        np.linspace(0.0, float(canvas), w, dtype=np.float32),
        np.linspace(0.0, float(canvas), h, dtype=np.float32),
    )
    Z = np.full_like(X, float(z_value), dtype=np.float32)

    face = img.copy()
    face[..., 3] = np.clip(face[..., 3] * float(np.clip(alpha_scale, 0.0, 1.0)), 0.0, 1.0)
    face = sanitize_rgba(face)

    ax.plot_surface(
        X, Y, Z,
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
    ax.plot(frame_x, frame_y, frame_z, color=frame_color, linewidth=frame_lw, alpha=frame_alpha, zorder=20)


def add_molecular_scatter_plane(
    ax,
    spot_xy: np.ndarray,
    values: np.ndarray,
    z_value: float,
    *,
    canvas: int,
    scatter_size: float,
    alpha: float,
    cmap_name: str,
    frame_color: str,
    frame_lw: float,
    frame_alpha: float,
):
    xy = orient_coords_pixels(spot_xy, canvas)
    cmap = get_cmap(cmap_name)
    colors = cmap(np.clip(values, 0.0, 1.0))
    colors[:, 3] = alpha

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        np.full(len(xy), float(z_value), dtype=np.float32),
        s=float(scatter_size),
        c=colors,
        depthshade=False,
        linewidths=0.0,
        edgecolors="none",
        zorder=10,
    )

    frame_x = np.asarray([0, canvas, canvas, 0, 0], dtype=float)
    frame_y = np.asarray([0, 0, canvas, canvas, 0], dtype=float)
    frame_z = np.full(5, float(z_value), dtype=float)
    ax.plot(frame_x, frame_y, frame_z, color=frame_color, linewidth=frame_lw, alpha=frame_alpha, zorder=20)


def style_vertical_stack(ax, z_positions: list[float], canvas: int, style: ViewStyle):
    zs = np.asarray(z_positions, dtype=float)
    z_span = max(float(np.ptp(zs)), 1.0)

    x0, x1 = -style.xy_pad, float(canvas) + style.xy_pad
    y0, y1 = -style.xy_pad, float(canvas) + style.xy_pad
    z0 = float(np.min(zs) - style.z_pad_low_ratio * z_span)
    z1 = float(np.max(zs) + style.z_pad_high_ratio * z_span)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)

    display_span = np.asarray([
        (x1 - x0) * style.spatial_display_scale,
        (y1 - y0) * style.spatial_display_scale,
        (z1 - z0),
    ], dtype=float)
    display_span /= max(float(np.max(display_span)), 1e-8)

    ax.view_init(elev=style.view_elev, azim=style.view_azim)
    ax.set_proj_type("ortho")
    ax.set_box_aspect(tuple(display_span))
    ax.set_axis_off()


def load_state(simu_dir: Path, tau_index: int):
    state_dir = simu_dir / f"tau_{tau_index:02d}"
    if not state_dir.exists():
        raise FileNotFoundError(f"State directory not found: {state_dir}")

    data = np.load(state_dir / "data.npz")
    anatomy_rgba = load_rgba(state_dir / "anatomy.png")

    return {
        "canvas": int(anatomy_rgba.shape[0]),
        "anatomy_rgba": anatomy_rgba,
        "native_spot_xy": data["native_spot_xy"].astype(np.float32),
        "tau": float(data["tau"]),
    }


def plot_highdimensional_stack(
    simu_dir: Path,
    output_dir: Path,
    *,
    tau_index: int = 1,
    n_channels: int = 5,
    save_or_show: str = "save",
    style: ViewStyle | None = None,
    multi_cmap: bool = False,
):
    style = style or ViewStyle()
    ensure_dir(output_dir)

    state = load_state(simu_dir, tau_index)
    canvas = state["canvas"]
    spot_xy = state["native_spot_xy"]

    fig = plt.figure(figsize=style.figsize)
    ax = fig.add_subplot(111, projection="3d")

    z_positions = []
    z_top = 0.0

    # Morphology plane at the top.
    add_rgba_z_plane_pixels(
        ax,
        state["anatomy_rgba"],
        z_top,
        canvas=canvas,
        target=style.downsample_target,
        alpha_scale=style.anatomy_alpha,
        frame_color=style.frame_color,
        frame_lw=style.frame_lw,
        frame_alpha=style.frame_alpha,
    )
    z_positions.append(z_top)

    # Molecular planes beneath morphology. Render as scatter only (no boundary),
    # which is substantially faster than drawing a full RGBA surface per channel.
    for ch in range(int(n_channels)):
        z = -float(ch + 1) * float(style.plane_gap)
        values = generate_molecular_channel(spot_xy, canvas, ch, n_channels)
        cmap_name = CMAP_CYCLE[ch % len(CMAP_CYCLE)] if multi_cmap else style.default_cmap
        add_molecular_scatter_plane(
            ax,
            spot_xy,
            values,
            z,
            canvas=canvas,
            scatter_size=style.scatter_size,
            alpha=style.molecular_alpha,
            cmap_name=cmap_name,
            frame_color=style.frame_color,
            frame_lw=style.frame_lw,
            frame_alpha=style.frame_alpha,
        )
        z_positions.append(z)

    style_vertical_stack(ax, z_positions, canvas, style)

    out_path = output_dir / f"highdimensional_stack_tau{tau_index:02d}_{n_channels}channels.png"
    if save_or_show == "show":
        plt.show()
    else:
        fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print(f"Saved figure to: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AXIS high-dimensional stacked view")
    p.add_argument("--simu-dir", type=Path, default=DEFAULT_SIMU_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--tau-index", type=int, default=1)
    p.add_argument("--num-channels", type=int, default=5)
    p.add_argument("--mode", type=str, default="show", choices=["save", "show"])
    p.add_argument("--plane-gap", type=float, default=110.0)
    p.add_argument("--view-elev", type=float, default=25.0)
    p.add_argument("--view-azim", type=float, default=-60.0)
    p.add_argument("--spatial-display-scale", type=float, default=0.24)
    p.add_argument("--xy-pad", type=float, default=70.0)
    p.add_argument("--cmap", type=str, default="viridis")
    p.add_argument("--multi-cmap", action="store_true", default=False)
    p.add_argument("--scatter-size", type=float, default=26.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    style = ViewStyle(
        plane_gap=args.plane_gap,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
        spatial_display_scale=args.spatial_display_scale,
        xy_pad=args.xy_pad,
        default_cmap=args.cmap,
        scatter_size=args.scatter_size,
    )
    plot_highdimensional_stack(
        args.simu_dir,
        args.output,
        tau_index=args.tau_index,
        n_channels=args.num_channels,
        save_or_show=args.mode,
        style=style,
        multi_cmap=args.multi_cmap,
    )


if __name__ == "__main__":
    main()
