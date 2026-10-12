"""Plotting helpers.

All functions draw on matplotlib axes and return the figure (or axes) so that
they can be composed and saved by the caller; nothing is shown implicitly.
Images use the canvas convention (origin top-left, y pointing down).
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "pseudoimage",
    "spatial",
    "matched",
    "registration",
    "flow",
    "axis_weights",
    "loss",
]


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _new_axes(ncols: int, size: float = 3.6, nrows: int = 1):
    plt = _plt()
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    return fig, axes.ravel()


def _finish(ax, invert: bool = True):
    ax.set_aspect("equal")
    if invert and not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _category_order(values) -> list:
    """Categories of an ``obs`` column: categorical order, else natural sort of the labels."""
    if hasattr(values, "cat"):
        return [str(c) for c in values.cat.categories]
    labels = {str(v) for v in np.asarray(values) if str(v) not in ("nan", "None")}
    return sorted(labels, key=lambda c: (0, int(c), c) if c.lstrip("-").isdigit() else (1, 0, c))


def _category_colors(palette: str, n: int):
    """``n`` colours; qualitative palettes keep their own colours so they are stable across plots."""
    plt = _plt()
    cmap = plt.get_cmap(palette)
    if not hasattr(cmap, "colors") or cmap.N < n:
        cmap = plt.get_cmap("tab20" if n <= 20 and palette == "tab10" else palette, max(n, 1))
    return [cmap(k % cmap.N) for k in range(n)]


def _to_display(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype != np.uint8 and img.max() > 1.0:
        img = img / img.max()
    return img


def pseudoimage(adata, channels: Optional[Sequence[int]] = None, uns_key: str = "axis", ncols: int = 5, cmap: str = "viridis"):
    """Show the molecular pseudo-image channels of one sample."""
    pseudo = np.asarray(adata.uns[uns_key]["pseudoimage"])
    channels = list(range(pseudo.shape[-1])) if channels is None else list(channels)
    nrows = int(np.ceil(len(channels) / ncols))
    fig, axes = _new_axes(min(ncols, len(channels)), size=2.4, nrows=nrows)
    for ax, c in zip(axes, channels):
        ax.imshow(pseudo[..., c], cmap=cmap)
        ax.set_title(f"PC {c + 1}", fontsize=9)
        _finish(ax, invert=False)
    for ax in axes[len(channels) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def spatial(
    adata,
    color: Optional[str] = None,
    *,
    spatial_key: str = "spatial",
    layer: Optional[str] = None,
    ax=None,
    s: float = 12,
    cmap: str = "magma",
    palette: str = "tab10",
    categories: Optional[Sequence[str]] = None,
    image: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    vmax: Optional[float] = None,
    mask_key: Optional[str] = "valid",
):
    """Scatter one AnnData by a gene (continuous) or an ``obs`` column (categorical).

    ``mask_key`` (``obs["valid"]`` for generated tissues) hides unsupported
    output locations. Category colours follow ``categories`` (default: the
    column's category order), so panels that share it use the same colours
    even when some categories are absent.
    """
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    xy = np.asarray(adata.obsm[spatial_key])[:, :2]
    keep = np.ones(len(xy), dtype=bool)
    if mask_key and mask_key in adata.obs:
        keep = np.asarray(adata.obs[mask_key]).astype(bool)
    if image is not None:
        ax.imshow(_to_display(image))
    if color is None:
        ax.scatter(xy[keep, 0], xy[keep, 1], s=s, c="0.3", linewidths=0)
    elif color in adata.obs and not np.issubdtype(np.asarray(adata.obs[color]).dtype, np.number):
        labels = adata.obs[color].astype(str).to_numpy()
        cats = [str(c) for c in categories] if categories is not None else _category_order(adata.obs[color])
        colors = _category_colors(palette, len(cats))
        for k, cat in enumerate(cats):
            sel = keep & (labels == cat)
            if sel.any():
                ax.scatter(xy[sel, 0], xy[sel, 1], s=s, color=colors[k], label=cat, linewidths=0)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=7,
            markerscale=1.5,
            title=color,
            title_fontsize=8,
        )
    else:
        if color in adata.obs:
            values = np.asarray(adata.obs[color], dtype=float)
        else:
            X = adata.layers[layer] if layer else adata.X
            col = X[:, adata.var_names.get_loc(color)]
            values = np.asarray(col.toarray() if hasattr(col, "toarray") else col, dtype=float).ravel()
        sc = ax.scatter(xy[keep, 0], xy[keep, 1], c=values[keep], s=s, cmap=cmap, vmax=vmax, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title(title if title is not None else (color or ""), fontsize=10)
    _finish(ax)
    return ax


def matched(
    adatas: Sequence,
    color: Optional[str] = None,
    *,
    key: str = "spatial_axis",
    before_key: str = "spatial_canvas",
    names: Optional[Sequence[str]] = None,
    s: float = 6,
):
    """Overlay all samples before (``before_key``) and after (``key``) matching.

    Points are coloured by sample, or by ``obs[color]`` with one marker per sample.
    """
    fig, axes = _new_axes(2, size=4.2)
    markers = ["o", "^", "s", "D", "v", "P", "X"]
    names = list(names) if names is not None else [f"sample {i}" for i in range(len(adatas))]
    if color is not None:  # one colour per category, shared by all samples
        cats = []
        for a in adatas:
            cats += [c for c in _category_order(a.obs[color]) if c not in cats]
        lut = dict(zip(cats, _category_colors("tab10", len(cats))))
    for ax, k, title in zip(axes, (before_key, key), ("before AXIS", "after AXIS")):
        for i, a in enumerate(adatas):
            xy = np.asarray(a.obsm[k])[:, :2]
            if color is None:
                ax.scatter(xy[:, 0], xy[:, 1], s=s, alpha=0.6, label=names[i], linewidths=0)
            else:
                labels = a.obs[color].astype(str).to_numpy()
                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=s,
                    c=[lut.get(lab, (0.6, 0.6, 0.6, 1.0)) for lab in labels],
                    marker=markers[i % len(markers)],
                    alpha=0.7,
                    linewidths=0,
                )
        ax.set_title(title, fontsize=10)
        _finish(ax)
    if color is None:
        axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8, markerscale=2)
    fig.tight_layout()
    return fig


def registration(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    field: np.ndarray,
    *,
    titles: Tuple[str, str, str] = ("fixed", "moving", "warped moving"),
):
    """Fixed, moving and warped moving images plus before/after overlays."""
    from ..registration.warp import warp_image

    warped = warp_image(moving_image, field)
    fig, axes = _new_axes(5, size=3.0)
    for ax, img, title in zip(axes[:3], (fixed_image, moving_image, warped), titles):
        ax.imshow(_to_display(img), cmap="gray")
        ax.set_title(title, fontsize=9)
        _finish(ax, invert=False)

    def overlay(a, b):
        ga = np.asarray(a, dtype=float).mean(axis=-1) if np.ndim(a) == 3 else np.asarray(a, dtype=float)
        gb = np.asarray(b, dtype=float).mean(axis=-1) if np.ndim(b) == 3 else np.asarray(b, dtype=float)
        norm = lambda x: (x - x.min()) / (np.ptp(x) + 1e-8)  # noqa: E731
        return np.stack([norm(ga), norm(gb), norm(ga)], axis=-1)

    axes[3].imshow(overlay(fixed_image, moving_image))
    axes[3].set_title("overlay before", fontsize=9)
    axes[4].imshow(overlay(fixed_image, warped))
    axes[4].set_title("overlay after", fontsize=9)
    for ax in axes[3:]:
        _finish(ax, invert=False)
    fig.tight_layout()
    return fig


def flow(field: np.ndarray, *, step: int = 24, ax=None, image: Optional[np.ndarray] = None, color: str = "tab:purple"):
    """Quiver plot of a ``(2, H, W)`` ``(dy, dx)`` transition field."""
    from ..registration.warp import ensure_flow_2hw

    plt = _plt()
    f = np.asarray(ensure_flow_2hw(np.asarray(field)))
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    if image is not None:
        ax.imshow(_to_display(image), alpha=0.6)
    H, W = f.shape[1:]
    ys, xs = np.arange(0, H, step), np.arange(0, W, step)
    X, Y = np.meshgrid(xs, ys)
    ax.quiver(X, Y, f[1][::step, ::step], f[0][::step, ::step], angles="xy", scale_units="xy", scale=1, color=color, width=0.004)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title("transition field", fontsize=10)
    _finish(ax, invert=False)
    return ax


def axis_weights(observed_positions: Sequence[float], query: float, bandwidth: Optional[float] = None, n: int = 400):
    """Normalised biological-axis weights of the anchors as a function of the query position."""
    from ..generation.evidence import compute_axis_kernel_weights, infer_axis_bandwidth

    plt = _plt()
    obs = np.asarray(observed_positions, dtype=float)
    bandwidth = infer_axis_bandwidth(obs) if bandwidth is None else bandwidth
    span = obs.max() - obs.min()
    grid = np.linspace(obs.min() - 0.3 * span, obs.max() + 0.3 * span, n)
    curves = np.asarray([compute_axis_kernel_weights(obs, t, bandwidth)[0] for t in grid])
    wq = compute_axis_kernel_weights(obs, query, bandwidth)[0]
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for i in range(len(obs)):
        ax.plot(grid, curves[:, i], lw=2, label=f"anchor @ {obs[i]:g}")
        ax.axvline(obs[i], ls="--", lw=0.8, color="0.6")
    ax.axvline(query, ls=":", lw=1.5, color="k")
    ax.scatter([query] * len(obs), wq, zorder=5, color="k", s=15)
    ax.set_xlabel("biological axis")
    ax.set_ylabel(r"anchor weight $\beta_r$")
    ax.set_title(rf"$\sigma_\tau$ = {bandwidth:.3g}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def loss(history: Mapping[str, Sequence[float]], smooth: int = 20):
    """Training curves of the loss terms in ``history`` (e.g. ``fields.history`` from [`axis.tl.fit_transition`][axis.tl.fit_transition])."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3))
    for key, values in history.items():
        v = np.asarray(values, dtype=float)
        if smooth > 1 and len(v) > smooth:
            v = np.convolve(v, np.ones(smooth) / smooth, mode="valid")
        ax.plot(v, label=key, lw=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
