#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Illustration of AXIS alignment innovation
-----------------------------------------
This script generates six clean schematic figures:

1. Tissue state manifold (3D surface)
2. Two observed tissue states (different deformation / sampling)
3. Linear transformation
4. Discrete matching (unordered)
5. Discrete matching (ordered)
6. Continuous mapping (AXIS-like)

Each figure can be shown interactively or saved to disk using the
save_or_show() helper.

Author: ChatGPT
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =============================================================================
# Global style
# =============================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

STATE_A_COLOR = "#E76F51"   # warm coral
STATE_B_COLOR = "#5FAF62"   # soft green
FIELD_COLOR = "#6C8EBF"     # muted blue
MATCH_COLOR = "#B5B5B5"     # light gray
RIBBON_COLOR = "#CAD8F0"    # very light blue
SURFACE_COLOR = plt.cm.BuPu
SURFACE_ALPHA = 0.78


# =============================================================================
# Utility functions
# =============================================================================
def save_or_show(fig, filename=None, outdir=None, show=True, close=True):
    """
    Save figure if filename is provided; otherwise just show.
    If both outdir and filename are provided, save to outdir/filename.
    """
    if filename is not None:
        if outdir is None:
            outdir = "."
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, filename)
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")

    if show:
        plt.show()

    if close:
        plt.close(fig)


def hide_2d_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def tissue_curve(x, amp=1.0, freq=1.0, phase=0.0, offset=0.0, trend=0.0):
    """
    Smooth tissue-state curve used for the schematic figures.
    """
    return amp * np.sin(freq * x + phase) + 0.18 * np.sin(2.2 * x + phase / 2) + trend * (x - x.mean()) + offset


def sample_points_on_curve(x, y, n=16, uneven=True, seed=0):
    rng = np.random.default_rng(seed)
    if uneven:
        w = np.linspace(0.6, 1.4, len(x))
        w = w / w.sum()
        idx = np.sort(rng.choice(np.arange(len(x)), size=n, replace=False, p=w))
    else:
        idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx], y[idx], idx


def draw_points(ax, xs, ys, color, marker="o", s=18, zorder=5, edgecolor="white", lw=0.4):
    ax.scatter(xs, ys, s=s, color=color, marker=marker, edgecolor=edgecolor, linewidth=lw, zorder=zorder)


def draw_curve(ax, x, y, color, lw=2.0, alpha=1.0, linestyle="-", zorder=2):
    ax.plot(x, y, color=color, lw=lw, alpha=alpha, linestyle=linestyle, zorder=zorder)


def add_panel_title(ax, title):
    ax.set_title(title, pad=10, weight="bold")


# =============================================================================
# Figure 1: Tissue state manifold
# =============================================================================
def make_figure1_manifold(show=True, outdir=None):
    fig = plt.figure(figsize=(6.8, 4.8))
    ax = fig.add_subplot(111, projection="3d")

    # Grid in a conceptual space: x=time, y=disease state
    x = np.linspace(0, 10, 90)
    y = np.linspace(0, 10, 90)
    X, Y = np.meshgrid(x, y)

    # Smooth nontrivial manifold
    Z = (
        0.8 * np.sin(X / 1.6)
        + 0.7 * np.cos(Y / 2.0)
        + 0.25 * np.sin((X + Y) / 2.5)
        - 0.12 * (Y - 5)
        + 0.06 * (X - 5)
    )

    surf = ax.plot_surface(
        X, Y, Z,
        cmap=SURFACE_COLOR,
        linewidth=0,
        antialiased=True,
        alpha=SURFACE_ALPHA,
        shade=True
    )

    # Conceptual path on the manifold
    t = np.linspace(0.8, 9.2, 120)
    d = 2.5 + 4.5 * (t / 10.0) + 0.8 * np.sin(t / 1.7)
    z = (
        0.8 * np.sin(t / 1.6)
        + 0.7 * np.cos(d / 2.0)
        + 0.25 * np.sin((t + d) / 2.5)
        - 0.12 * (d - 5)
        + 0.06 * (t - 5)
    )
    ax.plot(t, d, z + 0.02, color="#3B4D7A", lw=2.6, zorder=20)

    # Sampled observed states
    pts_idx = np.array([15, 40, 68, 95])
    ax.scatter(
        t[pts_idx], d[pts_idx], z[pts_idx] + 0.05,
        s=42, color="#F4A261", edgecolor="white", linewidth=0.6, zorder=30
    )

    # A "missing / unobserved" tissue state
    ax.scatter(
        [7.9], [7.5],
        [0.8 * np.sin(7.9 / 1.6) + 0.7 * np.cos(7.5 / 2.0) + 0.25 * np.sin((7.9 + 7.5) / 2.5) - 0.12 * (7.5 - 5) + 0.06 * (7.9 - 5) + 0.08],
        s=90, marker="*", color="#6D597A", edgecolor="white", linewidth=0.8, zorder=35
    )

    ax.set_xlabel("Time", labelpad=8)
    ax.set_ylabel("Disease", labelpad=8)
    ax.set_zlabel("Tissue state", labelpad=8)
    ax.set_title("Tissue state manifold", pad=14, weight="bold")

    ax.view_init(elev=24, azim=-60)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    save_or_show(fig, "fig1_tissue_state_manifold.png", outdir=outdir, show=show)


# =============================================================================
# Figure 2: Two observed tissue states
# =============================================================================
def make_figure2_observed_states(show=True, outdir=None):
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    x1 = np.linspace(0, 10.5, 300)
    y1 = tissue_curve(x1, amp=1.05, freq=0.92, phase=0.25, offset=1.1, trend=0.02)

    x2 = np.linspace(0, 8.8, 300)
    y2 = tissue_curve(x2, amp=0.72, freq=0.95, phase=-0.7, offset=-1.05, trend=-0.03)

    draw_curve(ax, x1, y1, STATE_A_COLOR, lw=2.2)
    draw_curve(ax, x2, y2, STATE_B_COLOR, lw=2.2)

    sx1, sy1, _ = sample_points_on_curve(x1, y1, n=18, uneven=True, seed=1)
    sx2, sy2, _ = sample_points_on_curve(x2, y2, n=14, uneven=True, seed=3)

    draw_points(ax, sx1, sy1, STATE_A_COLOR, marker="x", s=24, edgecolor=STATE_A_COLOR, lw=0.8)
    draw_points(ax, sx2, sy2, STATE_B_COLOR, marker="o", s=22)

    ax.text(10.75, y1[-1] + 0.02, "Observed tissue A", color=STATE_A_COLOR, va="center", ha="left", fontsize=11)
    ax.text(9.05, y2[-1] - 0.10, "Observed tissue B", color=STATE_B_COLOR, va="center", ha="left", fontsize=11)

    add_panel_title(ax, "Observed tissues")
    hide_2d_axes(ax)
    ax.set_xlim(-0.4, 12.1)
    ax.set_ylim(-2.4, 2.6)

    save_or_show(fig, "fig2_observed_tissues.png", outdir=outdir, show=show)


# =============================================================================
# Figure 3: Linear transformation
# =============================================================================
def make_figure3_linear_transform(show=True, outdir=None):
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    xA = np.linspace(0, 10, 300)
    yA = tissue_curve(xA, amp=1.0, freq=0.95, phase=0.15, offset=1.0, trend=0.01)

    xB = np.linspace(0, 7.2, 300)
    yB = tissue_curve(xB, amp=0.75, freq=0.95, phase=-0.55, offset=-1.0, trend=-0.02)

    # Linear transformation of B into approximate frame of A
    xB_lin = (xB - xB.min()) / (xB.max() - xB.min()) * (xA.max() - xA.min())
    yB_lin = (yB - yB.mean()) * 1.25 + 1.0  # shift and scale

    draw_curve(ax, xA, yA, STATE_A_COLOR, lw=2.1, alpha=0.9)
    draw_curve(ax, xB_lin, yB_lin, STATE_B_COLOR, lw=2.1, alpha=0.95)

    sxA, syA, _ = sample_points_on_curve(xA, yA, n=18, uneven=False, seed=2)
    sxB, syB, idxB = sample_points_on_curve(xB_lin, yB_lin, n=14, uneven=False, seed=5)

    draw_points(ax, sxA, syA, STATE_A_COLOR, marker="x", s=22, edgecolor=STATE_A_COLOR, lw=0.8)
    draw_points(ax, sxB, syB, STATE_B_COLOR, marker="o", s=22)

    # Add faint transformation arrow annotation
    ax.annotate(
        "Linear transformation",
        xy=(5.5, 2.2), xytext=(5.5, 2.8),
        ha="center", va="bottom", fontsize=11,
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#666666")
    )

    add_panel_title(ax, "Linear transformation")
    hide_2d_axes(ax)
    ax.set_xlim(-0.4, 10.6)
    ax.set_ylim(-2.2, 3.15)

    save_or_show(fig, "fig3_linear_transformation.png", outdir=outdir, show=show)


# =============================================================================
# Figure 4: Discrete matching (unordered)
# =============================================================================
def make_figure4_unordered_matching(show=True, outdir=None):
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    xA = np.linspace(0, 10, 300)
    yA = tissue_curve(xA, amp=1.0, freq=0.95, phase=0.15, offset=1.1, trend=0.01)

    xB = np.linspace(0, 10, 300)
    yB = tissue_curve(xB, amp=0.80, freq=0.95, phase=-0.55, offset=-1.0, trend=-0.01)

    draw_curve(ax, xA, yA, STATE_A_COLOR, lw=2.0)
    draw_curve(ax, xB, yB, STATE_B_COLOR, lw=2.0)

    sxA, syA, _ = sample_points_on_curve(xA, yA, n=16, uneven=False, seed=10)
    sxB, syB, _ = sample_points_on_curve(xB, yB, n=16, uneven=False, seed=11)

    draw_points(ax, sxA, syA, STATE_A_COLOR, marker="x", s=22, edgecolor=STATE_A_COLOR, lw=0.8)
    draw_points(ax, sxB, syB, STATE_B_COLOR, marker="o", s=22)

    # Unordered 1-1 matching = a permutation
    perm = np.array([8, 1, 12, 5, 15, 10, 3, 13, 0, 14, 6, 9, 2, 11, 4, 7])
    for i in range(len(sxA)):
        ax.plot([sxA[i], sxB[perm[i]]], [syA[i], syB[perm[i]]],
                color=MATCH_COLOR, lw=0.85, alpha=0.75, zorder=1)

    add_panel_title(ax, "Discrete matching (unordered)")
    hide_2d_axes(ax)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-2.2, 2.8)

    save_or_show(fig, "fig4_discrete_matching_unordered.png", outdir=outdir, show=show)


# =============================================================================
# Figure 5: Discrete matching (ordered)
# =============================================================================
def make_figure5_ordered_matching(show=True, outdir=None):
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    xA = np.linspace(0, 10, 300)
    yA = tissue_curve(xA, amp=1.0, freq=0.95, phase=0.15, offset=1.1, trend=0.01)

    xB = np.linspace(0, 10, 300)
    yB = tissue_curve(xB, amp=0.80, freq=0.95, phase=-0.55, offset=-1.0, trend=-0.01)

    draw_curve(ax, xA, yA, STATE_A_COLOR, lw=2.0)
    draw_curve(ax, xB, yB, STATE_B_COLOR, lw=2.0)

    sxA, syA, _ = sample_points_on_curve(xA, yA, n=16, uneven=False, seed=10)
    sxB, syB, _ = sample_points_on_curve(xB, yB, n=16, uneven=False, seed=11)

    draw_points(ax, sxA, syA, STATE_A_COLOR, marker="x", s=22, edgecolor=STATE_A_COLOR, lw=0.8)
    draw_points(ax, sxB, syB, STATE_B_COLOR, marker="o", s=22)

    # Ordered matching = monotonic correspondence
    for i in range(len(sxA)):
        ax.plot([sxA[i], sxB[i]], [syA[i], syB[i]],
                color=MATCH_COLOR, lw=0.85, alpha=0.75, zorder=1)

    add_panel_title(ax, "Discrete matching (ordered)")
    hide_2d_axes(ax)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-2.2, 2.8)

    save_or_show(fig, "fig5_discrete_matching_ordered.png", outdir=outdir, show=show)


# =============================================================================
# Figure 6: Continuous mapping (AXIS)
# =============================================================================
def make_figure6_continuous_mapping(show=True, outdir=None):
    fig, ax = plt.subplots(figsize=(7.4, 3.4))

    xA = np.linspace(0, 10, 500)
    yA = tissue_curve(xA, amp=1.0, freq=0.95, phase=0.15, offset=1.08, trend=0.01)

    xB = np.linspace(0, 10, 500)
    yB = tissue_curve(xB, amp=0.80, freq=0.95, phase=-0.55, offset=-0.92, trend=-0.01)

    # Intermediate continuous field: blend A/B smoothly along the whole shape
    x_dense = np.linspace(0, 10, 100)
    yA_i = np.interp(x_dense, xA, yA)
    yB_i = np.interp(x_dense, xB, yB)

    # Ribbon suggesting continuous correspondence field
    for xi, ya, yb in zip(x_dense[::2], yA_i[::2], yB_i[::2]):
        ax.plot([xi, xi], [yb, ya], color=RIBBON_COLOR, lw=1.8, alpha=0.55, zorder=0)

    # Faint intermediate warped curves
    for w in np.linspace(0.15, 0.85, 5):
        ym = (1 - w) * yB + w * yA
        ax.plot(xA, ym, color=FIELD_COLOR, lw=1.0, alpha=0.18, zorder=1)

    draw_curve(ax, xA, yA, STATE_A_COLOR, lw=2.2, alpha=0.95, zorder=4)
    draw_curve(ax, xB, yB, STATE_B_COLOR, lw=2.2, alpha=0.95, zorder=4)

    # Show sparse points but emphasize overall shape, not 1-1 links
    sxA, syA, _ = sample_points_on_curve(xA, yA, n=18, uneven=True, seed=21)
    sxB, syB, _ = sample_points_on_curve(xB, yB, n=14, uneven=True, seed=22)
    draw_points(ax, sxA, syA, STATE_A_COLOR, marker="x", s=22, edgecolor=STATE_A_COLOR, lw=0.8)
    draw_points(ax, sxB, syB, STATE_B_COLOR, marker="o", s=20)

    # A few field arrows showing global shape mapping
    arrow_x = np.array([1.1, 3.1, 5.3, 7.5, 9.0])
    for xi in arrow_x:
        ya = np.interp(xi, xA, yA)
        yb = np.interp(xi, xB, yB)
        arrow = FancyArrowPatch(
            (xi, yb + 0.12), (xi, ya - 0.12),
            arrowstyle='-|>', mutation_scale=10,
            lw=1.0, color=FIELD_COLOR, alpha=0.85
        )
        ax.add_patch(arrow)

    ax.text(9.75, 2.25, "Continuous field", color=FIELD_COLOR, ha="right", va="center", fontsize=11)

    add_panel_title(ax, "Continuous mapping (AXIS)")
    hide_2d_axes(ax)
    ax.set_xlim(-0.45, 10.45)
    ax.set_ylim(-2.1, 2.85)

    save_or_show(fig, "fig6_continuous_mapping_axis.png", outdir=outdir, show=show)


# =============================================================================
# Optional combined panel
# =============================================================================
def make_combined_panel(show=True, outdir=None):
    """
    Optional 2x3 combined panel for quick inspection.
    """
    fig = plt.figure(figsize=(14.5, 7.5))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.16)

    # Figure 1 in combined form
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    x = np.linspace(0, 10, 70)
    y = np.linspace(0, 10, 70)
    X, Y = np.meshgrid(x, y)
    Z = 0.8 * np.sin(X / 1.6) + 0.7 * np.cos(Y / 2.0) + 0.25 * np.sin((X + Y) / 2.5) - 0.12 * (Y - 5) + 0.06 * (X - 5)
    ax1.plot_surface(X, Y, Z, cmap=SURFACE_COLOR, linewidth=0, antialiased=True, alpha=0.78)
    t = np.linspace(0.8, 9.2, 100)
    d = 2.5 + 4.5 * (t / 10.0) + 0.8 * np.sin(t / 1.7)
    z = 0.8 * np.sin(t / 1.6) + 0.7 * np.cos(d / 2.0) + 0.25 * np.sin((t + d) / 2.5) - 0.12 * (d - 5) + 0.06 * (t - 5)
    ax1.plot(t, d, z + 0.02, color="#3B4D7A", lw=2.3)
    ax1.scatter(t[[15, 40, 68, 95]], d[[15, 40, 68, 95]], z[[15, 40, 68, 95]] + 0.05,
                s=30, color="#F4A261", edgecolor="white", linewidth=0.5)
    ax1.set_title("1. Tissue state manifold", pad=8, weight="bold", fontsize=12)
    ax1.set_xlabel("Time", labelpad=2)
    ax1.set_ylabel("Disease", labelpad=2)
    ax1.set_zlabel("State", labelpad=2)
    ax1.view_init(elev=24, azim=-60)
    ax1.grid(False)

    def small_ax_style(ax, title):
        ax.set_title(title, pad=8, weight="bold", fontsize=12)
        hide_2d_axes(ax)

    # Shared curves
    xA = np.linspace(0, 10, 300)
    yA = tissue_curve(xA, amp=1.0, freq=0.95, phase=0.15, offset=1.1, trend=0.01)
    xB = np.linspace(0, 10, 300)
    yB = tissue_curve(xB, amp=0.80, freq=0.95, phase=-0.55, offset=-1.0, trend=-0.01)

    # Fig 2
    ax2 = fig.add_subplot(gs[0, 1])
    draw_curve(ax2, xA, yA, STATE_A_COLOR)
    draw_curve(ax2, xB * 0.88, yB, STATE_B_COLOR)
    sxA, syA, _ = sample_points_on_curve(xA, yA, n=18, uneven=True, seed=1)
    sxB, syB, _ = sample_points_on_curve(xB * 0.88, yB, n=14, uneven=True, seed=3)
    draw_points(ax2, sxA, syA, STATE_A_COLOR, marker="x", s=20, edgecolor=STATE_A_COLOR, lw=0.8)
    draw_points(ax2, sxB, syB, STATE_B_COLOR, marker="o", s=20)
    small_ax_style(ax2, "2. Observed tissues")
    ax2.set_xlim(-0.4, 10.6)
    ax2.set_ylim(-2.2, 2.7)

    # Fig 3
    ax3 = fig.add_subplot(gs[0, 2])
    xB_lin = (xB - xB.min()) / (xB.max() - xB.min()) * (xA.max() - xA.min())
    yB_lin = (yB - yB.mean()) * 1.25 + 1.0
    draw_curve(ax3, xA, yA, STATE_A_COLOR)
    draw_curve(ax3, xB_lin, yB_lin, STATE_B_COLOR)
    sxA, syA, _ = sample_points_on_curve(xA, yA, n=16, uneven=False, seed=2)
    sxB, syB, _ = sample_points_on_curve(xB_lin, yB_lin, n=14, uneven=False, seed=5)
    draw_points(ax3, sxA, syA, STATE_A_COLOR, marker="x", s=20, edgecolor=STATE_A_COLOR, lw=0.8)
    draw_points(ax3, sxB, syB, STATE_B_COLOR, marker="o", s=20)
    small_ax_style(ax3, "3. Linear transformation")
    ax3.set_xlim(-0.4, 10.4)
    ax3.set_ylim(-2.2, 3.0)

    # Fig 4
    ax4 = fig.add_subplot(gs[1, 0])
    draw_curve(ax4, xA, yA, STATE_A_COLOR)
    draw_curve(ax4, xB, yB, STATE_B_COLOR)
    sxA, syA, _ = sample_points_on_curve(xA, yA, n=16, uneven=False, seed=10)
    sxB, syB, _ = sample_points_on_curve(xB, yB, n=16, uneven=False, seed=11)
    draw_points(ax4, sxA, syA, STATE_A_COLOR, marker="x", s=20, edgecolor=STATE_A_COLOR, lw=0.8)
    draw_points(ax4, sxB, syB, STATE_B_COLOR, marker="o", s=20)
    perm = np.array([8, 1, 12, 5, 15, 10, 3, 13, 0, 14, 6, 9, 2, 11, 4, 7])
    for i in range(len(sxA)):
        ax4.plot([sxA[i], sxB[perm[i]]], [syA[i], syB[perm[i]]], color=MATCH_COLOR, lw=0.8, alpha=0.7)
    small_ax_style(ax4, "4. Discrete matching (unordered)")
    ax4.set_xlim(-0.4, 10.4)
    ax4.set_ylim(-2.2, 2.7)

    # Fig 5
    ax5 = fig.add_subplot(gs[1, 1])
    draw_curve(ax5, xA, yA, STATE_A_COLOR)
    draw_curve(ax5, xB, yB, STATE_B_COLOR)
    draw_points(ax5, sxA, syA, STATE_A_COLOR, marker="x", s=20, edgecolor=STATE_A_COLOR, lw=0.8)
    draw_points(ax5, sxB, syB, STATE_B_COLOR, marker="o", s=20)
    for i in range(len(sxA)):
        ax5.plot([sxA[i], sxB[i]], [syA[i], syB[i]], color=MATCH_COLOR, lw=0.8, alpha=0.7)
    small_ax_style(ax5, "5. Discrete matching (ordered)")
    ax5.set_xlim(-0.4, 10.4)
    ax5.set_ylim(-2.2, 2.7)

    # Fig 6
    ax6 = fig.add_subplot(gs[1, 2])
    x_dense = np.linspace(0, 10, 70)
    yA_i = np.interp(x_dense, xA, yA)
    yB_i = np.interp(x_dense, xB, yB)
    for xi, ya, yb in zip(x_dense[::2], yA_i[::2], yB_i[::2]):
        ax6.plot([xi, xi], [yb, ya], color=RIBBON_COLOR, lw=1.8, alpha=0.5)
    for w in np.linspace(0.2, 0.8, 4):
        ym = (1 - w) * yB + w * yA
        ax6.plot(xA, ym, color=FIELD_COLOR, lw=1.0, alpha=0.18)
    draw_curve(ax6, xA, yA, STATE_A_COLOR)
    draw_curve(ax6, xB, yB, STATE_B_COLOR)
    draw_points(ax6, sxA, syA, STATE_A_COLOR, marker="x", s=20, edgecolor=STATE_A_COLOR, lw=0.8)
    draw_points(ax6, sxB[:14], syB[:14], STATE_B_COLOR, marker="o", s=20)
    small_ax_style(ax6, "6. Continuous mapping (AXIS)")
    ax6.set_xlim(-0.4, 10.4)
    ax6.set_ylim(-2.2, 2.7)

    save_or_show(fig, "combined_alignment_innovation_panel.png", outdir=outdir, show=show)


# =============================================================================
# Main
# =============================================================================
def main(show=True, outdir="/media/huifang/data/registration/paper_figure/Figure2/", also_make_combined=True):
    make_figure1_manifold(show=show, outdir=outdir)
    make_figure2_observed_states(show=show, outdir=outdir)
    make_figure3_linear_transform(show=show, outdir=outdir)
    make_figure4_unordered_matching(show=show, outdir=outdir)
    make_figure5_ordered_matching(show=show, outdir=outdir)
    make_figure6_continuous_mapping(show=show, outdir=outdir)

    if also_make_combined:
        make_combined_panel(show=show, outdir=outdir)


if __name__ == "__main__":
    # Example:
    #   show=True  -> display figures
    #   show=False -> only save to disk
    main(show=True, outdir="/media/huifang/data/registration/paper_figure/Figure2/", also_make_combined=True)