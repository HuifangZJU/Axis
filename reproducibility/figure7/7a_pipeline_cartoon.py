from axis.settings import data_path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.colors import to_rgb


# ============================================================
# 1. Gene-read profiles
# ============================================================

def plot_gene_read_profiles(
    n_profiles=5,
    n_bars=8,
    seed=12,
    figsize=(2.0, 7.5),
    save_path="gene_read_profiles.png",
    dpi=600,
):
    """
    Plot vertically stacked schematic gene-read distributions.

    Output:
    - PNG only
    - Transparent background
    - Only the bottom axis line is shown
    """

    rng = np.random.default_rng(seed)

    colors = [
        "#4C9BE8",  # blue
        "#4CCB6B",  # green
        "#F5A623",  # orange
        "#9B6BD6",  # purple
        "#EF6A67",  # coral/red
    ]

    fig, axes = plt.subplots(
        n_profiles,
        1,
        figsize=figsize,
        gridspec_kw={"hspace": 0.55},
    )

    if n_profiles == 1:
        axes = [axes]

    x = np.arange(n_bars)

    for i, ax in enumerate(axes):

        # Random but structured gene-read profile
        heights = rng.gamma(
            shape=rng.uniform(1.4, 3.0),
            scale=rng.uniform(0.6, 1.2),
            size=n_bars,
        )
        heights /= heights.max()

        ax.bar(
            x,
            heights,
            width=0.65,
            color=colors[i % len(colors)],
            edgecolor="none",
        )

        ax.set_xlim(-0.7, n_bars - 0.3)
        ax.set_ylim(0, 1.15)

        ax.set_xticks([])
        ax.set_yticks([])

        # Keep only the bottom baseline
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_linewidth(1.6)
        ax.spines["bottom"].set_color("#34495E")

        ax.tick_params(bottom=False, left=False)

        # Transparent axes background
        ax.set_facecolor("none")

    # Transparent figure background
    fig.patch.set_alpha(0)

    fig.savefig(
        save_path,
        dpi=dpi,
        format="png",
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )

    plt.close(fig)


# ============================================================
# 2. Protein-read indicator
# ============================================================

def _interpolate_color(color1, color2, t):
    """Linear RGB interpolation between two colors."""
    c1 = np.array(to_rgb(color1))
    c2 = np.array(to_rgb(color2))
    return tuple((1.0 - t) * c1 + t * c2)


def plot_protein_read_indicator(
    figsize=(3.2, 2.5),
    save_path="protein_read_indicator.png",
    dpi=600,
):
    """
    Plot a schematic protein-read indicator.

    Top:
        circular protein marker

    Bottom:
        rounded abundance bar
        light -> dark indicates low -> high protein read

    Output:
    - PNG only
    - Transparent background
    """

    fig, ax = plt.subplots(figsize=figsize)

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------
    circle_center = (0.5, 0.72)
    circle_radius = 0.24

    bar_x = 0.12
    bar_y = 0.24
    bar_width = 0.76
    bar_height = 0.16

    # --------------------------------------------------------
    # Protein marker
    # --------------------------------------------------------
    circle = Circle(
        circle_center,
        circle_radius,
        facecolor="#D88A90",
        edgecolor="#C6535D",
        linewidth=2.0,
    )
    ax.add_patch(circle)

    # --------------------------------------------------------
    # Rounded abundance bar
    # --------------------------------------------------------
    capsule = FancyBboxPatch(
        (bar_x, bar_y),
        bar_width,
        bar_height,
        boxstyle=f"round,pad=0,rounding_size={bar_height / 2}",
        facecolor="none",
        edgecolor="none",
    )
    ax.add_patch(capsule)

    light_color = "#FAEEEE"
    dark_color = "#B92129"

    # More segments produce a visually smooth gradient
    n_segments = 300
    segment_width = bar_width / n_segments

    for i in range(n_segments):
        t = i / (n_segments - 1)

        rect = Rectangle(
            (
                bar_x + i * segment_width,
                bar_y,
            ),
            segment_width * 1.05,
            bar_height,
            facecolor=_interpolate_color(
                light_color,
                dark_color,
                t,
            ),
            edgecolor="none",
            linewidth=0,
        )

        rect.set_clip_path(capsule)
        ax.add_patch(rect)

    # Subtle rounded outline
    outline = FancyBboxPatch(
        (bar_x, bar_y),
        bar_width,
        bar_height,
        boxstyle=f"round,pad=0,rounding_size={bar_height / 2}",
        facecolor="none",
        edgecolor="#EACBCD",
        linewidth=0.45,
    )
    ax.add_patch(outline)

    # --------------------------------------------------------
    # Clean layout
    # --------------------------------------------------------
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_facecolor("none")
    fig.patch.set_alpha(0)

    fig.savefig(
        save_path,
        dpi=dpi,
        format="png",
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )

    plt.close(fig)


# ============================================================
# Generate Figure 7a schematic elements
# ============================================================

output_dir = (
    data_path(""
    "paper_analysis/Figure7_modality/a_pipeline/")
)

plot_gene_read_profiles(
    n_profiles=5,
    n_bars=8,
    seed=8,
    save_path=output_dir + "gene_read_profiles.png",
)

plot_protein_read_indicator(
    save_path=output_dir + "protein_read_indicator.png",
)