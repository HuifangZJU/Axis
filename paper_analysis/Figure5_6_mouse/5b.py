#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render Allen mouse-brain anatomical annotations from LabelMe polygons.

Regions:
    - cortex
    - hippocampus
    - thalamus
    - hypothalamus

Visualization:
    1. Load Allen mouse-brain atlas image.
    2. Load LabelMe polygon annotations.
    3. Compute the average RGB color inside each annotated region.
    4. Compute a complementary / contrasting color from that mean color.
    5. Draw smooth dashed anatomical boundaries using the contrasting color.
    6. Add a clean right-side annotation panel:
         - rectangle fill = mean region color
         - dashed rectangle border = same contrasting color as contour
         - formal anatomical-region name
       No connector lines are drawn.
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
from scipy.interpolate import splprep, splev


# ============================================================
# PATHS
# ============================================================

IMAGE_PATH = (
    "/media/huifang/data/registration/"
    "paper_analysis/Figure56_mouse/5b/Allen_mouse.png"
)

JSON_PATH = (
    "/media/huifang/data/registration/"
    "paper_analysis/Figure56_mouse/5b/Allen_mouse.json"
)

OUTPUT_PATH = (
    "/media/huifang/data/registration/"
    "paper_analysis/Figure56_mouse/5b/"
    "Allen_mouse_annotated.png"
)


# ============================================================
# REGION CONFIGURATION
# ============================================================

TARGET_REGIONS = [
    "cortex",
    "hippocampus",
    "thalamus",
    "hypothalamus",
]

LABEL_ORDER = [
    "cortex",
    "hippocampus",
    "thalamus",
    "hypothalamus",
]

FORMAL_REGION_NAMES = {
    "cortex": "Cortex",
    "hippocampus": "Hippocampus",
    "thalamus": "Thalamus",
    "hypothalamus": "Hypothalamus",
}


# ============================================================
# CONTOUR STYLE
# ============================================================

BOUNDARY_LINEWIDTH = 3

BOUNDARY_LINESTYLE = (
    0,
    (6.0, 4.0),
)

BOUNDARY_POINTS_DENSITY = 12
SPLINE_SMOOTHNESS = 8.0

BOUNDARY_HALO_EXTRA_WIDTH = 1.8
BOUNDARY_HALO_ALPHA = 0.90


# ============================================================
# ANNOTATION STYLE
# ============================================================

LABEL_FONTSIZE = 20
LABEL_FONTWEIGHT = "normal"
LABEL_TEXT_COLOR = "#222222"

RIGHT_PANEL_EXTRA_WIDTH = 0.40

LABEL_Y_START = 0.20
LABEL_Y_END = 0.80

LEGEND_MARKER_WIDTH = 0.055
LEGEND_MARKER_HEIGHT = 0.045

LEGEND_MARKER_X = 1.035
LEGEND_TEXT_GAP = 0.025

TEXT_HALO_WIDTH = 3.0
TEXT_HALO_ALPHA = 0.95


# ============================================================
# FIGURE STYLE
# ============================================================

FIGSIZE = (12.5, 8.0)
FIG_DPI = 300
OUTPUT_PAD_INCHES = 0.02


# ============================================================
# LABELME LOADING
# ============================================================

def load_labelme_shapes(json_path: str):
    """
    Read LabelMe polygons and group them by anatomical region.

    Returns
    -------
    dict
        {
            "cortex": [polygon1, polygon2, ...],
            ...
        }

    Each polygon is an N x 2 numpy array containing (x, y).
    """

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    grouped = defaultdict(list)

    for shape in data.get("shapes", []):

        label = (
            str(
                shape.get(
                    "label",
                    "",
                )
            )
            .strip()
            .lower()
        )

        if label not in TARGET_REGIONS:
            continue

        points = shape.get(
            "points",
            [],
        )

        if len(points) < 3:
            print(
                f"[WARN] Skip {label!r}: "
                f"polygon contains only {len(points)} points."
            )
            continue

        polygon = np.asarray(
            points,
            dtype=float,
        )

        grouped[label].append(
            polygon
        )

    return dict(grouped)


# ============================================================
# REGION MASK / COLOR
# ============================================================

def polygon_mask(
    image_size,
    polygon_xy: np.ndarray,
) -> np.ndarray:
    """
    Rasterize one polygon into a boolean image mask.
    """

    width, height = image_size

    mask_image = Image.new(
        "L",
        (
            width,
            height,
        ),
        0,
    )

    drawer = ImageDraw.Draw(
        mask_image
    )

    drawer.polygon(
        [
            (
                float(point[0]),
                float(point[1]),
            )
            for point in polygon_xy
        ],
        outline=1,
        fill=1,
    )

    return (
        np.asarray(mask_image)
        > 0
    )


def build_region_mask(
    image_size,
    polygons,
) -> np.ndarray:
    """
    Build the union mask of all polygons belonging to one region.
    """

    width, height = image_size

    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=bool,
    )

    for polygon in polygons:

        mask |= polygon_mask(
            (
                width,
                height,
            ),
            polygon,
        )

    return mask


def compute_average_color(
    image_np: np.ndarray,
    polygons,
) -> np.ndarray:
    """
    Compute the mean RGB color inside one anatomical region.

    Returns RGB in [0, 1].
    """

    height, width = image_np.shape[:2]

    mask = build_region_mask(
        (
            width,
            height,
        ),
        polygons,
    )

    if not np.any(mask):

        print(
            "[WARN] Empty region mask. "
            "Using gray as fallback."
        )

        return np.asarray(
            [
                0.5,
                0.5,
                0.5,
            ],
            dtype=float,
        )

    pixels = np.asarray(
        image_np[mask],
        dtype=float,
    )

    rgb = np.mean(
        pixels[:, :3],
        axis=0,
    )

    return (
        rgb
        / 255.0
    )


def contrasting_color_from_average(
    average_rgb: np.ndarray,
) -> np.ndarray:
    """
    Generate a complementary / contrasting color from the region mean.

    Direct RGB complement:
        contrast = 1 - average_rgb

    The result is clipped slightly away from pure black / white so
    the contour remains visible and publication-friendly.
    """

    average_rgb = np.asarray(
        average_rgb,
        dtype=float,
    )

    contrast = (
        1.0
        - np.clip(
            average_rgb,
            0.0,
            1.0,
        )
    )

    contrast = np.clip(
        contrast,
        0.12,
        0.88,
    )

    return contrast


# ============================================================
# SMOOTH BOUNDARY
# ============================================================

def remove_duplicate_vertices(
    polygon: np.ndarray,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """
    Remove immediately repeated polygon vertices.
    """

    polygon = np.asarray(
        polygon,
        dtype=float,
    )

    if len(polygon) <= 1:
        return polygon

    keep = np.ones(
        len(polygon),
        dtype=bool,
    )

    delta = np.linalg.norm(
        np.diff(
            polygon,
            axis=0,
        ),
        axis=1,
    )

    keep[1:] = (
        delta
        > tolerance
    )

    return polygon[
        keep
    ]


def smooth_closed_polygon(
    polygon: np.ndarray,
    density: int = BOUNDARY_POINTS_DENSITY,
    smoothness: float = SPLINE_SMOOTHNESS,
) -> np.ndarray:
    """
    Convert a LabelMe polygon into a smooth closed periodic curve.
    """

    polygon = remove_duplicate_vertices(
        polygon
    )

    if len(polygon) == 0:
        return polygon

    if (
        len(polygon) > 1
        and np.allclose(
            polygon[0],
            polygon[-1],
        )
    ):
        polygon = polygon[:-1]

    if len(polygon) < 4:

        return np.vstack(
            [
                polygon,
                polygon[0],
            ]
        )

    x = polygon[:, 0]
    y = polygon[:, 1]

    try:

        tck, _ = splprep(
            [
                x,
                y,
            ],
            s=float(
                smoothness
            ),
            per=True,
        )

        n_points = max(
            int(
                len(polygon)
                * density
            ),
            250,
        )

        u_new = np.linspace(
            0.0,
            1.0,
            n_points,
            endpoint=True,
        )

        x_smooth, y_smooth = (
            splev(
                u_new,
                tck,
            )
        )

        return np.column_stack(
            [
                x_smooth,
                y_smooth,
            ]
        )

    except Exception as exc:

        print(
            "[WARN] Spline smoothing failed; "
            f"using original polygon. Error: {exc}"
        )

        return np.vstack(
            [
                polygon,
                polygon[0],
            ]
        )


# ============================================================
# DRAW ANATOMICAL BOUNDARIES
# ============================================================

def draw_region_boundaries(
    ax,
    polygons,
    boundary_color,
) -> None:
    """
    Draw all polygons of one anatomical region using a smooth,
    dashed, contrasting contour.
    """

    for polygon in polygons:

        smooth_polygon = (
            smooth_closed_polygon(
                polygon
            )
        )

        if len(smooth_polygon) < 2:
            continue

        line, = ax.plot(
            smooth_polygon[:, 0],
            smooth_polygon[:, 1],
            # color=boundary_color,
            color="black",
            linewidth=BOUNDARY_LINEWIDTH,
            # linestyle=BOUNDARY_LINESTYLE,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=10,
        )

        line.set_path_effects(
            [
                patheffects.Stroke(
                    linewidth=(
                        BOUNDARY_LINEWIDTH
                        + BOUNDARY_HALO_EXTRA_WIDTH
                    ),
                    foreground="white",
                    alpha=BOUNDARY_HALO_ALPHA,
                ),
                patheffects.Normal(),
            ]
        )


# ============================================================
# RIGHT-SIDE ANNOTATION
# ============================================================

def draw_right_annotation_panel(
    ax,
    image_width: float,
    image_height: float,
    region_polygons,
    region_mean_colors,
    region_boundary_colors,
) -> None:
    """
    Draw anatomical labels on the right.

    Marker design:
        fill   = mean RGB color of anatomical region
        border = contrasting RGB color
        border style = identical to anatomical contour

    No connector lines are drawn.
    """

    present_regions = [
        region
        for region in LABEL_ORDER
        if region in region_polygons
    ]

    if not present_regions:
        return

    y_positions = np.linspace(
        LABEL_Y_START
        * image_height,
        LABEL_Y_END
        * image_height,
        len(present_regions),
    )

    marker_width = (
        LEGEND_MARKER_WIDTH
        * image_width
    )

    marker_height = (
        LEGEND_MARKER_HEIGHT
        * image_height
    )

    marker_x = (
        LEGEND_MARKER_X
        * image_width
    )

    text_x = (
        marker_x
        + marker_width
        + LEGEND_TEXT_GAP
        * image_width
    )

    for region, y_position in zip(
        present_regions,
        y_positions,
    ):

        fill_color = (
            region_mean_colors[
                region
            ]
        )

        boundary_color = (
            region_boundary_colors[
                region
            ]
        )

        formal_name = (
            FORMAL_REGION_NAMES.get(
                region,
                region.title(),
            )
        )

        # ----------------------------------------------------
        # Rectangle marker
        # ----------------------------------------------------

        rectangle = Rectangle(
            (
                marker_x,
                y_position
                - 0.5
                * marker_height,
            ),
            marker_width,
            marker_height,

            # Fill = actual mean atlas-region color.
            facecolor=fill_color,

            # Border = same contrasting color as contour.
            edgecolor="black",

            linewidth=BOUNDARY_LINEWIDTH,
            # linestyle=BOUNDARY_LINESTYLE,
            joinstyle="round",
            zorder=20,
        )

        ax.add_patch(
            rectangle
        )

        rectangle.set_path_effects(
            [
                patheffects.Stroke(
                    linewidth=(
                        BOUNDARY_LINEWIDTH
                        + BOUNDARY_HALO_EXTRA_WIDTH
                    ),
                    foreground="white",
                    alpha=BOUNDARY_HALO_ALPHA,
                ),
                patheffects.Normal(),
            ]
        )

        # ----------------------------------------------------
        # Formal region name
        # ----------------------------------------------------

        text = ax.text(
            text_x,
            y_position,
            formal_name,
            ha="left",
            va="center",
            fontsize=LABEL_FONTSIZE,
            fontweight=LABEL_FONTWEIGHT,
            color=LABEL_TEXT_COLOR,
            zorder=21,
        )

        text.set_path_effects(
            [
                patheffects.Stroke(
                    linewidth=TEXT_HALO_WIDTH,
                    foreground="white",
                    alpha=TEXT_HALO_ALPHA,
                ),
                patheffects.Normal(),
            ]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Validate input paths
    # --------------------------------------------------------

    if not os.path.isfile(
        IMAGE_PATH
    ):
        raise FileNotFoundError(
            IMAGE_PATH
        )

    if not os.path.isfile(
        JSON_PATH
    ):
        raise FileNotFoundError(
            JSON_PATH
        )

    # --------------------------------------------------------
    # Load atlas image
    # --------------------------------------------------------

    image = Image.open(
        IMAGE_PATH
    ).convert(
        "RGB"
    )

    image_np = np.asarray(
        image
    )

    height, width = (
        image_np.shape[:2]
    )

    print(
        f"[IMAGE] {width} x {height}"
    )

    # --------------------------------------------------------
    # Load LabelMe annotations
    # --------------------------------------------------------

    region_polygons = (
        load_labelme_shapes(
            JSON_PATH
        )
    )

    for region in TARGET_REGIONS:

        n_polygons = len(
            region_polygons.get(
                region,
                [],
            )
        )

        print(
            f"[ANNOTATION] "
            f"{region}: "
            f"{n_polygons} polygon(s)"
        )

    missing_regions = [
        region
        for region in TARGET_REGIONS
        if region not in region_polygons
    ]

    if missing_regions:

        print(
            "[WARN] Missing LabelMe annotations: "
            + ", ".join(
                missing_regions
            )
        )

    # --------------------------------------------------------
    # Compute mean colors and contrasting boundary colors
    # --------------------------------------------------------

    region_mean_colors = {}
    region_boundary_colors = {}

    for region, polygons in (
        region_polygons.items()
    ):

        average_rgb = (
            compute_average_color(
                image_np,
                polygons,
            )
        )

        boundary_rgb = (
            contrasting_color_from_average(
                average_rgb
            )
        )

        region_mean_colors[
            region
        ] = average_rgb

        region_boundary_colors[
            region
        ] = boundary_rgb

        average_rgb_255 = np.round(
            255.0
            * average_rgb
        ).astype(
            int
        )

        boundary_rgb_255 = np.round(
            255.0
            * boundary_rgb
        ).astype(
            int
        )

        print(
            f"[COLOR] {region} | "
            f"mean RGB={tuple(average_rgb_255)} | "
            f"boundary RGB={tuple(boundary_rgb_255)}"
        )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
        dpi=FIG_DPI,
    )

    ax.imshow(
        image_np,
        interpolation="nearest",
        zorder=0,
    )

    # --------------------------------------------------------
    # Draw smooth dashed anatomical boundaries
    # --------------------------------------------------------

    for region in TARGET_REGIONS:

        if region not in region_polygons:
            continue

        draw_region_boundaries(
            ax,
            polygons=(
                region_polygons[
                    region
                ]
            ),
            boundary_color=(
                region_boundary_colors[
                    region
                ]
            ),
        )

    # --------------------------------------------------------
    # Draw right-side annotation panel
    # --------------------------------------------------------

    draw_right_annotation_panel(
        ax,
        image_width=float(
            width
        ),
        image_height=float(
            height
        ),
        region_polygons=(
            region_polygons
        ),
        region_mean_colors=(
            region_mean_colors
        ),
        region_boundary_colors=(
            region_boundary_colors
        ),
    )

    # --------------------------------------------------------
    # Canvas geometry
    # --------------------------------------------------------

    ax.set_xlim(
        0.0,
        (
            1.0
            + RIGHT_PANEL_EXTRA_WIDTH
        )
        * width,
    )

    ax.set_ylim(
        height,
        0.0,
    )

    ax.set_aspect(
        "equal"
    )

    ax.axis(
        "off"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_directory = (
        os.path.dirname(
            OUTPUT_PATH
        )
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    fig.savefig(
        OUTPUT_PATH,
        dpi=FIG_DPI,
        bbox_inches="tight",
        pad_inches=OUTPUT_PAD_INCHES,
        facecolor="white",
    )

    plt.close(
        fig
    )

    print(
        "[DONE] Saved annotated Allen mouse atlas:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()