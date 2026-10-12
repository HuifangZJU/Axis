"""
AXIS reusable simulation-data generator -- v7
=============================================

Styling updates
---------------
1. Morphology view:
   - the tissue outer boundary is drawn in blue.

2. Spatial-domain boundaries:
   - contours use the SAME color as the spot color of that domain.

3. Spatially matched pseudo-time states:
   - no anatomy is saved
   - one shared spot-level domain is saved once
   - each pseudo-time has separate molecular reads on the same spatial units

4. Pseudo-time molecular palettes:
   - molecular and pseudo-omics images use five smoothly transitioning palettes
   - progression:
         jade green
         -> teal
         -> indigo blue
         -> violet
         -> rose pink
   - each time point still uses a sequential low-to-high gradient
   - domain-related colors are unchanged
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import (
    LinearSegmentedColormap,
    to_rgb,
)

from scipy.ndimage import distance_transform_edt


# ============================================================
# PATHS
# ============================================================

DEFAULT_TEMPLATE = Path(
    "/media/huifang/data/registration/paper_analysis/"
    "Figure1/cartoons/brain_image.jpeg"
)

DEFAULT_OUTPUT = Path(
    "/media/huifang/data/registration/paper_analysis/"
    "Figure1/c_method"
)


# ============================================================
# DOMAIN COLORS
#
# IMPORTANT:
# These are unchanged.
# ============================================================

DOMAIN_COLORS_HEX = [
    "#8DBDD8",  # cortex-like
    "#DDAA69",  # hippocampal-like
    "#B79BCB",  # deep-gray-like
    "#AEBF9B",  # inner/white-matter-like
]


# Blue used for anatomy outer outline.
ANATOMY_OUTLINE_RGB = (
    43,
    125,
    188,
)


# ============================================================
# PSEUDO-TIME MOLECULAR PALETTES
# ============================================================
#
# Design principle
# ----------------
#
# We want the five states to be:
#
#     clearly distinguishable
#
# but also:
#
#     perceptually ordered / continuously transitioning
#
# Therefore the dominant hue moves gradually:
#
#     jade green
#       ↓
#     teal
#       ↓
#     indigo blue
#       ↓
#     violet
#       ↓
#     rose pink
#
#
# Importantly, EACH state remains a sequential colormap:
#
#     low molecular abundance
#         -> pale / nearly white
#
#     intermediate abundance
#         -> clear characteristic hue
#
#     high abundance
#         -> saturated darker hue
#
#
# This allows the molecular spatial pattern to remain highly
# visible within each state.
#
# Domain-related colors are completely independent from this.
# ============================================================

PSEUDOTIME_PALETTE_STOPS_HEX = [

    # --------------------------------------------------------
    # State 1
    # Jade green
    # --------------------------------------------------------
    [
        "#F7FAF8",
        "#DDEEE5",
        "#8BC7AE",
        "#3F987D",
        "#1D604C",
    ],

    # --------------------------------------------------------
    # State 2
    # Teal
    #
    # Transitional position between green and blue.
    # --------------------------------------------------------
    [
        "#F6FAFA",
        "#D9ECEC",
        "#83BDC1",
        "#3A8996",
        "#1D5C67",
    ],

    # --------------------------------------------------------
    # State 3
    # Indigo blue
    # --------------------------------------------------------
    [
        "#F7F8FB",
        "#DEE5F2",
        "#95A9CF",
        "#5D73AB",
        "#394B79",
    ],

    # --------------------------------------------------------
    # State 4
    # Violet
    #
    # Transitional position between blue and rose.
    # --------------------------------------------------------
    [
        "#FAF7FA",
        "#E9DFED",
        "#BD9AC4",
        "#86639D",
        "#593F70",
    ],

    # --------------------------------------------------------
    # State 5
    # Rose pink
    # --------------------------------------------------------
    [
        "#FCF7F9",
        "#F1DFE6",
        "#D89EAF",
        "#B86784",
        "#7C4059",
    ],
]


# ============================================================
# MOLECULAR DISPLAY CONTRAST
# ============================================================

# < 1 lifts weak/intermediate molecular signals.
# 1.0 = no adjustment
# 0.7 = mild
# 0.55 = recommended
# 0.4 = strong
MOLECULAR_DISPLAY_GAMMA = 0.45


def enhance_molecular_display(values):
    """
    Visualization-only nonlinear contrast enhancement.

    Keeps:
        0 -> 0
        1 -> 1

    while increasing visibility of weak/intermediate reads.

    Does NOT modify saved molecular_read or pseudo_omics data.
    """
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    values = np.clip(
        values,
        0.0,
        1.0,
    )

    return np.power(
        values,
        MOLECULAR_DISPLAY_GAMMA,
    )


def _interpolate_palette_stops(
    palette_a,
    palette_b,
    weight,
):
    """
    Interpolate corresponding RGB stops between two sequential palettes.

    This allows the script to remain valid even if n_states is changed
    from the default value of 5.
    """

    weight = float(
        np.clip(
            weight,
            0.0,
            1.0,
        )
    )

    output = []

    for color_a, color_b in zip(
        palette_a,
        palette_b,
    ):

        rgb_a = np.asarray(
            to_rgb(color_a),
            dtype=np.float64,
        )

        rgb_b = np.asarray(
            to_rgb(color_b),
            dtype=np.float64,
        )

        rgb = (
            (1.0 - weight) * rgb_a
            + weight * rgb_b
        )

        output.append(
            tuple(
                np.clip(
                    rgb,
                    0.0,
                    1.0,
                )
            )
        )

    return output


def pseudotime_colormap(
    state_index,
    n_states,
):
    """
    Return the sequential molecular colormap for one pseudo-time state.

    Parameters
    ----------
    state_index
        One-based pseudo-time state index.

    n_states
        Total number of simulated states.

    Notes
    -----
    When n_states == 5, the five designed anchor palettes above
    are used exactly.

    If n_states differs from 5, the palettes are smoothly
    interpolated rather than abruptly repeated.
    """

    n_anchors = len(
        PSEUDOTIME_PALETTE_STOPS_HEX
    )

    if n_states <= 1:

        position = 0.0

    else:

        position = (
            (float(state_index) - 1.0)
            / float(n_states - 1)
            * float(n_anchors - 1)
        )

    low_index = int(
        np.floor(position)
    )

    high_index = min(
        low_index + 1,
        n_anchors - 1,
    )

    fraction = (
        position
        - low_index
    )

    stops = _interpolate_palette_stops(
        PSEUDOTIME_PALETTE_STOPS_HEX[
            low_index
        ],
        PSEUDOTIME_PALETTE_STOPS_HEX[
            high_index
        ],
        fraction,
    )

    cmap = LinearSegmentedColormap.from_list(
        f"axis_pseudotime_{state_index:02d}",
        stops,
        N=256,
    )

    return cmap


# ============================================================
# PARAMETERS
# ============================================================

@dataclass
class DeformationParameters:

    strength: float = 1.0

    width_change: float = 0.045
    height_change: float = -0.025

    shear: float = 0.030

    bend_x: float = 0.070
    bend_y: float = 0.055

    cross_term: float = 0.028

    wave: float = 0.018

    linear_y: float = -0.030
    quadratic_y: float = 0.018


@dataclass
class SimulationParameters:

    canvas: int = 1000

    margin: int = 70

    n_states: int = 5

    spot_spacing: float = 30.0

    spot_radius: int = 30


    # --------------------------------------------------------
    # Pseudo-omics reconstruction
    # --------------------------------------------------------

    pseudo_sigma_px: float = 48.0


    # --------------------------------------------------------
    # Smooth joint domain partition
    # --------------------------------------------------------

    domain_score_sigma_px: float = 58.0

    domain_spot_anchor_boost: float = 3.0

    domain_contour_smooth_window: int = 31

    domain_contour_thickness_px: int = 2

    domain_partition_close_px: int = 19

    domain_min_component_area_px: int = 2500


    # --------------------------------------------------------
    # Morphology
    # --------------------------------------------------------

    anatomy_outline_thickness_px: int = 4

    line_threshold: int = 245

    morphology_kernel: int = 9


    seed: int = 17


# ============================================================
# UTILITY
# ============================================================

def hex_to_rgb255(h):

    h = h.lstrip("#")

    return np.array(
        [
            int(
                h[i:i + 2],
                16,
            )
            for i in (
                0,
                2,
                4,
            )
        ],
        dtype=np.uint8,
    )


def save_rgba_png(
    rgba,
    path,
):

    ok = cv2.imwrite(
        str(path),
        rgba[
            ...,
            [
                2,
                1,
                0,
                3,
            ]
        ],
    )

    if not ok:

        raise IOError(
            f"Failed to save {path}"
        )


# ============================================================
# TEMPLATE EXTRACTION
# ============================================================

def read_and_square_template(
    path,
    sim,
):

    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if img is None:

        raise FileNotFoundError(
            str(path)
        )

    dark = (
        img
        < sim.line_threshold
    ).astype(
        np.uint8
    ) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            sim.morphology_kernel,
            sim.morphology_kernel,
        ),
    )

    closed = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    closed = cv2.dilate(
        closed,
        kernel,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:

        raise RuntimeError(
            "No tissue contour found in template."
        )

    outer = max(
        contours,
        key=cv2.contourArea,
    )

    x, y, w, h = cv2.boundingRect(
        outer
    )

    pad = 12

    x0 = max(
        0,
        x - pad,
    )

    y0 = max(
        0,
        y - pad,
    )

    x1 = min(
        img.shape[1],
        x + w + pad,
    )

    y1 = min(
        img.shape[0],
        y + h + pad,
    )

    crop = img[
        y0:y1,
        x0:x1,
    ]

    dark_crop = (
        crop
        < sim.line_threshold
    ).astype(
        np.uint8
    ) * 255

    closed_crop = cv2.morphologyEx(
        dark_crop,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    closed_crop = cv2.dilate(
        closed_crop,
        kernel,
        iterations=1,
    )

    contours2, _ = cv2.findContours(
        closed_crop,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    outer2 = max(
        contours2,
        key=cv2.contourArea,
    )

    crop_mask = np.zeros_like(
        crop,
        dtype=np.uint8,
    )

    cv2.drawContours(
        crop_mask,
        [outer2],
        -1,
        255,
        thickness=-1,
    )

    usable = (
        sim.canvas
        - 2 * sim.margin
    )

    scale = min(
        usable / crop.shape[1],
        usable / crop.shape[0],
    )

    new_w = int(
        round(
            crop.shape[1]
            * scale
        )
    )

    new_h = int(
        round(
            crop.shape[0]
            * scale
        )
    )

    crop_r = cv2.resize(
        crop,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    mask_r = cv2.resize(
        crop_mask,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    anatomy = np.full(
        (
            sim.canvas,
            sim.canvas,
        ),
        255,
        dtype=np.uint8,
    )

    mask = np.zeros(
        (
            sim.canvas,
            sim.canvas,
        ),
        dtype=np.uint8,
    )

    ox = (
        sim.canvas
        - new_w
    ) // 2

    oy = (
        sim.canvas
        - new_h
    ) // 2

    anatomy[
        oy:oy + new_h,
        ox:ox + new_w,
    ] = crop_r

    mask[
        oy:oy + new_h,
        ox:ox + new_w,
    ] = mask_r

    dist = distance_transform_edt(
        mask > 0
    ).astype(
        np.float32
    )

    if dist.max() > 0:

        dist /= dist.max()

    fill = (
        243.0
        - 22.0 * dist
    )

    anatomy_f = anatomy.astype(
        np.float32
    )

    inside = (
        mask > 0
    )

    anatomy_f[
        inside
    ] = np.minimum(
        anatomy_f[
            inside
        ],
        fill[
            inside
        ],
    )

    anatomy_f = cv2.GaussianBlur(
        anatomy_f,
        (
            3,
            3,
        ),
        0.5,
    )

    return (
        np.clip(
            anatomy_f,
            0,
            255,
        ).astype(
            np.uint8
        ),
        mask,
    )


# ============================================================
# DEFORMATION
# ============================================================

def deformation_xy(
    x,
    y,
    tau,
    canvas,
    d,
):

    cx = cy = (
        canvas - 1
    ) / 2.0

    half = (
        canvas
        / 2.0
    )

    xn = (
        x - cx
    ) / half

    yn = (
        y - cy
    ) / half

    a = (
        float(tau)
        * d.strength
    )

    xg = (
        (
            1.0
            + d.width_change * a
        )
        * xn
        + d.shear
        * a
        * yn
    )

    yg = (
        1.0
        + d.height_change * a
    ) * yn

    dx = a * (
        d.bend_x
        * np.sin(
            np.pi * yn
        )
        + d.cross_term
        * xn
        * yn
        + d.wave
        * np.sin(
            2.0
            * np.pi
            * yn
        )
    )

    dy = a * (
        d.bend_y
        * np.sin(
            np.pi
            * xn
        )
        + d.linear_y
        * xn
        + d.quadratic_y
        * (
            xn ** 2
            - 0.35
        )
    )

    xo = (
        cx
        + half
        * (
            xg + dx
        )
    )

    yo = (
        cy
        + half
        * (
            yg + dy
        )
    )

    return xo, yo


def deform_points(
    points,
    tau,
    sim,
    d,
):

    xo, yo = deformation_xy(
        points[:, 0],
        points[:, 1],
        tau,
        sim.canvas,
        d,
    )

    return np.column_stack(
        [
            xo,
            yo,
        ]
    ).astype(
        np.float32
    )


def inverse_deformation_map(
    tau,
    sim,
    d,
):

    yy, xx = np.mgrid[
        0:sim.canvas,
        0:sim.canvas,
    ].astype(
        np.float32
    )

    sx = xx.copy()
    sy = yy.copy()

    for _ in range(8):

        fx, fy = deformation_xy(
            sx,
            sy,
            tau,
            sim.canvas,
            d,
        )

        sx += (
            xx - fx
        )

        sy += (
            yy - fy
        )

    return (
        sx.astype(
            np.float32
        ),
        sy.astype(
            np.float32
        ),
    )


# ============================================================
# SPOTS
# ============================================================

def make_hex_spots(
    mask,
    spacing,
):

    h, w = mask.shape

    points = []

    dy = (
        spacing
        * np.sqrt(3.0)
        / 2.0
    )

    row = 0

    y = spacing

    while y < h - spacing:

        offset = (
            spacing / 2.0
            if row % 2
            else 0.0
        )

        x = (
            spacing
            + offset
        )

        while x < w - spacing:

            xi = int(
                round(x)
            )

            yi = int(
                round(y)
            )

            if mask[
                yi,
                xi,
            ] > 0:

                points.append(
                    (
                        x,
                        y,
                    )
                )

            x += spacing

        row += 1

        y += dy

    return np.asarray(
        points,
        dtype=np.float32,
    )


def normalized_xy(
    points,
    canvas,
):

    center = (
        canvas - 1
    ) / 2.0

    half = (
        canvas
        / 2.0
    )

    x = (
        points[:, 0]
        - center
    ) / half

    y = (
        points[:, 1]
        - center
    ) / half

    return x, y


# ============================================================
# MOLECULAR READS
# ============================================================

def molecular_values(
    reference_spot_xy,
    tau,
    sim,
):

    x, y = normalized_xy(
        reference_spot_xy,
        sim.canvas,
    )

    def gauss(
        cx,
        cy,
        sx,
        sy,
    ):

        return np.exp(
            -0.5
            * (
                (
                    (
                        x - cx
                    )
                    / sx
                ) ** 2
                +
                (
                    (
                        y - cy
                    )
                    / sy
                ) ** 2
            )
        )

    p1 = (
        1.05
        - 0.45 * tau
    ) * gauss(
        -0.34
        + 0.08 * tau,
        -0.05,
        0.24,
        0.19,
    )

    p2 = (
        0.30
        + 0.85 * tau
    ) * gauss(
        0.30
        - 0.05 * tau,
        0.30
        - 0.05 * tau,
        0.22,
        0.20,
    )

    p3 = (
        0.20
        * np.clip(
            -y + 0.50,
            0.0,
            1.3,
        )
    )

    z = (
        p1
        + p2
        + p3
    )

    z -= z.min()

    z /= (
        z.max()
        + 1e-12
    )

    return z.astype(
        np.float32
    )


# ============================================================
# SPOT-LEVEL DOMAIN LABELS
# ============================================================

def assign_reference_domain_labels(
    reference_spot_xy,
    mask,
    sim,
):

    x, y = normalized_xy(
        reference_spot_xy,
        sim.canvas,
    )

    distance = distance_transform_edt(
        mask > 0
    )

    px = np.clip(
        np.rint(
            reference_spot_xy[
                :,
                0,
            ]
        ).astype(int),
        0,
        mask.shape[1] - 1,
    )

    py = np.clip(
        np.rint(
            reference_spot_xy[
                :,
                1,
            ]
        ).astype(int),
        0,
        mask.shape[0] - 1,
    )

    d = distance[
        py,
        px,
    ]

    labels = np.full(
        len(
            reference_spot_xy
        ),
        3,
        dtype=np.int16,
    )

    labels[
        d < 70
    ] = 0


    # --------------------------------------------------------
    # Hippocampal-like regions
    # --------------------------------------------------------

    left_hip = (
        (
            (
                x + 0.34
            ) / 0.22
        ) ** 2
        +
        (
            (
                y - 0.28
            ) / 0.15
        ) ** 2
    ) < 1.0

    right_hip = (
        (
            (
                x - 0.34
            ) / 0.22
        ) ** 2
        +
        (
            (
                y - 0.28
            ) / 0.15
        ) ** 2
    ) < 1.0

    hip = (
        (labels != 0)
        & (
            left_hip
            | right_hip
        )
    )

    labels[
        hip
    ] = 1


    # --------------------------------------------------------
    # Deep-gray-like region
    # --------------------------------------------------------

    deep = (
        (
            x / 0.27
        ) ** 2
        +
        (
            (
                y + 0.02
            ) / 0.24
        ) ** 2
    ) < 1.0

    deep = (
        (labels != 0)
        & (labels != 1)
        & deep
    )

    labels[
        deep
    ] = 2

    return labels


# ============================================================
# JOINT SMOOTH PARTITION FROM SPOT LABELS
# ============================================================

def smooth_closed_contour(
    points,
    window,
):

    pts = np.asarray(
        points,
        dtype=np.float32,
    )

    if len(pts) < 12:

        return pts

    window = max(
        5,
        int(window),
    )

    if window % 2 == 0:

        window += 1

    for _ in range(2):

        half = (
            window // 2
        )

        padded = np.vstack(
            [
                pts[-half:],
                pts,
                pts[:half],
            ]
        )

        kernel = (
            np.ones(
                window,
                dtype=np.float32,
            )
            / float(window)
        )

        xs = np.convolve(
            padded[:, 0],
            kernel,
            mode="valid",
        )

        ys = np.convolve(
            padded[:, 1],
            kernel,
            mode="valid",
        )

        pts = np.column_stack(
            [
                xs,
                ys,
            ]
        )

    return pts


def clean_partition_components(
    label_field,
    tissue_mask,
    n_domains,
    min_area,
    close_px,
):

    cleaned = label_field.copy()

    if close_px % 2 == 0:

        close_px += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            close_px,
            close_px,
        ),
    )

    fallback = cleaned.copy()

    for domain_id in range(
        n_domains
    ):

        binary = (
            (
                cleaned
                == domain_id
            )
            & (
                tissue_mask > 0
            )
        ).astype(
            np.uint8
        ) * 255

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        (
            n_components,
            component_labels,
            stats,
            _,
        ) = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        keep = np.zeros_like(
            binary,
            dtype=np.uint8,
        )

        for component_id in range(
            1,
            n_components,
        ):

            area = stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]

            if area >= min_area:

                keep[
                    component_labels
                    == component_id
                ] = 255

        remove = (
            (
                cleaned
                == domain_id
            )
            & (
                keep == 0
            )
            & (
                tissue_mask > 0
            )
        )

        cleaned[
            remove
        ] = -1

    unlabeled = (
        (cleaned < 0)
        & (
            tissue_mask > 0
        )
    )

    if np.any(
        unlabeled
    ):

        seeds = (
            cleaned >= 0
        ).astype(
            np.uint8
        )

        _, labels = cv2.distanceTransformWithLabels(
            1 - seeds,
            distanceType=cv2.DIST_L2,
            maskSize=5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )

        seed_coords = np.argwhere(
            seeds > 0
        )

        if len(seed_coords) > 0:

            ys, xs = np.where(
                unlabeled
            )

            nearest_label_ids = (
                labels[
                    ys,
                    xs,
                ]
                - 1
            )

            nearest_label_ids = np.clip(
                nearest_label_ids,
                0,
                len(seed_coords) - 1,
            )

            nearest_coords = seed_coords[
                nearest_label_ids
            ]

            nearest_y = nearest_coords[
                :,
                0,
            ]

            nearest_x = nearest_coords[
                :,
                1,
            ]

            cleaned[
                ys,
                xs,
            ] = fallback[
                nearest_y,
                nearest_x,
            ]

    return cleaned


def build_joint_domain_partition(
    spot_xy,
    spot_labels,
    tissue_mask,
    sim,
    n_domains=4,
):

    h, w = tissue_mask.shape

    scores = np.zeros(
        (
            n_domains,
            h,
            w,
        ),
        dtype=np.float32,
    )

    total_impulse = np.zeros(
        (
            h,
            w,
        ),
        dtype=np.float32,
    )

    for x, y in spot_xy:

        xi = int(
            round(x)
        )

        yi = int(
            round(y)
        )

        if (
            0 <= xi < w
            and 0 <= yi < h
        ):

            total_impulse[
                yi,
                xi,
            ] += 1.0

    total_density = cv2.GaussianBlur(
        total_impulse,
        (
            0,
            0,
        ),
        sigmaX=sim.domain_score_sigma_px,
        sigmaY=sim.domain_score_sigma_px,
        borderType=cv2.BORDER_CONSTANT,
    ) + 1e-8

    for domain_id in range(
        n_domains
    ):

        impulse = np.zeros(
            (
                h,
                w,
            ),
            dtype=np.float32,
        )

        pts = spot_xy[
            spot_labels
            == domain_id
        ]

        for x, y in pts:

            xi = int(
                round(x)
            )

            yi = int(
                round(y)
            )

            if (
                0 <= xi < w
                and 0 <= yi < h
            ):

                impulse[
                    yi,
                    xi,
                ] += 1.0

        density = cv2.GaussianBlur(
            impulse,
            (
                0,
                0,
            ),
            sigmaX=sim.domain_score_sigma_px,
            sigmaY=sim.domain_score_sigma_px,
            borderType=cv2.BORDER_CONSTANT,
        )

        density = (
            density
            / total_density
        )

        for x, y in pts:

            xi = int(
                round(x)
            )

            yi = int(
                round(y)
            )

            if (
                0 <= xi < w
                and 0 <= yi < h
            ):

                radius = max(
                    2,
                    sim.spot_radius - 2,
                )

                cv2.circle(
                    density,
                    (
                        xi,
                        yi,
                    ),
                    radius,
                    float(
                        sim.domain_spot_anchor_boost
                    ),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

        scores[
            domain_id
        ] = density

    scores[
        :,
        tissue_mask == 0,
    ] = -1e9

    partition = np.argmax(
        scores,
        axis=0,
    ).astype(
        np.int16
    )

    partition[
        tissue_mask == 0
    ] = -1

    partition = clean_partition_components(
        partition,
        tissue_mask,
        n_domains=n_domains,
        min_area=sim.domain_min_component_area_px,
        close_px=sim.domain_partition_close_px,
    )

    return (
        partition,
        scores,
    )


def extract_partition_contours(
    label_field,
    tissue_mask,
    sim,
    n_domains=4,
):

    contours_out = []

    for domain_id in range(
        n_domains
    ):

        binary = (
            (
                label_field
                == domain_id
            )
            & (
                tissue_mask > 0
            )
        ).astype(
            np.uint8
        ) * 255

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if (
                area
                < sim.domain_min_component_area_px
            ):

                continue

            points = contour[
                :,
                0,
                :,
            ].astype(
                np.float32
            )

            points = smooth_closed_contour(
                points,
                sim.domain_contour_smooth_window,
            )

            contours_out.append(
                (
                    domain_id,
                    points,
                )
            )

    return contours_out

# ============================================================
# RGBA RENDERERS
# ============================================================

def anatomy_rgba(
    anatomy,
    mask,
    sim,
):

    rgb = np.repeat(
        anatomy[
            ...,
            None,
        ],
        3,
        axis=2,
    )

    alpha = np.where(
        mask > 0,
        255,
        0,
    ).astype(
        np.uint8
    )

    rgba = np.dstack(
        [
            rgb,
            alpha,
        ]
    )

    contours, _ = cv2.findContours(
        (
            mask > 0
        ).astype(
            np.uint8
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if contours:

        outer = max(
            contours,
            key=cv2.contourArea,
        )

        pts = outer[
            :,
            0,
            :,
        ].astype(
            np.float32
        )

        pts = smooth_closed_contour(
            pts,
            19,
        )

        pts = np.rint(
            pts
        ).astype(
            np.int32
        )

        cv2.polylines(
            rgba,
            [
                pts
            ],
            isClosed=True,
            color=(
                int(
                    ANATOMY_OUTLINE_RGB[0]
                ),
                int(
                    ANATOMY_OUTLINE_RGB[1]
                ),
                int(
                    ANATOMY_OUTLINE_RGB[2]
                ),
                255,
            ),
            thickness=sim.anatomy_outline_thickness_px,
            lineType=cv2.LINE_AA,
        )

    return rgba


def render_domain_contours_rgba(
    contours,
    sim,
):

    rgba = np.zeros(
        (
            sim.canvas,
            sim.canvas,
            4,
        ),
        dtype=np.uint8,
    )

    for domain_id, points in contours:

        color = hex_to_rgb255(
            DOMAIN_COLORS_HEX[
                domain_id
            ]
        )

        pts = np.rint(
            points
        ).astype(
            np.int32
        )

        cv2.polylines(
            rgba,
            [
                pts
            ],
            isClosed=True,
            color=(
                int(
                    color[0]
                ),
                int(
                    color[1]
                ),
                int(
                    color[2]
                ),
                235,
            ),
            thickness=sim.domain_contour_thickness_px,
            lineType=cv2.LINE_AA,
        )

    return rgba


def render_domain_spots_and_contours(
    spot_xy,
    spot_labels,
    contours,
    sim,
):

    rgba = np.zeros(
        (
            sim.canvas,
            sim.canvas,
            4,
        ),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Spot-level domains
    #
    # DOMAIN PALETTE UNCHANGED.
    # --------------------------------------------------------

    for domain_id, hex_color in enumerate(
        DOMAIN_COLORS_HEX
    ):

        color = hex_to_rgb255(
            hex_color
        )

        pts = spot_xy[
            spot_labels
            == domain_id
        ]

        for x, y in pts:

            cv2.circle(
                rgba,
                (
                    int(
                        round(x)
                    ),
                    int(
                        round(y)
                    ),
                ),
                sim.spot_radius,
                (
                    int(
                        color[0]
                    ),
                    int(
                        color[1]
                    ),
                    int(
                        color[2]
                    ),
                    255,
                ),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # Same-color contours
    # --------------------------------------------------------

    for domain_id, points in contours:

        color = hex_to_rgb255(
            DOMAIN_COLORS_HEX[
                domain_id
            ]
        )

        pts = np.rint(
            points
        ).astype(
            np.int32
        )

        cv2.polylines(
            rgba,
            [
                pts
            ],
            isClosed=True,
            color=(
                int(
                    color[0]
                ),
                int(
                    color[1]
                ),
                int(
                    color[2]
                ),
                235,
            ),
            thickness=sim.domain_contour_thickness_px,
            lineType=cv2.LINE_AA,
        )

    return rgba


def render_continuous_domain_rgba(
    label_field,
    tissue_mask,
    contours,
    sim,
):
    """
    Render a pixel-wise continuous spatial-domain map.

    Every in-tissue pixel is filled using exactly the original
    DOMAIN_COLORS_HEX categorical palette.

    The molecular pseudo-time palette does NOT affect this function.
    """

    rgba = np.zeros(
        (
            sim.canvas,
            sim.canvas,
            4,
        ),
        dtype=np.uint8,
    )

    inside = (
        tissue_mask > 0
    )

    # --------------------------------------------------------
    # Fill every tissue pixel with categorical domain color
    # --------------------------------------------------------

    for domain_id, hex_color in enumerate(
        DOMAIN_COLORS_HEX
    ):

        color = hex_to_rgb255(
            hex_color
        )

        region = (
            inside
            & (
                label_field
                == domain_id
            )
        )

        rgba[
            region,
            0,
        ] = int(
            color[0]
        )

        rgba[
            region,
            1,
        ] = int(
            color[1]
        )

        rgba[
            region,
            2,
        ] = int(
            color[2]
        )

        rgba[
            region,
            3,
        ] = 220

    # --------------------------------------------------------
    # Overlay smooth domain contours
    # --------------------------------------------------------

    for domain_id, points in contours:

        color = hex_to_rgb255(
            DOMAIN_COLORS_HEX[
                domain_id
            ]
        )

        pts = np.rint(
            points
        ).astype(
            np.int32
        )

        cv2.polylines(
            rgba,
            [
                pts
            ],
            isClosed=True,
            color=(
                int(
                    color[0]
                ),
                int(
                    color[1]
                ),
                int(
                    color[2]
                ),
                255,
            ),
            thickness=sim.domain_contour_thickness_px,
            lineType=cv2.LINE_AA,
        )

    rgba[
        ~inside,
        3,
    ] = 0

    return rgba


# ============================================================
# MOLECULAR RENDERER
#
# NEW:
# Uses a pseudo-time-specific sequential palette.
# ============================================================

def molecular_rgba(
    spot_xy,
    values,
    sim,
    state_index,
):
    """
    Render molecular spots using the pseudo-time-specific palette.

    Low -> high abundance remains represented by a sequential
    light-to-saturated gradient within each pseudo-time state.
    """

    rgba = np.zeros(
        (
            sim.canvas,
            sim.canvas,
            4,
        ),
        dtype=np.uint8,
    )

    cmap = pseudotime_colormap(
        state_index,
        sim.n_states,
    )

    display_values = enhance_molecular_display(
        values
    )

    colors = cmap(
        display_values
    )

    for (
        x,
        y,
    ), color in zip(
        spot_xy,
        colors,
    ):

        rgb = tuple(
            int(
                round(
                    value * 255
                )
            )
            for value in color[:3]
        )

        cv2.circle(
            rgba,
            (
                int(
                    round(x)
                ),
                int(
                    round(y)
                ),
            ),
            sim.spot_radius,
            (
                rgb[0],
                rgb[1],
                rgb[2],
                255,
            ),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    return rgba


# ============================================================
# PSEUDO-OMICS RENDERER
#
# NEW:
# Uses exactly the same pseudo-time palette as molecular spots.
# ============================================================

def pseudo_omics_rgba(
    field,
    state_index,
    n_states,
):
    """
    Render continuous pseudo-omics using the same state-specific
    molecular palette.

    Therefore a given pseudo-time has consistent visual identity
    between:
        molecular spots
        pseudo-omics field
    """

    cmap = pseudotime_colormap(
        state_index,
        n_states,
    )

    display_field = enhance_molecular_display(
        np.nan_to_num(
            field,
            nan=0.0,
        )
    )

    mapped = cmap(
        display_field
    )

    rgba = np.zeros(
        (
            field.shape[0],
            field.shape[1],
            4,
        ),
        dtype=np.uint8,
    )

    rgba[
        ...,
        :3,
    ] = (
        mapped[
            ...,
            :3,
        ]
        * 255
    ).astype(
        np.uint8
    )

    rgba[
        ...,
        3,
    ] = np.where(
        np.isfinite(
            field
        ),
        255,
        0,
    ).astype(
        np.uint8
    )

    return rgba


# ============================================================
# PSEUDO-OMICS RECONSTRUCTION
# ============================================================

def render_pseudo_omics(
    spot_xy,
    values,
    tissue_mask,
    sigma_px,
):

    h, w = tissue_mask.shape

    numerator = np.zeros(
        (
            h,
            w,
        ),
        dtype=np.float32,
    )

    denominator = np.zeros(
        (
            h,
            w,
        ),
        dtype=np.float32,
    )

    for (
        x,
        y,
    ), value in zip(
        spot_xy,
        values,
    ):

        xi = int(
            round(x)
        )

        yi = int(
            round(y)
        )

        if (
            0 <= xi < w
            and 0 <= yi < h
        ):

            numerator[
                yi,
                xi,
            ] += float(
                value
            )

            denominator[
                yi,
                xi,
            ] += 1.0

    numerator = cv2.GaussianBlur(
        numerator,
        (
            0,
            0,
        ),
        sigmaX=sigma_px,
        sigmaY=sigma_px,
        borderType=cv2.BORDER_CONSTANT,
    )

    denominator = cv2.GaussianBlur(
        denominator,
        (
            0,
            0,
        ),
        sigmaX=sigma_px,
        sigmaY=sigma_px,
        borderType=cv2.BORDER_CONSTANT,
    )

    field = (
        numerator
        / (
            denominator
            + 1e-8
        )
    )

    inside = (
        tissue_mask > 0
    )

    valid = field[
        inside
    ]

    if valid.size:

        lo, hi = np.percentile(
            valid,
            [
                1,
                99,
            ],
        )

        field = np.clip(
            (
                field - lo
            )
            / (
                hi
                - lo
                + 1e-8
            ),
            0.0,
            1.0,
        )

    field = field.astype(
        np.float32
    )

    field[
        ~inside
    ] = np.nan

    return field

# ============================================================
# OVERVIEW FIGURES
# ============================================================

def save_overview(
    path,
    records,
):

    n = len(
        records
    )

    fig, axes = plt.subplots(
        5,
        n,
        figsize=(
            2.1 * n,
            10.0,
        ),
    )

    if n == 1:

        axes = axes[
            :,
            None,
        ]

    names = [
        "Anatomy",
        "Spot domain",
        "Continuous domain",
        "Molecular",
        "Pseudo-omics",
    ]

    for col, rec in enumerate(
        records
    ):

        paths = [
            rec[
                "anatomy"
            ],
            rec[
                "domain"
            ],
            rec[
                "domain_continuous"
            ],
            rec[
                "molecular"
            ],
            rec[
                "pseudo"
            ],
        ]

        for row, image_path in enumerate(
            paths
        ):

            image = cv2.imread(
                str(
                    image_path
                ),
                cv2.IMREAD_UNCHANGED,
            )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2RGBA,
            )

            axes[
                row,
                col,
            ].imshow(
                image
            )

            axes[
                row,
                col,
            ].axis(
                "off"
            )

            if col == 0:

                axes[
                    row,
                    col,
                ].set_ylabel(
                    names[
                        row
                    ],
                    fontsize=9,
                )

        axes[
            0,
            col,
        ].set_title(
            r"$\tau={:.2f}$".format(
                rec[
                    "tau"
                ]
            ),
            fontsize=9,
        )

    plt.tight_layout(
        pad=0.3,
        w_pad=0.1,
        h_pad=0.2,
    )

    fig.savefig(
        path,
        dpi=250,
        transparent=True,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_matched_overview(
    path,
    shared_domain_png,
    matched_records,
):
    """
    QC overview for spatially matched states.

    Top row:
        shared domain

    Bottom row:
        molecular reads from each pseudo-time

    Molecular colors therefore show the same pseudo-time palette
    progression as the native tissue outputs.
    """

    n = len(
        matched_records
    )

    fig, axes = plt.subplots(
        2,
        n,
        figsize=(
            2.1 * n,
            4.2,
        ),
    )

    if n == 1:

        axes = axes[
            :,
            None,
        ]

    shared = cv2.imread(
        str(
            shared_domain_png
        ),
        cv2.IMREAD_UNCHANGED,
    )

    shared = cv2.cvtColor(
        shared,
        cv2.COLOR_BGRA2RGBA,
    )

    for col, rec in enumerate(
        matched_records
    ):

        axes[
            0,
            col,
        ].imshow(
            shared
        )

        axes[
            0,
            col,
        ].axis(
            "off"
        )

        molecular = cv2.imread(
            str(
                rec[
                    "molecular"
                ]
            ),
            cv2.IMREAD_UNCHANGED,
        )

        molecular = cv2.cvtColor(
            molecular,
            cv2.COLOR_BGRA2RGBA,
        )

        axes[
            1,
            col,
        ].imshow(
            molecular
        )

        axes[
            1,
            col,
        ].axis(
            "off"
        )

        axes[
            0,
            col,
        ].set_title(
            r"$\tau={:.2f}$".format(
                rec[
                    "tau"
                ]
            ),
            fontsize=9,
        )

        if col == 0:

            axes[
                0,
                col,
            ].set_ylabel(
                "Shared domain",
                fontsize=9,
            )

            axes[
                1,
                col,
            ].set_ylabel(
                "Molecular",
                fontsize=9,
            )

    plt.tight_layout(
        pad=0.3,
        w_pad=0.1,
        h_pad=0.2,
    )

    fig.savefig(
        path,
        dpi=250,
        transparent=True,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# MAIN DATASET GENERATION
# ============================================================

def generate(
    template_path,
    output_root,
    sim,
    deform,
):

    data_root = (
        output_root
        / "simu_data"
    )

    data_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Reference anatomy and spots
    # --------------------------------------------------------

    anatomy_ref, mask_ref = read_and_square_template(
        template_path,
        sim,
    )

    reference_spots = make_hex_spots(
        mask_ref,
        sim.spot_spacing,
    )

    reference_domain_labels = assign_reference_domain_labels(
        reference_spots,
        mask_ref,
        sim,
    )

    tau_values = np.linspace(
        0.0,
        1.0,
        sim.n_states,
        dtype=np.float32,
    )

    np.savez_compressed(
        data_root
        / "reference_data.npz",

        anatomy_reference=anatomy_ref,

        mask_reference=mask_ref,

        reference_spot_xy=reference_spots,

        reference_domain_labels=reference_domain_labels,

        tau_values=tau_values,
    )


    # ========================================================
    # SPATIALLY MATCHED STATES
    # ========================================================
    #
    # All matched states:
    #
    #     same reference coordinates
    #     same shared domain
    #
    # Only molecular abundance changes with pseudo-time.
    # ========================================================

    matched_root = (
        data_root
        / "matched_states"
    )

    matched_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    shared_partition, shared_scores = build_joint_domain_partition(
        reference_spots,
        reference_domain_labels,
        mask_ref,
        sim,
        n_domains=4,
    )


    shared_domain_contours = extract_partition_contours(
        shared_partition,
        mask_ref,
        sim,
        n_domains=4,
    )


    shared_domain_rgba = render_domain_spots_and_contours(
        reference_spots,
        reference_domain_labels,
        shared_domain_contours,
        sim,
    )


    shared_boundary_rgba = render_domain_contours_rgba(
        shared_domain_contours,
        sim,
    )


    shared_continuous_domain_rgba = render_continuous_domain_rgba(
        shared_partition,
        mask_ref,
        shared_domain_contours,
        sim,
    )


    shared_domain_png = (
        matched_root
        / "shared_domain.png"
    )

    shared_continuous_domain_png = (
        matched_root
        / "shared_domain_continuous.png"
    )

    shared_boundary_png = (
        matched_root
        / "shared_domain_boundaries.png"
    )


    save_rgba_png(
        shared_domain_rgba,
        shared_domain_png,
    )


    save_rgba_png(
        shared_continuous_domain_rgba,
        shared_continuous_domain_png,
    )


    save_rgba_png(
        shared_boundary_rgba,
        shared_boundary_png,
    )


    shared_boundary_mask = (
        shared_boundary_rgba[
            ...,
            3,
        ]
        > 0
    ).astype(
        np.uint8
    ) * 255


    np.savez_compressed(
        matched_root
        / "shared_domain_data.npz",

        matched_spot_xy=reference_spots,

        domain_labels_at_spots=reference_domain_labels,

        domain_partition=shared_partition,

        domain_scores=shared_scores,

        domain_contour_mask=shared_boundary_mask,

        tissue_mask=mask_ref,
    )


    records = []

    matched_records = []


    # ========================================================
    # PSEUDO-TIME LOOP
    # ========================================================

    for state_index, tau in enumerate(
        tau_values,
        start=1,
    ):

        state_dir = (
            data_root
            / f"tau_{state_index:02d}"
        )

        state_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        # ----------------------------------------------------
        # Deformed anatomy
        # ----------------------------------------------------

        map_x, map_y = inverse_deformation_map(
            float(
                tau
            ),
            sim,
            deform,
        )


        anatomy_native = cv2.remap(
            anatomy_ref,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )


        mask_native = cv2.remap(
            mask_ref,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )


        native_spots = deform_points(
            reference_spots,
            float(
                tau
            ),
            sim,
            deform,
        )


        displacement = (
            native_spots
            - reference_spots
        ).astype(
            np.float32
        )


        # ----------------------------------------------------
        # Molecular values
        # ----------------------------------------------------

        molecular_read = molecular_values(
            reference_spots,
            float(
                tau
            ),
            sim,
        )


        # ====================================================
        # MATCHED MOLECULAR STATE
        # ====================================================
        #
        # Same shared reference coordinates for every time.
        # Each pseudo-time receives its own palette.
        # ====================================================

        matched_state_dir = (
            matched_root
            / f"tau_{state_index:02d}"
        )

        matched_state_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        matched_molecular_png = (
            matched_state_dir
            / "molecular.png"
        )


        save_rgba_png(
            molecular_rgba(
                reference_spots,
                molecular_read,
                sim,
                state_index,
            ),
            matched_molecular_png,
        )


        np.savez_compressed(
            matched_state_dir
            / "data.npz",

            state_index=np.int16(
                state_index
            ),

            tau=np.float32(
                tau
            ),

            matched_spot_xy=reference_spots,

            molecular_read=molecular_read,

            domain_labels_at_spots=reference_domain_labels,
        )


        matched_records.append(
            {
                "tau": float(
                    tau
                ),

                "molecular": matched_molecular_png,
            }
        )


        # ----------------------------------------------------
        # Continuous pseudo-omics field
        # ----------------------------------------------------

        pseudo_omics = render_pseudo_omics(
            native_spots,
            molecular_read,
            mask_native,
            sim.pseudo_sigma_px,
        )


        # ----------------------------------------------------
        # Domains
        #
        # Domain processing is unchanged.
        # ----------------------------------------------------

        joint_partition, joint_scores = build_joint_domain_partition(
            native_spots,
            reference_domain_labels,
            mask_native,
            sim,
            n_domains=4,
        )


        domain_contours = extract_partition_contours(
            joint_partition,
            mask_native,
            sim,
            n_domains=4,
        )


        contour_rgba = render_domain_contours_rgba(
            domain_contours,
            sim,
        )


        contour_mask = (
            contour_rgba[
                ...,
                3,
            ]
            > 0
        ).astype(
            np.uint8
        ) * 255


        continuous_domain_rgba = render_continuous_domain_rgba(
            joint_partition,
            mask_native,
            domain_contours,
            sim,
        )


        contours, _ = cv2.findContours(
            (
                mask_native > 0
            ).astype(
                np.uint8
            ),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )


        native_outline_xy = max(
            contours,
            key=cv2.contourArea,
        ).reshape(
            -1,
            2,
        ).astype(
            np.float32
        )


        # ----------------------------------------------------
        # Output paths
        # ----------------------------------------------------

        p_anatomy = (
            state_dir
            / "anatomy.png"
        )

        p_domain = (
            state_dir
            / "domain.png"
        )

        p_domain_continuous = (
            state_dir
            / "domain_continuous.png"
        )

        p_boundary = (
            state_dir
            / "domain_boundaries.png"
        )

        p_molecular = (
            state_dir
            / "molecular.png"
        )

        p_pseudo = (
            state_dir
            / "pseudo_omics.png"
        )


        # ----------------------------------------------------
        # Anatomy
        # ----------------------------------------------------

        save_rgba_png(
            anatomy_rgba(
                anatomy_native,
                mask_native,
                sim,
            ),
            p_anatomy,
        )


        # ----------------------------------------------------
        # Spot domains
        # ----------------------------------------------------

        save_rgba_png(
            render_domain_spots_and_contours(
                native_spots,
                reference_domain_labels,
                domain_contours,
                sim,
            ),
            p_domain,
        )


        # ----------------------------------------------------
        # Continuous domains
        # ----------------------------------------------------

        save_rgba_png(
            continuous_domain_rgba,
            p_domain_continuous,
        )


        save_rgba_png(
            contour_rgba,
            p_boundary,
        )


        # ====================================================
        # MOLECULAR OUTPUT
        #
        # State-specific palette
        # ====================================================

        save_rgba_png(
            molecular_rgba(
                native_spots,
                molecular_read,
                sim,
                state_index,
            ),
            p_molecular,
        )


        # ====================================================
        # PSEUDO-OMICS OUTPUT
        #
        # Same state-specific palette
        # ====================================================

        save_rgba_png(
            pseudo_omics_rgba(
                pseudo_omics,
                state_index,
                sim.n_states,
            ),
            p_pseudo,
        )


        # ----------------------------------------------------
        # Numerical data
        # ----------------------------------------------------

        np.savez_compressed(
            state_dir
            / "data.npz",

            state_index=np.int16(
                state_index
            ),

            tau=np.float32(
                tau
            ),

            anatomy=anatomy_native,

            tissue_mask=mask_native,

            reference_spot_xy=reference_spots,

            native_spot_xy=native_spots,

            spot_displacement=displacement,

            molecular_read=molecular_read,

            domain_labels_at_spots=reference_domain_labels,

            domain_partition=joint_partition,

            domain_contour_mask=contour_mask,

            domain_scores=joint_scores,

            pseudo_omics=pseudo_omics,

            native_outline_xy=native_outline_xy,
        )


        records.append(
            {
                "tau": float(
                    tau
                ),

                "anatomy": p_anatomy,

                "domain": p_domain,

                "domain_continuous": p_domain_continuous,

                "molecular": p_molecular,

                "pseudo": p_pseudo,
            }
        )

    # ========================================================
    # MATCHED OVERVIEW
    # ========================================================

    save_matched_overview(
        matched_root
        / "overview.png",
        shared_domain_png,
        matched_records,
    )


    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "template": str(
            template_path
        ),

        "simulation": asdict(
            sim
        ),

        "deformation": asdict(
            deform
        ),

        "domain_representation": (
            "spot-level categorical labels plus a pixel-wise continuous "
            "domain map; both use the same DOMAIN_COLORS_HEX palette and "
            "smooth non-overlapping partition inferred from all domain spots"
        ),

        "anatomy_outline_color_rgb": list(
            ANATOMY_OUTLINE_RGB
        ),

        # ----------------------------------------------------
        # Molecular palette information
        # ----------------------------------------------------

        "molecular_pseudotime_palette": {

            "description": (
                "state-specific sequential molecular palettes transitioning "
                "from jade green to teal to indigo blue to violet to rose pink"
            ),

            "applied_to": [
                "native molecular spots",
                "native pseudo-omics field",
                "matched-state molecular spots",
            ],

            "domain_colors_unchanged": True,

            "anchor_palettes": (
                PSEUDOTIME_PALETTE_STOPS_HEX
            ),
        },

        "matched_states": {

            "folder": (
                "matched_states"
            ),

            "shared_domain": (
                "matched_states/"
                "shared_domain.png"
            ),

            "shared_domain_continuous": (
                "matched_states/"
                "shared_domain_continuous.png"
            ),

            "shared_domain_boundaries": (
                "matched_states/"
                "shared_domain_boundaries.png"
            ),

            "representation": (
                "all pseudo-time states share reference_spot_xy and one "
                "reference domain annotation; molecular reads vary by pseudo-time"
            ),
        },

        "domain_labels": {

            "0": (
                "cortex_like"
            ),

            "1": (
                "hippocampal_like"
            ),

            "2": (
                "deep_gray_like"
            ),

            "3": (
                "inner_white_matter_like"
            ),
        },
    }


    with open(
        data_root
        / "metadata.json",
        "w",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


    # ========================================================
    # MAIN OVERVIEW
    # ========================================================

    save_overview(
        data_root
        / "overview.png",
        records,
    )


    print(
        "Saved reusable simulation dataset to:\n"
        f"  {data_root}\n"
        f"States: {sim.n_states}\n"
        f"Spots/state: {len(reference_spots)}"
    )


# ============================================================
# CLI
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser()


    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_TEMPLATE,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )


    # --------------------------------------------------------
    # General simulation
    # --------------------------------------------------------

    parser.add_argument(
        "--canvas",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--margin",
        type=int,
        default=70,
    )

    parser.add_argument(
        "--n-states",
        type=int,
        default=5,
    )


    # --------------------------------------------------------
    # Spots
    # --------------------------------------------------------

    parser.add_argument(
        "--spot-spacing",
        type=float,
        default=42.0,
    )

    parser.add_argument(
        "--spot-radius",
        type=int,
        default=15,
    )


    # --------------------------------------------------------
    # Pseudo-omics
    # --------------------------------------------------------

    parser.add_argument(
        "--pseudo-sigma",
        type=float,
        default=48.0,
    )


    # --------------------------------------------------------
    # Spatial domains
    # --------------------------------------------------------

    parser.add_argument(
        "--domain-score-sigma",
        type=float,
        default=58.0,
    )

    parser.add_argument(
        "--domain-spot-anchor-boost",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--domain-contour-window",
        type=int,
        default=31,
    )

    parser.add_argument(
        "--domain-contour-thickness",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--domain-partition-close",
        type=int,
        default=19,
    )

    parser.add_argument(
        "--domain-min-component-area",
        type=int,
        default=2500,
    )


    # --------------------------------------------------------
    # Anatomy
    # --------------------------------------------------------

    parser.add_argument(
        "--anatomy-outline-thickness",
        type=int,
        default=4,
    )


    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=17,
    )


    # --------------------------------------------------------
    # Deformation
    # --------------------------------------------------------

    parser.add_argument(
        "--deform-strength",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--width-change",
        type=float,
        default=0.045,
    )

    parser.add_argument(
        "--height-change",
        type=float,
        default=-0.025,
    )

    parser.add_argument(
        "--shear",
        type=float,
        default=0.030,
    )

    parser.add_argument(
        "--bend-x",
        type=float,
        default=0.070,
    )

    parser.add_argument(
        "--bend-y",
        type=float,
        default=0.055,
    )

    parser.add_argument(
        "--cross-term",
        type=float,
        default=0.028,
    )

    parser.add_argument(
        "--wave",
        type=float,
        default=0.018,
    )

    parser.add_argument(
        "--linear-y",
        type=float,
        default=-0.030,
    )

    parser.add_argument(
        "--quadratic-y",
        type=float,
        default=0.018,
    )


    return parser


# ============================================================
# MAIN
# ============================================================

def main():

    args = (
        build_parser()
        .parse_args()
    )


    sim = SimulationParameters(

        canvas=args.canvas,

        margin=args.margin,

        n_states=args.n_states,

        spot_spacing=args.spot_spacing,

        spot_radius=args.spot_radius,

        pseudo_sigma_px=args.pseudo_sigma,

        domain_score_sigma_px=(
            args.domain_score_sigma
        ),

        domain_spot_anchor_boost=(
            args.domain_spot_anchor_boost
        ),

        domain_contour_smooth_window=(
            args.domain_contour_window
        ),

        domain_contour_thickness_px=(
            args.domain_contour_thickness
        ),

        domain_partition_close_px=(
            args.domain_partition_close
        ),

        domain_min_component_area_px=(
            args.domain_min_component_area
        ),

        anatomy_outline_thickness_px=(
            args.anatomy_outline_thickness
        ),

        seed=args.seed,
    )


    deform = DeformationParameters(

        strength=(
            args.deform_strength
        ),

        width_change=(
            args.width_change
        ),

        height_change=(
            args.height_change
        ),

        shear=(
            args.shear
        ),

        bend_x=(
            args.bend_x
        ),

        bend_y=(
            args.bend_y
        ),

        cross_term=(
            args.cross_term
        ),

        wave=(
            args.wave
        ),

        linear_y=(
            args.linear_y
        ),

        quadratic_y=(
            args.quadratic_y
        ),
    )


    generate(
        args.input,
        args.output,
        sim,
        deform,
    )


if __name__ == "__main__":

    main()