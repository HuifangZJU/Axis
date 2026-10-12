#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate three same-sized schematic symbols for spatial data layouts:

1. Pixel/grid layout
2. Visium-like hexagonal spot layout
3. Single-cell layout with randomly connected cells

Default output folder:
    /media/huifang/data/registration/paper_analysis/Figure1/
"""

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


# ============================================================
# OUTPUT / GLOBAL STYLE
# ============================================================

OUTPUT_DIR = "/media/huifang/data/registration/paper_analysis/Figure1/"

# All three symbols use exactly the same canvas and box geometry.
FIGSIZE = (2.6, 2.6)
DPI = 300

BOX_LEFT = 0.06
BOX_BOTTOM = 0.06
BOX_SIZE = 0.88

BOX_LINEWIDTH = 1.8
BOX_COLOR = "#000000"
BACKGROUND_COLOR = "white"


# ============================================================
# FIGURE 1: PIXEL LAYOUT
# ============================================================

# Pixel side length in normalized axis coordinates.
# Smaller value -> more/smaller pixels.
PIXEL_SIZE = 0.15

PIXEL_LINEWIDTH = 1.5
PIXEL_LINECOLOR = "#000000"


# ============================================================
# FIGURE 2: VISIUM-LIKE LAYOUT
# ============================================================

# Center-to-center horizontal spot distance.
VISIUM_SPOT_GAP = 0.18

# Spot radius in normalized axis coordinates.
VISIUM_SPOT_RADIUS = 0.06

VISIUM_SPOT_FACE = "#DCE4EA"
VISIUM_SPOT_EDGE = "#000000"
VISIUM_SPOT_EDGEWIDTH = 0.65


# ============================================================
# FIGURE 3: SINGLE-CELL LAYOUT
# ============================================================

N_CELLS = 20
CELL_RADIUS = 0.04
N_CONNECTIONS = 12
RANDOM_SEED = 8

CELL_EDGE_COLOR = "#000000"
CELL_EDGE_WIDTH = 1.15

# Light, elegant palette.
CELL_COLORS = [
    "#DDE9F2",  # pale blue
    "#D7EADF",  # pale mint
    "#DDD8EF",  # pale lavender
    "#EEDBD3",  # pale peach
    "#D5E3F3",  # powder blue
    "#E9DDEF",  # pale lilac
    "#E2EAD6",  # pale sage
]

CONNECTION_COLOR = "#969BA3"
CONNECTION_WIDTH = 1.10
CONNECTION_ALPHA = 0.85

# Minimum center-to-center distance between random cells.
CELL_MIN_DISTANCE = 0.12


# ============================================================
# COMMON HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setup_symbol_axis(ax):
    """Create the same square frame for all three symbols."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor(BACKGROUND_COLOR)

    ax.add_patch(
        Rectangle(
            (BOX_LEFT, BOX_BOTTOM),
            BOX_SIZE,
            BOX_SIZE,
            facecolor="none",
            edgecolor=BOX_COLOR,
            linewidth=BOX_LINEWIDTH,
            zorder=10,
        )
    )


def save_symbol(fig, outpath):
    """
    Save with fixed canvas dimensions.

    Do NOT use bbox_inches='tight' here, because that can make the three
    exported images have slightly different pixel dimensions.
    """
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(
        outpath,
        dpi=DPI,
        facecolor=BACKGROUND_COLOR,
        edgecolor="none",
    )
    plt.close(fig)
    print(f"[SAVE] {outpath}")


# ============================================================
# FIGURE 1: PIXEL / GRID SYMBOL
# ============================================================

def plot_pixel_symbol(
    outpath,
    pixel_size=PIXEL_SIZE,
):
    """
    Draw a regular pixel grid inside the common square box.

    pixel_size is the approximate pixel side length in normalized coordinates.
    """
    if pixel_size <= 0:
        raise ValueError("pixel_size must be > 0.")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    setup_symbol_axis(ax)

    x0 = BOX_LEFT
    y0 = BOX_BOTTOM
    x1 = BOX_LEFT + BOX_SIZE
    y1 = BOX_BOTTOM + BOX_SIZE

    # Convert requested approximate pixel size to an integer number of cells,
    # then use equal-width cells so the grid exactly fills the frame.
    n_cells = max(1, int(round(BOX_SIZE / float(pixel_size))))
    actual_pixel_size = BOX_SIZE / n_cells

    # Internal grid lines only; outer border is already supplied by the box.
    for i in range(1, n_cells):
        x = x0 + i * actual_pixel_size
        y = y0 + i * actual_pixel_size

        ax.plot(
            [x, x],
            [y0, y1],
            color=PIXEL_LINECOLOR,
            linewidth=PIXEL_LINEWIDTH,
            zorder=2,
        )
        ax.plot(
            [x0, x1],
            [y, y],
            color=PIXEL_LINECOLOR,
            linewidth=PIXEL_LINEWIDTH,
            zorder=2,
        )

    save_symbol(fig, outpath)


# ============================================================
# FIGURE 2: VISIUM-LIKE SPOT SYMBOL
# ============================================================

def generate_visium_points(
    spot_gap=VISIUM_SPOT_GAP,
    spot_radius=VISIUM_SPOT_RADIUS,
):
    """
    Generate a staggered hexagonal Visium-like spot arrangement.
    """
    if spot_gap <= 0:
        raise ValueError("spot_gap must be > 0.")
    if spot_radius <= 0:
        raise ValueError("spot_radius must be > 0.")

    xmin = BOX_LEFT + spot_radius + 0.025
    xmax = BOX_LEFT + BOX_SIZE - spot_radius - 0.025
    ymin = BOX_BOTTOM + spot_radius + 0.025
    ymax = BOX_BOTTOM + BOX_SIZE - spot_radius - 0.025

    dy = spot_gap * math.sqrt(3.0) / 2.0

    points = []
    row = 0
    y = ymin

    while y <= ymax + 1e-12:
        offset = 0.0 if row % 2 == 0 else spot_gap / 2.0
        x = xmin + offset

        while x <= xmax + 1e-12:
            points.append((x, y))
            x += spot_gap

        y += dy
        row += 1

    return np.asarray(points, dtype=float)


def plot_visium_symbol(
    outpath,
    spot_gap=VISIUM_SPOT_GAP,
    spot_radius=VISIUM_SPOT_RADIUS,
):
    """
    Draw a Visium-like staggered spot layout.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    setup_symbol_axis(ax)

    points = generate_visium_points(
        spot_gap=spot_gap,
        spot_radius=spot_radius,
    )

    for x, y in points:
        ax.add_patch(
            Circle(
                (x, y),
                radius=spot_radius,
                facecolor=VISIUM_SPOT_FACE,
                edgecolor=VISIUM_SPOT_EDGE,
                linewidth=VISIUM_SPOT_EDGEWIDTH,
                zorder=3,
            )
        )

    save_symbol(fig, outpath)


# ============================================================
# FIGURE 3: SINGLE-CELL SYMBOL
# ============================================================

def sample_cell_positions(
    n_cells=N_CELLS,
    radius=CELL_RADIUS,
    min_distance=CELL_MIN_DISTANCE,
    seed=RANDOM_SEED,
):
    """
    Sample non-overlapping-ish cell centers inside the common square frame.
    """
    if n_cells < 1:
        raise ValueError("n_cells must be >= 1.")

    rng = np.random.default_rng(seed)

    margin = radius + 0.045
    xmin = BOX_LEFT + margin
    xmax = BOX_LEFT + BOX_SIZE - margin
    ymin = BOX_BOTTOM + margin
    ymax = BOX_BOTTOM + BOX_SIZE - margin

    points = []
    max_tries = 10000

    for _ in range(max_tries):
        if len(points) >= n_cells:
            break

        candidate = np.array(
            [
                rng.uniform(xmin, xmax),
                rng.uniform(ymin, ymax),
            ],
            dtype=float,
        )

        if not points:
            points.append(candidate)
            continue

        current = np.asarray(points)
        d = np.sqrt(np.sum((current - candidate) ** 2, axis=1))

        if np.all(d >= min_distance):
            points.append(candidate)

    if len(points) < n_cells:
        raise RuntimeError(
            f"Could only place {len(points)} of {n_cells} cells. "
            "Reduce CELL_MIN_DISTANCE or CELL_RADIUS."
        )

    return np.asarray(points)


def _orientation(a, b, c, eps=1e-12):
    """Orientation of ordered triplet (a, b, c)."""
    value = (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )
    if abs(value) <= eps:
        return 0
    return 1 if value > 0 else -1


def _segments_cross(p1, p2, q1, q2):
    """
    Return True when two line segments intersect/cross.

    Connections that share a cell endpoint are handled separately and are
    allowed; this function is used only for edges without a shared endpoint.
    """
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    # Proper crossing.
    if o1 != o2 and o3 != o4:
        return True

    # Collinear/touching cases are also rejected to prevent overlapping lines.
    def on_segment(a, b, p, eps=1e-12):
        return (
            min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
        )

    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, p2, q2):
        return True
    if o3 == 0 and on_segment(q1, q2, p1):
        return True
    if o4 == 0 and on_segment(q1, q2, p2):
        return True

    return False


def _segment_point_distance(a, b, p):
    """Shortest Euclidean distance from point p to segment a-b."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-15:
        return float(np.linalg.norm(p - a))

    t = float(np.dot(p - a, ab) / denom)
    t = np.clip(t, 0.0, 1.0)
    projection = a + t * ab
    return float(np.linalg.norm(p - projection))


def choose_cell_connections(
    points,
    n_connections=N_CONNECTIONS,
    seed=RANDOM_SEED,
    cell_radius=CELL_RADIUS,
):
    """
    Select a sparse set of short cell-cell connections with no line crossings.

    Rules
    -----
    1. Prefer nearby cells.
    2. Reject any candidate edge that crosses or overlaps an accepted edge.
    3. Shared endpoints are allowed, so multiple connections can meet at a cell.
    4. Reject lines that pass through an unrelated cell.
    """
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=float)
    n = len(points)

    if n < 2 or n_connections <= 0:
        return []

    # Build candidate edges from several nearest neighbors.
    candidates = []
    for i in range(n):
        dist2 = np.sum((points - points[i]) ** 2, axis=1)
        nearest = np.argsort(dist2)[1:min(6, n)]

        for j in nearest:
            a, b = sorted((i, int(j)))
            if a != b:
                length = float(np.sqrt(dist2[j]))
                candidates.append((a, b, length))

    # Remove duplicate edges.
    unique = {}
    for a, b, length in candidates:
        unique[(a, b)] = min(length, unique.get((a, b), np.inf))
    candidates = [(a, b, d) for (a, b), d in unique.items()]

    # Prefer short edges, with a small deterministic random jitter so the icon
    # does not look mechanically nearest-neighbor.
    scored = []
    for a, b, length in candidates:
        jitter = float(rng.uniform(0.0, 0.08))
        scored.append((length * (1.0 + jitter), a, b))
    scored.sort(key=lambda item: item[0])

    chosen = []
    clearance = 1.15 * float(cell_radius)

    for _, a, b in scored:
        p1 = points[a]
        p2 = points[b]

        # Avoid drawing a connection through a third cell.
        passes_through_cell = False
        for k, p in enumerate(points):
            if k in (a, b):
                continue
            if _segment_point_distance(p1, p2, p) < clearance:
                passes_through_cell = True
                break
        if passes_through_cell:
            continue

        # Reject crossings/overlaps with existing edges.
        intersects_existing = False
        for c, d in chosen:
            # Edges meeting at the same cell are allowed.
            if len({a, b, c, d}) < 4:
                continue

            if _segments_cross(
                points[a], points[b],
                points[c], points[d],
            ):
                intersects_existing = True
                break

        if intersects_existing:
            continue

        chosen.append((a, b))
        if len(chosen) >= int(n_connections):
            break

    return chosen


def plot_singlecell_symbol(
    outpath,
    n_cells=N_CELLS,
    cell_radius=CELL_RADIUS,
    n_connections=N_CONNECTIONS,
    seed=RANDOM_SEED,
):
    """
    Draw randomly placed colored single cells and a sparse connection network.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    setup_symbol_axis(ax)

    points = sample_cell_positions(
        n_cells=n_cells,
        radius=cell_radius,
        seed=seed,
    )

    edges = choose_cell_connections(
        points,
        n_connections=n_connections,
        seed=seed,
        cell_radius=cell_radius,
    )

    # Connections first so they remain behind cells.
    for i, j in edges:
        ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            color=CONNECTION_COLOR,
            linewidth=CONNECTION_WIDTH,
            alpha=CONNECTION_ALPHA,
            zorder=2,
        )

    for i, (x, y) in enumerate(points):
        ax.add_patch(
            Circle(
                (x, y),
                radius=cell_radius,
                facecolor=CELL_COLORS[i % len(CELL_COLORS)],
                edgecolor=CELL_EDGE_COLOR,
                linewidth=CELL_EDGE_WIDTH,
                zorder=4,
            )
        )

    save_symbol(fig, outpath)


# ============================================================
# MAIN
# ============================================================

def main(output_dir=OUTPUT_DIR):
    ensure_dir(output_dir)

    plot_pixel_symbol(
        os.path.join(output_dir, "symbol_pixel_layout.png"),
        pixel_size=PIXEL_SIZE,
    )

    plot_visium_symbol(
        os.path.join(output_dir, "symbol_visium_layout.png"),
        spot_gap=VISIUM_SPOT_GAP,
        spot_radius=VISIUM_SPOT_RADIUS,
    )

    plot_singlecell_symbol(
        os.path.join(output_dir, "symbol_singlecell_layout.png"),
        n_cells=N_CELLS,
        cell_radius=CELL_RADIUS,
        n_connections=N_CONNECTIONS,
        seed=RANDOM_SEED,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output folder for the three symbols.",
    )
    args = parser.parse_args()
    main(args.output_dir)
