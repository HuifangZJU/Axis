
"""
AXIS reusable simulation-data generator -- v6
=============================================

Styling updates in this version
-------------------------------
1. Morphology view:
   - the tissue outer boundary is drawn in blue, matching the earlier style.

2. Spatial-domain boundaries:
   - contours are drawn in the SAME color as the spot color of that domain.

Additional matched-state output:
3. Spatially matched pseudo-time states:
   - no anatomy is saved
   - one shared spot-level domain is saved once
   - each pseudo-time has separate molecular reads on the same spatial units
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


DEFAULT_TEMPLATE = Path(
    "/media/huifang/data/registration/paper_analysis/"
    "Figure1/cartoons/brain_image.jpeg"
)
DEFAULT_OUTPUT = Path(
    "/media/huifang/data/registration/paper_analysis/"
    "Figure1/c_method"
)

DOMAIN_COLORS_HEX = [
    "#8DBDD8",  # cortex-like
    "#DDAA69",  # hippocampal-like
    "#B79BCB",  # deep-gray-like
    "#AEBF9B",  # inner/white-matter-like
]

# Blue used for anatomy outer outline.
ANATOMY_OUTLINE_RGB = (43, 125, 188)


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
    spot_spacing: float = 42.0
    spot_radius: int = 11

    pseudo_sigma_px: float = 48.0

    # Smooth joint partition parameters.
    domain_score_sigma_px: float = 58.0
    domain_spot_anchor_boost: float = 3.0
    domain_contour_smooth_window: int = 31
    domain_contour_thickness_px: int = 2
    domain_partition_close_px: int = 19
    domain_min_component_area_px: int = 2500

    anatomy_outline_thickness_px: int = 4

    line_threshold: int = 245
    morphology_kernel: int = 9
    seed: int = 17


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def hex_to_rgb255(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def save_rgba_png(rgba, path):
    ok = cv2.imwrite(str(path), rgba[..., [2, 1, 0, 3]])
    if not ok:
        raise IOError(f"Failed to save {path}")


# ---------------------------------------------------------------------
# Template extraction
# ---------------------------------------------------------------------

def read_and_square_template(path, sim):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))

    dark = (img < sim.line_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (sim.morphology_kernel, sim.morphology_kernel)
    )

    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("No tissue contour found in template.")

    outer = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(outer)

    pad = 12
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad); y1 = min(img.shape[0], y + h + pad)

    crop = img[y0:y1, x0:x1]
    dark_crop = (crop < sim.line_threshold).astype(np.uint8) * 255
    closed_crop = cv2.morphologyEx(dark_crop, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed_crop = cv2.dilate(closed_crop, kernel, iterations=1)

    contours2, _ = cv2.findContours(
        closed_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    outer2 = max(contours2, key=cv2.contourArea)

    crop_mask = np.zeros_like(crop, dtype=np.uint8)
    cv2.drawContours(crop_mask, [outer2], -1, 255, thickness=-1)

    usable = sim.canvas - 2 * sim.margin
    scale = min(usable / crop.shape[1], usable / crop.shape[0])

    new_w = int(round(crop.shape[1] * scale))
    new_h = int(round(crop.shape[0] * scale))

    crop_r = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    anatomy = np.full((sim.canvas, sim.canvas), 255, dtype=np.uint8)
    mask = np.zeros((sim.canvas, sim.canvas), dtype=np.uint8)

    ox = (sim.canvas - new_w) // 2
    oy = (sim.canvas - new_h) // 2

    anatomy[oy:oy+new_h, ox:ox+new_w] = crop_r
    mask[oy:oy+new_h, ox:ox+new_w] = mask_r

    dist = distance_transform_edt(mask > 0).astype(np.float32)
    if dist.max() > 0:
        dist /= dist.max()

    fill = 243.0 - 22.0 * dist
    anatomy_f = anatomy.astype(np.float32)
    inside = mask > 0
    anatomy_f[inside] = np.minimum(anatomy_f[inside], fill[inside])
    anatomy_f = cv2.GaussianBlur(anatomy_f, (3, 3), 0.5)

    return np.clip(anatomy_f, 0, 255).astype(np.uint8), mask


# ---------------------------------------------------------------------
# Deformation
# ---------------------------------------------------------------------

def deformation_xy(x, y, tau, canvas, d):
    cx = cy = (canvas - 1) / 2.0
    half = canvas / 2.0

    xn = (x - cx) / half
    yn = (y - cy) / half
    a = float(tau) * d.strength

    xg = (1.0 + d.width_change * a) * xn + d.shear * a * yn
    yg = (1.0 + d.height_change * a) * yn

    dx = a * (
        d.bend_x * np.sin(np.pi * yn)
        + d.cross_term * xn * yn
        + d.wave * np.sin(2.0 * np.pi * yn)
    )
    dy = a * (
        d.bend_y * np.sin(np.pi * xn)
        + d.linear_y * xn
        + d.quadratic_y * (xn**2 - 0.35)
    )

    xo = cx + half * (xg + dx)
    yo = cy + half * (yg + dy)

    return xo, yo


def deform_points(points, tau, sim, d):
    xo, yo = deformation_xy(points[:, 0], points[:, 1], tau, sim.canvas, d)
    return np.column_stack([xo, yo]).astype(np.float32)


def inverse_deformation_map(tau, sim, d):
    yy, xx = np.mgrid[0:sim.canvas, 0:sim.canvas].astype(np.float32)
    sx = xx.copy()
    sy = yy.copy()

    for _ in range(8):
        fx, fy = deformation_xy(sx, sy, tau, sim.canvas, d)
        sx += xx - fx
        sy += yy - fy

    return sx.astype(np.float32), sy.astype(np.float32)


# ---------------------------------------------------------------------
# Spots
# ---------------------------------------------------------------------

def make_hex_spots(mask, spacing):
    h, w = mask.shape
    points = []

    dy = spacing * np.sqrt(3.0) / 2.0
    row = 0
    y = spacing

    while y < h - spacing:
        offset = spacing / 2.0 if row % 2 else 0.0
        x = spacing + offset
        while x < w - spacing:
            xi = int(round(x)); yi = int(round(y))
            if mask[yi, xi] > 0:
                points.append((x, y))
            x += spacing
        row += 1
        y += dy

    return np.asarray(points, dtype=np.float32)


def normalized_xy(points, canvas):
    center = (canvas - 1) / 2.0
    half = canvas / 2.0
    x = (points[:, 0] - center) / half
    y = (points[:, 1] - center) / half
    return x, y


# ---------------------------------------------------------------------
# Molecular reads
# ---------------------------------------------------------------------

def molecular_values(reference_spot_xy, tau, sim):
    x, y = normalized_xy(reference_spot_xy, sim.canvas)

    def gauss(cx, cy, sx, sy):
        return np.exp(-0.5 * (((x - cx) / sx)**2 + ((y - cy) / sy)**2))

    p1 = (1.05 - 0.45 * tau) * gauss(-0.34 + 0.08 * tau, -0.05, 0.24, 0.19)
    p2 = (0.30 + 0.85 * tau) * gauss(0.30 - 0.05 * tau, 0.30 - 0.05 * tau, 0.22, 0.20)
    p3 = 0.20 * np.clip(-y + 0.50, 0.0, 1.3)

    z = p1 + p2 + p3
    z -= z.min()
    z /= z.max() + 1e-12
    return z.astype(np.float32)


# ---------------------------------------------------------------------
# Spot-level domain labels
# ---------------------------------------------------------------------

def assign_reference_domain_labels(reference_spot_xy, mask, sim):
    x, y = normalized_xy(reference_spot_xy, sim.canvas)

    distance = distance_transform_edt(mask > 0)
    px = np.clip(np.rint(reference_spot_xy[:, 0]).astype(int), 0, mask.shape[1]-1)
    py = np.clip(np.rint(reference_spot_xy[:, 1]).astype(int), 0, mask.shape[0]-1)
    d = distance[py, px]

    labels = np.full(len(reference_spot_xy), 3, dtype=np.int16)
    labels[d < 70] = 0

    left_hip = (((x + 0.34) / 0.22)**2 + ((y - 0.28) / 0.15)**2) < 1.0
    right_hip = (((x - 0.34) / 0.22)**2 + ((y - 0.28) / 0.15)**2) < 1.0
    hip = (labels != 0) & (left_hip | right_hip)
    labels[hip] = 1

    deep = ((x / 0.27)**2 + ((y + 0.02) / 0.24)**2) < 1.0
    deep = (labels != 0) & (labels != 1) & deep
    labels[deep] = 2

    return labels


# ---------------------------------------------------------------------
# Joint smooth partition from spot labels
# ---------------------------------------------------------------------

def smooth_closed_contour(points, window):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 12:
        return pts

    window = max(5, int(window))
    if window % 2 == 0:
        window += 1

    for _ in range(2):
        half = window // 2
        padded = np.vstack([pts[-half:], pts, pts[:half]])
        kernel = np.ones(window, dtype=np.float32) / float(window)
        xs = np.convolve(padded[:, 0], kernel, mode="valid")
        ys = np.convolve(padded[:, 1], kernel, mode="valid")
        pts = np.column_stack([xs, ys])

    return pts


def clean_partition_components(label_field, tissue_mask, n_domains, min_area, close_px):
    cleaned = label_field.copy()

    if close_px % 2 == 0:
        close_px += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))

    fallback = cleaned.copy()

    for domain_id in range(n_domains):
        binary = ((cleaned == domain_id) & (tissue_mask > 0)).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        n, cc, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros_like(binary, dtype=np.uint8)

        for comp in range(1, n):
            area = stats[comp, cv2.CC_STAT_AREA]
            if area >= min_area:
                keep[cc == comp] = 255

        remove = (cleaned == domain_id) & (keep == 0) & (tissue_mask > 0)
        cleaned[remove] = -1

    unlabeled = (cleaned < 0) & (tissue_mask > 0)
    if np.any(unlabeled):
        seeds = (cleaned >= 0).astype(np.uint8)
        dist, labels = cv2.distanceTransformWithLabels(
            1 - seeds,
            distanceType=cv2.DIST_L2,
            maskSize=5,
            labelType=cv2.DIST_LABEL_PIXEL
        )
        seed_coords = np.argwhere(seeds > 0)
        if len(seed_coords) > 0:
            ys, xs = np.where(unlabeled)
            nearest_label_ids = labels[ys, xs] - 1
            nearest_label_ids = np.clip(nearest_label_ids, 0, len(seed_coords)-1)
            nys, nxs = seed_coords[nearest_label_ids][:, 0], seed_coords[nearest_label_ids][:, 1]
            cleaned[ys, xs] = fallback[nys, nxs]

    return cleaned


def build_joint_domain_partition(spot_xy, spot_labels, tissue_mask, sim, n_domains=4):
    h, w = tissue_mask.shape
    scores = np.zeros((n_domains, h, w), dtype=np.float32)

    total_impulse = np.zeros((h, w), dtype=np.float32)
    for x, y in spot_xy:
        xi = int(round(x)); yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            total_impulse[yi, xi] += 1.0

    total_density = cv2.GaussianBlur(
        total_impulse,
        (0, 0),
        sigmaX=sim.domain_score_sigma_px,
        sigmaY=sim.domain_score_sigma_px,
        borderType=cv2.BORDER_CONSTANT,
    ) + 1e-8

    for domain_id in range(n_domains):
        impulse = np.zeros((h, w), dtype=np.float32)

        pts = spot_xy[spot_labels == domain_id]
        for x, y in pts:
            xi = int(round(x)); yi = int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                impulse[yi, xi] += 1.0

        density = cv2.GaussianBlur(
            impulse,
            (0, 0),
            sigmaX=sim.domain_score_sigma_px,
            sigmaY=sim.domain_score_sigma_px,
            borderType=cv2.BORDER_CONSTANT,
        )

        density = density / total_density

        for x, y in pts:
            xi = int(round(x)); yi = int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                rr = max(2, sim.spot_radius - 2)
                cv2.circle(
                    density,
                    (xi, yi),
                    rr,
                    float(sim.domain_spot_anchor_boost),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

        scores[domain_id] = density

    scores[:, tissue_mask == 0] = -1e9
    partition = np.argmax(scores, axis=0).astype(np.int16)
    partition[tissue_mask == 0] = -1

    partition = clean_partition_components(
        partition,
        tissue_mask,
        n_domains=n_domains,
        min_area=sim.domain_min_component_area_px,
        close_px=sim.domain_partition_close_px,
    )

    return partition, scores


def extract_partition_contours(label_field, tissue_mask, sim, n_domains=4):
    contours_out = []

    for domain_id in range(n_domains):
        binary = ((label_field == domain_id) & (tissue_mask > 0)).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < sim.domain_min_component_area_px:
                continue

            points = contour[:, 0, :].astype(np.float32)
            points = smooth_closed_contour(points, sim.domain_contour_smooth_window)
            contours_out.append((domain_id, points))

    return contours_out


# ---------------------------------------------------------------------
# RGBA renderers
# ---------------------------------------------------------------------

def anatomy_rgba(anatomy, mask, sim):
    rgb = np.repeat(anatomy[..., None], 3, axis=2)
    alpha = np.where(mask > 0, 255, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])

    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if contours:
        outer = max(contours, key=cv2.contourArea)
        pts = outer[:, 0, :].astype(np.float32)
        pts = smooth_closed_contour(pts, 19)
        pts = np.rint(pts).astype(np.int32)

        cv2.polylines(
            rgba,
            [pts],
            isClosed=True,
            color=(
                int(ANATOMY_OUTLINE_RGB[0]),
                int(ANATOMY_OUTLINE_RGB[1]),
                int(ANATOMY_OUTLINE_RGB[2]),
                255,
            ),
            thickness=sim.anatomy_outline_thickness_px,
            lineType=cv2.LINE_AA,
        )

    return rgba


def render_domain_contours_rgba(contours, sim):
    rgba = np.zeros((sim.canvas, sim.canvas, 4), dtype=np.uint8)

    for domain_id, points in contours:
        color = hex_to_rgb255(DOMAIN_COLORS_HEX[domain_id])
        pts = np.rint(points).astype(np.int32)

        cv2.polylines(
            rgba,
            [pts],
            isClosed=True,
            color=(
                int(color[0]),
                int(color[1]),
                int(color[2]),
                235,
            ),
            thickness=sim.domain_contour_thickness_px,
            lineType=cv2.LINE_AA,
        )

    return rgba


def render_domain_spots_and_contours(spot_xy, spot_labels, contours, sim):
    rgba = np.zeros((sim.canvas, sim.canvas, 4), dtype=np.uint8)

    # Spot-level domains.
    for domain_id, hex_color in enumerate(DOMAIN_COLORS_HEX):
        color = hex_to_rgb255(hex_color)
        pts = spot_xy[spot_labels == domain_id]

        for x, y in pts:
            cv2.circle(
                rgba,
                (int(round(x)), int(round(y))),
                sim.spot_radius,
                (int(color[0]), int(color[1]), int(color[2]), 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    # Same-color contours.
    for domain_id, points in contours:
        color = hex_to_rgb255(DOMAIN_COLORS_HEX[domain_id])
        pts = np.rint(points).astype(np.int32)
        cv2.polylines(
            rgba,
            [pts],
            isClosed=True,
            color=(
                int(color[0]),
                int(color[1]),
                int(color[2]),
                235,
            ),
            thickness=sim.domain_contour_thickness_px,
            lineType=cv2.LINE_AA,
        )

    return rgba


def render_continuous_domain_rgba(label_field, tissue_mask, contours, sim):
    """
    Render a pixel-wise continuous spatial-domain map.

    Every in-tissue pixel is filled according to its domain label in
    ``label_field`` using the EXACT same DOMAIN_COLORS_HEX palette as the
    spot-level domain map. The outside-tissue background is transparent.

    The saved smooth domain contours are overlaid using the same per-domain
    colors with full opacity, so the continuous map remains visually
    consistent with the existing spot-based domain representation.
    """
    rgba = np.zeros((sim.canvas, sim.canvas, 4), dtype=np.uint8)
    inside = tissue_mask > 0

    # Fill every tissue pixel with its categorical domain color.
    for domain_id, hex_color in enumerate(DOMAIN_COLORS_HEX):
        color = hex_to_rgb255(hex_color)
        region = inside & (label_field == domain_id)
        rgba[region, 0] = int(color[0])
        rgba[region, 1] = int(color[1])
        rgba[region, 2] = int(color[2])
        rgba[region, 3] = 220

    # Overlay the existing smooth domain contours using the same palette.
    for domain_id, points in contours:
        color = hex_to_rgb255(DOMAIN_COLORS_HEX[domain_id])
        pts = np.rint(points).astype(np.int32)
        cv2.polylines(
            rgba,
            [pts],
            isClosed=True,
            color=(
                int(color[0]),
                int(color[1]),
                int(color[2]),
                255,
            ),
            thickness=sim.domain_contour_thickness_px,
            lineType=cv2.LINE_AA,
        )

    # Guarantee transparent background outside the tissue.
    rgba[~inside, 3] = 0
    return rgba


def molecular_rgba(spot_xy, values, sim):
    rgba = np.zeros((sim.canvas, sim.canvas, 4), dtype=np.uint8)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.clip(values, 0.0, 1.0))

    for (x, y), c in zip(spot_xy, colors):
        rgb = tuple(int(round(v * 255)) for v in c[:3])
        cv2.circle(
            rgba,
            (int(round(x)), int(round(y))),
            sim.spot_radius,
            (rgb[0], rgb[1], rgb[2], 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return rgba


def pseudo_omics_rgba(field):
    cmap = plt.get_cmap("viridis")
    mapped = cmap(np.clip(np.nan_to_num(field, nan=0.0), 0.0, 1.0))
    rgba = np.zeros((field.shape[0], field.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = (mapped[..., :3] * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(field), 255, 0).astype(np.uint8)
    return rgba


# ---------------------------------------------------------------------
# Pseudo-omics reconstruction
# ---------------------------------------------------------------------

def render_pseudo_omics(spot_xy, values, tissue_mask, sigma_px):
    h, w = tissue_mask.shape
    numerator = np.zeros((h, w), dtype=np.float32)
    denominator = np.zeros((h, w), dtype=np.float32)

    for (x, y), value in zip(spot_xy, values):
        xi = int(round(x)); yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            numerator[yi, xi] += float(value)
            denominator[yi, xi] += 1.0

    numerator = cv2.GaussianBlur(
        numerator, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px,
        borderType=cv2.BORDER_CONSTANT
    )
    denominator = cv2.GaussianBlur(
        denominator, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px,
        borderType=cv2.BORDER_CONSTANT
    )

    field = numerator / (denominator + 1e-8)
    inside = tissue_mask > 0
    valid = field[inside]

    if valid.size:
        lo, hi = np.percentile(valid, [1, 99])
        field = np.clip((field - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    field = field.astype(np.float32)
    field[~inside] = np.nan
    return field


# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------

def save_overview(path, records):
    n = len(records)
    fig, axes = plt.subplots(5, n, figsize=(2.1*n, 10.0))
    if n == 1:
        axes = axes[:, None]

    names = [
        "Anatomy",
        "Spot domain",
        "Continuous domain",
        "Molecular",
        "Pseudo-omics",
    ]

    for col, rec in enumerate(records):
        paths = [
            rec["anatomy"],
            rec["domain"],
            rec["domain_continuous"],
            rec["molecular"],
            rec["pseudo"],
        ]
        for row, p in enumerate(paths):
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
            axes[row, col].imshow(im)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(names[row], fontsize=9)
        axes[0, col].set_title(r"$\tau={:.2f}$".format(rec["tau"]), fontsize=9)

    plt.tight_layout(pad=0.3, w_pad=0.1, h_pad=0.2)
    fig.savefig(path, dpi=250, transparent=True, bbox_inches="tight")
    plt.close(fig)



def save_matched_overview(path, shared_domain_png, matched_records):
    """
    QC overview for the spatially matched states.

    Top row: the single shared domain
    Bottom row: molecular reads from each pseudo-time, all on the same
                matched/reference spatial coordinates.
    """
    n = len(matched_records)

    fig, axes = plt.subplots(
        2,
        n,
        figsize=(2.1 * n, 4.2),
    )

    if n == 1:
        axes = axes[:, None]

    shared = cv2.imread(
        str(shared_domain_png),
        cv2.IMREAD_UNCHANGED,
    )
    shared = cv2.cvtColor(
        shared,
        cv2.COLOR_BGRA2RGBA,
    )

    for col, rec in enumerate(matched_records):
        axes[0, col].imshow(shared)
        axes[0, col].axis("off")

        mol = cv2.imread(
            str(rec["molecular"]),
            cv2.IMREAD_UNCHANGED,
        )
        mol = cv2.cvtColor(
            mol,
            cv2.COLOR_BGRA2RGBA,
        )

        axes[1, col].imshow(mol)
        axes[1, col].axis("off")

        axes[0, col].set_title(
            r"$\tau={:.2f}$".format(rec["tau"]),
            fontsize=9,
        )

        if col == 0:
            axes[0, col].set_ylabel(
                "Shared domain",
                fontsize=9,
            )
            axes[1, col].set_ylabel(
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

    plt.close(fig)

# ---------------------------------------------------------------------
# Main dataset generation
# ---------------------------------------------------------------------

def generate(template_path, output_root, sim, deform):
    data_root = output_root / "simu_data"
    data_root.mkdir(parents=True, exist_ok=True)

    anatomy_ref, mask_ref = read_and_square_template(template_path, sim)
    reference_spots = make_hex_spots(mask_ref, sim.spot_spacing)
    reference_domain_labels = assign_reference_domain_labels(reference_spots, mask_ref, sim)
    tau_values = np.linspace(0.0, 1.0, sim.n_states, dtype=np.float32)

    np.savez_compressed(
        data_root / "reference_data.npz",
        anatomy_reference=anatomy_ref,
        mask_reference=mask_ref,
        reference_spot_xy=reference_spots,
        reference_domain_labels=reference_domain_labels,
        tau_values=tau_values,
    )

    # -----------------------------------------------------------------
    # Spatially matched states
    # -----------------------------------------------------------------
    # All matched states use the same reference coordinates and the same
    # shared domain annotation. Only molecular reads vary with pseudo-time.
    matched_root = data_root / "matched_states"
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

    shared_domain_png = matched_root / "shared_domain.png"
    shared_continuous_domain_png = matched_root / "shared_domain_continuous.png"
    shared_boundary_png = matched_root / "shared_domain_boundaries.png"

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
        shared_boundary_rgba[..., 3] > 0
    ).astype(np.uint8) * 255

    np.savez_compressed(
        matched_root / "shared_domain_data.npz",
        matched_spot_xy=reference_spots,
        domain_labels_at_spots=reference_domain_labels,
        domain_partition=shared_partition,
        domain_scores=shared_scores,
        domain_contour_mask=shared_boundary_mask,
        tissue_mask=mask_ref,
    )

    records = []
    matched_records = []

    for state_index, tau in enumerate(tau_values, start=1):
        state_dir = data_root / f"tau_{state_index:02d}"
        state_dir.mkdir(parents=True, exist_ok=True)

        map_x, map_y = inverse_deformation_map(float(tau), sim, deform)

        anatomy_native = cv2.remap(
            anatomy_ref, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

        mask_native = cv2.remap(
            mask_ref, map_x, map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        native_spots = deform_points(reference_spots, float(tau), sim, deform)
        displacement = (native_spots - reference_spots).astype(np.float32)

        molecular_read = molecular_values(reference_spots, float(tau), sim)

        # -------------------------------------------------------------
        # Matched molecular state
        # -------------------------------------------------------------
        # Molecular values are evaluated at the shared/reference coordinates,
        # so every pseudo-time has exactly the same spatial units.
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
            ),
            matched_molecular_png,
        )

        np.savez_compressed(
            matched_state_dir / "data.npz",
            state_index=np.int16(state_index),
            tau=np.float32(tau),
            matched_spot_xy=reference_spots,
            molecular_read=molecular_read,
            domain_labels_at_spots=reference_domain_labels,
        )

        matched_records.append(
            {
                "tau": float(tau),
                "molecular": matched_molecular_png,
            }
        )

        pseudo_omics = render_pseudo_omics(
            native_spots, molecular_read, mask_native, sim.pseudo_sigma_px
        )

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

        contour_rgba = render_domain_contours_rgba(domain_contours, sim)
        contour_mask = (contour_rgba[..., 3] > 0).astype(np.uint8) * 255

        continuous_domain_rgba = render_continuous_domain_rgba(
            joint_partition,
            mask_native,
            domain_contours,
            sim,
        )

        contours, _ = cv2.findContours(
            (mask_native > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        native_outline_xy = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)

        p_anatomy = state_dir / "anatomy.png"
        p_domain = state_dir / "domain.png"
        p_domain_continuous = state_dir / "domain_continuous.png"
        p_boundary = state_dir / "domain_boundaries.png"
        p_molecular = state_dir / "molecular.png"
        p_pseudo = state_dir / "pseudo_omics.png"

        save_rgba_png(anatomy_rgba(anatomy_native, mask_native, sim), p_anatomy)
        save_rgba_png(
            render_domain_spots_and_contours(
                native_spots, reference_domain_labels, domain_contours, sim
            ),
            p_domain,
        )
        save_rgba_png(continuous_domain_rgba, p_domain_continuous)
        save_rgba_png(contour_rgba, p_boundary)
        save_rgba_png(molecular_rgba(native_spots, molecular_read, sim), p_molecular)
        save_rgba_png(pseudo_omics_rgba(pseudo_omics), p_pseudo)

        np.savez_compressed(
            state_dir / "data.npz",
            state_index=np.int16(state_index),
            tau=np.float32(tau),

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
                "tau": float(tau),
                "anatomy": p_anatomy,
                "domain": p_domain,
                "domain_continuous": p_domain_continuous,
                "molecular": p_molecular,
                "pseudo": p_pseudo,
            }
        )

    save_matched_overview(
        matched_root / "overview.png",
        shared_domain_png,
        matched_records,
    )

    metadata = {
        "template": str(template_path),
        "simulation": asdict(sim),
        "deformation": asdict(deform),
        "domain_representation": (
            "spot-level categorical labels plus a pixel-wise continuous domain map; "
            "both use the same DOMAIN_COLORS_HEX palette and smooth non-overlapping "
            "partition inferred from all domain spots"
        ),
        "anatomy_outline_color_rgb": list(ANATOMY_OUTLINE_RGB),
        "matched_states": {
            "folder": "matched_states",
            "shared_domain": "matched_states/shared_domain.png",
            "shared_domain_continuous": "matched_states/shared_domain_continuous.png",
            "shared_domain_boundaries": "matched_states/shared_domain_boundaries.png",
            "representation": (
                "all pseudo-time states share reference_spot_xy and one "
                "reference domain annotation; molecular reads vary by pseudo-time"
            ),
        },
        "domain_labels": {
            "0": "cortex_like",
            "1": "hippocampal_like",
            "2": "deep_gray_like",
            "3": "inner_white_matter_like",
        },
    }

    with open(data_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    save_overview(data_root / "overview.png", records)

    print(
        "Saved reusable simulation dataset to:\n"
        f"  {data_root}\n"
        f"States: {sim.n_states}\n"
        f"Spots/state: {len(reference_spots)}"
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--input", type=Path, default=DEFAULT_TEMPLATE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    p.add_argument("--canvas", type=int, default=1000)
    p.add_argument("--margin", type=int, default=70)
    p.add_argument("--n-states", type=int, default=5)

    p.add_argument("--spot-spacing", type=float, default=42.0)
    p.add_argument("--spot-radius", type=int, default=11)

    p.add_argument("--pseudo-sigma", type=float, default=48.0)

    p.add_argument("--domain-score-sigma", type=float, default=58.0)
    p.add_argument("--domain-spot-anchor-boost", type=float, default=3.0)
    p.add_argument("--domain-contour-window", type=int, default=31)
    p.add_argument("--domain-contour-thickness", type=int, default=2)
    p.add_argument("--domain-partition-close", type=int, default=19)
    p.add_argument("--domain-min-component-area", type=int, default=2500)

    p.add_argument("--anatomy-outline-thickness", type=int, default=4)

    p.add_argument("--seed", type=int, default=17)

    p.add_argument("--deform-strength", type=float, default=1.0)
    p.add_argument("--width-change", type=float, default=0.045)
    p.add_argument("--height-change", type=float, default=-0.025)
    p.add_argument("--shear", type=float, default=0.030)
    p.add_argument("--bend-x", type=float, default=0.070)
    p.add_argument("--bend-y", type=float, default=0.055)
    p.add_argument("--cross-term", type=float, default=0.028)
    p.add_argument("--wave", type=float, default=0.018)
    p.add_argument("--linear-y", type=float, default=-0.030)
    p.add_argument("--quadratic-y", type=float, default=0.018)

    return p


def main():
    a = build_parser().parse_args()

    sim = SimulationParameters(
        canvas=a.canvas,
        margin=a.margin,
        n_states=a.n_states,
        spot_spacing=a.spot_spacing,
        spot_radius=a.spot_radius,
        pseudo_sigma_px=a.pseudo_sigma,

        domain_score_sigma_px=a.domain_score_sigma,
        domain_spot_anchor_boost=a.domain_spot_anchor_boost,
        domain_contour_smooth_window=a.domain_contour_window,
        domain_contour_thickness_px=a.domain_contour_thickness,
        domain_partition_close_px=a.domain_partition_close,
        domain_min_component_area_px=a.domain_min_component_area,

        anatomy_outline_thickness_px=a.anatomy_outline_thickness,

        seed=a.seed,
    )

    deform = DeformationParameters(
        strength=a.deform_strength,
        width_change=a.width_change,
        height_change=a.height_change,
        shear=a.shear,
        bend_x=a.bend_x,
        bend_y=a.bend_y,
        cross_term=a.cross_term,
        wave=a.wave,
        linear_y=a.linear_y,
        quadratic_y=a.quadratic_y,
    )

    generate(a.input, a.output, sim, deform)


if __name__ == "__main__":
    main()
