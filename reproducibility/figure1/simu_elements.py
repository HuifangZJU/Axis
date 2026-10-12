from __future__ import annotations
from axis.settings import data_path

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import Circle
from scipy.ndimage import distance_transform_edt, gaussian_filter

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


# =============================================================================
# Style
# =============================================================================

@dataclass
class SupplementStyle:
    dpi: int = 300
    bg_color: str = "white"
    query_color: str = "#202020"
    center_color: str = "#202020"
    bio_color_a: str = "#7EB6D9"
    bio_color_b: str = "#F0AA6C"
    spatial_point_color: str = "#5D8DD6"
    molecular_point_color: str = "#A875D6"
    spatial_line_color: str = "#2F5E9E"
    molecular_line_color: str = "#7A4AA8"
    dashed_circle_color: str = "#9A9A9A"
    axis_text_color: str = "#444444"

    domain_colors: Tuple[str, ...] = (
        "#8DBDD8",  # cortex-like
        "#DDAA69",  # hippocampal-like
        "#B79BCB",  # deep-gray-like
        "#AEBF9B",  # inner/white-matter-like
    )


# =============================================================================
# Data store
# =============================================================================

class SimulationStore:
    def __init__(self, simu_dir: Optional[Path] = None):
        self.simu_dir = simu_dir
        self.canvas = 512
        self.reference_mask: np.ndarray
        self.reference_spots: np.ndarray
        self.reference_reads: np.ndarray
        self.tau01_domain_partition: np.ndarray
        self.tau01_tissue_mask: np.ndarray
        self.tau01_domain_contour_mask: np.ndarray
        self.tau01_domain_scores: np.ndarray
        self.tau01_native_spots: np.ndarray
        self.tau01_domain_labels_at_spots: np.ndarray
        self.tau01_domain_boundaries_rgba: Optional[np.ndarray] = None
        self.tau01_domain_continuous_rgba: Optional[np.ndarray] = None
        self._load_or_fallback()

    def _load_npz(self, path: Path) -> Dict[str, np.ndarray]:
        arr = np.load(path, allow_pickle=True)
        return {k: arr[k] for k in arr.files}

    def _load_rgba_png(self, path: Path) -> Optional[np.ndarray]:
        if cv2 is None or not path.exists():
            return None
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            return None
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
        elif arr.shape[-1] == 3:
            alpha = np.full(arr.shape[:2] + (1,), 255, dtype=arr.dtype)
            arr = np.concatenate([arr, alpha], axis=-1)
        # BGRA -> RGBA
        arr = arr[..., [2, 1, 0, 3]]
        return arr

    def _load_or_fallback(self) -> None:
        if self.simu_dir is not None and self.simu_dir.exists():
            ref = self.simu_dir / "reference_data.npz"
            tau1 = self.simu_dir / "tau_01" / "data.npz"
            tau1_cont = self.simu_dir / "tau_01" / "domain_continuous.png"
            tau1_boundary = self.simu_dir / "tau_01" / "domain_boundaries.png"
            if ref.exists() and tau1.exists():
                ref_data = self._load_npz(ref)
                tau1_data = self._load_npz(tau1)
                self.reference_mask = ref_data["mask_reference"].astype(np.uint8)
                self.reference_spots = ref_data["reference_spot_xy"].astype(np.float32)
                self.canvas = int(self.reference_mask.shape[0])
                self.reference_reads = tau1_data.get(
                    "molecular_read", np.zeros(len(self.reference_spots), dtype=np.float32)
                ).astype(np.float32)
                if "domain_partition" in tau1_data:
                    self.tau01_domain_partition = tau1_data["domain_partition"].astype(np.int32)
                else:
                    self.tau01_domain_partition = self._make_fallback_domain_partition(self.reference_mask)
                self.tau01_tissue_mask = (tau1_data.get("tissue_mask", self.reference_mask) > 0).astype(np.uint8)
                self.tau01_domain_contour_mask = (
                    tau1_data.get("domain_contour_mask", np.zeros_like(self.tau01_tissue_mask)) > 0
                ).astype(np.uint8)

                # Figure 2 uses the PRE-SAVED spatial-domain information from data.npz.
                # domain_scores are generated together with domain_partition in the
                # simulation-data script and are used only to soften colors near boundaries.
                if "domain_scores" in tau1_data:
                    self.tau01_domain_scores = tau1_data["domain_scores"].astype(np.float32)
                else:
                    self.tau01_domain_scores = self._scores_from_partition(
                        self.tau01_domain_partition,
                        self.tau01_tissue_mask,
                    )

                self.tau01_native_spots = tau1_data.get(
                    "native_spot_xy", self.reference_spots
                ).astype(np.float32)
                self.tau01_domain_labels_at_spots = tau1_data.get(
                    "domain_labels_at_spots",
                    ref_data.get("reference_domain_labels", np.zeros(len(self.tau01_native_spots), dtype=np.int16)),
                ).astype(np.int16)

                # Figure 2 uses this PRE-SAVED smooth contour image as the geometry
                # of the continuous domains. This avoids the spot-anchor islands that
                # can exist in domain_partition/domain_scores.
                self.tau01_domain_boundaries_rgba = self._load_rgba_png(tau1_boundary)
                if self.tau01_domain_boundaries_rgba is None:
                    raise FileNotFoundError(
                        f"Figure 2 requires the saved boundary image: {tau1_boundary}"
                    )

                self.tau01_domain_continuous_rgba = self._load_rgba_png(tau1_cont)
                return
        self._build_fallback()

    def _build_fallback(self) -> None:
        canvas = self.canvas
        yy, xx = np.mgrid[0:canvas, 0:canvas]
        x = (xx - canvas * 0.50) / (canvas * 0.34)
        y = (yy - canvas * 0.54) / (canvas * 0.28)
        mask = (
            (x**2 + (1.12 * y) ** 2 < 1.0)
            | (((x + 0.36) / 0.68) ** 2 + ((y + 0.02) / 0.80) ** 2 < 1.0)
        )
        notch = (((x - 0.48) / 0.34) ** 2 + ((y + 0.02) / 0.24) ** 2) < 1.0
        mask = (mask & (~notch)).astype(np.uint8)
        self.reference_mask = mask
        self.tau01_tissue_mask = mask.copy()

        spacing = 28.0
        xs = np.arange(32, canvas - 32, spacing)
        ys = np.arange(32, canvas - 32, spacing * np.sqrt(3) / 2.0)
        pts: List[Tuple[float, float]] = []
        for i, y0 in enumerate(ys):
            offset = 0.5 * spacing if (i % 2) else 0.0
            for x0 in xs + offset:
                xi = int(round(x0))
                yi = int(round(y0))
                if 0 <= xi < canvas and 0 <= yi < canvas and mask[yi, xi] > 0:
                    pts.append((x0, y0))
        self.reference_spots = np.asarray(pts, dtype=np.float32)

        xy = self.reference_spots / float(canvas)
        reads = (
            0.55 * np.exp(-((xy[:, 0] - 0.32) ** 2 + (xy[:, 1] - 0.32) ** 2) / 0.025)
            + 0.75 * np.exp(-((xy[:, 0] - 0.64) ** 2 + (xy[:, 1] - 0.55) ** 2) / 0.040)
            + 0.25 * np.sin(6.0 * xy[:, 0]) * np.cos(4.0 * xy[:, 1])
        )
        self.reference_reads = ((reads - reads.min()) / (reads.max() - reads.min() + 1e-8)).astype(np.float32)
        self.tau01_domain_partition = self._make_fallback_domain_partition(mask)
        self.tau01_domain_contour_mask = self._make_fallback_contour_mask(self.tau01_domain_partition, mask)
        self.tau01_domain_scores = self._scores_from_partition(self.tau01_domain_partition, mask)
        self.tau01_native_spots = self.reference_spots.copy()
        xi = np.clip(np.rint(self.tau01_native_spots[:, 0]).astype(int), 0, canvas - 1)
        yi = np.clip(np.rint(self.tau01_native_spots[:, 1]).astype(int), 0, canvas - 1)
        self.tau01_domain_labels_at_spots = self.tau01_domain_partition[yi, xi].astype(np.int16)
        self.tau01_domain_boundaries_rgba = self._fallback_boundary_rgba(
            self.tau01_domain_partition,
            self.tau01_tissue_mask,
        )
        self.tau01_domain_continuous_rgba = None

    def _make_fallback_domain_partition(self, tissue_mask: np.ndarray) -> np.ndarray:
        h, w = tissue_mask.shape
        yy, xx = np.mgrid[0:h, 0:w]
        seeds = np.array([
            [0.33 * w, 0.33 * h],
            [0.70 * w, 0.36 * h],
            [0.60 * w, 0.68 * h],
            [0.28 * w, 0.65 * h],
        ], dtype=np.float32)
        sigma = 90.0
        scores = []
        for sx, sy in seeds:
            d2 = (xx - sx) ** 2 + (yy - sy) ** 2
            scores.append(np.exp(-d2 / (2 * sigma * sigma)))
        scores = np.stack(scores, axis=0)
        part = np.argmax(scores, axis=0).astype(np.int32)
        part[tissue_mask == 0] = -1
        return part

    def _make_fallback_contour_mask(self, partition: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
        edges = np.zeros_like(tissue_mask, dtype=np.uint8)
        valid = partition >= 0
        edges[:-1, :] |= ((partition[:-1, :] != partition[1:, :]) & valid[:-1, :] & valid[1:, :]).astype(np.uint8)
        edges[:, :-1] |= ((partition[:, :-1] != partition[:, 1:]) & valid[:, :-1] & valid[:, 1:]).astype(np.uint8)
        if cv2 is not None:
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges[tissue_mask <= 0] = 0
        return edges
    def _fallback_boundary_rgba(self, partition: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
        h, w = partition.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        if cv2 is None:
            rgba[..., 3] = self._make_fallback_contour_mask(partition, tissue_mask) * 255
            return rgba
        palette = [
            (141, 189, 216),
            (221, 170, 105),
            (183, 155, 203),
            (174, 191, 155),
        ]
        for domain_id, color in enumerate(palette):
            binary = ((partition == domain_id) & (tissue_mask > 0)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgba, contours, -1, (*color, 235), thickness=2, lineType=cv2.LINE_AA)
        return rgba

    def _scores_from_partition(self, partition: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
        """Construct simple smooth per-domain scores if they are not pre-saved."""
        domain_ids = [int(v) for v in np.unique(partition) if v >= 0]
        h, w = partition.shape
        yy, xx = np.mgrid[0:h, 0:w]
        scores = []
        for d in domain_ids:
            ys, xs = np.where(partition == d)
            if len(xs) == 0:
                cx, cy = 0.5 * w, 0.5 * h
                sigma = 80.0
            else:
                cx, cy = float(xs.mean()), float(ys.mean())
                sigma = max(45.0, 0.55 * np.sqrt(len(xs)))
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            s = np.exp(-d2 / (2 * sigma * sigma)).astype(np.float32)
            s[tissue_mask <= 0] = 0.0
            scores.append(s)
        if len(scores) == 0:
            return np.zeros((0, h, w), dtype=np.float32)
        return np.stack(scores, axis=0)


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_or_show(fig: plt.Figure, path: Optional[Path], mode: str, dpi: int) -> None:
    if mode == "save":
        assert path is not None
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        plt.show()


def normalize_xy(points: np.ndarray, canvas: int) -> np.ndarray:
    out = np.asarray(points, dtype=np.float32).copy()
    out[:, 0] = out[:, 0] / float(canvas)
    out[:, 1] = 1.0 - out[:, 1] / float(canvas)
    return out


def gaussian_weight(d: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(d ** 2) / (2 * sigma ** 2))


def hex_to_rgba01(h: str, alpha: float = 1.0) -> np.ndarray:
    rgb = np.asarray(mcolors.to_rgb(h), dtype=np.float32)
    return np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=np.float32)


# =============================================================================
# Figure 1
# =============================================================================

def sample_balanced_scattered_points(points: np.ndarray, reads: np.ndarray, canvas: int,
                                    query_xy: Tuple[float, float] = (0.58, 0.46),
                                    radius: float = 0.17,
                                    n_points: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    xy = normalize_xy(points, canvas)
    q = np.asarray(query_xy, dtype=np.float32)
    d = np.sqrt(np.sum((xy - q[None, :]) ** 2, axis=1))
    candidate = np.where(d <= radius)[0]
    if len(candidate) < n_points:
        candidate = np.argsort(d)[: max(n_points * 2, len(candidate))]

    cand_xy = xy[candidate]
    cand_reads = reads[candidate]
    cand_d = np.sqrt(np.sum((cand_xy - q[None, :]) ** 2, axis=1))
    cand_ang = np.arctan2(cand_xy[:, 1] - q[1], cand_xy[:, 0] - q[0])

    sectors = np.linspace(-np.pi, np.pi, n_points + 1)
    chosen: List[int] = []
    used = set()
    for i in range(n_points):
        loc = np.where((cand_ang >= sectors[i]) & (cand_ang < sectors[i + 1]))[0]
        if len(loc) == 0:
            continue
        pick = loc[np.argmin(cand_d[loc])] if i % 2 == 0 else loc[np.argmax(cand_d[loc])]
        if int(pick) not in used:
            chosen.append(int(pick))
            used.add(int(pick))

    if len(chosen) < n_points:
        order = np.argsort(cand_d)
        for idx in order:
            if int(idx) not in used:
                chosen.append(int(idx))
                used.add(int(idx))
            if len(chosen) >= n_points:
                break

    chosen = chosen[:n_points]
    nbr_xy = cand_xy[chosen]
    nbr_reads = cand_reads[chosen]
    nbr_d = np.sqrt(np.sum((nbr_xy - q[None, :]) ** 2, axis=1))
    dashed_radius = float(np.max(nbr_d))
    farthest = int(np.argmax(nbr_d))
    if nbr_d[farthest] > 1e-8:
        direction = (nbr_xy[farthest] - q) / nbr_d[farthest]
        nbr_xy[farthest] = q + dashed_radius * direction
    return q, nbr_xy, nbr_reads, dashed_radius


def draw_weight_panel(ax, center_xy: np.ndarray, nbr_xy: np.ndarray, weights: np.ndarray,
                      dashed_radius: float, point_color: str, line_color: str,
                      title: str, point_sizes: Optional[np.ndarray] = None,
                      seed: int = 0):
    ax.add_patch(Circle(center_xy, dashed_radius, facecolor="none",
                        edgecolor="#9A9A9A", lw=1.2, linestyle=(0, (3, 3))))

    w = np.asarray(weights, dtype=float)
    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
    rng = np.random.default_rng(seed)
    order = np.argsort(w_norm)
    label_vals = np.linspace(0.12, 0.96, len(w_norm))
    label_map = np.empty_like(label_vals)
    label_map[order] = label_vals

    for i, p in enumerate(nbr_xy):
        lw = 1.0 + 5.0 * w_norm[i]
        alpha = 0.18 + 0.78 * w_norm[i]
        ax.plot([center_xy[0], p[0]], [center_xy[1], p[1]], color=line_color, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=1)
        s = 88.0 if point_sizes is None else float(point_sizes[i])
        ax.scatter([p[0]], [p[1]], s=s, color=point_color, edgecolor="white", linewidth=0.8, zorder=3)

        mid_t = 0.56 + 0.05 * rng.uniform(-1.0, 1.0)
        tx = center_xy[0] * (1 - mid_t) + p[0] * mid_t
        ty = center_xy[1] * (1 - mid_t) + p[1] * mid_t
        dx = p[0] - center_xy[0]
        dy = p[1] - center_xy[1]
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-8)
        ox = -dy / norm * 0.014 * rng.uniform(-1.0, 1.0)
        oy = dx / norm * 0.014 * rng.uniform(-1.0, 1.0)
        ax.text(tx + ox, ty + oy, f"{label_map[i]:.2f}", color=line_color, fontsize=8.3,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.72),
                alpha=max(0.45, alpha), zorder=4)

    ax.scatter([center_xy[0]], [center_xy[1]], s=108, color="#1F1F1F",
               edgecolor="white", linewidth=0.9, zorder=5)
    pad = 0.10
    ax.set_xlim(center_xy[0] - dashed_radius - pad, center_xy[0] + dashed_radius + pad)
    ax.set_ylim(center_xy[1] - dashed_radius - pad, center_xy[1] + dashed_radius + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    for s in ax.spines.values():
        s.set_visible(False)


def plot_weight_schematic(store: SimulationStore, out_dir: Path, mode: str, style: SupplementStyle) -> None:
    ensure_dir(out_dir)
    center_xy, nbr_xy, nbr_reads, dashed_radius = sample_balanced_scattered_points(
        store.reference_spots, store.reference_reads, store.canvas,
        query_xy=(0.60, 0.47), radius=0.17, n_points=10,
    )

    fig = plt.figure(figsize=(12.8, 3.9), facecolor=style.bg_color)
    gs = fig.add_gridspec(1, 3, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(0.0, 1.0, 300)
    qa = 0.69
    k_a = gaussian_weight(np.abs(x - 0.0), sigma=0.30)
    k_b = gaussian_weight(np.abs(x - 1.0), sigma=0.30)
    w_a = k_a / (k_a + k_b + 1e-8)
    w_b = k_b / (k_a + k_b + 1e-8)
    ax.fill_between(x, 0, w_a, color=style.bio_color_a, alpha=0.55)
    ax.fill_between(x, 0, w_b, color=style.bio_color_b, alpha=0.40)
    ax.plot(x, w_a, color=style.bio_color_a, lw=2.3)
    ax.plot(x, w_b, color=style.bio_color_b, lw=2.3)
    qa_wa = float(np.interp(qa, x, w_a))
    qa_wb = float(np.interp(qa, x, w_b))
    ax.axvline(qa, color=style.query_color, lw=1.4, linestyle=(0, (3, 3)))
    ax.scatter([0.0, qa, 1.0], [0.0, 0.0, 0.0], s=[70, 85, 70],
               color=[style.bio_color_a, style.query_color, style.bio_color_b], zorder=5)
    ax.scatter([qa, qa], [qa_wa, qa_wb], s=[42, 42],
               color=[style.bio_color_a, style.bio_color_b], edgecolor="white", linewidth=0.8, zorder=6)
    ax.text(0.0, -0.10, "anchor a", ha="center", va="top", fontsize=10)
    ax.text(qa, -0.10, "query", ha="center", va="top", fontsize=10)
    ax.text(1.0, -0.10, "anchor b", ha="center", va="top", fontsize=10)
    ax.text(0.10, 0.87, r"$\omega_a(\tau_q)$", color=style.bio_color_a, fontsize=11)
    ax.text(0.73, 0.87, r"$\omega_b(\tau_q)$", color=style.bio_color_b, fontsize=11)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Biological weight", fontsize=12)
    for s in ax.spines.values():
        s.set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    d = np.sqrt(np.sum((nbr_xy - center_xy[None, :]) ** 2, axis=1))
    spatial_w = gaussian_weight(d, sigma=max(0.06, dashed_radius * 0.50))
    draw_weight_panel(ax, center_xy, nbr_xy, spatial_w, dashed_radius,
                      point_color=style.spatial_point_color, line_color=style.spatial_line_color,
                      title="Spatial weight", seed=11)

    ax = fig.add_subplot(gs[0, 2])
    mol_w = (nbr_reads - float(np.min(nbr_reads))) / (float(np.max(nbr_reads) - np.min(nbr_reads)) + 1e-8)
    point_sizes = 52.0 + 230.0 * mol_w
    draw_weight_panel(ax, center_xy, nbr_xy, mol_w, dashed_radius,
                      point_color=style.molecular_point_color, line_color=style.molecular_line_color,
                      title="Molecular weight", point_sizes=point_sizes, seed=23)

    save_or_show(fig, out_dir / "supp_fig1_weight_schematics.png", mode, style.dpi)


# =============================================================================
# Figure 2
# =============================================================================

def build_smooth_domain_labels_from_saved_boundaries(
    boundary_rgba: np.ndarray,
    tissue_mask: np.ndarray,
    spot_xy: np.ndarray,
    spot_labels: np.ndarray,
    fallback_partition: np.ndarray,
    n_domains: int,
    boundary_alpha_threshold: int = 8,
    boundary_expand_px: int = 1,
) -> np.ndarray:
    """Build clean continuous domains from the PRE-SAVED domain_boundaries.png.

    The saved boundary image contains the smooth large-scale contours extracted
    by the simulation generator and therefore excludes the small spot-anchor
    islands visible in ``domain_partition``. We use those contours as barriers,
    split the tissue into connected regions, and assign each region a domain
    label by majority vote of the saved spot labels. ``fallback_partition`` is
    consulted only for a region that happens to contain no spot center.
    """
    if cv2 is None:
        raise RuntimeError("Figure 2 boundary-based rendering requires OpenCV.")

    inside = tissue_mask > 0
    alpha = boundary_rgba[..., 3]
    line_mask = (alpha >= int(boundary_alpha_threshold)).astype(np.uint8)

    # Close tiny antialiasing gaps and make the contour a reliable barrier.
    line_mask = cv2.morphologyEx(
        line_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    if boundary_expand_px > 0:
        k = 2 * int(boundary_expand_px) + 1
        line_mask = cv2.dilate(
            line_mask,
            np.ones((k, k), dtype=np.uint8),
            iterations=1,
        )

    open_region = (inside & (line_mask == 0)).astype(np.uint8)
    n_comp, comp = cv2.connectedComponents(open_region, connectivity=8)

    # Component-wise spot votes.
    votes = np.zeros((n_comp, n_domains), dtype=np.int32)
    h, w = tissue_mask.shape
    for (x, y), lab in zip(spot_xy, spot_labels):
        lab = int(lab)
        if lab < 0 or lab >= n_domains:
            continue
        xi = int(np.clip(round(float(x)), 0, w - 1))
        yi = int(np.clip(round(float(y)), 0, h - 1))
        cid = int(comp[yi, xi])
        if cid > 0:
            votes[cid, lab] += 1

    hard = np.full((h, w), -1, dtype=np.int16)
    for cid in range(1, n_comp):
        region = comp == cid
        area = int(region.sum())
        if area == 0:
            continue

        if votes[cid].sum() > 0:
            label = int(np.argmax(votes[cid]))
        else:
            vals = fallback_partition[region]
            vals = vals[(vals >= 0) & (vals < n_domains)]
            if vals.size == 0:
                continue
            label = int(np.bincount(vals.astype(np.int64), minlength=n_domains).argmax())
        hard[region] = label

    # Fill the contour pixels themselves by nearest labeled interior pixel.
    missing = inside & (hard < 0)
    if np.any(missing):
        valid_seed = hard >= 0
        if np.any(valid_seed):
            _, nearest = distance_transform_edt(
                ~valid_seed,
                return_distances=True,
                return_indices=True,
            )
            ny, nx = nearest
            hard[missing] = hard[ny[missing], nx[missing]]

    # Very defensive fallback if a closed contour failed to generate a seed.
    still_missing = inside & (hard < 0)
    hard[still_missing] = fallback_partition[still_missing].astype(np.int16)
    hard[~inside] = -1
    return hard


def render_boundary_based_continuous_domain_rgba(
    boundary_rgba: np.ndarray,
    tissue_mask: np.ndarray,
    spot_xy: np.ndarray,
    spot_labels: np.ndarray,
    fallback_partition: np.ndarray,
    domain_colors: Sequence[str],
    blend_margin_px: int = 18,
    boundary_alpha_threshold: int = 8,
    boundary_expand_px: int = 1,
    fill_alpha: int = 220,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a smooth continuous map from saved domain boundaries.

    No boundary strokes are drawn in the output. Instead, the clean regions
    inferred from ``domain_boundaries.png`` are filled with the same palette as
    the spot map, then softly mixed only within ``blend_margin_px`` of adjacent
    domain borders.
    """
    inside = tissue_mask > 0
    n_domains = len(domain_colors)
    hard = build_smooth_domain_labels_from_saved_boundaries(
        boundary_rgba=boundary_rgba,
        tissue_mask=tissue_mask,
        spot_xy=spot_xy,
        spot_labels=spot_labels,
        fallback_partition=fallback_partition,
        n_domains=n_domains,
        boundary_alpha_threshold=boundary_alpha_threshold,
        boundary_expand_px=boundary_expand_px,
    )

    h, w = hard.shape
    palette = np.stack([
        255.0 * np.asarray(mcolors.to_rgb(hx), dtype=np.float32)
        for hx in domain_colors
    ], axis=0)

    hard_w = np.zeros((n_domains, h, w), dtype=np.float32)
    for d in range(n_domains):
        hard_w[d] = (hard == d).astype(np.float32)

    if int(blend_margin_px) <= 0:
        weights = hard_w
    else:
        # Smooth candidate mixture from the CLEAN component map, not from the
        # original spot-anchored scores.
        sigma = max(0.7, float(blend_margin_px) / 3.0)
        soft_w = np.stack([
            gaussian_filter(hard_w[d], sigma=sigma, mode="nearest")
            for d in range(n_domains)
        ], axis=0)
        soft_w[:, ~inside] = 0.0
        soft_w /= (soft_w.sum(axis=0, keepdims=True) + 1e-8)

        # Restrict mixing to a finite margin around true domain borders.
        border = np.zeros((h, w), dtype=bool)
        valid = hard >= 0
        border[:-1, :] |= (hard[:-1, :] != hard[1:, :]) & valid[:-1, :] & valid[1:, :]
        border[:, :-1] |= (hard[:, :-1] != hard[:, 1:]) & valid[:, :-1] & valid[:, 1:]
        dist = distance_transform_edt(~border).astype(np.float32)
        margin = max(1.0, float(blend_margin_px))
        strength = np.clip(1.0 - dist / margin, 0.0, 1.0)
        # smoothstep gives a cleaner visual transition than a linear ramp
        strength = strength * strength * (3.0 - 2.0 * strength)
        strength[~inside] = 0.0
        weights = (1.0 - strength[None, :, :]) * hard_w + strength[None, :, :] * soft_w
        weights /= (weights.sum(axis=0, keepdims=True) + 1e-8)

    rgb = np.tensordot(weights.transpose(1, 2, 0), palette, axes=([2], [0]))
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(inside, fill_alpha, 0).astype(np.uint8)
    return rgba, hard


def sample_square_crops(tissue_mask: np.ndarray, square_size: int, n_squares: int,
                        rng: np.random.Generator,
                        min_tissue_fraction: float = 0.45) -> List[Tuple[int, int]]:
    """Sample same-size square crops from tissue-supported locations."""
    h, w = tissue_mask.shape
    if square_size <= 0:
        raise ValueError("square_size must be positive")
    if square_size > min(h, w):
        raise ValueError(
            f"square_size={square_size} exceeds image size {(h, w)}"
        )

    half_lo = square_size // 2
    half_hi = square_size - half_lo
    valid_yx = np.argwhere(tissue_mask > 0)
    if len(valid_yx) == 0:
        raise RuntimeError("No tissue-supported pixels found for Figure 2 crop sampling.")

    crops: List[Tuple[int, int]] = []
    attempts = 0
    min_center_dist = square_size * 0.55

    while len(crops) < n_squares and attempts < 10000:
        attempts += 1
        idx = int(rng.integers(0, len(valid_yx)))
        cy, cx = valid_yx[idx]
        cy = int(np.clip(cy, half_lo, h - half_hi))
        cx = int(np.clip(cx, half_lo, w - half_hi))

        y0, y1 = cy - half_lo, cy + half_hi
        x0, x1 = cx - half_lo, cx + half_hi
        patch_mask = tissue_mask[y0:y1, x0:x1]
        if patch_mask.shape != (square_size, square_size):
            continue
        if float(np.mean(patch_mask > 0)) < min_tissue_fraction:
            continue

        if any((cy - py) ** 2 + (cx - px) ** 2 < min_center_dist ** 2
               for py, px in crops):
            continue

        crops.append((cy, cx))

    # Relax spacing only if necessary, while keeping the tissue-fraction criterion.
    while len(crops) < n_squares and attempts < 20000:
        attempts += 1
        idx = int(rng.integers(0, len(valid_yx)))
        cy, cx = valid_yx[idx]
        cy = int(np.clip(cy, half_lo, h - half_hi))
        cx = int(np.clip(cx, half_lo, w - half_hi))
        y0, y1 = cy - half_lo, cy + half_hi
        x0, x1 = cx - half_lo, cx + half_hi
        patch_mask = tissue_mask[y0:y1, x0:x1]
        if patch_mask.shape == (square_size, square_size) and \
                float(np.mean(patch_mask > 0)) >= min_tissue_fraction:
            crops.append((cy, cx))

    if len(crops) < n_squares:
        raise RuntimeError(
            f"Could only sample {len(crops)} valid patches; requested {n_squares}. "
            "Try a smaller --fig2-square-size or lower tissue-fraction threshold."
        )

    return crops


def plot_domain_patch_grid(store: SimulationStore, out_dir: Path, mode: str,
                           style: SupplementStyle, square_size: int = 180,
                           n_squares: int = 8,
                           min_tissue_fraction: float = 0.45,
                           blend_margin_px: int = 18,
                           boundary_alpha_threshold: int = 8,
                           boundary_expand_px: int = 1) -> None:
    """Figure 2: whole smooth spatial-domain map + cropped patch grid."""
    ensure_dir(out_dir)

    if store.tau01_domain_boundaries_rgba is None:
        raise RuntimeError(
            "Figure 2 requires simu_data/tau_01/domain_boundaries.png."
        )

    domain_img, clean_labels = render_boundary_based_continuous_domain_rgba(
        boundary_rgba=store.tau01_domain_boundaries_rgba,
        tissue_mask=store.tau01_tissue_mask,
        spot_xy=store.tau01_native_spots,
        spot_labels=store.tau01_domain_labels_at_spots,
        fallback_partition=store.tau01_domain_partition,
        domain_colors=style.domain_colors,
        blend_margin_px=blend_margin_px,
        boundary_alpha_threshold=boundary_alpha_threshold,
        boundary_expand_px=boundary_expand_px,
    )

    rendered_tissue_mask = (domain_img[..., 3] > 0).astype(np.uint8)
    rng = np.random.default_rng(13)
    centers = sample_square_crops(
        rendered_tissue_mask,
        square_size=square_size,
        n_squares=n_squares,
        rng=rng,
        min_tissue_fraction=min_tissue_fraction,
    )

    cols = min(4, n_squares)
    rows = int(math.ceil(n_squares / cols))

    # Whole map on the left, patch grid on the right.
    fig_h = max(5.3, 2.55 * rows)
    fig = plt.figure(figsize=(15.0, fig_h), facecolor=style.bg_color)
    outer = fig.add_gridspec(
        1, 2,
        width_ratios=[1.05, 2.15],
        wspace=0.055,
    )

    ax_full = fig.add_subplot(outer[0, 0])
    ax_full.imshow(domain_img)
    ax_full.set_aspect("equal")
    ax_full.set_xticks([])
    ax_full.set_yticks([])
    for spine in ax_full.spines.values():
        spine.set_visible(False)

    patch_gs = outer[0, 1].subgridspec(
        rows, cols,
        wspace=0.055,
        hspace=0.055,
    )

    half_lo = square_size // 2
    half_hi = square_size - half_lo
    for i in range(rows * cols):
        ax = fig.add_subplot(patch_gs[i // cols, i % cols])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i >= n_squares:
            ax.axis("off")
            continue

        cy, cx = centers[i]
        y0, y1 = cy - half_lo, cy + half_hi
        x0, x1 = cx - half_lo, cx + half_hi
        crop = domain_img[y0:y1, x0:x1]
        ax.imshow(crop)
        ax.set_aspect("equal")

    save_or_show(
        fig,
        out_dir / "supp_fig2_domain_map_and_patch_grid.png",
        mode,
        style.dpi,
    )

    # Also save the clean whole-domain rendering as a convenient QC asset.
    if mode == "save":
        whole_path = out_dir / "supp_fig2_continuous_domain.png"
        if cv2 is not None:
            cv2.imwrite(str(whole_path), domain_img[..., [2, 1, 0, 3]])

# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supplementary AXIS method-element illustration script.")
    p.add_argument("--simu-dir", type=str,
                 default=data_path("paper_analysis/Figure1/c_method/simu_data/"),
                 help="Path to the generated simu_data directory (optional).")
    p.add_argument("--output", type=str, default=data_path("paper_analysis/Figure1/c_method/"),
                   help="Output directory.")
    p.add_argument("--mode", type=str, default="show", choices=["save", "show"],
                   help="Save figures or show interactively.")
    p.add_argument("--which", type=str, default="all", choices=["all", "fig1", "fig2"],
                   help="Which figure set to generate.")
    p.add_argument("--fig2-square-size", type=int, default=180,
                   help="Square crop size in pixels for Figure 2.")
    p.add_argument("--fig2-num-squares", type=int, default=8,
                   help="Number of square crops to show in Figure 2.")
    p.add_argument("--fig2-min-tissue-fraction", type=float, default=0.45,
                   help="Minimum fraction of rendered tissue pixels required in each Figure 2 crop.")
    p.add_argument("--fig2-blend-margin", type=int, default=18,
                   help="Width in pixels of the gradual color-overlap zone near domain boundaries.")
    p.add_argument("--fig2-boundary-alpha-threshold", type=int, default=8,
                   help="Alpha threshold used to read the saved domain_boundaries.png contour geometry.")
    p.add_argument("--fig2-boundary-expand", type=int, default=1,
                   help="Small contour expansion in pixels to close antialiased gaps before filling domains.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    style = SupplementStyle()
    simu_dir = Path(args.simu_dir) if args.simu_dir else None
    out_dir = Path(args.output)
    ensure_dir(out_dir)

    store = SimulationStore(simu_dir)

    if args.which in {"all", "fig1"}:
        plot_weight_schematic(store, out_dir, args.mode, style)

    if args.which in {"all", "fig2"}:
        plot_domain_patch_grid(
            store,
            out_dir,
            args.mode,
            style,
            square_size=args.fig2_square_size,
            n_squares=args.fig2_num_squares,
            min_tissue_fraction=args.fig2_min_tissue_fraction,
            blend_margin_px=args.fig2_blend_margin,
            boundary_alpha_threshold=args.fig2_boundary_alpha_threshold,
            boundary_expand_px=args.fig2_boundary_expand,
        )


if __name__ == "__main__":
    main()
