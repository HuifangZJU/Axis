"""Metrics for generated and spatially matched tissues (Methods, "Evaluation").

Functions take plain arrays so they can be applied to any method's output.
Their numerics are identical to the implementations used for the figures
(``reproducibility/figure3`` and ``figure4``).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

__all__ = [
    "safe_pearson",
    "masked_ncc",
    "shifted_map_ncc",
    "orientation_histogram",
    "cluster_composition",
    "symmetric_relative_abs_error",
    "min_positive_spot_distance",
    "within_cluster_nn_distances",
    "local_structure_distance",
    "neighborhood_connectivity",
    "js_distance",
    "total_variation_distance",
    "kruskal_eta_h_squared",
    "direction_invariant_auroc",
    "adjacent_layer_aurocs",
    "contrast_agreement",
]


def safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation of finite pairs; ``NaN`` for < 3 pairs or constant input."""
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return np.nan
    x, y = x[keep], y[keep]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


# =============================================================================
# Spatial molecular patterns
# =============================================================================
def masked_ncc(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None, min_overlap: int = 20) -> float:
    """NCC of two rasterised maps over grid cells valid in both (Pearson correlation)."""
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    mm = np.ones(aa.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    keep = mm & np.isfinite(aa) & np.isfinite(bb)
    if int(keep.sum()) < min_overlap:
        return np.nan
    return safe_pearson(aa[keep], bb[keep])


def shifted_map_ncc(
    a: np.ndarray, b: np.ndarray, max_shift_y: int = 3, max_shift_x: int = 3, min_overlap: int = 20
) -> Tuple[float, int, int]:
    """Shift-tolerant NCC: maximum Pearson correlation over integer shifts.

    Returns ``(best_ncc, best_dy, best_dx)``.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    h, w = a.shape
    best, best_dy, best_dx = -np.inf, 0, 0
    for dy in range(-int(max_shift_y), int(max_shift_y) + 1):
        for dx in range(-int(max_shift_x), int(max_shift_x) + 1):
            y0a, y1a, y0b, y1b = max(0, -dy), min(h, h - dy), max(0, dy), min(h, h + dy)
            x0a, x1a, x0b, x1b = max(0, -dx), min(w, w - dx), max(0, dx), min(w, w + dx)
            if y1a <= y0a or y1b <= y0b or x1a <= x0a or x1b <= x0b:
                continue
            aa, bb = a[y0a:y1a, x0a:x1a].ravel(), b[y0b:y1b, x0b:x1b].ravel()
            keep = np.isfinite(aa) & np.isfinite(bb)
            if keep.sum() < min_overlap:
                continue
            r = safe_pearson(aa[keep], bb[keep])
            if np.isfinite(r) and r > best:
                best, best_dy, best_dx = r, dy, dx
    return (float(best) if np.isfinite(best) else np.nan), int(best_dy), int(best_dx)


def orientation_histogram(gx: np.ndarray, gy: np.ndarray, strength: Optional[np.ndarray] = None, n_bins: int = 12) -> np.ndarray:
    """Axial gradient-orientation profile on ``[0, pi)``.

    With ``strength=None`` every element counts once (unweighted profile);
    otherwise elements are weighted by e.g. the gradient magnitude.
    """
    gx, gy = np.asarray(gx, dtype=np.float64).ravel(), np.asarray(gy, dtype=np.float64).ravel()
    strength = np.ones_like(gx) if strength is None else np.asarray(strength, dtype=np.float64).ravel()
    keep = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(strength) & (strength > 0)
    if keep.sum() < 5:
        return np.full(n_bins, np.nan, dtype=np.float32)
    theta = np.mod(np.arctan2(gy[keep], gx[keep]), np.pi)
    hist, _ = np.histogram(theta, bins=n_bins, range=(0.0, np.pi), weights=strength[keep])
    if hist.sum() <= 0:
        return np.full(n_bins, np.nan, dtype=np.float32)
    return (hist / hist.sum()).astype(np.float32)


# =============================================================================
# Spatial cluster patterns
# =============================================================================
def cluster_composition(labels: Sequence, cluster_order: Sequence[str]) -> np.ndarray:
    """Fraction of observations in every cluster of ``cluster_order``."""
    counts = pd.Series(np.asarray(labels)).astype(str).value_counts()
    values = np.asarray([counts.get(str(c), 0) for c in cluster_order], dtype=float)
    return values / values.sum() if values.sum() > 0 else values


def symmetric_relative_abs_error(reference, candidate, eps: float = 1e-8) -> np.ndarray:
    """``2 |c - r| / (|r| + |c| + eps)``."""
    reference, candidate = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    return 2.0 * np.abs(candidate - reference) / (np.abs(reference) + np.abs(candidate) + eps)


def min_positive_spot_distance(coords: np.ndarray, eps: float = 1e-8) -> float:
    """Characteristic spot spacing: minimum positive nearest-neighbour distance."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return np.nan
    dist, _ = cKDTree(coords).query(coords, k=2, workers=-1)
    nearest = dist[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > eps)]
    return float(np.min(nearest)) if nearest.size else np.nan


def within_cluster_nn_distances(coords: np.ndarray, labels: Sequence, cluster: str, spot_unit: Optional[float]) -> np.ndarray:
    """Nearest same-cluster neighbour distance of every observation, in spot units."""
    coords, labels = np.asarray(coords, dtype=float), np.asarray(labels).astype(str)
    pts = coords[labels == str(cluster)]
    if len(pts) < 2:
        return np.asarray([], dtype=float)
    dist, _ = cKDTree(pts).query(pts, k=2, workers=-1)
    nearest = dist[:, 1]
    nearest = nearest[np.isfinite(nearest)]
    if spot_unit is None or not np.isfinite(spot_unit) or spot_unit <= 0:
        return np.full_like(nearest, np.nan, dtype=float)
    return nearest / float(spot_unit)


def local_structure_distance(real_distances: np.ndarray, candidate_distances: np.ndarray) -> float:
    """1-D Wasserstein distance between two nearest-neighbour distance distributions."""
    from scipy.stats import wasserstein_distance

    real = np.asarray(real_distances, dtype=float)
    cand = np.asarray(candidate_distances, dtype=float)
    real, cand = real[np.isfinite(real)], cand[np.isfinite(cand)]
    if real.size == 0 or cand.size == 0:
        return np.nan
    return float(wasserstein_distance(real, cand))


def neighborhood_connectivity(coords: np.ndarray, labels: Sequence, cluster_order: Sequence[str], k: int = 6) -> np.ndarray:
    """Row-normalised cluster-to-neighbour-cluster fractions over a ``k``-NN graph."""
    coords, labels = np.asarray(coords, dtype=float), np.asarray(labels).astype(str)
    n = len(coords)
    matrix = np.full((len(cluster_order), len(cluster_order)), np.nan, dtype=float)
    if n < 2:
        return matrix
    _, idx = cKDTree(coords).query(coords, k=min(k + 1, n), workers=-1)
    idx = np.asarray(idx, dtype=np.int64)
    idx = idx[:, None] if idx.ndim == 1 else idx
    neighbors = idx[:, 1:]
    for row, source in enumerate(cluster_order):
        src = labels == str(source)
        if not np.any(src) or neighbors.shape[1] == 0:
            continue
        nb = labels[neighbors[src].ravel()]
        counts = pd.Series(nb).value_counts()
        for col, target in enumerate(cluster_order):
            matrix[row, col] = float(counts.get(str(target), 0)) / float(len(nb))
    return matrix


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance (base 2, in ``[0, 1]``) of two non-negative vectors."""
    p, q = np.asarray(p, dtype=float).ravel(), np.asarray(q, dtype=float).ravel()
    keep = np.isfinite(p) & np.isfinite(q)
    if not np.any(keep):
        return np.nan
    p, q = np.clip(p[keep], 0.0, np.inf), np.clip(q[keep], 0.0, np.inf)
    if p.sum() <= 0 or q.sum() <= 0:
        return np.nan
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        v = (a > 0) & (b > 0)
        return float(np.sum(a[v] * np.log(a[v] / b[v]))) if np.any(v) else 0.0

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(np.clip(np.sqrt(max(jsd, 0.0) / np.log(2.0)), 0.0, 1.0))


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """``0.5 * sum |p - q|`` of two normalised distributions."""
    p, q = np.asarray(p, dtype=float).ravel(), np.asarray(q, dtype=float).ravel()
    keep = np.isfinite(p) & np.isfinite(q)
    if not np.any(keep) or p[keep].sum() <= 0 or q[keep].sum() <= 0:
        return np.nan
    p, q = p[keep] / p[keep].sum(), q[keep] / q[keep].sum()
    return float(0.5 * np.abs(p - q).sum())


# =============================================================================
# Laminar organisation (population-level generation)
# =============================================================================
def kruskal_eta_h_squared(groups: Sequence[np.ndarray]) -> Tuple[float, float, float, int, int]:
    """Kruskal-Wallis effect size ``eta^2_H = max(0, (H - K + 1) / (n - K))``.

    Returns ``(eta2, H, p, n_total, K)``.
    """
    from scipy.stats import kruskal

    valid = [np.asarray(x, dtype=float) for x in groups if len(x) > 0]
    n_total, k = int(sum(len(x) for x in valid)), len(valid)
    if k < 2 or n_total <= k:
        return np.nan, np.nan, np.nan, n_total, k
    try:
        result = kruskal(*valid)
        h_stat, p_value = float(result[0]), float(result[1])
    except ValueError as exc:
        if "identical" not in str(exc).lower():
            raise
        h_stat, p_value = 0.0, 1.0
    return float(max(0.0, (h_stat - k + 1.0) / (n_total - k))), h_stat, p_value, n_total, k


def direction_invariant_auroc(left_values: np.ndarray, right_values: np.ndarray) -> Tuple[float, float, str]:
    """``max(A, 1 - A)`` of the AUROC separating two groups; returns ``(sep, raw, higher_side)``."""
    from sklearn.metrics import roc_auc_score

    left, right = np.asarray(left_values, dtype=float), np.asarray(right_values, dtype=float)
    left, right = left[np.isfinite(left)], right[np.isfinite(right)]
    if len(left) == 0 or len(right) == 0:
        return np.nan, np.nan, "NA"
    scores = np.concatenate([left, right])
    labels = np.concatenate([np.zeros(len(left), dtype=np.int8), np.ones(len(right), dtype=np.int8)])
    raw = 0.5 if np.nanmax(scores) - np.nanmin(scores) < 1e-12 else float(roc_auc_score(labels, scores))
    return float(max(raw, 1.0 - raw)), float(raw), "right" if raw >= 0.5 else "left"


def adjacent_layer_aurocs(values: np.ndarray, layers: Sequence, layer_order: Sequence[str]) -> Dict[str, float]:
    """Mean/median/min/max direction-invariant AUROC over anatomically adjacent layer pairs."""
    values, layers = np.asarray(values, dtype=float), np.asarray(layers).astype(str)
    aucs = []
    for left, right in zip(layer_order[:-1], layer_order[1:]):
        lv, rv = values[layers == str(left)], values[layers == str(right)]
        lv, rv = lv[np.isfinite(lv)], rv[np.isfinite(rv)]
        if len(lv) and len(rv):
            aucs.append(direction_invariant_auroc(lv, rv)[0])
    aucs = np.asarray([a for a in aucs if np.isfinite(a)])
    if not len(aucs):
        return {
            "adjacent_auc_mean": np.nan,
            "adjacent_auc_median": np.nan,
            "adjacent_auc_min": np.nan,
            "adjacent_auc_max": np.nan,
            "n_adjacent_pairs": 0,
        }
    return {
        "adjacent_auc_mean": float(np.mean(aucs)),
        "adjacent_auc_median": float(np.median(aucs)),
        "adjacent_auc_min": float(np.min(aucs)),
        "adjacent_auc_max": float(np.max(aucs)),
        "n_adjacent_pairs": int(len(aucs)),
    }


# =============================================================================
# Condition-matched contrasts
# =============================================================================
def contrast_agreement(
    measured: np.ndarray, virtual: np.ndarray, effect_threshold: float = 0.10, top_fraction: float = 0.10
) -> Dict[str, float]:
    """Agreement between measured and virtual domain-level contrasts across genes.

    Pearson, Spearman and cosine similarity; sign accuracy over genes with
    ``|measured| >= effect_threshold``; AUROC of ``|virtual|`` for recovering
    the top ``top_fraction`` genes by ``|measured|``.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    m, v = np.asarray(measured, dtype=float), np.asarray(virtual, dtype=float)
    keep = np.isfinite(m) & np.isfinite(v)
    m, v = m[keep], v[keep]
    out = {
        "pearson": safe_pearson(m, v),
        "spearman": np.nan,
        "cosine": np.nan,
        "sign_accuracy": np.nan,
        "top_auroc": np.nan,
        "n_genes": int(len(m)),
    }
    if len(m) >= 3:
        out["spearman"] = float(spearmanr(m, v)[0])
    denom = np.linalg.norm(m) * np.linalg.norm(v)
    if denom > 0:
        out["cosine"] = float(np.dot(m, v) / denom)
    strong = np.abs(m) >= effect_threshold
    if strong.any():
        out["sign_accuracy"] = float(np.mean(np.sign(m[strong]) == np.sign(v[strong])))
    if len(m) >= 10:
        cutoff = np.quantile(np.abs(m), 1.0 - top_fraction)
        is_top = (np.abs(m) >= cutoff).astype(int)
        if 0 < is_top.sum() < len(is_top):
            out["top_auroc"] = float(roc_auc_score(is_top, np.abs(v)))
    return out
