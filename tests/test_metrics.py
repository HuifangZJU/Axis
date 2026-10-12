import numpy as np
import pytest

from axis import metrics


def test_neighbourhood_metrics_perfect_alignment(rng):
    coords = rng.uniform(0, 100, size=(300, 2))
    labels = np.where(coords[:, 0] < 50, "a", "b")
    summary, per_spot = metrics.evaluate_neighborhood_metrics(coords, labels, coords, labels, distance_threshold=5)
    assert summary["neighbor_accuracy"].min() > 0.9
    assert len(per_spot) == 300


def test_domain_metrics_identity(rng):
    pytest.importorskip("alphashape")
    coords = rng.uniform(0, 100, size=(400, 2))
    labels = np.where(coords[:, 0] < 50, "a", "b")
    df, _, _ = metrics.evaluate_domain_metrics(coords, labels, coords, labels, mask_shape=(100, 100), alpha=0.05)
    assert np.allclose(df["dice"], 1.0) and np.allclose(df["center_shift"], 0.0)
    assert np.allclose(df["spatial_corr"], 1.0)


def test_cluster_metrics():
    comp = metrics.cluster_composition(["a", "a", "b", "c"], ["a", "b", "c"])
    assert np.allclose(comp, [0.5, 0.25, 0.25])
    assert np.allclose(metrics.symmetric_relative_abs_error(comp, comp), 0)
    assert metrics.js_distance([1, 0], [0, 1]) == pytest.approx(1.0)
    assert metrics.js_distance([1, 1], [1, 1]) == pytest.approx(0.0)
    assert metrics.total_variation_distance([1, 0], [0, 1]) == pytest.approx(1.0)


def test_map_ncc():
    a = np.arange(100, dtype=float).reshape(10, 10)
    assert metrics.masked_ncc(a, 2 * a + 1) == pytest.approx(1.0)
    shifted = np.roll(a, 1, axis=0)
    best, dy, _ = metrics.shifted_map_ncc(a, shifted, max_shift_y=2, max_shift_x=0)
    assert best == pytest.approx(1.0) and dy == 1


def test_laminar_metrics(rng):
    groups = [rng.normal(m, 0.1, 50) for m in (0, 1, 2)]
    eta2, h, p, n, k = metrics.kruskal_eta_h_squared(groups)
    assert eta2 > 0.8 and p < 1e-10 and n == 150 and k == 3
    sep, raw, side = metrics.direction_invariant_auroc(groups[1], groups[0])
    assert sep == pytest.approx(1.0) and side == "left"
    values = np.concatenate(groups)
    layers = np.repeat(["L1", "L2", "L3"], 50)
    assert metrics.adjacent_layer_aurocs(values, layers, ["L1", "L2", "L3"])["adjacent_auc_mean"] == pytest.approx(1.0)


def test_contrast_agreement(rng):
    m = rng.normal(0, 1, 200)
    out = metrics.contrast_agreement(m, m + rng.normal(0, 0.01, 200))
    assert out["pearson"] > 0.99 and out["sign_accuracy"] > 0.95 and out["top_auroc"] > 0.95
