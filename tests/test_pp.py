import numpy as np
import pytest

import axis
from axis.pp import get_gene_feature_matrix, get_gene_feature_matrix_soft, rasterize_pseudoimage


def _soft_reference(coords, values, size, patch, sigma):
    """Straightforward loop implementation of Eq. (1) restricted to 3x3 neighbourhoods."""
    n = size // patch
    num = np.zeros((n, n, values.shape[1]))
    den = np.zeros((n, n))
    for (x, y), v in zip(coords, values):
        if not (0 <= x < size and 0 <= y < size):
            continue
        px, py = int(x) // patch, int(y) // patch
        cells, w = [], []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cy, cx = py + dy, px + dx
                if 0 <= cx < n and 0 <= cy < n:
                    cells.append((cy, cx))
                    w.append(np.exp(-(np.hypot(x - (cx + 0.5) * patch, y - (cy + 0.5) * patch) ** 2) / (2 * sigma**2)))
        w = np.array(w) / np.sum(w)
        for (cy, cx), wi in zip(cells, w):
            num[cy, cx] += wi * v
            den[cy, cx] += wi
    out = np.zeros_like(num)
    out[den > 0] = num[den > 0] / den[den > 0, None]
    return out, (den > 0).astype(np.uint8)


def test_soft_rasterizer_matches_reference(rng):
    coords = rng.uniform(-2, 66, size=(300, 2))
    values = rng.random((300, 3))
    ref, ref_mask = _soft_reference(coords, values, 64, 8, 4.0)
    out, mask = get_gene_feature_matrix_soft(coords, values, (64, 64), patch_size=8)
    assert np.allclose(out, ref, atol=1e-6) and np.array_equal(mask, ref_mask)


def test_hard_binning(rng):
    coords = np.array([[1.0, 1.0], [2.0, 3.0], [20.0, 1.0]])
    values = np.array([[1.0], [3.0], [5.0]])
    out, mask = get_gene_feature_matrix(coords, values, (32, 32), patch_size=16)
    assert out[0, 0, 0] == 2.0 and out[0, 1, 0] == 5.0 and mask.sum() == 2


def test_rasterize_pseudoimage_resolution(rng):
    coords = rng.uniform(0, 128, size=(400, 2))
    emb = rng.random((400, 5))
    pseudo, mask = rasterize_pseudoimage(coords, emb, 128, preset="visium")
    assert pseudo.shape == (16, 16, 5) and mask.shape == (16, 16)
    pseudo, _ = rasterize_pseudoimage(coords, emb, 128, preset="xenium")
    assert pseudo.shape == (16, 16, 5)
    with pytest.raises(ValueError):
        rasterize_pseudoimage(coords, emb, 128, preset=None, patch_size=4, soft=True)


def test_build_pseudoimages(toy_series):
    for adata in toy_series:
        store = adata.uns["axis"]
        assert store["pseudoimage"].shape == (8, 8, 4)
        assert store["mask"].any()
        assert adata.obsm["X_axis_pca"].shape == (adata.n_obs, 4)
        assert 0.0 <= store["pseudoimage"].min() and store["pseudoimage"].max() <= 1.0


def test_prepare_canvas_pads_and_scales():
    import anndata as ad

    a = ad.AnnData(np.ones((3, 2), dtype=np.float32))
    a.obsm["spatial"] = np.array([[0.0, 0.0], [99.0, 49.0], [50.0, 25.0]])
    image = np.full((50, 100, 3), 200, dtype=np.uint8)
    axis.pp.prepare_canvas(a, image, canvas_size=32)
    assert a.uns["axis"]["image"].shape == (32, 32, 3)
    assert np.allclose(a.obsm["spatial_canvas"][1], [99 * 0.32, 49 * 0.32])
    axis.pp.prepare_canvas(a, image, canvas_size=32, grayscale=True)
    assert a.uns["axis"]["image"].shape == (32, 32)
