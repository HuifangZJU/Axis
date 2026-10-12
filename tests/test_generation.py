import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from axis.generation import (
    compute_axis_kernel_weights,
    compute_dense_tissue_mask_on_any_grid,
    generate_virtual_tissue_fixed_grid,
    get_anchor_pair_for_query,
    infer_axis_bandwidth,
    make_common_grid_from_coords,
    relative_position_on_axis,
    stack_weighted_evidence,
)


def test_relative_position():
    assert relative_position_on_axis(10, 0, 20) == 0.5
    assert relative_position_on_axis(30, 0, 20) == 1.5
    assert relative_position_on_axis(30, 0, 20, clip=(0, 1)) == 1.0
    with pytest.raises(ValueError):
        relative_position_on_axis(1, 2, 2)


def test_axis_weights_follow_methods():
    # sigma_tau = one third of the anchor spacing -> weight ~0.011 at the other anchor
    assert infer_axis_bandwidth([0, 20]) == pytest.approx(20 / 3)
    w, bw = compute_axis_kernel_weights([0, 20], 0)
    assert w.sum() == pytest.approx(1.0)
    assert w[1] / w[0] == pytest.approx(np.exp(-4.5), rel=1e-5)
    w, _ = compute_axis_kernel_weights([0, 20], 10)
    assert np.allclose(w, [0.5, 0.5])


def test_anchor_pair():
    pos = {0: 0, 1: 10, 2: 310, 3: 320}
    assert get_anchor_pair_for_query(100, pos, [0, 1, 2, 3]) == (1, 2)
    assert get_anchor_pair_for_query(-5, pos, [0, 1, 2, 3]) == (0, 1)
    assert get_anchor_pair_for_query(400, pos, [0, 1, 2, 3]) == (2, 3)


def test_grids():
    pix, (xe, ye) = make_common_grid_from_coords(10, type="pixel", max_x=100, max_y=100)
    assert pix.shape == (100, 2) and xe.shape == (11,)
    hexes, meta = make_common_grid_from_coords(20, type="visium", max_x=100, max_y=100)
    assert meta["layout"] == "staggered_hex" and abs(len(hexes) - 400) < 80


def test_support_mask():
    grid, _ = make_common_grid_from_coords(20, type="pixel", max_x=100, max_y=100)
    spots = np.array([[x, y] for x in range(30, 70, 5) for y in range(30, 70, 5)], dtype=np.float32)
    mask = compute_dense_tissue_mask_on_any_grid(grid, spots, max_dist_px=4)
    inside = (grid[:, 0] > 35) & (grid[:, 0] < 65) & (grid[:, 1] > 35) & (grid[:, 1] < 65)
    assert mask[inside].all() and not mask[(grid[:, 0] < 15)].any()


def test_generation_constant_field_and_labels(rng):
    coords = rng.uniform(10, 90, size=(400, 2)).astype(np.float32)
    X = sp.csr_matrix(np.full((400, 3), 4.0, dtype=np.float32))
    attrs = pd.DataFrame({"domain": np.where(coords[:, 0] < 50, "left", "right")})
    grid, _ = make_common_grid_from_coords(10, type="pixel", max_x=100, max_y=100)
    mask = compute_dense_tissue_mask_on_any_grid(grid, coords, max_dist_px=8)
    out = generate_virtual_tissue_fixed_grid(
        coords, X, grid, var_names=["a", "b", "c"], attrs=attrs, grid_support_mask=mask, spatial_bandwidth=5
    )
    active = out.obs["active_query_support"].to_numpy()
    assert np.allclose(out.X[active], 4.0)
    labels = out.obs["domain"].astype(str).to_numpy()
    assert (labels[active & (grid[:, 0] < 30)] == "left").all()
    assert (labels[active & (grid[:, 0] > 70)] == "right").all()
    assert out.obsm["domain_probs"].shape == (100, 2)


def test_stack_weighted_evidence():
    c, X, w = stack_weighted_evidence(
        [np.zeros((2, 2)), np.ones((3, 2))], [np.zeros((2, 4)), sp.csr_matrix(np.ones((3, 4)))], [0.2, 0.8]
    )
    assert c.shape == (5, 2) and sp.issparse(X) and np.allclose(w, [0.2, 0.2, 0.8, 0.8, 0.8])
