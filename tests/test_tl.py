import numpy as np
import pytest

import axis


def test_fit_transition_returns_all_pairs(toy_fields):
    assert len(toy_fields) == 6
    assert toy_fields[(0, 1)].shape == (2, 64, 64)
    assert np.all(toy_fields[(1, 1)] == 0)
    assert {"loss", "morph", "mol", "smooth"} <= set(toy_fields.history)


def test_fields_roundtrip(toy_fields, tmp_path):
    toy_fields.save(tmp_path)
    loaded = axis.tl.TransitionFields.load(tmp_path)
    assert set(loaded) == set(toy_fields)
    assert np.array_equal(loaded[(2, 0)], toy_fields[(2, 0)])


def test_match_and_contrast(toy_series, toy_fields):
    matched = axis.tl.match(toy_series, toy_fields, reference=0, reconstruct=True, attr_keys=["domain"], max_dist=4)
    assert all("spatial_axis" in a.obsm for a in toy_series)
    assert np.allclose(toy_series[0].obsm["spatial_axis"], toy_series[0].obsm["spatial_canvas"])
    assert len(matched) == 3 and matched[1].n_obs == toy_series[0].n_obs
    delta = axis.tl.contrast(matched[1], matched[0])
    assert delta.n_vars == matched[0].n_vars


def test_generate(toy_series, toy_fields):
    vt = axis.tl.generate(
        toy_series,
        toy_fields,
        positions=[0, 10, 20],
        query=5,
        grid="pixel",
        grid_size=16,
        attr_keys=["domain"],
        spatial_bandwidth=2,
        max_dist=4,
    )
    meta = vt.uns["axis_generation"]
    assert meta["anchors"] == [0, 1] and meta["alpha"] == pytest.approx(0.5)
    assert vt.n_obs == 256 and vt.obs["valid"].any()
    assert set(vt.obs["domain"].dropna().astype(str)) <= {"D0", "D1", "D2"}


def test_integrate(toy_series, toy_fields):
    pop = axis.tl.integrate(
        toy_series, toy_fields, reference=0, grid_size=16, attr_keys=["domain"], device="cpu", max_dist=4, fuse_radius=8
    )
    assert pop.n_obs > 0 and pop.n_vars == toy_series[0].n_vars
    assert "domain" in pop.obs


def test_plots(toy_series, toy_fields):
    import matplotlib.pyplot as plt

    axis.tl.match(toy_series, toy_fields, reference=0)
    axis.pl.pseudoimage(toy_series[0])
    axis.pl.matched(toy_series, color="domain")
    axis.pl.registration(toy_series[0].uns["axis"]["image"], toy_series[1].uns["axis"]["image"], toy_fields[(0, 1)])
    axis.pl.flow(toy_fields[(0, 1)], step=8)
    axis.pl.spatial(toy_series[0], "domain", spatial_key="spatial_canvas")
    axis.pl.spatial(toy_series[0], "gradient", spatial_key="spatial_canvas")
    axis.pl.axis_weights([0, 20], 10)
    axis.pl.loss(toy_fields.history, smooth=1)
    plt.close("all")


def test_category_colours_do_not_depend_on_present_categories():
    import anndata as ad
    import matplotlib.pyplot as plt

    def colours(labels):
        a = ad.AnnData(np.zeros((len(labels), 1)))
        a.obs["c"] = labels
        a.obsm["spatial"] = np.random.default_rng(0).random((len(labels), 2))
        ax = axis.pl.spatial(a, "c", categories=["0", "1", "2"])
        return {t.get_text(): tuple(h.get_facecolor()[0]) for t, h in zip(ax.get_legend().texts, ax.get_legend().legend_handles)}

    full, partial = colours(["0", "1", "2"] * 3), colours(["0", "2"] * 3)
    assert partial["2"] == full["2"]
    plt.close("all")
