import numpy as np
import pytest

from axis.matching import SearchConfig, build_stage1_structure_candidates, build_stage2_boosting_schedules


def test_xgb_search_space_matches_methods():
    schedules = build_stage2_boosting_schedules(SearchConfig())
    rounds = [s["n_estimators"] for s in schedules]
    rates = [round(s["learning_rate"], 3) for s in schedules]
    assert rounds == [1200, 1600, 2000, 2500, 3000]
    assert rates == [0.07, 0.06, 0.05, 0.04, 0.03]
    stage1 = build_stage1_structure_candidates(SearchConfig())
    assert len(stage1) == len({tuple(sorted(c.items())) for c in stage1})


def test_fit_cross_modality_small(rng):
    pytest.importorskip("xgboost")
    from axis.matching import fit_cross_modality

    X = rng.normal(size=(300, 5))
    Y = np.column_stack([X[:, 0] * 2 + rng.normal(0, 0.1, 300), X[:, 1] - X[:, 2]])
    search = SearchConfig(n_estimators=40, learning_rate=0.2)
    res = fit_cross_modality(X, Y, ["p1", "p2"], n_jobs=1, search=search)
    assert res.metrics["pearson_r"].min() > 0.8
    assert res.predict(X[:3]).shape == (3, 2)
