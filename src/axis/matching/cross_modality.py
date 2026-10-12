"""Cross-modality prediction (Methods, "Cross-modality prediction").

Two independently measured modalities are reconstructed on the same support;
the matched locations are then used to train, for every target feature, a
separate XGBoost regressor from the predictor modality. Hyper-parameters are
selected on a validation split by RMSE in two stages:

1. tree structure / regularisation (depth, column subsampling, minimum child
   weight, L1/L2) with the boosting schedule fixed;
2. the boosting schedule (about 1,200-3,000 rounds with learning rates
   0.07-0.03) with the selected structure fixed.

The selected model is refit on training + validation locations and evaluated
once on the held-out test locations (20 % test, 10 % of the remainder for
validation by default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "SearchConfig",
    "build_stage1_structure_candidates",
    "build_stage2_boosting_schedules",
    "select_xgb_params_two_stage",
    "safe_corr",
    "per_target_metrics",
    "CrossModalityResult",
    "fit_cross_modality",
]

HYPERPARAM_SELECTION_METRIC = "rmse"


@dataclass
class SearchConfig:
    """Baseline XGBoost settings around which the search is built.

    Any object with the same attribute names (e.g. an ``argparse.Namespace``)
    can be passed wherever a ``SearchConfig`` is expected.
    """

    n_estimators: int = 2000
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    min_child_weight: float = 0.3
    reg_lambda: float = 0.3
    reg_alpha: float = 0.0


def _unique_dicts(dicts: Sequence[dict]) -> List[dict]:
    out, seen = [], set()
    for d in dicts:
        key = tuple(sorted(d.items()))
        if key not in seen:
            seen.add(key)
            out.append(dict(d))
    return out


def build_stage1_structure_candidates(args=None) -> List[dict]:
    """Stage-1 candidates: the user baseline plus a focused structure/regularisation grid."""
    args = args or SearchConfig()
    base = {"max_depth": 6, "colsample_bytree": 0.8, "min_child_weight": 3.0, "reg_lambda": 3.0, "reg_alpha": 0.0}

    def v(**kw):
        return {**base, **kw}

    refined = [
        v(),
        v(max_depth=4),
        v(max_depth=5),
        v(max_depth=7),
        v(max_depth=8),
        v(colsample_bytree=0.7),
        v(colsample_bytree=0.9),
        v(colsample_bytree=1.0),
        v(min_child_weight=1.0),
        v(min_child_weight=5.0),
        v(min_child_weight=8.0),
        v(reg_lambda=1.0),
        v(reg_lambda=5.0),
        v(reg_lambda=8.0),
        v(reg_alpha=0.1),
        {"max_depth": 8, "colsample_bytree": 0.8, "min_child_weight": 1.0, "reg_lambda": 1.0, "reg_alpha": 0.1},
        {"max_depth": 7, "colsample_bytree": 0.9, "min_child_weight": 2.0, "reg_lambda": 2.0, "reg_alpha": 0.05},
        {"max_depth": 5, "colsample_bytree": 0.7, "min_child_weight": 5.0, "reg_lambda": 5.0, "reg_alpha": 0.1},
        {"max_depth": 4, "colsample_bytree": 0.7, "min_child_weight": 8.0, "reg_lambda": 8.0, "reg_alpha": 0.3},
    ]
    user = {
        "max_depth": int(args.max_depth),
        "colsample_bytree": float(args.colsample_bytree),
        "min_child_weight": float(args.min_child_weight),
        "reg_lambda": float(args.reg_lambda),
        "reg_alpha": float(args.reg_alpha),
    }
    return _unique_dicts([user] + refined)


def build_stage2_boosting_schedules(args=None) -> List[dict]:
    """Stage-2 schedules: 0.6x/0.8x/1x/1.25x/1.5x rounds with inversely scaled learning rate."""
    args = args or SearchConfig()
    n, lr = int(args.n_estimators), float(args.learning_rate)

    def sched(fn, fl):
        return {"n_estimators": max(200, int(round(n * fn))), "learning_rate": min(0.30, max(0.005, lr * fl))}

    return _unique_dicts(
        [sched(0.60, 1.40), sched(0.80, 1.20), {"n_estimators": n, "learning_rate": lr}, sched(1.25, 0.80), sched(1.50, 0.60)]
    )


def _validation_rmse_for_params(X_train, y_train, X_val, y_val, params: dict) -> float:
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    return float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))


def select_xgb_params_two_stage(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    base_params: dict,
    args=None,
    structure_candidates: Optional[Sequence[dict]] = None,
    boosting_schedules: Optional[Sequence[dict]] = None,
) -> Tuple[dict, float, float, List[dict], List[dict]]:
    """Select per-target XGBoost parameters in two validation stages (no early stopping).

    Returns ``(selected_params, best_val_rmse, best_stage1_rmse, stage1_scores, stage2_scores)``.
    """
    args = args or SearchConfig()
    structure_candidates = structure_candidates or build_stage1_structure_candidates(args)
    boosting_schedules = boosting_schedules or build_stage2_boosting_schedules(args)

    fixed_stage1 = {
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "subsample": float(args.subsample),
    }
    stage1_scores, best_structure, best_stage1_rmse = [], None, np.inf
    for candidate_id, structure in enumerate(structure_candidates):
        score = _validation_rmse_for_params(X_train, y_train, X_val, y_val, {**base_params, **fixed_stage1, **structure})
        stage1_scores.append(
            {
                "candidate_id": int(candidate_id),
                **{k: (int(v) if k == "max_depth" else float(v)) for k, v in structure.items()},
                "validation_rmse": float(score),
            }
        )
        if np.isfinite(score) and score < best_stage1_rmse:
            best_stage1_rmse, best_structure = score, dict(structure)
    if best_structure is None:
        raise RuntimeError("Stage-1 validation search failed to produce a finite RMSE.")

    default = {"n_estimators": int(args.n_estimators), "learning_rate": float(args.learning_rate)}
    best_schedule, best_final_rmse, stage2_scores = dict(default), float(best_stage1_rmse), []
    for schedule_id, schedule in enumerate(boosting_schedules):
        is_default = int(schedule["n_estimators"]) == default["n_estimators"] and np.isclose(
            float(schedule["learning_rate"]), default["learning_rate"]
        )
        if is_default:
            score = float(best_stage1_rmse)
        else:
            params = {**base_params, "subsample": float(args.subsample), **best_structure, **schedule}
            score = _validation_rmse_for_params(X_train, y_train, X_val, y_val, params)
        stage2_scores.append(
            {
                "schedule_id": int(schedule_id),
                "n_estimators": int(schedule["n_estimators"]),
                "learning_rate": float(schedule["learning_rate"]),
                "validation_rmse": float(score),
            }
        )
        if np.isfinite(score) and score < best_final_rmse:
            best_final_rmse, best_schedule = score, dict(schedule)

    selected = {
        "n_estimators": int(best_schedule["n_estimators"]),
        "learning_rate": float(best_schedule["learning_rate"]),
        "max_depth": int(best_structure["max_depth"]),
        "subsample": float(args.subsample),
        "colsample_bytree": float(best_structure["colsample_bytree"]),
        "min_child_weight": float(best_structure["min_child_weight"]),
        "reg_lambda": float(best_structure["reg_lambda"]),
        "reg_alpha": float(best_structure["reg_alpha"]),
    }
    return selected, float(best_final_rmse), float(best_stage1_rmse), stage1_scores, stage2_scores


def safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    """Pearson / Spearman correlation, ``NaN`` for constant or tiny inputs."""
    from scipy.stats import pearsonr, spearmanr

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be pearson or spearman")
    try:
        return float((pearsonr if method == "pearson" else spearmanr)(x, y)[0])
    except Exception:
        return np.nan


def per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: Sequence[str], name_col: str = "protein") -> pd.DataFrame:
    """Pearson, Spearman, R², MAE and RMSE for every target column."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rows = []
    for j, name in enumerate(names):
        yt, yp = np.asarray(y_true[:, j], dtype=float), np.asarray(y_pred[:, j], dtype=float)
        rows.append(
            {
                name_col: str(name),
                "pearson_r": safe_corr(yt, yp, "pearson"),
                "spearman_r": safe_corr(yt, yp, "spearman"),
                "r2": float(r2_score(yt, yp)) if np.std(yt) > 0 else np.nan,
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class CrossModalityResult:
    """Output of [`fit_cross_modality`][axis.matching.cross_modality.fit_cross_modality]."""

    models: Dict[str, object]
    metrics: pd.DataFrame
    selected_params: Dict[str, dict]
    split: np.ndarray
    test_prediction: np.ndarray
    target_names: List[str] = field(default_factory=list)

    def predict(self, X) -> np.ndarray:
        """Predict every target for new predictor-modality observations."""
        return np.column_stack([np.asarray(self.models[n].predict(X), dtype=np.float32) for n in self.target_names])


def fit_cross_modality(
    X,
    Y: np.ndarray,
    target_names: Sequence[str],
    *,
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 0,
    use_gpu: bool = False,
    n_jobs: int = 8,
    search: Optional[SearchConfig] = None,
) -> CrossModalityResult:
    """Train one regressor per target modality feature on spatially matched locations.

    Parameters
    ----------
    X
        Predictor-modality matrix ``(n, p)`` (dense or sparse).
    Y
        Target-modality matrix ``(n, t)``.
    target_names
        Names of the ``t`` targets.
    """
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    search = search or SearchConfig()
    Y = np.asarray(Y, dtype=np.float32)
    n = Y.shape[0]
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size, random_state=seed, shuffle=True)
    split = np.array(["train"] * n, dtype=object)
    split[val_idx], split[test_idx] = "val", "test"
    trainval_idx = np.concatenate([train_idx, val_idx])

    base_params = dict(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        n_jobs=n_jobs,
        random_state=seed,
        verbosity=0,
    )
    structures, schedules = build_stage1_structure_candidates(search), build_stage2_boosting_schedules(search)
    models, selected = {}, {}
    pred_test = np.zeros((len(test_idx), Y.shape[1]), dtype=np.float32)
    for j, name in enumerate(target_names):
        params, *_ = select_xgb_params_two_stage(
            X[train_idx], Y[train_idx, j], X[val_idx], Y[val_idx, j], base_params, search, structures, schedules
        )
        model = XGBRegressor(**{**base_params, **params})
        model.fit(X[trainval_idx], Y[trainval_idx, j], verbose=False)
        pred_test[:, j] = model.predict(X[test_idx])
        models[str(name)], selected[str(name)] = model, params
    metrics = per_target_metrics(Y[test_idx], pred_test, target_names, name_col="target")
    return CrossModalityResult(models, metrics, selected, split, pred_test, [str(t) for t in target_names])
