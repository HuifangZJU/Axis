#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train Xenium-to-protein XGBoost models on selected paired samples and predict
all requested protein channels for selected held-out pairs.

Input naming convention under ROOT_DIR
--------------------------------------
virtual_sennet_<pair_id>_xenium.h5ad
virtual_sennet_<pair_id>_protein.h5ad

Output
------
One h5ad prediction file for each held-out test pair:
virtual_sennet_<pair_id>_protein_predicted.h5ad

Predictions in .X are saved on the original protein scale. Each file stores
its pair ID in both obs["pair_id"] and uns["pair_id"].
"""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# =============================================================================
# User configuration
# =============================================================================

ROOT_DIR = Path("/media/huifang/data/sennet/Axis_generated_proteins")
OUTPUT_DIR = ROOT_DIR / "xgb_test_pair_predictions"

PROTEIN_NAMES = [
    "p53",
    "p16",
    "pH2AX",
    "HSP47",
    "PD1",
    "PDL1",
    "CD32b",
    "SPP1",
    "CD23",
]

# IDs 0-43 excluding 8, 25, and 26 produce 41 valid IDs, not 40.
# The default split below uses 30 training pairs and 10 test pairs and leaves
# pair 43 unused. Replace these lists with your intended split if needed.
ALL_AVAILABLE_PAIR_IDS = [i for i in range(43) if i not in {8, 25, 26}]
TEST_PAIR_IDS=[0,1,16,23,24]
TRAIN_PAIR_IDS = [i for i in ALL_AVAILABLE_PAIR_IDS if i not in TEST_PAIR_IDS]


# Use the final three training pairs only for early-stopping model selection.
# After selecting the number of boosting rounds, every model is refit using all
# 30 training pairs.
N_VALIDATION_PAIRS = 3

COORD_OBSM_KEY = "spatial"
COORD_ROUND_DECIMALS = 6

LOG1P_GENE = True
LOG1P_PROTEIN = True

USE_GPU = True
SEED = 0
N_ESTIMATORS = 2000
LEARNING_RATE = 0.05
MAX_DEPTH = 6
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
MIN_CHILD_WEIGHT = 1.0
REG_LAMBDA = 1.0
REG_ALPHA = 0.0
EARLY_STOPPING_ROUNDS = 50
N_JOBS = 8

# Store observed test proteins in the same output h5ad for direct evaluation.
# Set to False if only predictions are desired.
INCLUDE_TRUTH_LAYER = True


# =============================================================================
# Logging and matrix utilities
# =============================================================================


def log(message: str) -> None:
    print(message, flush=True)


def to_dense_2d(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def stack_feature_matrices(parts: Sequence):
    if any(sp.issparse(x) for x in parts):
        sparse_parts = [
            x.astype(np.float32).tocsr()
            if sp.issparse(x)
            else sp.csr_matrix(np.asarray(x, dtype=np.float32))
            for x in parts
        ]
        return sp.vstack(sparse_parts, format="csr")

    return np.vstack([
        np.asarray(x, dtype=np.float32)
        for x in parts
    ])


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    if method == "pearson":
        return float(pearsonr(x, y)[0])
    if method == "spearman":
        return float(spearmanr(x, y)[0])
    raise ValueError("method must be 'pearson' or 'spearman'.")


# =============================================================================
# Paths and input validation
# =============================================================================


def gene_path(pair_id: int) -> Path:
    return ROOT_DIR / f"virtual_sennet_{pair_id}_xenium.h5ad"


def protein_path(pair_id: int) -> Path:
    return ROOT_DIR / f"virtual_sennet_{pair_id}_protein.h5ad"


def validate_pair_lists() -> None:
    train_set = set(TRAIN_PAIR_IDS)
    test_set = set(TEST_PAIR_IDS)

    if len(TRAIN_PAIR_IDS) != 35:
        raise ValueError(
            f"TRAIN_PAIR_IDS must contain 35 IDs, but contains {len(TRAIN_PAIR_IDS)}."
        )
    if len(TEST_PAIR_IDS) != 5:
        raise ValueError(
            f"TEST_PAIR_IDS must contain 5 IDs, but contains {len(TEST_PAIR_IDS)}."
        )
    if len(train_set) != len(TRAIN_PAIR_IDS):
        raise ValueError("TRAIN_PAIR_IDS contains duplicate IDs.")
    if len(test_set) != len(TEST_PAIR_IDS):
        raise ValueError("TEST_PAIR_IDS contains duplicate IDs.")
    overlap = train_set.intersection(test_set)
    if overlap:
        raise ValueError(f"Train and test pair IDs overlap: {sorted(overlap)}")

    invalid = (train_set | test_set) - set(ALL_AVAILABLE_PAIR_IDS)
    if invalid:
        raise ValueError(
            "The split contains excluded or out-of-range pair IDs: "
            f"{sorted(invalid)}"
        )

    if N_VALIDATION_PAIRS <= 0 or N_VALIDATION_PAIRS >= len(TRAIN_PAIR_IDS):
        raise ValueError(
            "N_VALIDATION_PAIRS must be greater than 0 and smaller than the "
            "number of training pairs."
        )

    missing_files = []
    for pair_id in TRAIN_PAIR_IDS + TEST_PAIR_IDS:
        for path in (gene_path(pair_id), protein_path(pair_id)):
            if not path.exists():
                missing_files.append(str(path))

    if missing_files:
        raise FileNotFoundError(
            "The following required files are missing:\n" + "\n".join(missing_files)
        )


# =============================================================================
# Coordinate extraction and pair alignment
# =============================================================================


def get_spatial_coordinates(adata: ad.AnnData) -> np.ndarray:
    if COORD_OBSM_KEY not in adata.obsm:
        raise KeyError(
            f"Expected adata.obsm['{COORD_OBSM_KEY}']; available keys are "
            f"{list(adata.obsm.keys())}."
        )

    coords = np.asarray(adata.obsm[COORD_OBSM_KEY], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['{COORD_OBSM_KEY}'] must have shape (n_cells, >=2), "
            f"but has shape {coords.shape}."
        )
    if not np.isfinite(coords[:, :2]).all():
        raise ValueError("Spatial coordinates contain non-finite values.")

    return coords[:, :2]


def coordinate_keys(coords: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(coords, dtype=np.float64), COORD_ROUND_DECIMALS)
    return np.asarray([
        f"{x:.{COORD_ROUND_DECIMALS}f}__{y:.{COORD_ROUND_DECIMALS}f}"
        for x, y in rounded
    ])


def align_pair(
    gene_adata: ad.AnnData,
    protein_adata: ad.AnnData,
    pair_id: int,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Align one Xenium/protein pair to the same cell order."""

    # Same names and same order.
    if (
        gene_adata.n_obs == protein_adata.n_obs
        and np.array_equal(gene_adata.obs_names, protein_adata.obs_names)
    ):
        return gene_adata, protein_adata

    # Same complete name set, but different order.
    if (
        gene_adata.n_obs == protein_adata.n_obs
        and set(gene_adata.obs_names) == set(protein_adata.obs_names)
    ):
        return gene_adata, protein_adata[gene_adata.obs_names].copy()

    gene_coords = get_spatial_coordinates(gene_adata)
    protein_coords = get_spatial_coordinates(protein_adata)

    # Same row count and coordinates already in the same order.
    if (
        gene_adata.n_obs == protein_adata.n_obs
        and np.allclose(
            gene_coords,
            protein_coords,
            atol=10.0 ** (-COORD_ROUND_DECIMALS),
            rtol=0.0,
        )
    ):
        return gene_adata, protein_adata

    # Reorder by unique rounded coordinate keys.
    gene_keys = coordinate_keys(gene_coords)
    protein_keys = coordinate_keys(protein_coords)

    if len(np.unique(gene_keys)) != len(gene_keys):
        raise ValueError(
            f"Pair {pair_id}: duplicate Xenium coordinates prevent unique alignment."
        )
    if len(np.unique(protein_keys)) != len(protein_keys):
        raise ValueError(
            f"Pair {pair_id}: duplicate protein coordinates prevent unique alignment."
        )

    protein_index = {key: index for index, key in enumerate(protein_keys)}
    missing = [key for key in gene_keys if key not in protein_index]
    if missing:
        raise ValueError(
            f"Pair {pair_id}: {len(missing)} Xenium coordinates do not have "
            "matching protein coordinates."
        )

    reorder = np.asarray([protein_index[key] for key in gene_keys], dtype=int)
    aligned_protein = protein_adata[reorder].copy()

    if aligned_protein.n_obs != gene_adata.n_obs:
        raise RuntimeError(f"Pair {pair_id}: alignment produced unequal cell counts.")

    return gene_adata, aligned_protein


# =============================================================================
# Feature harmonization and extraction
# =============================================================================


def find_common_genes(pair_ids: Sequence[int]) -> List[str]:
    """Find genes present in every selected Xenium file."""
    common = None
    first_order = None

    for pair_id in pair_ids:
        adata = ad.read_h5ad(gene_path(pair_id), backed="r")
        names = [str(x) for x in adata.var_names]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Pair {pair_id}: duplicated Xenium var_names are not supported."
            )

        name_set = set(names)
        if common is None:
            common = name_set
            first_order = names
        else:
            common &= name_set

        adata.file.close()

    assert common is not None and first_order is not None
    ordered_common = [gene for gene in first_order if gene in common]

    if not ordered_common:
        raise ValueError("No genes are shared by all selected Xenium files.")

    return ordered_common


def validate_protein_channels(pair_ids: Sequence[int]) -> None:
    for pair_id in pair_ids:
        adata = ad.read_h5ad(protein_path(pair_id), backed="r")
        available = set(str(x) for x in adata.var_names)
        missing = [protein for protein in PROTEIN_NAMES if protein not in available]
        adata.file.close()

        if missing:
            raise KeyError(
                f"Pair {pair_id}: missing requested protein channels: {missing}"
            )


def extract_gene_matrix(
    gene_adata: ad.AnnData,
    common_genes: Sequence[str],
):
    x = gene_adata[:, list(common_genes)].X

    if sp.issparse(x):
        x = x.astype(np.float32).tocsr()
        if LOG1P_GENE:
            if x.data.size and np.min(x.data) < 0:
                raise ValueError("Gene matrix contains negative values; log1p is invalid.")
            x = x.copy()
            x.data = np.log1p(x.data)
        return x

    x = np.asarray(x, dtype=np.float32)
    if LOG1P_GENE:
        if np.nanmin(x) < 0:
            raise ValueError("Gene matrix contains negative values; log1p is invalid.")
        x = np.log1p(x)
    return x


def extract_protein_matrix(
    protein_adata: ad.AnnData,
    apply_log1p: bool,
) -> np.ndarray:
    y = to_dense_2d(protein_adata[:, PROTEIN_NAMES].X).astype(np.float32)
    y = np.clip(y, a_min=0.0, a_max=None)

    if apply_log1p:
        y = np.log1p(y)

    return y


# =============================================================================
# Training-data assembly
# =============================================================================


def load_training_data(
    pair_ids: Sequence[int],
    common_genes: Sequence[str],
):
    x_parts = []
    y_parts = []
    pair_labels = []

    for pair_id in pair_ids:
        start = time.time()
        gene_adata = ad.read_h5ad(gene_path(pair_id))
        protein_adata = ad.read_h5ad(protein_path(pair_id))
        gene_adata, protein_adata = align_pair(gene_adata, protein_adata, pair_id)

        x_pair = extract_gene_matrix(gene_adata, common_genes)
        y_pair = extract_protein_matrix(
            protein_adata,
            apply_log1p=LOG1P_PROTEIN,
        )

        if x_pair.shape[0] != y_pair.shape[0]:
            raise ValueError(
                f"Pair {pair_id}: gene/protein row mismatch after alignment: "
                f"{x_pair.shape[0]} versus {y_pair.shape[0]}."
            )

        x_parts.append(x_pair)
        y_parts.append(y_pair)
        pair_labels.append(np.full(gene_adata.n_obs, pair_id, dtype=np.int16))

        log(
            f"Loaded training pair {pair_id}: {gene_adata.n_obs} cells "
            f"in {time.time() - start:.1f}s"
        )

    x_all = stack_feature_matrices(x_parts)
    y_all = np.vstack(y_parts).astype(np.float32)
    pair_labels_all = np.concatenate(pair_labels)

    return x_all, y_all, pair_labels_all


# =============================================================================
# XGBoost training
# =============================================================================


def xgb_base_parameters() -> Dict:
    return {
        "learning_rate": LEARNING_RATE,
        "max_depth": MAX_DEPTH,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "min_child_weight": MIN_CHILD_WEIGHT,
        "reg_lambda": REG_LAMBDA,
        "reg_alpha": REG_ALPHA,
        "objective": "reg:squarederror",
        "tree_method": "gpu_hist" if USE_GPU else "hist",
        "predictor": "gpu_predictor" if USE_GPU else "auto",
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "verbosity": 0,
    }


def train_models(
    x_all,
    y_all: np.ndarray,
    pair_labels: np.ndarray,
) -> Tuple[List[XGBRegressor], List[int]]:
    validation_pair_ids = TRAIN_PAIR_IDS[-N_VALIDATION_PAIRS:]
    validation_mask = np.isin(pair_labels, validation_pair_ids)
    fit_mask = ~validation_mask

    if fit_mask.sum() == 0 or validation_mask.sum() == 0:
        raise ValueError("Training or validation subset is empty.")

    x_fit = x_all[fit_mask]
    y_fit = y_all[fit_mask]
    x_val = x_all[validation_mask]
    y_val = y_all[validation_mask]

    log(f"Early-stopping fit pairs: {TRAIN_PAIR_IDS[:-N_VALIDATION_PAIRS]}")
    log(f"Early-stopping validation pairs: {validation_pair_ids}")
    log(f"Fit cells: {x_fit.shape[0]:,}; validation cells: {x_val.shape[0]:,}")

    base_params = xgb_base_parameters()
    final_models = []
    selected_rounds = []

    for protein_index, protein_name in enumerate(PROTEIN_NAMES):
        start = time.time()

        selector = XGBRegressor(
            n_estimators=N_ESTIMATORS,
            **base_params,
        )
        selector.fit(
            x_fit,
            y_fit[:, protein_index],
            eval_set=[(x_val, y_val[:, protein_index])],
            eval_metric="rmse",
            verbose=False,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )

        best_iteration = getattr(selector, "best_iteration", None)
        if best_iteration is None:
            n_rounds = N_ESTIMATORS
        else:
            n_rounds = int(best_iteration) + 1

        # Refit on all 30 training pairs using the selected number of rounds.
        final_model = XGBRegressor(
            n_estimators=n_rounds,
            **base_params,
        )
        final_model.fit(
            x_all,
            y_all[:, protein_index],
            verbose=False,
        )

        final_models.append(final_model)
        selected_rounds.append(n_rounds)

        log(
            f"[{protein_index + 1}/{len(PROTEIN_NAMES)}] {protein_name}: "
            f"selected {n_rounds} rounds and refit on all training pairs "
            f"in {time.time() - start:.1f}s"
        )

    return final_models, selected_rounds


# =============================================================================
# Test prediction and output assembly
# =============================================================================


def predict_and_save_test_pairs(
    models: Sequence[XGBRegressor],
    common_genes: Sequence[str],
    selected_rounds: Sequence[int],
) -> List[Path]:
    """Predict and save one h5ad file for each held-out test pair."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for pair_id in TEST_PAIR_IDS:
        start = time.time()
        gene_adata = ad.read_h5ad(gene_path(pair_id))
        protein_adata = ad.read_h5ad(protein_path(pair_id))
        gene_adata, protein_adata = align_pair(gene_adata, protein_adata, pair_id)

        x_test = extract_gene_matrix(gene_adata, common_genes)
        pred_model_scale = np.column_stack([
            model.predict(x_test).astype(np.float32, copy=False)
            for model in models
        ])

        # Convert model outputs back to the original protein scale before saving.
        if LOG1P_PROTEIN:
            predictions = np.expm1(pred_model_scale)
        else:
            predictions = pred_model_scale
        predictions = np.clip(
            predictions,
            a_min=0.0,
            a_max=None,
        ).astype(np.float32)

        truth_original = extract_protein_matrix(
            protein_adata,
            apply_log1p=False,
        ).astype(np.float32)

        coords = get_spatial_coordinates(gene_adata).astype(np.float32)

        # Because each pair is saved separately, retain its original cell names.
        output_obs = pd.DataFrame(index=gene_adata.obs_names.astype(str).copy())
        output_obs.index.name = "cell"
        output_obs["pair_id"] = np.full(
            gene_adata.n_obs,
            pair_id,
            dtype=np.int16,
        )
        output_obs["source_obs_name"] = gene_adata.obs_names.astype(str).to_numpy()
        output_obs["cell_index_within_pair"] = np.arange(
            gene_adata.n_obs,
            dtype=np.int32,
        )

        # Preserve only compact, useful cell metadata when available.
        for column in ("cell_id", "region_id"):
            if column in gene_adata.obs.columns:
                output_obs[column] = gene_adata.obs[column].astype(str).to_numpy()

        output_var = pd.DataFrame(
            index=pd.Index(PROTEIN_NAMES, name="protein")
        )

        output = ad.AnnData(
            X=predictions,
            obs=output_obs,
            var=output_var,
        )
        output.obsm[COORD_OBSM_KEY] = coords

        if INCLUDE_TRUTH_LAYER:
            output.layers["truth"] = truth_original
            output.layers["abs_error"] = np.abs(
                predictions - truth_original
            ).astype(np.float32)

        # Pair-specific metrics.
        metric_rows = []
        for protein_index, protein_name in enumerate(PROTEIN_NAMES):
            truth = truth_original[:, protein_index]
            prediction = predictions[:, protein_index]
            metric_rows.append({
                "protein": protein_name,
                "pearson_r": safe_corr(truth, prediction, "pearson"),
                "spearman_r": safe_corr(truth, prediction, "spearman"),
                "r2": (
                    float(r2_score(truth, prediction))
                    if np.std(truth) > 0
                    else np.nan
                ),
                "mae": float(mean_absolute_error(truth, prediction)),
                "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
            })

        output.uns["description"] = (
            "XGBoost-predicted protein expression for one held-out Xenium "
            "test pair. Predictions in X are on the original protein scale."
        )
        output.uns["pair_id"] = int(pair_id)
        output.uns["train_pair_ids"] = np.asarray(
            TRAIN_PAIR_IDS,
            dtype=np.int16,
        )
        output.uns["test_pair_ids"] = np.asarray(
            TEST_PAIR_IDS,
            dtype=np.int16,
        )
        output.uns["protein_names"] = np.asarray(PROTEIN_NAMES, dtype=str)
        output.uns["n_common_genes"] = int(len(common_genes))
        output.uns["log1p_gene_for_training"] = bool(LOG1P_GENE)
        output.uns["log1p_protein_for_training"] = bool(LOG1P_PROTEIN)
        output.uns["selected_boosting_rounds"] = {
            protein: int(rounds)
            for protein, rounds in zip(PROTEIN_NAMES, selected_rounds)
        }
        output.uns["test_metrics"] = pd.DataFrame(
            metric_rows
        ).set_index("protein")

        output_path = (
            OUTPUT_DIR
            / f"virtual_sennet_{pair_id}_protein_predicted.h5ad"
        )
        output.write_h5ad(
            output_path,
            compression="gzip",
        )
        saved_paths.append(output_path)

        log(
            f"Predicted and saved test pair {pair_id}: "
            f"{gene_adata.n_obs} cells -> {output_path} "
            f"in {time.time() - start:.1f}s"
        )

    return saved_paths


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    validate_pair_lists()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_pair_ids = TRAIN_PAIR_IDS + TEST_PAIR_IDS
    unused_pair_ids = sorted(set(ALL_AVAILABLE_PAIR_IDS) - set(selected_pair_ids))

    log(f"ROOT_DIR: {ROOT_DIR}")
    log(f"Training pairs ({len(TRAIN_PAIR_IDS)}): {TRAIN_PAIR_IDS}")
    log(f"Test pairs ({len(TEST_PAIR_IDS)}): {TEST_PAIR_IDS}")
    log(f"Unused valid pairs: {unused_pair_ids}")
    log(f"Proteins: {PROTEIN_NAMES}")

    log("Finding genes shared by all selected Xenium files...")
    common_genes = find_common_genes(selected_pair_ids)
    log(f"Common genes: {len(common_genes)}")

    log("Validating requested protein channels...")
    validate_protein_channels(selected_pair_ids)

    log("Loading the 30 training pairs...")
    x_train_all, y_train_all, train_pair_labels = load_training_data(
        TRAIN_PAIR_IDS,
        common_genes,
    )
    log(
        f"Combined training data: {x_train_all.shape[0]:,} cells x "
        f"{x_train_all.shape[1]:,} genes"
    )

    log("Training nine protein models...")
    models, selected_rounds = train_models(
        x_train_all,
        y_train_all,
        train_pair_labels,
    )

    log("Predicting and saving the 10 held-out pairs...")
    saved_paths = predict_and_save_test_pairs(
        models=models,
        common_genes=common_genes,
        selected_rounds=selected_rounds,
    )

    log(f"Saved {len(saved_paths)} pair-specific prediction files:")
    for output_path in saved_paths:
        log(f"  {output_path}")
    log("Done.")


if __name__ == "__main__":
    main()
