from virtual_utils import *
from axis.coordinate_utils import find_inverse_warp_coords
import os
import numpy as np
import scipy.sparse as sp

# =========================
# CONFIG
# =========================
BIOLOGICAL_AXIS_NAME = "dlpfc_z"
MODE = "analysis"   # "query" or "analysis"

AXIS_POSITIONS = {
    0: 0,
    1: 10,
    2: 310,
    3: 320,
}

GROUP_ID = 1
ANCHOR_A_ID = 2
ANCHOR_B_ID = 3
QUERY_AXIS_VALUES = np.arange(310, 321, 10).tolist()

ROOT = "/media/huifang/data/registration/DLPFC/huifang"
NET_IN = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ad")
AXIS_OUT = os.path.join(ROOT, "pairwise_output", f"group_{GROUP_ID}")

GRID_SIZE = 256
MAX_DIST_PX = 5.0

# preprocessing / weighting
APPLY_LOG1P_BEFORE_GENERATION = False
AXIS_WEIGHT_KERNEL = "gaussian"
AXIS_WEIGHT_BANDWIDTH = None
SPATIAL_BANDWIDTH = None
KNN = 48
MIN_NEIGHBORS = 6
RADIUS = None

# query-support policy
# query mode: keep only locally trustworthy locations
# analysis mode: usually disabled for dense, directly comparable maps
QUERY_MAX_SUPPORT_DIST = None           # None -> handled inside generator
QUERY_MIN_CONFIDENCE_ABS = 1e-8         # tiny absolute floor
QUERY_MIN_CONFIDENCE_RATIO = 0.0        # disable dataset-relative confidence pruning
QUERY_CONFIDENCE_REF = "median"

ANALYSIS_ENFORCE_QUERY_SUPPORT = False
ANALYSIS_MAX_SUPPORT_DIST = None
ANALYSIS_MIN_CONFIDENCE_ABS = None
ANALYSIS_MIN_CONFIDENCE_RATIO = 0.0
ANALYSIS_CONFIDENCE_REF = "median"

# gene generation
GENE_MODE = "mean"      # "mean", "kl", "latent_kl"
N_PROGRAMS = 32         # used only for latent_kl

# optional attributes
USE_ATTRIBUTES = True
ATTRIBUTE_COLUMNS = ["layer_guess_reordered"]     # example
ATTRIBUTE_SCHEMA = {"layer_guess_reordered": "categorical"}

# gene subset (set to None to use all genes)
GENE_SUBSET = ["PCP4", "MOBP", "MFGE8"]

OUT_DIR = os.path.join(
    ROOT,
    "virtual_tissue",
    f"{BIOLOGICAL_AXIS_NAME}_group{GROUP_ID}_grid{GRID_SIZE}",
    MODE,
)
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# HELPERS
# =========================
def maybe_subset_genes(expr_a_raw, expr_b_raw, genes, gene_subset):
    if gene_subset is None:
        return expr_a_raw, expr_b_raw, genes

    gene_to_idx = {g: i for i, g in enumerate(genes.tolist())}
    keep_idx = [gene_to_idx[g] for g in gene_subset if g in gene_to_idx]
    if len(keep_idx) == 0:
        raise ValueError("None of the requested genes found in common gene list.")

    genes = genes[keep_idx]
    expr_a_raw = expr_a_raw[:, keep_idx]
    expr_b_raw = expr_b_raw[:, keep_idx]
    return expr_a_raw, expr_b_raw, genes


def maybe_log1p(X):
    return matrix_log1p(X) if APPLY_LOG1P_BEFORE_GENERATION else X


def resolve_support_kwargs(mode):
    """
    Streamlined support policy:
      - query: only enforce local trustworthiness
      - analysis: default off for dense aligned maps
    """
    if mode == "query":
        return dict(
            enforce_query_support=False,
            max_support_dist=QUERY_MAX_SUPPORT_DIST,
            min_confidence_abs=QUERY_MIN_CONFIDENCE_ABS,
            min_confidence_ratio=QUERY_MIN_CONFIDENCE_RATIO,
            confidence_ref=QUERY_CONFIDENCE_REF,
        )
    elif mode == "analysis":
        return dict(
            enforce_query_support=ANALYSIS_ENFORCE_QUERY_SUPPORT,
            max_support_dist=ANALYSIS_MAX_SUPPORT_DIST,
            min_confidence_abs=ANALYSIS_MIN_CONFIDENCE_ABS,
            min_confidence_ratio=ANALYSIS_MIN_CONFIDENCE_RATIO,
            confidence_ref=ANALYSIS_CONFIDENCE_REF,
        )
    else:
        raise ValueError("MODE must be one of {'query','analysis'}")


def build_mode_context():
    """
    Returns:
      axis_a, axis_b,
      coords_a, coords_b,
      expr_a_raw, expr_b_raw, genes,
      attrs_a_raw, attrs_b_raw,
      flow_a_to_b, flow_b_to_a,
      fixed_grid_pts, fixed_grid_support
    """
    axis_a = float(AXIS_POSITIONS[ANCHOR_A_ID])
    axis_b = float(AXIS_POSITIONS[ANCHOR_B_ID])

    if MODE == "analysis":
        expr_a_raw, genes_a, coords_a, attrs_a_raw = load_h5ad_X_and_genes_and_coords(
            H5AD_DIR, GROUP_ID, ANCHOR_A_ID, attr_columns=ATTRIBUTE_COLUMNS if USE_ATTRIBUTES else None
        )
        expr_b_raw, genes_b, coords_b, attrs_b_raw = load_h5ad_X_and_genes_and_coords(
            H5AD_DIR, GROUP_ID, ANCHOR_B_ID, attr_columns=ATTRIBUTE_COLUMNS if USE_ATTRIBUTES else None
        )

        flow_a_to_b = None
        flow_b_to_a = None

        union_coords = np.vstack([coords_a, coords_b]).astype(np.float32)
        fixed_grid_pts, _ = make_common_grid_from_coords(union_coords, GRID_SIZE)
        fixed_grid_support = compute_dense_tissue_mask_on_grid(
            grid_pts=fixed_grid_pts,
            union_spots=union_coords,
            grid_size=GRID_SIZE,
            max_dist_px=MAX_DIST_PX,
            closing_iters=1,
            keep_largest=True,
        )

    elif MODE == "query":
        coords_a = load_coords_npz(NET_IN, GROUP_ID, ANCHOR_A_ID)
        coords_b = load_coords_npz(NET_IN, GROUP_ID, ANCHOR_B_ID)

        flow_a_to_b = load_axis_flow(AXIS_OUT, fixed_id=ANCHOR_B_ID, moving_id=ANCHOR_A_ID)
        flow_b_to_a = load_axis_flow(AXIS_OUT, fixed_id=ANCHOR_A_ID, moving_id=ANCHOR_B_ID)

        expr_a_raw, genes_a, attrs_a_raw = load_h5ad_X_and_genes(
            H5AD_DIR, GROUP_ID, ANCHOR_A_ID, attr_columns=ATTRIBUTE_COLUMNS if USE_ATTRIBUTES else None
        )
        expr_b_raw, genes_b, attrs_b_raw = load_h5ad_X_and_genes(
            H5AD_DIR, GROUP_ID, ANCHOR_B_ID, attr_columns=ATTRIBUTE_COLUMNS if USE_ATTRIBUTES else None
        )

        fixed_grid_pts = None
        fixed_grid_support = None

    else:
        raise ValueError("MODE must be one of {'query','analysis'}")

    expr_a_raw, expr_b_raw, genes = ensure_same_genes_keep_order(expr_a_raw, genes_a, expr_b_raw, genes_b)
    expr_a_raw, expr_b_raw, genes = maybe_subset_genes(expr_a_raw, expr_b_raw, genes, GENE_SUBSET)

    # gene subsetting does not change attrs
    # attrs already aligned row-wise to coords / expr rows

    return (
        axis_a, axis_b,
        coords_a, coords_b,
        expr_a_raw, expr_b_raw, genes,
        attrs_a_raw, attrs_b_raw,
        flow_a_to_b, flow_b_to_a,
        fixed_grid_pts, fixed_grid_support,
    )


def build_query_grid_and_evidence(
    alpha,
    coords_a,
    coords_b,
    expr_a,
    expr_b,
    flow_a_to_b,
    flow_b_to_a,
    source_weights,
):
    """
    Query mode:
      1) warp endpoint coordinates to query state
      2) build query-specific grid + support from warped union
      3) stack weighted evidence
    """
    coords_a_q = find_inverse_warp_coords(
        coords_a,
        (alpha * flow_a_to_b).astype(np.float32)
    ).astype(np.float32)

    coords_b_q = find_inverse_warp_coords(
        coords_b,
        ((1.0 - alpha) * flow_b_to_a).astype(np.float32)
    ).astype(np.float32)

    union_coords = np.vstack([coords_a_q, coords_b_q]).astype(np.float32)

    grid_pts, _ = make_common_grid_from_coords(union_coords, GRID_SIZE)
    grid_support = compute_dense_tissue_mask_on_grid(
        grid_pts=grid_pts,
        union_spots=union_coords,
        grid_size=GRID_SIZE,
        max_dist_px=MAX_DIST_PX,
        closing_iters=1,
        keep_largest=True,
    )

    coords_cat, X_cat, point_weights = stack_weighted_evidence(
        coords_blocks=[coords_a_q, coords_b_q],
        X_blocks=[expr_a, expr_b],
        source_weights=source_weights,
    )

    return grid_pts, grid_support, coords_cat, X_cat, point_weights


def build_analysis_grid_and_evidence(
    coords_a,
    coords_b,
    expr_a,
    expr_b,
    source_weights,
    fixed_grid_pts,
    fixed_grid_support,
):
    """
    Analysis mode:
      - keep fixed aligned coordinates and fixed common grid
      - no query-specific shape tracking here
    """
    coords_cat, X_cat, point_weights = stack_weighted_evidence(
        coords_blocks=[coords_a, coords_b],
        X_blocks=[expr_a, expr_b],
        source_weights=source_weights,
    )
    return fixed_grid_pts, fixed_grid_support, coords_cat, X_cat, point_weights


def maybe_load_attrs():
    """
    Placeholder for future attribute generation.
    Keep straightforward for now.
    """
    if not USE_ATTRIBUTES:
        return None, None
    return  ATTRIBUTE_COLUMNS, ATTRIBUTE_SCHEMA


# =========================
# LOAD CONTEXT
# =========================
(
    axis_a, axis_b,
    coords_a, coords_b,
    expr_a_raw, expr_b_raw, genes,
    attrs_a_raw, attrs_b_raw,
    flow_a_to_b, flow_b_to_a,
    fixed_grid_pts, fixed_grid_support,
) = build_mode_context()

expr_a = maybe_log1p(expr_a_raw)
expr_b = maybe_log1p(expr_b_raw)

attrs, attr_schema = maybe_load_attrs()

source_axis_positions = np.array([axis_a, axis_b], dtype=np.float32)
support_kwargs = resolve_support_kwargs(MODE)

# optional: fit once and reuse for latent_kl
program_model = None
if GENE_MODE == "latent_kl":
    if sp.issparse(expr_a) or sp.issparse(expr_b):
        X_for_program = sp.vstack([expr_a, expr_b]).tocsr()
    else:
        X_for_program = np.vstack([np.asarray(expr_a), np.asarray(expr_b)]).astype(np.float32)

    program_model = fit_latent_gene_program_model(
        X=X_for_program,
        n_programs=N_PROGRAMS,
        random_state=0,
    )

# =========================
# GENERATE VIRTUAL TISSUES
# =========================
for query_axis_value in QUERY_AXIS_VALUES:
    query_axis_value = float(query_axis_value)

    alpha = relative_position_on_axis(
        query_axis_value=query_axis_value,
        axis_start_value=axis_a,
        axis_end_value=axis_b,
        clip=None,
    )

    print(f"[GEN] {BIOLOGICAL_AXIS_NAME}={query_axis_value:.3f}  alpha={alpha:.3f}")

    source_weights, used_axis_bandwidth = compute_axis_kernel_weights(
        observed_axis_positions=source_axis_positions,
        query_axis_position=query_axis_value,
        bandwidth=AXIS_WEIGHT_BANDWIDTH,
    )

    if MODE == "query":
        grid_pts, grid_support, coords_cat, X_cat, point_weights = build_query_grid_and_evidence(
            alpha=alpha,
            coords_a=coords_a,
            coords_b=coords_b,
            expr_a=expr_a,
            expr_b=expr_b,
            flow_a_to_b=flow_a_to_b,
            flow_b_to_a=flow_b_to_a,
            source_weights=source_weights,
        )
    elif MODE == "analysis":
        grid_pts, grid_support, coords_cat, X_cat, point_weights = build_analysis_grid_and_evidence(
            coords_a=coords_a,
            coords_b=coords_b,
            expr_a=expr_a,
            expr_b=expr_b,
            source_weights=source_weights,
            fixed_grid_pts=fixed_grid_pts,
            fixed_grid_support=fixed_grid_support,
        )
    else:
        raise ValueError("MODE must be one of {'query','analysis'}")

    if USE_ATTRIBUTES:
        attrs_cat = stack_attribute_evidence(attrs_a_raw, attrs_b_raw)
    else:
        attrs_cat = None

    ad_grid = generate_virtual_tissue_fixed_grid(
        coords=coords_cat,
        X=X_cat,
        grid_pts=grid_pts,
        var_names=genes,
        attrs=attrs_cat,
        attr_schema=None,  # let the function infer internally
        grid_support_mask=grid_support,
        point_weights=point_weights,
        spatial_bandwidth=SPATIAL_BANDWIDTH,
        knn=KNN,
        min_neighbors=MIN_NEIGHBORS,
        radius=RADIUS,
        chunk_size=64,
        fill_value=0.0,
        gene_mode=GENE_MODE,
        program_model=program_model,
        n_programs=N_PROGRAMS,
        **support_kwargs,
    )

    # metadata
    ad_grid.uns["biological_axis_name"] = BIOLOGICAL_AXIS_NAME
    ad_grid.uns["mode"] = MODE
    ad_grid.uns["query_axis_value"] = float(query_axis_value)
    ad_grid.uns["relative_position_alpha"] = float(alpha)
    ad_grid.uns["grid_size"] = int(GRID_SIZE)
    ad_grid.uns["apply_log1p_before_generation"] = bool(APPLY_LOG1P_BEFORE_GENERATION)
    ad_grid.uns["source_anchor_ids"] = [int(ANCHOR_A_ID), int(ANCHOR_B_ID)]
    ad_grid.uns["source_axis_positions"] = source_axis_positions.tolist()
    ad_grid.uns["source_weights"] = source_weights.tolist()
    ad_grid.uns["axis_weight_kernel"] = AXIS_WEIGHT_KERNEL
    ad_grid.uns["axis_weight_bandwidth"] = float(used_axis_bandwidth)

    axis_tag = str(BIOLOGICAL_AXIS_NAME).replace(" ", "_")
    out_path = os.path.join(
        OUT_DIR,
        f"virtual_{axis_tag}_{query_axis_value:.3f}.h5ad"
    )
    ad_grid.write_h5ad(out_path)
    print("  saved:", out_path)

print("DONE. Output folder:", OUT_DIR)