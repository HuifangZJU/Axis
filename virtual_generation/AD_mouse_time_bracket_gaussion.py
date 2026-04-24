from virtual_utils import *
from axis.coordinate_utils import find_inverse_warp_coords
import os
import numpy as np
import scipy.sparse as sp

# =========================
# CONFIG
# =========================
BIOLOGICAL_AXIS_NAME = "Tg_time"
MODE = "analysis"   # "query" or "analysis"

AXIS_POSITIONS = {
    0: 2.5,
    1: 5.7,
    2: 17.9,
}

# instead of one fixed global pair, provide all candidate anchors
ANCHOR_IDS = [0, 1, 2]

# analysis mode uses one fixed reference slice/grid
ANALYSIS_REFERENCE_ANCHOR_ID = 0

GROUP_ID = 0
QUERY_AXIS_VALUES = np.arange(2.5, 20, 1.0).tolist()
# QUERY_AXIS_VALUES = [5.7,17.9]

ROOT = "/media/huifang/data/registration/xenium/mouse_brain"
NET_IN = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ads")
AXIS_OUT = os.path.join(ROOT, "pairwise_output", f"group_{GROUP_ID}")

GRID_SIZE = 256
MAX_DIST_PX = 5.0

# preprocessing / weighting
APPLY_LOG1P_BEFORE_GENERATION = False
AXIS_WEIGHT_BANDWIDTH = None
SPATIAL_BANDWIDTH = None
KNN = 48
MIN_NEIGHBORS = 6
RADIUS = None

# query-support policy
QUERY_MAX_SUPPORT_DIST = None
QUERY_MIN_CONFIDENCE_ABS = 1e-8
QUERY_MIN_CONFIDENCE_RATIO = 0.0
QUERY_CONFIDENCE_REF = "median"

ANALYSIS_ENFORCE_QUERY_SUPPORT = False
ANALYSIS_MAX_SUPPORT_DIST = None
ANALYSIS_MIN_CONFIDENCE_ABS = None
ANALYSIS_MIN_CONFIDENCE_RATIO = 0.0
ANALYSIS_CONFIDENCE_REF = "median"

# gene generation
GENE_MODE = "mean"      # "mean", "kl", "latent_kl"
N_PROGRAMS = 32

# optional attributes
USE_ATTRIBUTES = True
ATTRIBUTE_COLUMNS = ["axis_joint_domain"]
ATTRIBUTE_SCHEMA = {"axis_joint_domain": "categorical"}

# gene subset
GENE_SUBSET = ["Arc","Satb2","Pou3f1"]

OUT_DIR = os.path.join(
    ROOT,
    "virtual_tissue",
    f"{BIOLOGICAL_AXIS_NAME}_grid{GRID_SIZE}_refer0",
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


def align_coords_expr_attrs(coords, expr, attrs):
    n = min(coords.shape[0], expr.shape[0])
    if attrs is not None:
        n = min(n, len(attrs))
    coords = coords[:n]
    expr = expr[:n]
    if attrs is not None:
        attrs = attrs.iloc[:n].reset_index(drop=True)
    return coords, expr, attrs


def get_anchor_pair_for_query(query_axis_value, axis_positions, anchor_ids):
    """
    Interpolation:
      choose the bracket pair.
    Short-range extrapolation:
      choose the nearest two slices at the corresponding edge.
    """
    anchor_ids_sorted = sorted(anchor_ids, key=lambda k: float(axis_positions[k]))
    z_sorted = [float(axis_positions[k]) for k in anchor_ids_sorted]
    q = float(query_axis_value)

    # left extrapolation
    if q <= z_sorted[0]:
        return anchor_ids_sorted[0], anchor_ids_sorted[1]

    # right extrapolation
    if q >= z_sorted[-1]:
        return anchor_ids_sorted[-2], anchor_ids_sorted[-1]

    # bracket pair
    for i in range(len(anchor_ids_sorted) - 1):
        z0 = z_sorted[i]
        z1 = z_sorted[i + 1]
        if z0 <= q <= z1:
            return anchor_ids_sorted[i], anchor_ids_sorted[i + 1]

    raise RuntimeError("Failed to find anchor pair for query.")


def load_anchor_bank():
    """
    Load all anchors once. The selected pair is chosen later per query.
    """
    bank = {}
    for aid in ANCHOR_IDS:
        if MODE == "analysis":
            # expr_raw, genes_i, coords_i, attrs_i = load_h5ad_X_and_genes_and_coords(
            #     H5AD_DIR, "Tg", AXIS_POSITIONS[aid],
            #     attr_columns=ATTRIBUTE_COLUMNS if USE_ATTRIBUTES else None
            # )
            coords_i = load_coords_npz(NET_IN, GROUP_ID, aid)
            expr_raw, genes_i, attrs_i = load_h5ad_X_and_genes(
                H5AD_DIR, GROUP_ID, AXIS_POSITIONS[aid],
                attr_columns=ATTRIBUTE_COLUMNS if USE_ATTRIBUTES else None
            )
            if aid != ANALYSIS_REFERENCE_ANCHOR_ID:
                flow_i_to_reference = load_axis_flow(
                    AXIS_OUT, fixed_id=ANALYSIS_REFERENCE_ANCHOR_ID, moving_id=aid
                )
                coords_i = find_inverse_warp_coords(
                    coords_i,
                    (flow_i_to_reference).astype(np.float32)
                ).astype(np.float32)
        elif MODE == "query":
            coords_i = load_coords_npz(NET_IN, GROUP_ID, aid)
            expr_raw, genes_i, attrs_i = load_h5ad_X_and_genes(
                H5AD_DIR, "Tg", AXIS_POSITIONS[aid],
                attr_columns=ATTRIBUTE_COLUMNS if USE_ATTRIBUTES else None
            )
        else:
            raise ValueError("MODE must be one of {'query','analysis'}")

        coords_i, expr_raw, attrs_i = align_coords_expr_attrs(coords_i, expr_raw, attrs_i)

        bank[aid] = {
            "axis_value": float(AXIS_POSITIONS[aid]),
            "coords": coords_i,
            "expr_raw": expr_raw,
            "genes": np.asarray(genes_i),
            "attrs": attrs_i,
        }
    return bank


def build_fixed_analysis_grid(anchor_bank, reference_anchor_id):
    """
    In analysis mode, all coordinates are assumed already in the same aligned frame.
    Build one fixed grid/support from the chosen reference anchor only.
    """
    coords_ref = np.asarray(anchor_bank[reference_anchor_id]["coords"], dtype=np.float32)
    grid_pts, _ = make_common_grid_from_coords(coords_ref, GRID_SIZE)
    grid_support = compute_dense_tissue_mask_on_grid(
        grid_pts=grid_pts,
        union_spots=coords_ref,
        grid_size=GRID_SIZE,
        max_dist_px=MAX_DIST_PX,
        closing_iters=1,
        keep_largest=True,
    )
    return grid_pts, grid_support


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
    coords_cat, X_cat, point_weights = stack_weighted_evidence(
        coords_blocks=[coords_a, coords_b],
        X_blocks=[expr_a, expr_b],
        source_weights=source_weights,
    )
    return fixed_grid_pts, fixed_grid_support, coords_cat, X_cat, point_weights


# =========================
# LOAD ALL ANCHORS ONCE
# =========================
anchor_bank = load_anchor_bank()
support_kwargs = resolve_support_kwargs(MODE)

fixed_grid_pts = None
fixed_grid_support = None
if MODE == "analysis":
    fixed_grid_pts, fixed_grid_support = build_fixed_analysis_grid(
        anchor_bank=anchor_bank,
        reference_anchor_id=ANALYSIS_REFERENCE_ANCHOR_ID,
    )


# =========================
# GENERATE VIRTUAL TISSUES
# =========================
for query_axis_value in QUERY_AXIS_VALUES:
    query_axis_value = float(query_axis_value)

    # choose dynamic two-anchor pair for this query
    anchor_a_id, anchor_b_id = get_anchor_pair_for_query(
        query_axis_value=query_axis_value,
        axis_positions=AXIS_POSITIONS,
        anchor_ids=ANCHOR_IDS,
    )

    axis_a = float(AXIS_POSITIONS[anchor_a_id])
    axis_b = float(AXIS_POSITIONS[anchor_b_id])

    rec_a = anchor_bank[anchor_a_id]
    rec_b = anchor_bank[anchor_b_id]

    expr_a_raw = rec_a["expr_raw"]
    expr_b_raw = rec_b["expr_raw"]
    genes_a = rec_a["genes"]
    genes_b = rec_b["genes"]
    attrs_a_raw = rec_a["attrs"]
    attrs_b_raw = rec_b["attrs"]
    coords_a = rec_a["coords"]
    coords_b = rec_b["coords"]

    # harmonize genes only for the selected pair
    expr_a_raw_sel, expr_b_raw_sel, genes = ensure_same_genes_keep_order(
        expr_a_raw, genes_a, expr_b_raw, genes_b
    )
    expr_a_raw_sel, expr_b_raw_sel, genes = maybe_subset_genes(
        expr_a_raw_sel, expr_b_raw_sel, genes, GENE_SUBSET
    )

    expr_a = maybe_log1p(expr_a_raw_sel)
    expr_b = maybe_log1p(expr_b_raw_sel)

    source_axis_positions = np.array([axis_a, axis_b], dtype=np.float32)

    alpha = relative_position_on_axis(
        query_axis_value=query_axis_value,
        axis_start_value=axis_a,
        axis_end_value=axis_b,
        clip=None,
    )

    print(
        f"[GEN] {BIOLOGICAL_AXIS_NAME}={query_axis_value:.3f}  "
        f"anchors=({anchor_a_id},{anchor_b_id})  alpha={alpha:.3f}"
    )

    source_weights, used_axis_bandwidth = compute_axis_kernel_weights(
        observed_axis_positions=source_axis_positions,
        query_axis_position=query_axis_value,
        bandwidth=AXIS_WEIGHT_BANDWIDTH,
    )



    # pair-specific latent model if needed
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

    if MODE == "query":
        flow_a_to_b = load_axis_flow(
            AXIS_OUT, fixed_id=anchor_b_id, moving_id=anchor_a_id
        )
        flow_b_to_a = load_axis_flow(
            AXIS_OUT, fixed_id=anchor_a_id, moving_id=anchor_b_id
        )

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
        attr_schema=ATTRIBUTE_SCHEMA,
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
    ad_grid.uns["source_anchor_ids"] = [int(anchor_a_id), int(anchor_b_id)]
    ad_grid.uns["source_axis_positions"] = source_axis_positions.tolist()
    ad_grid.uns["source_weights"] = source_weights.tolist()
    ad_grid.uns["axis_weight_bandwidth"] = float(used_axis_bandwidth)

    if MODE == "analysis":
        ad_grid.uns["analysis_reference_anchor_id"] = int(ANALYSIS_REFERENCE_ANCHOR_ID)

    axis_tag = str(BIOLOGICAL_AXIS_NAME).replace(" ", "_")
    out_path = os.path.join(
        OUT_DIR,
        f"virtual_{axis_tag}_{query_axis_value:.3f}.h5ad"
    )
    ad_grid.write_h5ad(out_path)
    print("  saved:", out_path)

print("DONE. Output folder:", OUT_DIR)