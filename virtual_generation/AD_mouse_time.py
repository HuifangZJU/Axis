
from virtual_utils import *
from axis.coordinate_utils import find_inverse_warp_coords

# =========================
# CONFIG
# =========================


BIOLOGICAL_AXIS_NAME = "Tg_time"   # e.g. "depth_um", "time_day", "disease_score"
# MODE = 'analysis'
MODE = 'query'
AXIS_POSITIONS = {
    0: 2.5,
    1: 5.7,
    2: 17.9,
}
GROUP_ID = 0
ANCHOR_A_ID = 0
ANCHOR_B_ID = 1
QUERY_AXIS_VALUES= np.arange(2, 5.7, 1.0).tolist()
# ANCHOR_A_ID = 1
# ANCHOR_B_ID = 2
# QUERY_AXIS_VALUES= np.arange(5.7, 19, 1.0).tolist()
# QUERY_AXIS_VALUES = [17.9]          # works for interpolation and extrapolation

ROOT = "/media/huifang/data/registration/xenium/mouse_brain"
NET_IN = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ads/Tg")
AXIS_OUT = os.path.join(ROOT, "pairwise_output", f"group_{GROUP_ID}")

GRID_SIZE = 256
MAX_DIST_PX = 5.0

# Virtual slice generator settings
APPLY_LOG1P_BEFORE_REGRESSION = False
AXIS_WEIGHT_BANDWIDTH = None       # None -> inferred from axis spacing
SPATIAL_BANDWIDTH = None            # None -> inferred from grid spacing
KNN = 48
MIN_NEIGHBORS = 6
RADIUS = None

MAX_SUPPORT_DIST = None          # None -> 1.5 * spatial_bandwidth
MIN_CONFIDENCE_ABS = None
MIN_CONFIDENCE_RATIO = 0.05
CONFIDENCE_REF = "median"

OUT_DIR = os.path.join(ROOT, "virtual_tissue", f"{BIOLOGICAL_AXIS_NAME}_grid{GRID_SIZE}",f"{MODE}")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD ANCHOR DATA
# =========================
axis_a = float(AXIS_POSITIONS[ANCHOR_A_ID])
axis_b = float(AXIS_POSITIONS[ANCHOR_B_ID])




if MODE == 'analysis':
    expr_a_raw, genes_a, coords_a= load_h5ad_X_and_genes_and_coords(H5AD_DIR, "Tg", axis_a)
    expr_b_raw, genes_b, coords_b= load_h5ad_X_and_genes_and_coords(H5AD_DIR, "Tg", axis_b)
elif MODE =='query':
    coords_a = load_coords_npz(NET_IN, GROUP_ID, ANCHOR_A_ID)
    coords_b = load_coords_npz(NET_IN, GROUP_ID, ANCHOR_B_ID)
    # Convention: fixed_id, moving_id => moving -> fixed
    flow_a_to_b = load_axis_flow(AXIS_OUT, fixed_id=ANCHOR_B_ID, moving_id=ANCHOR_A_ID)  # A -> B
    flow_b_to_a = load_axis_flow(AXIS_OUT, fixed_id=ANCHOR_A_ID, moving_id=ANCHOR_B_ID)  # B -> A

    expr_a_raw, genes_a = load_h5ad_X_and_genes(H5AD_DIR, "Tg", axis_a)
    expr_b_raw, genes_b = load_h5ad_X_and_genes(H5AD_DIR, "Tg", axis_b)
else:
    raise ValueError("Generation mode must be one of {'analysis','query'}")

expr_a_raw, expr_b_raw, genes = ensure_same_genes_keep_order(expr_a_raw, genes_a, expr_b_raw, genes_b)


# =========================
# subset on selected genes
# =========================
GENE_SUBSET = ["Arc"]
gene_to_idx = {g: i for i, g in enumerate(genes.tolist())}
keep_idx = [gene_to_idx[g] for g in GENE_SUBSET if g in gene_to_idx]
if len(keep_idx) == 0:
    raise ValueError("None of the requested genes found in common gene list.")
genes = genes[keep_idx]
expr_a_raw = expr_a_raw[:, keep_idx]
expr_b_raw = expr_b_raw[:, keep_idx]


# =========================
# BUILD ONE COMMON GRID for analysis mode
# Reference frame = anchor A frame
# =========================
if MODE =='analysis':
    union_coords = np.vstack([coords_a, coords_b]).astype(np.float32)
    grid_pts, (x_edges, y_edges) = make_common_grid_from_coords(union_coords, GRID_SIZE)
    grid_support = compute_dense_tissue_mask_on_grid(
        grid_pts=grid_pts,
        union_spots=union_coords,
        grid_size=GRID_SIZE,
        max_dist_px=MAX_DIST_PX,
        closing_iters=1,
        keep_largest=True,
    )

# Optional preprocessing once per source
expr_a = matrix_log1p(expr_a_raw) if APPLY_LOG1P_BEFORE_REGRESSION else expr_a_raw
expr_b = matrix_log1p(expr_b_raw) if APPLY_LOG1P_BEFORE_REGRESSION else expr_b_raw
#

# =========================
# GENERATE VIRTUAL TISSUES
# =========================
source_axis_positions = np.array([axis_a, axis_b], dtype=np.float32)

for query_axis_value in QUERY_AXIS_VALUES:
    query_axis_value = float(query_axis_value)

    alpha = relative_position_on_axis(
        query_axis_value=query_axis_value,
        axis_start_value=axis_a,
        axis_end_value=axis_b,
        clip=None,   # keep None to allow extrapolation
    )
    print(f"[GEN] {BIOLOGICAL_AXIS_NAME}={query_axis_value:.3f}  alpha={alpha:.3f}")
    # Axis-aware source weights (same formula for interpolation/extrapolation)
    source_weights, used_axis_bandwidth = compute_axis_kernel_weights(
        observed_axis_positions=source_axis_positions,
        query_axis_position=query_axis_value,
        bandwidth=AXIS_WEIGHT_BANDWIDTH,
    )

    if MODE=='query':
        coords_a_q = find_inverse_warp_coords(
            coords_a,
            (alpha * flow_a_to_b).astype(np.float32)
        ).astype(np.float32)

        coords_b_q = find_inverse_warp_coords(
            coords_b,
            ((1.0 - alpha) * flow_b_to_a).astype(np.float32)
        ).astype(np.float32)

        union_coords = np.vstack([coords_a_q, coords_b_q]).astype(np.float32)
        grid_pts, (x_edges, y_edges) = make_common_grid_from_coords(union_coords, GRID_SIZE)
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
    elif MODE=='analysis':
        coords_cat, X_cat, point_weights = stack_weighted_evidence(
            coords_blocks=[coords_a, coords_b],
            X_blocks=[expr_a, expr_b],
            source_weights=source_weights,
        )
    else:
        raise ValueError("Generation mode must be one of {'analysis','query'}")


    ad_grid = spatial_resample_evidence_regression_fixed_grid(
        coords=coords_cat,
        X=X_cat,
        grid_pts=grid_pts,
        grid_support_mask=grid_support,
        point_weights=point_weights,
        spatial_bandwidth=SPATIAL_BANDWIDTH,
        knn=KNN,
        min_neighbors=MIN_NEIGHBORS,
        radius=RADIUS,
        chunk_size=256,
        fill_value=0.0,
        enforce_query_support=False,
        max_support_dist=MAX_SUPPORT_DIST,
        min_confidence_abs=MIN_CONFIDENCE_ABS,
        min_confidence_ratio=MIN_CONFIDENCE_RATIO,
        confidence_ref=CONFIDENCE_REF,
    )

    # metadata
    ad_grid.var_names = genes
    ad_grid.uns["biological_axis_name"] = BIOLOGICAL_AXIS_NAME
    ad_grid.uns["query_axis_value"] = float(query_axis_value)
    ad_grid.uns["relative_position_alpha"] = float(alpha)
    ad_grid.uns["grid_size"] = int(GRID_SIZE)
    ad_grid.uns["apply_log1p_before_regression"] = bool(APPLY_LOG1P_BEFORE_REGRESSION)
    ad_grid.uns["source_anchor_ids"] = [int(ANCHOR_A_ID), int(ANCHOR_B_ID)]
    ad_grid.uns["source_axis_positions"] = source_axis_positions.tolist()
    ad_grid.uns["source_weights"] = source_weights.tolist()
    ad_grid.uns["axis_weight_bandwidth"] = float(used_axis_bandwidth)

    axis_tag = str(BIOLOGICAL_AXIS_NAME).replace(" ", "_")
    out_path = os.path.join(
        OUT_DIR,
        f"virtual_{axis_tag}_{query_axis_value:.3f}.h5ad"
    )
    ad_grid.write_h5ad(out_path)
    print("  saved:", out_path)

print("DONE. Output folder:", OUT_DIR)