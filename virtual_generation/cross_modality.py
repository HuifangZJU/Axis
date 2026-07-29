from virtual_utils import *
from axis.coordinate_utils import find_inverse_warp_coords
import os
import numpy as np
import scipy.sparse as sp




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
        # coords_i = load_coords_npz(NET_IN, GROUP_ID, AXIS_POSITIONS[aid])
        # expr_raw, genes_i, attrs_i = load_h5ad_X_and_genes(
        #     H5AD_DIR, GROUP_ID, AXIS_POSITIONS[aid],
        #     attr_columns=ATTRIBUTE_COLUMNS if USE_ATTRIBUTES else None
        # )
        expr_raw, genes_i, attrs_i,coords_i = load_h5ad_X_and_genes_and_coords(
            H5AD_DIR, GROUP_ID, AXIS_POSITIONS[aid],
            attr_columns=ATTRIBUTE_COLUMNS if USE_ATTRIBUTES else None
        )
        print(expr_raw.shape)
        test = input()

        coords_i, expr_raw, attrs_i = align_coords_expr_attrs(coords_i, expr_raw, attrs_i)


        bank[aid] = {
            "axis_value": AXIS_POSITIONS[aid],
            "coords": coords_i,
            "expr_raw": expr_raw,
            "genes": np.asarray(genes_i),
            "attrs": attrs_i,
        }
    return bank

def build_analysis_evidence(
    coords,
    grid_pts,
    expr,
    flow_to_f
):
    coords_to_f = find_inverse_warp_coords(
        coords,
        (flow_to_f).astype(np.float32)
    ).astype(np.float32)

    grid_support = compute_dense_tissue_mask_on_any_grid(
        grid_pts=grid_pts,
        union_spots=coords_to_f,
        max_dist_px=MAX_DIST_PX,
        keep_largest=True,
    )
    point_weights = np.full(expr.shape[0], 1, dtype=float)

    return grid_support, coords_to_f, point_weights


def run_analysis():
    refer = anchor_bank[ANALYSIS_REFERENCE_ANCHOR_ID]
    grid_pts = refer["coords"]
    for anchor_id in ANCHOR_IDS:
        if anchor_id == ANALYSIS_REFERENCE_ANCHOR_ID:
            continue
        axis_value = AXIS_POSITIONS[anchor_id]
        rec = anchor_bank[anchor_id]

        expr_raw = rec["expr_raw"]
        genes = rec["genes"]
        attrs_raw = rec["attrs"]
        coords = rec["coords"]

        expr = maybe_log1p(expr_raw)

        print(
            f"[GEN] {BIOLOGICAL_AXIS_NAME}={axis_value}"
            f"data anchors=({anchor_id}), location anchors = ({ANALYSIS_REFERENCE_ANCHOR_ID})"
        )

        flow_to_f = load_axis_flow(
            AXIS_OUT, fixed_id=AXIS_POSITIONS[ANALYSIS_REFERENCE_ANCHOR_ID], moving_id=AXIS_POSITIONS[anchor_id]
        )

        grid_support, coords_cat,point_weights = build_analysis_evidence(
            coords=coords,
            grid_pts=grid_pts,
            expr=expr,
            flow_to_f=flow_to_f
        )

        if USE_ATTRIBUTES:
            attrs_cat = attrs_raw
        else:
            attrs_cat = None

        ad_grid = generate_virtual_tissue_fixed_grid(
            coords=coords_cat,
            X=expr,
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
        )

        # metadata
        ad_grid.uns["biological_axis_name"] = BIOLOGICAL_AXIS_NAME
        ad_grid.uns["mode"] = MODE
        ad_grid.uns["apply_log1p_before_generation"] = bool(APPLY_LOG1P_BEFORE_GENERATION)
        ad_grid.uns["analysis_reference_anchor_id"] = int(ANALYSIS_REFERENCE_ANCHOR_ID)

        axis_tag = str(BIOLOGICAL_AXIS_NAME).replace(" ", "_")
        out_path = os.path.join(
            OUT_DIR,
            f"virtual_{axis_tag}_{axis_value}.h5ad"
        )
        ad_grid.write_h5ad(out_path)
        print("  saved:", out_path)

    print("DONE. Output folder:", OUT_DIR)




# =========================
# CONFIG
# =========================
BIOLOGICAL_AXIS_NAME = "modality"


AXIS_POSITIONS = {
    0: 'xenium',
    1: 'protein'
}

# analysis mode uses one fixed reference slice/grid
ANALYSIS_REFERENCE_ANCHOR_ID = 0

ROOT = "/media/huifang/data/registration/phenocycler/huifang"
NET_IN = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ad")

MAX_DIST_PX = 5.0

# preprocessing / weighting
APPLY_LOG1P_BEFORE_GENERATION = False
# for query, to sue neighboring information
AXIS_WEIGHT_BANDWIDTH = None
SPATIAL_BANDWIDTH = None


KNN = 48
MIN_NEIGHBORS = 6
RADIUS = None


# gene generation
GENE_MODE = "mean"

# optional attributes
USE_ATTRIBUTES = False
ATTRIBUTE_COLUMNS = None
ATTRIBUTE_SCHEMA = None
GENE_SUBSET=None


for GROUP_ID in ['LUAD_2_A']:
    print(f"group id {GROUP_ID}")
    AXIS_OUT = os.path.join(ROOT, "pairwise_output", f"{GROUP_ID}")
    for MODE in ["analysis"]:
        OUT_DIR = os.path.join(
            ROOT,
            "virtual_tissue",
            f"{BIOLOGICAL_AXIS_NAME}_{GROUP_ID}",
            MODE,
        )
        os.makedirs(OUT_DIR, exist_ok=True)
        # =========================
        # LOAD ALL ANCHORS ONCE
        # =========================
        if MODE == "analysis":
            ANCHOR_IDS = [0, 1]
            anchor_bank = load_anchor_bank()
            run_analysis()
