from virtual_utils import *
import os
import numpy as np
import scipy.sparse as sp

# =========================
# CONFIG
# =========================
BIOLOGICAL_AXIS_NAME = "protein"


AXIS_POSITIONS = {
    0: 'xenium',
    1: 'protein'
} # for cross-modality, we use categrical positions

# instead of one fixed global pair, provide all candidate anchors
ANCHOR_IDS = [0, 1]

# analysis mode uses one fixed reference slice/grid
ANALYSIS_REFERENCE_ANCHOR_ID = 0
MODALITY_REFERENCE_ANCHOR_ID = 1

QUERY_AXIS_VALUES = [0]
GENE_SUBSET=None


ROOT = "/media/huifang/data/registration/phenocycler/huifang"
NET_IN = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ad")

GROUP_ID = 'LUAD_2_A'
AXIS_OUT = os.path.join(ROOT, "pairwise_output", f"{GROUP_ID}")

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
USE_ATTRIBUTES = False
ATTRIBUTE_COLUMNS = None
ATTRIBUTE_SCHEMA = None

OUT_DIR = os.path.join(
    ROOT,
    "virtual_tissue",
    f"{BIOLOGICAL_AXIS_NAME}_{GROUP_ID}",
)
os.makedirs(OUT_DIR, exist_ok=True)



def maybe_log1p(X):
    return matrix_log1p(X) if APPLY_LOG1P_BEFORE_GENERATION else X


def resolve_support_kwargs():
    return dict(
        enforce_query_support=ANALYSIS_ENFORCE_QUERY_SUPPORT,
        max_support_dist=ANALYSIS_MAX_SUPPORT_DIST,
        min_confidence_abs=ANALYSIS_MIN_CONFIDENCE_ABS,
        min_confidence_ratio=ANALYSIS_MIN_CONFIDENCE_RATIO,
        confidence_ref=ANALYSIS_CONFIDENCE_REF,
    )

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
        expr_raw, genes_i, coords_i, attrs_i = load_h5ad_X_and_genes_and_coords(
            H5AD_DIR, GROUP_ID, AXIS_POSITIONS[aid],
            attr_columns=ATTRIBUTE_COLUMNS if USE_ATTRIBUTES else None
        )
        coords_i, expr_raw, attrs_i = align_coords_expr_attrs(coords_i, expr_raw, attrs_i)
        bank[aid] = {
            "axis_value": AXIS_POSITIONS[aid],
            "coords": coords_i,
            "expr_raw": expr_raw,
            "genes": np.asarray(genes_i),
            "attrs": attrs_i,
        }
    return bank


def build_fixed_analysis_grid(anchor_bank, reference_anchor_id):
    """
    In multi-modality analysis mode, use single cell locations as the reference_anchor_ids
    """
    coords_ref = np.asarray(anchor_bank[reference_anchor_id]["coords"], dtype=np.float32)
    grid_support = np.full(coords_ref.shape[0], True, dtype=bool)
    return coords_ref, grid_support




def build_analysis_grid_and_evidence(
    coords,
    expr,
    fixed_grid_pts,
    fixed_grid_support,
):
    coords_cat = coords
    X_cat=expr
    point_weights=np.full(X_cat.shape[0], 1, dtype=float)
    return fixed_grid_pts, fixed_grid_support, coords_cat, X_cat, point_weights


# =========================
# LOAD ALL ANCHORS ONCE
# =========================
anchor_bank = load_anchor_bank()
support_kwargs = resolve_support_kwargs()

fixed_grid_pts = None
fixed_grid_support = None

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
    anchor_a_id, anchor_b_id = ANALYSIS_REFERENCE_ANCHOR_ID,MODALITY_REFERENCE_ANCHOR_ID
    axis_a = AXIS_POSITIONS[anchor_a_id]
    axis_b = AXIS_POSITIONS[anchor_b_id]


    rec = anchor_bank[MODALITY_REFERENCE_ANCHOR_ID]
    expr_raw = rec["expr_raw"]
    genes = rec["genes"]
    attrs_raw = rec["attrs"]
    coords = rec["coords"]


    expr = maybe_log1p(expr_raw)

    # pair-specific latent model if needed
    program_model = None
    if GENE_MODE == "latent_kl":
        if sp.issparse(expr) :
            X_for_program = expr.tocsr()
        else:
            X_for_program = np.asarray(expr).astype(np.float32)

        program_model = fit_latent_gene_program_model(
            X=X_for_program,
            n_programs=N_PROGRAMS,
            random_state=0,
        )


    grid_pts, grid_support, coords_cat, X_cat, point_weights = build_analysis_grid_and_evidence(
        coords=coords,
        expr=expr,
        fixed_grid_pts=fixed_grid_pts,
        fixed_grid_support=fixed_grid_support,
    )


    if USE_ATTRIBUTES:
        attrs = attrs
    else:
        attrs = None

    ad_grid = generate_virtual_tissue_fixed_grid(
        coords=coords_cat,
        X=X_cat,
        grid_pts=grid_pts,
        var_names=genes,
        attrs=attrs,
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
    ad_grid.uns["apply_log1p_before_generation"] = bool(APPLY_LOG1P_BEFORE_GENERATION)
    ad_grid.uns["source_anchor_ids"] = [int(anchor_a_id), int(anchor_b_id)]
    ad_grid.uns["analysis_reference_anchor_id"] = int(ANALYSIS_REFERENCE_ANCHOR_ID)

    axis_tag = str(BIOLOGICAL_AXIS_NAME).replace(" ", "_")
    out_path = os.path.join(
        OUT_DIR,
        f"virtual_{axis_tag}_{query_axis_value:.3f}.h5ad"
    )
    ad_grid.write_h5ad(out_path)
    print("  saved:", out_path)

print("DONE. Output folder:", OUT_DIR)