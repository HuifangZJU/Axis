from axis.config import load_config
from axis.registration import find_inverse_warp_coords
from axis.generation import (
    compute_dense_tissue_mask_on_any_grid,
    generate_virtual_tissue_fixed_grid,
    make_common_grid_from_coords,
    matrix_log1p,
    save_generation_meta,
    stack_weighted_evidence,
)
from axis.io import load_axis_flow, load_coords_npz, load_h5ad_X_and_genes
import os
import numpy as np
import scipy.sparse as sp
import pandas as pd




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


def load_anchor_bank():
    """
    Load all anchors once. The selected pair is chosen later per query.
    """
    bank = {}
    for aid in ANCHOR_IDS:
        coords_i = load_coords_npz(NET_IN, GROUP_ID, aid)
        expr_raw, genes_i, attrs_i = load_h5ad_X_and_genes(
            H5AD_DIR, GROUP_ID, aid,
            attr_columns=ATTRIBUTE_COLUMNS if USE_ATTRIBUTES else None
        )
        coords_i, expr_raw, attrs_i = align_coords_expr_attrs(coords_i, expr_raw, attrs_i)

        bank[aid] = {
            "axis_value": float(AXIS_POSITIONS[aid]),
            "coords": coords_i,
            "expr_raw": expr_raw,
            "genes": np.asarray(genes_i),
            "attrs": attrs_i,
        }
    return bank


def _validate_anchor_ids_for_integration(anchor_ids, reference_anchor_id):
    """Validate and deterministically order integration anchors."""
    anchor_ids = list(anchor_ids)
    if len(anchor_ids) == 0:
        raise ValueError("Integration requires at least one anchor.")
    if len(anchor_ids) != len(set(anchor_ids)):
        raise ValueError("ANCHOR_IDS contains duplicate anchor IDs.")
    if reference_anchor_id not in anchor_ids:
        raise ValueError(
            "ANALYSIS_REFERENCE_ANCHOR_ID must be included in ANCHOR_IDS "
            "for integration."
        )

    missing_bank = [aid for aid in anchor_ids if aid not in anchor_bank]
    if missing_bank:
        raise KeyError(
            "The following integration anchors are missing from anchor_bank: "
            + ", ".join(map(str, missing_bank))
        )

    missing_axis = [aid for aid in anchor_ids if aid not in AXIS_POSITIONS]
    if missing_axis:
        raise KeyError(
            "The following integration anchors are missing from AXIS_POSITIONS: "
            + ", ".join(map(str, missing_axis))
        )

    # Deterministic biological-axis ordering. This ordering is used consistently
    # for coordinates, expression, attributes, weights, and metadata.
    return sorted(anchor_ids, key=lambda aid: float(AXIS_POSITIONS[aid]))


def _check_unique_gene_names(genes, anchor_id):
    genes = np.asarray(genes).astype(str)
    if len(np.unique(genes)) != len(genes):
        duplicated = pd.Series(genes)[pd.Series(genes).duplicated()].unique()
        raise ValueError(
            f"Anchor {anchor_id} contains duplicated gene names, for example: "
            + ", ".join(map(str, duplicated[:10]))
        )
    return genes


def harmonize_all_anchor_genes(anchor_ids, gene_subset=None):
    """
    Harmonize all integration anchors to one common gene order.

    The first anchor in biological-axis order defines the order of the common
    genes. When ``gene_subset`` is supplied, its requested order is retained.
    No expression is pooled at this stage; every anchor remains an independent
    evidence block.
    """
    anchor_ids = list(anchor_ids)
    first_id = anchor_ids[0]
    first_genes = _check_unique_gene_names(
        anchor_bank[first_id]["genes"],
        first_id,
    )

    common_gene_set = set(first_genes.tolist())
    gene_arrays = {first_id: first_genes}

    for aid in anchor_ids[1:]:
        genes_i = _check_unique_gene_names(
            anchor_bank[aid]["genes"],
            aid,
        )
        gene_arrays[aid] = genes_i
        common_gene_set.intersection_update(genes_i.tolist())

    common_genes = np.asarray(
        [gene for gene in first_genes if gene in common_gene_set],
        dtype=object,
    )

    if gene_subset is not None:
        common_set = set(common_genes.tolist())
        requested = [str(gene) for gene in gene_subset]
        common_genes = np.asarray(
            [gene for gene in requested if gene in common_set],
            dtype=object,
        )

    if len(common_genes) == 0:
        raise ValueError(
            "No genes remain after intersecting all integration anchors "
            "and applying GENE_SUBSET."
        )

    expression_blocks = []
    for aid in anchor_ids:
        genes_i = gene_arrays.get(aid)
        if genes_i is None:
            genes_i = _check_unique_gene_names(
                anchor_bank[aid]["genes"],
                aid,
            )

        index_lookup = {gene: idx for idx, gene in enumerate(genes_i.tolist())}
        indices = np.asarray(
            [index_lookup[gene] for gene in common_genes],
            dtype=np.int64,
        )

        expr_i = anchor_bank[aid]["expr_raw"][:, indices]
        expr_i = maybe_log1p(expr_i)
        expression_blocks.append(expr_i)

    return expression_blocks, common_genes


def transfer_anchor_to_reference(anchor_id, reference_anchor_id):
    """
    Transform one anchor's coordinates into the fixed reference frame.

    The reference anchor is already in the target frame and therefore uses an
    exact identity transformation. Every other anchor is transferred using the
    same full inverse warp used by analysis mode.
    """
    coords = np.asarray(
        anchor_bank[anchor_id]["coords"],
        dtype=np.float32,
    )

    if anchor_id == reference_anchor_id:
        return coords.copy()

    flow_to_reference = load_axis_flow(
        AXIS_OUT,
        fixed_id=reference_anchor_id,
        moving_id=anchor_id,
    )

    return find_inverse_warp_coords(
        coords,
        flow_to_reference.astype(np.float32),
    ).astype(np.float32)


def concatenate_attribute_blocks(attribute_blocks):
    """Concatenate attributes in exactly the same order as evidence blocks."""
    if not USE_ATTRIBUTES:
        return None

    if len(attribute_blocks) == 0:
        raise ValueError(
            "USE_ATTRIBUTES=True, but no attribute blocks were collected."
        )

    missing = [
        i for i, attrs in enumerate(attribute_blocks)
        if attrs is None
    ]
    if missing:
        raise ValueError(
            "USE_ATTRIBUTES=True, but attributes are missing for source "
            f"blocks {missing}."
        )

    attrs_cat = pd.concat(
        [
            attrs.loc[:, ATTRIBUTE_COLUMNS].reset_index(drop=True)
            for attrs in attribute_blocks
        ],
        axis=0,
        ignore_index=True,
    )
    return attrs_cat


def build_integration_grid_and_evidence(
    fixed_grid_pts,
    anchor_ids,
    reference_anchor_id,
):
    """
    Collect all anchors in one reference frame and build one evidence bank.

    Integration differs from the other modes as follows:

    * query mode uses two local anchors, partially transfers both toward a
      requested axis position, and uses query-dependent axis-kernel weights;
    * analysis mode fully transfers every anchor to one fixed frame but
      generates each anchor independently;
    * integration mode fully transfers every anchor to the same fixed frame,
      harmonizes genes across all anchors, concatenates every evidence block,
      assigns the same biological source weight to every anchor, and returns
      one joint evidence bank for one generation call.

    Equal biological weighting follows the same source-weight semantics as
    ``build_query_evidence``: every anchor receives source weight 1 / N, with no
    preference based on axis position or reference-anchor identity.
    """
    anchor_ids = _validate_anchor_ids_for_integration(
        anchor_ids=anchor_ids,
        reference_anchor_id=reference_anchor_id,
    )

    expression_blocks, genes = harmonize_all_anchor_genes(
        anchor_ids=anchor_ids,
        gene_subset=GENE_SUBSET,
    )

    coordinate_blocks = []
    attribute_blocks = []
    source_point_counts = []

    for aid, expr_i in zip(anchor_ids, expression_blocks):
        coords_i_reference = transfer_anchor_to_reference(
            anchor_id=aid,
            reference_anchor_id=reference_anchor_id,
        )

        attrs_i = anchor_bank[aid]["attrs"] if USE_ATTRIBUTES else None

        # The anchor bank was aligned at loading time. Revalidate here after
        # gene subsetting so all blocks share the exact same row count.
        coords_i_reference, expr_i, attrs_i = align_coords_expr_attrs(
            coords_i_reference,
            expr_i,
            attrs_i,
        )

        coordinate_blocks.append(coords_i_reference)
        expression_blocks[anchor_ids.index(aid)] = expr_i
        attribute_blocks.append(attrs_i)
        source_point_counts.append(int(coords_i_reference.shape[0]))

        print(
            f"  [INTEGRATE] anchor={aid} "
            f"axis={float(AXIS_POSITIONS[aid]):.3f} "
            f"points={coords_i_reference.shape[0]:,}"
        )

    n_sources = len(anchor_ids)
    source_weights = np.full(
        n_sources,
        1.0 / n_sources,
        dtype=np.float32,
    )

    # Reuse the same evidence-stacking mechanism as query mode. Concatenation
    # order is anchor-by-anchor and is preserved for coordinates, expression,
    # attributes, and source-point metadata.
    coords_cat, X_cat, point_weights = stack_weighted_evidence(
        coords_blocks=coordinate_blocks,
        X_blocks=expression_blocks,
        source_weights=source_weights,
    )

    attrs_cat = concatenate_attribute_blocks(attribute_blocks)

    expected_rows = int(sum(source_point_counts))
    if coords_cat.shape[0] != expected_rows:
        raise RuntimeError(
            f"Coordinate concatenation produced {coords_cat.shape[0]} rows; "
            f"expected {expected_rows}."
        )
    if X_cat.shape[0] != expected_rows:
        raise RuntimeError(
            f"Expression concatenation produced {X_cat.shape[0]} rows; "
            f"expected {expected_rows}."
        )
    if len(point_weights) != expected_rows:
        raise RuntimeError(
            f"Point-weight concatenation produced {len(point_weights)} rows; "
            f"expected {expected_rows}."
        )
    if attrs_cat is not None and len(attrs_cat) != expected_rows:
        raise RuntimeError(
            f"Attribute concatenation produced {len(attrs_cat)} rows; "
            f"expected {expected_rows}."
        )

    union_coords = np.vstack(coordinate_blocks).astype(np.float32)
    grid_support = compute_dense_tissue_mask_on_any_grid(
        grid_pts=fixed_grid_pts,
        union_spots=union_coords,
        max_dist_px=MAX_DIST_PX,
        keep_largest=True,
    )

    source_anchor_per_point = np.concatenate(
        [
            np.full(count, aid, dtype=np.int32)
            for aid, count in zip(anchor_ids, source_point_counts)
        ]
    )

    return {
        "grid_pts": fixed_grid_pts,
        "grid_support": grid_support,
        "coords_cat": coords_cat,
        "X_cat": X_cat,
        "genes": genes,
        "attrs_cat": attrs_cat,
        "point_weights": point_weights,
        "source_anchor_ids": anchor_ids,
        "source_anchor_per_point": source_anchor_per_point,
        "source_axis_positions": np.asarray(
            [AXIS_POSITIONS[aid] for aid in anchor_ids],
            dtype=np.float32,
        ),
        "source_weights": source_weights,
        "source_point_counts": source_point_counts,
    }


def run_integration(save_meta=True):
    """
    Generate one equal-weight integrated slice in the reference coordinate frame.

    There is intentionally no loop over anchors and no query-axis loop. All
    aligned evidence is sent to ``generate_virtual_tissue_fixed_grid`` once.
    """
    grid_pts, _ = make_common_grid_from_coords(
        GRID_SIZE,
        type="pixel",
    )

    evidence = build_integration_grid_and_evidence(
        fixed_grid_pts=grid_pts,
        anchor_ids=ANCHOR_IDS,
        reference_anchor_id=ANALYSIS_REFERENCE_ANCHOR_ID,
    )

    print(
        f"[GEN-INTEGRATION] sources={evidence['source_anchor_ids']}  "
        f"reference={ANALYSIS_REFERENCE_ANCHOR_ID}  "
        f"points={evidence['coords_cat'].shape[0]:,}  "
        f"genes={len(evidence['genes']):,}"
    )
    print(
        "  equal biological source weights:",
        evidence["source_weights"].tolist(),
    )

    ad_grid = generate_virtual_tissue_fixed_grid(
        coords=evidence["coords_cat"],
        X=evidence["X_cat"],
        grid_pts=evidence["grid_pts"],
        var_names=evidence["genes"],
        attrs=evidence["attrs_cat"],
        attr_schema=ATTRIBUTE_SCHEMA,
        grid_support_mask=evidence["grid_support"],
        point_weights=evidence["point_weights"],
        spatial_bandwidth=SPATIAL_BANDWIDTH,
        attr_spatial_bandwidth=ATTR_SPATIAL_BANDWIDTH,
        knn=KNN,
        min_neighbors=MIN_NEIGHBORS,
        radius=RADIUS,
        chunk_size=64,
        fill_value=0.0,
    )

    # Integration metadata. The reference anchor defines only the coordinate
    # frame; it does not receive greater biological weight.
    ad_grid.uns["biological_axis_name"] = BIOLOGICAL_AXIS_NAME
    ad_grid.uns["mode"] = "integration"
    ad_grid.uns["grid_size"] = int(GRID_SIZE)
    ad_grid.uns["apply_log1p_before_generation"] = bool(
        APPLY_LOG1P_BEFORE_GENERATION
    )
    ad_grid.uns["integration_reference_anchor_id"] = int(
        ANALYSIS_REFERENCE_ANCHOR_ID
    )
    ad_grid.uns["integration_reference_axis_value"] = float(
        AXIS_POSITIONS[ANALYSIS_REFERENCE_ANCHOR_ID]
    )
    ad_grid.uns["integration_source_anchor_ids"] = [
        int(x) for x in evidence["source_anchor_ids"]
    ]
    ad_grid.uns["integration_source_axis_positions"] = (
        evidence["source_axis_positions"].astype(float).tolist()
    )
    ad_grid.uns["integration_source_weights"] = (
        evidence["source_weights"].astype(float).tolist()
    )
    ad_grid.uns["integration_source_point_counts"] = [
        int(x) for x in evidence["source_point_counts"]
    ]
    ad_grid.uns["integration_equal_biological_weight"] = True
    ad_grid.uns["integration_weight_definition"] = (
        "uniform source coefficient 1/N assigned through "
        "stack_weighted_evidence"
    )
    ad_grid.uns["integration_n_evidence_points"] = int(
        evidence["coords_cat"].shape[0]
    )
    ad_grid.uns["integration_n_genes"] = int(
        len(evidence["genes"])
    )
    ad_grid.uns["integration_coordinate_frame"] = (
        f"anchor_{ANALYSIS_REFERENCE_ANCHOR_ID}"
    )

    axis_tag = str(BIOLOGICAL_AXIS_NAME).replace(" ", "_")
    out_path = os.path.join(
        OUT_DIR,
        (
            f"integrated_{axis_tag}_"
            f"ref_{ANALYSIS_REFERENCE_ANCHOR_ID}.h5ad"
        ),
    )
    ad_grid.write_h5ad(out_path)
    print("  saved:", out_path)
    print("DONE. Output folder:", OUT_DIR)

    if save_meta:
        meta_path = save_generation_meta(
            virtual_tissue_path=out_path,

            # User-defined spatial support
            grid_pts=grid_pts,
            grid_support_mask=evidence["grid_support"],

            # Collected spatial evidence
            coords_cat=evidence["coords_cat"],
            X_cat=evidence["X_cat"],
            genes=evidence["genes"],
            attrs_cat=evidence["attrs_cat"],
            attr_schema=ATTRIBUTE_SCHEMA,
            # Preserve anchor identity for each evidence point.
            # This assumes build_query_evidence concatenates anchor A
            # followed by anchor B without deleting or duplicating rows.
            source_anchor_ids=ANCHOR_IDS,
            source_point_counts=evidence["source_point_counts"]
        )

        print("  saved generation metadata:", meta_path)
    return ad_grid, evidence


# =========================
# CONFIG
# =========================
BIOLOGICAL_AXIS_NAME = "dlpfc_z"


AXIS_POSITIONS = {
    0: 0,
    1: 10,
    2: 310,
    3: 320,
}


# instead of one fixed global pair, provide all candidate anchors



# Analysis and integration use one fixed reference coordinate frame.
# In integration mode, this anchor defines geometry only; all anchors
# receive the same biological source weight.
ANALYSIS_REFERENCE_ANCHOR_ID = 0

ROOT = load_config('dlpfc').root
NET_IN = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ad")


GRID_SIZE = 150
MAX_DIST_PX = 5.0


# preprocessing / weighting
APPLY_LOG1P_BEFORE_GENERATION = False
AXIS_WEIGHT_BANDWIDTH = None
SPATIAL_BANDWIDTH = 0.3
# plot_axis_kernel_weight_debug([0,320],10,bandwidth=AXIS_WEIGHT_BANDWIDTH)

KNN = 48
MIN_NEIGHBORS = 1
RADIUS = None

# gene generation
GENE_MODE = "mean"

# optional attributes
USE_ATTRIBUTES = True
ATTRIBUTE_COLUMNS = ["layer_guess_reordered"]
ATTRIBUTE_SCHEMA = {"layer_guess_reordered": "categorical"}
ATTR_SPATIAL_BANDWIDTH = 0.3



for GROUP_ID in [1]:
# for GROUP_ID in [1]:
    print(f"group id {GROUP_ID}")
    AXIS_OUT = os.path.join(ROOT, "pairwise_output", f"group_{GROUP_ID}")
    # gene subset
    # for MODE in ["query","analysis","virtual_cf"]:
    for MODE in ["integration"]:
        OUT_DIR = os.path.join(
            ROOT,
            "virtual_tissue",
            f"{BIOLOGICAL_AXIS_NAME}_group{GROUP_ID}_grid{GRID_SIZE}_spatial{SPATIAL_BANDWIDTH}",
            MODE,
        )
        os.makedirs(OUT_DIR, exist_ok=True)
        # =========================
        # LOAD ALL ANCHORS ONCE
        # =========================
        if MODE == "integration":
            ANCHOR_IDS = [0, 1, 2, 3]
            GENE_SUBSET = ["PCP4"]
            anchor_bank = load_anchor_bank()
            run_integration()
