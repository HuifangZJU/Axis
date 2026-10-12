from axis.config import load_config
from axis.registration import find_inverse_warp_coords
from axis.generation import (
    compute_axis_kernel_weights,
    compute_dense_tissue_mask_on_any_grid,
    ensure_same_genes_keep_order,
    generate_virtual_tissue_fixed_grid,
    get_anchor_pair_for_query,
    make_common_grid_from_coords,
    matrix_log1p,
    relative_position_on_axis,
    save_generation_meta,
    stack_attribute_evidence,
    stack_weighted_evidence,
)
from axis.io import load_axis_flow, load_coords_npz, load_h5ad_X_and_genes
from axis.matching.population import spatial_regrid_fuse_visium_gpu_robust
import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
from anndata import AnnData
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA




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

def build_analysis_grid_and_evidence(
    coords,
    fixed_grid_pts,
    expr,
    flow_to_f
):
    coords_to_f = find_inverse_warp_coords(
        coords,
        (flow_to_f).astype(np.float32)
    ).astype(np.float32)

    grid_support = compute_dense_tissue_mask_on_any_grid(
        grid_pts=fixed_grid_pts,
        union_spots=coords_to_f,
        max_dist_px=MAX_DIST_PX,
        keep_largest=True,
    )
    point_weights = np.full(expr.shape[0], 1, dtype=float)

    X_cat = expr
    return fixed_grid_pts, grid_support, coords_to_f, X_cat, point_weights



def build_query_evidence(
    alpha,
    grid_pts,
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


    grid_support = compute_dense_tissue_mask_on_any_grid(
        grid_pts=grid_pts,
        union_spots=union_coords,
        max_dist_px=MAX_DIST_PX,
        keep_largest=True,
    )

    coords_cat, X_cat, point_weights = stack_weighted_evidence(
        coords_blocks=[coords_a_q, coords_b_q],
        X_blocks=[expr_a, expr_b],
        source_weights=source_weights,
    )

    return grid_support, coords_cat, X_cat, point_weights


def build_virtual_counterfactual_evidence(
    grid_pts,
    coords_a,
    coords_b,
    expr_a,
    expr_b,
    source_weights,
):
    union_coords = np.vstack([coords_a, coords_b]).astype(np.float32)


    grid_support = compute_dense_tissue_mask_on_any_grid(
        grid_pts=grid_pts,
        union_spots=union_coords,
        max_dist_px=MAX_DIST_PX,
        keep_largest=True,
    )

    coords_cat, X_cat, point_weights = stack_weighted_evidence(
        coords_blocks=[coords_a, coords_b],
        X_blocks=[expr_a, expr_b],
        source_weights=source_weights,
    )

    return grid_support, coords_cat, X_cat, point_weights

def build_synchronization_evidence(
    coords,
    grid_pts,
    expr
):
    grid_support = compute_dense_tissue_mask_on_any_grid(
        grid_pts=grid_pts,
        union_spots=coords,
        max_dist_px=MAX_DIST_PX,
        keep_largest=True,
    )
    point_weights = np.full(expr.shape[0], 1, dtype=float)

    return grid_support, point_weights



def run_query(save_meta=False):
    grid_pts, _ = make_common_grid_from_coords(GRID_SIZE, type='visium')
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

        flow_a_to_b = load_axis_flow(
            AXIS_OUT, fixed_id=anchor_b_id, moving_id=anchor_a_id
        )
        flow_b_to_a = load_axis_flow(
            AXIS_OUT, fixed_id=anchor_a_id, moving_id=anchor_b_id
        )

        grid_support, coords_cat, X_cat, point_weights = build_query_evidence(
            grid_pts=grid_pts,
            alpha=alpha,
            coords_a=coords_a,
            coords_b=coords_b,
            expr_a=expr_a,
            expr_b=expr_b,
            flow_a_to_b=flow_a_to_b,
            flow_b_to_a=flow_b_to_a,
            source_weights=source_weights,
        )

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
            attr_spatial_bandwidth=ATTR_SPATIAL_BANDWIDTH,
            knn=KNN,
            min_neighbors=MIN_NEIGHBORS,
            radius=RADIUS,
            chunk_size=64,
            fill_value=0.0,
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
        if save_meta:
            meta_path = save_generation_meta(
                virtual_tissue_path=out_path,

                # User-defined spatial support
                grid_pts=grid_pts,
                grid_support_mask=grid_support,

                # Collected spatial evidence
                coords_cat=coords_cat,
                X_cat=X_cat,
                genes=genes,
                attrs_cat=attrs_cat,
                attr_schema=ATTRIBUTE_SCHEMA,
                point_weights=point_weights,

                # Preserve anchor identity for each evidence point.
                # This assumes build_query_evidence concatenates anchor A
                # followed by anchor B without deleting or duplicating rows.
                source_anchor_ids=[
                    anchor_a_id,
                    anchor_b_id,
                ],
                source_point_counts=[
                    len(coords_a),
                    len(coords_b),
                ],

                # Generation context
                source_axis_positions=source_axis_positions,
                source_weights=source_weights,
                biological_axis_name=BIOLOGICAL_AXIS_NAME,
                query_axis_value=query_axis_value,
                relative_position_alpha=float(alpha),
                mode=MODE,
                apply_log1p_before_generation=bool(
                    APPLY_LOG1P_BEFORE_GENERATION
                ),
                axis_weight_bandwidth=float(
                    used_axis_bandwidth
                ),
                spatial_bandwidth=SPATIAL_BANDWIDTH,
                knn=KNN,
                min_neighbors=MIN_NEIGHBORS,
                radius=RADIUS,
            )

            print("  saved generation metadata:", meta_path)


    print("DONE. Output folder:", OUT_DIR)


def run_virtual_counterfactual():
    grid_pts, _ = make_common_grid_from_coords(GRID_SIZE, type='visium')
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

        flow_a_to_f = load_axis_flow(
            AXIS_OUT, fixed_id=ANALYSIS_REFERENCE_ANCHOR_ID, moving_id=anchor_a_id
        )
        flow_b_to_f = load_axis_flow(
            AXIS_OUT, fixed_id=ANALYSIS_REFERENCE_ANCHOR_ID, moving_id=anchor_b_id
        )

        coords_a_f = find_inverse_warp_coords(
            coords_a,
            flow_a_to_f.astype(np.float32)
        ).astype(np.float32)

        coords_b_f = find_inverse_warp_coords(
            coords_b,
            flow_b_to_f.astype(np.float32)
        ).astype(np.float32)


        grid_support, coords_cat, X_cat, point_weights = build_virtual_counterfactual_evidence(
            grid_pts=grid_pts,
            coords_a=coords_a_f,
            coords_b=coords_b_f,
            expr_a=expr_a,
            expr_b=expr_b,
            source_weights=source_weights,
        )

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
            attr_spatial_bandwidth=ATTR_SPATIAL_BANDWIDTH,
            knn=KNN,
            min_neighbors=MIN_NEIGHBORS,
            radius=RADIUS,
            chunk_size=64,
            fill_value=0.0,
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

def run_analysis():
    grid_pts, _ = make_common_grid_from_coords(GRID_SIZE, type='visium')
    for anchor_id in ANCHOR_IDS:
        axis_value = float(AXIS_POSITIONS[anchor_id])
        rec = anchor_bank[anchor_id]

        expr_raw = rec["expr_raw"]
        genes = rec["genes"]
        attrs_raw = rec["attrs"]
        coords = rec["coords"]

        expr = maybe_log1p(expr_raw)

        print(
            f"[GEN] {BIOLOGICAL_AXIS_NAME}={axis_value:.3f}  "
            f"anchors=({anchor_id})"
        )

        flow_to_f = load_axis_flow(
            AXIS_OUT, fixed_id=ANALYSIS_REFERENCE_ANCHOR_ID, moving_id=anchor_id
        )

        grid_pts, grid_support, coords_cat, X_cat, point_weights = build_analysis_grid_and_evidence(
            coords=coords,
            fixed_grid_pts=grid_pts,
            expr=expr,
            flow_to_f=flow_to_f
        )

        if USE_ATTRIBUTES:
            attrs_cat = attrs_raw
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
            attr_spatial_bandwidth=ATTR_SPATIAL_BANDWIDTH,
            knn=KNN,
            min_neighbors=MIN_NEIGHBORS,
            radius=RADIUS,
            chunk_size=64,
            fill_value=0.0,
        )

        # metadata
        ad_grid.uns["biological_axis_name"] = BIOLOGICAL_AXIS_NAME
        ad_grid.uns["mode"] = MODE
        ad_grid.uns["grid_size"] = int(GRID_SIZE)
        ad_grid.uns["apply_log1p_before_generation"] = bool(APPLY_LOG1P_BEFORE_GENERATION)

        if MODE == "analysis":
            ad_grid.uns["analysis_reference_anchor_id"] = int(ANALYSIS_REFERENCE_ANCHOR_ID)

        axis_tag = str(BIOLOGICAL_AXIS_NAME).replace(" ", "_")
        out_path = os.path.join(
            OUT_DIR,
            f"virtual_{axis_tag}_{axis_value:.3f}.h5ad"
        )
        ad_grid.write_h5ad(out_path)
        print("  saved:", out_path)

    print("DONE. Output folder:", OUT_DIR)




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
    genes. A requested GENE_SUBSET retains the user's requested order.
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
        available = set(common_genes.tolist())
        common_genes = np.asarray(
            [str(gene) for gene in gene_subset if str(gene) in available],
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

        lookup = {gene: idx for idx, gene in enumerate(genes_i.tolist())}
        indices = np.asarray(
            [lookup[gene] for gene in common_genes],
            dtype=np.int64,
        )
        expr_i = anchor_bank[aid]["expr_raw"][:, indices]
        expr_i = maybe_log1p(expr_i)
        expression_blocks.append(expr_i)

    return expression_blocks, common_genes


def transfer_anchor_to_reference(anchor_id, reference_anchor_id):
    """
    Fully transform one anchor into the fixed reference coordinate frame.

    The reference anchor uses an identity transform. The reference defines
    geometry only and receives no additional biological mass.
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


def _stack_expression_blocks(expression_blocks):
    """Stack dense or sparse expression matrices without changing values."""
    if any(sp.issparse(block) for block in expression_blocks):
        return sp.vstack(
            [
                block.tocsr() if sp.issparse(block) else sp.csr_matrix(block)
                for block in expression_blocks
            ],
            format="csr",
        )
    return np.vstack(
        [np.asarray(block) for block in expression_blocks]
    )


def concatenate_attribute_blocks(attribute_blocks):
    """Concatenate attributes in the same row order as coordinates/expression."""
    if not USE_ATTRIBUTES:
        return None

    if len(attribute_blocks) == 0:
        raise ValueError(
            "USE_ATTRIBUTES=True, but no attribute blocks were collected."
        )

    missing = [
        index
        for index, attrs in enumerate(attribute_blocks)
        if attrs is None
    ]
    if missing:
        raise ValueError(
            "USE_ATTRIBUTES=True, but attributes are missing for source "
            f"blocks {missing}."
        )

    return pd.concat(
        [
            attrs.loc[:, ATTRIBUTE_COLUMNS].reset_index(drop=True)
            for attrs in attribute_blocks
        ],
        axis=0,
        ignore_index=True,
    )


def build_integration_grid_and_evidence(
    fixed_grid_pts,
    anchor_ids,
    reference_anchor_id,
):
    """
    Build one equal-biological-weight integration evidence bank.

    Every anchor is fully transferred into the reference frame. Coordinates,
    expression, and attributes are concatenated once. Equal biological weight
    is represented as a source probability mass: each anchor receives total
    mass 1 / N, divided uniformly among that anchor's observations.
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
    aligned_expression_blocks = []
    attribute_blocks = []
    source_point_counts = []

    for aid, expr_i in zip(anchor_ids, expression_blocks):
        coords_i = transfer_anchor_to_reference(
            anchor_id=aid,
            reference_anchor_id=reference_anchor_id,
        )
        attrs_i = anchor_bank[aid]["attrs"] if USE_ATTRIBUTES else None

        coords_i, expr_i, attrs_i = align_coords_expr_attrs(
            coords_i,
            expr_i,
            attrs_i,
        )

        if coords_i.shape[0] == 0:
            raise ValueError(
                f"Anchor {aid} has no aligned observations for integration."
            )

        coordinate_blocks.append(
            np.asarray(coords_i, dtype=np.float32)
        )
        aligned_expression_blocks.append(expr_i)
        attribute_blocks.append(attrs_i)
        source_point_counts.append(int(coords_i.shape[0]))

        print(
            f"  [INTEGRATE] anchor={aid} "
            f"axis={float(AXIS_POSITIONS[aid]):.3f} "
            f"points={coords_i.shape[0]:,}"
        )

    coords_cat = np.vstack(coordinate_blocks).astype(np.float32)
    X_cat = _stack_expression_blocks(aligned_expression_blocks)
    attrs_cat = concatenate_attribute_blocks(attribute_blocks)

    n_sources = len(anchor_ids)
    source_weights = np.full(
        n_sources,
        1.0 / n_sources,
        dtype=np.float64,
    )

    # Each source contributes exactly 1/N total OT mass, regardless of how
    # many spots are present in that source.
    point_masses = np.concatenate(
        [
            np.full(
                count,
                source_weight / count,
                dtype=np.float64,
            )
            for source_weight, count in zip(
                source_weights,
                source_point_counts,
            )
        ]
    )
    point_masses /= point_masses.sum()

    expected_rows = int(sum(source_point_counts))
    if coords_cat.shape[0] != expected_rows:
        raise RuntimeError("Coordinate concatenation row mismatch.")
    if X_cat.shape[0] != expected_rows:
        raise RuntimeError("Expression concatenation row mismatch.")
    if attrs_cat is not None and len(attrs_cat) != expected_rows:
        raise RuntimeError("Attribute concatenation row mismatch.")
    if len(point_masses) != expected_rows:
        raise RuntimeError("Point-mass concatenation row mismatch.")

    grid_support = compute_dense_tissue_mask_on_any_grid(
        grid_pts=fixed_grid_pts,
        union_spots=coords_cat,
        max_dist_px=MAX_DIST_PX,
        keep_largest=True,
    )
    grid_support = np.asarray(grid_support, dtype=bool)

    if grid_support.shape[0] != np.asarray(fixed_grid_pts).shape[0]:
        raise ValueError(
            "The Visium support mask is not aligned with fixed_grid_pts."
        )
    if not np.any(grid_support):
        raise ValueError("The integration Visium support mask is empty.")

    source_anchor_per_point = np.concatenate(
        [
            np.full(count, aid, dtype=np.int32)
            for aid, count in zip(anchor_ids, source_point_counts)
        ]
    )

    return {
        "grid_pts": np.asarray(fixed_grid_pts, dtype=np.float32),
        "grid_support": grid_support,
        "coords_cat": coords_cat,
        "X_cat": X_cat,
        "genes": genes,
        "attrs_cat": attrs_cat,
        "point_masses": point_masses,
        "source_anchor_ids": anchor_ids,
        "source_anchor_per_point": source_anchor_per_point,
        "source_axis_positions": np.asarray(
            [AXIS_POSITIONS[aid] for aid in anchor_ids],
            dtype=np.float32,
        ),
        "source_weights": source_weights,
        "source_point_counts": source_point_counts,
    }


def _build_integration_anndata(evidence):
    """Package the aligned source evidence for the OT fusion function."""
    var = pd.DataFrame(
        index=pd.Index(
            evidence["genes"].astype(str),
            name="gene",
        )
    )
    obs = (
        evidence["attrs_cat"].copy()
        if evidence["attrs_cat"] is not None
        else pd.DataFrame(
            index=np.arange(
                evidence["coords_cat"].shape[0]
            )
        )
    )

    obs["source_anchor_id"] = pd.Categorical(
        evidence["source_anchor_per_point"].astype(str)
    )
    obs["source_point_mass"] = evidence[
        "point_masses"
    ].astype(np.float64)

    adata_full = AnnData(
        X=evidence["X_cat"],
        obs=obs,
        var=var,
    )
    adata_full.obsm["spatial"] = evidence[
        "coords_cat"
    ].astype(np.float32)
    adata_full.obs_names_make_unique()

    return adata_full


def run_integration(save_meta=True):
    """
    Generate one integrated slice using fixed-Visium sparse OT fusion.

    All anchors are transferred to the reference frame and concatenated once.
    The common local generator is not used in this mode.
    """
    grid_pts, _ = make_common_grid_from_coords(
        GRID_SIZE,
        type="visium",
    )

    evidence = build_integration_grid_and_evidence(
        fixed_grid_pts=grid_pts,
        anchor_ids=ANCHOR_IDS,
        reference_anchor_id=ANALYSIS_REFERENCE_ANCHOR_ID,
    )
    adata_full = _build_integration_anndata(evidence)

    print(
        f"[GEN-INTEGRATION-OT] "
        f"sources={evidence['source_anchor_ids']} "
        f"reference={ANALYSIS_REFERENCE_ANCHOR_ID} "
        f"points={adata_full.n_obs:,} "
        f"genes={adata_full.n_vars:,} "
        f"supported_grid={int(evidence['grid_support'].sum()):,}"
    )
    print(
        "  equal biological source weights:",
        evidence["source_weights"].tolist(),
    )

    # Exactly the parameter values from the successful pixel-grid call.
    adata_fused = spatial_regrid_fuse_visium_gpu_robust(
        adata_full,
        fixed_grid_pts=evidence["grid_pts"],
        grid_support_mask=evidence["grid_support"],
        point_masses=evidence["point_masses"],
        fuse_radius=10,
        keep_nuc_frac=0.6,
        knee_mode=False,
        alpha=0.6,
        pca_dim=32,
        device="cuda",
        use_raw=False,
        attr_schema=(
            ATTRIBUTE_SCHEMA
            if USE_ATTRIBUTES
            else None
        ),
    )

    adata_fused.uns["biological_axis_name"] = BIOLOGICAL_AXIS_NAME
    adata_fused.uns["mode"] = "integration"
    adata_fused.uns["grid_size"] = int(GRID_SIZE)
    adata_fused.uns["max_dist_px"] = float(MAX_DIST_PX)
    adata_fused.uns["apply_log1p_before_generation"] = bool(
        APPLY_LOG1P_BEFORE_GENERATION
    )
    adata_fused.uns["integration_reference_anchor_id"] = int(
        ANALYSIS_REFERENCE_ANCHOR_ID
    )
    adata_fused.uns["integration_reference_axis_value"] = float(
        AXIS_POSITIONS[ANALYSIS_REFERENCE_ANCHOR_ID]
    )
    adata_fused.uns["integration_source_anchor_ids"] = [
        int(x)
        for x in evidence["source_anchor_ids"]
    ]
    adata_fused.uns["integration_source_axis_positions"] = (
        evidence["source_axis_positions"].astype(float).tolist()
    )
    adata_fused.uns["integration_source_weights"] = (
        evidence["source_weights"].astype(float).tolist()
    )
    adata_fused.uns["integration_source_point_counts"] = [
        int(x)
        for x in evidence["source_point_counts"]
    ]
    adata_fused.uns["integration_equal_biological_weight"] = True
    adata_fused.uns["integration_weight_definition"] = (
        "Each anchor has total OT source mass 1/N, distributed uniformly "
        "among that anchor's observations."
    )
    adata_fused.uns["integration_n_evidence_points"] = int(
        evidence["coords_cat"].shape[0]
    )
    adata_fused.uns["integration_n_genes"] = int(
        len(evidence["genes"])
    )
    adata_fused.uns["integration_coordinate_frame"] = (
        f"anchor_{ANALYSIS_REFERENCE_ANCHOR_ID}"
    )

    axis_tag = str(BIOLOGICAL_AXIS_NAME).replace(
        " ",
        "_",
    )
    out_path = os.path.join(
        OUT_DIR,
        (
            f"integrated_{axis_tag}_"
            f"ref_{ANALYSIS_REFERENCE_ANCHOR_ID}_"
            "visium_ot.h5ad"
        ),
    )
    adata_fused.write_h5ad(out_path)
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

    return adata_fused, evidence


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
# receive equal total OT source mass.
ANALYSIS_REFERENCE_ANCHOR_ID = 0

ROOT = load_config('dlpfc').root
NET_IN = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ad")


GRID_SIZE = 80
MAX_DIST_PX = 5.0


# preprocessing / weighting
APPLY_LOG1P_BEFORE_GENERATION = False
AXIS_WEIGHT_BANDWIDTH = None
SPATIAL_BANDWIDTH = 1
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
ATTR_SPATIAL_BANDWIDTH = 6



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
            # GENE_SUBSET = ["PCP4"]
            GENE_SUBSET = None
            anchor_bank = load_anchor_bank()
            run_integration()


        if MODE=="query":
            ANCHOR_IDS = [1, 2]
            QUERY_AXIS_VALUES = [80, 160, 240]

            # QUERY_AXIS_VALUES = np.arange(20, 301, 10).tolist()
            GENE_SUBSET = ["PCP4"]
            # GENE_SUBSET=None
            anchor_bank = load_anchor_bank()
            run_query(save_meta=True)


        if MODE == "analysis":
            ANCHOR_IDS = [0, 1, 2, 3]
            GENE_SUBSET = ["PCP4"]
            anchor_bank = load_anchor_bank()
            run_analysis()
