from virtual_utils import *
from axis.coordinate_utils import find_inverse_warp_coords
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


# ---------------------------------------------------------------------
# FIXED-VISIUM SPARSE OT FUSION
# ---------------------------------------------------------------------

def _to_dense_local(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _safe_l2_normalize_np(X):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[~np.isfinite(norms) | (norms < 1e-8)] = 1.0
    out = X / norms
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _build_visium_lattice_edges(grid):
    """
    Build first-ring edges directly from the fixed Visium coordinates.

    This replaces the square-pixel `_build_lattice_edges(grid, grid_size)`
    operation. The lattice spacing is inferred from the supported Visium grid,
    preserving the actual coordinate system rather than rebuilding a grid.
    """
    grid = np.asarray(grid, dtype=np.float32)
    m = grid.shape[0]
    if m < 2:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.nan,
        )

    tree = cKDTree(grid)
    k_query = min(8, m)
    dists, inds = tree.query(grid, k=k_query)

    if k_query == 1:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.nan,
        )

    first_positive = dists[:, 1]
    first_positive = first_positive[
        np.isfinite(first_positive) & (first_positive > 0)
    ]
    if first_positive.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.nan,
        )

    lattice_spacing = float(np.median(first_positive))
    radius = 1.35 * lattice_spacing

    edges = set()
    for i in range(m):
        for distance, j in zip(
            np.atleast_1d(dists[i, 1:]),
            np.atleast_1d(inds[i, 1:]),
        ):
            if not np.isfinite(distance) or distance > radius:
                continue
            j = int(j)
            if i == j:
                continue
            edges.add((min(i, j), max(i, j)))

    if not edges:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            lattice_spacing,
        )

    edges = sorted(edges)
    return (
        np.asarray([x[0] for x in edges], dtype=np.int64),
        np.asarray([x[1] for x in edges], dtype=np.int64),
        lattice_spacing,
    )


def _sinkhorn_sparse_local(
    a,
    b,
    row_idx,
    col_idx,
    kernel_values,
    n_cols,
    eps_iter=300,
    tol=1e-6,
):
    """Sparse Sinkhorn scaling on a neighbor-limited kernel."""
    n_rows = int(a.shape[0])
    indices = torch.stack([row_idx, col_idx], dim=0)

    Ksp = torch.sparse_coo_tensor(
        indices,
        kernel_values,
        size=(n_rows, n_cols),
        device=a.device,
        dtype=a.dtype,
    ).coalesce()

    KTsp = torch.sparse_coo_tensor(
        torch.stack([col_idx, row_idx], dim=0),
        kernel_values,
        size=(n_cols, n_rows),
        device=a.device,
        dtype=a.dtype,
    ).coalesce()

    tiny = torch.tensor(1e-20, device=a.device, dtype=a.dtype)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for iteration in range(int(eps_iter)):
        Kv = torch.sparse.mm(
            Ksp,
            v.unsqueeze(1),
        ).squeeze(1)
        if torch.any(Kv <= 0):
            raise FloatingPointError(
                "Sparse Sinkhorn encountered a row with zero kernel mass."
            )
        u_new = a / Kv.clamp_min(tiny)

        KTu = torch.sparse.mm(
            KTsp,
            u_new.unsqueeze(1),
        ).squeeze(1)
        if torch.any(KTu <= 0):
            raise FloatingPointError(
                "Sparse Sinkhorn encountered a grid column with zero mass."
            )
        v_new = b / KTu.clamp_min(tiny)

        if iteration % 10 == 0:
            du = torch.max(
                torch.abs(u_new - u)
                / torch.clamp(torch.abs(u), min=1e-8)
            )
            dv = torch.max(
                torch.abs(v_new - v)
                / torch.clamp(torch.abs(v), min=1e-8)
            )
            if max(float(du), float(dv)) < tol:
                u, v = u_new, v_new
                break

        u, v = u_new, v_new

    return u, v, Ksp, KTsp


def _select_adaptive_rank(singular_values, keep_nuc_frac, knee):
    singular_values = np.asarray(singular_values, dtype=float)
    singular_values = singular_values[
        np.isfinite(singular_values) & (singular_values > 0)
    ]
    if singular_values.size == 0:
        return 0

    if knee and singular_values.size >= 3:
        y = np.log1p(singular_values)
        x = np.linspace(0.0, 1.0, len(y))
        y_scaled = (y - y[-1]) / max(y[0] - y[-1], 1e-12)
        line = 1.0 - x
        knee_index = int(np.argmax(y_scaled - line))
        return max(1, knee_index + 1)

    keep_nuc_frac = float(np.clip(keep_nuc_frac, 1e-6, 1.0))
    cumulative = np.cumsum(singular_values)
    threshold = keep_nuc_frac * cumulative[-1]
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def _svt_adaptive_local(
    X,
    keep_nuc_frac=0.90,
    knee=False,
):
    """
    Adaptive singular-value thresholding.

    The retained rank is selected by cumulative nuclear-norm fraction or the
    singular-value knee. Values below the first discarded singular value are
    removed through soft thresholding.
    """
    if X.ndim != 2:
        raise ValueError("SVT input must be a two-dimensional matrix.")
    if min(X.shape) == 0:
        return X, X[:, :0], X[:0, :]

    U, S, Vt = torch.linalg.svd(
        X,
        full_matrices=False,
    )

    rank = _select_adaptive_rank(
        S.detach().cpu().numpy(),
        keep_nuc_frac=keep_nuc_frac,
        knee=knee,
    )
    if rank <= 0:
        return (
            torch.zeros_like(X),
            X[:, :0],
            X[:0, :],
        )

    rank = min(rank, int(S.numel()))
    threshold = (
        S[rank]
        if rank < int(S.numel())
        else torch.zeros((), device=X.device, dtype=X.dtype)
    )
    shrunk = torch.clamp(S[:rank] - threshold, min=0.0)

    # Prevent a degenerate all-zero output when the spectrum is flat.
    if not torch.any(shrunk > 0):
        shrunk = S[:rank]

    A_latent = U[:, :rank] * shrunk.unsqueeze(0)
    Vt_keep = Vt[:rank, :]
    X_reconstructed = A_latent @ Vt_keep

    return X_reconstructed, A_latent, Vt_keep


def _latent_aniso_local(
    latent,
    edge_i,
    edge_j,
    iters=10,
    tau=0.15,
    kappa=1.0,
):
    """Edge-preserving anisotropic diffusion on the fixed Visium lattice."""
    if (
        int(iters) <= 0
        or latent.shape[1] == 0
        or len(edge_i) == 0
    ):
        return latent

    edge_i_t = torch.as_tensor(
        edge_i,
        device=latent.device,
        dtype=torch.long,
    )
    edge_j_t = torch.as_tensor(
        edge_j,
        device=latent.device,
        dtype=torch.long,
    )

    tau = float(tau)
    kappa_sq = max(float(kappa) ** 2, 1e-12)

    A = latent
    for _ in range(int(iters)):
        delta = A[edge_j_t] - A[edge_i_t]
        squared_distance = torch.mean(delta * delta, dim=1)
        conductance = torch.exp(
            -squared_distance / kappa_sq
        ).unsqueeze(1)
        flux = conductance * delta

        update = torch.zeros_like(A)
        degree = torch.zeros(
            (A.shape[0], 1),
            device=A.device,
            dtype=A.dtype,
        )

        update.index_add_(0, edge_i_t, flux)
        update.index_add_(0, edge_j_t, -flux)
        degree.index_add_(
            0,
            edge_i_t,
            conductance,
        )
        degree.index_add_(
            0,
            edge_j_t,
            conductance,
        )

        A = A + tau * update / degree.clamp_min(1e-8)

    return A


def _ensure_every_grid_column_has_an_edge(
    coords,
    grid,
    neighbor_lists,
):
    """
    Add a nearest-spot edge for any supported Visium point with no incoming edge.

    This guarantees that the output retains exactly the supplied supported
    Visium grid instead of silently dropping unsupported columns.
    """
    n_grid = grid.shape[0]
    counts = np.zeros(n_grid, dtype=np.int64)
    for neighbors in neighbor_lists:
        counts[np.asarray(neighbors, dtype=np.int64)] += 1

    orphan_grid = np.where(counts == 0)[0]
    if orphan_grid.size == 0:
        return neighbor_lists

    spot_tree = cKDTree(coords)
    _, nearest_spot = spot_tree.query(
        grid[orphan_grid],
        k=1,
    )

    neighbor_sets = [set(items) for items in neighbor_lists]
    for grid_index, spot_index in zip(
        orphan_grid.tolist(),
        np.atleast_1d(nearest_spot).astype(int).tolist(),
    ):
        neighbor_sets[spot_index].add(int(grid_index))

    return [sorted(items) for items in neighbor_sets]


def _transport_attributes_from_plan(
    attrs,
    attr_schema,
    row_idx,
    col_idx,
    edge_mass,
    n_grid,
):
    """
    Transport attributes through the final OT plan.

    Categorical attributes are assigned by maximum transported probability.
    Continuous attributes are assigned by transported weighted mean.
    """
    if attrs is None or attr_schema is None:
        return pd.DataFrame(index=np.arange(n_grid)), {}, {}

    attrs = attrs.reset_index(drop=True)
    if len(attrs) != int(row_idx.max().item()) + 1:
        # row_idx should cover all source rows because each source has a neighbor.
        if len(attrs) < int(row_idx.max().item()) + 1:
            raise ValueError(
                "Attribute rows are not aligned with the OT source rows."
            )

    row_np = row_idx.detach().cpu().numpy().astype(np.int64)
    col_np = col_idx.detach().cpu().numpy().astype(np.int64)
    mass_np = edge_mass.detach().cpu().numpy().astype(np.float64)

    obs_out = pd.DataFrame(index=np.arange(n_grid))
    probability_matrices = {}
    category_metadata = {}

    for column, kind in attr_schema.items():
        if column not in attrs:
            raise KeyError(
                f"Attribute column {column!r} is absent from attrs."
            )

        kind = str(kind).lower()
        values = attrs[column]

        if kind == "categorical":
            categorical = pd.Categorical(values)
            categories = [str(x) for x in categorical.categories]
            codes = categorical.codes.astype(np.int64)
            n_categories = len(categories)

            probabilities = np.zeros(
                (n_grid, n_categories),
                dtype=np.float64,
            )
            valid_source = codes[row_np] >= 0
            np.add.at(
                probabilities,
                (
                    col_np[valid_source],
                    codes[row_np[valid_source]],
                ),
                mass_np[valid_source],
            )

            row_sum = probabilities.sum(axis=1, keepdims=True)
            normalized = np.divide(
                probabilities,
                row_sum,
                out=np.zeros_like(probabilities),
                where=row_sum > 0,
            )

            best = np.argmax(normalized, axis=1)
            labels = np.asarray(categories, dtype=object)[best]
            labels[row_sum[:, 0] <= 0] = pd.NA
            obs_out[column] = pd.Categorical(
                labels,
                categories=categories,
            )

            probability_matrices[column] = normalized.astype(np.float32)
            category_metadata[column] = categories

        elif kind in {"continuous", "numeric", "float"}:
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
            valid_source = np.isfinite(numeric[row_np])

            weighted_sum = np.zeros(n_grid, dtype=np.float64)
            weight_sum = np.zeros(n_grid, dtype=np.float64)
            np.add.at(
                weighted_sum,
                col_np[valid_source],
                mass_np[valid_source] * numeric[row_np[valid_source]],
            )
            np.add.at(
                weight_sum,
                col_np[valid_source],
                mass_np[valid_source],
            )
            obs_out[column] = np.divide(
                weighted_sum,
                weight_sum,
                out=np.full(n_grid, np.nan, dtype=np.float64),
                where=weight_sum > 0,
            )
        else:
            raise ValueError(
                f"Unsupported attribute type {kind!r} for {column!r}."
            )

    return obs_out, probability_matrices, category_metadata


def spatial_regrid_fuse_visium_gpu_robust(
    adata: AnnData,
    fixed_grid_pts,
    grid_support_mask,
    point_masses=None,
    fuse_radius: float = None,
    k_mask: int = 64,
    alpha: float = 0.5,
    pca_dim: int = 32,
    outer_loops: int = 2,
    keep_nuc_frac: float = 0.90,
    knee_mode: bool = False,
    aniso_iters: int = 10,
    aniso_tau: float = 0.15,
    aniso_kappa: float = 1.0,
    gene_chunk: int = 1024,
    epsilon: float = 3.0,
    sinkhorn_iters: int = 300,
    device: str = "cuda",
    use_raw: bool = False,
    attr_schema=None,
):
    """
    Sparse GPU OT fusion on a fixed supported Visium grid.

    This is the fixed-grid adaptation of `spatial_regrid_fuse_gpu_robust`.
    The original `_make_grid(coords, grid_size)` step is deliberately removed.
    Output positions are exactly:

        fixed_grid_pts[grid_support_mask]

    The `grid_size` argument is retained so the exposed call uses the same
    parameter values as the original method. For the fixed-grid adaptation it
    is recorded as provenance; lattice geometry is inferred from the supplied
    Visium coordinates.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.")
        device = "cpu"

    coords = np.asarray(
        adata.obsm["spatial"],
        dtype=np.float32,
    )
    X_source = (
        adata.raw.X
        if use_raw and adata.raw is not None
        else adata.X
    )
    X_spot_np = _to_dense_local(X_source).astype(np.float32)
    var_ref = (
        adata.raw.var.copy()
        if use_raw and adata.raw is not None
        else adata.var.copy()
    )

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] must have shape (n, >=2).")
    coords = coords[:, :2]
    if coords.shape[0] != X_spot_np.shape[0]:
        raise ValueError("Coordinate and expression row counts differ.")
    if not np.all(np.isfinite(coords)):
        raise ValueError("Source coordinates contain non-finite values.")
    if not np.all(np.isfinite(X_spot_np)):
        raise ValueError("Source expression contains non-finite values.")
    if np.any(X_spot_np < 0):
        raise ValueError(
            "OT fusion expects non-negative expression values."
        )

    n, n_genes = X_spot_np.shape
    if n < 2:
        raise ValueError("At least two source observations are required.")
    if n_genes < 1:
        raise ValueError("At least one gene is required.")

    grid_full = np.asarray(
        fixed_grid_pts,
        dtype=np.float32,
    )
    support_mask = np.asarray(
        grid_support_mask,
        dtype=bool,
    )
    if (
        grid_full.ndim != 2
        or grid_full.shape[1] < 2
        or support_mask.shape != (grid_full.shape[0],)
    ):
        raise ValueError(
            "fixed_grid_pts and grid_support_mask are incompatible."
        )

    grid = grid_full[support_mask, :2].copy()
    if grid.shape[0] == 0:
        raise ValueError("No Visium grid positions are supported.")

    if point_masses is None:
        point_masses = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        point_masses = np.asarray(point_masses, dtype=np.float64)
        if point_masses.shape != (n,):
            raise ValueError(
                f"point_masses must have shape ({n},)."
            )
        if (
            not np.all(np.isfinite(point_masses))
            or np.any(point_masses <= 0)
        ):
            raise ValueError(
                "point_masses must be finite and strictly positive."
            )
        point_masses = point_masses / point_masses.sum()

    x_spot = torch.as_tensor(
        coords,
        device=device,
        dtype=torch.float32,
    )
    X_spot = torch.as_tensor(
        X_spot_np,
        device=device,
        dtype=torch.float32,
    )
    x_grid = torch.as_tensor(
        grid,
        device=device,
        dtype=torch.float32,
    )
    n_grid = grid.shape[0]

    # Gene embedding used only for the OT cost.
    library_size = np.clip(
        X_spot_np.sum(axis=1, keepdims=True),
        1.0,
        None,
    )
    X_norm = np.log1p(
        X_spot_np / library_size * 1e4
    )
    X_norm = np.nan_to_num(
        X_norm,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    n_components = max(
        1,
        min(
            int(pca_dim),
            int(n),
            int(X_norm.shape[1]),
        ),
    )
    pca = PCA(
        n_components=n_components,
        random_state=0,
    )
    Xp_spot_np = pca.fit_transform(
        X_norm
    ).astype(np.float32)
    Xp_spot_np = _safe_l2_normalize_np(Xp_spot_np)
    Xp_spot = torch.as_tensor(
        Xp_spot_np,
        device=device,
        dtype=torch.float32,
    )

    # Neighbor-limited sparse support from source spots to fixed Visium points.
    grid_tree = cKDTree(grid)
    neighbor_lists = []

    for source_coord in coords:
        if fuse_radius is None:
            indices = list(range(n_grid))
        else:
            indices = grid_tree.query_ball_point(
                source_coord,
                r=float(fuse_radius),
            )

        if len(indices) == 0:
            _, nearest = grid_tree.query(
                source_coord,
                k=1,
            )
            indices = [int(nearest)]

        if k_mask is not None and len(indices) > int(k_mask):
            candidate = grid[np.asarray(indices, dtype=np.int64)]
            distances = np.linalg.norm(
                candidate - source_coord[None, :],
                axis=1,
            )
            order = np.argsort(distances)[: int(k_mask)]
            indices = list(
                np.asarray(indices, dtype=np.int64)[order]
            )

        neighbor_lists.append(
            [int(index) for index in indices]
        )

    neighbor_lists = _ensure_every_grid_column_has_an_edge(
        coords=coords,
        grid=grid,
        neighbor_lists=neighbor_lists,
    )

    column_counts = np.zeros(n_grid, dtype=np.int64)
    for indices in neighbor_lists:
        column_counts[np.asarray(indices, dtype=np.int64)] += 1

    if np.any(column_counts == 0):
        raise RuntimeError(
            "A supported Visium point has no incoming OT edge."
        )

    max_neighbors = max(len(items) for items in neighbor_lists)
    neighbor_index = -np.ones(
        (n, max_neighbors),
        dtype=np.int64,
    )
    for i, items in enumerate(neighbor_lists):
        neighbor_index[i, : len(items)] = items

    neighbor_index_t = torch.as_tensor(
        neighbor_index,
        device=device,
        dtype=torch.long,
    )
    valid_mask = neighbor_index_t >= 0

    grid_neighbors = x_grid[
        neighbor_index_t.clamp_min(0)
    ]
    source_expanded = x_spot.unsqueeze(1).expand_as(
        grid_neighbors
    )

    spatial_cost = (
        source_expanded - grid_neighbors
    ).pow(2).sum(dim=2)
    spatial_median = (
        torch.median(spatial_cost[valid_mask])
        if valid_mask.any()
        else torch.tensor(1.0, device=device)
    )
    spatial_cost = spatial_cost / spatial_median.clamp(min=1e-6)

    # Initialize the target gene embedding from geometry, while respecting the
    # supplied equal-source point masses.
    with torch.no_grad():
        distances = torch.linalg.norm(
            grid_neighbors - source_expanded,
            dim=2,
        ).clamp(min=1e-12)

        local_weights = (
            1.0 / distances
        ) * valid_mask.float()
        local_weights = local_weights / local_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp(min=1e-8)

        mass_t = torch.as_tensor(
            point_masses,
            device=device,
            dtype=torch.float32,
        )
        local_weights = local_weights * mass_t.unsqueeze(1)

        Xp_grid = torch.zeros(
            (n_grid, Xp_spot.shape[1]),
            device=device,
            dtype=torch.float32,
        )
        init_mass = torch.zeros(
            (n_grid, 1),
            device=device,
            dtype=torch.float32,
        )

        for neighbor_slot in range(max_neighbors):
            slot_mask = valid_mask[:, neighbor_slot]
            target_index = neighbor_index_t[
                slot_mask,
                neighbor_slot,
            ]
            slot_weight = local_weights[
                slot_mask,
                neighbor_slot,
            ].unsqueeze(1)

            Xp_grid.index_add_(
                0,
                target_index,
                slot_weight * Xp_spot[slot_mask],
            )
            init_mass.index_add_(
                0,
                target_index,
                slot_weight,
            )

        Xp_grid = Xp_grid / init_mass.clamp_min(1e-12)
        Xp_grid = Xp_grid / (
            Xp_grid.norm(dim=1, keepdim=True) + 1e-8
        )

    edge_i, edge_j, inferred_lattice_spacing = (
        _build_visium_lattice_edges(grid)
    )

    a = torch.as_tensor(
        point_masses,
        device=device,
        dtype=torch.float32,
    )
    b = torch.as_tensor(
        column_counts / column_counts.sum(),
        device=device,
        dtype=torch.float32,
    )

    final_row_idx = None
    final_col_idx = None
    final_kernel_values = None
    final_u = None
    final_v = None
    final_column_mass = None
    X_grid = None

    for outer_iteration in range(int(outer_loops)):
        target_neighbor_embedding = Xp_grid[
            neighbor_index_t.clamp_min(0)
        ]
        cosine_similarity = (
            Xp_spot.unsqueeze(1)
            * target_neighbor_embedding
        ).sum(dim=2)
        gene_cost = 1.0 - cosine_similarity

        gene_median = (
            torch.median(gene_cost[valid_mask])
            if valid_mask.any()
            else torch.tensor(1.0, device=device)
        )
        gene_cost = gene_cost / gene_median.clamp(min=1e-6)

        total_cost = (
            float(alpha) * spatial_cost
            + (1.0 - float(alpha)) * gene_cost
        )

        kernel = torch.zeros_like(total_cost)
        kernel[valid_mask] = torch.exp(
            -total_cost[valid_mask]
            / max(float(epsilon), 1e-12)
        ).clamp_min(1e-30)

        row_ids = torch.arange(
            n,
            device=device,
        ).unsqueeze(1).expand(
            n,
            max_neighbors,
        )
        row_idx = row_ids[valid_mask]
        col_idx = neighbor_index_t[valid_mask]
        kernel_values = kernel[valid_mask]

        u, v, Ksp, KTsp = _sinkhorn_sparse_local(
            a=a,
            b=b,
            row_idx=row_idx,
            col_idx=col_idx,
            kernel_values=kernel_values,
            n_cols=n_grid,
            eps_iter=sinkhorn_iters,
            tol=1e-6,
        )

        column_mass = (
            torch.sparse.mm(
                KTsp,
                u.unsqueeze(1),
            ).squeeze(1)
            * v
        ).clamp(min=1e-12)

        X_grid = torch.zeros(
            (n_grid, n_genes),
            device=device,
            dtype=torch.float32,
        )

        for start in range(0, n_genes, int(gene_chunk)):
            end = min(n_genes, start + int(gene_chunk))
            weighted_source = (
                u.unsqueeze(1)
                * X_spot[:, start:end]
            )
            target_chunk = torch.sparse.mm(
                KTsp,
                weighted_source,
            )
            target_chunk = (
                v.unsqueeze(1) * target_chunk
            )
            X_grid[:, start:end] = (
                target_chunk
                / column_mass.unsqueeze(1)
            )

        X_grid, latent, Vt = _svt_adaptive_local(
            X_grid,
            keep_nuc_frac=keep_nuc_frac,
            knee=knee_mode,
        )

        if (
            int(aniso_iters) > 0
            and latent.shape[1] > 0
            and len(edge_i) > 0
        ):
            latent = _latent_aniso_local(
                latent,
                edge_i=edge_i,
                edge_j=edge_j,
                iters=aniso_iters,
                tau=aniso_tau,
                kappa=aniso_kappa,
            )
            X_grid = latent @ Vt

        grid_expression = (
            X_grid.detach().cpu().numpy()
        )
        grid_expression = np.nan_to_num(
            grid_expression,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        grid_expression[
            grid_expression < 0.0
        ] = 0.0

        row_sum = grid_expression.sum(
            axis=1,
            keepdims=True,
        )
        row_sum[row_sum == 0.0] = 1.0
        grid_normalized = np.log1p(
            grid_expression / row_sum * 1e4
        )
        grid_embedding = pca.transform(
            grid_normalized
        )
        grid_embedding = _safe_l2_normalize_np(
            grid_embedding
        )
        Xp_grid = torch.as_tensor(
            grid_embedding,
            device=device,
            dtype=torch.float32,
        )

        b = (
            column_mass / column_mass.sum()
        ).detach()

        final_row_idx = row_idx
        final_col_idx = col_idx
        final_kernel_values = kernel_values
        final_u = u
        final_v = v
        final_column_mass = column_mass

        print(
            f"  [OT] outer={outer_iteration + 1}/{outer_loops} "
            f"edges={len(kernel_values):,} "
            f"latent_rank={latent.shape[1]}"
        )

    if X_grid is None:
        raise RuntimeError("OT fusion did not execute any outer loop.")

    fused_expression = X_grid.detach().cpu().numpy()
    fused_expression = np.nan_to_num(
        fused_expression,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    fused_expression[fused_expression < 0.0] = 0.0

    final_edge_mass = (
        final_u[final_row_idx]
        * final_kernel_values
        * final_v[final_col_idx]
    )

    attrs_source = (
        adata.obs.loc[:, list(attr_schema)].copy()
        if attr_schema
        else None
    )
    obs_out, attr_probabilities, attr_categories = (
        _transport_attributes_from_plan(
            attrs=attrs_source,
            attr_schema=attr_schema,
            row_idx=final_row_idx,
            col_idx=final_col_idx,
            edge_mass=final_edge_mass,
            n_grid=n_grid,
        )
    )

    new_adata = AnnData(
        X=fused_expression,
        obs=obs_out,
        var=var_ref.copy(),
    )
    new_adata.obsm["spatial"] = grid.astype(np.float32)
    new_adata.obs_names = [
        f"visium_grid_{i}"
        for i in range(new_adata.n_obs)
    ]

    for attribute, probabilities in attr_probabilities.items():
        new_adata.obsm[
            f"{attribute}_probabilities"
        ] = probabilities
        new_adata.uns[
            f"{attribute}_categories"
        ] = attr_categories[attribute]

    new_adata.obs["fixed_grid_index"] = np.flatnonzero(
        support_mask
    ).astype(np.int64)
    new_adata.obs["ot_target_mass"] = (
        final_column_mass.detach().cpu().numpy().astype(np.float32)
    )

    new_adata.uns.update(
        dict(
            integration_generator=(
                "spatial_regrid_fuse_visium_gpu_robust"
            ),
            fixed_visium_grid=True,
            inferred_visium_lattice_spacing=(
                None
                if not np.isfinite(inferred_lattice_spacing)
                else float(inferred_lattice_spacing)
            ),
            fuse_radius=(
                None
                if fuse_radius is None
                else float(fuse_radius)
            ),
            alpha=float(alpha),
            pca_dim=int(pca_dim),
            outer_loops=int(outer_loops),
            keep_nuc_frac=float(keep_nuc_frac),
            knee_mode=bool(knee_mode),
            aniso_iters=int(aniso_iters),
            aniso_tau=float(aniso_tau),
            aniso_kappa=float(aniso_kappa),
            epsilon=float(epsilon),
            sinkhorn_iters=int(sinkhorn_iters),
            gene_chunk=int(gene_chunk),
            sparse_mode=True,
            k_mask=int(k_mask),
            n_ot_edges=int(len(final_kernel_values)),
            n_full_visium_grid_points=int(grid_full.shape[0]),
            n_supported_visium_grid_points=int(grid.shape[0]),
        )
    )

    return new_adata


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

ROOT = "/media/huifang/data/registration/DLPFC/huifang"
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
