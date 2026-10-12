"""Reference-based reconstruction of tissue samples (Methods, "Spatially matched analysis").

Every sample is transported into a common reference sample with its own
transition field and reconstructed independently on the reference support
with the local spatial and measurement-quality weights; no cross-state
biological weighting is applied (``beta_r = 1``).
"""

from __future__ import annotations

import glob
import os
from typing import Mapping, Optional, Sequence

import numpy as np

from ..generation.generate import generate_virtual_tissue_fixed_grid
from ..generation.support import compute_dense_tissue_mask_on_any_grid
from ..registration.warp import find_inverse_warp_coords
from ..settings import log

__all__ = ["build_analysis_evidence", "reconstruct_on_reference", "assign_spatial_axis"]


def build_analysis_evidence(coords, grid_pts, expr, flow_to_f, max_dist_px: float = 10.0):
    """Transport one sample into the reference frame and compute its grid support.

    Returns ``(grid_support, coords_in_reference, point_weights)``.
    """
    coords_to_f = np.asarray(
        find_inverse_warp_coords(np.asarray(coords, np.float32), np.asarray(flow_to_f, np.float32)), np.float32
    )
    grid_support = compute_dense_tissue_mask_on_any_grid(
        grid_pts=grid_pts, union_spots=coords_to_f, max_dist_px=max_dist_px, keep_largest=True
    )
    return grid_support, coords_to_f, np.ones(expr.shape[0], dtype=float)


def reconstruct_on_reference(
    coords,
    expr,
    flow_to_reference,
    grid_pts,
    *,
    var_names=None,
    attrs=None,
    attr_schema: Optional[Mapping[str, str]] = None,
    max_dist_px: float = 10.0,
    spatial_bandwidth: Optional[float] = None,
    attr_spatial_bandwidth: Optional[float] = None,
    knn: int = 48,
    min_neighbors: int = 1,
    radius: Optional[float] = None,
    use_gene_weight: bool = True,
):
    """Reconstruct one measured sample on the support of the reference sample.

    Parameters
    ----------
    coords, expr
        Canvas coordinates ``(N, 2)`` and molecular values ``(N, G)`` of the sample.
    flow_to_reference
        Field registering the sample (moving) onto the reference (fixed); zeros
        for the reference itself.
    grid_pts
        Output support in the reference frame (e.g. the reference spot or cell
        coordinates, or a Visium-like lattice).
    """
    grid_support, coords_ref, weights = build_analysis_evidence(
        coords, grid_pts, expr, flow_to_reference, max_dist_px=max_dist_px
    )
    return generate_virtual_tissue_fixed_grid(
        coords=coords_ref,
        X=expr,
        grid_pts=grid_pts,
        var_names=var_names,
        attrs=attrs,
        attr_schema=attr_schema,
        grid_support_mask=grid_support,
        point_weights=weights,
        spatial_bandwidth=spatial_bandwidth,
        attr_spatial_bandwidth=attr_spatial_bandwidth,
        use_gene_weight=use_gene_weight,
        knn=knn,
        min_neighbors=min_neighbors,
        radius=radius,
    )


def assign_spatial_axis(root: str, group_ids: Sequence, reference_id: int = 0, key: str = "spatial_axis") -> None:
    """Write coordinates in the reference frame to ``obsm[key]`` of every ``h5ad/{group}_*.h5ad``.

    Reads ``network_input/{group}_{ref}_validation.npz`` for the reference and
    ``pairwise_output/group_{group}/{ref}_{sample}_warped_coor.npy`` for the
    other samples; files are updated in place.
    """
    import anndata as ad

    h5ad_dir, net_dir = os.path.join(root, "h5ad"), os.path.join(root, "network_input")
    for group_id in group_ids:
        files = sorted(glob.glob(os.path.join(h5ad_dir, f"{group_id}_*.h5ad")))
        if not files:
            log(f"[axis] no h5ad files for group {group_id}")
            continue
        ref_npz = os.path.join(net_dir, f"{group_id}_{reference_id}_validation.npz")
        if not os.path.exists(ref_npz):
            raise FileNotFoundError(f"Reference network coord file not found: {ref_npz}")
        ref_coord = np.load(ref_npz)["coord"].astype(np.float32)

        for path in files:
            sample_id = int(os.path.basename(path).replace(".h5ad", "").split("_")[1])
            adata = ad.read_h5ad(path)
            if sample_id == reference_id:
                coord, source = ref_coord, ref_npz
            else:
                source = os.path.join(root, "pairwise_output", f"group_{group_id}", f"{reference_id}_{sample_id}_warped_coor.npy")
                if not os.path.exists(source):
                    raise FileNotFoundError(f"Warped coord file not found: {source}")
                coord = np.load(source).astype(np.float32)
            coord = coord[:, :2]
            if coord.shape[0] != adata.n_obs:
                raise ValueError(
                    f"Length mismatch in group {group_id}, sample {sample_id}: n_obs={adata.n_obs}, coord={coord.shape}, source={source}"
                )
            adata.obsm[key] = coord
            adata.write(path)
            log(f"[axis] group={group_id} sample={sample_id} {key}={coord.shape} <- {os.path.basename(source)}")
