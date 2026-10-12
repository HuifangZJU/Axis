"""Example data.

* [`load_scc_sample`][axis.datasets.load_scc_sample] reads the three serial SCC Visium sections of one
  patient in the format distributed with the tutorial
  (``{patient}_{i}.h5ad``, i = 0, 1, 2 at 0, 10 and 20 µm).
* [`toy_series`][axis.datasets.toy_series] simulates a small series of deformed sections with
  matching morphology images; it needs no download and is used by the tests and
  the documentation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np

__all__ = ["load_scc_sample", "toy_series", "SCC_DEPTHS_UM"]

SCC_DEPTHS_UM = (0.0, 10.0, 20.0)
"""Depths (µm) of SCC sections 0, 1 and 2; [`load_scc_sample`][axis.datasets.load_scc_sample] stores each section's depth in ``uns["depth_um"]``."""


def load_scc_sample(directory: Union[str, os.PathLike], patient: int = 10, sections: Sequence[int] = (0, 1, 2)) -> List:
    """Load the serial SCC sections of one patient.

    Each file ``{directory}/{patient}_{i}.h5ad`` holds raw counts in ``X``,
    the 512-px H&E canvas in ``uns["image_array"]``, canvas coordinates in
    ``obsm["spatial_image_coor"]``, spot clusters in ``obs["original_clusters"]``
    and, if present, the published AXIS alignment to section 0 in
    ``obsm["spatial_axis"]``.

    The section depth is stored in ``uns["depth_um"]``.
    """
    import anndata as ad

    directory = Path(directory)
    out = []
    for i in sections:
        path = directory / f"{patient}_{i}.h5ad"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Download the sample described in the README (Quickstart).")
        adata = ad.read_h5ad(path)
        adata.uns["depth_um"] = float(SCC_DEPTHS_UM[i]) if i < len(SCC_DEPTHS_UM) else float(10 * i)
        adata.uns["sample_name"] = f"patient{patient}_section{i}"
        out.append(adata)
    return out


def _hex_lattice(canvas: int, spacing: float) -> np.ndarray:
    pts = []
    sy = spacing * np.sqrt(3) / 2
    for r, y in enumerate(np.arange(spacing / 2, canvas - spacing / 2, sy)):
        xs = np.arange(spacing / 2 + (spacing / 2 if r % 2 else 0), canvas - spacing / 2, spacing)
        pts.append(np.column_stack([xs, np.full_like(xs, y)]))
    return np.vstack(pts)


def toy_series(
    n_samples: int = 3,
    canvas_size: int = 128,
    n_genes: int = 40,
    spot_spacing: float = 4.0,
    deformation: float = 3.0,
    seed: int = 0,
) -> List:
    """Simulate ``n_samples`` deformed sections of a layered tissue.

    The tissue is an ellipse with three concentric domains; every section is a
    smooth random deformation of it. Spots on a hexagonal lattice carry Poisson
    counts with domain-specific programmes and a smooth gradient gene
    (``"gradient"``), ``obs["domain"]`` labels, canvas coordinates in
    ``obsm["spatial_canvas"]`` and an RGB morphology canvas in
    ``uns["axis"]["image"]``, i.e. the data are ready for
    [`axis.pp.build_pseudoimages`][axis.pp.build_pseudoimages].
    """
    import anndata as ad
    import pandas as pd
    from scipy.ndimage import gaussian_filter, map_coordinates

    from .pp.image import random_deform_field

    rng = np.random.default_rng(seed)
    H = W = int(canvas_size)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    c, ry, rx = H / 2.0, 0.40 * H, 0.32 * W
    programmes = rng.gamma(0.6, 1.0, size=(3, n_genes)) * 4
    programmes[:, :6] = np.array([[8, 0.2, 0.2], [0.2, 8, 0.2], [0.2, 0.2, 8]]).repeat(2, axis=1)
    programmes[:, -1] = 0.0
    lattice = _hex_lattice(H, spot_spacing)
    out = []
    for k in range(n_samples):
        dy, dx = random_deform_field(
            H, W, grid_res=max(8, H // 8), smooth_sigma=2.0, amplitude=deformation * (k + 1) / n_samples, seed=seed + 17 * k
        )
        shift = np.array([1.5 * k, -1.0 * k], dtype=np.float32)
        # radial "depth" coordinate of the deformed tissue on the canvas grid
        u = (xx + dx - c - shift[0]) / rx
        v = (yy + dy - c - shift[1]) / ry
        r = np.sqrt(u**2 + v**2) * (1.0 + 0.05 * k)
        domain_img = np.digitize(r, [0.45, 0.75, 1.0])  # 0, 1, 2 inside; 3 outside
        palette = np.array([[180, 90, 160], [220, 140, 190], [240, 200, 225], [250, 250, 250]], dtype=np.float32)
        img = palette[domain_img] + rng.normal(0, 10, (H, W, 1))
        img = np.stack([gaussian_filter(img[..., ch], 0.8) for ch in range(3)], axis=-1)
        img = np.clip(img, 0, 255).astype(np.uint8)

        sample_r = map_coordinates(r, [lattice[:, 1], lattice[:, 0]], order=1)
        keep = sample_r < 1.0
        coords = lattice[keep]
        dom = np.digitize(sample_r[keep], [0.45, 0.75])
        rates = programmes[dom].copy()
        rates[:, -1] = 6.0 * (coords[:, 0] / W)
        depth = rng.uniform(0.6, 1.4, size=(len(coords), 1))
        X = rng.poisson(rates * depth).astype(np.float32)

        adata = ad.AnnData(X=X)
        adata.var_names = [f"gene{g}" for g in range(n_genes - 1)] + ["gradient"]
        adata.obs_names = [f"s{k}_spot{i}" for i in range(len(coords))]
        adata.obs["domain"] = pd.Categorical([f"D{d}" for d in dom])
        adata.obsm["spatial_canvas"] = coords.astype(np.float32)
        adata.uns["axis"] = {"image": img, "canvas_size": H}
        adata.uns["position"] = float(10 * k)
        out.append(adata)
    return out
