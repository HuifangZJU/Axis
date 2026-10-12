"""High-level tools (scanpy-style API).

A typical AXIS session::

    import anndata as ad, axis

    adatas = [ad.read_h5ad(f"10_{i}.h5ad") for i in range(3)]
    for a in adatas:
        axis.pp.prepare_canvas(a, spatial_key="spatial_image_coor", canvas_size=512)
    axis.pp.build_pseudoimages(adatas, n_comps=10)

    fields = axis.tl.fit_transition(adatas, epochs=200)           # transition fields
    vt = axis.tl.generate(adatas, fields, positions=[0, 10, 20],   # virtual tissue at 10 µm
                          query=10, anchors=[0, 2], attr_keys=["original_clusters"])
    matched = axis.tl.match(adatas, fields, reference=0, reconstruct=True)

All functions take a list of AnnData objects (the measured samples, ordered as
in ``positions``) and refer to samples by their index in that list.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from itertools import permutations
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .generation.evidence import (
    build_query_evidence,
    compute_axis_kernel_weights,
    ensure_same_genes_keep_order,
    get_anchor_pair_for_query,
    matrix_log1p,
    relative_position_on_axis,
    stack_attribute_evidence,
    subset_genes,
)
from .generation.generate import generate_virtual_tissue_fixed_grid
from .generation.support import compute_dense_tissue_mask_on_any_grid, make_common_grid_from_coords
from .registration.warp import ensure_flow_2hw, find_inverse_warp_coords
from .settings import log, settings

__all__ = [
    "TransitionFields",
    "fit_transition",
    "make_grid",
    "generate",
    "match",
    "integrate",
    "contrast",
    "predict_modality",
]

Pair = Tuple[int, int]


# =============================================================================
# Transition fields container
# =============================================================================
class TransitionFields(Mapping):
    """Pairwise transition fields of a series of samples.

    ``fields[(fixed, moving)]`` is the ``(2, H, W)`` ``(dy, dx)`` field that
    registers sample ``moving`` onto sample ``fixed`` on the shared canvas.
    ``fields[(i, i)]`` is the identity (zeros).
    """

    def __init__(self, fields: Dict[Pair, np.ndarray], canvas_size: int, model=None, history: Optional[dict] = None):
        self._fields = {
            tuple(map(int, k)): np.asarray(ensure_flow_2hw(np.asarray(v)), dtype=np.float32) for k, v in fields.items()
        }
        self.canvas_size = int(canvas_size)
        self.model = model
        self.history = history or {}

    def __getitem__(self, key: Pair) -> np.ndarray:
        fixed, moving = map(int, key)
        if fixed == moving:
            return np.zeros((2, self.canvas_size, self.canvas_size), dtype=np.float32)
        return self._fields[(fixed, moving)]

    def __iter__(self) -> Iterator[Pair]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __contains__(self, key) -> bool:
        fixed, moving = map(int, key)
        return fixed == moving or (fixed, moving) in self._fields

    def __repr__(self) -> str:
        return f"TransitionFields(n_pairs={len(self)}, canvas_size={self.canvas_size})"

    @property
    def samples(self) -> List[int]:
        return sorted({i for pair in self._fields for i in pair})

    def transport(self, coords: np.ndarray, fixed: int, moving: int, alpha: float = 1.0) -> np.ndarray:
        """Transport moving-sample coordinates with ``alpha * D_{moving -> fixed}``."""
        if fixed == moving or alpha == 0:
            return np.asarray(coords, dtype=np.float32).copy()
        flow = (float(alpha) * self[(fixed, moving)]).astype(np.float32)
        return np.asarray(find_inverse_warp_coords(np.asarray(coords, dtype=np.float32), flow), dtype=np.float32)

    def save(self, directory: Union[str, os.PathLike]) -> Path:
        """Write ``{fixed}_{moving}_flow.npy`` files (the pipeline layout) and ``fields.json``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for (f, m), flow in self._fields.items():
            np.save(directory / f"{f}_{m}_flow.npy", flow)
        (directory / "fields.json").write_text(
            json.dumps({"canvas_size": self.canvas_size, "pairs": [list(p) for p in self._fields]}, indent=2)
        )
        return directory

    @classmethod
    def load(cls, directory: Union[str, os.PathLike], canvas_size: Optional[int] = None) -> TransitionFields:
        """Read all ``{fixed}_{moving}_flow.npy`` files from ``directory``."""
        directory = Path(directory)
        fields = {}
        for path in sorted(directory.glob("*_*_flow.npy")):
            parts = path.name[: -len("_flow.npy")].split("_")
            if len(parts) == 2 and all(p.lstrip("-").isdigit() for p in parts):
                fields[(int(parts[0]), int(parts[1]))] = np.load(path)
        if not fields:
            raise FileNotFoundError(f"No '*_*_flow.npy' files in {directory}")
        if canvas_size is None:
            canvas_size = next(iter(fields.values())).shape[-1]
        return cls(fields, canvas_size)


# =============================================================================
# Transition-field learning
# =============================================================================
def _axis_inputs(adatas, uns_key: str):
    images, pseudo, masks = [], [], []
    for i, adata in enumerate(adatas):
        store = adata.uns.get(uns_key, {})
        missing = [k for k in ("image", "pseudoimage", "mask") if k not in store]
        if missing:
            raise KeyError(
                f"Sample {i} lacks uns[{uns_key!r}]{missing}; run axis.pp.prepare_canvas and axis.pp.build_pseudoimages."
            )
        images.append(np.asarray(store["image"]))
        pseudo.append(np.asarray(store["pseudoimage"]))
        masks.append(np.asarray(store["mask"]))
    sizes = {im.shape[:2] for im in images}
    if len(sizes) != 1:
        raise ValueError(f"All canvases must have the same size, got {sizes}")
    return images, pseudo, masks


def fit_transition(
    adatas: Sequence,
    *,
    pairs: Optional[Iterable[Pair]] = None,
    checkpoint: Optional[Union[str, os.PathLike]] = None,
    model=None,
    epochs: int = 1000,
    max_steps: Optional[int] = None,
    lr: float = 1e-4,
    fusion: str = "transformer",
    lambda_morph: float = 1.0,
    lambda_mol: float = 1.0,
    lambda_smooth: float = 1.0,
    morph_win: int = 21,
    mol_win: int = 15,
    device: Optional[str] = None,
    seed: Optional[int] = 0,
    out_dir: Optional[Union[str, os.PathLike]] = None,
    save_every: Optional[int] = None,
    uns_key: str = "axis",
) -> TransitionFields:
    """Learn bidirectional transition fields between samples (Methods Eqs. (3)-(23)).

    A single multimodal network is trained on all ordered sample pairs (or the
    given ``pairs``) and then applied to every ordered pair.

    Parameters
    ----------
    adatas
        Samples prepared with :func:`axis.pp.prepare_canvas` and
        :func:`axis.pp.build_pseudoimages`.
    pairs
        Ordered ``(fixed, moving)`` pairs used for training and prediction.
        Defaults to all ordered pairs.
    checkpoint
        Optional checkpoint to start from (AXIS ≥ 1.0 format or an original
        ``state_dict``). With ``epochs=0`` the checkpoint is used as is.
    model
        A pre-built :class:`axis.registration.TransitionFieldNet` (overrides
        ``fusion`` and ``checkpoint``).
    epochs, max_steps, lr
        Training schedule (Adam, batch size 1). ``max_steps`` caps the total
        number of optimisation steps, e.g. for quick runs on CPU.
    fusion
        ``"transformer"`` (Visium-sized inputs) or ``"linear"`` (large inputs).
    lambda_morph, lambda_mol, lambda_smooth, morph_win, mol_win
        Loss weights and local-NCC windows (Eqs. (20)-(22)).
    out_dir, save_every
        Optional checkpoint directory / frequency (epochs).

    Returns
    -------
    :class:`TransitionFields` with the trained ``model`` and loss ``history``.
    """
    import torch

    from .registration.data import PairDataset
    from .registration.inference import predict_field
    from .registration.losses import RegistrationLoss
    from .registration.model import TransitionFieldNet, load_state_dict_flexible
    from .registration.train import train_transition_network

    device = settings.resolve_device(device)
    images, pseudo, masks = _axis_inputs(adatas, uns_key)
    n = len(adatas)
    pairs = [tuple(int(i) for i in p) for p in pairs] if pairs is not None else list(permutations(range(n), 2))
    in_channels = 1 if images[0].ndim == 2 else images[0].shape[2]
    n_comp = pseudo[0].shape[-1]

    if model is None:
        model = TransitionFieldNet(in_channels=in_channels, in_gene_channels=n_comp, fusion_module=fusion)
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            loaded, skipped = load_state_dict_flexible(model, state)
            log(f"[axis] loaded {len(loaded)} tensors from {checkpoint} ({len(skipped)} skipped)")

    dataset = PairDataset(images, pseudo, masks, pairs=pairs)
    history: dict = {}
    if epochs and (max_steps is None or max_steps > 0):
        loss_fn = RegistrationLoss(
            lambda_morph=lambda_morph,
            lambda_mol=lambda_mol,
            lambda_smooth=lambda_smooth,
            morph_win=morph_win,
            mol_win=mol_win,
            n_mol_channels=n_comp,
        )
        history = train_transition_network(
            model,
            dataset,
            epochs=epochs,
            lr=lr,
            loss_fn=loss_fn,
            device=device,
            max_steps=max_steps,
            out_dir=out_dir,
            save_every=save_every,
            seed=seed,
        )

    fields = {}
    for f, m in pairs:
        item = dataset[pairs.index((f, m))]
        fields[(f, m)] = predict_field(
            model, item["fixed_image"], item["moving_image"], item["fixed_gene"], item["moving_gene"], device=device
        )
    return TransitionFields(fields, canvas_size=images[0].shape[0], model=model, history=history)


# =============================================================================
# Helpers shared by generation and matching
# =============================================================================
def _expr(adata, layer: Optional[str]):
    return adata.layers[layer] if layer is not None else adata.X


def _coords(adata, spatial_key: str) -> np.ndarray:
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm[{spatial_key!r}] missing; run axis.pp.prepare_canvas first.")
    return np.asarray(adata.obsm[spatial_key], dtype=np.float32)[:, :2]


def _attrs(adata, attr_keys: Optional[Sequence[str]]) -> Optional[pd.DataFrame]:
    if not attr_keys:
        return None
    missing = [k for k in attr_keys if k not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing obs columns: {missing}")
    return adata.obs.loc[:, list(attr_keys)].copy().reset_index(drop=True)


def make_grid(kind: Union[str, np.ndarray] = "visium", grid_size: int = 40, canvas_size: int = 512) -> np.ndarray:
    """Output support on the canvas: ``"visium"`` (hexagonal), ``"pixel"`` (square) or explicit ``(M, 2)`` points."""
    if not isinstance(kind, str):
        return np.asarray(kind, dtype=np.float32)
    grid, _ = make_common_grid_from_coords(grid_size, type=kind, max_x=canvas_size, max_y=canvas_size)
    return grid


# =============================================================================
# Virtual-tissue generation
# =============================================================================
def generate(
    adatas: Sequence,
    fields: TransitionFields,
    positions: Union[Sequence[float], Mapping[int, float]],
    query: float,
    *,
    anchors: Optional[Sequence[int]] = None,
    grid: Union[str, np.ndarray] = "visium",
    grid_size: int = 40,
    spatial_key: str = "spatial_canvas",
    layer: Optional[str] = None,
    genes: Optional[Sequence[str]] = None,
    attr_keys: Optional[Sequence[str]] = None,
    attr_schema: Optional[Mapping[str, str]] = None,
    spatial_bandwidth: Optional[float] = None,
    attr_spatial_bandwidth: Optional[float] = None,
    axis_bandwidth: Optional[float] = None,
    max_dist: float = 8.0,
    knn: int = 48,
    min_neighbors: int = 1,
    radius: Optional[float] = None,
    use_quality_weight: bool = True,
    log1p: bool = False,
):
    """Generate a virtual tissue at an unmeasured position ``query`` (Methods Eqs. (24)-(35)).

    Parameters
    ----------
    adatas, fields
        Measured samples and their transition fields.
    positions
        Position of every sample along the biological axis (list aligned with
        ``adatas`` or ``{index: position}``).
    query
        Queried position ``tau_q``.
    anchors
        Candidate anchor samples (default: all). The two anchors bracketing
        ``query`` are used.
    grid, grid_size
        Output support: ``"visium"``/``"pixel"`` lattice with about
        ``grid_size**2`` locations, or explicit ``(M, 2)`` canvas coordinates.
    layer
        Molecular values to generate (default ``X``; counts are expected when
        ``use_quality_weight`` is on).
    genes
        Optional subset of features.
    attr_keys, attr_schema
        ``obs`` annotations to generate (e.g. ``["original_clusters"]``).
    spatial_bandwidth, attr_spatial_bandwidth
        ``sigma_x`` for molecules / annotations, in canvas pixels.
    axis_bandwidth
        ``sigma_tau``; defaults to one third of the anchor spacing.
    max_dist
        Tissue-support threshold (``max_distance``, canvas pixels; 8 by default).

    Returns
    -------
    AnnData on the output support with generated ``X``, annotations, support
    diagnostics in ``obs`` and provenance in ``uns["axis_generation"]``.
    """
    pos = dict(positions) if isinstance(positions, Mapping) else {i: float(p) for i, p in enumerate(positions)}
    anchors = list(anchors) if anchors is not None else sorted(pos)
    a, b = get_anchor_pair_for_query(query, pos, anchors)
    alpha = relative_position_on_axis(query, pos[a], pos[b])
    weights, bw = compute_axis_kernel_weights([pos[a], pos[b]], query, bandwidth=axis_bandwidth)
    log(f"[axis] generate query={query} anchors=({a},{b}) alpha={alpha:.3f} weights={np.round(weights, 3).tolist()}")

    ad_a, ad_b = adatas[a], adatas[b]
    Xa, Xb, var = ensure_same_genes_keep_order(
        _expr(ad_a, layer), np.asarray(ad_a.var_names), _expr(ad_b, layer), np.asarray(ad_b.var_names)
    )
    var, Xa, Xb = subset_genes(var, genes, Xa, Xb)
    if log1p:
        Xa, Xb = matrix_log1p(Xa), matrix_log1p(Xb)

    grid_pts = make_grid(grid, grid_size, fields.canvas_size)
    support, coords_cat, X_cat, point_weights = build_query_evidence(
        alpha,
        grid_pts,
        _coords(ad_a, spatial_key),
        _coords(ad_b, spatial_key),
        Xa,
        Xb,
        fields[(b, a)],
        fields[(a, b)],
        weights,
        max_dist_px=max_dist,
    )
    attrs = stack_attribute_evidence(_attrs(ad_a, attr_keys), _attrs(ad_b, attr_keys)) if attr_keys else None
    out = generate_virtual_tissue_fixed_grid(
        coords=coords_cat,
        X=X_cat,
        grid_pts=grid_pts,
        var_names=var,
        attrs=attrs,
        attr_schema=attr_schema,
        grid_support_mask=support,
        point_weights=point_weights,
        spatial_bandwidth=spatial_bandwidth,
        attr_spatial_bandwidth=attr_spatial_bandwidth,
        use_gene_weight=use_quality_weight,
        knn=knn,
        min_neighbors=min_neighbors,
        radius=radius,
    )
    out.uns["axis_generation"] = {
        "query": float(query),
        "anchors": [int(a), int(b)],
        "anchor_positions": [float(pos[a]), float(pos[b])],
        "alpha": float(alpha),
        "anchor_weights": [float(w) for w in weights],
        "axis_bandwidth": float(bw),
        "max_dist": float(max_dist),
        "n_evidence": int(coords_cat.shape[0]),
    }
    return out


# =============================================================================
# Spatially matched analysis
# =============================================================================
def match(
    adatas: Sequence,
    fields: TransitionFields,
    reference: int = 0,
    *,
    spatial_key: str = "spatial_canvas",
    key_added: str = "spatial_axis",
    reconstruct: bool = False,
    grid: Optional[Union[str, np.ndarray]] = None,
    grid_size: int = 40,
    layer: Optional[str] = None,
    genes: Optional[Sequence[str]] = None,
    attr_keys: Optional[Sequence[str]] = None,
    attr_schema: Optional[Mapping[str, str]] = None,
    spatial_bandwidth: Optional[float] = None,
    attr_spatial_bandwidth: Optional[float] = None,
    max_dist: float = 8.0,
    knn: int = 48,
    min_neighbors: int = 1,
    use_quality_weight: bool = True,
):
    """Place samples in the anatomical frame of ``reference`` (Methods, "Spatially matched analysis").

    Always writes the transported coordinates to ``obsm[key_added]`` of every
    sample. With ``reconstruct=True`` every sample is additionally
    reconstructed on a common support in the reference frame (by default the
    reference sample's own locations; ``beta_r = 1``), giving location-matched
    AnnData objects that can be compared directly (see :func:`contrast`).

    Returns
    -------
    ``None``, or the list of reconstructed AnnData objects (same order as
    ``adatas``) when ``reconstruct=True``.
    """
    for i, adata in enumerate(adatas):
        adata.obsm[key_added] = fields.transport(_coords(adata, spatial_key), reference, i)
    if not reconstruct:
        return None

    from .matching.reference import reconstruct_on_reference

    grid_pts = _coords(adatas[reference], spatial_key) if grid is None else make_grid(grid, grid_size, fields.canvas_size)
    common = None
    for adata in adatas:
        names = np.asarray(adata.var_names)
        common = names if common is None else common[np.isin(common, names)]
    if genes is not None:
        common = np.asarray([g for g in genes if g in set(common.tolist())])
    out = []
    for i, adata in enumerate(adatas):
        idx = pd.Index(adata.var_names).get_indexer(common)
        X = _expr(adata, layer)[:, idx]
        rec = reconstruct_on_reference(
            _coords(adata, spatial_key),
            X,
            fields[(reference, i)],
            grid_pts,
            var_names=common,
            attrs=_attrs(adata, attr_keys),
            attr_schema=attr_schema,
            max_dist_px=max_dist,
            spatial_bandwidth=spatial_bandwidth,
            attr_spatial_bandwidth=attr_spatial_bandwidth,
            knn=knn,
            min_neighbors=min_neighbors,
            use_gene_weight=use_quality_weight,
        )
        rec.uns["axis_matching"] = {"reference": int(reference), "source": int(i)}
        out.append(rec)
    return out


def integrate(
    adatas: Sequence,
    fields: TransitionFields,
    reference: int = 0,
    *,
    grid: Union[str, np.ndarray] = "visium",
    grid_size: int = 80,
    spatial_key: str = "spatial_canvas",
    layer: Optional[str] = None,
    genes: Optional[Sequence[str]] = None,
    attr_keys: Optional[Sequence[str]] = None,
    max_dist: float = 5.0,
    fuse_radius: float = 10.0,
    alpha: float = 0.6,
    keep_nuc_frac: float = 0.6,
    pca_dim: int = 32,
    outer_loops: int = 2,
    device: Optional[str] = None,
):
    """Population-level tissue representation (Methods Eqs. (36)-(39)).

    All samples are transported into the ``reference`` frame, pooled with equal
    total mass per sample, coupled to the reference support by sparse
    Sinkhorn transport and denoised by adaptive low-rank reconstruction.
    Defaults follow the DLPFC analysis.
    """
    import anndata as ad

    from .matching.population import integrate_population

    common = None
    for adata in adatas:
        names = np.asarray(adata.var_names)
        common = names if common is None else common[np.isin(common, names)]
    if genes is not None:
        common = np.asarray([g for g in genes if g in set(common.tolist())])

    coords, blocks, attrs, counts, sources = [], [], [], [], []
    for i, adata in enumerate(adatas):
        coords.append(fields.transport(_coords(adata, spatial_key), reference, i))
        X = _expr(adata, layer)[:, pd.Index(adata.var_names).get_indexer(common)]
        blocks.append(X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X)))
        attrs.append(_attrs(adata, attr_keys))
        counts.append(adata.n_obs)
        sources.append(np.full(adata.n_obs, i))
    coords_cat = np.vstack(coords).astype(np.float32)
    masses = np.concatenate([np.full(c, 1.0 / len(counts) / c) for c in counts])
    masses /= masses.sum()

    grid_pts = make_grid(grid, grid_size, fields.canvas_size)
    support = compute_dense_tissue_mask_on_any_grid(grid_pts, coords_cat, max_dist_px=max_dist, keep_largest=True)
    obs = pd.concat(attrs, ignore_index=True) if attr_keys else pd.DataFrame(index=np.arange(len(coords_cat)))
    obs.index = obs.index.astype(str)
    obs["source_sample"] = pd.Categorical(np.concatenate(sources).astype(str))
    pooled = ad.AnnData(X=sp.vstack(blocks).tocsr(), obs=obs, var=pd.DataFrame(index=pd.Index(common.astype(str), name="gene")))
    pooled.obsm["spatial"] = coords_cat

    out = integrate_population(
        pooled,
        fixed_grid_pts=grid_pts,
        grid_support_mask=support,
        point_masses=masses,
        fuse_radius=fuse_radius,
        alpha=alpha,
        keep_nuc_frac=keep_nuc_frac,
        pca_dim=pca_dim,
        outer_loops=outer_loops,
        device=settings.resolve_device(device),
        attr_schema={k: "categorical" for k in attr_keys} if attr_keys else None,
    )
    out.uns["axis_integration"] = {
        "reference": int(reference),
        "n_samples": len(adatas),
        "source_point_counts": [int(c) for c in counts],
    }
    return out


def contrast(
    adata_a,
    adata_b,
    genes: Optional[Sequence[str]] = None,
    *,
    normalize_total: bool = True,
    target_sum: Optional[float] = None,
    use_log1p: bool = True,
    domain_key: Optional[str] = None,
):
    """Location-wise molecular contrast ``a - b`` of two spatially matched tissues.

    Thin wrapper around :func:`axis.matching.contrast.compute_spatial_delta`;
    the inputs must share the same support (e.g. outputs of :func:`match` with
    ``reconstruct=True``).
    """
    from .matching.contrast import compute_spatial_delta

    kwargs = dict(genes=genes, normalize_total=normalize_total, use_log1p=use_log1p, domain_key=domain_key)
    if target_sum is not None:
        kwargs["target_sum"] = target_sum
    return compute_spatial_delta(adata_a, adata_b, **kwargs)


def predict_modality(X, Y, target_names: Sequence[str], **kwargs):
    """Cross-modality prediction from matched locations; see :func:`axis.matching.fit_cross_modality`."""
    from .matching.cross_modality import fit_cross_modality

    return fit_cross_modality(X, Y, target_names, **kwargs)
