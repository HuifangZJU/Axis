#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replace mouse-brain h5ad spatial coordinates from validation NPZ files.

Expected files
--------------
H5AD:
    $AXIS_DATA_DIR/mouse_brain/h5ads/0_[sampleid].h5ad
NPZ:
    $AXIS_DATA_DIR/mouse_brain/network_input/0_[sampleid]_validation.npz

For each matched sample, this script:
1. loads the original h5ad;
2. reads the replacement coordinates from npz["coord"];
3. replaces adata.obsm["spatial"];
4. saves the updated object as 0_[sampleid]_new.h5ad in the same h5ad directory;
5. plots Arc expression before and after coordinate replacement using a shared color scale.

Only spatial coordinates are changed. Expression values, obs, var, layers, and other
AnnData fields are preserved.
"""

from __future__ import annotations
from axis.settings import data_path

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
H5AD_DIR = Path(data_path("mouse_brain/h5ads"))
NPZ_DIR = Path(data_path("mouse_brain/network_input"))
SPATIAL_KEY = "spatial"
COORD_KEY = "coord"
MARKER_GENE = "Arc"
INVERT_Y = True
POINT_SIZE = 8.0
PLOT_DPI = 320
VMAX_PERCENTILE = 99.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def find_input_h5ads(h5ad_dir: Path, sample_ids: Optional[Iterable[str]] = None):
    """Return original 0_[sampleid].h5ad files, excluding previously generated *_new files."""
    if sample_ids:
        paths = [h5ad_dir / f"0_{sid}.h5ad" for sid in sample_ids]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing requested h5ad file(s):\n  " + "\n  ".join(missing))
        return paths

    paths = []
    pattern = re.compile(r"^0_(.+)\.h5ad$")
    for path in sorted(h5ad_dir.glob("0_*.h5ad")):
        if path.stem.endswith("_new"):
            continue
        if pattern.match(path.name):
            paths.append(path)
    return paths


def sample_id_from_h5ad(path: Path) -> str:
    match = re.match(r"^0_(.+)\.h5ad$", path.name)
    if match is None:
        raise ValueError(f"Unexpected h5ad filename: {path.name}")
    return match.group(1)


def load_replacement_coords(npz_path: Path, coord_key: str = COORD_KEY) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"Coordinate npz not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        if coord_key not in data:
            raise KeyError(
                f"{npz_path} does not contain key {coord_key!r}. "
                f"Available keys: {list(data.keys())}"
            )
        coords = np.asarray(data[coord_key], dtype=np.float32)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"Expected npz[{coord_key!r}] to have shape (n_obs, >=2), got {coords.shape}"
        )

    # obsm['spatial'] conventionally stores x/y in its first two columns.
    coords = coords[:, :2]

    if not np.isfinite(coords).all():
        raise ValueError(f"Replacement coordinates contain NaN or Inf: {npz_path}")

    return coords


def extract_gene_vector(adata: ad.AnnData, gene: str) -> tuple[str, np.ndarray]:
    """Return a case-insensitively resolved gene symbol and a dense 1D expression vector."""
    names = np.asarray(adata.var_names.astype(str))

    if gene in names:
        resolved = gene
    else:
        matches = names[np.char.upper(names.astype(str)) == gene.upper()]
        if len(matches) != 1:
            raise KeyError(
                f"Gene {gene!r} was not uniquely found in {adata.n_vars} features. "
                f"Case-insensitive matches: {matches.tolist()}"
            )
        resolved = str(matches[0])

    x = adata[:, resolved].X
    if sparse.issparse(x):
        values = x.toarray().reshape(-1)
    else:
        values = np.asarray(x).reshape(-1)

    values = values.astype(float, copy=False)
    return resolved, values


def robust_limits(values: np.ndarray, upper_percentile: float = VMAX_PERCENTILE):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanpercentile(finite, upper_percentile))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def plot_before_after(
    old_coords: np.ndarray,
    new_coords: np.ndarray,
    values: np.ndarray,
    gene: str,
    sample_id: str,
    out_png: Path,
    *,
    invert_y: bool = INVERT_Y,
    point_size: float = POINT_SIZE,
    dpi: int = PLOT_DPI,
) -> None:
    """Plot the same Arc expression values at old and replacement coordinates."""
    vmin, vmax = robust_limits(values)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), constrained_layout=True)

    panels = [
        (axes[0], old_coords, "Before"),
        (axes[1], new_coords, "After"),
    ]

    scatter = None
    for ax, coords, title in panels:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            s=point_size,
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(title, fontsize=11, fontweight="normal")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if invert_y:
            ax.invert_yaxis()

    fig.suptitle(f"Sample {sample_id} | {gene}", fontsize=12, fontweight="normal")
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label(f"{gene} expression", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def process_one_sample(
    h5ad_path: Path,
    npz_dir: Path,
    *,
    spatial_key: str = SPATIAL_KEY,
    coord_key: str = COORD_KEY,
    marker_gene: str = MARKER_GENE,
    overwrite: bool = False,
) -> Path:
    sample_id = sample_id_from_h5ad(h5ad_path)
    npz_path = npz_dir / f"0_{sample_id}_validation.npz"
    out_h5ad = h5ad_path.with_name(f"{h5ad_path.stem}_new.h5ad")
    out_plot = h5ad_path.with_name(f"{h5ad_path.stem}_Arc_spatial_before_after.png")

    if out_h5ad.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {out_h5ad}\n"
            "Use --overwrite to replace it."
        )

    print(f"[INFO] sample {sample_id}")
    print(f"       h5ad: {h5ad_path}")
    print(f"       npz:   {npz_path}")

    adata = ad.read_h5ad(h5ad_path)

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"{h5ad_path} does not contain adata.obsm[{spatial_key!r}]. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    old_coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    if old_coords.ndim != 2 or old_coords.shape[1] < 2:
        raise ValueError(
            f"Existing adata.obsm[{spatial_key!r}] has invalid shape {old_coords.shape}"
        )
    old_coords_xy = old_coords[:, :2].copy()

    new_coords = load_replacement_coords(npz_path, coord_key=coord_key)

    if new_coords.shape[0] != adata.n_obs:
        raise ValueError(
            f"Coordinate row count mismatch for sample {sample_id}: "
            f"h5ad n_obs={adata.n_obs}, npz[{coord_key!r}] rows={new_coords.shape[0]}. "
            "The script will not reorder cells automatically."
        )

    resolved_gene, values = extract_gene_vector(adata, marker_gene)

    # Diagnostic plot is generated from the exact same expression vector before
    # modifying the AnnData object; only coordinates differ between panels.
    plot_before_after(
        old_coords=old_coords_xy,
        new_coords=new_coords,
        values=values,
        gene=resolved_gene,
        sample_id=sample_id,
        out_png=out_plot,
    )

    # Preserve any extra spatial columns only if present. The requested x/y
    # coordinates are replaced by the new coordinates.
    if old_coords.shape[1] == 2:
        adata.obsm[spatial_key] = new_coords
    else:
        updated = old_coords.copy()
        updated[:, :2] = new_coords
        adata.obsm[spatial_key] = updated

    adata.write_h5ad(out_h5ad)

    print(f"[OK] saved updated h5ad: {out_h5ad}")
    print(f"[OK] saved Arc check plot: {out_plot}")
    return out_h5ad


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replace adata.obsm['spatial'] using 0_[sampleid]_validation.npz['coord'] "
            "and save *_new.h5ad files."
        )
    )
    parser.add_argument("--h5ad-dir", type=Path, default=H5AD_DIR)
    parser.add_argument("--npz-dir", type=Path, default=NPZ_DIR)
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help=(
            "Optional sample IDs to process, e.g. --sample-ids 1 2 3. "
            "If omitted, all matching 0_*.h5ad files are processed."
        ),
    )
    parser.add_argument("--spatial-key", default=SPATIAL_KEY)
    parser.add_argument("--coord-key", default=COORD_KEY)
    parser.add_argument("--marker-gene", default=MARKER_GENE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.h5ad_dir.exists():
        raise FileNotFoundError(f"h5ad directory not found: {args.h5ad_dir}")
    if not args.npz_dir.exists():
        raise FileNotFoundError(f"npz directory not found: {args.npz_dir}")

    h5ad_paths = find_input_h5ads(args.h5ad_dir, sample_ids=args.sample_ids)
    if not h5ad_paths:
        raise FileNotFoundError(
            f"No original files matching 0_[sampleid].h5ad were found in {args.h5ad_dir}"
        )

    print(f"[INFO] Found {len(h5ad_paths)} sample(s).")

    completed = 0
    for h5ad_path in h5ad_paths:
        process_one_sample(
            h5ad_path=h5ad_path,
            npz_dir=args.npz_dir,
            spatial_key=args.spatial_key,
            coord_key=args.coord_key,
            marker_gene=args.marker_gene,
            overwrite=args.overwrite,
        )
        completed += 1

    print(f"[DONE] Updated {completed} sample(s).")


if __name__ == "__main__":
    main()
