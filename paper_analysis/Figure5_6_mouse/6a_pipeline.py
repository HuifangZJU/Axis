#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combined 3D vertical pipeline figures for AD-vs-WT disease analysis.

This revision simplifies the previous script by:
1. removing unnecessary path-construction / alternate-coordinate controls,
2. using distinct colormaps for AD and WT state maps,
3. merging AD/WT views into paired subplot figures, and
4. keeping virtual states consistently dashed wherever they appear.

Final outputs
-------------
Figure 1  AD raw states + WT raw states.
Figure 2  AD common-frame states + WT common-frame validation states.
Figure 3  AD common-frame virtual-generation states + WT common-frame
          virtual-generation states.
Figure 4  AD matched states + spatial AD-WT delta + WT matched states
          in a shared 1 x 3 age-aligned layout.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D


# ============================================================
# USER CONFIGURATION
# ============================================================
GENE_NAME = "Gfap"
COORD_KEY = "spatial"
VALID_KEY = "valid"

OUTDIR = (
    "/media/huifang/data/registration/paper_figure/"
    "Figure6_disease/pipeline_3d_vertical"
)

# ------------------------------------------------------------
# Direct input file specifications.
# ------------------------------------------------------------
AD_RAW_SPECS = [
    {
        "age": 2.5,
        "label": "2.5m",
        "h5ad_path": "/media/huifang/data/registration/mouse_brain/h5ads/0_0.h5ad",
        "coord_path": "/media/huifang/data/registration/mouse_brain/network_input/0_0_validation.npz",
    },
    {
        "age": 5.7,
        "label": "5.7m",
        "h5ad_path": "/media/huifang/data/registration/mouse_brain/h5ads/0_1.h5ad",
        "coord_path": "/media/huifang/data/registration/mouse_brain/network_input/0_1_validation.npz",
    },
    {
        "age": 17.9,
        "label": "17.9m",
        "h5ad_path": "/media/huifang/data/registration/mouse_brain/h5ads/0_2.h5ad",
        "coord_path": "/media/huifang/data/registration/mouse_brain/network_input/0_2_validation.npz",
    },
]

WT_RAW_SPECS = [
    {
        "age": 2.5,
        "label": "2.5m",
        "h5ad_path": "/media/huifang/data/registration/mouse_brain/h5ads/0_3.h5ad",
        "coord_path": "/media/huifang/data/registration/mouse_brain/network_input/0_3_validation.npz",
    },
    {
        "age": 5.7,
        "label": "5.7m",
        "h5ad_path": "/media/huifang/data/registration/mouse_brain/h5ads/0_4.h5ad",
        "coord_path": "/media/huifang/data/registration/mouse_brain/network_input/0_4_validation.npz",
    },
    {
        "age": 13.4,
        "label": "13.4m",
        "h5ad_path": "/media/huifang/data/registration/mouse_brain/h5ads/0_5.h5ad",
        "coord_path": "/media/huifang/data/registration/mouse_brain/network_input/0_5_validation.npz",
    },
]

COMMON_FRAME_ROOT = (
    "/media/huifang/data/registration/mouse_brain/"
    "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf"
)

AD_COMMON_FRAME: Dict[str, dict] = {
    "2.5m": {
        "age": 2.5,
        "label": "2.5m",
        "h5ad_path": os.path.join(COMMON_FRAME_ROOT, "virtual_Disease_time_2.500.h5ad"),
    },
    "5.7m": {
        "age": 5.7,
        "label": "5.7m",
        "h5ad_path": os.path.join(COMMON_FRAME_ROOT, "virtual_Disease_time_5.700.h5ad"),
    },
    "13.4m": {
        "age": 13.4,
        "label": "13.4m",
        "h5ad_path": os.path.join(COMMON_FRAME_ROOT, "virtual_Disease_time_13.400.h5ad"),
    },
    "17.9m": {
        "age": 17.9,
        "label": "17.9m",
        "h5ad_path": os.path.join(COMMON_FRAME_ROOT, "virtual_Disease_time_17.900.h5ad"),
    },
}

WT_COMMON_FRAME: Dict[str, dict] = {
    "2.5m": {
        "age": 2.5,
        "label": "2.5m",
        "h5ad_path": os.path.join(COMMON_FRAME_ROOT, "virtual_Disease_time_102.500.h5ad"),
    },
    "5.7m": {
        "age": 5.7,
        "label": "5.7m",
        "h5ad_path": os.path.join(COMMON_FRAME_ROOT, "virtual_Disease_time_105.700.h5ad"),
    },
    "13.4m": {
        "age": 13.4,
        "label": "13.4m",
        "h5ad_path": os.path.join(COMMON_FRAME_ROOT, "virtual_Disease_time_113.400.h5ad"),
    },
}

DELTA_DIR = (
    "/media/huifang/data/registration/paper_analysis/"
    "Figure56_mouse/6_trend_heatmap_hybrid/saved_spatial_delta_maps"
)
DELTA_SPECS: Dict[str, dict] = {
    "2.5m": {
        "age": 2.5,
        "label": "2.5m",
        "h5ad_path": os.path.join(DELTA_DIR, "spatial_delta_2.5m.h5ad"),
    },
    "5.7m": {
        "age": 5.7,
        "label": "5.7m",
        "h5ad_path": os.path.join(DELTA_DIR, "spatial_delta_5.7m.h5ad"),
    },
    "13.4m": {
        "age": 13.4,
        "label": "13.4m",
        "h5ad_path": os.path.join(DELTA_DIR, "spatial_delta_13.4m.h5ad"),
    },
}


# ============================================================
# PLOTTING CONFIGURATION
# ============================================================
FIG_DPI = 400
FIGURE_MODE = "show"  # save / show / both
SHOW_BLOCK = True

FIXED_BOX = (0.0, 1024.0, 0.0, 1024.0)
SWAP_XY = True

# Cohort-specific spatial orientation controls.
AD_INVERT_X = True
AD_INVERT_Y = False
WT_INVERT_X = False
WT_INVERT_Y = False

# Delta maps can be controlled independently as well.
DELTA_INVERT_X = True
DELTA_INVERT_Y = False

STACK_GAP = 300.0
SPATIAL_DISPLAY_SCALE = 0.70

POINT_SIZE_STATE = 1.2
POINT_SIZE_DELTA = 1.4
MAX_POINTS_PER_PLANE = 30000
STATE_ALPHA = 0.6
DELTA_ALPHA = 0.76

# Colormaps.
AD_STATE_CMAP = "Reds"
WT_STATE_CMAP = "Blues"  # elegant, clean, and high-contrast against AD reds
DELTA_CMAP = "coolwarm"

REAL_FRAME_COLOR = "#9A9A9A"
VIRTUAL_FRAME_COLOR = "#888888"
TIME_AXIS_COLOR = "#2F2F2F"
REAL_LABEL_COLOR = "#1F1F1F"
VIRTUAL_LABEL_COLOR = "#7C7C7C"

SOLID_LW = 1.4
DASHED_LW = 1.45
DASHED_STYLE = (0, (4.0, 2.4))

TIME_AXIS_LW = 1.8
TIME_ARROW_LENGTH_RATIO = 0.045
TIME_POINT_SIZE = 68.0
TIME_POINT_EDGE_COLOR = "white"
TIME_POINT_EDGE_WIDTH = 0.9

# Independent camera controls.
AD_VIEW_ELEV = 25.0
AD_VIEW_AZIM = -60.0
WT_VIEW_ELEV = 25.0
WT_VIEW_AZIM = -120.0
DELTA_VIEW_ELEV = 25.0
DELTA_VIEW_AZIM = -60.0


# ============================================================
# DATA STRUCTURE
# ============================================================
@dataclass
class SliceRecord:
    age: float
    label: str
    path: str
    coords: np.ndarray
    values: np.ndarray
    is_virtual: bool = False


# ============================================================
# BASIC HELPERS
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finalize_figure(fig, save_path: str) -> None:
    mode = str(FIGURE_MODE).lower()
    if mode not in {"save", "show", "both"}:
        raise ValueError(f"Invalid FIGURE_MODE: {mode!r}")

    if mode in {"save", "both"}:
        ensure_dir(os.path.dirname(save_path) or ".")
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        print(f"[SAVE] {save_path}")

    if mode in {"show", "both"}:
        plt.show(block=SHOW_BLOCK)

    plt.close(fig)


def to_dense_1d(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=float).ravel()


def validate_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected spatial coordinates (N,2+), got {coords.shape}")
    return coords[:, :2].copy()


def transform_coords(
    coords: np.ndarray,
    *,
    invert_x: bool,
    invert_y: bool,
) -> np.ndarray:
    """Apply shared x/y swap plus cohort-specific x/y inversion."""
    coords = validate_coords(coords)
    if SWAP_XY:
        coords = coords[:, [1, 0]]

    xmin, xmax, ymin, ymax = FIXED_BOX
    if invert_x:
        coords[:, 0] = xmin + xmax - coords[:, 0]
    if invert_y:
        coords[:, 1] = ymin + ymax - coords[:, 1]
    return coords


def sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    idx = np.arange(n, dtype=int)
    if n <= max_points:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_points, replace=False))


def stack_positions(n: int) -> np.ndarray:
    return -np.arange(n, dtype=float) * STACK_GAP


def build_shared_age_scaffold(
    left_records: Sequence[SliceRecord],
    right_records: Sequence[SliceRecord],
    reference_ages: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build one common schematic z scaffold for a paired AD/WT figure.

    If ``reference_ages`` is supplied, those ages define the COMPLETE scaffold,
    including ages that are absent from one side. This is important for the
    observed/common-frame paired figures: 13.4m and 17.9m must occupy different
    vertical positions even though each is present in only one cohort.

    The scaffold uses equal schematic spacing between ordered ages. It therefore
    emphasizes progression stages rather than literal elapsed-month distance.
    """
    if reference_ages is None:
        ages = {float(r.age) for r in left_records} | {float(r.age) for r in right_records}
        shared_ages = np.asarray(sorted(ages), dtype=float)
    else:
        shared_ages = np.asarray(sorted({float(a) for a in reference_ages}), dtype=float)

    if shared_ages.size == 0:
        empty = np.asarray([], dtype=float)
        return shared_ages, empty, empty, empty

    shared_z = stack_positions(len(shared_ages))
    age_to_z = {round(float(age), 6): float(z) for age, z in zip(shared_ages, shared_z)}

    def map_records(records: Sequence[SliceRecord]) -> np.ndarray:
        out = []
        for record in records:
            key = round(float(record.age), 6)
            if key not in age_to_z:
                raise ValueError(
                    f"Age {record.age:g}m is missing from the shared reference scaffold "
                    f"{shared_ages.tolist()}"
                )
            out.append(age_to_z[key])
        return np.asarray(out, dtype=float)

    return shared_ages, shared_z, map_records(left_records), map_records(right_records)


def robust_positive_vmax(arrays: Sequence[np.ndarray], q: float = 99.0) -> float:
    x = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    return max(float(np.percentile(x, q)), 1e-8)


def robust_symmetric_vmax(arrays: Sequence[np.ndarray], q: float = 99.0) -> float:
    x = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    return max(float(np.percentile(np.abs(x), q)), 1e-8)


# ============================================================
# DATA LOADERS
# ============================================================
def extract_state_gene(ad: sc.AnnData, gene: str) -> np.ndarray:
    if gene not in ad.var_names:
        raise KeyError(f"Gene {gene!r} not found")
    values = to_dense_1d(ad[:, [gene]].X)
    return np.log1p(np.clip(values, a_min=0.0, a_max=None))


def load_raw_record(spec: dict, gene: str, *, invert_x: bool, invert_y: bool) -> SliceRecord:
    h5ad_path = str(spec["h5ad_path"]).strip()
    coord_path = str(spec["coord_path"]).strip()
    if not os.path.isfile(h5ad_path):
        raise FileNotFoundError(h5ad_path)
    if not os.path.isfile(coord_path):
        raise FileNotFoundError(coord_path)

    ad = sc.read_h5ad(h5ad_path)
    values = extract_state_gene(ad, gene)
    payload = np.load(coord_path)
    key = "coord" if "coord" in payload.files else payload.files[0]
    coords = transform_coords(
        np.asarray(payload[key], dtype=float),
        invert_x=invert_x,
        invert_y=invert_y,
    )

    if len(coords) != len(values):
        raise ValueError(
            f"Row mismatch at {spec['label']}: coords={len(coords)}, values={len(values)}"
        )

    return SliceRecord(
        age=float(spec["age"]),
        label=str(spec["label"]),
        path=h5ad_path,
        coords=coords,
        values=values,
        is_virtual=bool(spec.get("is_virtual", False)),
    )


def load_common_record(spec: dict, gene: str, *, invert_x: bool, invert_y: bool) -> SliceRecord:
    h5ad_path = str(spec["h5ad_path"]).strip()
    if not os.path.isfile(h5ad_path):
        raise FileNotFoundError(h5ad_path)

    ad = sc.read_h5ad(h5ad_path)
    if VALID_KEY in ad.obs.columns:
        ad = ad[ad.obs[VALID_KEY].to_numpy().astype(bool)].copy()
    if COORD_KEY not in ad.obsm:
        raise KeyError(f"{COORD_KEY!r} not found in ad.obsm: {h5ad_path}")

    coords = transform_coords(
        np.asarray(ad.obsm[COORD_KEY], dtype=float),
        invert_x=invert_x,
        invert_y=invert_y,
    )
    values = extract_state_gene(ad, gene)

    if len(coords) != len(values):
        raise ValueError(
            f"Row mismatch at {spec['label']}: coords={len(coords)}, values={len(values)}"
        )

    return SliceRecord(
        age=float(spec["age"]),
        label=str(spec["label"]),
        path=h5ad_path,
        coords=coords,
        values=values,
        is_virtual=bool(spec.get("is_virtual", False)),
    )


def load_delta_record(spec: dict, gene: str, *, invert_x: bool, invert_y: bool) -> SliceRecord:
    h5ad_path = str(spec["h5ad_path"]).strip()
    if not os.path.isfile(h5ad_path):
        raise FileNotFoundError(h5ad_path)

    ad = sc.read_h5ad(h5ad_path)
    if COORD_KEY not in ad.obsm:
        raise KeyError(f"{COORD_KEY!r} not found in ad.obsm: {h5ad_path}")
    if gene not in ad.var_names:
        raise KeyError(f"Gene {gene!r} not found in delta map {h5ad_path}")

    coords = transform_coords(
        np.asarray(ad.obsm[COORD_KEY], dtype=float),
        invert_x=invert_x,
        invert_y=invert_y,
    )
    values = to_dense_1d(ad[:, [gene]].X)

    if len(coords) != len(values):
        raise ValueError(
            f"Row mismatch at {spec['label']}: coords={len(coords)}, values={len(values)}"
        )

    return SliceRecord(
        age=float(spec["age"]),
        label=str(spec["label"]),
        path=h5ad_path,
        coords=coords,
        values=values,
        is_virtual=False,
    )


def load_sequence(
    specs: Sequence[dict],
    loader: Callable,
    gene: str,
    figure_name: str,
    *,
    invert_x: bool,
    invert_y: bool,
) -> List[SliceRecord]:
    records: List[SliceRecord] = []
    for spec in specs:
        try:
            records.append(
                loader(
                    spec,
                    gene,
                    invert_x=invert_x,
                    invert_y=invert_y,
                )
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[SKIP] {figure_name}: {spec.get('label', '?')}: {exc}")
            return []
    return sorted(records, key=lambda r: r.age)


# ============================================================
# 3D DRAWING HELPERS
# ============================================================
def state_facecolors(values: np.ndarray, vmax: float, cmap_name: str) -> np.ndarray:
    norm = np.clip(np.asarray(values, dtype=float) / max(vmax, 1e-8), 0.0, 1.0)
    colors = plt.get_cmap(cmap_name)(norm)
    colors[:, 3] = np.clip((0.08 + 0.92 * norm) * STATE_ALPHA, 0.02, 1.0)
    return colors


def delta_facecolors(values: np.ndarray, vmax: float) -> np.ndarray:
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    colors = plt.get_cmap(DELTA_CMAP)(norm(np.asarray(values, dtype=float)))
    colors[:, 3] = DELTA_ALPHA
    return colors


def draw_plane_frame(ax, z_value: float, is_virtual: bool) -> None:
    xmin, xmax, ymin, ymax = FIXED_BOX
    x = np.asarray([xmin, xmax, xmax, xmin, xmin], dtype=float)
    y = np.asarray([ymin, ymin, ymax, ymax, ymin], dtype=float)
    z = np.full(5, float(z_value), dtype=float)

    ax.plot(
        x,
        y,
        z,
        color=VIRTUAL_FRAME_COLOR if is_virtual else REAL_FRAME_COLOR,
        linewidth=DASHED_LW if is_virtual else SOLID_LW,
        linestyle=DASHED_STYLE if is_virtual else "-",
        alpha=0.96,
        zorder=20,
    )


def draw_time_axis(
    ax,
    records: Sequence[SliceRecord],
    z_positions: Sequence[float],
    side: str,
    title_label: str = "Age",
    axis_reference_z: Optional[Sequence[float]] = None,
    held_out_timepoints: Optional[Sequence[Tuple[float, str]]] = None,
    placeholder_timepoints: Optional[Sequence[Tuple[float, str]]] = None,
) -> None:
    """
    Draw the time axis using the shared pair-wise z extent.

    Parameters
    ----------
    held_out_timepoints
        Optional ``(z, label)`` entries for observed ages intentionally withheld
        from this panel. They use the normal time-point style plus a ``Held out``
        annotation on the opposite side of the axis.
    placeholder_timepoints
        Optional ``(z, label)`` entries for genuinely unmeasured/missing states.
        They use the virtual-state color for both the time-point marker and age
        label, plus a ``Missing disease`` annotation on the opposite side.
    """
    xmin, xmax, ymin, _ = FIXED_BOX
    x0 = xmin if side == "left" else xmax
    y0 = ymin

    zs = np.asarray(z_positions, dtype=float)
    ref_zs = zs if axis_reference_z is None else np.asarray(axis_reference_z, dtype=float)
    span = max(float(np.ptp(ref_zs)), 1.0)
    z_top = float(np.max(ref_zs) + 0.08 * span)
    z_bottom = float(np.min(ref_zs) - 0.10 * span)

    ax.plot(
        [x0, x0], [y0, y0], [z_top, z_bottom],
        color=TIME_AXIS_COLOR, linewidth=TIME_AXIS_LW, alpha=0.98, zorder=40,
    )
    ax.quiver(
        x0, y0, z_top, 0.0, 0.0, z_bottom - z_top,
        arrow_length_ratio=TIME_ARROW_LENGTH_RATIO,
        color=TIME_AXIS_COLOR, linewidth=TIME_AXIS_LW, zorder=41,
    )

    marker_colors = [
        VIRTUAL_LABEL_COLOR if r.is_virtual else REAL_LABEL_COLOR
        for r in records
    ]
    ax.scatter(
        np.full(len(zs), x0), np.full(len(zs), y0), zs,
        s=TIME_POINT_SIZE, c=marker_colors,
        edgecolors=TIME_POINT_EDGE_COLOR, linewidths=TIME_POINT_EDGE_WIDTH,
        depthshade=False, zorder=50,
    )

    dx = -30.0 if side == "left" else +30.0
    ha = "right" if side == "left" else "left"
    ax.text(
        x0 + dx, y0 - 25.0, z_top, title_label,
        fontsize=15.5, fontweight="bold", color=TIME_AXIS_COLOR,
        ha=ha, va="bottom",
    )

    for record, z in zip(records, zs):
        ax.text(
            x0 + dx, y0 - 14.0, float(z), record.label,
            fontsize=12.3, fontweight="bold",
            color=VIRTUAL_LABEL_COLOR if record.is_virtual else REAL_LABEL_COLOR,
            ha=ha, va="center", zorder=60,
        )

    # Optional axis-only time points, e.g. the WT 5.7m state withheld from
    # Figure 2. The marker and age label look like ordinary time points, while
    # the status annotation is placed on the opposite side of the axis.
    if held_out_timepoints:
        held_z = np.asarray([float(item[0]) for item in held_out_timepoints], dtype=float)
        ax.scatter(
            np.full(len(held_z), x0),
            np.full(len(held_z), y0),
            held_z,
            s=TIME_POINT_SIZE,
            c=REAL_LABEL_COLOR,
            edgecolors=TIME_POINT_EDGE_COLOR,
            linewidths=TIME_POINT_EDGE_WIDTH,
            depthshade=False,
            zorder=51,
        )

        opposite_dx = -2*dx
        opposite_ha = "left" if ha == "right" else "right"

        for z, label in held_out_timepoints:
            z = float(z)

            # Normal age label on the same side as the other timeline labels.
            ax.text(
                x0 + dx,
                y0 - 14.0,
                z,
                str(label),
                fontsize=12.3,
                fontweight="bold",
                color=REAL_LABEL_COLOR,
                ha=ha,
                va="center",
                zorder=61,
            )

            # Held-out status on the opposite side of the axis.
            ax.text(
                x0 + opposite_dx,
                y0 + 50.0,
                z,
                "Held out",
                fontsize=10.5,
                fontstyle="italic",
                fontweight="bold",
                color=VIRTUAL_LABEL_COLOR,
                ha=opposite_ha,
                va="center",
                zorder=61,
            )

    # Optional missing/unmeasured disease states. These use the virtual-state
    # color on the time axis to distinguish them from measured or held-out data.
    if placeholder_timepoints:
        missing_z = np.asarray(
            [float(item[0]) for item in placeholder_timepoints],
            dtype=float,
        )
        ax.scatter(
            np.full(len(missing_z), x0),
            np.full(len(missing_z), y0),
            missing_z,
            s=TIME_POINT_SIZE,
            c=VIRTUAL_LABEL_COLOR,
            edgecolors=TIME_POINT_EDGE_COLOR,
            linewidths=TIME_POINT_EDGE_WIDTH,
            depthshade=False,
            zorder=51,
        )

        opposite_dx = -2*dx
        opposite_ha = "left" if ha == "right" else "right"

        for z, label in placeholder_timepoints:
            z = float(z)

            # Missing age label uses the same gray color as virtual states.
            ax.text(
                x0 + dx,
                y0 - 14.0,
                z,
                str(label),
                fontsize=12.3,
                fontweight="bold",
                color=VIRTUAL_LABEL_COLOR,
                ha=ha,
                va="center",
                zorder=61,
            )

            # Missing-state annotation mirrors the Held out annotation style.
            # ax.text(
            #     x0 - 2*opposite_dx,
            #     y0 - 270.0,
            #     z,
            #     "(Unmeasured)",
            #     fontsize=10.5,
            #     fontstyle="italic",
            #     fontweight="bold",
            #     color=VIRTUAL_LABEL_COLOR,
            #     ha=opposite_ha,
            #     va="center",
            #     zorder=61,
            # )


def style_3d_axis(
    ax,
    z_positions: Sequence[float],
    view_elev: float,
    view_azim: float,
    reference_z_positions: Optional[Sequence[float]] = None,
) -> None:
    """Use identical z limits/aspect for both subplots in a paired figure."""
    xmin, xmax, ymin, ymax = FIXED_BOX
    zs = np.asarray(z_positions, dtype=float)
    ref_zs = zs if reference_z_positions is None else np.asarray(reference_z_positions, dtype=float)
    z_span = max(float(np.ptp(ref_zs)), 1.0)

    x_pad = 70.0
    y_pad = 70.0
    z_pad_low = 0.12 * z_span
    z_pad_high = 0.10 * z_span

    x0, x1 = xmin - x_pad, xmax + x_pad
    y0, y1 = ymin - y_pad, ymax + y_pad
    z0 = float(np.min(ref_zs) - z_pad_low)
    z1 = float(np.max(ref_zs) + z_pad_high)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)

    display_span = np.asarray(
        [
            (x1 - x0) * SPATIAL_DISPLAY_SCALE,
            (y1 - y0) * SPATIAL_DISPLAY_SCALE,
            (z1 - z0),
        ],
        dtype=float,
    )
    display_span /= max(float(np.max(display_span)), 1e-8)

    ax.view_init(elev=float(view_elev), azim=float(view_azim))
    ax.set_proj_type("ortho")
    ax.set_box_aspect(tuple(display_span))
    ax.set_axis_off()


def plot_state_plane(ax, record: SliceRecord, z_value: float, vmax: float, seed: int, cmap_name: str) -> None:
    idx = sample_indices(len(record.coords), MAX_POINTS_PER_PLANE, seed)
    coords = record.coords[idx]
    values = record.values[idx]
    colors = state_facecolors(values, vmax, cmap_name)

    ax.scatter(coords[:, 0], coords[:, 1], np.full(len(coords), float(z_value)), c=colors, s=POINT_SIZE_STATE, linewidths=0, depthshade=False, rasterized=True, zorder=3)
    draw_plane_frame(ax, z_value, is_virtual=record.is_virtual)


def plot_delta_plane(ax, record: SliceRecord, z_value: float, vmax: float, seed: int) -> None:
    idx = sample_indices(len(record.coords), MAX_POINTS_PER_PLANE, seed)
    coords = record.coords[idx]
    values = record.values[idx]
    colors = delta_facecolors(values, vmax)

    ax.scatter(coords[:, 0], coords[:, 1], np.full(len(coords), float(z_value)), c=colors, s=POINT_SIZE_DELTA, linewidths=0, depthshade=False, rasterized=True, zorder=3)
    draw_plane_frame(ax, z_value, is_virtual=False)


# ============================================================
# AXIS RENDERERS
# ============================================================
def render_stack_on_axis(
    ax,
    records: Sequence[SliceRecord],
    *,
    mode: str,
    axis_side: str,
    view_elev: float,
    view_azim: float,
    title: str,
    state_cmap: Optional[str] = None,
    z_positions: Optional[Sequence[float]] = None,
    reference_z_positions: Optional[Sequence[float]] = None,
    held_out_timepoints: Optional[Sequence[Tuple[float, str]]] = None,
    placeholder_timepoints: Optional[Sequence[Tuple[float, str]]] = None,
    show_time_axis: bool = True,
):
    """Render one stacked 3D panel on a supplied shared vertical scaffold."""
    if not records:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(missing input)", fontsize=12.5, pad=8)
        return None

    ordered = sorted(records, key=lambda r: r.age)
    if z_positions is None:
        z_positions = stack_positions(len(ordered))
    z_positions = np.asarray(z_positions, dtype=float)

    if len(z_positions) != len(ordered):
        raise ValueError(
            f"z_positions length {len(z_positions)} does not match "
            f"record count {len(ordered)} for {title!r}."
        )

    if reference_z_positions is None:
        reference_z_positions = z_positions
    reference_z_positions = np.asarray(reference_z_positions, dtype=float)

    if mode == "state":
        vmax = robust_positive_vmax([r.values for r in ordered], q=99.0)
        cmap_name = AD_STATE_CMAP if state_cmap is None else state_cmap
        for i, (record, z) in enumerate(zip(ordered, z_positions)):
            plot_state_plane(
                ax, record, float(z), vmax, seed=100 + i, cmap_name=cmap_name
            )
        mappable = ScalarMappable(
            norm=Normalize(vmin=0.0, vmax=vmax), cmap=cmap_name
        )
        colorbar_label = f"log1p {GENE_NAME} expression"
    elif mode == "delta":
        vmax = robust_symmetric_vmax([r.values for r in ordered], q=99.0)
        for i, (record, z) in enumerate(zip(ordered, z_positions)):
            plot_delta_plane(ax, record, float(z), vmax, seed=200 + i)
        mappable = ScalarMappable(
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            cmap=DELTA_CMAP,
        )
        colorbar_label = f"{GENE_NAME} spatial delta (AD−WT)"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Optional empty states can be shown as dashed placeholder planes.
    # ``held_out_timepoints`` are withheld observations (e.g. WT 5.7m), while
    # ``placeholder_timepoints`` are unmeasured states (e.g. AD 13.4m).
    if placeholder_timepoints:
        for z_placeholder, _label in placeholder_timepoints:
            draw_plane_frame(ax, float(z_placeholder), is_virtual=True)

    if held_out_timepoints:
        for z_hold, _label in held_out_timepoints:
            draw_plane_frame(ax, float(z_hold), is_virtual=True)

    if show_time_axis:
        draw_time_axis(
            ax,
            ordered,
            z_positions,
            side=axis_side,
            title_label="Age",
            axis_reference_z=reference_z_positions,
            held_out_timepoints=held_out_timepoints,
            placeholder_timepoints=placeholder_timepoints,
        )

    style_3d_axis(
        ax,
        z_positions,
        view_elev=view_elev,
        view_azim=view_azim,
        reference_z_positions=reference_z_positions,
    )
    ax.set_title(title, pad=10, fontsize=13.5, fontweight="bold")

    mappable.set_array([])
    return {
        "mappable": mappable,
        "colorbar_label": colorbar_label,
        "has_virtual": (
            any(r.is_virtual for r in ordered)
            or bool(held_out_timepoints)
            or bool(placeholder_timepoints)
        ),
    }


def add_virtual_legend(ax) -> None:
    handles = [
        Line2D([0], [0], color=REAL_FRAME_COLOR, linewidth=1.5, linestyle="-", label="Observed / retained state"),
        Line2D([0], [0], color=VIRTUAL_FRAME_COLOR, linewidth=1.5, linestyle=DASHED_STYLE, label="Virtual state"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.98, 0.98), frameon=False, fontsize=8.8)


def plot_paired_state_figure(
    *,
    left_records: Sequence[SliceRecord],
    right_records: Sequence[SliceRecord],
    left_title: str,
    right_title: str,
    left_axis_side: str,
    right_axis_side: str,
    left_cmap: str,
    right_cmap: str,
    left_view: Tuple[float, float],
    right_view: Tuple[float, float],
    save_name: str,
    suptitle: Optional[str] = None,
    reference_ages: Optional[Sequence[float]] = None,
    left_held_out_ages: Optional[Sequence[float]] = None,
    right_held_out_ages: Optional[Sequence[float]] = None,
    left_placeholder_ages: Optional[Sequence[float]] = None,
    right_placeholder_ages: Optional[Sequence[float]] = None,
) -> None:
    """
    Plot AD/WT panels on one shared age-aware schematic z scaffold.

    The union of ages in the two panels determines the shared positions. This
    aligns identical ages and separates different ages while preserving equal
    schematic spacing between progression stages.
    """
    set_style()
    fig = plt.figure(figsize=(15.5, 8.3))
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")

    shared_ages, shared_z, left_z, right_z = build_shared_age_scaffold(
        left_records,
        right_records,
        reference_ages=reference_ages,
    )

    # Convert requested held-out ages to their exact shared-scaffold z values.
    age_to_z = {
        round(float(age), 6): float(z)
        for age, z in zip(shared_ages, shared_z)
    }

    def _held_out_points(ages: Optional[Sequence[float]]):
        points = []
        for age in ages or []:
            key = round(float(age), 6)
            if key not in age_to_z:
                raise ValueError(
                    f"Held-out age {age:g}m is not present in shared scaffold "
                    f"{shared_ages.tolist()}"
                )
            points.append((age_to_z[key], f"{float(age):g}m"))
        return points

    left_held_out_points = _held_out_points(left_held_out_ages)
    right_held_out_points = _held_out_points(right_held_out_ages)
    left_placeholder_points = _held_out_points(left_placeholder_ages)
    right_placeholder_points = _held_out_points(right_placeholder_ages)

    print(
        "[INFO] Shared paired-age scaffold: "
        + ", ".join(
            f"{age:g}m -> z={z:g}"
            for age, z in zip(shared_ages, shared_z)
        )
    )
    print(
        "[INFO] Left map positions: "
        + ", ".join(f"{r.age:g}m:{z:g}" for r, z in zip(left_records, left_z))
    )
    print(
        "[INFO] Right map positions: "
        + ", ".join(f"{r.age:g}m:{z:g}" for r, z in zip(right_records, right_z))
    )

    left_meta = render_stack_on_axis(
        ax_left,
        left_records,
        mode="state",
        axis_side=left_axis_side,
        view_elev=left_view[0],
        view_azim=left_view[1],
        title=left_title,
        state_cmap=left_cmap,
        z_positions=left_z,
        reference_z_positions=shared_z,
        held_out_timepoints=left_held_out_points,
        placeholder_timepoints=left_placeholder_points,
    )
    right_meta = render_stack_on_axis(
        ax_right,
        right_records,
        mode="state",
        axis_side=right_axis_side,
        view_elev=right_view[0],
        view_azim=right_view[1],
        title=right_title,
        state_cmap=right_cmap,
        z_positions=right_z,
        reference_z_positions=shared_z,
        held_out_timepoints=right_held_out_points,
        placeholder_timepoints=right_placeholder_points,
    )

    if left_meta is not None:
        cb1 = fig.colorbar(
            left_meta["mappable"], ax=ax_left, fraction=0.030, pad=0.02, shrink=0.68
        )
        cb1.set_label(left_meta["colorbar_label"], fontsize=10)
        cb1.ax.tick_params(labelsize=9)
        if left_meta["has_virtual"]:
            add_virtual_legend(ax_left)

    if right_meta is not None:
        cb2 = fig.colorbar(
            right_meta["mappable"], ax=ax_right, fraction=0.030, pad=0.02, shrink=0.68
        )
        cb2.set_label(right_meta["colorbar_label"], fontsize=10)
        cb2.ax.tick_params(labelsize=9)
        if right_meta["has_virtual"]:
            add_virtual_legend(ax_right)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=0.98)
    finalize_figure(fig, os.path.join(OUTDIR, save_name))


def plot_single_delta_figure(records: Sequence[SliceRecord], *, title: str, save_name: str) -> None:
    set_style()
    fig = plt.figure(figsize=(8.2, 8.4))
    ax = fig.add_subplot(111, projection="3d")

    meta = render_stack_on_axis(
        ax,
        records,
        mode="delta",
        axis_side="left",
        view_elev=DELTA_VIEW_ELEV,
        view_azim=DELTA_VIEW_AZIM,
        title=title,
    )
    if meta is not None:
        cb = fig.colorbar(meta["mappable"], ax=ax, fraction=0.030, pad=0.02, shrink=0.68)
        cb.set_label(meta["colorbar_label"], fontsize=10)
        cb.ax.tick_params(labelsize=9)

    finalize_figure(fig, os.path.join(OUTDIR, save_name))


# ============================================================
# FIGURE BUILDERS
# ============================================================
def build_figure1(gene: str) -> None:
    ad_records = load_sequence(
        AD_RAW_SPECS, load_raw_record, gene, "Figure 1 AD",
        invert_x=AD_INVERT_X, invert_y=AD_INVERT_Y,
    )
    wt_records = load_sequence(
        WT_RAW_SPECS, load_raw_record, gene, "Figure 1 WT",
        invert_x=WT_INVERT_X, invert_y=WT_INVERT_Y,
    )
    plot_paired_state_figure(
        left_records=ad_records,
        right_records=wt_records,
        left_title="AD observed age states",
        right_title="WT observed age states",
        left_axis_side="left",
        right_axis_side="right",
        left_cmap=AD_STATE_CMAP,
        right_cmap=WT_STATE_CMAP,
        left_view=(AD_VIEW_ELEV, AD_VIEW_AZIM),
        right_view=(WT_VIEW_ELEV, WT_VIEW_AZIM),
        save_name=f"figure1_observed_AD_WT_{gene}.png",
        suptitle="Observed, incompletely paired states",
        reference_ages=(2.5, 5.7, 13.4, 17.9),
    )


def build_figure2(gene: str) -> None:
    ad_specs = [AD_COMMON_FRAME[k].copy() for k in ("2.5m", "5.7m", "17.9m")]
    wt_specs = [WT_COMMON_FRAME[k].copy() for k in ("2.5m", "13.4m")]
    for spec in ad_specs:
        spec["is_virtual"] = False
    for spec in wt_specs:
        spec["is_virtual"] = False


    ad_records = load_sequence(
        ad_specs, load_common_record, gene, "Figure 2 AD",
        invert_x=AD_INVERT_X, invert_y=AD_INVERT_Y,
    )
    wt_records = load_sequence(
        wt_specs, load_common_record, gene, "Figure 2 WT",
        invert_x=WT_INVERT_X, invert_y=WT_INVERT_Y,
    )

    # The paired plotter automatically builds the shared age scaffold from
    # 2.5, 5.7, 13.4, and 17.9m. Thus the withheld WT 5.7m position remains
    # empty, WT 13.4m occupies its own stage, and AD 17.9m is lower still.
    plot_paired_state_figure(
        left_records=ad_records,
        right_records=wt_records,
        left_title="AD age-matched states in a shared spatial frame",
        right_title="WT common-frame states (5.7m withheld)",
        left_axis_side="left",
        right_axis_side="right",
        left_cmap=AD_STATE_CMAP,
        right_cmap=WT_STATE_CMAP,
        left_view=(AD_VIEW_ELEV, AD_VIEW_AZIM),
        right_view=(WT_VIEW_ELEV, WT_VIEW_AZIM),
        save_name=f"figure2_common_frame_AD_WT_{gene}.png",
        suptitle="Age-matched paired states in a shared spatial frame",
        reference_ages=(2.5, 5.7, 13.4, 17.9),
        left_placeholder_ages=(13.4,),
        right_held_out_ages=(5.7,),
    )


def build_figure3(gene: str) -> None:
    ad_specs = [AD_COMMON_FRAME[k].copy() for k in ("2.5m", "5.7m", "13.4m", "17.9m")]
    wt_specs = [WT_COMMON_FRAME[k].copy() for k in ("2.5m", "5.7m", "13.4m")]
    for spec in ad_specs:
        spec["is_virtual"] = spec["label"] == "13.4m"
    for spec in wt_specs:
        spec["is_virtual"] = spec["label"] == "5.7m"



    ad_records = load_sequence(
        ad_specs, load_common_record, gene, "Figure 3 AD",
        invert_x=AD_INVERT_X, invert_y=AD_INVERT_Y,
    )
    wt_records = load_sequence(
        wt_specs, load_common_record, gene, "Figure 3 WT",
        invert_x=WT_INVERT_X, invert_y=WT_INVERT_Y,
    )
    plot_paired_state_figure(
        left_records=ad_records,
        right_records=wt_records,
        left_title="AD common-frame states with virtual generation",
        right_title="WT common-frame states with virtual generation",
        left_axis_side="left",
        right_axis_side="right",
        left_cmap=AD_STATE_CMAP,
        right_cmap=WT_STATE_CMAP,
        left_view=(AD_VIEW_ELEV, AD_VIEW_AZIM),
        right_view=(WT_VIEW_ELEV, WT_VIEW_AZIM),
        save_name=f"figure3_virtual_generation_AD_WT_{gene}.png",
        suptitle="Common-frame reconstruction and missing-age virtual generation",
    )


def build_figure4(gene: str) -> None:
    """
    Merge the previous matched-state Figure 4 and delta Figure 5 into one
    1 x 3 layout:

        AD  |  AD-WT spatial delta  |  WT

    All three panels use exactly the same schematic z scaffold for
    2.5m, 5.7m, and 13.4m. The center delta panel intentionally does not draw
    a time axis; its map planes are aligned to the AD/WT age positions.
    """
    ad_specs = [AD_COMMON_FRAME[k].copy() for k in ("2.5m", "5.7m", "13.4m")]
    wt_specs = [WT_COMMON_FRAME[k].copy() for k in ("2.5m", "5.7m", "13.4m")]
    delta_specs = [DELTA_SPECS[k].copy() for k in ("2.5m", "5.7m", "13.4m")]

    # Preserve the generated-state convention from the previous Figure 4.
    for spec in ad_specs:
        spec["is_virtual"] = spec["label"] == "13.4m"
    for spec in wt_specs:
        spec["is_virtual"] = spec["label"] == "5.7m"

    ad_records = load_sequence(
        ad_specs,
        load_common_record,
        gene,
        "Figure 4 AD",
        invert_x=AD_INVERT_X,
        invert_y=AD_INVERT_Y,
    )
    delta_records = load_sequence(
        delta_specs,
        load_delta_record,
        gene,
        "Figure 4 Delta",
        invert_x=DELTA_INVERT_X,
        invert_y=DELTA_INVERT_Y,
    )
    wt_records = load_sequence(
        wt_specs,
        load_common_record,
        gene,
        "Figure 4 WT",
        invert_x=WT_INVERT_X,
        invert_y=WT_INVERT_Y,
    )

    if not ad_records or not delta_records or not wt_records:
        print("[SKIP] Figure 4: one or more AD/Delta/WT inputs are missing.")
        return

    # One explicit age scaffold shared by all three panels.
    shared_ages = np.asarray([2.5, 5.7, 13.4], dtype=float)
    shared_z = stack_positions(len(shared_ages))
    age_to_z = {
        round(float(age), 6): float(z)
        for age, z in zip(shared_ages, shared_z)
    }

    def map_records_to_shared_z(records: Sequence[SliceRecord]) -> np.ndarray:
        return np.asarray(
            [age_to_z[round(float(record.age), 6)] for record in records],
            dtype=float,
        )

    ad_z = map_records_to_shared_z(ad_records)
    delta_z = map_records_to_shared_z(delta_records)
    wt_z = map_records_to_shared_z(wt_records)

    print(
        "[INFO] Figure 4 shared AD/Delta/WT scaffold: "
        + ", ".join(
            f"{age:g}m -> z={z:g}"
            for age, z in zip(shared_ages, shared_z)
        )
    )

    set_style()
    fig = plt.figure(figsize=(21.5, 8.4))
    ax_ad = fig.add_subplot(1, 3, 1, projection="3d")
    ax_delta = fig.add_subplot(1, 3, 2, projection="3d")
    ax_wt = fig.add_subplot(1, 3, 3, projection="3d")

    ad_meta = render_stack_on_axis(
        ax_ad,
        ad_records,
        mode="state",
        axis_side="left",
        view_elev=AD_VIEW_ELEV,
        view_azim=AD_VIEW_AZIM,
        title="AD matched states",
        state_cmap=AD_STATE_CMAP,
        z_positions=ad_z,
        reference_z_positions=shared_z,
        show_time_axis=True,
    )

    delta_meta = render_stack_on_axis(
        ax_delta,
        delta_records,
        mode="delta",
        axis_side="left",  # ignored because show_time_axis=False
        view_elev=DELTA_VIEW_ELEV,
        view_azim=DELTA_VIEW_AZIM,
        title="Spatial AD−WT delta",
        z_positions=delta_z,
        reference_z_positions=shared_z,
        show_time_axis=False,
    )

    wt_meta = render_stack_on_axis(
        ax_wt,
        wt_records,
        mode="state",
        axis_side="right",
        view_elev=WT_VIEW_ELEV,
        view_azim=WT_VIEW_AZIM,
        title="WT matched states",
        state_cmap=WT_STATE_CMAP,
        z_positions=wt_z,
        reference_z_positions=shared_z,
        show_time_axis=True,
    )

    # Keep one color scale for each data type/panel, as in the previous
    # separate Figure 4 and Figure 5.
    if ad_meta is not None:
        cb_ad = fig.colorbar(
            ad_meta["mappable"],
            ax=ax_ad,
            fraction=0.028,
            pad=0.015,
            shrink=0.66,
        )
        cb_ad.set_label(ad_meta["colorbar_label"], fontsize=9.5)
        cb_ad.ax.tick_params(labelsize=8.5)
        if ad_meta["has_virtual"]:
            add_virtual_legend(ax_ad)

    if delta_meta is not None:
        cb_delta = fig.colorbar(
            delta_meta["mappable"],
            ax=ax_delta,
            fraction=0.028,
            pad=0.015,
            shrink=0.66,
        )
        cb_delta.set_label(delta_meta["colorbar_label"], fontsize=9.5)
        cb_delta.ax.tick_params(labelsize=8.5)

    if wt_meta is not None:
        cb_wt = fig.colorbar(
            wt_meta["mappable"],
            ax=ax_wt,
            fraction=0.028,
            pad=0.015,
            shrink=0.66,
        )
        cb_wt.set_label(wt_meta["colorbar_label"], fontsize=9.5)
        cb_wt.ax.tick_params(labelsize=8.5)
        if wt_meta["has_virtual"]:
            add_virtual_legend(ax_wt)

    fig.suptitle(
        "Age-matched AD–WT spatial comparison",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    finalize_figure(
        fig,
        os.path.join(OUTDIR, f"figure4_AD_Delta_WT_{gene}.png"),
    )


# ============================================================
# RUN
# ============================================================
def run_all(gene: str) -> None:
    ensure_dir(OUTDIR)
    # build_figure1(gene)
    build_figure2(gene)
    # build_figure3(gene)
    build_figure4(gene)
    print(f"[DONE] Combined disease-analysis pipeline figures written/displayed from: {OUTDIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate combined vertical 3D pipeline figures for AD-vs-WT disease analysis.")
    parser.add_argument("--figure-mode", choices=["save", "show", "both"], default=FIGURE_MODE)
    parser.add_argument("--gene", default=GENE_NAME)
    parser.add_argument("--stack-gap", type=float, default=STACK_GAP)
    parser.add_argument("--spatial-display-scale", type=float, default=SPATIAL_DISPLAY_SCALE)
    parser.add_argument("--ad-view-elev", type=float, default=AD_VIEW_ELEV)
    parser.add_argument("--ad-view-azim", type=float, default=AD_VIEW_AZIM)
    parser.add_argument("--wt-view-elev", type=float, default=WT_VIEW_ELEV)
    parser.add_argument("--wt-view-azim", type=float, default=WT_VIEW_AZIM)
    parser.add_argument("--delta-view-elev", type=float, default=DELTA_VIEW_ELEV)
    parser.add_argument("--delta-view-azim", type=float, default=DELTA_VIEW_AZIM)
    parser.add_argument("--ad-invert-x", type=int, choices=[0, 1], default=int(AD_INVERT_X))
    parser.add_argument("--ad-invert-y", type=int, choices=[0, 1], default=int(AD_INVERT_Y))
    parser.add_argument("--wt-invert-x", type=int, choices=[0, 1], default=int(WT_INVERT_X))
    parser.add_argument("--wt-invert-y", type=int, choices=[0, 1], default=int(WT_INVERT_Y))
    args = parser.parse_args()

    globals()["FIGURE_MODE"] = str(args.figure_mode)
    globals()["GENE_NAME"] = str(args.gene)
    globals()["STACK_GAP"] = float(args.stack_gap)
    globals()["SPATIAL_DISPLAY_SCALE"] = float(args.spatial_display_scale)
    globals()["AD_VIEW_ELEV"] = float(args.ad_view_elev)
    globals()["AD_VIEW_AZIM"] = float(args.ad_view_azim)
    globals()["WT_VIEW_ELEV"] = float(args.wt_view_elev)
    globals()["WT_VIEW_AZIM"] = float(args.wt_view_azim)
    globals()["DELTA_VIEW_ELEV"] = float(args.delta_view_elev)
    globals()["DELTA_VIEW_AZIM"] = float(args.delta_view_azim)
    globals()["AD_INVERT_X"] = bool(args.ad_invert_x)
    globals()["AD_INVERT_Y"] = bool(args.ad_invert_y)
    globals()["WT_INVERT_X"] = bool(args.wt_invert_x)
    globals()["WT_INVERT_Y"] = bool(args.wt_invert_y)

    print(f"[INFO] FIGURE_MODE = {FIGURE_MODE}")
    print(f"[INFO] GENE_NAME = {GENE_NAME}")
    print(f"[INFO] AD colormap = {AD_STATE_CMAP}")
    print(f"[INFO] WT colormap = {WT_STATE_CMAP}")
    print(f"[INFO] AD camera = elev {AD_VIEW_ELEV}, azim {AD_VIEW_AZIM}")
    print(f"[INFO] WT camera = elev {WT_VIEW_ELEV}, azim {WT_VIEW_AZIM}")
    print(f"[INFO] Delta camera = elev {DELTA_VIEW_ELEV}, azim {DELTA_VIEW_AZIM}")
    print(f"[INFO] AD invert = x:{AD_INVERT_X}, y:{AD_INVERT_Y}")
    print(f"[INFO] WT invert = x:{WT_INVERT_X}, y:{WT_INVERT_Y}")

    run_all(gene=GENE_NAME)


if __name__ == "__main__":
    main()
