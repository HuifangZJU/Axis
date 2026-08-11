from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import scanpy as sc


OFFSET = 5.0
TARGET_RANGE = 1000.0


@dataclass(frozen=True)
class NormalizationMeta:
    """Parameters used to place a coordinate pair in one common frame."""

    origin_x: float
    origin_y: float
    scale: float
    offset: float
    target_range: float
    mask_height: int
    mask_width: int

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.origin_x,
                self.origin_y,
                self.scale,
                self.offset,
                self.target_range,
                self.mask_height,
                self.mask_width,
            ],
            dtype=np.float64,
        )


# =============================================================================
# Validation and normalization
# =============================================================================
def as_xy(coords: np.ndarray, source: str) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"{source}: expected coordinates with shape [N, 2+], got {coords.shape}")
    coords = coords[:, :2]
    if not np.all(np.isfinite(coords)):
        raise ValueError(f"{source}: coordinates contain NaN or infinite values")
    return coords


def as_labels(labels: np.ndarray, source: str) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    if labels.dtype.kind == "S":
        labels = labels.astype(str)
    elif labels.dtype == object:
        labels = np.asarray(
            [x.decode("utf-8") if isinstance(x, bytes) else x for x in labels],
            dtype=object,
        )
    if np.any(np.asarray([x is None for x in labels], dtype=bool)):
        raise ValueError(f"{source}: labels contain None")
    return labels

_MISSING_LABEL_STRINGS = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "<na>",
}


def filter_missing_domain_labels(
    coords: np.ndarray,
    labels: np.ndarray,
    source: str,
    canonicalize_numeric: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove points whose domain label is missing.

    Missing labels include:
      - np.nan
      - None
      - pandas-style <NA>
      - string representations such as "nan", "None", and ""

    Coordinates and labels are filtered together to preserve row alignment.

    When ``canonicalize_numeric=True``, numeric labels are returned as
    canonical strings:
        "4.0" -> "4"
        4.0   -> "4"
        4     -> "4"
    """
    coords = as_xy(coords, f"{source}: coordinates")
    labels = np.asarray(labels).reshape(-1)

    if coords.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{source}: coordinates ({coords.shape[0]}) and labels "
            f"({labels.shape[0]}) have different lengths"
        )

    keep_mask = np.ones(labels.shape[0], dtype=bool)
    cleaned_labels = []

    for i, value in enumerate(labels):
        if isinstance(value, bytes):
            value = value.decode("utf-8")

        text = str(value).strip()
        text_lower = text.lower()

        is_missing = (
            value is None
            or text_lower in _MISSING_LABEL_STRINGS
        )

        # Catch numeric NaN or infinity.
        if not is_missing:
            try:
                numeric_value = float(value)
                if not np.isfinite(numeric_value):
                    is_missing = True
            except (TypeError, ValueError):
                pass

        if is_missing:
            keep_mask[i] = False
            cleaned_labels.append(None)
            continue

        if canonicalize_numeric:
            try:
                numeric_value = float(value)

                if numeric_value.is_integer():
                    cleaned_value = str(int(numeric_value))
                else:
                    cleaned_value = f"{numeric_value:g}"

            except (TypeError, ValueError):
                cleaned_value = text
        else:
            cleaned_value = value

        cleaned_labels.append(cleaned_value)

    n_removed = int((~keep_mask).sum())
    if n_removed > 0:
        print(
            f"[INFO] {source}: removed {n_removed} points with missing "
            f"domain labels; retained {int(keep_mask.sum())}."
        )

    filtered_coords = coords[keep_mask]
    filtered_labels = np.asarray(
        [cleaned_labels[i] for i in np.flatnonzero(keep_mask)],
        dtype=str if canonicalize_numeric else object,
    )

    if filtered_coords.shape[0] == 0:
        raise ValueError(
            f"{source}: no points remain after removing missing domain labels"
        )

    return filtered_coords, filtered_labels

def validate_pair(
    coords_fixed: np.ndarray,
    labels_fixed: np.ndarray,
    coords_moving: np.ndarray,
    labels_moving: np.ndarray,
    source: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate one pair without silently truncating coordinates or labels."""

    coords_fixed = as_xy(coords_fixed, f"{source}: fixed coordinates")
    coords_moving = as_xy(coords_moving, f"{source}: moving coordinates")
    labels_fixed = as_labels(labels_fixed, f"{source}: fixed labels")
    labels_moving = as_labels(labels_moving, f"{source}: moving labels")

    if coords_fixed.shape[0] != labels_fixed.shape[0]:
        raise ValueError(
            f"{source}: fixed coordinates ({coords_fixed.shape[0]}) and labels "
            f"({labels_fixed.shape[0]}) have different lengths"
        )
    if coords_moving.shape[0] != labels_moving.shape[0]:
        raise ValueError(
            f"{source}: moving coordinates ({coords_moving.shape[0]}) and labels "
            f"({labels_moving.shape[0]}) have different lengths"
        )
    return coords_fixed, coords_moving, labels_fixed, labels_moving


def normalize_pair_by_fixed_frame(
    coords_fixed: np.ndarray,
    coords_moving: np.ndarray,
    offset: float = OFFSET,
    target_range: float = TARGET_RANGE,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], NormalizationMeta]:
    """
    Normalize both slices with one isotropic scale derived from the fixed slice.

    A single scale preserves angles, distances, and tissue aspect ratio. Both
    slices are translated by their combined minimum so they remain in the same
    coordinate frame.
    """

    coords_fixed = as_xy(coords_fixed, "coords_fixed")
    coords_moving = as_xy(coords_moving, "coords_moving")

    fixed_span = np.ptp(coords_fixed, axis=0)
    scale = float(np.max(fixed_span))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("The fixed slice has zero or invalid spatial extent")

    origin = np.min(np.vstack([coords_fixed, coords_moving]), axis=0)
    factor = float(target_range) / scale

    fixed_norm = (coords_fixed - origin) * factor + float(offset)
    moving_norm = (coords_moving - origin) * factor + float(offset)

    combined = np.vstack([fixed_norm, moving_norm])
    width = int(np.ceil(combined[:, 0].max() + offset)) + 1
    height = int(np.ceil(combined[:, 1].max() + offset)) + 1
    mask_shape = (height, width)

    meta = NormalizationMeta(
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        scale=scale,
        offset=float(offset),
        target_range=float(target_range),
        mask_height=height,
        mask_width=width,
    )
    return fixed_norm.astype(np.float32), moving_norm.astype(np.float32), mask_shape, meta


# =============================================================================
# Data loaders
# =============================================================================
def load_h5ad_pair(
    fixed_path: Path,
    moving_path: Path,
    coord_key: str,
    label_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fixed_path = Path(fixed_path)
    moving_path = Path(moving_path)
    if not fixed_path.exists():
        raise FileNotFoundError(fixed_path)
    if not moving_path.exists():
        raise FileNotFoundError(moving_path)

    adata_fixed = sc.read_h5ad(fixed_path)
    adata_moving = sc.read_h5ad(moving_path)

    if coord_key not in adata_fixed.obsm or coord_key not in adata_moving.obsm:
        raise KeyError(f"Both h5ad files must contain obsm['{coord_key}']")
    if label_key not in adata_fixed.obs or label_key not in adata_moving.obs:
        raise KeyError(f"Both h5ad files must contain obs['{label_key}']")

    return validate_pair(
        adata_fixed.obsm[coord_key],
        adata_fixed.obs[label_key].to_numpy(),
        adata_moving.obsm[coord_key],
        adata_moving.obs[label_key].to_numpy(),
        source=f"h5ad pair {fixed_path.name} -> {moving_path.name}",
    )


def load_competing_method_pair(
    method_dir: Path,
    fixed_id: int,
    moving_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(method_dir) / f"{fixed_id}_{moving_id}_result.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as data:
        required = {"pts1", "pts2", "label1", "label2"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path}: missing arrays {sorted(missing)}")
        return validate_pair(
            data["pts1"],
            data["label1"],
            data["pts2"],
            data["label2"],
            source=str(path),
        )


def load_axis_validation_npz(
    path: Path,
    coord_key: str = "coord",
    label_key: str = "label",
) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        if coord_key not in data.files:
            raise KeyError(f"{path}: missing coordinate array '{coord_key}'")
        if label_key not in data.files:
            raise KeyError(f"{path}: missing label array '{label_key}'")
        coords = as_xy(data[coord_key], f"{path}:{coord_key}")
        labels = as_labels(data[label_key], f"{path}:{label_key}")
    if len(coords) != len(labels):
        raise ValueError(f"{path}: coordinates and labels have different lengths")
    return coords, labels


def load_axis_pair(
    group_id: int,
    fixed_id: int,
    moving_id: int,
    network_input_dir: Path,
    pairwise_dir: Path,
    validation_coord_key: str = "coord",
    validation_label_key: str = "label",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    network_input_dir = Path(network_input_dir)
    pairwise_dir = Path(pairwise_dir)

    fixed_path = network_input_dir / f"{group_id}_{fixed_id}_validation.npz"
    moving_path = network_input_dir / f"{group_id}_{moving_id}_validation.npz"
    coords_fixed, labels_fixed = load_axis_validation_npz(
        fixed_path, coord_key=validation_coord_key, label_key=validation_label_key
    )
    _, labels_moving = load_axis_validation_npz(
        moving_path, coord_key=validation_coord_key, label_key=validation_label_key
    )

    warped_path = pairwise_dir / f"group_{group_id}" / f"{fixed_id}_{moving_id}_warped_coor.npy"
    if not warped_path.exists():
        raise FileNotFoundError(warped_path)
    coords_moving = as_xy(np.load(warped_path), str(warped_path))

    return validate_pair(
        coords_fixed,
        labels_fixed,
        coords_moving,
        labels_moving,
        source=f"AXIS group={group_id}, pair={fixed_id}->{moving_id}",
    )

def load_axis_unaligned_pair(
    group_id: int,
    fixed_id: int,
    moving_id: int,
    network_input_dir: Path,
    validation_coord_key: str = "coord",
    validation_label_key: str = "label",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    network_input_dir = Path(network_input_dir)

    fixed_path = network_input_dir / f"{group_id}_{fixed_id}_validation.npz"
    moving_path = network_input_dir / f"{group_id}_{moving_id}_validation.npz"
    coords_fixed, labels_fixed = load_axis_validation_npz(
        fixed_path, coord_key=validation_coord_key, label_key=validation_label_key
    )
    coords_moving, labels_moving = load_axis_validation_npz(
        moving_path, coord_key=validation_coord_key, label_key=validation_label_key
    )
    return validate_pair(
        coords_fixed,
        labels_fixed,
        coords_moving,
        labels_moving,
        source=f"AXIS group={group_id}, pair={fixed_id}->{moving_id}",
    )


def load_pair_for_method(
    *,
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
    competing_methods: Sequence[str],
    competing_root: Path,
    h5ad_dir: Path,
    h5ad_coord_key: str,
    h5ad_label_key: str,
    axis_network_input_dir: Path,
    axis_pairwise_dir: Path,
    axis_validation_coord_key: str = "coord",
    axis_validation_label_key: str = "label",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if method == "unaligned":
        return load_axis_unaligned_pair(
            group_id=group_id,
            fixed_id=fixed_id,
            moving_id=moving_id,
            network_input_dir=axis_network_input_dir,
            validation_coord_key=axis_validation_coord_key,
            validation_label_key=axis_validation_label_key,
        )

        # return load_h5ad_pair(
        #     Path(h5ad_dir) / f"{group_id}_{fixed_id}.h5ad",
        #     Path(h5ad_dir) / f"{group_id}_{moving_id}.h5ad",
        #     coord_key=h5ad_coord_key,
        #     label_key=h5ad_label_key,
        # )

    if method == "axis":
        return load_axis_pair(
            group_id=group_id,
            fixed_id=fixed_id,
            moving_id=moving_id,
            network_input_dir=axis_network_input_dir,
            pairwise_dir=axis_pairwise_dir,
            validation_coord_key=axis_validation_coord_key,
            validation_label_key=axis_validation_label_key,
        )
    if method in competing_methods:
        return load_competing_method_pair(
            Path(competing_root) / method / f"group_{group_id}", fixed_id, moving_id
        )
    raise ValueError(f"Unknown registration method: {method}")


# =============================================================================
# Result transfer
# =============================================================================
def case_metadata_path(
    metadata_dir: Path,
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
) -> Path:
    return (
        Path(metadata_dir)
        / f"group_{group_id}"
        / f"{fixed_id}_{moving_id}"
        / f"{method}.npz"
    )


def save_case_metadata_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    """Save plotting metadata using NumPy arrays only; no Polygon objects."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        if arr.dtype == object:
            raise TypeError(f"Metadata field '{key}' has object dtype and is not portable")
        safe_payload[key] = arr
    np.savez_compressed(path, **safe_payload)


def load_case_metadata_npz(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def save_run_config(path: Path, config: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    path.write_text(json.dumps(convert(dict(config)), indent=2, sort_keys=True))
