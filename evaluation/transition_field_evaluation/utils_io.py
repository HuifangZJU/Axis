
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from utils_metrics import evaluate_domain_metrics





# Coordinate normalization
OFFSET = 5.0
TARGET_RANGE = 1000.0

def ensure_flow_hw2(flow):
    """
    Accept [H, W, 2] or [2, H, W], return [H, W, 2].
    """
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3:
        raise ValueError(f"flow must be 3D, got shape {flow.shape}")

    if flow.shape[-1] == 2:
        return flow
    if flow.shape[0] == 2:
        return np.moveaxis(flow, 0, -1)

    raise ValueError(f"Cannot interpret flow shape {flow.shape}")

# =========================
# Utility helpers
# =========================
def clean_labels(labels: np.ndarray) -> np.ndarray:
    """Normalize labels into a clean NumPy array, decoding bytes if needed."""
    labels = np.asarray(labels).squeeze().ravel()

    if labels.dtype.kind == "S":
        return labels.astype(str)

    if labels.dtype == object:
        cleaned = []
        for x in labels:
            if isinstance(x, bytes):
                cleaned.append(x.decode("utf-8"))
            else:
                cleaned.append(x)
        return np.asarray(cleaned, dtype=object)

    return labels


def ensure_2d_coords(coords: np.ndarray, source: str) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"{source}: expected coords shape [N,2+], got {coords.shape}")
    return coords[:, :2]


def trim_coords_and_labels(
    coords: np.ndarray,
    labels: np.ndarray,
    source: str,
) -> Tuple[np.ndarray, np.ndarray]:
    coords = ensure_2d_coords(coords, source=source)
    labels = clean_labels(labels)

    n = min(coords.shape[0], len(labels))

    if coords.shape[0] != len(labels):
        print(
            f"[WARN] {source}: coords length {coords.shape[0]} != labels length {len(labels)}. "
            f"Using first {n}."
        )

    return coords[:n], labels[:n]


def npz_get_first(data: np.lib.npyio.NpzFile, keys: Iterable[str], source: str) -> np.ndarray:
    for key in keys:
        if key in data.files:
            return data[key]
    raise KeyError(f"{source}: none of these keys found: {list(keys)}. Available keys: {data.files}")





def normalize_pair_by_fixed_frame(
    pts_fixed: np.ndarray,
    pts_moving: np.ndarray,
    offset: float = OFFSET,
    target_range: float = TARGET_RANGE,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Normalize both fixed and moving coordinates using the fixed-slice span.

    Follows the idea you described:
      - use fixed slice extent for scale
      - allow combined minimum so both point sets land in the same normalized frame
      - map roughly into [offset, offset + target_range]

    Returns
    -------
    pts_fixed_norm, pts_moving_norm, mask_shape
        mask_shape is (height, width) = (max_y, max_x) in raster order.
    """
    pts_fixed = ensure_2d_coords(pts_fixed, "pts_fixed")
    pts_moving = ensure_2d_coords(pts_moving, "pts_moving")

    fixed_min = pts_fixed.min(axis=0)
    fixed_max = pts_fixed.max(axis=0)
    scale = fixed_max - fixed_min
    scale = np.where(scale <= 1e-8, 1.0, scale)

    pts_combined = np.vstack([pts_fixed, pts_moving])
    combined_min = pts_combined.min(axis=0)

    pts_fixed_norm = (pts_fixed - combined_min) / scale * target_range + offset
    pts_moving_norm = (pts_moving - combined_min) / scale * target_range + offset

    pts_norm_combined = np.vstack([pts_fixed_norm, pts_moving_norm])
    max_x = int(np.ceil(pts_norm_combined[:, 0].max()) + offset)
    max_y = int(np.ceil(pts_norm_combined[:, 1].max()) + offset)

    # IMPORTANT: mask shape should be (rows, cols) = (y, x)
    mask_shape = (max_y, max_x)

    return pts_fixed_norm.astype(np.float32), pts_moving_norm.astype(np.float32), mask_shape







# =========================
# Loaders
# =========================
def load_h5ad_pair(
    fixed_path: Path,
    moving_path: Path,
    coord_key: str,
    label_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ad_fixed = sc.read_h5ad(fixed_path)
    ad_moving = sc.read_h5ad(moving_path)

    if coord_key not in ad_fixed.obsm:
        raise KeyError(f"{fixed_path}: obsm['{coord_key}'] not found")
    if coord_key not in ad_moving.obsm:
        raise KeyError(f"{moving_path}: obsm['{coord_key}'] not found")
    if label_key not in ad_fixed.obs:
        raise KeyError(f"{fixed_path}: obs['{label_key}'] not found")
    if label_key not in ad_moving.obs:
        raise KeyError(f"{moving_path}: obs['{label_key}'] not found")

    pts1, label1 = trim_coords_and_labels(
        ad_fixed.obsm[coord_key],
        ad_fixed.obs[label_key].to_numpy(),
        source=str(fixed_path),
    )
    pts2, label2 = trim_coords_and_labels(
        ad_moving.obsm[coord_key],
        ad_moving.obs[label_key].to_numpy(),
        source=str(moving_path),
    )
    return pts1, pts2, label1, label2


def load_competing_method_pair(
    method_dir: Path,
    fixed_id: int,
    moving_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = method_dir / f"{fixed_id}_{moving_id}_result.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)

    pts1 = npz_get_first(data, ["pts1"], source=str(path))
    pts2 = npz_get_first(data, ["pts2"], source=str(path))
    label1 = npz_get_first(data, ["label1"], source=str(path))
    label2 = npz_get_first(data, ["label2"], source=str(path))



    pts1, label1 = trim_coords_and_labels(pts1, label1, source=f"{path}: fixed")
    pts2, label2 = trim_coords_and_labels(pts2, label2, source=f"{path}: moving")
    return pts1, pts2, label1, label2


def load_axis_validation_npz(
    path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)

    coord = npz_get_first(
        data,
        keys=["coord", "coords", "pts", "points", "spatial"],
        source=str(path),
    )

    label = npz_get_first(
        data,
        keys=[
            "label",
            "labels",
            "layer_guess_reordered",
            "layer",
            "domain",
            "domains",
            "y",
            "target",
        ],
        source=str(path),
    )

    coord, label = trim_coords_and_labels(coord, label, source=str(path))
    return coord, label


def load_axis_pair(
    group_id: int,
    fixed_id: int,
    moving_id: int,
    AXIS_NETWORK_INPUT_DIR,
    AXIS_PAIRWISE_DIR
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fixed_npz = AXIS_NETWORK_INPUT_DIR / f"{group_id}_{fixed_id}_validation.npz"
    moving_npz = AXIS_NETWORK_INPUT_DIR / f"{group_id}_{moving_id}_validation.npz"

    fixed_pts, fixed_label = load_axis_validation_npz(fixed_npz)
    _, moving_label = load_axis_validation_npz(moving_npz)


    warped_path = AXIS_PAIRWISE_DIR / f"group_{group_id}" / f"{fixed_id}_{moving_id}_warped_coor.npy"

    moving_pts_warped = np.load(warped_path, allow_pickle=True)
    moving_pts_warped = ensure_2d_coords(moving_pts_warped, source=str(warped_path))

    # moving labels should match warped coordinate order
    n = min(moving_pts_warped.shape[0], len(moving_label))
    if moving_pts_warped.shape[0] != len(moving_label):
        print(
            f"[WARN] {warped_path}: warped coords length {moving_pts_warped.shape[0]} "
            f"!= moving labels length {len(moving_label)}. Using first {n}."
        )

    moving_pts_warped = moving_pts_warped[:n]
    moving_label = moving_label[:n]

    return fixed_pts, moving_pts_warped, fixed_label, moving_label


# =========================
# Pair dispatcher
# =========================
def load_pair_for_method(
    H5AD_DIR,
    COMPETING_METHODS,
    COMPETING_ROOT,
    H5AD_COORD_KEY,
    H5AD_LABEL_KEY,
    AXIS_PAIRWISE_DIR,
    AXIS_NETWORK_INPUT_DIR,
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if method == "unaligned":
        fixed_path = H5AD_DIR / f"{group_id}_{fixed_id}.h5ad"
        moving_path = H5AD_DIR / f"{group_id}_{moving_id}.h5ad"
        return load_h5ad_pair(fixed_path, moving_path,H5AD_COORD_KEY,H5AD_LABEL_KEY)

    if method in COMPETING_METHODS:
        method_dir = COMPETING_ROOT / method / f"group_{group_id}"
        return load_competing_method_pair(method_dir, fixed_id, moving_id)

    if method == "axis":
        return load_axis_pair(group_id, fixed_id, moving_id,AXIS_NETWORK_INPUT_DIR,AXIS_PAIRWISE_DIR)

    raise ValueError(f"Unknown method: {method}")


