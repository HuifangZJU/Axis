#!/usr/bin/env python
"""
Unify all slice coordinates in a group to a single reference slice,
using AXIS pairwise outputs in group_dir and explicit sample IDs.

Pairwise outputs expected in group_dir:
    fixed_{F}_moving_{M}_flow.npy          # flow from moving M to fixed F
    fixed_{F}_moving_{M}_warped_coor.npy   # coordinates of slice M in frame F

Unified coordinates are saved to:
    group_dir / "unified" / f"coords_slice{sid}_to_{ref_id}.npy"

Example usage:

    from unify_axis import unify_group_to_reference

    group_dir = "/media/huifang/data/registration/SCC/huifang/output/group_2"
    sample_ids = [0, 1, 2, 3]

    # Direct mode (default): min(sample_ids) as reference
    unify_group_to_reference(group_dir, sample_ids=sample_ids)

    # Custom cascade:
    paths = {
        0: [0],
        1: [1, 0],
        2: [2, 1, 0],
        3: [3, 0],
    }
    unify_group_to_reference(group_dir, sample_ids=sample_ids, paths=paths)
"""


from typing import Dict, List, Tuple, Optional, Union
from axis.coordinate_utils import find_inverse_warp_coords
from pathlib import Path
import json
import numpy as np
import scanpy as sc

# ---------------- Pairwise loading helpers ---------------- #

def _load_pairwise_files(
    group_dir: Path,
    sample_ids: List[int],
) -> Tuple[List[int], Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    """
    Load pairwise flow and warped_coor files for the given sample_ids.

    Expects files:
        fixed_{F}_moving_{M}_flow.npy
        fixed_{F}_moving_{M}_warped_coor.npy

    for all F, M in sample_ids with F != M.

    Returns
    -------
    slice_ids : sorted list[int]
        Just the sorted sample_ids.
    flows : dict[(int, int), np.ndarray]
        flows[(fixed, moving)] = flow array.
    warped_coords : dict[(int, int), np.ndarray]
        warped_coords[(fixed, moving)] = (N_spots_moving, 2) coordinates.
    """
    flows: Dict[Tuple[int, int], np.ndarray] = {}
    warped_coords: Dict[Tuple[int, int], np.ndarray] = {}

    for fixed_id in sample_ids:
        for moving_id in sample_ids:
            if fixed_id == moving_id:
                continue

            flow_path = group_dir / f"{fixed_id}_{moving_id}_flow.npy"
            warped_path = group_dir / f"{fixed_id}_{moving_id}_warped_coor.npy"

            if not flow_path.is_file():
                print(
                    f"[WARN] Missing flow file: {flow_path}. "
                    f"Skipping pair fixed={fixed_id}, moving={moving_id}."
                )
            else:
                flows[(fixed_id, moving_id)] = np.load(flow_path)

            if not warped_path.is_file():
                print(
                    f"[WARN] Missing warped coords file: {warped_path}. "
                    f"Skipping pair fixed={fixed_id}, moving={moving_id}."
                )
            else:
                warped_coords[(fixed_id, moving_id)] = np.load(warped_path)

    slice_ids = sorted(sample_ids)
    if not flows and not warped_coords:
        raise RuntimeError(f"No pairwise files found for sample_ids={sample_ids} in {group_dir}")

    return slice_ids, flows, warped_coords


def _infer_direct_paths(
    slice_ids: List[int],
    warped_coords: Dict[Tuple[int, int], np.ndarray],
) -> Dict[int, List[int]]:
    """
    Build default direct-mode paths: each slice k has path [k, ref_id],
    where ref_id is the smallest slice id and a direct warped_coor exists.

    Returns
    -------
    paths : dict[int, list[int]]
        slice_id -> path (list of slice ids from slice_id to ref)
    """
    ref_id = slice_ids[0]  # smallest id as reference
    paths: Dict[int, List[int]] = {}

    for sid in slice_ids:
        if sid == ref_id:
            paths[sid] = [sid]
        else:
            if (ref_id, sid) not in warped_coords:
                raise RuntimeError(
                    f"No direct warped coords for fixed={ref_id}, moving={sid}; "
                    f"cannot form direct path in direct mode."
                )
            paths[sid] = [sid, ref_id]

    return paths


def _deduce_reference_from_paths(paths: Dict[int, List[int]]) -> int:
    """
    Deduce unique reference slice from paths: require that all paths end
    with the same slice id.
    """
    refs = {p[-1] for p in paths.values()}
    if len(refs) != 1:
        raise ValueError(f"Paths must all end in the same reference slice; got {refs}")
    return next(iter(refs))


# ---------------- Main unify function ---------------- #

def unify_group_to_reference(
    group_dir: str,
    sample_ids: List[int],
    paths: Optional[Dict[int, List[int]]] = None,
    ref_coords: Optional[np.ndarray] = None,
    save_meta: bool = True,
    unified_subdir: str = "unified",
):
    """
    Unify all slices in a group to a single reference slice, using AXIS pairwise
    outputs in group_dir, and save unified coordinates for *all* slices
    (including the reference) into one folder.

    Parameters
    ----------
    group_dir : str
        Path to the group folder, e.g. ".../output/group_2".
        Must contain the pairwise AXIS outputs:
            fixed_{F}_moving_{M}_flow.npy
            fixed_{F}_moving_{M}_warped_coor.npy
    sample_ids : list[int]
        List of slice IDs to consider, e.g. [0, 1, 2, 3].
    paths : dict[int, list[int]] or None
        If provided: explicit cascade paths for each slice.
        Example:
            {0: [0], 1: [1,0], 2: [2,1,0], 3: [3,0]}
        Each path is a list [start, ..., ref] describing how to go from
        that slice to the reference.

        If None: direct mode is used, where the smallest slice id is the
        reference and each slice k has path [k, ref].
    ref_coords : np.ndarray or None
        Optional coordinates for the reference slice in its own frame
        (e.g. from network_input or from the original h5ad).
        If provided, they will be saved as:
            coords_slice{ref_id}_to_{ref_id}.npy
        so that later you have a complete coordinate set for all slices.
    save_meta : bool
        If True, write a meta.json with reference and paths.
    unified_subdir : str
        Name of the subfolder under group_dir where unified coords will be
        written. Default "unified".

    Output
    ------
    group_dir/{unified_subdir}/coords_slice{sid}_to_{ref_id}.npy
        for all slices sid for which we have coordinates in the reference frame.
        If ref_coords is provided, this includes coords_slice{ref_id}_to_{ref_id}.npy.

    Returns
    -------
    coords_dict : dict[int, np.ndarray]
        Coordinates for each slice sid in reference frame.
    """
    group_dir = Path(group_dir)
    if not group_dir.is_dir():
        raise NotADirectoryError(f"Group dir not found: {group_dir}")

    # Load all pairwise flows and warped coords, restricted to sample_ids
    slice_ids, flows, warped_coords = _load_pairwise_files(group_dir, sample_ids)
    print(f"[INFO] Found slices (from sample_ids) in {group_dir.name}: {slice_ids}")

    # Build or validate paths
    if paths is None:
        print("[INFO] No paths provided, using direct mode.")
        paths = _infer_direct_paths(slice_ids, warped_coords)
    else:
        # Coerce keys and entries to int
        paths = {int(k): [int(x) for x in v] for k, v in paths.items()}

    ref_id = _deduce_reference_from_paths(paths)
    print(f"[INFO] Using reference slice: {ref_id}")

    out_dir = group_dir / unified_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    coords_dict: Dict[int, np.ndarray] = {}

    # If we have explicit reference coordinates, save them as well
    if ref_coords is not None:
        ref_coords = np.asarray(ref_coords, dtype=np.float32)
        coords_dict[ref_id] = ref_coords
        ref_out_path = out_dir / f"coords_slice{ref_id}_to_ref.npy"
        np.save(ref_out_path, ref_coords)
        print(
            f"[INFO] Saved reference coords for slice {ref_id} to {ref_out_path} "
            f"with shape {ref_coords.shape}"
        )

    # For each non-reference slice, compute unified coords in reference frame
    for sid, path in paths.items():
        if sid not in slice_ids:
            print(f"[WARN] Path provided for sid={sid} which is not in sample_ids; skipping.")
            continue

        if sid == ref_id:
            # We already handled reference above (if ref_coords is given)
            continue

        if len(path) == 0:
            raise ValueError(f"Empty path for slice {sid}")

        if sid != path[0]:
            raise ValueError(f"Path for slice {sid} must start with {sid}, got {path}")

        if path[-1] != ref_id:
            raise ValueError(f"Path for slice {sid} must end in ref {ref_id}, got {path}")

        # -------- Step 1: initial mapping sid -> path[1] using warped_coor -------- #
        if len(path) >= 2:
            src = path[0]
            nxt = path[1]
            key = (nxt, src)  # fixed=nxt, moving=src
            if key not in warped_coords:
                raise RuntimeError(
                    f"Missing warped_coor for first step: fixed={nxt}, moving={src}"
                )
            coords_current = warped_coords[key].astype(np.float32)
            print(
                f"[INFO] Slice {sid}: initial step {src}->{nxt} via "
                f"fixed_{nxt}_moving_{src}_warped_coor.npy (coords shape={coords_current.shape})"
            )
        else:
            raise RuntimeError(f"Path for slice {sid} has length 1 but sid!=ref_id")

        # -------- Step 2: cascade through remaining nodes using flows -------- #
        for j in range(1, len(path) - 1):
            frame_src = path[j]        # current frame of coords_current
            frame_dst = path[j + 1]    # next desired frame

            if frame_dst == frame_src:
                continue

            flow_key = (frame_dst, frame_src)  # fixed=frame_dst, moving=frame_src
            if flow_key not in flows:
                raise RuntimeError(
                    f"Missing flow for step {frame_src}->{frame_dst}: "
                    f"fixed_{frame_dst}_moving_{frame_src}_flow.npy"
                )
            flow = flows[flow_key]
            print(
                f"[INFO] Slice {sid}: cascading {frame_src}->{frame_dst} "
                f"using fixed_{frame_dst}_moving_{frame_src}_flow.npy"
            )

            # Use AXIS helper to warp coords_current into frame_dst
            coords_current = find_inverse_warp_coords(coords_current, flow)
            coords_current = np.asarray(coords_current, dtype=np.float32)

        # Save result for this slice
        coords_dict[sid] = coords_current
        out_path = out_dir / f"coords_slice{sid}_to_ref.npy"
        np.save(out_path, coords_current)
        print(f"[INFO] Saved unified coords for slice {sid} -> {ref_id} to {out_path}")

    # Optional meta.json
    if save_meta:
        meta = {
            "reference_slice": ref_id,
            "paths": {int(k): v for k, v in paths.items()},
            "sample_ids": slice_ids,
            "has_ref_coords": ref_coords is not None,
            "note": (
                "coords_slice{sid}_to_{ref_id}.npy store slice sid coordinates in "
                "reference slice ref_id frame."
            ),
        }
        meta_path = out_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] Saved meta.json to {meta_path}")

    return coords_dict

