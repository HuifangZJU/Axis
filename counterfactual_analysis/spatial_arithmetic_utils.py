import os
import glob
import copy
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData





# ============================================================
# Helpers
# ============================================================
def parse_axis_value_from_filename(path, biological_axis_name):
    """
    Parse query axis value from file name:
      virtual_{biological_axis_name}_{value}.h5ad
    """
    base = os.path.basename(path)
    prefix = f"virtual_{biological_axis_name}_"
    suffix = ".h5ad"

    if not (base.startswith(prefix) and base.endswith(suffix)):
        raise ValueError(f"Unexpected file name format: {base}")

    s = base[len(prefix):-len(suffix)]
    return float(s)


def get_axis_value_from_adata_or_filename(path, adata, biological_axis_name):
    """
    Prefer adata.uns['query_axis_value']; fallback to filename.
    """
    if "query_axis_value" in adata.uns:
        return float(adata.uns["query_axis_value"])
    return parse_axis_value_from_filename(path, biological_axis_name)


def to_float32_X(X):
    """
    Ensure X is float32, preserving sparse/dense structure.
    """
    if sp.issparse(X):
        return X.astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def matrix_maximum(A, B):
    """
    Elementwise maximum preserving sparse when both are sparse.
    """
    if sp.issparse(A) and sp.issparse(B):
        return A.maximum(B).tocsr()
    elif (not sp.issparse(A)) and (not sp.issparse(B)):
        return np.maximum(A, B).astype(np.float32, copy=False)
    else:
        # Mixed types: convert both to dense as fallback
        if sp.issparse(A):
            A = A.toarray()
        if sp.issparse(B):
            B = B.toarray()
        return np.maximum(
            np.asarray(A, dtype=np.float32),
            np.asarray(B, dtype=np.float32),
        ).astype(np.float32, copy=False)


def matrix_subtract(A, B):
    """
    Compute A - B, preserving sparse when both are sparse.
    """
    if sp.issparse(A) and sp.issparse(B):
        return (A.tocsr() - B.tocsr()).tocsr().astype(np.float32)
    elif (not sp.issparse(A)) and (not sp.issparse(B)):
        return (np.asarray(A, dtype=np.float32) - np.asarray(B, dtype=np.float32)).astype(np.float32, copy=False)
    else:
        # Mixed types: convert both to dense as fallback
        if sp.issparse(A):
            A = A.toarray()
        if sp.issparse(B):
            B = B.toarray()
        return (np.asarray(A, dtype=np.float32) - np.asarray(B, dtype=np.float32)).astype(np.float32, copy=False)


def assert_same_geometry_and_genes(ad_ref, ad_cur, ref_path, cur_path, atol=1e-6):
    """
    Ensure analysis-mode virtual slices are directly comparable.
    """
    if ad_ref.n_obs != ad_cur.n_obs:
        raise ValueError(f"n_obs mismatch: {ref_path} vs {cur_path}")

    if ad_ref.n_vars != ad_cur.n_vars:
        raise ValueError(f"n_vars mismatch: {ref_path} vs {cur_path}")

    if not np.array_equal(np.asarray(ad_ref.var_names), np.asarray(ad_cur.var_names)):
        raise ValueError(f"var_names mismatch: {ref_path} vs {cur_path}")

    coords_ref = np.asarray(ad_ref.obsm["spatial"], dtype=np.float32)
    coords_cur = np.asarray(ad_cur.obsm["spatial"], dtype=np.float32)

    if coords_ref.shape != coords_cur.shape:
        raise ValueError(f"spatial shape mismatch: {ref_path} vs {cur_path}")

    if not np.allclose(coords_ref, coords_cur, atol=atol):
        raise ValueError(f"spatial coordinates mismatch: {ref_path} vs {cur_path}")


def load_virtual_records(input_dir, biological_axis_name):
    """
    Load all analysis-mode virtual h5ad files with axis values.
    """
    pattern = os.path.join(input_dir, f"*.h5ad")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found with pattern: {pattern}")

    records = []
    for f in files:
        ad = sc.read_h5ad(f)
        z = get_axis_value_from_adata_or_filename(f, ad, biological_axis_name)
        records.append({
            "file": f,
            "z": float(z),
            "adata": ad,
        })

    records = sorted(records, key=lambda x: x["z"])
    return records


def choose_reference_record(records, reference_axis_value):
    """
    Pick the existing slice closest to the requested reference z.
    """
    return min(records, key=lambda r: abs(r["z"] - float(reference_axis_value)))


def sanitize_uns_for_write(uns_dict):
    """
    Keep .uns simple and write-safe.
    """
    out = {}
    for k, v in dict(uns_dict).items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x.item() if isinstance(x, np.generic) else x for x in v]
        elif isinstance(v, np.ndarray):
            out[k] = v
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            # keep unknown objects as string to avoid h5ad write/read issues
            out[k] = str(v)
    return out


# ============================================================
# 1) Aggregation: max across all virtual slices
# ============================================================
def aggregate_virtual_slices_max(
    input_dir,
    biological_axis_name="dlpfc_z",
    reference_axis_value=10.0,
    output_subdir="aggregation",
    output_name=None,
):
    """
    Aggregate all analysis-mode virtual slices by taking coordinate-wise max
    on the common gene set shared by all slices.

    Keeps:
      - obs / obsm / uns from the selected reference z slice
    Replaces:
      - X with max across all slices, restricted to common genes

    Notes
    -----
    - All slices must share the same spatial geometry (same obs / coordinates layout).
    - Gene sets may differ; this function first intersects var_names across all slices.
    """
    records = load_virtual_records(input_dir, biological_axis_name)
    if len(records) == 0:
        raise ValueError(f"No virtual slices found in: {input_dir}")

    ref_rec = choose_reference_record(records, reference_axis_value)
    ref_ad = ref_rec["adata"]

    print(
        f"[AGG] using reference slice: z={ref_rec['z']:.3f}  "
        f"file={os.path.basename(ref_rec['file'])}"
    )

    # --------------------------------------------------------
    # 1) Find common genes across all slices
    # --------------------------------------------------------
    common_genes = ref_ad.var_names.copy()
    for rec in records:
        common_genes = common_genes.intersection(rec["adata"].var_names)
    if len(common_genes) == 0:
        raise ValueError("No common genes found across all virtual slices.")

    print(f"[AGG] common genes across all slices: {len(common_genes)}")

    # keep gene order following the reference slice
    common_genes = ref_ad.var_names[ref_ad.var_names.isin(common_genes)]

    # --------------------------------------------------------
    # 2) Subset reference slice to common genes
    # --------------------------------------------------------
    ref_ad_common = ref_ad[:, common_genes].copy()

    # initialize aggregation matrix
    agg_X = to_float32_X(ref_ad_common.X)

    # --------------------------------------------------------
    # 3) Aggregate remaining slices on the same common gene set
    # --------------------------------------------------------
    for rec in records:
        cur_ad = rec["adata"][:, common_genes].copy()

        # geometry check only, since genes are now harmonized
        if cur_ad.n_obs != ref_ad_common.n_obs:
            raise ValueError(
                f"Geometry mismatch in number of spots/cells:\n"
                f"  reference: {ref_rec['file']} -> n_obs={ref_ad_common.n_obs}\n"
                f"  current:   {rec['file']} -> n_obs={cur_ad.n_obs}"
            )

        if "spatial" in ref_ad_common.obsm and "spatial" in cur_ad.obsm:
            if ref_ad_common.obsm["spatial"].shape != cur_ad.obsm["spatial"].shape:
                raise ValueError(
                    f"Spatial coordinate shape mismatch:\n"
                    f"  reference: {ref_ad_common.obsm['spatial'].shape}\n"
                    f"  current:   {cur_ad.obsm['spatial'].shape}"
                )

        agg_X = matrix_maximum(agg_X, to_float32_X(cur_ad.X))

    # --------------------------------------------------------
    # 4) Write output using reference slice metadata
    # --------------------------------------------------------
    ad_out = ref_ad_common.copy()
    ad_out.X = agg_X

    uns_new = sanitize_uns_for_write(ad_out.uns)
    uns_new["derived_operation"] = "aggregation_max"
    uns_new["derived_from_n_slices"] = int(len(records))
    uns_new["derived_reference_axis_value"] = float(ref_rec["z"])
    uns_new["derived_source_axis_values"] = [float(r["z"]) for r in records]
    uns_new["derived_common_genes_only"] = True
    uns_new["derived_n_common_genes"] = int(len(common_genes))
    ad_out.uns = uns_new

    out_dir = os.path.join(input_dir, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    if output_name is None:
        output_name = f"aggregate_max_ref_{float(ref_rec['z']):.3f}.h5ad"

    out_path = os.path.join(out_dir, output_name)
    ad_out.write_h5ad(out_path)

    print(f"[AGG] saved: {out_path}")
    return out_path


# ============================================================
# 2) Adjacent differences
# ============================================================
def adjacent_virtual_differences(
    input_dir,
    biological_axis_name="dlpfc_z",
    mode="forward",   # "forward" or "inverse"
    output_subdir="differences",
):
    """
    Generate adjacent difference maps from analysis-mode virtual slices.

    mode == "forward":
      diff = X_{i+1} - X_i
      keep attrs/obs/obsm/uns from slice (i+1)

    mode == "inverse":
      diff = X_i - X_{i+1}
      keep attrs/obs/obsm/uns from slice i
    """
    if mode not in {"forward", "inverse"}:
        raise ValueError("mode must be one of {'forward', 'inverse'}")

    records = load_virtual_records(input_dir, biological_axis_name)
    if len(records) < 2:
        raise ValueError("Need at least 2 virtual slices to compute adjacent differences.")

    out_dir = os.path.join(input_dir, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    out_paths = []

    for i in range(len(records) - 1):
        rec_a = records[i]
        rec_b = records[i + 1]
        ad_a = rec_a["adata"]
        ad_b = rec_b["adata"]

        assert_same_geometry_and_genes(ad_a, ad_b, rec_a["file"], rec_b["file"])

        Xa = to_float32_X(ad_a.X)
        Xb = to_float32_X(ad_b.X)

        if mode == "forward":
            Xdiff = matrix_subtract(Xb, Xa)   # (i+1) - i
            ad_keep = ad_b.copy()
            keep_z = rec_b["z"]
            out_name = f"diff_forward_{float(rec_b['z']):.3f}_minus_{float(rec_a['z']):.3f}.h5ad"
        else:
            Xdiff = matrix_subtract(Xa, Xb)   # i - (i+1)
            ad_keep = ad_a.copy()
            keep_z = rec_a["z"]
            out_name = f"diff_inverse_{float(rec_a['z']):.3f}_minus_{float(rec_b['z']):.3f}.h5ad"

        ad_keep.X = Xdiff

        uns_new = sanitize_uns_for_write(ad_keep.uns)
        uns_new["derived_operation"] = f"adjacent_difference_{mode}"
        uns_new["derived_keep_axis_value"] = float(keep_z)
        uns_new["derived_axis_value_a"] = float(rec_a["z"])
        uns_new["derived_axis_value_b"] = float(rec_b["z"])
        uns_new["derived_file_a"] = os.path.basename(rec_a["file"])
        uns_new["derived_file_b"] = os.path.basename(rec_b["file"])
        ad_keep.uns = uns_new

        out_path = os.path.join(out_dir, out_name)
        ad_keep.write_h5ad(out_path)
        out_paths.append(out_path)
        print(f"[DIFF] saved: {out_path}")

    return out_paths

def matrix_lerp(A, B, t):
    """
    Linear interpolation between two matrices:
        (1-t) * A + t * B

    Supports dense or sparse matrices.
    """
    t = float(t)

    if sp.issparse(A) and sp.issparse(B):
        return (A.multiply(1.0 - t) + B.multiply(t)).tocsr().astype(np.float32)

    if sp.issparse(A):
        A = A.toarray()
    if sp.issparse(B):
        B = B.toarray()

    return ((1.0 - t) * np.asarray(A, dtype=np.float32) +
            t * np.asarray(B, dtype=np.float32)).astype(np.float32, copy=False)


def choose_template_ad(ad_a, ad_b, t, attr_source="nearest"):
    """
    Choose which slice provides non-X metadata (obs/obsm/uns) template.

    attr_source:
      - "nearest": use ad_a if t < 0.5 else ad_b
      - "left": always use ad_a
      - "right": always use ad_b
    """
    if attr_source == "nearest":
        return ad_a if t < 0.5 else ad_b
    elif attr_source == "left":
        return ad_a
    elif attr_source == "right":
        return ad_b
    else:
        raise ValueError("attr_source must be one of {'nearest','left','right'}")


def interpolate_adjacent_virtual_slices(
    input_dir,
    biological_axis_name="dlpfc_z",
    n_pieces=5,
    mode="forward",   # "forward" or "inverse"
    output_subdir="interpolations",
    attr_source="nearest",
    include_endpoint=False,
):
    """
    Generate equally spaced slices inside each adjacent interval.

    If n_pieces = 5, then by default the interval [x_i, x_{i+1}] is divided into
    5 equal pieces and the function writes the internal slices at:
        t = 1/5, 2/5, 3/5, 4/5
    i.e. 4 new slices between the two existing ones.

    Parameters
    ----------
    input_dir : str
        Directory containing analysis-mode virtual .h5ad slices.
    biological_axis_name : str
        Used by load_virtual_records().
    n_pieces : int
        Number of equal pieces per adjacent interval.
        Must be >= 2.
    mode : {"forward", "inverse"}
        - "forward": output fractions in increasing order: 1/n, 2/n, ..., (n-1)/n
        - "inverse": output fractions in decreasing order: (n-1)/n, ..., 2/n, 1/n
        The generated physical states are the same; only the output order/naming differs.
    output_subdir : str
        Subfolder name for saving outputs.
    attr_source : {"nearest", "left", "right"}
        Which adjacent endpoint contributes obs/obsm/uns template.
    include_endpoint : bool
        If True, also write the upper endpoint t=1.0 as an output file.
        By default False, because that slice already exists in the input set.
    """
    if mode not in {"forward", "inverse"}:
        raise ValueError("mode must be one of {'forward', 'inverse'}")
    if n_pieces < 2:
        raise ValueError("n_pieces must be >= 2")

    records = load_virtual_records(input_dir, biological_axis_name)
    if len(records) < 2:
        raise ValueError("Need at least 2 virtual slices to interpolate adjacent intervals.")

    out_dir = os.path.join(input_dir, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    out_paths = []

    # fractions inside each interval
    if include_endpoint:
        frac_list = np.arange(1, n_pieces + 1, dtype=np.float32) / float(n_pieces)
    else:
        frac_list = np.arange(1, n_pieces, dtype=np.float32) / float(n_pieces)

    if mode == "inverse":
        frac_list = frac_list[::-1]

    for i in range(len(records) - 1):
        rec_a = records[i]
        rec_b = records[i + 1]
        ad_a = rec_a["adata"]
        ad_b = rec_b["adata"]

        assert_same_geometry_and_genes(ad_a, ad_b, rec_a["file"], rec_b["file"])

        Xa = to_float32_X(ad_a.X)
        Xb = to_float32_X(ad_b.X)

        za = float(rec_a["z"])
        zb = float(rec_b["z"])

        for t in frac_list:
            t = float(t)

            # interpolate the matrix
            Xt = matrix_lerp(Xa, Xb, t)

            # physical queried axis value
            zt = za + t * (zb - za)

            # choose metadata template
            ad_template = choose_template_ad(ad_a, ad_b, t, attr_source=attr_source)
            ad_out = ad_template.copy()
            ad_out.X = Xt

            # metadata
            uns_new = sanitize_uns_for_write(ad_out.uns)
            uns_new["derived_operation"] = f"adjacent_interpolation_{mode}"
            uns_new["derived_axis_value_a"] = float(za)
            uns_new["derived_axis_value_b"] = float(zb)
            uns_new["derived_query_axis_value"] = float(zt)
            uns_new["derived_fraction"] = float(t)
            uns_new["derived_n_pieces"] = int(n_pieces)
            uns_new["derived_attr_source"] = str(attr_source)
            uns_new["derived_file_a"] = os.path.basename(rec_a["file"])
            uns_new["derived_file_b"] = os.path.basename(rec_b["file"])
            ad_out.uns = uns_new

            out_name = (
                f"interp_{mode}_"
                f"{float(za):.3f}_to_{float(zb):.3f}_"
                f"frac_{float(t):.3f}_"
                f"{biological_axis_name}_{float(zt):.3f}.h5ad"
            )
            out_path = os.path.join(out_dir, out_name)
            ad_out.write_h5ad(out_path)
            out_paths.append(out_path)
            print(f"[INTERP] saved: {out_path}")

    return out_paths