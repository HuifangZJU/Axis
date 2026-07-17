import os
import glob
import math
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.lines import Line2D
import pandas as pd

NCOLS = 9

ALPHA = 0.9
GEN_USE_VALID_MASK = True
USE_LOG1P = False
# VMIN = 0.0
VMAX_PCT_REAL = 0.995
VMAX_PCT_VIRTUAL = 0.995
# VMAX_PCT_REAL = 0.96
# VMAX_PCT_VIRTUAL = 0.96

VMAX_HARD_CAP_REAL = None
VMAX_HARD_CAP_VIRTUAL = None

# layout margins for colorbars
RIGHT_MARGIN = 0.88  # leave space on the right for two colorbars
CBAR_X = 0.90
CBAR_W = 0.02

# --------------------
# Helpers
# --------------------
def load_coords_npz(NET_IN,group_id, sample_id):
    f = os.path.join(NET_IN, f"{group_id}_{sample_id}_validation.npz")
    d = np.load(f)
    return d["coord"].astype(np.float32)


def parse_z_from_filename(path):
    base = os.path.basename(path)
    s = base.split("_")[-1][:-5]
    return float(s)


def get_valid_mask(adata):
    if (not GEN_USE_VALID_MASK) or ("valid" not in adata.obs.columns):
        return np.ones(adata.n_obs, dtype=bool)
    return np.asarray(adata.obs["valid"].values, dtype=bool)

def extract_gene_vals(adata, gene):

    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in var_names.")
    j = int(np.where(adata.var_names == gene)[0][0])
    v = adata.X[:, j]
    if sp.issparse(v):
        v = np.asarray(v.toarray()).ravel().astype(np.float32)
    else:
        v = np.asarray(v).ravel().astype(np.float32)

    if USE_LOG1P:
        v = np.log1p(v)

    return v














import numpy as np
import pandas as pd
import re


def _natural_sort_key(x):
    """
    Sort strings naturally:
    '1', '2', '10' instead of '1', '10', '2'.
    """
    x = str(x)
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", x)
    ]


def extract_obs_vals(adata,obs_key):
    """
    Convert an obs column to numeric values for plotting/evaluation.

    Rules:
    1. Numeric columns stay numeric.
    2. Numeric-like categorical/string labels such as '0', '1', '12'
       are converted to their actual numeric values.
    3. Non-numeric string labels are converted to stable categorical codes
       using natural sorting.
    """

    # Convert to pandas Series if needed.
    s = adata.obs[obs_key]
    s = pd.Series(s)

    # Keep missing mask.
    missing = s.isna()

    # --------------------------------------------------
    # Case 1: already numeric
    # --------------------------------------------------
    if pd.api.types.is_numeric_dtype(s):
        vals = s.astype(np.float32).to_numpy()
        vals[missing.to_numpy()] = np.nan

        meta = {
            "kind": "numeric",
            "categories": None,
            "label_to_code": None,
            "is_discrete": False,
        }
        return vals, meta

    # --------------------------------------------------
    # Case 2: categorical/object/string but numeric-like
    # Example: '0', '1', '10', '12'
    # --------------------------------------------------
    s_str = s.astype("string")

    numeric_vals = pd.to_numeric(s_str, errors="coerce")

    non_missing = ~missing
    all_non_missing_numeric = numeric_vals[non_missing].notna().all()

    if all_non_missing_numeric:
        vals = numeric_vals.astype(np.float32).to_numpy()
        vals[missing.to_numpy()] = np.nan
        # categories are the true numeric label values, sorted numerically
        cats = sorted(np.unique(vals[~np.isnan(vals)]).tolist())
        meta = {
            "kind": "numeric_categorical",
            "categories": cats,
            "label_to_code": {str(int(c)) if float(c).is_integer() else str(c): c for c in cats},
            "is_discrete": True,
        }
        return vals, meta

    # --------------------------------------------------
    # Case 3: real categorical strings
    # Example: 'Astro', 'Micro', 'Layer2'
    # --------------------------------------------------
    valid_labels = s_str[~missing].astype(str).to_numpy()
    categories = sorted(set(valid_labels), key=_natural_sort_key)

    label_to_code = {
        lab: i for i, lab in enumerate(categories)
    }

    vals = np.full(len(s), np.nan, dtype=np.float32)

    for i, x in enumerate(s_str):
        if pd.isna(x):
            continue
        vals[i] = label_to_code[str(x)]

    meta = {
        "kind": "categorical",
        "categories": categories,
        "label_to_code": label_to_code,
        "is_discrete": True,
    }
    return vals, meta


# def extract_obs_vals(
#     adata,
#     obs_key,
#     use_log1p_continuous=False,
# ):
#     """
#     Extract obs values in a visualization-friendly numeric format.
#
#     Returns
#     -------
#     vals : np.ndarray, shape (n_obs,), float32
#         Numeric values ready for scatter(c=vals, ...).
#         For categorical attributes, this is category code as float32.
#         Missing values are returned as np.nan.
#     meta : dict
#         {
#             "kind": "continuous" | "categorical" | "boolean",
#             "categories": list[str] or None,
#             "is_discrete": bool,
#         }
#     """
#
#     if obs_key not in adata.obs.columns:
#         raise ValueError(f"obs key '{obs_key}' not found in adata.obs.")
#
#     s = adata.obs[obs_key]
#
#     # ---- boolean ----
#     if pd.api.types.is_bool_dtype(s):
#         vals = s.astype(np.float32).to_numpy()
#         meta = {
#             "kind": "boolean",
#             "categories": ["False", "True"],
#             "is_discrete": True,
#         }
#         return vals, meta
#
#     # ---- numeric continuous ----
#     if pd.api.types.is_numeric_dtype(s):
#         vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)
#         if use_log1p_continuous:
#             valid = np.isfinite(vals)
#             if np.any(vals[valid] < 0):
#                 raise ValueError(f"obs['{obs_key}'] contains negative values; cannot apply log1p safely.")
#             vals[valid] = np.log1p(vals[valid]).astype(np.float32)
#         meta = {
#             "kind": "continuous",
#             "categories": None,
#             "is_discrete": False,
#         }
#         return vals, meta
#
#     # ---- categorical / object / string ----
#
#     cat = pd.Categorical(s)
#     codes = cat.codes.astype(np.float32)
#     codes[codes < 0] = np.nan   # missing values become NaN
#
#
#
#     meta = {
#         "kind": "categorical",
#         "categories": None,
#         "is_discrete": True,
#     }
#     return codes, meta

def robust_vmax(vals_list, pct, hard_cap):
    if len(vals_list) == 0:
        return 1.0
    allv = np.concatenate(vals_list, axis=0)
    vmax = float(np.quantile(allv, pct)) if allv.size > 0 else 1.0
    if hard_cap is not None:
        vmax = min(vmax, float(hard_cap))
    return max(vmax, 1e-6)

def robust_vrange(vals_list, low_pct=0.01, high_pct=0.99, hard_cap=None, hard_floor=None):
    if len(vals_list) == 0:
        return 0.0, 1.0

    allv = np.concatenate(vals_list, axis=0)
    if allv.size == 0:
        return 0.0, 1.0

    vmin = float(np.quantile(allv, low_pct))
    vmax = float(np.quantile(allv, high_pct))

    if hard_floor is not None:
        vmin = max(vmin, float(hard_floor))
    if hard_cap is not None:
        vmax = min(vmax, float(hard_cap))

    if vmin >= vmax:
        vmax = vmin + 1e-6

    return vmin, vmax

def plot_panel(ax, coords, vals, title, norm, s, invert_y, xlim=None, ylim=None,danshedline=None):
    # all_labels = [0,1,2,3,4,5,6,7,8,9,10,11]
    # all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12]
    # base_cmap = plt.get_cmap("tab20")
    #
    # label_to_color = {
    #     lab: base_cmap(i)
    #     for i, lab in enumerate(all_labels)
    # }
    # colors = [label_to_color[v] for v in vals]
    # ax.scatter(
    #     coords[:, 0],
    #     coords[:, 1],
    #     c=colors,
    #     s=s,
    #     alpha=ALPHA,
    #     linewidths=0,
    #     rasterized=True
    # )


    # ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=s, alpha=ALPHA,cmap='tab20',
    #            norm=norm, linewidths=0, rasterized=True)
    ax.scatter(coords[:, 1], coords[:, 0], c=vals, s=s, alpha=ALPHA,cmap="coolwarm",
               norm=norm, linewidths=0, rasterized=True)
    # ax.scatter(coords[:, 1], coords[:, 0], c=vals, s=s, alpha=ALPHA, cmap="Reds",
    #            norm=norm, linewidths=0, rasterized=True)

    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    pad_ratio=0.12




    if xlim is not None:
        ax.set_xlim(xlim)          # e.g. (xmin, xmax)
        ax.set_ylim(ylim)  # e.g. (ymin, ymax)
    else:

        min_limit = np.min([ np.min(coords[:, 1])])
        max_limit = np.max([ np.max(coords[:, 1])])
        pad = pad_ratio * (max_limit - min_limit)
        y_length = 2*pad +max_limit-min_limit
        x_middle = 0.5*(np.min([ np.min(coords[:, 0])])+np.max([ np.max(coords[:, 0])]))

        ax.set_xlim([x_middle-y_length/2, x_middle+y_length/2])
        ax.set_ylim([min_limit - pad, max_limit + pad])
    if invert_y:
        ax.invert_yaxis()

    if danshedline:
        for spine in ax.spines.values():
            spine.set_linestyle((0, (6, 4)))   # 6 on, 6 off
            spine.set_linewidth(1.0)
            spine.set_edgecolor("black")
    ax.invert_xaxis()


def standardize_cluster_categories(ad, cluster_key="original_clusters", labels=None):
    if labels is None:
        labels = [str(i) for i in range(13)]

    ad.obs[cluster_key] = ad.obs[cluster_key].astype(str)
    ad.obs[cluster_key] = pd.Categorical(
        ad.obs[cluster_key],
        categories=labels,
        ordered=True,
    )
    return ad

def visualize_slices(NET_IN,ORIG_SAMPLES,ORIG_Z_UM,H5AD_DIR,GROUP_ID,GENE,GEN_PATTERN,OUT_PNG,S_REAL,S_VIRTUAL,INVERT_Y,H5ID=None):
    real_panels = []
    real_vals_for_scale = []

    virtual_panels = []
    virtual_vals_for_scale = []

    for sid in ORIG_SAMPLES:
        t = ORIG_Z_UM[sid]
        if H5ID:
            ad = sc.read_h5ad(os.path.join(H5AD_DIR, f"{H5ID}_{t}.h5ad"))
        else:
            try:
                ad = sc.read_h5ad(os.path.join(H5AD_DIR,  f"{GROUP_ID}_{t}.h5ad"))
            except FileNotFoundError:
                ad = sc.read_h5ad(os.path.join(H5AD_DIR, f"{GROUP_ID}_{sid}.h5ad"))


        coords = load_coords_npz(NET_IN,GROUP_ID, sid)
        try:
            vals = extract_gene_vals(ad, GENE)
        except ValueError:
            try:
                vals, meta = extract_obs_vals(ad, GENE)
            except ValueError:
                break
        # domain_ids = ['8', '11']
        # mask = ad.obs["axis_joint_domain"].isin(domain_ids)
        # coords = coords[mask]
        # vals = vals[mask]



        z = ORIG_Z_UM.get(sid, None)
        # title = f"Real ({z}um)" if z is not None else f"Real (sample {sid})"
        title = f"Real ({z}m)" if z is not None else f"Real (sample {sid})"
        real_panels.append({"title": title, "coords": coords, "vals": vals})
        real_vals_for_scale.append(vals)

    # --------------------
    # Load Virtual
    # --------------------
    gen_files = sorted(glob.glob(GEN_PATTERN), key=parse_z_from_filename)

    for f in gen_files:
        z = parse_z_from_filename(f)
        # if z not in [80,160,240]:
        #     continue

        ad = sc.read_h5ad(f)


        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)
        try:
            vals = extract_gene_vals(ad, GENE)
        except ValueError:
            vals, meta = extract_obs_vals(ad, GENE)
        mask_valid = get_valid_mask(ad)
        # domain_ids = ['8', '11']
        # domain_ids=['1','4','6','10']
        # mask_domain = ad.obs["axis_joint_domain"].isin(domain_ids)
        # mask = mask_domain*mask_valid
        mask = mask_valid
        coords = coords[mask]
        vals = vals[mask]
        # print(np.min(coords,axis=0))
        # print(np.max(coords,axis=0))


        # virtual_panels.append({"title": f"Virtual ({z}um)", "coords": coords, "vals": vals})
        virtual_panels.append({"title": f"Virtual ({z}m)", "coords": coords, "vals": vals})
        virtual_vals_for_scale.append(vals)

    print(f"[INFO] real={len(real_panels)} virtual={len(virtual_panels)}")

    # norms
    # vmax_real = robust_vmax(real_vals_for_scale, VMAX_PCT_REAL, VMAX_HARD_CAP_REAL)
    # vmax_virtual = robust_vmax(virtual_vals_for_scale, VMAX_PCT_VIRTUAL, VMAX_HARD_CAP_VIRTUAL)
    # norm_real = Normalize(vmin=float(VMIN), vmax=float(vmax_real))
    # norm_virtual = Normalize(vmin=float(VMIN), vmax=float(vmax_virtual))
    #
    # print(f"[SCALE] Real: vmin={VMIN}, vmax={vmax_real:.4f}")
    # print(f"[SCALE] Virtual: vmin={VMIN}, vmax={vmax_virtual:.4f}")

    vmin_real, vmax_real = robust_vrange(real_vals_for_scale, 1-VMAX_PCT_REAL, VMAX_PCT_REAL)
    vmin_virtual, vmax_virtual = robust_vrange(virtual_vals_for_scale, 1-VMAX_PCT_VIRTUAL, VMAX_PCT_VIRTUAL)

    norm_real = Normalize(vmin=float(vmin_real), vmax=float(vmax_real))
    norm_virtual = Normalize(vmin=float(vmin_virtual), vmax=float(vmax_virtual))

    print(f"[SCALE] Real: vmin={vmin_real}, vmax={vmax_real:.4f}")
    print(f"[SCALE] Virtual: vmin={vmin_virtual}, vmax={vmax_virtual:.4f}")

    # --------------------
    # Figure layout
    # --------------------
    n_virtual = len(virtual_panels)
    n_rows_virtual = int(math.ceil(n_virtual / float(NCOLS))) if n_virtual > 0 else 0
    n_rows_total = 1 + n_rows_virtual

    fig_w = 3.0 * NCOLS
    fig_h = 4.5 * n_rows_total
    fig, axes = plt.subplots(n_rows_total, NCOLS, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(n_rows_total, NCOLS)

    # leave right margin for colorbars + a bit more top space for suptitle
    # fig.subplots_adjust(left=0.02, right=RIGHT_MARGIN, top=0.96, bottom=0.02, wspace=-0.8, hspace=0.0)
    fig.subplots_adjust(left=0.02, right=RIGHT_MARGIN, top=0.96, bottom=0.02, wspace=0.04, hspace=0.0)

    # Row 1: Real
    for c in range(NCOLS):
        ax = axes[0, c]
        if c < len(real_panels):
            p = real_panels[c]
            # plot_panel(ax, p["coords"], p["vals"], p["title"], norm_real, s=S_REAL, invert_y=INVERT_Y)
            plot_panel(ax, p["coords"], p["vals"], p["title"], norm_real, s=S_REAL, invert_y=INVERT_Y, xlim=(0, 1024),
                       ylim=(0, 1024))
            # plot_panel(ax, p["coords"], p["vals"], p["title"], norm_virtual, s=S_VIRTUAL, invert_y=INVERT_Y,
            #            xlim=(520, 870), ylim=(0, 700), danshedline=True)
        else:
            ax.axis("off")

    # Rows 2+: Virtual
    idx = 0
    for r in range(1, n_rows_total):
        for c in range(NCOLS):
            ax = axes[r, c]
            if idx < n_virtual:
                p = virtual_panels[idx]
                plot_panel(ax, p["coords"], p["vals"], p["title"], norm_virtual, s=S_VIRTUAL, invert_y=INVERT_Y,danshedline=True)
                # plot_panel(ax, p["coords"], p["vals"], p["title"], norm_virtual, s=S_VIRTUAL, invert_y=INVERT_Y,
                #            xlim=(520, 870), ylim=(0, 700),danshedline=True)
                idx += 1
            else:
                ax.axis("off")

    # --------------------
    # Dedicated colorbar axes (no overlap)
    # --------------------
    sm_real = ScalarMappable(norm=norm_real); sm_real.set_array([])
    sm_virtual = ScalarMappable(norm=norm_virtual); sm_virtual.set_array([])

    # colorbar vertical spans in figure coordinates:
    # - real bar near top
    # - virtual bar below it
    cax_real = fig.add_axes([CBAR_X, 0.62, CBAR_W, 0.28])    # [x, y, w, h]
    cbar_real = fig.colorbar(sm_real, cax=cax_real)
    cbar_real.set_label(f"{GENE} log1p(counts) (Real)" if USE_LOG1P else f"{GENE} (Real)", rotation=90)

    if n_rows_total > 1:
        cax_virtual = fig.add_axes([CBAR_X, 0.12, CBAR_W, 0.42])
        cbar_virtual = fig.colorbar(sm_virtual, cax=cax_virtual)
        cbar_virtual.set_label(f"{GENE} log1p(counts) (Virtual)" if USE_LOG1P else f"{GENE} (Virtual)", rotation=90)

    # fig.suptitle(f"{GENE}: Real (row 1) vs Virtual (rows 2+), separate legends", y=0.995, fontsize=12)
    if not OUT_PNG:
        plt.show()
    else:
        plt.savefig(OUT_PNG, dpi=300)
        plt.close(fig)

        print("Saved:", OUT_PNG)


# def make_rgba_with_value_alpha_abs(vals, norm, cmap_name, alpha_min, alpha_max, gamma=1.0):
#     cmap = get_cmap(cmap_name)
#     vals = np.asarray(vals, dtype=np.float32)
#
#     if hasattr(cmap, "colors"):
#         colors = np.asarray(cmap.colors, dtype=np.float32)
#         idx = np.clip(vals.astype(int), 0, len(colors) - 1)
#         base = colors[idx]
#
#         if base.shape[1] == 3:
#             rgba = np.concatenate(
#                 [base, np.ones((base.shape[0], 1), dtype=np.float32)],
#                 axis=1
#             )
#         else:
#             rgba = base.copy()
#
#         # for categorical labels, use constant alpha
#         rgba[:, 3] = alpha_max
#     else:
#         frac = np.clip(norm(vals), 0.0, 1.0)
#         rgba = np.asarray(cmap(frac), dtype=np.float32)
#         frac_alpha = np.clip(np.abs(norm(vals)), 0.0, 1.0) ** gamma
#         alpha = alpha_min + (alpha_max - alpha_min) * frac_alpha
#         rgba[:, 3] = alpha
#     return rgba
def make_rgba_with_value_alpha_abs(vals, norm, cmap_name, alpha_min, alpha_max, gamma=1.0):
    cmap = get_cmap(cmap_name)


    vals = np.asarray(vals, dtype=np.float32)

    # For color: keep the original signed normalization
    frac = norm(vals)
    frac = np.clip(frac, 0.0, 1.0)
    rgba = cmap(frac)

    # For alpha: use absolute distance from real zero
    abs_vals = np.abs(vals)

    # Normalize abs(vals) to [0, 1] using the largest magnitude in the data
    abs_max = np.max(abs_vals) if abs_vals.size > 0 else 1.0
    if abs_max < 1e-12:
        alpha_frac = np.zeros_like(abs_vals, dtype=np.float32)
    else:
        alpha_frac = abs_vals / abs_max

    alpha = alpha_min + (alpha_frac ** gamma) * (alpha_max - alpha_min)
    rgba[:, 3] = np.clip(alpha, 0.0, 1.0)

    return rgba

def scatter_slice_horizontal_depth(
    ax,
    coords,
    vals,
    z_um,
    norm,
    cmap_name,
    s,
    alpha_max,
    label_prefix,
    DEPTH_SCALE,
    ALPHA_MIN,
    ALPHA_GAMMA
):
    # Horizontal stacking direction:
    #   horizontal axis = slice depth (original z * DEPTH_SCALE)
    #   vertical tissue plane = original x/y
    #
    # Use depth on X so the slice progression reads left-to-right.
    depth_x = np.full(coords.shape[0], z_um * DEPTH_SCALE, dtype=np.float32)
    tissue_y = coords[:, 0]
    tissue_z = coords[:, 1]

    rgba = make_rgba_with_value_alpha_abs(
        vals=vals,
        norm=norm,
        cmap_name=cmap_name,
        alpha_min=ALPHA_MIN,
        alpha_max=alpha_max,
        gamma=ALPHA_GAMMA,
    )
    # rgba = make_rgba_with_value_alpha(
    #     vals=vals,
    #     norm=norm,
    #     cmap_name=cmap_name,
    #     alpha_min=ALPHA_MIN,
    #     alpha_max=alpha_max,
    #     gamma=ALPHA_GAMMA,
    # )

    ax.scatter(
        depth_x,
        tissue_y,
        tissue_z,
        c=rgba,
        s=s,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )

    # slice label
    # ax.text(
    #     z_um * DEPTH_SCALE,
    #     X_LIM[1]-100 ,
    #     Y_LIM[1]+ 170,
    #     f"{label_prefix} {z_um:.2f} month",
    #     fontsize=8,
    # )

def plot_3d(GEN_PATTERN,GENE,DEPTH_SCALE,X_LIM,Y_LIM,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA,CMAP="viridis",INVERT_Y=False,INVERT_X=False):
    # # --------------------
    # # Load Real
    # # --------------------
    # real_panels = []
    # real_vals_for_scale = []
    #
    # for sid in ORIG_SAMPLES:
    #     z = ORIG_Z_UM[sid]
    #     ad = sc.read_h5ad(os.path.join(H5AD_DIR, f"Tg_{z}.h5ad"))
    #     coords = load_coords_npz(GROUP_ID, sid)
    #     vals = extract_gene_vals(ad, GENE)
    #
    #     n = min(coords.shape[0], vals.shape[0])
    #     coords = coords[:n]
    #     vals = vals[:n]
    #     coords, vals = maybe_downsample(coords, vals, MAX_POINTS_PER_SLICE, RANDOM_SEED)
    #
    #     real_panels.append({
    #         "title": f"Real ({z}um)",
    #         "coords": coords,
    #         "vals": vals,
    #         "z": float(z),
    #     })
    #     real_vals_for_scale.append(vals)

    # --------------------
    # Load Virtual
    # --------------------
    gen_files = sorted(glob.glob(GEN_PATTERN), key=parse_z_from_filename)
    virtual_panels = []
    virtual_vals_for_scale = []


    for f in gen_files:
        z = parse_z_from_filename(f)
        ad = sc.read_h5ad(f)

        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)
        if GENE is None:
            vals = np.full(coords.shape[0], 1, dtype=float)
        else:
            try:
                vals = extract_gene_vals(ad, GENE)
            except ValueError:
                vals, meta = extract_obs_vals(ad, GENE)
        mask = get_valid_mask(ad)

        coords = coords[mask]
        vals = vals[mask]
        virtual_panels.append({
            "title": f"Virtual ({z}month)",
            "coords": coords,
            "vals": vals,
            "z": float(-z),
            # "z": float(5.7),
        })

        virtual_vals_for_scale.append(vals)

    # print(f"[INFO] real={len(real_panels)} virtual={len(virtual_panels)}")

    # --------------------
    # Norms
    # --------------------
    # # vmax_real = robust_vmax(real_vals_for_scale, VMAX_PCT_REAL, VMAX_HARD_CAP_REAL)
    # vmax_virtual = robust_vmax(virtual_vals_for_scale, VMAX_PCT_VIRTUAL, VMAX_HARD_CAP_VIRTUAL)
    #
    # # norm_real = Normalize(vmin=float(VMIN), vmax=float(vmax_real))
    # norm_virtual = Normalize(vmin=float(VMIN), vmax=float(vmax_virtual))
    #
    # # print(f"[SCALE] Real:    vmin={VMIN}, vmax={vmax_real:.4f}")
    # print(f"[SCALE] Virtual: vmin={VMIN}, vmax={vmax_virtual:.4f}")

    # vmin_real, vmax_real = robust_vrange(real_vals_for_scale, 1 - VMAX_PCT_REAL, VMAX_PCT_REAL)
    vmin_virtual, vmax_virtual = robust_vrange(virtual_vals_for_scale, 1 - VMAX_PCT_VIRTUAL, VMAX_PCT_VIRTUAL)

    # norm_real = Normalize(vmin=float(vmin_real), vmax=float(vmax_real))
    norm_virtual = Normalize(vmin=float(vmin_virtual), vmax=float(vmax_virtual))

    # print(f"[SCALE] Real: vmin={vmin_real}, vmax={vmax_real:.4f}")
    print(f"[SCALE] Virtual: vmin={vmin_virtual}, vmax={vmax_virtual:.4f}")

    # --------------------
    # 3D Figure
    # --------------------
    fig = plt.figure(figsize=(20, 9))
    ax = fig.add_subplot(111, projection="3d")

    # make the main 3D axes actually large
    ax.set_position([0.03, 0.08, 0.80, 0.84])

    # scientific-looking projection
    ax.set_proj_type("ortho")

    # set your limits tightly
    all_z_um = [p["z"] for p in virtual_panels]
    xmin_depth = min(all_z_um) * DEPTH_SCALE
    xmax_depth = max(all_z_um) * DEPTH_SCALE
    #
    # ax.set_xlim(xmin_depth, xmax_depth)
    ax.set_ylim(X_LIM)
    ax.set_zlim(Y_LIM)

    # visual aspect, not data aspect
    ax.set_box_aspect((10, 1.3, 1.3))


    # real
    # for p in real_panels:
    #     scatter_slice_horizontal_depth(
    #         ax=ax,
    #         coords=p["coords"],
    #         vals=p["vals"],
    #         z_um=p["z"],
    #         norm=norm_real,
    #         cmap_name="viridis",
    #         s=S_REAL,
    #         alpha_max=ALPHA_MAX_REAL,
    #         label_prefix="Real",
    #             DEPTH_SCALE,
    #             ALPHA_MIN,
    #             ALPHA_GAMMA
    #     )

    # virtual

    for p in virtual_panels:
        scatter_slice_horizontal_depth(
            ax=ax,
            coords=p["coords"],
            vals=p["vals"],
            z_um=p["z"],
            norm=norm_virtual,
            cmap_name=CMAP,
            s=S_VIRTUAL,
            alpha_max=ALPHA_MAX_VIRTUAL,
            label_prefix="Virtual",
            DEPTH_SCALE=DEPTH_SCALE,
            ALPHA_MIN=ALPHA_MIN,
            ALPHA_GAMMA=ALPHA_GAMMA
        )

    # axes
    # all_z_um = [p["z"] for p in real_panels] + [p["z"] for p in virtual_panels]
    # xmin_depth = min(all_z_um) * DEPTH_SCALE if all_z_um else 0.0
    # xmax_depth = max(all_z_um) * DEPTH_SCALE if all_z_um else 1.0
    # xpad = 0.08 * max(xmax_depth - xmin_depth, 1.0)
    #
    # ax.set_xlim(xmin_depth - xpad, xmax_depth + xpad)

    if INVERT_Y:
        ax.set_zlim(Y_LIM[::-1])
    else:
        ax.set_zlim(Y_LIM)

    if INVERT_X:
        ax.set_ylim(X_LIM[::-1])
    else:
        ax.set_ylim(X_LIM)

    # ax.set_zlim(Y_LIM)

    ax.set_xlabel(f"slice depth (original z * {DEPTH_SCALE:g})")
    ax.set_ylabel("x")
    ax.set_zlabel("y")

    ax.view_init(elev=ELEV, azim=AZIM)

    # keep depth wide and tissue plane readable
    # try:
    #     ax.set_box_aspect((
    #         max(xmax_depth - xmin_depth, 1.0) + 2 * xpad,
    #         X_LIM[1] - X_LIM[0],
    #         Y_LIM[1] - Y_LIM[0],
    #     ))
    # except Exception:
    #     pass
    ax.set_aspect("auto")
    ax.grid(False)

    fig.suptitle(
        f"{GENE}: 3D stack with horizontal slice depth and value-dependent transparency",
        y=0.97,
        fontsize=14,
    )

    # legend for slice type
    legend_handles = [
        # Line2D([0], [0], marker='o', color='w', label='Real slices',
        #        markerfacecolor='gray', markersize=8, alpha=0.9),
        Line2D([0], [0], marker='o', color='w', label='Virtual slices',
               markerfacecolor='gray', markersize=8, alpha=0.6),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98))
    ax.set_proj_type("ortho")
    # # separate colorbars
    # sm_real = ScalarMappable(norm=norm_real, cmap="viridis")
    # sm_real.set_array([])
    sm_virtual = ScalarMappable(norm=norm_virtual, cmap="plasma")
    sm_virtual.set_array([])

    fig.subplots_adjust(left=0.02, right=0.84, top=0.94, bottom=0.04)

    # cax_real = fig.add_axes([0.87, 0.58, 0.02, 0.25])
    # cbar_real = fig.colorbar(sm_real, cax=cax_real)
    # cbar_real.set_label(
    #     f"{GENE} log1p(counts) (Real)" if USE_LOG1P else f"{GENE} (Real)",
    #     rotation=90
    # )

    cax_virtual = fig.add_axes([0.87, 0.18, 0.02, 0.25])
    cbar_virtual = fig.colorbar(sm_virtual, cax=cax_virtual)
    cbar_virtual.set_label(
        f"{GENE} log1p(counts) (Virtual)" if USE_LOG1P else f"{GENE} (Virtual)",
        rotation=90
    )
    plt.show()



def downsample_adata(adata, dsrate):
    n_sub = max(1, int(adata.n_obs / dsrate))
    print(f"Downsampling from {adata.n_obs} → {n_sub} cells")

    # deterministic: evenly spaced indices
    step = max(1, int(np.floor(adata.n_obs / n_sub)))
    subset_idx = np.arange(0, adata.n_obs, step)[:n_sub]

    adata_sub = adata[subset_idx].copy()
    return adata_sub

def plot_channel_3d(GEN_FILES,GENE_set,DEPTH_SCALE,X_LIM,Y_LIM,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA,INVERT_Y,INVERT_X=False):
    virtual_panels = []
    virtual_vals_for_scale = []
    channel_z=0
    cmapid=0
    # cmaps=["Blues","Oranges"]
    cmaps = ["viridis", "Oranges"]
    for f,GENE in zip (GEN_FILES,GENE_set):

        ad = sc.read_h5ad(f)
        # ad = downsample_adata(ad,20000)
        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)
        for gene in GENE:
            try:
                vals = extract_gene_vals(ad, GENE)
            except ValueError:
                vals, meta = extract_obs_vals(ad, GENE)
            if cmapid==0:
                vals = vals*2
            virtual_panels.append({
                "title": f"Virtual ({gene})",
                "coords": coords,
                "vals": vals,
                "z": float(channel_z),
                'cmap':cmaps[cmapid]

            })
            virtual_vals_for_scale.append(vals)
            channel_z = channel_z+1
        cmapid = cmapid+1
    # --------------------
    # Norms
    # --------------------

    vmin_virtual, vmax_virtual = robust_vrange(virtual_vals_for_scale, 1 - VMAX_PCT_VIRTUAL, VMAX_PCT_VIRTUAL)
    norm_virtual = Normalize(vmin=float(vmin_virtual), vmax=float(vmax_virtual))
    print(f"[SCALE] Virtual: vmin={vmin_virtual}, vmax={vmax_virtual:.4f}")

    # vmax_virtual = robust_vmax(virtual_vals_for_scale, VMAX_PCT_VIRTUAL, VMAX_HARD_CAP_VIRTUAL)
    # norm_virtual = Normalize(vmin=float(VMIN), vmax=float(vmax_virtual))
    # print(f"[SCALE] Virtual: vmin={VMIN}, vmax={vmax_virtual:.4f}")

    # --------------------
    # 3D Figure
    # --------------------
    fig = plt.figure(figsize=(20, 9))
    ax = fig.add_subplot(111, projection="3d")

    # make the main 3D axes actually large
    ax.set_position([0.03, 0.08, 0.80, 0.84])

    # scientific-looking projection
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.3, 1.3, 1.3))
    # ax.set_box_aspect((1.3, 1.3, 1.3))

    # virtual
    for p in virtual_panels:
        scatter_slice_horizontal_depth(
            ax=ax,
            coords=p["coords"],
            vals=p["vals"],
            z_um=p["z"],
            norm=norm_virtual,
            cmap_name=p['cmap'],
            s=S_VIRTUAL,
            alpha_max=ALPHA_MAX_VIRTUAL,
            label_prefix="Virtual",
            DEPTH_SCALE=DEPTH_SCALE,
            ALPHA_MIN=ALPHA_MIN,
            ALPHA_GAMMA=ALPHA_GAMMA
        )


    if INVERT_Y:
        ax.set_zlim(Y_LIM[::-1])
    else:
        ax.set_zlim(Y_LIM)

    if INVERT_X:
        ax.set_ylim(X_LIM[::-1])
    else:
        ax.set_ylim(X_LIM)

    # ax.set_xlim(-10,25)

    ax.set_xlabel(f"slice depth (original z * {DEPTH_SCALE:g})")
    ax.set_ylabel("x")
    ax.set_zlabel("y")

    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_aspect("auto")
    ax.grid(False)

    fig.suptitle(
        f"{GENE}: 3D stack with horizontal slice depth and value-dependent transparency",
        y=0.97,
        fontsize=14,
    )

    # legend for slice type
    legend_handles = [

        Line2D([0], [0], marker='o', color='w', label='Virtual slices',
               markerfacecolor='gray', markersize=8, alpha=0.6),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98))
    ax.set_proj_type("ortho")
    sm_virtual = ScalarMappable(norm=norm_virtual, cmap="plasma")
    sm_virtual.set_array([])

    fig.subplots_adjust(left=0.02, right=0.84, top=0.94, bottom=0.04)

    cax_virtual = fig.add_axes([0.87, 0.18, 0.02, 0.25])
    cbar_virtual = fig.colorbar(sm_virtual, cax=cax_virtual)
    cbar_virtual.set_label(
        f"{GENE} log1p(counts) (Virtual)" if USE_LOG1P else f"{GENE} (Virtual)",
        rotation=90
    )
    plt.show()
