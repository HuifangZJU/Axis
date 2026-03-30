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

NCOLS = 6

ALPHA = 0.9
GEN_USE_VALID_MASK = True
USE_LOG1P = True
VMIN = 0.0
VMAX_PCT_REAL = 0.995
VMAX_PCT_VIRTUAL = 0.995
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

def extract_obs_vals(
    adata,
    obs_key,
    use_log1p_continuous=False,
    categorical_as_codes=True,
):
    """
    Extract obs values in a visualization-friendly numeric format.

    Returns
    -------
    vals : np.ndarray, shape (n_obs,), float32
        Numeric values ready for scatter(c=vals, ...).
        For categorical attributes, this is category code as float32.
        Missing values are returned as np.nan.
    meta : dict
        {
            "kind": "continuous" | "categorical" | "boolean",
            "categories": list[str] or None,
            "is_discrete": bool,
        }
    """

    if obs_key not in adata.obs.columns:
        raise ValueError(f"obs key '{obs_key}' not found in adata.obs.")

    s = adata.obs[obs_key]

    # ---- boolean ----
    if pd.api.types.is_bool_dtype(s):
        vals = s.astype(np.float32).to_numpy()
        meta = {
            "kind": "boolean",
            "categories": ["False", "True"],
            "is_discrete": True,
        }
        return vals, meta

    # ---- numeric continuous ----
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)
        if use_log1p_continuous:
            valid = np.isfinite(vals)
            if np.any(vals[valid] < 0):
                raise ValueError(f"obs['{obs_key}'] contains negative values; cannot apply log1p safely.")
            vals[valid] = np.log1p(vals[valid]).astype(np.float32)
        meta = {
            "kind": "continuous",
            "categories": None,
            "is_discrete": False,
        }
        return vals, meta

    # ---- categorical / object / string ----
    cat = pd.Categorical(s)
    codes = cat.codes.astype(np.float32)
    codes[codes < 0] = np.nan   # missing values become NaN

    meta = {
        "kind": "categorical",
        "categories": [str(x) for x in cat.categories.tolist()],
        "is_discrete": True,
    }
    return codes, meta

def robust_vmax(vals_list, pct, hard_cap):
    if len(vals_list) == 0:
        return 1.0
    allv = np.concatenate(vals_list, axis=0)
    vmax = float(np.quantile(allv, pct)) if allv.size > 0 else 1.0
    if hard_cap is not None:
        vmax = min(vmax, float(hard_cap))
    return max(vmax, 1e-6)

def plot_panel(ax, coords, vals, title, norm, s, invert_y, xlim=None, ylim=None):
    ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=s, alpha=ALPHA,
               norm=norm, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    if xlim is not None:
        ax.set_xlim(xlim)          # e.g. (xmin, xmax)
    if ylim is not None:
        ax.set_ylim(ylim)          # e.g. (ymin, ymax)

    if invert_y:
        ax.invert_yaxis()

def visualize_slices(NET_IN,ORIG_SAMPLES,ORIG_Z_UM,H5AD_DIR,GROUP_ID,GENE,GEN_PATTERN,OUT_PNG,S_REAL,S_VIRTUAL,INVERT_Y,H5ID=None):
    real_panels = []
    real_vals_for_scale = []
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

        n = min(coords.shape[0], vals.shape[0])
        coords = coords[:n]
        vals = vals[:n]

        z = ORIG_Z_UM.get(sid, None)
        title = f"Real ({z}m)" if z is not None else f"Real (sample {sid})"
        real_panels.append({"title": title, "coords": coords, "vals": vals})
        real_vals_for_scale.append(vals)

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
        try:
            vals = extract_gene_vals(ad, GENE)
        except ValueError:
            vals, meta = extract_obs_vals(ad, GENE)
        mask = get_valid_mask(ad)

        coords = coords[mask]
        vals = vals[mask]

        virtual_panels.append({"title": f"Virtual ({z}m)", "coords": coords, "vals": vals})
        virtual_vals_for_scale.append(vals)

    print(f"[INFO] real={len(real_panels)} virtual={len(virtual_panels)}")

    # norms
    vmax_real = robust_vmax(real_vals_for_scale, VMAX_PCT_REAL, VMAX_HARD_CAP_REAL)
    vmax_virtual = robust_vmax(virtual_vals_for_scale, VMAX_PCT_VIRTUAL, VMAX_HARD_CAP_VIRTUAL)

    norm_real = Normalize(vmin=float(VMIN), vmax=float(vmax_real))
    norm_virtual = Normalize(vmin=float(VMIN), vmax=float(vmax_virtual))

    print(f"[SCALE] Real: vmin={VMIN}, vmax={vmax_real:.4f}")
    print(f"[SCALE] Virtual: vmin={VMIN}, vmax={vmax_virtual:.4f}")

    # --------------------
    # Figure layout
    # --------------------
    n_virtual = len(virtual_panels)
    n_rows_virtual = int(math.ceil(n_virtual / float(NCOLS))) if n_virtual > 0 else 0
    n_rows_total = 1 + n_rows_virtual

    fig_w = 3.0 * NCOLS
    fig_h = 3.0 * n_rows_total
    fig, axes = plt.subplots(n_rows_total, NCOLS, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(n_rows_total, NCOLS)

    # leave right margin for colorbars + a bit more top space for suptitle
    fig.subplots_adjust(left=0.02, right=RIGHT_MARGIN, top=0.96, bottom=0.02, wspace=0.15, hspace=0.25)

    # Row 1: Real
    for c in range(NCOLS):
        ax = axes[0, c]
        if c < len(real_panels):
            p = real_panels[c]
            plot_panel(ax, p["coords"], p["vals"], p["title"], norm_real, s=S_REAL, invert_y=INVERT_Y)
            # plot_panel(ax, p["coords"], p["vals"], p["title"], norm_real, s=S_REAL, invert_y=INVERT_Y, xlim=(-50, 1050),
            #            ylim=(-50, 1050))
        else:
            ax.axis("off")

    # Rows 2+: Virtual
    idx = 0
    for r in range(1, n_rows_total):
        for c in range(NCOLS):
            ax = axes[r, c]
            if idx < n_virtual:
                p = virtual_panels[idx]
                plot_panel(ax, p["coords"], p["vals"], p["title"], norm_virtual, s=S_VIRTUAL, invert_y=INVERT_Y)
                # plot_panel(ax, p["coords"], p["vals"], p["title"], norm_virtual, s=S_VIRTUAL, invert_y=INVERT_Y,
                #            xlim=(-50, 1050), ylim=(-50, 1050))
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

    fig.suptitle(f"{GENE}: Real (row 1) vs Virtual (rows 2+), separate legends", y=0.995, fontsize=12)

    plt.savefig(OUT_PNG, dpi=220)
    plt.close(fig)

    print("Saved:", OUT_PNG)





def make_rgba_with_value_alpha(vals, norm, cmap_name, alpha_min, alpha_max, gamma=1.0):
    cmap = get_cmap(cmap_name)
    frac = norm(vals)
    frac = np.clip(np.asarray(frac, dtype=np.float32), 0.0, 1.0)

    rgba = cmap(frac)
    alpha = alpha_min + (frac ** gamma) * (alpha_max - alpha_min)
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

    rgba = make_rgba_with_value_alpha(
        vals=vals,
        norm=norm,
        cmap_name=cmap_name,
        alpha_min=ALPHA_MIN,
        alpha_max=alpha_max,
        gamma=ALPHA_GAMMA,
    )

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

def plot_3d(GEN_PATTERN,GENE,DEPTH_SCALE,X_LIM,Y_LIM,S_VIRTUAL,ALPHA_MAX_VIRTUAL,ELEV,AZIM,ALPHA_MIN,ALPHA_GAMMA,INVERT_Y,INVERT_X=False):
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
            "z": float(z),
        })

        virtual_vals_for_scale.append(vals)

    # print(f"[INFO] real={len(real_panels)} virtual={len(virtual_panels)}")

    # --------------------
    # Norms
    # --------------------
    # vmax_real = robust_vmax(real_vals_for_scale, VMAX_PCT_REAL, VMAX_HARD_CAP_REAL)
    vmax_virtual = robust_vmax(virtual_vals_for_scale, VMAX_PCT_VIRTUAL, VMAX_HARD_CAP_VIRTUAL)

    # norm_real = Normalize(vmin=float(VMIN), vmax=float(vmax_real))
    norm_virtual = Normalize(vmin=float(VMIN), vmax=float(vmax_virtual))

    # print(f"[SCALE] Real:    vmin={VMIN}, vmax={vmax_real:.4f}")
    print(f"[SCALE] Virtual: vmin={VMIN}, vmax={vmax_virtual:.4f}")

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

    # ax.set_xlim(xmin_depth, xmax_depth)
    ax.set_ylim(X_LIM)
    ax.set_zlim(Y_LIM)

    # visual aspect, not data aspect
    # ax.set_box_aspect((8, 1.3, 1.3))
    ax.set_box_aspect((1.3, 1.3, 1.3))


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
            cmap_name="viridis",
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