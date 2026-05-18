
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from visualization_utils import *
import seaborn as sns
# --------------------
# Config
# --------------------
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
import glob
import os
import scanpy as sc
import matplotlib.pyplot as plt

def draw_slice_plane_frame(
    ax,
    z_um,
    x_lim_2d,
    y_lim_2d,
    depth_scale=1.0,
    linestyle="-",
    color="0.55",
    linewidth=1.2,
    alpha=0.8,
    label=None,
    label_color=None,
    label_fontsize=10,
    label_y_frac=0.01,
    label_z_frac=0.08,
):
    """
    Draw a rectangular slice frame at constant depth in the 3D stack.

    Parameters
    ----------
    ax : mpl 3d axis
    z_um : float
        Slice depth before scaling.
    x_lim_2d : tuple
        2D x-coordinate limits for the slice plane (mapped to ax.set_ylabel("x")).
    y_lim_2d : tuple
        2D y-coordinate limits for the slice plane (mapped to ax.set_zlabel("y")).
    depth_scale : float
        Scale applied to z_um before plotting on the 3D depth axis.
    linestyle : str
        "-" for real, "--" for virtual.
    color : str
        Frame line color.
    linewidth : float
        Frame line width.
    alpha : float
        Frame transparency.
    label : str or None
        Optional text shown near the upper-left corner of the frame.
    """

    x_depth = float(z_um) * float(depth_scale)

    y0, y1 = float(min(x_lim_2d)), float(max(x_lim_2d))
    z0, z1 = float(min(y_lim_2d)), float(max(y_lim_2d))

    xs = np.full(5, x_depth, dtype=np.float32)
    ys = np.array([y0, y1, y1, y0, y0], dtype=np.float32)
    zs = np.array([z0, z0, z1, z1, z0], dtype=np.float32)

    ax.plot(
        xs,
        ys,
        zs,
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=0,
    )

    if label is not None:
        if label_color is None:
            label_color = color

        y_text = y0 + label_y_frac * (y1 - y0)
        z_text = z0 - label_z_frac * (z1 - z0)

        ax.text(
            x_depth,
            y_text,
            z_text,
            label,
            color=label_color,
            fontsize=label_fontsize,
            ha="left",
            va="top",
            zorder=10,
        )




def plot_virtual_3d(
    GENE,
    CMAP="viridis",
    REAL_FRAME_COLOR="black",
    VIRTUAL_FRAME_COLOR="0.15",
    REAL_FRAME_STYLE="-",
    VIRTUAL_FRAME_STYLE="--",
    REAL_FRAME_WIDTH=1,
    VIRTUAL_FRAME_WIDTH=1,
    SHOW_SLICE_FRAME=True,
    SHOW_SLICE_LABEL=True,
    VIRTUAL_UNIT_LABEL="um",   # change if your virtual axis is not in microns
):
    # --------------------
    # Load Real
    # --------------------
    real_panels = []
    real_vals_for_scale = []
    for sid in ORIG_SAMPLES:
        z = ORIG_Z_UM[sid]
        ad = sc.read_h5ad(os.path.join(H5AD_DIR, f"{GROUP_ID}_{sid}.h5ad"))
        coords = load_coords_npz(NET_IN, GROUP_ID, sid)

        try:
            vals = extract_gene_vals(ad, GENE)
        except ValueError:
            vals, _ = extract_obs_vals(ad, GENE)

        n = min(coords.shape[0], vals.shape[0])
        coords = coords[:n]
        vals = vals[:n]

        real_panels.append({
            "title": f"Real ({z}um)",
            "coords": coords,
            "vals": vals,
            "z": float(z),
        })
        real_vals_for_scale.append(vals)

    # --------------------
    # Load Virtual
    # --------------------
    GEN_PATTERN = os.path.join(GEN_DIR, "*.h5ad")
    gen_files = sorted(glob.glob(GEN_PATTERN), key=parse_z_from_filename)
    virtual_panels = []
    virtual_vals_for_scale = []

    for f in gen_files:
        z = parse_z_from_filename(f)
        z = int(z)
        ad = sc.read_h5ad(f)

        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)

        if GENE is None:
            vals = np.full(coords.shape[0], 1.0, dtype=np.float32)
        else:
            try:
                vals = extract_gene_vals(ad, GENE)
            except ValueError:
                vals, _ = extract_obs_vals(ad, GENE)

        mask = get_valid_mask(ad)
        coords = coords[mask]
        vals = vals[mask]

        virtual_panels.append({
            "title": f"Virtual ({z}{VIRTUAL_UNIT_LABEL})",
            "coords": coords,
            "vals": vals,
            "z": int(z),
        })
        virtual_vals_for_scale.append(vals)

    # --------------------
    # Norms
    # --------------------
    vmin_real, vmax_real = robust_vrange(
        real_vals_for_scale,
        1 - VMAX_PCT_REAL,
        VMAX_PCT_REAL,
    )
    vmin_virtual, vmax_virtual = robust_vrange(
        virtual_vals_for_scale,
        1 - VMAX_PCT_VIRTUAL,
        VMAX_PCT_VIRTUAL,
    )

    norm_real = Normalize(vmin=float(vmin_real), vmax=float(vmax_real))
    norm_virtual = Normalize(vmin=float(vmin_virtual), vmax=float(vmax_virtual))

    print(f"[SCALE] Real:    vmin={vmin_real:.4f}, vmax={vmax_real:.4f}")
    print(f"[SCALE] Virtual: vmin={vmin_virtual:.4f}, vmax={vmax_virtual:.4f}")

    # --------------------
    # 3D Figure
    # --------------------
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    # visual aspect, not data aspect
    # ax.set_box_aspect((10, 1.3, 1.3))
    y_range = abs(X_LIM[1] - X_LIM[0])
    z_range = abs(Y_LIM[1] - Y_LIM[0])

    # optional exaggeration of depth for visual clarity
    ax.set_box_aspect((1.4 * y_range, y_range, z_range))
    ax.set_position([0.03, 0.08, 0.80, 0.84])

    ax.set_proj_type("ortho")

    # 2D coordinate limits on the slice planes
    ax.set_ylim(X_LIM)
    ax.set_zlim(Y_LIM)

    # 2D coordinate limits on the slice planes


    ax.set_ylim(X_LIM)
    ax.set_zlim(Y_LIM)

    # --------------------
    # Draw slice frames first
    # --------------------
    if SHOW_SLICE_FRAME:
        for p in real_panels:
            draw_slice_plane_frame(
                ax=ax,
                z_um=p["z"],
                x_lim_2d=X_LIM,
                y_lim_2d=Y_LIM,
                depth_scale=DEPTH_SCALE,
                linestyle=REAL_FRAME_STYLE,
                color=REAL_FRAME_COLOR,
                linewidth=REAL_FRAME_WIDTH,
                alpha=0.75,
                label=p["title"] if SHOW_SLICE_LABEL else None,
                label_color=REAL_FRAME_COLOR,
                label_fontsize=14,
            )

        for p in virtual_panels:

            if int(p["z"]/20)==p["z"]/20:
                continue
                SHOW_SLICE_LABEL = False
            else:
                SHOW_SLICE_LABEL= True
            draw_slice_plane_frame(
                ax=ax,
                z_um=p["z"],
                x_lim_2d=X_LIM,
                y_lim_2d=Y_LIM,
                depth_scale=DEPTH_SCALE,
                linestyle=VIRTUAL_FRAME_STYLE,
                color=VIRTUAL_FRAME_COLOR,
                linewidth=VIRTUAL_FRAME_WIDTH,
                alpha=0.70,
                label=p["title"] if SHOW_SLICE_LABEL else None,
                label_color=VIRTUAL_FRAME_COLOR,
                label_fontsize=14,
            )

    # --------------------
    # Scatter real slices
    # --------------------
    for p in real_panels:
        scatter_slice_horizontal_depth(
            ax=ax,
            coords=p["coords"],
            vals=p["vals"],
            z_um=p["z"],
            norm=norm_real,
            cmap_name=CMAP,
            s=S_REAL,
            alpha_max=ALPHA_MAX_REAL,
            label_prefix="Real",
            DEPTH_SCALE=DEPTH_SCALE,
            ALPHA_MIN=ALPHA_MIN_REAL,
            ALPHA_GAMMA=ALPHA_GAMMA,
        )

    # --------------------
    # Scatter virtual slices
    # --------------------
    for p in virtual_panels:
        if int(p["z"] / 20) == p["z"] / 20:
            continue
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
            ALPHA_MIN=ALPHA_MIN_VIRTUAL,
            ALPHA_GAMMA=ALPHA_GAMMA,
        )

    # --------------------
    # Axes direction
    # --------------------
    if INVERT_Y:
        ax.set_zlim(Y_LIM[::-1])
    else:
        ax.set_zlim(Y_LIM)

    if INVERT_X:
        ax.set_ylim(X_LIM[::-1])
    else:
        ax.set_ylim(X_LIM)

    # --------------------
    # Labels and title
    # --------------------
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=ELEV, azim=AZIM)
    fig.suptitle(
        f"{GENE}: 3D stack of real and virtual slices",
        y=0.97,
        fontsize=14,
    )

    # --------------------
    # Legend
    # --------------------
    legend_handles = [
        Line2D(
            [0], [0],
            color=REAL_FRAME_COLOR,
            linestyle=REAL_FRAME_STYLE,
            linewidth=REAL_FRAME_WIDTH,
            label="Real slices (solid frame)",
        ),
        Line2D(
            [0], [0],
            color=VIRTUAL_FRAME_COLOR,
            linestyle=VIRTUAL_FRAME_STYLE,
            linewidth=VIRTUAL_FRAME_WIDTH,
            label="Virtual slices (dashed frame)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
    )
    #
    # --------------------
    # Colorbars
    # --------------------
    sm_real = ScalarMappable(norm=norm_real, cmap=CMAP)
    sm_real.set_array([])

    sm_virtual = ScalarMappable(norm=norm_virtual, cmap=CMAP)
    sm_virtual.set_array([])

    fig.subplots_adjust(left=0.02, right=0.84, top=0.94, bottom=0.04)

    cax_real = fig.add_axes([0.87, 0.58, 0.02, 0.21])
    cbar_real = fig.colorbar(sm_real, cax=cax_real)
    cbar_real.set_label(
        f"{GENE} log1p(counts) " if USE_LOG1P else f"{GENE} ",
        rotation=90,
    )

    cax_virtual = fig.add_axes([0.87, 0.18, 0.02, 0.25])
    cbar_virtual = fig.colorbar(sm_virtual, cax=cax_virtual)
    cbar_virtual.set_label(
        f"{GENE} log1p(counts)" if USE_LOG1P else f"{GENE}",
        rotation=90,
    )

    plt.show()



def plot_real_3d(
    GENE,
    CMAP="viridis",
    REAL_FRAME_COLOR="0.15",
    REAL_FRAME_STYLE="-",
    REAL_FRAME_WIDTH=1,
    SHOW_SLICE_FRAME=True,
    SHOW_SLICE_LABEL=True,
):
    # --------------------
    # Load Real
    # --------------------
    real_panels = []
    real_vals_for_scale = []
    for sid in ORIG_SAMPLES:
        z = ORIG_Z_UM[sid]
        ad = sc.read_h5ad(os.path.join(H5AD_DIR, f"{GROUP_ID}_{sid}.h5ad"))
        coords = load_coords_npz(NET_IN, GROUP_ID, sid)

        try:
            vals = extract_gene_vals(ad, GENE)
        except ValueError:
            vals, _ = extract_obs_vals(ad, GENE)

        n = min(coords.shape[0], vals.shape[0])
        coords = coords[:n]
        vals = vals[:n]

        real_panels.append({
            "title": f"Real ({z}um)",
            "coords": coords,
            "vals": vals,
            # "z": float(z),
            "z":int(sid)
        })
        real_vals_for_scale.append(vals)

    # --------------------
    # Norms
    # --------------------
    vmin_real, vmax_real = robust_vrange(
        real_vals_for_scale,
        1 - VMAX_PCT_REAL,
        VMAX_PCT_REAL,
    )


    norm_real = Normalize(vmin=float(vmin_real), vmax=float(vmax_real))
    print(f"[SCALE] Real:    vmin={vmin_real:.4f}, vmax={vmax_real:.4f}")
    # --------------------
    # 3D Figure
    # --------------------


    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    # visual aspect, not data aspect
    # ax.set_box_aspect((10, 1.3, 1.3))
    y_range = abs(X_LIM[1] - X_LIM[0])
    z_range = abs(Y_LIM[1] - Y_LIM[0])

    # optional exaggeration of depth for visual clarity
    ax.set_box_aspect((1.4*y_range, y_range, z_range))
    ax.set_position([0.03, 0.08, 0.80, 0.84])

    ax.set_proj_type("ortho")

    # 2D coordinate limits on the slice planes
    ax.set_ylim(X_LIM)
    ax.set_zlim(Y_LIM)

    # --------------------
    # Draw slice frames first
    # --------------------
    if SHOW_SLICE_FRAME:
        for p in real_panels:
            draw_slice_plane_frame(
                ax=ax,
                z_um=p["z"],
                x_lim_2d=X_LIM,
                y_lim_2d=Y_LIM,
                depth_scale=DEPTH_SCALE,
                linestyle=REAL_FRAME_STYLE,
                color=REAL_FRAME_COLOR,
                linewidth=REAL_FRAME_WIDTH,
                alpha=0.75,
                label=p["title"] if SHOW_SLICE_LABEL else None,
                label_color="black",
                label_fontsize=16,
            )
    # --------------------
    # Scatter real slices
    # --------------------
    for p in real_panels:
        scatter_slice_horizontal_depth(
            ax=ax,
            coords=p["coords"],
            vals=p["vals"],
            z_um=p["z"],
            norm=norm_real,
            cmap_name=CMAP,
            s=S_REAL,
            alpha_max=ALPHA_MAX_REAL,
            label_prefix="Real",
            DEPTH_SCALE=DEPTH_SCALE,
            ALPHA_MIN=ALPHA_MIN_REAL,
            ALPHA_GAMMA=ALPHA_GAMMA,
        )

    # --------------------
    # Axes direction
    # --------------------
    if INVERT_Y:
        ax.set_zlim(Y_LIM[::-1])
    else:
        ax.set_zlim(Y_LIM)

    if INVERT_X:
        ax.set_ylim(X_LIM[::-1])
    else:
        ax.set_ylim(X_LIM)

    # --------------------
    # Labels and title
    # --------------------
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=ELEV, azim=AZIM)
    fig.suptitle(
        f"{GENE}: 3D stack of real and virtual slices",
        y=0.97,
        fontsize=14,
    )

    # --------------------
    # Legend
    # --------------------
    #
    # --------------------
    # Colorbars
    # --------------------
    sm_real = ScalarMappable(norm=norm_real, cmap=CMAP)
    sm_real.set_array([])


    fig.subplots_adjust(left=0.02, right=0.84, top=0.94, bottom=0.04)

    cax_real = fig.add_axes([0.87, 0.58, 0.02, 0.21])
    cbar_real = fig.colorbar(sm_real, cax=cax_real)
    cbar_real.set_label(
        f"{GENE} log1p(counts) " if USE_LOG1P else f"{GENE} ",
        rotation=90,
    )


    plt.show()


def plot_counterfactual_3d(
    GENE,
    CMAP="viridis",
    VIRTUAL_FRAME_COLOR="0.35",
    VIRTUAL_FRAME_STYLE="--",
    VIRTUAL_FRAME_WIDTH=1.4,
    SHOW_SLICE_FRAME=True,
    SHOW_SLICE_LABEL=True,
    VIRTUAL_UNIT_LABEL="um",   # change if your virtual axis is not in microns
):
    # --------------------
    # Load Virtual
    # --------------------
    GEN_PATTERN = os.path.join(GEN_DIR, "*.h5ad")
    gen_files = sorted(glob.glob(GEN_PATTERN), key=parse_z_from_filename)
    virtual_panels = []
    virtual_vals_for_scale = []


    for id,f in enumerate(gen_files):
        z = parse_z_from_filename(f)
        z = int(z)
        ad = sc.read_h5ad(f)
        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)

        if GENE is None:
            vals = np.full(coords.shape[0], 1.0, dtype=np.float32)
        else:
            try:
                vals = extract_gene_vals(ad, GENE)
            except ValueError:
                vals, _ = extract_obs_vals(ad, GENE)

        mask = get_valid_mask(ad)
        coords = coords[mask]
        vals = vals[mask]
        # virtual_panels.append({
        #     "title": f"Counterfactual \n ({z}{VIRTUAL_UNIT_LABEL})",
        #     "coords": coords,
        #     "vals": vals,
        #     "z": int(id),
        # })
        virtual_panels.append({
            "title": f"Aggregated gene reads",
            "coords": coords,
            "vals": vals,
            "z": int(1),
            # "z": int(id),
        })
        virtual_vals_for_scale.append(vals)

    # --------------------
    # Norms
    # --------------------
    vmin_virtual, vmax_virtual = robust_vrange(
        virtual_vals_for_scale,
        1 - VMAX_PCT_VIRTUAL,
        VMAX_PCT_VIRTUAL,
    )

    norm_virtual = Normalize(vmin=float(vmin_virtual), vmax=float(vmax_virtual))
    print(f"[SCALE] Virtual: vmin={vmin_virtual:.4f}, vmax={vmax_virtual:.4f}")

    # --------------------
    # 3D Figure
    # --------------------
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    # all_depths = [p["z"] * DEPTH_SCALE for p in virtual_panels]
    # x_min = min(all_depths)
    # x_max = max(all_depths)
    # x_pad = 0.02 * max(x_max - x_min, 1.0)
    # # visual aspect, not data aspect
    # # ax.set_box_aspect((10, 1.3, 1.3))
    # x_range = (x_max + x_pad) - (x_min - x_pad)
    y_range = abs(X_LIM[1] - X_LIM[0])
    z_range = abs(Y_LIM[1] - Y_LIM[0])

    # optional exaggeration of depth for visual clarity
    ax.set_box_aspect((1.5*y_range, y_range, z_range), zoom=1.2)
    # ax.set_box_aspect((1. * y_range, y_range, z_range), zoom=1.2)
    ax.set_position([0.03, 0.08, 0.80, 0.84])

    ax.set_proj_type("ortho")

    # 2D coordinate limits on the slice planes


    # ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(X_LIM)
    ax.set_zlim(Y_LIM)

    # --------------------
    # Draw slice frames first
    # --------------------
    if SHOW_SLICE_FRAME:
        for i,p in enumerate(virtual_panels):
            # if i!=0:
            #     SHOW_SLICE_LABEL=False
            draw_slice_plane_frame(
                ax=ax,
                z_um=p["z"],
                x_lim_2d=X_LIM,
                y_lim_2d=Y_LIM,
                depth_scale=DEPTH_SCALE,
                linestyle=VIRTUAL_FRAME_STYLE,
                color=VIRTUAL_FRAME_COLOR,
                linewidth=VIRTUAL_FRAME_WIDTH,
                alpha=0.70,
                label=p["title"] if SHOW_SLICE_LABEL else None,
                label_color=VIRTUAL_FRAME_COLOR,
                label_fontsize=14,
                label_y_frac=0.15,
                label_z_frac=0.1,
            )

    # --------------------
    # Scatter virtual slices
    # --------------------
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
            ALPHA_MIN=ALPHA_MIN_VIRTUAL,
            ALPHA_GAMMA=ALPHA_GAMMA,
        )

    # --------------------
    # Axes direction
    # --------------------
    if INVERT_Y:
        ax.set_zlim(Y_LIM[::-1])
    else:
        ax.set_zlim(Y_LIM)

    if INVERT_X:
        ax.set_ylim(X_LIM[::-1])
    else:
        ax.set_ylim(X_LIM)

    # --------------------
    # Labels and title
    # --------------------
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=ELEV, azim=AZIM)
    fig.suptitle(
        f"{GENE}: 3D stack of real and virtual slices",
        y=0.97,
        fontsize=14,
    )

    # --------------------
    # Colorbars


    sm_virtual = ScalarMappable(norm=norm_virtual, cmap=CMAP)
    sm_virtual.set_array([])

    fig.subplots_adjust(left=0.02, right=0.84, top=0.94, bottom=0.04)


    cax_virtual = fig.add_axes([0.87, 0.18, 0.02, 0.25])
    cbar_virtual = fig.colorbar(sm_virtual, cax=cax_virtual)
    cbar_virtual.set_label(
        f"{GENE} log1p(counts)" if USE_LOG1P else f"{GENE}",
        rotation=90,
    )

    plt.show()



GROUP_ID = 10
ORIG_Z_UM = {0: 0, 1: 10, 2: 20}


ROOT = "/media/huifang/data/registration/SCC/huifang"
NET_IN = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ad")


INVERT_X = True
INVERT_Y = True
GEN_USE_VALID_MASK = True
USE_LOG1P = True

VMIN = 0.0
VMAX_PCT_REAL = 0.995
VMAX_PCT_VIRTUAL = 0.995
VMAX_HARD_CAP_REAL = None
VMAX_HARD_CAP_VIRTUAL = None

# tissue coordinate display range
X_LIM = (-50, 550)
Y_LIM = (-50, 550)

# Magnify slice spacing while preserving actual z ordering/ratios
DEPTH_SCALE = 1


def get_3a_paras():
    ORIG_SAMPLES = [0, 2]
    dir = os.path.join(ROOT, "virtual_tissue", f"scc_z_group{GROUP_ID}_grid48/query")
    ALPHA_MIN_REAL = 0.2  # almost invisible near zero
    ALPHA_MIN_VIRTUAL=0.2
    ALPHA_MAX_REAL = 1
    ALPHA_MAX_VIRTUAL = 1
    ALPHA_GAMMA = 1.6  # >1 makes low values fade more aggressively
    ELEV = 15
    AZIM = 145
    S_REAL = 15
    S_VIRTUAL = 15
    return ORIG_SAMPLES,dir,ALPHA_MIN_REAL,ALPHA_MIN_VIRTUAL,ALPHA_MAX_REAL,ALPHA_MAX_VIRTUAL,ALPHA_GAMMA,ELEV,AZIM,S_REAL,S_VIRTUAL

def get_3b_paras():
    ORIG_SAMPLES = [ 0, 2]
    dir = os.path.join(ROOT, "virtual_tissue", f"scc_z_group{GROUP_ID}_grid40_spatial3_gene_weight/query")
    ALPHA_MIN_REAL = 0.2  # almost invisible near zero
    ALPHA_MIN_VIRTUAL = 0.2
    ALPHA_MAX_REAL = 1.0
    ALPHA_MAX_VIRTUAL = 1
    ALPHA_GAMMA = 1.6  # >1 makes low values fade more aggressively
    ELEV = 15
    AZIM = 145
    S_REAL = 15
    S_VIRTUAL = 15
    return ORIG_SAMPLES,dir,ALPHA_MIN_REAL,ALPHA_MIN_VIRTUAL,ALPHA_MAX_REAL,ALPHA_MAX_VIRTUAL,ALPHA_GAMMA,ELEV,AZIM,S_REAL,S_VIRTUAL



def get_3c_paras():
    ORIG_SAMPLES = [0,1,2]
    dir = os.path.join(ROOT, "virtual_tissue", f"scc_z_group{GROUP_ID}_grid40_spatial3_gene_weight/analysis")
    ALPHA_MIN_REAL = 0.0  # almost invisible near zero
    ALPHA_MIN_VIRTUAL=0.0
    ALPHA_MAX_REAL = 1.0
    ALPHA_MAX_VIRTUAL = 1.0
    ALPHA_GAMMA = 1.6  # >1 makes low values fade more aggressively
    ELEV = 15
    AZIM = 145
    S_REAL = 15
    S_VIRTUAL = 15
    return ORIG_SAMPLES,dir,ALPHA_MIN_REAL,ALPHA_MIN_VIRTUAL,ALPHA_MAX_REAL,ALPHA_MAX_VIRTUAL,ALPHA_GAMMA,ELEV,AZIM,S_REAL,S_VIRTUAL

ORIG_SAMPLES,GEN_DIR, ALPHA_MIN_REAL,ALPHA_MIN_VIRTUAL,ALPHA_MAX_REAL,ALPHA_MAX_VIRTUAL,ALPHA_GAMMA,ELEV,AZIM,S_REAL,S_VIRTUAL= get_3a_paras()
plot_real_3d(GENE = "KRT1", CMAP="Reds")
# plot_real_3d(GENE = "original_clusters", CMAP='tab20')
#
ORIG_SAMPLES,GEN_DIR, ALPHA_MIN_REAL,ALPHA_MIN_VIRTUAL,ALPHA_MAX_REAL,ALPHA_MAX_VIRTUAL,ALPHA_GAMMA,ELEV,AZIM,S_REAL,S_VIRTUAL = get_3b_paras()
plot_virtual_3d(GENE = "KRT1", CMAP="Reds")
plot_virtual_3d(GENE = "original_clusters", CMAP='tab20')


ORIG_SAMPLES,GEN_DIR, ALPHA_MIN_REAL,ALPHA_MIN_VIRTUAL,ALPHA_MAX_REAL,ALPHA_MAX_VIRTUAL,ALPHA_GAMMA,ELEV,AZIM,S_REAL,S_VIRTUAL= get_3c_paras()
plot_counterfactual_3d(GENE = "KRT1", CMAP="Reds")
# plot_counterfactual_3d(GENE = "PCP4", CMAP="Reds")