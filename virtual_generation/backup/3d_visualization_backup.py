import os
import glob
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D

# --------------------
# Config
# --------------------
GROUP_ID = 0
ORIG_SAMPLES = [0, 1, 2]
# ORIG_SAMPLES = [3, 4, 5]

ORIG_Z_UM = {0: 2.5, 1: 5.7, 2: 17.9}
# ORIG_Z_UM = {3: 2.5, 4: 5.7, 5: 13.4}

GENE = "Arc"

ROOT = "/media/huifang/data/registration/xenium/mouse_brain"
NET_IN = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ads/Tg")
# H5AD_DIR = os.path.join(ROOT, "h5ads/widetype")

GEN_DIR = os.path.join(ROOT, "virtual_tissue", "Tg_time_grid256/analysis")
GEN_PATTERN = os.path.join(GEN_DIR, "virtual_Tg_time_*.h5ad")
OUT_PNG = os.path.join(GEN_DIR, f"{GENE}_real_virtual_3d_horizontal.png")

INVERT_Y = True
GEN_USE_VALID_MASK = True
USE_LOG1P = True

S_REAL = 1
S_VIRTUAL = 5

# transparency control
ALPHA_MIN = 0.2     # almost invisible near zero
ALPHA_MAX_REAL = 0.95
ALPHA_MAX_VIRTUAL = 0.85
ALPHA_GAMMA = 1.6    # >1 makes low values fade more aggressively

VMIN = 0.0
VMAX_PCT_REAL = 0.995
VMAX_PCT_VIRTUAL = 0.995
VMAX_HARD_CAP_REAL = None
VMAX_HARD_CAP_VIRTUAL = None

# tissue coordinate display range
X_LIM = (-50, 1050)
Y_LIM = (-50, 1050)

# Magnify slice spacing while preserving actual z ordering/ratios
DEPTH_SCALE = 260.0

# view
ELEV = 12
AZIM = -92

# optional speed-up
MAX_POINTS_PER_SLICE = None
RANDOM_SEED = 0

# --------------------
# Helpers
# --------------------
def load_coords_npz(group_id, sample_id):
    f = os.path.join(NET_IN, f"{group_id}_{sample_id}_validation.npz")
    d = np.load(f)
    return d["coord"].astype(np.float32)

def parse_z_from_filename(path):
    base = os.path.basename(path)
    s = base.split("virtual_Tg_time_")[1][:-5]
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

def robust_vmax(vals_list, pct, hard_cap):
    if len(vals_list) == 0:
        return 1.0
    allv = np.concatenate(vals_list, axis=0)
    vmax = float(np.quantile(allv, pct)) if allv.size > 0 else 1.0
    if hard_cap is not None:
        vmax = min(vmax, float(hard_cap))
    return max(vmax, 1e-6)

def maybe_downsample(coords, vals, max_points=None, seed=0):
    if max_points is None or len(vals) <= max_points:
        return coords, vals
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vals), size=max_points, replace=False)
    return coords[idx], vals[idx]

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
    vals = extract_gene_vals(ad, GENE)
    mask = get_valid_mask(ad)

    coords = coords[mask]
    vals = vals[mask]
    coords, vals = maybe_downsample(coords, vals, MAX_POINTS_PER_SLICE, RANDOM_SEED)

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

ax.set_xlim(xmin_depth, xmax_depth)
ax.set_ylim(X_LIM)
ax.set_zlim(Y_LIM)
# ax.axis('off')
# visual aspect, not data aspect
ax.set_box_aspect((4.2, 1.3, 1.3))

# camera
ax.view_init(elev=10, azim=-90)
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
    )

# axes
# all_z_um = [p["z"] for p in real_panels] + [p["z"] for p in virtual_panels]
# xmin_depth = min(all_z_um) * DEPTH_SCALE if all_z_um else 0.0
# xmax_depth = max(all_z_um) * DEPTH_SCALE if all_z_um else 1.0
# xpad = 0.08 * max(xmax_depth - xmin_depth, 1.0)
#
# ax.set_xlim(xmin_depth - xpad, xmax_depth + xpad)

if INVERT_Y:
    ax.set_ylim(X_LIM[::-1])
else:
    ax.set_ylim(X_LIM)

ax.set_zlim(Y_LIM)

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
ax.set_box_aspect((6.8, 1.2, 1.2))
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

# plt.savefig(OUT_PNG, dpi=240)
# plt.close(fig)
#
# print("Saved:", OUT_PNG)