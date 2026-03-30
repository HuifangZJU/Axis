import os
import glob
import math
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# --------------------
# Config
# --------------------
GROUP_ID = 1
ORIG_SAMPLES = [0, 1, 2, 3]
ORIG_Z_UM = {0: 0, 1: 10, 2: 310, 3: 320}
GENE = "PCP4"
# GENE = "MFGE8"
# GENE = "MOBP"
ROOT = "/media/huifang/data/registration/DLPFC/huifang"
H5AD_DIR = os.path.join(ROOT, "h5ad")
NET_IN = os.path.join(ROOT, "network_input")

GEN_DIR = os.path.join(ROOT, "generated_3d_fixedgrid_per_slice_nmf", "generated_grid96_step10um_ot_subset")
GEN_PATTERN = os.path.join(GEN_DIR, "gen_grid_z*um.h5ad")
OUT_PNG = os.path.join(GEN_DIR, f"{GENE}_real_virtual_montage_log1p_separate_legends_margin.png")

NCOLS = 6
INVERT_Y = True

S_REAL = 6
S_VIRTUAL = 8
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
def load_coords_npz(group_id, sample_id):
    f = os.path.join(NET_IN, f"{group_id}_{sample_id}_validation.npz")
    d = np.load(f)
    return d["coord"].astype(np.float32)

def parse_z_from_filename(path):
    base = os.path.basename(path)
    s = base.split("gen_grid_z")[1].split("um")[0]
    return int(s)

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

def plot_panel(ax, coords, vals, title, norm, s, invert_y):
    ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=s, alpha=ALPHA,
               norm=norm, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()

# --------------------
# Load Real
# --------------------
real_panels = []
real_vals_for_scale = []
for sid in ORIG_SAMPLES:
    ad = sc.read_h5ad(os.path.join(H5AD_DIR, f"{GROUP_ID}_{sid}.h5ad"))
    coords = load_coords_npz(GROUP_ID, sid)
    vals = extract_gene_vals(ad, GENE)

    n = min(coords.shape[0], vals.shape[0])
    coords = coords[:n]
    vals = vals[:n]

    z = ORIG_Z_UM.get(sid, None)
    title = f"Real ({z}um)" if z is not None else f"Real (sample {sid})"
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
    vals = extract_gene_vals(ad, GENE)
    mask = get_valid_mask(ad)

    coords = coords[mask]
    vals = vals[mask]

    virtual_panels.append({"title": f"Virtual ({z}um)", "coords": coords, "vals": vals})
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
