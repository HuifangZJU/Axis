import os
import glob
import math
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from axis.coordinate_utils import find_inverse_warp_coords


# ====================
# Config
# ====================
GROUP_ID = 0

# Real measured slices you want to show in the first row
# ORIG_SAMPLES = [3, 4, 5]
# ORIG_AXIS_VALUES = {3: 2.5, 4: 5.7, 5: 13.4}   # e.g. time/depth values for real slices
ORIG_SAMPLES = [1, 2, 3]
ORIG_AXIS_VALUES = {1: 2.5, 2: 5.7, 3: 17.9}   # e.g. time/depth values for real slices

GENE = "Arc"

ROOT = "/media/huifang/data/registration/xenium/mouse_brain"
NET_IN = os.path.join(ROOT, "network_input")
AXIS_OUT = os.path.join(ROOT, "pairwise_output", f"group_{GROUP_ID}")

H5AD_DIR_REAL = os.path.join(ROOT, "h5ads/Tg")
GEN_DIR = os.path.join(ROOT, "virtual_tissue", "Tg_time_grid256_gaussian")
GEN_PATTERN = os.path.join(GEN_DIR, "virtual_time_month_*.h5ad")

OUT_PNG = os.path.join(GEN_DIR, f"{GENE}_real_virtual_warped_to_reference.png")

# plotting
NCOLS = 6
INVERT_Y = False
S_REAL = 1.0
S_VIRTUAL = 1.0
POINT_ALPHA = 0.9

USE_LOG1P = True
VMIN = 0.0
VMAX_PCT_REAL = 0.995
VMAX_PCT_VIRTUAL = 0.995
VMAX_HARD_CAP_REAL = None
VMAX_HARD_CAP_VIRTUAL = None

RIGHT_MARGIN = 0.88
CBAR_X = 0.90
CBAR_W = 0.02

XLIM = (-50, 1050)
YLIM = (-50, 1050)

# choose reference virtual state
REFERENCE_MODE = "nearest_axis_value"   # "nearest_axis_value" or "file"
REFERENCE_AXIS_VALUE = 5.7
REFERENCE_FILE = None

# prefer query-specific support if present
PREFER_ACTIVE_MASK = True


# ====================
# Helpers
# ====================
def load_coords_npz(group_id, sample_id):
    f = os.path.join(NET_IN, f"{group_id}_{sample_id}_validation.npz")
    d = np.load(f)
    return d["coord"].astype(np.float32)


def load_axis_flow(axis_out_dir, fixed_id, moving_id):
    """
    Convention:
      {fixed_id}_{moving_id}_flow.npy means moving -> fixed
    """
    f = os.path.join(axis_out_dir, f"{fixed_id}_{moving_id}_flow.npy")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Flow not found: {f}")
    return np.asarray(np.load(f), dtype=np.float32)


def parse_axis_value_from_filename(path, prefix="virtual_Tg_time_"):
    """
    Fallback parser only.
    Example:
      virtual_WT_time_5.700.h5ad -> 5.7
    """
    base = os.path.basename(path)
    if not base.endswith(".h5ad"):
        raise ValueError(f"Unexpected file name: {base}")
    stem = base[:-5]
    if prefix not in stem:
        raise ValueError(f"Prefix '{prefix}' not found in file name: {base}")
    s = stem.split(prefix, 1)[1]
    return float(s)


def parse_virtual_meta(adata, fallback_axis_value=None):
    """
    Read virtual-slice metadata with backward compatibility.

    Preferred keys:
      uns["relative_position_alpha"]
      uns["source_anchor_ids"]
      uns["query_axis_value"]

    Backward-compatible keys:
      uns["t"]
      uns["source_anchors"]
    """
    uns = adata.uns

    if "relative_position_alpha" in uns:
        alpha = float(uns["relative_position_alpha"])
    elif "t" in uns:
        alpha = float(uns["t"])
    else:
        raise KeyError("Missing alpha in .uns['relative_position_alpha'] or .uns['t'].")

    if "source_anchor_ids" in uns:
        anchor_ids = tuple(int(x) for x in uns["source_anchor_ids"])
    elif "source_anchors" in uns:
        anchor_ids = tuple(int(x) for x in uns["source_anchors"])
    else:
        raise KeyError("Missing source anchors in .uns['source_anchor_ids'] or .uns['source_anchors'].")

    if len(anchor_ids) != 2:
        raise ValueError(f"Expected exactly 2 anchors, got {anchor_ids}")

    if "query_axis_value" in uns:
        axis_value = float(uns["query_axis_value"])
    else:
        axis_value = None if fallback_axis_value is None else float(fallback_axis_value)

    return {
        "alpha": alpha,
        "anchor_ids": anchor_ids,
        "axis_value": axis_value,
    }


def get_plot_mask(adata):
    if PREFER_ACTIVE_MASK and ("active_query_support" in adata.obs.columns):
        return np.asarray(adata.obs["active_query_support"].values, dtype=bool)
    if "valid" in adata.obs.columns:
        return np.asarray(adata.obs["valid"].values, dtype=bool)
    return np.ones(adata.n_obs, dtype=bool)


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


def robust_vmax(vals_list, pct, hard_cap=None):
    if len(vals_list) == 0:
        return 1.0
    allv = np.concatenate(vals_list, axis=0)
    vmax = float(np.quantile(allv, pct)) if allv.size > 0 else 1.0
    if hard_cap is not None:
        vmax = min(vmax, float(hard_cap))
    return max(vmax, 1e-6)


def plot_panel(ax, coords, vals, title, norm, s, invert_y, xlim=None, ylim=None):
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=vals,
        s=s,
        alpha=POINT_ALPHA,
        norm=norm,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if invert_y:
        ax.invert_yaxis()


def choose_reference_record(records, mode="nearest_axis_value", reference_axis_value=None, reference_file=None):
    if len(records) == 0:
        raise ValueError("No virtual records loaded.")

    if mode == "file":
        if reference_file is None:
            raise ValueError("REFERENCE_FILE must be set when REFERENCE_MODE='file'.")
        ref_abs = os.path.abspath(reference_file)
        for r in records:
            if os.path.abspath(r["file"]) == ref_abs:
                return r
        raise FileNotFoundError(f"Reference file not found among loaded records: {reference_file}")

    if mode == "nearest_axis_value":
        if reference_axis_value is None:
            raise ValueError("REFERENCE_AXIS_VALUE must be set when REFERENCE_MODE='nearest_axis_value'.")
        return min(
            records,
            key=lambda r: abs(float(r["meta"]["axis_value"]) - float(reference_axis_value))
        )

    raise ValueError(f"Unsupported REFERENCE_MODE: {mode}")


class VirtualFlowHandler:
    """
    Relative virtual-to-virtual coordinate warping for slices generated from
    the same ordered anchor pair.
    """

    def __init__(self, axis_out_dir):
        self.axis_out_dir = axis_out_dir
        self._flow_cache = {}

    def get_anchor_pair_flow(self, anchor_a, anchor_b):
        """
        Returns flow for anchor_a -> anchor_b
        using stored convention fixed=anchor_b, moving=anchor_a.
        """
        key = (int(anchor_a), int(anchor_b))
        if key not in self._flow_cache:
            self._flow_cache[key] = load_axis_flow(
                self.axis_out_dir,
                fixed_id=int(anchor_b),
                moving_id=int(anchor_a),
            ).astype(np.float32)
        return self._flow_cache[key]

    def build_virtual_to_virtual_flow(self, src_meta, ref_meta):
        """
        Relative flow from source virtual state to reference virtual state.

        If both are generated from the same ordered anchor pair (A,B):
          src alpha = a_s
          ref alpha = a_r
        then:
          flow src->ref = (a_r - a_s) * flow_A->B
        """
        src_anchors = tuple(src_meta["anchor_ids"])
        ref_anchors = tuple(ref_meta["anchor_ids"])

        if src_anchors != ref_anchors:
            raise ValueError(
                "This script currently requires all virtual slices to come from the same "
                f"ordered anchor pair. Got src={src_anchors}, ref={ref_anchors}"
            )

        anchor_a, anchor_b = src_anchors
        flow_ab = self.get_anchor_pair_flow(anchor_a, anchor_b)

        delta_alpha = float(ref_meta["alpha"] - src_meta["alpha"])
        rel_flow = (delta_alpha * flow_ab).astype(np.float32)
        return rel_flow

    def warp_coords_to_reference(self, coords, src_meta, ref_meta):
        rel_flow = self.build_virtual_to_virtual_flow(src_meta, ref_meta)
        coords_ref = find_inverse_warp_coords(
            np.asarray(coords, dtype=np.float32),
            rel_flow.astype(np.float32)
        ).astype(np.float32)
        return coords_ref


def make_real_anchor_meta(anchor_a, anchor_b, which_anchor, axis_value=None):
    """
    Represent a real anchor slice inside the same alpha parameterization:
      anchor_a -> alpha=0
      anchor_b -> alpha=1
    """
    if which_anchor == anchor_a:
        alpha = 0.0
    elif which_anchor == anchor_b:
        alpha = 1.0
    else:
        raise ValueError(
            f"Real sample {which_anchor} is not one of the source anchors {(anchor_a, anchor_b)}"
        )

    return {
        "alpha": float(alpha),
        "anchor_ids": (int(anchor_a), int(anchor_b)),
        "axis_value": None if axis_value is None else float(axis_value),
    }


# ====================
# Load virtual slices
# ====================
gen_files = sorted(glob.glob(GEN_PATTERN))
if len(gen_files) == 0:
    raise FileNotFoundError(f"No virtual files found with pattern: {GEN_PATTERN}")

virtual_records = []
for f in gen_files:
    ad = sc.read_h5ad(f)

    # fallback filename parser if query_axis_value is not saved
    try:
        fallback_axis_value = parse_axis_value_from_filename(f, prefix="virtual_Tg_time_")
    except Exception:
        fallback_axis_value = None

    meta = parse_virtual_meta(ad, fallback_axis_value=fallback_axis_value)

    coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)
    vals = extract_gene_vals(ad, GENE)
    mask = get_plot_mask(ad)

    coords = coords[mask]
    vals = vals[mask]

    virtual_records.append({
        "file": f,
        "adata": ad,
        "meta": meta,
        "coords": coords,
        "vals": vals,
    })

virtual_records = sorted(
    virtual_records,
    key=lambda r: float(r["meta"]["axis_value"]) if r["meta"]["axis_value"] is not None else 0.0
)

print(f"[INFO] loaded virtual slices: {len(virtual_records)}")

# choose reference virtual state
ref_record = choose_reference_record(
    virtual_records,
    mode=REFERENCE_MODE,
    reference_axis_value=REFERENCE_AXIS_VALUE,
    reference_file=REFERENCE_FILE,
)
ref_meta = ref_record["meta"]

print("[REF] selected reference:")
print("      file:", os.path.basename(ref_record["file"]))
print("      axis_value:", ref_meta["axis_value"])
print("      alpha:", ref_meta["alpha"])
print("      anchors:", ref_meta["anchor_ids"])

flow_handler = VirtualFlowHandler(AXIS_OUT)


# ====================
# Load real slices and warp them to the same reference state if possible
# ====================
anchor_a, anchor_b = ref_meta["anchor_ids"]

real_panels = []
real_vals_for_scale = []

for sid in ORIG_SAMPLES:
    axis_value = float(ORIG_AXIS_VALUES[sid])
    ad_real = sc.read_h5ad(os.path.join(H5AD_DIR_REAL, f"Tg_{axis_value}.h5ad"))
    coords_real = load_coords_npz(GROUP_ID, sid)
    vals_real = extract_gene_vals(ad_real, GENE)

    n = min(coords_real.shape[0], vals_real.shape[0])
    coords_real = coords_real[:n]
    vals_real = vals_real[:n]

    # If this real slice is one of the current anchor pair, warp it to reference.
    # Otherwise, keep original coordinates and label clearly.
    if sid in (anchor_a, anchor_b):
        real_meta = make_real_anchor_meta(
            anchor_a=anchor_a,
            anchor_b=anchor_b,
            which_anchor=sid,
            axis_value=axis_value,
        )
        coords_real_ref = flow_handler.warp_coords_to_reference(
            coords=coords_real,
            src_meta=real_meta,
            ref_meta=ref_meta,
        )
        title = f"Real {sid} ({axis_value}) → ref"
        coords_plot = coords_real_ref
    else:
        title = f"Real {sid} ({axis_value}) [unwarped]"
        coords_plot = coords_real

    real_panels.append({
        "title": title,
        "coords": coords_plot,
        "vals": vals_real,
    })
    real_vals_for_scale.append(vals_real)


# ====================
# Warp all virtual slices to the reference state
# ====================
virtual_panels = []
virtual_vals_for_scale = []

for r in virtual_records:
    src_meta = r["meta"]
    coords_src = r["coords"]
    vals = r["vals"]

    coords_ref = flow_handler.warp_coords_to_reference(
        coords=coords_src,
        src_meta=src_meta,
        ref_meta=ref_meta,
    )

    title = f"Virtual ({src_meta['axis_value']:.3f}) → ref"
    virtual_panels.append({
        "title": title,
        "coords": coords_ref,
        "vals": vals,
    })
    virtual_vals_for_scale.append(vals)

print(f"[INFO] real={len(real_panels)} virtual={len(virtual_panels)}")


# ====================
# Color scales
# ====================
vmax_real = robust_vmax(real_vals_for_scale, VMAX_PCT_REAL, VMAX_HARD_CAP_REAL)
vmax_virtual = robust_vmax(virtual_vals_for_scale, VMAX_PCT_VIRTUAL, VMAX_HARD_CAP_VIRTUAL)

norm_real = Normalize(vmin=float(VMIN), vmax=float(vmax_real))
norm_virtual = Normalize(vmin=float(VMIN), vmax=float(vmax_virtual))

print(f"[SCALE] Real:    vmin={VMIN}, vmax={vmax_real:.4f}")
print(f"[SCALE] Virtual: vmin={VMIN}, vmax={vmax_virtual:.4f}")


# ====================
# Figure layout
# ====================
n_virtual = len(virtual_panels)
n_rows_virtual = int(math.ceil(n_virtual / float(NCOLS))) if n_virtual > 0 else 0
n_rows_total = 1 + n_rows_virtual

fig_w = 3.0 * NCOLS
fig_h = 3.0 * n_rows_total
fig, axes = plt.subplots(n_rows_total, NCOLS, figsize=(fig_w, fig_h))
axes = np.array(axes).reshape(n_rows_total, NCOLS)

fig.subplots_adjust(left=0.02, right=RIGHT_MARGIN, top=0.96, bottom=0.02, wspace=0.15, hspace=0.25)

# Row 1: real
for c in range(NCOLS):
    ax = axes[0, c]
    if c < len(real_panels):
        p = real_panels[c]
        plot_panel(
            ax,
            p["coords"],
            p["vals"],
            p["title"],
            norm_real,
            s=S_REAL,
            invert_y=INVERT_Y,
            xlim=XLIM,
            ylim=YLIM,
        )
    else:
        ax.axis("off")

# Rows 2+: virtual warped to reference
idx = 0
for r in range(1, n_rows_total):
    for c in range(NCOLS):
        ax = axes[r, c]
        if idx < n_virtual:
            p = virtual_panels[idx]
            plot_panel(
                ax,
                p["coords"],
                p["vals"],
                p["title"],
                norm_virtual,
                s=S_VIRTUAL,
                invert_y=INVERT_Y,
                xlim=XLIM,
                ylim=YLIM,
            )
            idx += 1
        else:
            ax.axis("off")


# ====================
# Colorbars
# ====================
sm_real = ScalarMappable(norm=norm_real)
sm_real.set_array([])

sm_virtual = ScalarMappable(norm=norm_virtual)
sm_virtual.set_array([])

cax_real = fig.add_axes([CBAR_X, 0.62, CBAR_W, 0.28])
cbar_real = fig.colorbar(sm_real, cax=cax_real)
cbar_real.set_label(
    f"{GENE} log1p(counts) (Real)" if USE_LOG1P else f"{GENE} (Real)",
    rotation=90
)

if n_rows_total > 1:
    cax_virtual = fig.add_axes([CBAR_X, 0.12, CBAR_W, 0.42])
    cbar_virtual = fig.colorbar(sm_virtual, cax=cax_virtual)
    cbar_virtual.set_label(
        f"{GENE} log1p(counts) (Virtual)" if USE_LOG1P else f"{GENE} (Virtual)",
        rotation=90
    )

fig.suptitle(
    f"{GENE}: real + virtual slices warped to reference state "
    f"({ref_meta['axis_value']:.3f}, alpha={ref_meta['alpha']:.3f})",
    y=0.995,
    fontsize=12,
)

plt.savefig(OUT_PNG, dpi=220)
plt.close(fig)

print("Saved:", OUT_PNG)