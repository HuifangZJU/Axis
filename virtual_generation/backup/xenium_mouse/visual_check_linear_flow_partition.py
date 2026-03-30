import os
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from axis.coordinate_utils import find_inverse_warp_coords

# -----------------------------
# Config
# -----------------------------
GROUP_ID = 0
S0, S2 = 0, 2
GENE = "Arc"

ROOT = "/media/huifang/data/registration/xenium/mouse_brain"
NET_IN  = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ads/Tg")
AXIS_OUT = os.path.join(ROOT, "pairwise_output", f"group_{GROUP_ID}")  # adjust if needed


# z positions of slice0 and slice3 (um) in this group
t_0 = 2.5
t_2 = 17.9

# which z to visualize (pick a few first for speed)
t_GRID = [5, 7.5, 10, 12.5, 15]  # edit as you like

# plot style
INVERT_Y = False
S = 10
ALPHA_FLOOR = 0.02
ALPHA_POW = 0.7
VMAX_Q = 0.995

# speed: subsample points (set None to disable)
MAX_PTS_PER_SET = 6000

# -----------------------------
# IO
# -----------------------------
def load_coords_npz(group_id, sample_id):
    path = os.path.join(NET_IN, f"{group_id}_{sample_id}_validation.npz")
    d = np.load(path)
    return d["coord"].astype(np.float32)

def load_axis_flow(fixed_id, moving_id):
    # file names: [fixed]_[moving]_flow.npy
    path = os.path.join(AXIS_OUT, f"{fixed_id}_{moving_id}_flow.npy")
    if not os.path.exists(path):
        alt = os.path.join(ROOT, "output", f"group_[{GROUP_ID}]", f"{fixed_id}_{moving_id}_flow.npy")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)

def load_single_gene(t, gene):
    path = os.path.join(H5AD_DIR, f"xenium_Tg_{t}m.h5ad")
    ad = sc.read_h5ad(path)

    if gene not in ad.var_names:
        raise ValueError(f"Gene {gene} not found in {path}")
    j = int(np.where(ad.var_names == gene)[0][0])
    v = ad.X[:, j]
    if sp.issparse(v):
        v = np.asarray(v.toarray()).ravel().astype(np.float32)
    else:
        v = np.asarray(v).ravel().astype(np.float32)
    return v

def maybe_subsample(X, g, max_n, seed=0):
    if (max_n is None) or (X.shape[0] <= max_n):
        return X, g
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_n, replace=False)
    return X[idx], g[idx]

# -----------------------------
# Visualization
# -----------------------------
def alpha_from_gene(g, vmax):
    a = np.clip(g / vmax, 0.0, 1.0).astype(np.float32)
    a = np.power(a, ALPHA_POW)
    a = np.maximum(a, ALPHA_FLOOR)
    return a

def plot_two_sets(X0_t, g0, X3_t, g3, z, t):
    # robust vmax shared for alpha scaling (so contrast is comparable)
    allg = np.concatenate([g0, g3], axis=0)
    vmax = float(np.quantile(allg, VMAX_Q)) if allg.size > 0 else 1.0
    vmax = max(vmax, 1e-6)

    a0 = alpha_from_gene(g0, vmax)
    a3 = alpha_from_gene(g3, vmax)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    # color fixed (red/blue), alpha shows PCP4 intensity
    ax.scatter(X0_t[:, 0], X0_t[:, 1], s=S, c="red",  alpha=a0, linewidths=0, rasterized=True, label=f"slice0→t ({GENE})")
    ax.scatter(X3_t[:, 0], X3_t[:, 1], s=S, c="blue", alpha=a3, linewidths=0, rasterized=True, label=f"slice2→t ({GENE})")

    ax.set_title(f"PCP4 point overlay  z={z}um  t={t:.3f}\nred=0→t, blue=3→t  (alpha∝PCP4, vmax≈{vmax:.3g})", fontsize=10)
    # ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", frameon=True, fontsize=8)


    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    # Load coords
    X0 = load_coords_npz(GROUP_ID, S0)  # (N0,2)
    X3 = load_coords_npz(GROUP_ID, S2)  # (N3,2)

    # Load flows (dense, HxWx2)
    # convention: fixed_moving means moving->fixed
    flow_3_0 = load_axis_flow(fixed_id=2, moving_id=0)  # 0 -> 3
    flow_0_3 = load_axis_flow(fixed_id=0, moving_id=2)  # 3 -> 0

    # Load PCP4 counts only
    g0 = load_single_gene(t_0, GENE)
    g2 = load_single_gene(t_2, GENE)

    # Align lengths defensively (in case of tiny mismatch)
    n0 = min(X0.shape[0], g0.shape[0])
    n3 = min(X3.shape[0], g2.shape[0])
    X0, g0 = X0[:n0], g0[:n0]
    X3, g3 = X3[:n3], g2[:n3]

    denom = float(t_2 - t_0) if (t_2 != t_0) else 1.0

    for t in t_GRID:
        t_norm = float((t - t_0) / denom)
        t_norm = float(np.clip(t_norm, 0.0, 1.0))

        # Linear partial warp (points only)
        X0_t = find_inverse_warp_coords(X0, (t_norm * flow_3_0).astype(np.float32)).astype(np.float32)
        X3_t = find_inverse_warp_coords(X3, ((1.0 - t_norm) * flow_0_3).astype(np.float32)).astype(np.float32)
        # console debug: mean shifts (should scale with t and 1-t roughly)
        mean_shift0 = float(np.linalg.norm(X0_t - X0, axis=1).mean())
        mean_shift3 = float(np.linalg.norm(X3_t - X3, axis=1).mean())
        print(f"[DBG] z={z} t={t:.3f} mean_shift0={mean_shift0:.3f} mean_shift3={mean_shift3:.3f} ")


        # Optional subsample for speed (keeps layer pattern visible)
        X0_s, g0_s = maybe_subsample(X0_t, g0, MAX_PTS_PER_SET, seed=int(t) + 0)
        X3_s, g3_s = maybe_subsample(X3_t, g3, MAX_PTS_PER_SET, seed=int(t) + 1)

        plot_two_sets(X0_s, g0_s, X3_s, g3_s, z=int(t), t=t_norm)


if __name__ == "__main__":
    main()
