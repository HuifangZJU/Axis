import os
from pathlib import Path

import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt


# -----------------------------
# 1. Load SCC data
# -----------------------------
def load_scc_data(root_dir):
    path_to_h5ads = os.path.join(root_dir, "H5ADs")
    patient_2, patient_5, patient_9, patient_10 = [], [], [], []
    patients = {
        "2": patient_2,
        "5": patient_5,
        "9": patient_9,
        "10": patient_10,
    }

    for k in patients.keys():
        p_id = int(k)
        for i in range(3):
            fn = os.path.join(path_to_h5ads, f"scc_p{p_id}_layer{i + 1}.h5ad")
            data = sc.read_h5ad(fn)
            patients[k].append(data)

    return patients


# -----------------------------
# 2. Joint counts
# -----------------------------
def build_joint_counts(slices):
    all_genes = sorted(set().union(*[list(ad.var_names) for ad in slices]))
    n_genes = len(all_genes)
    gene_index = {g: i for i, g in enumerate(all_genes)}

    slice_mats = []
    all_spot_ids = []

    for s_idx, ad in enumerate(slices):
        X = ad.X
        if sparse.issparse(X):
            X = X.tocsr()
        else:
            X = sparse.csr_matrix(X)

        var_idx = [gene_index[g] for g in ad.var_names]

        slice_full = sparse.lil_matrix((ad.n_obs, n_genes), dtype=X.dtype)
        slice_full[:, var_idx] = X
        slice_mats.append(slice_full.tocsr())

        base_obs_names = ad.obs_names.astype(str)
        spot_ids = [f"s{s_idx}_{obs}" for obs in base_obs_names]
        all_spot_ids.extend(spot_ids)

    joint_X = sparse.vstack(slice_mats)
    joint_X_dense = joint_X.toarray()

    return all_spot_ids, all_genes, joint_X_dense


# -----------------------------
# 3. Low→hi scale per slice
# -----------------------------
def estimate_scale_factor(adata, low_key="spatial_image_coor", high_key="spatial"):
    low = adata.obsm[low_key]
    high = adata.obsm[high_key]

    def _range(v):
        return float(v.max() - v.min())

    low_x_rng = _range(low[:, 0])
    low_y_rng = _range(low[:, 1])
    high_x_rng = _range(high[:, 0])
    high_y_rng = _range(high[:, 1])

    scale_x = high_x_rng / low_x_rng if low_x_rng > 0 else 1.0
    scale_y = high_y_rng / low_y_rng if low_y_rng > 0 else 1.0
    scale = 0.5 * (scale_x + scale_y)

    if abs(scale_x - scale_y) / max(scale, 1e-6) > 0.05:
        print(
            f"[WARN] Anisotropic scale: "
            f"scale_x={scale_x:.3f}, scale_y={scale_y:.3f}, using mean={scale:.3f}"
        )

    return scale


# -----------------------------
# 4. Build low-res coords in axis k
# -----------------------------
def build_locs_for_axis_lowres(
    root_dir,
    group_id,
    slices,
    all_spot_ids,
    target_axis_idx,
    coord_key="spatial_image_coor",
):
    reg_group_dir = os.path.join(root_dir, "output", f"group_{group_id}")

    all_coords = []

    for s_idx, ad in enumerate(slices):
        if s_idx == target_axis_idx:
            coords = ad.obsm[coord_key]
        else:
            reg_fn = os.path.join(
                reg_group_dir,
                f"{target_axis_idx}_{s_idx}_warped_coor.npy",
            )
            coords = np.load(reg_fn)
        all_coords.append(coords)

    joint_coords = np.vstack(all_coords)
    assert joint_coords.shape[0] == len(all_spot_ids)

    return joint_coords  # low-res coords only


# -----------------------------
# 5. Plot overlay on hi-res
# -----------------------------
def plot_overlay(image_path, coords_hires, title=None, point_size=2):
    if not os.path.exists(image_path):
        print(f"[WARN] Image not found: {image_path}")
        return

    img = plt.imread(image_path)

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.scatter(coords_hires[:, 0], coords_hires[:, 1],
                s=point_size, alpha=0.4)
    # plt.gca().invert_yaxis()  # enable if needed
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 6. Main: save hi-res locs
# -----------------------------
def export_group_istar_multi_axes(
    root_dir,
    group_id,
    slices,
    lowres_coord_key="spatial_image_coor",
    hires_coord_key="spatial",
    save=True,        # now we actually save hi-res locs
    visualize=True,
):
    out_dir = Path(root_dir) / "output" / f"group_{group_id}" / "istar_input"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) counts
    all_spot_ids, all_genes, joint_X_dense = build_joint_counts(slices)
    cnts_df = pd.DataFrame(joint_X_dense, index=all_spot_ids, columns=all_genes)
    cnts_df.insert(0, "spot_id", all_spot_ids)
    cnts_path = out_dir / "cnts.tsv"

    # 2) per-slice low→hi scale
    scale_factors = []
    for s_idx, ad in enumerate(slices):
        sf = estimate_scale_factor(
            ad,
            low_key=lowres_coord_key,
            high_key=hires_coord_key,
        )
        scale_factors.append(sf)
        print(f"[group {group_id}] slice {s_idx}: low→hi scale ≈ {sf:.4f}")

    # 3) locs-k.tsv in HI-RES for each axis k
    n_slices = len(slices)
    for target_axis_idx in range(n_slices):
        # low-res coords in axis k
        joint_coords_lowres = build_locs_for_axis_lowres(
            root_dir=root_dir,
            group_id=group_id,
            slices=slices,
            all_spot_ids=all_spot_ids,
            target_axis_idx=target_axis_idx,
            coord_key=lowres_coord_key,
        )

        # convert to hi-res using axis-k scale
        sf = scale_factors[target_axis_idx]
        joint_coords_hires = joint_coords_lowres * sf

        # build DataFrame with HI-RES coords
        locs_df = pd.DataFrame(
            {
                "spot_id": all_spot_ids,
                "x": joint_coords_hires[:, 0],
                "y": joint_coords_hires[:, 1],
            }
        )

        image_path = (
            Path(root_dir) / "network_input" /
            f"{group_id}_{target_axis_idx}_hires_image.jpg"
        )

        if visualize:
            plot_overlay(
                image_path=str(image_path),
                coords_hires=joint_coords_hires,
                title=f"group {group_id} – locs-{target_axis_idx} (hi-res)",
                point_size=10,
            )

        if save:
            locs_path = out_dir / f"locs-{target_axis_idx}.tsv"
            locs_df.to_csv(locs_path, sep="\t", index=False)
            print(f"[group {group_id}] wrote {locs_path}")

    if save:
        cnts_df.to_csv(cnts_path, sep="\t", index=False)
        print(f"[group {group_id}] wrote {cnts_path}")



# --------------------------------------
# Single-bundle export with FULL visualization
# --------------------------------------
def export_group_istar_single(
    root_dir,
    group_id,
    slices,
    lowres_coord_key="spatial_image_coor",
    hires_coord_key="spatial",
    save=True,
    visualize=True,
):
    """
    Single bundle: output/istar_input_single/group_{group_id}/

    For a group with S slices:

      - cnts-{s}.tsv: per-sample counts

      - subfolder {k}/: for each target axis k
          locs-s0.tsv, locs-s1.tsv, ... (each sample in axis k, hi-res)

    Visualization:
      For each target axis k, create a figure with S subplots:
        - subplot s shows coords of sample s in axis k (hi-res)
          overlaid on:
            * original hi-res if s == k
            * warped hi-res if s != k
    """
    base_out_dir = (
        Path(root_dir) / "output" / "istar_input_single" / f"group_{group_id}"
    )
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # (1) per-sample counts
    # --------------------------
    for s_idx, ad in enumerate(slices):
        cnts_path = base_out_dir / f"cnts-{s_idx}.tsv"
        spot_ids = ad.obs_names.astype(str)
        X = ad.X
        if sparse.issparse(X):
            X = X.toarray()
        cnts_df = pd.DataFrame(X, index=spot_ids, columns=ad.var_names)
        cnts_df.insert(0, "spot_id", spot_ids)

        if save:
            cnts_df.to_csv(cnts_path, sep="\t", index=False)
            print(f"[single] [group {group_id}] wrote {cnts_path}")

    # --------------------------
    # (2) per-slice scale factor
    # --------------------------
    scale_factors = []
    for s_idx, ad in enumerate(slices):
        sf = estimate_scale_factor(ad, low_key=lowres_coord_key, high_key=hires_coord_key)
        scale_factors.append(sf)
        print(f"[single] [group {group_id}] slice {s_idx}: low→hi scale ≈ {sf:.4f}")

    reg_group_dir = Path(root_dir) / "output" / f"group_{group_id}"
    n_slices = len(slices)

    # --------------------------
    # (3) locs for each target axis k
    # --------------------------
    for target_axis_idx in range(n_slices):
        axis_dir = base_out_dir / f"{target_axis_idx}"
        axis_dir.mkdir(parents=True, exist_ok=True)

        sf = scale_factors[target_axis_idx]

        if visualize:
            fig, axes = plt.subplots(
                1,
                n_slices,
                figsize=(4 * n_slices, 4),
                squeeze=False,
            )
            axes = axes[0]  # 1 row

        for s_idx, ad in enumerate(slices):
            spot_ids = ad.obs_names.astype(str)

            # ----- low-res coords in axis k -----
            if s_idx == target_axis_idx:
                coords_lowres = ad.obsm[lowres_coord_key]
            else:
                warp_path = reg_group_dir / f"{target_axis_idx}_{s_idx}_warped_coor.npy"
                coords_lowres = np.load(warp_path)

            # ----- high-res coords in axis k -----
            coords_hires = coords_lowres * sf

            # ----- save per-sample locs (HI-RES) -----
            locs_df = pd.DataFrame(
                {
                    "spot_id": spot_ids,
                    "x": coords_hires[:, 0],
                    "y": coords_hires[:, 1],
                }
            )
            locs_path = axis_dir / f"locs-s{s_idx}.tsv"

            if save:
                locs_df.to_csv(locs_path, sep="\t", index=False)
                print(f"[single] [group {group_id}] wrote {locs_path}")

            # ----- visualization: subplot per sample -----
            if visualize:
                ax = axes[s_idx]
                if s_idx == target_axis_idx:
                    # original hi-res image for target slice
                    img_path = (
                        Path(root_dir)
                        / "network_input"
                        / f"{group_id}_{target_axis_idx}_hires_image.jpg"
                    )
                else:
                    # warped hi-res image (target_axis_idx <- s_idx)
                    img_path = (
                        reg_group_dir
                        / f"{target_axis_idx}_{s_idx}_hires_warped_img.png"
                    )

                if os.path.exists(img_path):
                    img = plt.imread(img_path)
                    ax.imshow(img)
                    ax.scatter(
                        coords_hires[:, 0],
                        coords_hires[:, 1],
                        s=2,
                        alpha=0.4,
                    )
                    ax.set_title(f"s{s_idx}")
                    ax.axis("off")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"Missing image\n{img_path.name}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                    ax.axis("off")

        if visualize:
            fig.suptitle(f"group {group_id} – target axis {target_axis_idx}", y=0.98)
            plt.tight_layout()
            plt.show()
# -----------------------------
# 7. Run
# -----------------------------
if __name__ == "__main__":
    root_dir = "/media/huifang/data/registration/SCC/huifang/"

    patients = load_scc_data(root_dir)  # {"2": [ann0, ann1, ann2], ...}

    for group_id, slices in patients.items():
        # export_group_istar_multi_axes(
        #     root_dir=root_dir,
        #     group_id=group_id,
        #     slices=slices,
        #     lowres_coord_key="spatial_image_coor",
        #     hires_coord_key="spatial",
        #     save=True,       # hi-res locs saved
        #     visualize=False,  # keep True if you still want overlays
        # )
        # Single bundle (what you just described)
        for group_id, slices in patients.items():
            export_group_istar_single(
                root_dir=root_dir,
                group_id=group_id,
                slices=slices,
                lowres_coord_key="spatial_image_coor",
                hires_coord_key="spatial",
                save=True,
                visualize=False,  # set False later if you only want files
            )
