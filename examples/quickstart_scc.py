"""AXIS quickstart on three serial SCC Visium sections (patient 10; 0, 10 and 20 µm).

    python examples/quickstart_scc.py --data-dir path/to/scc_patient10 --out quickstart_out

Steps
-----
1. Place every section on a square canvas and build molecular pseudo-images.
2. Learn transition fields between all section pairs.
3. Spatially match sections 1 and 2 to section 0 (and compare with the
   published AXIS alignment stored in ``obsm["spatial_axis"]``).
4. Generate the 10-µm section from the 0- and 20-µm anchors with the settings
   of the manuscript's Fig. 3 and compare it with the measured (held-out)
   10-µm section using the Fig. 3c cluster metrics.

On a GPU use the default 512-px canvas. On a CPU use
``--canvas-size 256 --max-steps 400`` (about 55 min on 4 cores; training at
512 px needs more than 14 GB of RAM). With a trained model,
``--checkpoint model_final.pth --epochs 0`` skips training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import axis  # noqa: E402


def neighbourhood_agreement(ref_xy, ref_labels, xy, labels, radius):
    """Fraction of spots whose cluster is the majority cluster of reference spots within ``radius``."""
    tree = cKDTree(ref_xy)
    ref_labels = np.asarray(ref_labels).astype(str)
    hits, n = 0, 0
    for p, lab in zip(xy, np.asarray(labels).astype(str)):
        nb = tree.query_ball_point(p, radius)
        if not nb:
            continue
        values, counts = np.unique(ref_labels[nb], return_counts=True)
        hits += lab in set(values[counts == counts.max()])
        n += 1
    return hits / max(n, 1)


def cluster_fidelity(measured, generated, key="original_clusters"):
    """Fig. 3c metrics, averaged over clusters (as in ``reproducibility/figure3/3bc_cluster_evaluation.py``).

    Cluster composition (symmetric relative absolute error), within-cluster
    nearest-neighbour distances in spot units (1-D Wasserstein distance) and
    6-nearest-neighbour cluster connectivity (Jensen-Shannon distance); lower
    is better. Each tissue uses its own ``obsm["spatial"]`` and spot spacing.
    """
    m = axis.metrics
    xy = [np.asarray(a.obsm["spatial"], dtype=float)[:, :2] for a in (measured, generated)]
    lab = [a.obs[key].astype(str).to_numpy() for a in (measured, generated)]
    order = sorted(set(lab[0]) | set(lab[1]), key=lambda c: (0, float(c)) if c.replace(".", "", 1).isdigit() else (1, c))
    srae = m.symmetric_relative_abs_error(m.cluster_composition(lab[0], order), m.cluster_composition(lab[1], order))
    unit = [m.min_positive_spot_distance(c) for c in xy]
    wasserstein = [
        m.local_structure_distance(*(m.within_cluster_nn_distances(xy[i], lab[i], c, unit[i]) for i in (0, 1))) for c in order
    ]
    conn = [m.neighborhood_connectivity(xy[i], lab[i], order, k=6) for i in (0, 1)]
    js = [m.js_distance(conn[0][i], conn[1][i]) for i in range(len(order))]
    return {
        "cluster_composition_srae": float(np.nanmean(srae)),
        "within_cluster_nn_wasserstein": float(np.nanmean(wasserstein)),
        "connectivity_js_distance": float(np.nanmean(js)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Folder with 10_0.h5ad, 10_1.h5ad, 10_2.h5ad")
    parser.add_argument("--out", default="quickstart_out")
    parser.add_argument("--canvas-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--checkpoint", default=None, help="Optional pretrained checkpoint")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scale = args.canvas_size / 512.0

    # 1 -- canvases and pseudo-images -------------------------------------------------
    adatas = axis.datasets.load_scc_sample(args.data_dir, patient=10)
    for a in adatas:
        axis.pp.prepare_canvas(a, spatial_key="spatial_image_coor", canvas_size=args.canvas_size)
    axis.pp.build_pseudoimages(adatas, n_comps=10, preset="visium")
    axis.pl.pseudoimage(adatas[0]).savefig(out / "pseudoimage_section0.png", dpi=150)

    # 2 -- transition fields (SCC settings: morphology window 9, molecular window 5, lambda_mol 0.5)
    fields = axis.tl.fit_transition(
        adatas,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        max_steps=args.max_steps,
        fusion="transformer",
        morph_win=9,
        mol_win=5,
        lambda_mol=0.5,
        device=args.device,
        out_dir=out / "model",
    )
    fields.save(out / "fields")
    if fields.history:
        axis.pl.loss(fields.history).savefig(out / "training_loss.png", dpi=150)
    t_train = time.time() - t0

    # 3 -- spatial matching to section 0 --------------------------------------------
    axis.tl.match(adatas, fields, reference=0, key_added="spatial_axis_new")
    radius = 1.5 * 13.5 * scale  # 1.5 spot distances
    ref_xy, ref_lab = adatas[0].obsm["spatial_canvas"], adatas[0].obs["original_clusters"]
    report = {
        "canvas_size": args.canvas_size,
        "training_steps": len(next(iter(fields.history.values()), [])),
        "train_seconds": round(t_train, 1),
    }
    for k in (1, 2):
        a = adatas[k]
        lab = a.obs["original_clusters"]
        entry = {
            "neighbourhood_agreement_before": neighbourhood_agreement(ref_xy, ref_lab, a.obsm["spatial_canvas"], lab, radius),
            "neighbourhood_agreement_after": neighbourhood_agreement(ref_xy, ref_lab, a.obsm["spatial_axis_new"], lab, radius),
        }
        if "spatial_axis" in a.obsm:  # published AXIS result (512-px canvas)
            published = np.asarray(a.obsm["spatial_axis"]) * scale
            entry["neighbourhood_agreement_published"] = neighbourhood_agreement(ref_xy, ref_lab, published, lab, radius)
            disp_new = a.obsm["spatial_axis_new"] - a.obsm["spatial_canvas"]
            disp_pub = published - a.obsm["spatial_canvas"]
            entry["displacement_median_px_new"] = float(np.median(np.linalg.norm(disp_new, axis=1)))
            entry["displacement_median_px_published"] = float(np.median(np.linalg.norm(disp_pub, axis=1)))
            entry["distance_to_published_median_px"] = float(
                np.median(np.linalg.norm(a.obsm["spatial_axis_new"] - published, axis=1))
            )
        report[f"section{k}_to_0"] = entry
    axis.pl.matched(adatas, color="original_clusters", key="spatial_axis_new").savefig(out / "matched.png", dpi=150)
    axis.pl.registration(adatas[0].uns["axis"]["image"], adatas[2].uns["axis"]["image"], fields[(0, 2)]).savefig(
        out / "registration_2_to_0.png", dpi=150
    )

    # 4 -- generate the held-out 10-µm section from the 0/20-µm anchors ------------------
    # Fig. 3 settings (reproducibility/pipelines/generation/scc_depth.py): clusters are
    # generated with the molecular bandwidth (no separate attr_spatial_bandwidth).
    vt = axis.tl.generate(
        adatas,
        fields,
        positions=[0, 10, 20],
        query=10,
        anchors=[0, 2],
        grid="visium",
        grid_size=40,
        attr_keys=["original_clusters"],
        spatial_bandwidth=3 * scale,
        max_dist=10 * scale,
    )
    vt.write_h5ad(out / "virtual_section_10um.h5ad")
    supported = vt[vt.obs["valid"].astype(bool)]
    report["generation_10um"] = {"n_supported_spots": int(supported.n_obs), **cluster_fidelity(adatas[1], supported)}
    import matplotlib.pyplot as plt

    cats = sorted({c for a in adatas for c in a.obs["original_clusters"].astype(str)}, key=int)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, (a, key, title) in zip(
        axes,
        [
            (adatas[0], "spatial_canvas", "measured 0 µm (anchor)"),
            (vt, "spatial", "AXIS virtual 10 µm"),
            (adatas[1], "spatial_canvas", "measured 10 µm (held out)"),
            (adatas[2], "spatial_canvas", "measured 20 µm (anchor)"),
        ],
    ):
        axis.pl.spatial(a, "original_clusters", spatial_key=key, categories=cats, ax=ax, title=title)
    fig.tight_layout()
    fig.savefig(out / "virtual_section.png", dpi=150)

    report["total_seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
