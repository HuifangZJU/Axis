"""AXIS quickstart on three serial SCC Visium sections (patient 10; 0, 10 and 20 µm).

    python examples/quickstart_scc.py --data-dir path/to/scc_patient10 --out quickstart_out

Steps
-----
1. Place every section on a square canvas and build molecular pseudo-images.
2. Learn transition fields between all section pairs.
3. Spatially match sections 1 and 2 to section 0 (and compare with the
   published AXIS alignment stored in ``obsm["spatial_axis"]``).
4. Generate the 10-µm section from the 0- and 20-µm anchors and compare it
   with the measured (held-out) 10-µm section.

On a GPU use the default 512-px canvas. On a laptop CPU use
``--canvas-size 256 --max-steps 300`` (about 25 min on 4 cores).
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
        attr_spatial_bandwidth=4 * scale,
        max_dist=10 * scale,
    )
    vt.write_h5ad(out / "virtual_section_10um.h5ad")
    measured = adatas[1]
    supported = vt[vt.obs["valid"].astype(bool)]
    clusters = sorted(set(measured.obs["original_clusters"].astype(str)))
    comp_measured = axis.metrics.cluster_composition(measured.obs["original_clusters"], clusters)
    comp_virtual = axis.metrics.cluster_composition(supported.obs["original_clusters"].astype(str), clusters)
    comp_anchor = axis.metrics.cluster_composition(adatas[0].obs["original_clusters"], clusters)
    report["generation_10um"] = {
        "n_supported_spots": int(supported.n_obs),
        "cluster_composition_SRAE_virtual": float(
            np.mean(axis.metrics.symmetric_relative_abs_error(comp_measured, comp_virtual))
        ),
        "cluster_composition_SRAE_copy_anchor0": float(
            np.mean(axis.metrics.symmetric_relative_abs_error(comp_measured, comp_anchor))
        ),
    }
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axis.pl.spatial(adatas[0], "original_clusters", spatial_key="spatial_canvas", ax=axes[0], title="measured 0 µm")
    axis.pl.spatial(vt, "original_clusters", ax=axes[1], title="AXIS virtual 10 µm")
    axis.pl.spatial(adatas[2], "original_clusters", spatial_key="spatial_canvas", ax=axes[2], title="measured 20 µm")
    fig.tight_layout()
    fig.savefig(out / "virtual_section.png", dpi=150)

    report["total_seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
