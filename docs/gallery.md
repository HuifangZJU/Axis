# Gallery

Outputs of the quickstart (`examples/quickstart_scc.py`) on SCC patient 10,
run on a 4-core CPU without a GPU:

```bash
python examples/quickstart_scc.py --data-dir scc_patient10 --canvas-size 256 --max-steps 400
```

The network was trained from scratch for 400 steps on a 256-px canvas
(55 min); the paper's models use 512-px canvases and 1000 epochs. Rerunning
from the saved model with `--checkpoint quickstart_out/model/model_final.pth --epochs 0`
reproduces the same numbers in 81 s.

## Molecular pseudo-image

The first 10 principal components of section 0, rasterised at 1/8 of the
canvas resolution (Methods, Eq. 1).

![Pseudo-image channels of section 0](assets/gallery/scc_pseudoimage_section0.jpg)

## Transition field: 20 µm → 0 µm

Fixed and moving H&E, the moving section warped by the learned field, and
green/magenta overlays before and after.

![Registration of section 2 onto section 0](assets/gallery/scc_registration_2_to_0.jpg)

## Spatially matched sections

All three sections before and after transport into the frame of section 0
(colour: cluster, marker: section).

![Sections before and after matching](assets/gallery/scc_matched.jpg)

## Virtual tissue at 10 µm

The 10-µm section generated from the 0- and 20-µm anchors (α = 0.5) on a
Visium-like grid with the settings of the manuscript's Fig. 3, next to the
measured 10-µm section that was held out.

![Virtual 10-µm section](assets/gallery/scc_virtual_section.jpg)

## Numbers (`report.json`)

Neighbourhood agreement is the fraction of spots whose cluster matches the
majority cluster of section-0 spots within 1.5 spot distances. "Published"
is the AXIS alignment stored with the sample (`obsm["spatial_axis"]`). The
last column runs the same script with the SCC patient-10 model used in the
manuscript (512 px, `--checkpoint ... --epochs 0`, 69 s on the same CPU).

| | before | this run | manuscript model | published |
|---|---|---|---|---|
| 10 µm → 0 µm, neighbourhood agreement | 0.624 | 0.676 | 0.706 | 0.715 |
| 20 µm → 0 µm, neighbourhood agreement | 0.551 | 0.662 | 0.707 | 0.714 |

This run's transported spots lie 3.5 px (10 µm) and 4.6 px (20 µm) from the
published ones on the 256-px canvas, where neighbouring spots are 6.75 px
apart. With the manuscript model, the median distance is 0 px: 67–74% of the
spots land on exactly the published pixel and 90–95% within 1 px. The rest
differ because the published positions are whole pixels chosen by a
nearest-pixel inverse of the field, so sub-pixel differences between the
original GPU computation and this CPU run move some spots to a neighbouring
pixel.

Generated 10-µm section against the held-out measurement, with the cluster
metrics of Fig. 3c (means over clusters; lower is better):

| | this run | manuscript model |
|---|---|---|
| supported spots | 882 | 849 |
| cluster composition (symmetric relative error) | 0.166 | 0.198 |
| within-cluster nearest-neighbour distance (Wasserstein) | 0.307 | 0.271 |
| cluster connectivity (Jensen–Shannon distance) | 0.168 | 0.158 |

With the manuscript model, these values match the Fig. 3 evaluation script
(`reproducibility/figure3/3bc_cluster_evaluation.py`) exactly.
