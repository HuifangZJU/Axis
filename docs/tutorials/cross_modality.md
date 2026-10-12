# Cross-modality analysis

Transcriptomic (Xenium) and proteomic (PhenoCycler) sections are registered
with a shared molecular representation built from a few cognate
protein–gene pairs (IDO1–*IDO1*, CD45RO–*PTPRC*, Podoplanin–*PDPN*) plus DAPI.
The full protein panel is then reconstructed at the Xenium cell coordinates:

```python
# xenium, protein: AnnData of the two sections, each placed on the canvas with
# axis.pp.prepare_canvas(a, canvas_size=1024, enhance_dapi=True)      # CLAHE-enhanced DAPI
markers = {"IDO1": "IDO1", "CD45RO": "PTPRC", "Podoplanin": "PDPN"}   # protein -> cognate gene
xen_m = xenium[:, list(markers.values())].copy()
pro_m = protein[:, list(markers)].copy()
pro_m.var_names = list(markers.values())       # pseudo-images use the features shared by name
axis.pp.build_pseudoimages([xen_m, pro_m], n_comps=3, preset="xenium")   # configs/phenocycler_luad.yaml
fields = axis.tl.fit_transition([xen_m, pro_m], fusion="linear", morph_win=41, mol_win=31, lr=5e-5)

# axis.tl.match(..., reconstruct=True) keeps only features shared by all samples,
# so reconstruct the complete protein panel at the Xenium cells directly:
protein_on_cells = axis.matching.reconstruct_on_reference(
    protein.obsm["spatial_canvas"], protein.X, fields[(0, 1)],
    xenium.obsm["spatial_canvas"], var_names=protein.var_names,
)
```

## Predicting an unmeasured modality

```python
valid = protein_on_cells.obs["valid"].to_numpy()   # locations supported by both modalities
gene_matrix = xenium.X[valid]                       # predictors: Xenium genes at the matched cells
protein_matrix = protein_on_cells.X[valid]           # targets: reconstructed protein panel
protein_names = list(protein_on_cells.var_names)

res = axis.tl.predict_modality(gene_matrix, protein_matrix, protein_names)
res.metrics          # one row per protein ("target"): Pearson / Spearman / R² / MAE / RMSE on the 20 % held-out cells
res.predict(new_gene_matrix)   # new Xenium cells, same genes in the same order
```

One XGBoost regressor is trained per protein with the two-stage validation
search of Methods (≈1,200–3,000 rounds at learning rates 0.07–0.03), then refit
on training + validation cells. For Fig. 7f the Xenium predictors were first
smoothed with PreTSA (`reproducibility/figure7/analysis/pretsa_spatial.py`);
`predict_modality` does not smooth its inputs.
