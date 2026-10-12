# Cross-modality analysis

Transcriptomic (Xenium) and proteomic (PhenoCycler) sections are registered
with a shared molecular representation built from a few cognate
protein–gene pairs (IDO1–*IDO1*, CD45RO–*PTPRC*, Podoplanin–*PDPN*) plus DAPI.
The full protein panel is then reconstructed at the Xenium cell coordinates:

```python
# adatas = [xenium, protein]; pseudo-images built from the shared markers only
fields = axis.tl.fit_transition(adatas, fusion="linear")
protein_on_cells = axis.tl.match([adatas[0], adatas[1]], fields, reference=0, reconstruct=True)[1]
```

## Predicting an unmeasured modality

```python
res = axis.tl.predict_modality(gene_matrix, protein_matrix, protein_names)
res.metrics          # per-protein Pearson / Spearman / R² / MAE / RMSE on held-out cells
res.predict(new_gene_matrix)
```

One XGBoost regressor is trained per protein with the two-stage validation
search of Methods (≈1,200–3,000 rounds at learning rates 0.07–0.03).
