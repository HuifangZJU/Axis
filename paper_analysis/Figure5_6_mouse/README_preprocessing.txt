Unified preprocessing used by the updated scripts
=================================================

Default global settings:
    LIBRARY_SIZE_NORMALIZE = True
    NORMALIZATION_TARGET_SUM = 1e4
    LOG1P_BEFORE_ANALYSIS = True

Canonical source-expression preprocessing:
    raw/count-like expression
      -> per-location library-size normalization using ALL genes in adata.X
      -> optional log1p
      -> subtraction / temporal comparison

Disease spatial delta:
    preprocess(disease_i) - preprocess(normal_i)

5.7-month domain-level fidelity evaluation:
    mean(preprocess(disease) within domain)
      - mean(preprocess(normal) within domain)

The real and virtual domain-level evaluation keeps independent valid masks,
matching the original V2 design. The spatial virtual delta uses aligned paired
locations and common-valid support only for location-level analyses/maps.

Important:
- Library sizes are calculated from the complete gene panel in each h5ad, even
  when only a marker-gene subset is extracted.
- Delta AnnData values are never normalized/log-transformed a second time.
