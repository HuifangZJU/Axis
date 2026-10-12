# Changelog

## 1.0.0

First public release, restructured from the submission snapshot
(tag `v0-submission-snapshot`).

* `axis` package with a scanpy-style API: `axis.pp`, `axis.tl`, `axis.pl`,
  `axis.metrics`, and the building blocks `axis.registration`,
  `axis.generation`, `axis.matching`.
* One transition-field implementation replaces five near-identical network
  files; parameter names are unchanged, so existing checkpoints load as is.
* All machine-specific paths replaced by `AXIS_DATA_DIR` /
  `AXIS_EXPERIMENT_DIR` and per-dataset YAML configs.
* Figure scripts moved to `reproducibility/figure1` … `figure7`; pipelines to
  `reproducibility/pipelines`.
* Compatibility with current NumPy / pandas / anndata / matplotlib releases.
* Tests, continuous integration and this documentation site.
