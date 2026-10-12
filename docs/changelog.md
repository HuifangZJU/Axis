# Changelog

## 1.0.0

First public release, restructured from the submission snapshot
(tag `v0-submission-snapshot`).

* `axis` package with a scanpy-style API: `axis.pp`, `axis.tl`, `axis.pl`,
  `axis.metrics`, and the building blocks `axis.registration`,
  `axis.generation`, `axis.matching`.
* One transition-field implementation replaces five near-identical network
  files; parameter names are unchanged, so existing checkpoints load as is.
* Checkpoints written by `axis.registration.save_checkpoint` (also used by
  `axis.tl.fit_transition(out_dir=...)`) leave out the deterministic Swin
  `relative_position_index` buffers, which shrinks them from about 2.2 GB to
  36 MB; `axis.registration.load_model` accepts checkpoints with or without
  them.
* `axis.tl.fit_transition(checkpoint=..., epochs=0)` raises `ValueError` when
  the checkpoint leaves network tensors untrained (e.g. a different `fusion`,
  number of pseudo-image components or number of image channels), instead of
  predicting fields with a partly initialised network.
* `axis.pp.prepare_canvas` drops an alpha channel and rescales images with
  values above 255 (e.g. 16-bit DAPI) linearly by their maximum instead of
  clipping them; other inputs are unchanged.
* `axis.pl.spatial` and `axis.pl.matched` colour categories in a fixed order
  (the column's categorical order, otherwise a natural sort of the labels).
  `axis.pl.matched` gives each category one colour across all samples;
  `axis.pl.spatial` keeps the colours of panels that share the categories
  (a categorical column with the same categories, or the same `categories=`)
  when a section lacks one.
* All machine-specific paths replaced by `AXIS_DATA_DIR` /
  `AXIS_EXPERIMENT_DIR` and per-dataset YAML configs. The wheel ships the
  configs as `axis/_configs` (`setup.py`, `MANIFEST.in`), so
  `axis.config.load_config` also works after `pip install`.
* Figure scripts moved to `reproducibility/figure1` … `figure7`; pipelines to
  `reproducibility/pipelines`.
* Fixes in the reproducibility scripts:
    - `pipelines/training/train_multimodal_xenium.py` used an undefined
      `displacement_loss` (`NameError`);
    - a blocking `input()` debug pause was removed from
      `pipelines/generation/lung_cross_modality.py`;
    - pairwise outputs are written to `pairwise_output/` (the submission code
      wrote `output/`), where the later stages read them;
    - `matplotlib.cm.get_cmap` (removed in matplotlib 3.9) was replaced by
      `matplotlib.pyplot.get_cmap`;
    - `run_virtual_counterfactual` in `pipelines/generation/mouse_ad_time.py`
      passed a misspelled keyword `basezize=` to `make_common_grid_from_coords`
      (`TypeError`); it now passes `max_x=1024, max_y=1024`.
* Compatibility with current NumPy / pandas / anndata / matplotlib releases.
* Tests, continuous integration and this documentation site. The
  documentation build validates internal links and anchors (`mkdocs build --strict`
  fails on broken ones), and `tests/test_docs.py` checks the `axis` names and
  calls in every Python code block of `README.md` and `docs/` against the API
  without running them.
