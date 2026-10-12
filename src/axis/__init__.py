"""AXIS: virtual tissue generation and spatially matched analysis from spatial omics.

Public API (scanpy-style):

* ``axis.pp`` – canvases and molecular pseudo-images
  ([`axis.pp.prepare_canvas`][axis.pp.prepare_canvas],
  [`axis.pp.build_pseudoimages`][axis.pp.build_pseudoimages])
* ``axis.tl`` – transition fields, generation and matching
  ([`axis.tl.fit_transition`][axis.tl.fit_transition], [`axis.tl.generate`][axis.tl.generate],
  [`axis.tl.match`][axis.tl.match], [`axis.tl.integrate`][axis.tl.integrate])
* ``axis.pl`` – plotting
* ``axis.metrics`` – evaluation metrics

Lower-level building blocks live in ``axis.registration``,
``axis.generation`` and ``axis.matching``. Data locations are configured
through [`axis.settings`][axis.settings] (``AXIS_DATA_DIR``) and ``axis.config``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ._version import __version__
from .settings import data_path, experiment_path, settings

_SUBMODULES = {
    "pp",
    "tl",
    "pl",
    "metrics",
    "registration",
    "generation",
    "matching",
    "io",
    "config",
    "datasets",
}

__all__ = ["__version__", "settings", "data_path", "experiment_path", *sorted(_SUBMODULES)]


def __getattr__(name: str):
    # Submodules are imported lazily so that ``import axis`` stays fast and does
    # not initialise PyTorch until it is needed.
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module 'axis' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)


if TYPE_CHECKING:  # pragma: no cover
    from . import config, datasets, generation, io, matching, metrics, pl, pp, registration, tl
