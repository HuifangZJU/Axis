"""Global settings for AXIS.

All file-system locations used by AXIS and by the reproducibility scripts are
resolved through this module, so that no absolute, machine-specific path needs
to appear in the code base.

Two environment variables are recognised:

``AXIS_DATA_DIR``
    Root directory that holds the processed datasets (one sub-directory per
    dataset, e.g. ``SCC/huifang``, ``DLPFC/huifang``, ``xenium/mouse_brain``)
    and the ``paper_analysis`` output folder used by the figure scripts.
    Defaults to ``./data`` relative to the current working directory.

``AXIS_EXPERIMENT_DIR``
    Root directory for trained models, training logs and intermediate
    training figures. Defaults to ``$AXIS_DATA_DIR/experiments``.

Both can also be changed at run time::

    import axis
    axis.settings.data_dir = "/path/to/axis_data"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike]

__all__ = ["settings", "data_path", "experiment_path", "AxisSettings"]


class AxisSettings:
    """Container for run-time settings.

    Attributes
    ----------
    data_dir
        Root of the processed datasets (``AXIS_DATA_DIR``).
    experiment_dir
        Root of trained models and logs (``AXIS_EXPERIMENT_DIR``).
    device
        Default torch device. ``"auto"`` selects CUDA when available.
    verbosity
        0 = silent, 1 = progress messages (default), 2 = debug.
    """

    def __init__(self) -> None:
        self._data_dir: Optional[Path] = None
        self._experiment_dir: Optional[Path] = None
        self.device: str = os.environ.get("AXIS_DEVICE", "auto")
        self.verbosity: int = int(os.environ.get("AXIS_VERBOSITY", "1"))

    # ------------------------------------------------------------------ data
    @property
    def data_dir(self) -> Path:
        if self._data_dir is not None:
            return self._data_dir
        return Path(os.environ.get("AXIS_DATA_DIR", "data")).expanduser()

    @data_dir.setter
    def data_dir(self, value: Optional[PathLike]) -> None:
        self._data_dir = None if value is None else Path(value).expanduser()

    # ----------------------------------------------------------- experiments
    @property
    def experiment_dir(self) -> Path:
        if self._experiment_dir is not None:
            return self._experiment_dir
        env = os.environ.get("AXIS_EXPERIMENT_DIR")
        if env:
            return Path(env).expanduser()
        return self.data_dir / "experiments"

    @experiment_dir.setter
    def experiment_dir(self, value: Optional[PathLike]) -> None:
        self._experiment_dir = None if value is None else Path(value).expanduser()

    # ---------------------------------------------------------------- device
    def resolve_device(self, device: Optional[str] = None) -> str:
        """Return a concrete torch device string (``"cuda"``, ``"cpu"`` ...)."""
        device = device or self.device
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover - torch is a hard dependency
                return "cpu"
        return device

    def __repr__(self) -> str:
        return (
            f"AxisSettings(data_dir={str(self.data_dir)!r}, "
            f"experiment_dir={str(self.experiment_dir)!r}, device={self.device!r}, "
            f"verbosity={self.verbosity})"
        )


settings = AxisSettings()


def _join(root: Path, parts) -> str:
    # ``os.path.join`` keeps a trailing separator ("SCC/huifang/" -> ".../SCC/huifang/"),
    # which matters for scripts that build paths by string concatenation.
    return os.path.join(str(root), *[str(p) for p in parts]) if parts else str(root)


def data_path(*parts: PathLike) -> str:
    """Return a path below :attr:`AxisSettings.data_dir` as a string.

    Examples
    --------
    >>> data_path("SCC/huifang", "network_input")  # doctest: +SKIP
    '/path/to/axis_data/SCC/huifang/network_input'
    """
    return _join(settings.data_dir, parts)


def experiment_path(*parts: PathLike) -> str:
    """Return a path below :attr:`AxisSettings.experiment_dir` as a string."""
    return _join(settings.experiment_dir, parts)


def log(message: str, level: int = 1) -> None:
    """Print ``message`` if the global verbosity is at least ``level``."""
    if settings.verbosity >= level:
        print(message, flush=True)
