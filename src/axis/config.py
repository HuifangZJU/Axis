"""Per-dataset YAML configuration.

Every dataset analysed in the manuscript has a YAML file in ``configs/``
(``scc.yaml``, ``dlpfc.yaml``, ``xenium_mouse_brain.yaml``,
``xenium_breast.yaml``, ``phenocycler_luad.yaml``). A config records where the
dataset lives (relative to ``AXIS_DATA_DIR``) and the default parameters used
for pseudo-image construction, transition-field learning and virtual-tissue
generation (Methods, "AXIS analysis of real datasets").

String values may reference environment variables or the two AXIS roots::

    root: ${AXIS_DATA_DIR}/SCC/huifang
    checkpoint: ${AXIS_EXPERIMENT_DIR}/combined/scc_patient10/model_1000.pth

Relative ``root`` values are interpreted relative to ``AXIS_DATA_DIR``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .settings import settings

__all__ = ["load_config", "list_configs", "config_dir", "DatasetConfig"]

_VAR = re.compile(r"\$\{([^}]+)\}")


def config_dir() -> Path:
    """Directory holding the bundled dataset configs.

    Looks for ``configs/`` next to the package (source checkout) and falls back
    to the copy shipped inside the wheel (``axis/_configs``). ``AXIS_CONFIG_DIR``
    overrides both.
    """
    env = os.environ.get("AXIS_CONFIG_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for candidate in (here.parents[2] / "configs", here.parent / "_configs"):
        if candidate.is_dir():
            return candidate
    return here.parents[2] / "configs"


def list_configs() -> list[str]:
    """Names of the available dataset configs."""
    return sorted(p.stem for p in config_dir().glob("*.yaml"))


def _expand(value: Any) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key == "AXIS_DATA_DIR":
                return str(settings.data_dir)
            if key == "AXIS_EXPERIMENT_DIR":
                return str(settings.experiment_dir)
            return os.environ.get(key, match.group(0))

        return os.path.expanduser(_VAR.sub(repl, value))
    if isinstance(value, Mapping):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


class DatasetConfig(dict):
    """A ``dict`` with attribute access and path helpers.

    >>> cfg = load_config("scc")          # doctest: +SKIP
    >>> cfg.root                          # doctest: +SKIP
    '/path/to/axis_data/SCC/huifang'
    >>> cfg.path("network_input")         # doctest: +SKIP
    '/path/to/axis_data/SCC/huifang/network_input'
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - mirrors attribute semantics
            raise AttributeError(item) from exc
        return DatasetConfig(value) if isinstance(value, dict) else value

    def path(self, *parts: Union[str, os.PathLike]) -> str:
        """Join ``parts`` below the dataset ``root``."""
        return os.path.join(self["root"], *[str(p) for p in parts])


def load_config(name_or_path: Union[str, os.PathLike], overrides: Optional[Dict[str, Any]] = None) -> DatasetConfig:
    """Load a dataset config by name (``"scc"``) or by file path.

    Parameters
    ----------
    name_or_path
        Either a config name found in :func:`config_dir` or a path to a YAML file.
    overrides
        Optional mapping merged (shallowly per top-level section) on top of the file.
    """
    import yaml

    path = Path(name_or_path)
    if not path.suffix:
        path = config_dir() / f"{name_or_path}.yaml"
    if not path.is_file():
        available = ", ".join(list_configs()) or "<none>"
        raise FileNotFoundError(f"Config {name_or_path!r} not found (available: {available}).")

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if overrides:
        for key, value in overrides.items():
            if isinstance(value, Mapping) and isinstance(raw.get(key), Mapping):
                raw[key] = {**raw[key], **value}
            else:
                raw[key] = value

    cfg = DatasetConfig(_expand(raw))
    root = cfg.get("root")
    if root is not None and not os.path.isabs(root):
        cfg["root"] = os.path.join(str(settings.data_dir), root)
    cfg.setdefault("name", path.stem)
    return cfg
