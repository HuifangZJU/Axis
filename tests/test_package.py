import importlib

import pytest

import axis

SUBMODULES = ["pp", "tl", "pl", "metrics", "registration", "generation", "matching", "io", "config", "datasets"]


def test_version():
    assert isinstance(axis.__version__, str) and axis.__version__.count(".") >= 2


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodules_import(name):
    module = getattr(axis, name)
    assert module is importlib.import_module(f"axis.{name}")
    for public in getattr(module, "__all__", []):
        assert hasattr(module, public), f"axis.{name}.{public} listed in __all__ but missing"


def test_public_api():
    for fn in ["build_pseudoimages", "prepare_canvas"]:
        assert callable(getattr(axis.pp, fn))
    for fn in ["fit_transition", "generate", "match", "integrate", "contrast", "predict_modality"]:
        assert callable(getattr(axis.tl, fn))


def test_unknown_attribute():
    with pytest.raises(AttributeError):
        axis.not_a_module  # noqa: B018
