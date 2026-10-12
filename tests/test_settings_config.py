import os

import pytest

import axis
from axis.config import list_configs, load_config


def test_data_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AXIS_DATA_DIR", str(tmp_path))
    axis.settings.data_dir = None
    assert axis.data_path("SCC/huifang") == os.path.join(str(tmp_path), "SCC/huifang")
    assert axis.data_path("SCC/huifang/").endswith("/")
    assert axis.experiment_path("combined") == os.path.join(str(tmp_path), "experiments", "combined")


def test_data_dir_override(tmp_path):
    axis.settings.data_dir = tmp_path
    try:
        assert axis.data_path("x") == os.path.join(str(tmp_path), "x")
    finally:
        axis.settings.data_dir = None


@pytest.mark.parametrize("name", ["scc", "dlpfc", "xenium_mouse_brain", "xenium_breast", "phenocycler_luad"])
def test_bundled_configs(name, monkeypatch, tmp_path):
    monkeypatch.setenv("AXIS_DATA_DIR", str(tmp_path))
    axis.settings.data_dir = None
    cfg = load_config(name)
    assert cfg.name == name
    assert cfg.root.startswith(str(tmp_path))
    assert "${" not in str(cfg)


def test_list_and_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("AXIS_DATA_DIR", str(tmp_path))
    assert "scc" in list_configs()
    cfg = load_config("scc", overrides={"generation": {"grid_size": 12}})
    assert cfg.generation.grid_size == 12 and cfg.generation.knn == 48
    assert cfg.path("network_input").endswith("SCC/huifang/network_input")


def test_missing_config():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")
