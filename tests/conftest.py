import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
os.environ.setdefault("AXIS_VERBOSITY", "0")


@pytest.fixture(scope="session")
def toy_series():
    """Three small simulated sections on a 64-px canvas with pseudo-images."""
    import axis

    adatas = axis.datasets.toy_series(n_samples=3, canvas_size=64, n_genes=24, spot_spacing=3.0, seed=0)
    axis.pp.build_pseudoimages(adatas, n_comps=4, preset="visium")
    return adatas


@pytest.fixture(scope="session")
def toy_fields(toy_series):
    import axis

    return axis.tl.fit_transition(toy_series, epochs=1, max_steps=2, device="cpu", morph_win=5, mol_win=3, seed=0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
