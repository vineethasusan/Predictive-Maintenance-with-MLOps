import numpy as np

from fleetpm.monitor import psi


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)) < 0.02


def test_psi_flags_shift():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 5000), rng.normal(1.0, 1, 5000)) > 0.25
