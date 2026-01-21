import numpy as np
import pytest


def test_corr_bounds() -> None:
    pytest.importorskip("sklearn")
    from experiment.bridge.stage_F4_1_alignment import check_leakage

    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 5))
    Y = rng.normal(size=(40, 4))

    leakage = check_leakage(X, Y)

    assert leakage["cross_corr_checked"] is True
    assert leakage["max_abs_corr"] <= 1.0 + 1e-12


def test_identical_columns_abort() -> None:
    pytest.importorskip("sklearn")
    from experiment.bridge.stage_F4_1_alignment import check_leakage

    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 3))
    Y = rng.normal(size=(30, 4))
    Y[:, 2] = X[:, 1]

    leakage = check_leakage(X, Y)

    assert leakage["ok"] is False
    assert "i=1" in leakage["reason"]
    assert "j=2" in leakage["reason"]
    assert "rmse=" in leakage["reason"]
    assert leakage["pair_max"]["stats"]["rmse"] < 1e-6
