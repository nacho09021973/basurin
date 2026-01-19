import importlib.util
from pathlib import Path

import numpy as np


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "05_tangentes_locales.py"
    spec = importlib.util.spec_from_file_location("tangentes_locales", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_ratio_features_masses():
    module = load_module()
    masses = np.array([[2.0, 4.0, 6.0]])
    X = module.compute_ratio_features(masses, 2, "masses")
    expected = np.array([[4.0, 9.0]])
    assert np.allclose(X, expected)


def test_compute_ratio_features_M2():
    module = load_module()
    m2 = np.array([[4.0, 16.0, 36.0]])
    X = module.compute_ratio_features(m2, 2, "M2")
    expected = np.array([[4.0, 9.0]])
    assert np.allclose(X, expected)
