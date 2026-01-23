"""Tests de residuales para el solver Sturm-Liouville.

Notas de implementación:
  - Requiere mockear experiment.geometry.geometry_from_json que puede no existir.
  - Tests deterministas que no ejecutan el solver completo.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from unittest import mock

import numpy as np


def _create_geometry_mock():
    """Crea un mock para experiment.geometry.geometry_from_json."""
    mock_module = mock.MagicMock()
    mock_module.DEFAULT_Z_MAX = 1.0
    mock_module.DEFAULT_Z_MIN = 0.01
    mock_module.__version__ = "mock-1.0"
    mock_module.compile_geometry_numeric = mock.MagicMock(return_value={
        "z": [0.01, 0.5, 1.0],
        "A": [0.0, 0.0, 0.0],
        "f": [1.0, 1.0, 1.0],
        "d": 3,
        "L": 1.0,
        "z_min": 0.01,
        "z_max": 1.0,
        "N": 3,
        "family": "mock",
    })
    mock_module.load_geometry_json = mock.MagicMock(return_value={"raw": {}})
    mock_module.write_geometry_numeric = mock.MagicMock(return_value="mock-hash")
    return mock_module


def _load_sturm_module():
    """Carga el módulo 03_sturm_liouville.py con mocks necesarios."""
    # Pre-instalar mocks para evitar ImportError
    if "experiment.geometry.geometry_from_json" not in sys.modules:
        sys.modules["experiment.geometry.geometry_from_json"] = _create_geometry_mock()
    if "experiment.geometry" not in sys.modules:
        sys.modules["experiment.geometry"] = mock.MagicMock()
    if "experiment" not in sys.modules:
        sys.modules["experiment"] = mock.MagicMock()
    
    repo_root = Path(__file__).resolve().parents[0]
    module_path = repo_root / "03_sturm_liouville.py"
    spec = spec_from_file_location("sturm_liouville", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_residuals_backward_error_exact_pair():
    """Test con par eigenvalor/eigenvector exacto: residuos deben ser ~0."""
    sl = _load_sturm_module()
    
    # Matrices diagonales simples: K = diag(2,3), W = I
    K = np.array([[2.0, 0.0], [0.0, 3.0]])
    W = np.eye(2)
    eigenvalues = np.array([2.0, 3.0])
    eigenvectors = np.eye(2)

    cfg = sl.Config(run="test")
    result = sl.validate_residuals(K, W, eigenvalues, eigenvectors, cfg=cfg)

    # Verificar estructura del resultado
    assert result["residual_metric"] == "backward_error"
    assert result["residual_max"] < 1e-12
    assert result["residual_argmax_mode"] == 0
    assert result["residual_argmax_value"] == 0.0
    
    # Verificar campos adicionales
    assert "residual_mean" in result
    assert "residual_per_mode" in result
    assert "residuals_ok" in result
    assert result["residuals_ok"] is True


def test_validate_residuals_has_expected_keys():
    """Verifica que validate_residuals devuelve todas las claves esperadas."""
    sl = _load_sturm_module()
    
    K = np.eye(3) * np.array([1.0, 2.0, 3.0])
    W = np.eye(3)
    eigenvalues = np.array([1.0, 2.0, 3.0])
    eigenvectors = np.eye(3)

    cfg = sl.Config(run="test")
    result = sl.validate_residuals(K, W, eigenvalues, eigenvectors, cfg=cfg)

    expected_keys = {
        "residual_metric",
        "residual_max",
        "residual_mean",
        "residual_per_mode",
        "residual_threshold",
        "residuals_ok",
        "residual_argmax_mode",
        "residual_argmax_value",
    }
    
    for key in expected_keys:
        assert key in result, f"Falta clave esperada: {key}"


def test_validate_residuals_detects_perturbation():
    """Verifica que residuos aumentan con eigenvalores perturbados."""
    sl = _load_sturm_module()
    
    K = np.array([[2.0, 0.0], [0.0, 3.0]])
    W = np.eye(2)
    # Eigenvalores con pequeña perturbación
    eigenvalues_perturbed = np.array([2.01, 2.99])
    eigenvectors = np.eye(2)

    cfg = sl.Config(run="test")
    result = sl.validate_residuals(K, W, eigenvalues_perturbed, eigenvectors, cfg=cfg)

    # Residuos deben ser no-cero pero pequeños
    assert result["residual_max"] > 1e-14  # No exactamente cero
    assert result["residual_max"] < 1e-1   # Pero pequeños para perturbación de 0.01
