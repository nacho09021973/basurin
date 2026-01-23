"""Tests para validate_residuals en 03_sturm_liouville.py"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest import mock
import sys

import numpy as np


def _create_geometry_mock():
    """Crea un mock del módulo experiment.geometry.geometry_from_json.
    
    Este módulo puede no existir en el repo, pero 03_sturm_liouville.py lo importa.
    """
    mock_module = mock.MagicMock()
    mock_module.DEFAULT_Z_MAX = 1.0
    mock_module.DEFAULT_Z_MIN = 0.01
    mock_module.__version__ = "mock-1.0"
    mock_module.compile_geometry_numeric = mock.MagicMock(return_value={})
    mock_module.load_geometry_json = mock.MagicMock(return_value={})
    mock_module.write_geometry_numeric = mock.MagicMock(return_value="mock-hash")
    return mock_module


def _load_sturm_module():
    """Load 03_sturm_liouville.py module dynamically.
    
    Pre-installs mocks for dependencies that may not exist.
    """
    # Pre-install mock for experiment.geometry before import
    if "experiment.geometry.geometry_from_json" not in sys.modules:
        sys.modules["experiment"] = mock.MagicMock()
        sys.modules["experiment.geometry"] = mock.MagicMock()
        sys.modules["experiment.geometry.geometry_from_json"] = _create_geometry_mock()
    
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "03_sturm_liouville.py"
    
    if not module_path.exists():
        raise FileNotFoundError(f"No se encuentra el módulo: {module_path}")
    
    spec = spec_from_file_location("sturm_liouville", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar spec para {module_path}")
    
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_residuals_backward_error_exact_pair():
    """Test validate_residuals con autovalores/autovectores exactos."""
    sl = _load_sturm_module()
    K = np.array([[2.0, 0.0], [0.0, 3.0]])
    W = np.eye(2)
    eigenvalues = np.array([2.0, 3.0])
    eigenvectors = np.eye(2)

    cfg = sl.Config(run="test")
    result = sl.validate_residuals(K, W, eigenvalues, eigenvectors, cfg=cfg)

    assert result["residual_metric"] == "backward_error"
    assert result["residual_max"] < 1e-12
    assert result["residual_argmax_mode"] == 0
    assert result["residual_argmax_value"] == 0.0


def test_validate_residuals_structure():
    """Test que validate_residuals devuelve todas las claves esperadas."""
    sl = _load_sturm_module()
    K = np.array([[4.0, 0.0], [0.0, 9.0]])
    W = np.eye(2)
    eigenvalues = np.array([4.0, 9.0])
    eigenvectors = np.eye(2)

    cfg = sl.Config(run="test")
    result = sl.validate_residuals(K, W, eigenvalues, eigenvectors, cfg=cfg)

    # Claves obligatorias
    assert "residual_metric" in result
    assert "residual_max" in result
    assert "residual_mean" in result
    assert "residual_per_mode" in result
    assert "residual_threshold" in result
    assert "residuals_ok" in result
    assert "residual_argmax_mode" in result
    assert "residual_argmax_value" in result
    
    # Claves nuevas (non-breaking)
    assert "residual_first_k" in result
    assert "residual_max_first_k" in result
    assert "residuals_ok_first_k" in result
    assert "residual_status" in result


def test_validate_residuals_with_perturbation():
    """Test validate_residuals detecta residuos cuando hay perturbación."""
    sl = _load_sturm_module()
    K = np.array([[2.0, 0.0], [0.0, 3.0]])
    W = np.eye(2)
    # Autovalores ligeramente perturbados
    eigenvalues = np.array([2.01, 3.01])
    eigenvectors = np.eye(2)

    cfg = sl.Config(run="test")
    result = sl.validate_residuals(K, W, eigenvalues, eigenvectors, cfg=cfg)

    # Debe detectar residuos no nulos
    assert result["residual_max"] > 0
    assert len(result["residual_per_mode"]) == 2
