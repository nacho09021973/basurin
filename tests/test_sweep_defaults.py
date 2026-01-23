"""Tests para resolve_sweep_defaults y validate_config en 03_sturm_liouville.py"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest import mock
import sys

import pytest


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


def test_resolve_sweep_defaults_raises_only_for_explicit_delta_min():
    """Test que resolve_sweep_defaults ajusta delta_min para d=5 (BF bound)."""
    sl = _load_sturm_module()
    cfg = sl.Config(run="r1", mode="sweep_delta")

    sl.resolve_sweep_defaults(cfg, d=5)

    # Para d=5, BF bound es 2.5, así que delta_min se ajusta a 2.501
    assert cfg.delta_min == pytest.approx(2.501)
    sl.validate_config(cfg, d=5)

    # Con delta_min explícito, no se modifica
    explicit = sl.Config(run="r1", mode="sweep_delta", delta_min=2.4)
    sl.resolve_sweep_defaults(explicit, d=5)
    assert explicit.delta_min == pytest.approx(2.4)
    sl.validate_config(explicit, d=5)


def test_resolve_sweep_defaults_keeps_delta_for_d3():
    """Test que resolve_sweep_defaults mantiene delta_min=1.55 para d=3."""
    sl = _load_sturm_module()
    cfg = sl.Config(run="r1", mode="sweep_delta", delta_min=1.55)

    sl.resolve_sweep_defaults(cfg, d=3)

    # Para d=3, BF bound es 1.5, así que delta_min=1.55 es válido y no se modifica
    assert cfg.delta_min == pytest.approx(1.55)
    sl.validate_config(cfg, d=3)


def test_validate_config_rejects_invalid_sweep():
    """Test que validate_config rechaza configuraciones inválidas."""
    sl = _load_sturm_module()
    
    # delta_max <= delta_min
    cfg_invalid = sl.Config(run="r1", mode="sweep_delta", delta_min=5.0, delta_max=4.0)
    with pytest.raises(ValueError, match="delta_max"):
        sl.validate_config(cfg_invalid, d=3)
    
    # n_delta < 2
    cfg_few = sl.Config(run="r1", mode="sweep_delta", n_delta=1)
    with pytest.raises(ValueError, match="n_delta"):
        sl.validate_config(cfg_few, d=3)


def test_validate_config_accepts_valid_sweep():
    """Test que validate_config acepta configuraciones válidas."""
    sl = _load_sturm_module()
    
    cfg = sl.Config(run="r1", mode="sweep_delta", delta_min=1.6, delta_max=5.0, n_delta=10)
    # No debe lanzar excepción
    sl.validate_config(cfg, d=3)
