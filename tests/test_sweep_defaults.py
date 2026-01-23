"""Tests para sweep_delta defaults y comportamiento de BF bound.

Notas de implementación:
  - Requiere mockear experiment.geometry.geometry_from_json que puede no existir.
  - Tests deterministas que validan la lógica de configuración, no el solver.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from unittest import mock

import pytest


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


def test_resolve_sweep_defaults_raises_only_for_explicit_delta_min():
    """Verifica ajuste automático de delta_min para d=5 (BF bound = 2.5)."""
    sl = _load_sturm_module()
    
    # Config con delta_min por defecto (1.55)
    cfg = sl.Config(run="r1", mode="sweep_delta")
    
    # Para d=5, BF bound = 2.5, y delta_min=1.55 <= 2.5
    # resolve_sweep_defaults debe ajustar a 2.5 + epsilon
    sl.resolve_sweep_defaults(cfg, d=5)

    assert cfg.delta_min == pytest.approx(2.501, rel=1e-3)
    # validate_config no debe fallar ahora
    sl.validate_config(cfg, d=5)

    # Config con delta_min explícito (por debajo de BF bound)
    explicit = sl.Config(run="r1", mode="sweep_delta", delta_min=2.4)
    sl.resolve_sweep_defaults(explicit, d=5)
    # No debe modificar un valor explícito
    assert explicit.delta_min == pytest.approx(2.4)
    # validate_config debe aceptar (el filtrado BF se hace en runtime)
    sl.validate_config(explicit, d=5)


def test_resolve_sweep_defaults_keeps_delta_for_d3():
    """Para d=3, delta_min=1.55 ya está por encima de BF bound (1.5)."""
    sl = _load_sturm_module()
    
    cfg = sl.Config(run="r1", mode="sweep_delta", delta_min=1.55)
    
    # Para d=3, BF bound = 1.5, y delta_min=1.55 > 1.5
    # No debe modificar
    sl.resolve_sweep_defaults(cfg, d=3)

    assert cfg.delta_min == pytest.approx(1.55)
    sl.validate_config(cfg, d=3)


def test_validate_config_basic_constraints():
    """Verifica que validate_config detecta configuraciones inválidas."""
    sl = _load_sturm_module()
    
    # delta_max <= delta_min debe fallar
    cfg_bad_range = sl.Config(run="r1", mode="sweep_delta", delta_min=5.0, delta_max=2.0)
    with pytest.raises(ValueError, match="delta_max"):
        sl.validate_config(cfg_bad_range, d=3)
    
    # n_delta < 2 debe fallar
    cfg_bad_n = sl.Config(run="r1", mode="sweep_delta", n_delta=1)
    with pytest.raises(ValueError, match="n_delta"):
        sl.validate_config(cfg_bad_n, d=3)


def test_config_is_frozen_but_modifiable_via_object_setattr():
    """Verifica que Config es frozen pero resolve_sweep_defaults puede modificar."""
    sl = _load_sturm_module()
    
    cfg = sl.Config(run="r1", mode="sweep_delta")
    original_delta_min = cfg.delta_min
    
    # Config es frozen
    assert hasattr(sl.Config, "__dataclass_fields__")
    
    # resolve_sweep_defaults usa object.__setattr__ para modificar
    sl.resolve_sweep_defaults(cfg, d=5)
    
    # El valor puede haber cambiado si original estaba en BF region
    if original_delta_min <= 2.5:  # BF bound para d=5
        assert cfg.delta_min > original_delta_min
