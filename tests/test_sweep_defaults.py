from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


def _load_sturm_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "03_sturm_liouville.py"
    spec = spec_from_file_location("sturm_liouville", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_sweep_defaults_raises_only_for_explicit_delta_min():
    sl = _load_sturm_module()
    cfg = sl.Config(run="r1", mode="sweep_delta")

    sl.resolve_sweep_defaults(cfg, d=5)

    assert cfg.delta_min == pytest.approx(2.501)
    sl.validate_config(cfg, d=5)

    explicit = sl.Config(run="r1", mode="sweep_delta", delta_min=2.4)
    sl.resolve_sweep_defaults(explicit, d=5)
    assert explicit.delta_min == pytest.approx(2.4)
    sl.validate_config(explicit, d=5)


def test_resolve_sweep_defaults_keeps_delta_for_d3():
    sl = _load_sturm_module()
    cfg = sl.Config(run="r1", mode="sweep_delta", delta_min=1.55)

    sl.resolve_sweep_defaults(cfg, d=3)

    assert cfg.delta_min == pytest.approx(1.55)
    sl.validate_config(cfg, d=3)
