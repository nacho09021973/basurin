"""Test de regresión: BF-skip per-Δ en sweep_delta (Stage 03).

Contrato fijado:
  - Δ <= d/2 se marca skipped, no aborta
  - bf_per_delta registra cada punto con bf_ok/skipped
  - valid_deltas contiene solo Δ > d/2, en orden creciente
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _import_stage03():
    """Import dinámico para módulo con nombre numérico."""
    module_path = Path(__file__).resolve().parents[1] / "03_sturm_liouville.py"
    spec = importlib.util.spec_from_file_location("stage03_sl", module_path)
    assert spec is not None and spec.loader is not None, f"No se pudo cargar {module_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage03_sl"] = module
    spec.loader.exec_module(module)
    return module


stage03 = _import_stage03()
assert hasattr(stage03, "bf_filter_sweep_deltas")


def test_bf_skip_in_sweep_delta_border_point():
    """Δ = d/2 exacto se salta; el resto pasa."""
    d = 5
    delta_min = 2.5  # == d/2
    delta_max = 5.5
    n_delta = 25

    deltas = np.linspace(delta_min, delta_max, n_delta)
    bf_per_delta, valid_deltas = stage03.bf_filter_sweep_deltas(d, deltas)

    # 1) No aborta, devuelve estructuras coherentes
    assert len(bf_per_delta) == n_delta

    # 2) Punto exacto Δ = d/2 marcado skipped
    assert bf_per_delta[0]["delta"] == float(delta_min)
    assert bf_per_delta[0]["bf_bound"] == float(d / 2.0)
    assert bf_per_delta[0]["bf_ok"] is False
    assert bf_per_delta[0]["skipped"] is True

    # 3) Quedan n_delta - 1 válidos
    assert len(valid_deltas) == n_delta - 1
    assert valid_deltas[0] > d / 2.0

    # 3b) El primer válido coincide con el segundo punto del linspace
    assert valid_deltas[0] == float(deltas[1])

    # 4) Orden preservado y estrictamente creciente
    assert np.all(np.diff(valid_deltas) > 0)


def test_bf_skip_all_invalid_returns_empty():
    """Si todos Δ <= d/2, valid_deltas queda vacío (no aborta aquí)."""
    d = 5
    deltas = np.array([2.0, 2.3, 2.5])

    bf_per_delta, valid_deltas = stage03.bf_filter_sweep_deltas(d, deltas)

    assert len(bf_per_delta) == 3
    assert all(rec["skipped"] for rec in bf_per_delta)
    assert len(valid_deltas) == 0


def test_bf_skip_all_valid():
    """Si todos Δ > d/2, ninguno se salta."""
    d = 5
    deltas = np.array([3.0, 4.0, 5.0])

    bf_per_delta, valid_deltas = stage03.bf_filter_sweep_deltas(d, deltas)

    assert len(bf_per_delta) == 3
    assert all(rec["bf_ok"] for rec in bf_per_delta)
    assert len(valid_deltas) == 3
    assert np.all(np.diff(valid_deltas) > 0)
