#!/usr/bin/env python3
"""
exp_05 wrapper: module re-export + entrypoint.

Loads canonical implementation from exp_04 and re-exports the API expected by tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

_HERE = Path(__file__).resolve().parent
_IMPL_PATH = (_HERE.parent / "exp_04_tangentes_locales" / "05_tangentes_locales.py").resolve()

if not _IMPL_PATH.exists():
    raise FileNotFoundError(f"No se encontró la implementación canónica en {_IMPL_PATH}")

_spec = importlib.util.spec_from_file_location("tangentes_locales_impl", _IMPL_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"No se pudo crear spec para {_IMPL_PATH}")

_impl: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)  # type: ignore[attr-defined]

_required = ["resolve_spectrum_path", "compute_ratio_features"]
_missing = [n for n in _required if not hasattr(_impl, n)]
if _missing:
    raise AttributeError(f"Implementación no expone {_missing}. Revisa {_IMPL_PATH}")

resolve_spectrum_path = getattr(_impl, "resolve_spectrum_path")
compute_ratio_features = getattr(_impl, "compute_ratio_features")

def __getattr__(name: str) -> Any:
    return getattr(_impl, name)

def __dir__():
    return sorted(set(globals().keys()) | set(dir(_impl)))

__all__ = ["resolve_spectrum_path", "compute_ratio_features"]

if __name__ == "__main__":
    if hasattr(_impl, "main"):
        _impl.main()
