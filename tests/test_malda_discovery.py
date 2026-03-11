from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "malda" / "11_kan_pysr_discovery.py"
_SPEC = importlib.util.spec_from_file_location("malda_11_kan_pysr_discovery", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _FakeEquations:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def iterrows(self):
        for idx, row in enumerate(self._rows):
            yield idx, row


class _FakePySRRegressor:
    last_kwargs: dict[str, object] | None = None
    last_fit: dict[str, object] | None = None

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs
        self.equations_ = _FakeEquations([
            {
                "complexity": 3,
                "loss": 0.125,
                "score": 0.875,
                "equation": "x0 + x1",
                "sympy_format": "x0 + x1",
            }
        ])

    def fit(self, X: np.ndarray, y: np.ndarray, variable_names: list[str]) -> None:
        type(self).last_fit = {
            "X_shape": X.shape,
            "y_shape": y.shape,
            "variable_names": list(variable_names),
        }


def test_run_pysr_uses_backend_directory_and_exports_pareto_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pysr = types.ModuleType("pysr")
    fake_pysr.PySRRegressor = _FakePySRRegressor
    monkeypatch.setitem(sys.modules, "pysr", fake_pysr)

    out_dir = tmp_path / "runs" / "demo_run" / "experiment" / "malda_discovery" / "outputs"
    result = _MODULE.run_pysr(
        X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        y=np.array([0.5, 0.75], dtype=np.float64),
        feat_names=["m1_src", "m2_src"],
        target_name="af",
        out_dir=out_dir,
        n_iterations=5,
        maxsize=7,
        use_gpu=False,
        seed=123,
    )

    kwargs = _FakePySRRegressor.last_kwargs
    assert kwargs is not None
    assert "equation_file" not in kwargs
    assert kwargs["output_directory"] == str(out_dir / "pysr_backend")
    assert kwargs["run_id"] == "af_123"
    assert kwargs["temp_equation_file"] is False

    pareto_path = out_dir / "pysr_pareto_af.csv"
    rows = list(csv.DictReader(pareto_path.open("r", encoding="utf-8")))

    assert pareto_path.exists()
    assert rows == [
        {
            "complexity": "3",
            "loss": "0.125",
            "score": "0.875",
            "equation": "x0 + x1",
            "sympy_format": "x0 + x1",
        }
    ]
    assert result["pareto_csv"] == str(pareto_path)
    assert _FakePySRRegressor.last_fit == {
        "X_shape": (2, 2),
        "y_shape": (2,),
        "variable_names": ["m1_src", "m2_src"],
    }
