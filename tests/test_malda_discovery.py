from __future__ import annotations

import csv
import importlib.util
import json
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


def _write_feature_table(path: Path) -> None:
    fieldnames = list(dict.fromkeys(_MODULE.PREMERGER_ONLY + list(_MODULE.TARGETS.keys())))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(5):
            m1 = 35.0 + idx
            m2 = 25.0 + idx
            total = m1 + m2
            chirp = 24.0 + idx
            q = m2 / m1
            row = {
                "m1_src": m1,
                "m2_src": m2,
                "M_total": total,
                "Mchirp": chirp,
                "chi_eff": 0.02 * idx,
                "q": q,
                "eta": (m1 * m2) / (total**2),
                "delta": (m1 - m2) / total,
                "log_q": np.log(q),
                "Mchirp_over_Mtotal": chirp / total,
                "E_rad_frac": 0.04 + 0.002 * idx,
                "af": 0.67 + 0.01 * idx,
                "S_f": 3200.0 + 100.0 * idx,
                "delta_S": 420.0 + 8.0 * idx,
                "S_ratio": 1.18 + 0.01 * idx,
                "Q_220": 3.2 + 0.05 * idx,
                "F_220_dimless": 0.55 + 0.01 * idx,
                "f_ratio_221_220": 1.28 + 0.01 * idx,
            }
            writer.writerow(row)


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


@pytest.mark.parametrize(
    ("target_name", "blocked_features"),
    [
        ("S_f", {"Mf", "xi_f", "S_f", "log_Mf"}),
        ("E_rad_frac", {"Mf", "E_rad_Msun", "E_rad_frac", "log_Mf"}),
        ("Q_220", {"af", "Q_220", "F_220_dimless", "Q_ratio_221_220", "log_af"}),
    ],
)
def test_strict_premerger_blocks_leaky_inputs_and_transforms(
    target_name: str,
    blocked_features: set[str],
) -> None:
    selected, analysis_mode = _MODULE.resolve_input_features(target_name, "strict_premerger")

    assert analysis_mode == "discovery"
    assert all(feature in _MODULE.PREMERGER_ONLY for feature in selected)
    assert blocked_features.isdisjoint(selected)

    cols = list(
        dict.fromkeys(
            _MODULE.PREMERGER_ONLY
            + sorted(base for base in blocked_features if not base.startswith("log_"))
            + list(_MODULE.TARGETS.keys())
        )
    )
    data = np.array(
        [
            [10.0 + row_idx + col_idx for col_idx in range(len(cols))]
            for row_idx in range(5)
        ],
        dtype=np.float64,
    )
    _, _, feat_names = _MODULE.prepare_XY(cols, data, target_name, selected)

    assert blocked_features.isdisjoint(feat_names)
    assert "log_Mf" not in feat_names
    assert "log_xi_f" not in feat_names


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


def test_main_records_strict_feature_policy_and_final_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runsroot"
    table_path = tmp_path / "inputs" / "event_features.csv"
    _write_feature_table(table_path)

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    rc = _MODULE.main(
        [
            "--run-id",
            "malda_feature_policy_smoke",
            "--feature-table",
            str(table_path),
            "--feature-policy",
            "strict_premerger",
            "--no-kan",
            "--no-pysr",
        ]
    )

    stage_dir = runs_root / "malda_feature_policy_smoke" / "experiment" / "malda_discovery"
    stage_summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert stage_summary["config"]["feature_policy"] == "strict_premerger"
    assert stage_summary["results"]["features_by_target"]["S_f"] == [
        "log_m1_src",
        "log_m2_src",
        "log_M_total",
        "log_Mchirp",
        "chi_eff",
        "q",
        "eta",
        "delta",
        "log_q",
        "Mchirp_over_Mtotal",
    ]
    assert "Mf" not in stage_summary["results"]["features_by_target"]["S_f"]
    assert "xi_f" not in stage_summary["results"]["features_by_target"]["S_f"]
    assert stage_summary["results"]["analysis_mode_by_target"]["Q_220"] == "discovery"
