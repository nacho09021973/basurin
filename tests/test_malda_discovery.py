from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
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


class _FakeKAN:
    last_init: dict[str, object] | None = None
    last_fit: dict[str, object] | None = None

    def __init__(self, **kwargs: object) -> None:
        type(self).last_init = kwargs
        self.act_fun = []

    def fit(self, dataset: dict[str, object], **kwargs: object) -> dict[str, list[float]]:
        type(self).last_fit = {
            "dataset_keys": sorted(dataset.keys()),
            "kwargs": kwargs,
        }
        return {"train_loss": [0.1], "test_loss": [0.2]}

    def train(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("run_kan must use model.fit, not model.train")

    def plot(self, folder: str, beta: int) -> None:
        Path(folder).mkdir(parents=True, exist_ok=True)


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


def test_claim_grade_reduces_inputs_to_primitive_inspiral_features() -> None:
    selected, analysis_mode = _MODULE.resolve_input_features("S_f", "claim_grade")

    assert analysis_mode == "claim_grade"
    assert selected == ["m1_src", "m2_src", "chi_eff"]

    cols = ["m1_src", "m2_src", "chi_eff", "q", "eta", "S_f"]
    data = np.array(
        [
            [30.0, 20.0, 0.0, 20.0 / 30.0, (30.0 * 20.0) / (50.0**2), 3000.0],
            [31.0, 21.0, 0.1, 21.0 / 31.0, (31.0 * 21.0) / (52.0**2), 3200.0],
            [32.0, 22.0, 0.2, 22.0 / 32.0, (32.0 * 22.0) / (54.0**2), 3400.0],
            [33.0, 23.0, 0.3, 23.0 / 33.0, (33.0 * 23.0) / (56.0**2), 3600.0],
            [34.0, 24.0, 0.4, 24.0 / 34.0, (34.0 * 24.0) / (58.0**2), 3800.0],
        ],
        dtype=np.float64,
    )
    _, _, feat_names = _MODULE.prepare_XY(cols, data, "S_f", selected)

    assert feat_names == ["log_m1_src", "log_m2_src", "chi_eff"]
    assert "q" not in feat_names
    assert "eta" not in feat_names


def test_claim_grade_symmetric_reduces_inputs_to_q_eta_chi_eff() -> None:
    selected, analysis_mode = _MODULE.resolve_input_features("E_rad_frac", "claim_grade_symmetric")

    assert analysis_mode == "claim_grade_symmetric"
    assert selected == ["q", "eta", "chi_eff"]

    cols = ["m1_src", "m2_src", "q", "eta", "chi_eff", "E_rad_frac"]
    data = np.array(
        [
            [30.0, 20.0, 20.0 / 30.0, (30.0 * 20.0) / (50.0**2), 0.0, 0.04],
            [31.0, 21.0, 21.0 / 31.0, (31.0 * 21.0) / (52.0**2), 0.1, 0.042],
            [32.0, 22.0, 22.0 / 32.0, (32.0 * 22.0) / (54.0**2), 0.2, 0.044],
            [33.0, 23.0, 23.0 / 33.0, (33.0 * 23.0) / (56.0**2), 0.3, 0.046],
            [34.0, 24.0, 24.0 / 34.0, (34.0 * 24.0) / (58.0**2), 0.4, 0.048],
        ],
        dtype=np.float64,
    )
    _, _, feat_names = _MODULE.prepare_XY(cols, data, "E_rad_frac", selected)

    assert feat_names == ["q", "eta", "chi_eff"]
    assert "log_m1_src" not in feat_names
    assert "log_m2_src" not in feat_names


def test_runtime_timeline_emits_heartbeat_and_persists_jsonl(tmp_path: Path) -> None:
    timeline = _MODULE.RuntimeTimeline(tmp_path / "timeline.jsonl")

    result = _MODULE.run_with_heartbeat(
        lambda: (time.sleep(0.03), "ok")[1],
        timeline=timeline,
        target_name="af",
        step_name="pysr_fit",
        heartbeat_seconds=0.01,
    )

    events = timeline.snapshot()
    event_names = [event["event"] for event in events]
    lines = (tmp_path / "timeline.jsonl").read_text(encoding="utf-8").splitlines()

    assert result == "ok"
    assert event_names[0] == "step_started"
    assert "step_heartbeat" in event_names
    assert event_names[-1] == "step_completed"
    assert len(lines) == len(events)


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
        parsimony=0.123,
        populations=9,
        population_size=11,
        ncycles_per_iteration=13,
        use_gpu=False,
        seed=123,
    )

    kwargs = _FakePySRRegressor.last_kwargs
    assert kwargs is not None
    assert "equation_file" not in kwargs
    assert kwargs["output_directory"] == str(out_dir / "pysr_backend")
    assert kwargs["run_id"] == "af_123"
    assert kwargs["temp_equation_file"] is False
    assert kwargs["parsimony"] == 0.123
    assert kwargs["populations"] == 9
    assert kwargs["population_size"] == 11
    assert kwargs["ncycles_per_iteration"] == 13

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


def test_run_kan_uses_fit_and_keeps_backend_under_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_kan = types.ModuleType("kan")
    fake_kan.KAN = _FakeKAN
    monkeypatch.setitem(sys.modules, "kan", fake_kan)

    out_dir = tmp_path / "runs" / "demo_run" / "experiment" / "malda_discovery" / "outputs"
    result = _MODULE.run_kan(
        X=np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0]], dtype=np.float64),
        y=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
        feat_names=["m1_src", "m2_src"],
        target_name="E_rad_frac",
        out_dir=out_dir,
        n_epochs=3,
        seed=123,
    )

    assert _FakeKAN.last_init is not None
    assert _FakeKAN.last_init["auto_save"] is False
    assert _FakeKAN.last_init["seed"] == 123
    assert _FakeKAN.last_init["ckpt_path"] == str(out_dir / "kan_backend" / "E_rad_frac")

    assert _FakeKAN.last_fit == {
        "dataset_keys": ["test_input", "test_label", "train_input", "train_label"],
        "kwargs": {"opt": "Adam", "steps": 3, "lamb": 0.01, "display_metrics": None},
    }
    assert result["status"] == "ok"
    assert result["target"] == "E_rad_frac"


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
    assert stage_summary["config"]["heartbeat_seconds"] == 30.0
    assert stage_summary["config"]["pysr_iterations"] == _MODULE.SEARCH_DEFAULTS["strict_premerger"]["pysr_iterations"]
    assert stage_summary["config"]["pysr_maxsize"] == _MODULE.SEARCH_DEFAULTS["strict_premerger"]["pysr_maxsize"]
    assert stage_summary["config"]["pysr_parsimony"] == _MODULE.SEARCH_DEFAULTS["strict_premerger"]["pysr_parsimony"]
    assert stage_summary["config"]["pysr_populations"] == _MODULE.SEARCH_DEFAULTS["strict_premerger"]["pysr_populations"]
    assert stage_summary["config"]["pysr_population_size"] == _MODULE.SEARCH_DEFAULTS["strict_premerger"]["pysr_population_size"]
    assert (
        stage_summary["config"]["pysr_ncycles_per_iteration"]
        == _MODULE.SEARCH_DEFAULTS["strict_premerger"]["pysr_ncycles_per_iteration"]
    )
    assert stage_summary["config"]["kan_epochs"] == _MODULE.SEARCH_DEFAULTS["strict_premerger"]["kan_epochs"]
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
    assert (stage_dir / "outputs" / "runtime_timeline.json").exists()
    assert (stage_dir / "outputs" / "runtime_timeline.jsonl").exists()
    assert "runtime_timeline_json" in stage_summary["outputs"]
    assert "runtime_timeline_jsonl" in stage_summary["outputs"]


def test_main_records_claim_grade_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runsroot_claim"
    table_path = tmp_path / "inputs" / "event_features_claim.csv"
    _write_feature_table(table_path)

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    rc = _MODULE.main(
        [
            "--run-id",
            "malda_claim_grade_smoke",
            "--feature-table",
            str(table_path),
            "--feature-policy",
            "claim_grade",
            "--no-kan",
            "--no-pysr",
        ]
    )

    stage_dir = runs_root / "malda_claim_grade_smoke" / "experiment" / "malda_discovery"
    stage_summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert stage_summary["config"]["feature_policy"] == "claim_grade"
    assert stage_summary["config"]["heartbeat_seconds"] == 30.0
    assert stage_summary["config"]["pysr_maxsize"] == _MODULE.SEARCH_DEFAULTS["claim_grade"]["pysr_maxsize"]
    assert stage_summary["config"]["pysr_parsimony"] == _MODULE.SEARCH_DEFAULTS["claim_grade"]["pysr_parsimony"]
    assert stage_summary["config"]["pysr_iterations"] == _MODULE.SEARCH_DEFAULTS["claim_grade"]["pysr_iterations"]
    assert stage_summary["results"]["features_by_target"]["S_f"] == ["log_m1_src", "log_m2_src", "chi_eff"]
    assert stage_summary["results"]["analysis_mode_by_target"]["S_f"] == "claim_grade"
    assert (stage_dir / "outputs" / "runtime_timeline.json").exists()
    assert (stage_dir / "outputs" / "runtime_timeline.jsonl").exists()


def test_main_records_claim_grade_symmetric_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runsroot_claim_symmetric"
    table_path = tmp_path / "inputs" / "event_features_claim_symmetric.csv"
    _write_feature_table(table_path)

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    rc = _MODULE.main(
        [
            "--run-id",
            "malda_claim_grade_symmetric_smoke",
            "--feature-table",
            str(table_path),
            "--feature-policy",
            "claim_grade_symmetric",
            "--no-kan",
            "--no-pysr",
        ]
    )

    stage_dir = runs_root / "malda_claim_grade_symmetric_smoke" / "experiment" / "malda_discovery"
    stage_summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert stage_summary["config"]["feature_policy"] == "claim_grade_symmetric"
    assert (
        stage_summary["config"]["pysr_maxsize"]
        == _MODULE.SEARCH_DEFAULTS["claim_grade_symmetric"]["pysr_maxsize"]
    )
    assert (
        stage_summary["config"]["pysr_parsimony"]
        == _MODULE.SEARCH_DEFAULTS["claim_grade_symmetric"]["pysr_parsimony"]
    )
    assert (
        stage_summary["config"]["pysr_iterations"]
        == _MODULE.SEARCH_DEFAULTS["claim_grade_symmetric"]["pysr_iterations"]
    )
    assert stage_summary["results"]["features_by_target"]["E_rad_frac"] == ["q", "eta", "chi_eff"]
    assert stage_summary["results"]["analysis_mode_by_target"]["E_rad_frac"] == "claim_grade_symmetric"
    assert (stage_dir / "outputs" / "runtime_timeline.json").exists()
    assert (stage_dir / "outputs" / "runtime_timeline.jsonl").exists()


def test_main_runs_pysr_before_kan_when_both_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runsroot_order"
    table_path = tmp_path / "inputs" / "event_features_order.csv"
    _write_feature_table(table_path)

    call_order: list[str] = []

    def _fake_run_pysr(*args, **kwargs):
        call_order.append("pysr")
        return {
            "status": "ok",
            "target": "af",
            "n_events": 5,
            "best_equation": {},
            "pareto_equations": [],
            "pareto_csv": None,
        }

    def _fake_run_kan(*args, **kwargs):
        call_order.append("kan")
        return {
            "status": "ok",
            "target": "af",
            "n_events": 5,
            "train_loss": 0.1,
            "test_loss": 0.2,
            "feature_importances": {},
            "symbolic_suggestions": [],
        }

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    monkeypatch.setattr(_MODULE, "run_pysr", _fake_run_pysr)
    monkeypatch.setattr(_MODULE, "run_kan", _fake_run_kan)

    rc = _MODULE.main(
        [
            "--run-id",
            "malda_order_smoke",
            "--feature-table",
            str(table_path),
            "--targets",
            "af",
        ]
    )

    assert rc == 0
    assert call_order == ["pysr", "kan"]
