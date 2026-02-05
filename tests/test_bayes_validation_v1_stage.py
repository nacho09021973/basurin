from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


STAGE = "stages/bayes_validation_v1_stage.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _setup_run_valid(run_dir: Path, verdict: str = "PASS") -> None:
    _write_json(run_dir / "RUN_VALID" / "outputs" / "run_valid.json", {"overall_verdict": verdict})


def _setup_spectrum_input(run_dir: Path, content: bytes = b"dummy-spectrum") -> Path:
    path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _run_stage(runs_root: Path, run_id: str, extra_args: list[str] | None = None, env: dict[str, str] | None = None):
    cmd = ["python", STAGE, "--run", run_id]
    if extra_args:
        cmd.extend(extra_args)
    run_env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    if env:
        run_env.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, check=False, env=run_env)


def test_gate_run_valid(tmp_path: Path) -> None:
    run_id = "2040-01-01__unit__bayes_gate"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    _setup_run_valid(run_dir, verdict="FAIL")

    res = _run_stage(runs_root, run_id)

    assert res.returncode == 2
    assert not (run_dir / "bayes_validation_v1").exists()


def test_output_contract_files_exist(tmp_path: Path) -> None:
    run_id = "2040-01-01__unit__bayes_contract"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    _setup_run_valid(run_dir)
    _setup_spectrum_input(run_dir)

    res = _run_stage(runs_root, run_id)

    assert res.returncode == 0, res.stderr
    stage_dir = run_dir / "bayes_validation_v1"
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (stage_dir / "outputs" / "bayes_validation.json").exists()


def test_determinism_same_seed_same_output(tmp_path: Path) -> None:
    run_id = "2040-01-01__unit__bayes_determinism"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    _setup_run_valid(run_dir)
    _setup_spectrum_input(run_dir, content=b"deterministic-spectrum")

    res_a = _run_stage(runs_root, run_id, ["--stage-name", "bayes_validation_a", "--seed", "123"])
    res_b = _run_stage(runs_root, run_id, ["--stage-name", "bayes_validation_b", "--seed", "123"])

    assert res_a.returncode == 0, res_a.stderr
    assert res_b.returncode == 0, res_b.stderr

    out_a = run_dir / "bayes_validation_a" / "outputs" / "bayes_validation.json"
    out_b = run_dir / "bayes_validation_b" / "outputs" / "bayes_validation.json"
    hash_a = hashlib.sha256(out_a.read_bytes()).hexdigest()
    hash_b = hashlib.sha256(out_b.read_bytes()).hexdigest()
    assert hash_a == hash_b


def test_c4_score_semantics_and_bf_proxy(tmp_path: Path) -> None:
    """C4 output must contain score semantics, BIC, and a BF proxy."""
    run_id = "2040-01-01__unit__bayes_semantics"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    _setup_run_valid(run_dir)
    _setup_spectrum_input(run_dir, content=b"semantics-spectrum-data-payload")

    res = _run_stage(runs_root, run_id)
    assert res.returncode == 0, res.stderr

    payload = json.loads(
        (run_dir / "bayes_validation_v1" / "outputs" / "bayes_validation.json").read_text()
    )
    c4 = payload["results"]["C4_model_selection"]

    # Score semantics present
    assert c4["score_name"] == "neg_mse"
    assert c4["score_higher_is_better"] is True
    assert isinstance(c4["delta_score"], float)
    assert c4["delta_score"] >= 0.0

    # BIC values present for each model
    assert "bic" in c4
    assert set(c4["bic"].keys()) == set(c4["model_scores"].keys())

    # BF proxy computed (2 models → must be non-null)
    assert c4["log_bayes_factor_proxy"] is not None
    assert isinstance(c4["log_bayes_factor_proxy"], float)
    assert c4["bf_proxy_method"] == "bic_schwarz"

    # selection_consistent flag present
    assert isinstance(c4["selection_consistent"], bool)


def test_verdict_pass_requires_bf_proxy(tmp_path: Path) -> None:
    """Verdict must not be PASS when log_bayes_factor_proxy is null."""
    from bayes_contracts import validate_bayes_output

    payload = {
        "schema_version": "bayes_validation_v1",
        "timestamp_utc": "1970-01-01T00:00:00+00:00",
        "parameters": {
            "seed": 42,
            "n_monte_carlo": 500,
            "k_features": 3,
            "models": ["linear", "poly2"],
            "prior_precision": 1e-6,
        },
        "inputs": [{"path": "test.h5", "sha256": "abc123"}],
        "results": {
            "C4_model_selection": {
                "best_model": "linear",
                "model_scores": {"linear": -0.01, "poly2": -0.02},
                "score_name": "neg_mse",
                "score_higher_is_better": True,
                "delta_score": 0.01,
                "log_bayes_factor_proxy": None,
                "bf_proxy_method": None,
                "selection_consistent": True,
                "posterior_means": {"linear": [1.0], "poly2": [1.0, 0.1]},
                "seed": 42,
                "n_monte_carlo": 500,
            }
        },
        "verdict": "PASS",
        "reasons": [],
    }
    ok, reasons = validate_bayes_output(payload)
    assert not ok
    assert any("C4_pass_without_bayes_factor" in r for r in reasons)


def test_contract_detects_best_model_inconsistency(tmp_path: Path) -> None:
    """Contract must flag when best_model does not match score semantics."""
    from bayes_contracts import validate_bayes_output

    payload = {
        "schema_version": "bayes_validation_v1",
        "timestamp_utc": "1970-01-01T00:00:00+00:00",
        "parameters": {
            "seed": 42,
            "n_monte_carlo": 500,
            "k_features": 3,
            "models": ["linear", "poly2"],
            "prior_precision": 1e-6,
        },
        "inputs": [{"path": "test.h5", "sha256": "abc123"}],
        "results": {
            "C4_model_selection": {
                "best_model": "poly2",
                "model_scores": {"linear": -0.01, "poly2": -0.02},
                "score_name": "neg_mse",
                "score_higher_is_better": True,
                "delta_score": 0.01,
                "log_bayes_factor_proxy": 5.0,
                "bf_proxy_method": "bic_schwarz",
                "selection_consistent": False,
                "posterior_means": {"linear": [1.0], "poly2": [1.0, 0.1]},
                "seed": 42,
                "n_monte_carlo": 500,
            }
        },
        "verdict": "INSPECT",
        "reasons": ["inconsistency"],
    }
    ok, reasons = validate_bayes_output(payload)
    assert not ok
    assert any("C4_best_model_inconsistent" in r for r in reasons)


def test_verdict_inspect_when_selection_inconsistent(tmp_path: Path) -> None:
    """Stage output must contain selection_consistent flag as a bool."""
    run_id = "2040-01-01__unit__bayes_consistency"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    _setup_run_valid(run_dir)
    _setup_spectrum_input(run_dir, content=b"consistency-check-data")

    res = _run_stage(runs_root, run_id)
    assert res.returncode == 0, res.stderr

    payload = json.loads(
        (run_dir / "bayes_validation_v1" / "outputs" / "bayes_validation.json").read_text()
    )
    c4 = payload["results"]["C4_model_selection"]
    assert isinstance(c4["selection_consistent"], bool)

    # Verdict must reflect consistency: PASS only when consistent and BF available
    if not c4["selection_consistent"]:
        assert payload["verdict"] == "INSPECT"
        assert any("BIC" in r for r in payload["reasons"])


def test_scipy_missing_killswitch(tmp_path: Path) -> None:
    run_id = "2040-01-01__unit__bayes_no_scipy"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    _setup_run_valid(run_dir)
    _setup_spectrum_input(run_dir)

    fake_pkg = tmp_path / "fakepkg"
    fake_pkg.mkdir(parents=True, exist_ok=True)
    (fake_pkg / "scipy.py").write_text("raise ImportError('simulated scipy missing')\n", encoding="utf-8")

    env = {"PYTHONPATH": f"{fake_pkg}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    res = _run_stage(runs_root, run_id, env=env)

    assert res.returncode == 2
    assert "scipy_missing" in (res.stderr + res.stdout)
