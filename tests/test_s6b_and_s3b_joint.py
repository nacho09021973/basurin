"""Joint regression tests: model_comparison (ΔBIC) canonical output for s3b.

Sections
--------
1. model_comparison_v1 schema validation (JSON strict, schema_version, field types)
2. ΔBIC decreases on a clean synthetic 2-mode signal  →  two_mode_preferred=True
3. Pure noise: no crash, decision deterministic, valid_2mode may be True or False
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    import unittest
    raise unittest.SkipTest("integration requires numpy")

# ---------------------------------------------------------------------------
# Load s3b module (same pattern as test_s3b_multimode_estimates.py)
# ---------------------------------------------------------------------------
_MODULE_PATH = Path(__file__).resolve().parents[1] / "mvp" / "s3b_multimode_estimates.py"
_SPEC = importlib.util.spec_from_file_location("mvp_s3b_multimode_estimates_joint", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

compute_model_comparison = _MODULE.compute_model_comparison

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_two_mode_signal(
    *,
    fs: float = 4096.0,
    duration: float = 0.3,
    f220: float = 220.0,
    tau220: float = 0.08,
    f221: float = 332.0,
    tau221: float = 0.05,
    amp220: float = 1.0,
    amp221: float = 0.6,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Noiseless 2-mode exponentially-decaying sinusoid (deterministic)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs), dtype=float) / fs
    signal = (
        amp220 * np.exp(-t / tau220) * np.cos(2.0 * math.pi * f220 * t)
        + amp221 * np.exp(-t / tau221) * np.cos(2.0 * math.pi * f221 * t + 0.3)
    )
    # tiny noise so numerical conditioning stays finite
    signal = signal + rng.normal(0.0, 1e-6, signal.size)
    return signal, fs


def _make_mode_dict(label: str, f_hz: float, q: float) -> dict:
    return {
        "label": label,
        "ln_f": math.log(f_hz),
        "ln_Q": math.log(q),
    }


def _write_minimal_s2_inputs(run_root: Path, run_id: str) -> None:
    """Replicates the helper from test_s3b_multimode_estimates.py."""
    run_dir = run_root / run_id
    rv_dir = run_dir / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    (rv_dir / "verdict.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")

    s2_outputs = run_dir / "s2_ringdown_window" / "outputs"
    s2_outputs.mkdir(parents=True, exist_ok=True)
    signal, fs = _make_two_mode_signal()
    np.savez(s2_outputs / "H1_rd.npz", strain=signal, sample_rate_hz=np.array([fs]))

    (run_dir / "s2_ringdown_window" / "manifest.json").write_text(
        json.dumps({"artifacts": {"H1_rd": "s2_ringdown_window/outputs/H1_rd.npz"}}),
        encoding="utf-8",
    )

    s3_outputs = run_dir / "s3_ringdown_estimates" / "outputs"
    s3_outputs.mkdir(parents=True, exist_ok=True)
    (s3_outputs / "estimates.json").write_text(
        json.dumps({
            "schema_version": "ringdown_estimates_v1",
            "results": {
                "best_detector": "H1",
                "bandpass_hz": [150.0, 400.0],
                "detectors": {"H1": {"f0_hz": 220.0, "tau_s": 0.04}},
            },
        }),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Section 1 – schema validation (model_comparison_v1, JSON strict)
# ---------------------------------------------------------------------------

def test_model_comparison_schema_version_and_required_fields() -> None:
    """Unit: compute_model_comparison returns model_comparison_v1 with all required fields."""
    signal, fs = _make_two_mode_signal()
    mode_220 = _make_mode_dict("220", 220.0, 12.0)
    mode_221 = _make_mode_dict("221", 332.0, 8.0)

    result = compute_model_comparison(signal, fs, mode_220, mode_221, ok_220=True, ok_221=True)

    assert result["schema_version"] == "model_comparison_v1"
    assert isinstance(result["n_samples"], int) and result["n_samples"] > 0
    assert result["k_1mode"] == 4
    assert result["k_2mode"] == 8
    assert isinstance(result["thresholds"], dict)
    assert "two_mode_preferred_delta_bic" in result["thresholds"]
    assert isinstance(result["decision"], dict)
    assert "two_mode_preferred" in result["decision"]
    assert isinstance(result["decision"]["two_mode_preferred"], bool)
    assert isinstance(result["valid_1mode"], bool)
    assert isinstance(result["valid_2mode"], bool)
    assert "trace" in result
    assert "conventions" in result
    conventions = result["conventions"]
    assert conventions["delta_bic_definition"] == "bic_2mode - bic_1mode"
    assert conventions["bic_formula"] == "k*ln(n) + n*ln(rss/n)"
    assert conventions["tie_break_rule"] == "|ΔBIC|<eps => ΔBIC=0.0 and prefer 1-mode"
    assert conventions["delta_bic_tie_eps"] == 1e-10
    assert "design_matrix_columns" in conventions
    assert conventions["design_matrix_columns"] == ["220_cos", "220_sin", "221_cos", "221_sin"]
    assert "k_definition" in conventions
    assert conventions["k_definition"]["k_1mode"] == 4
    assert conventions["k_definition"]["k_2mode"] == 8
    assert isinstance(result["trace"]["rss_floored_1mode"], bool)
    assert isinstance(result["trace"]["rss_floored_2mode"], bool)


def test_model_comparison_delta_bic_is_finite_on_valid_two_mode() -> None:
    """Unit: delta_bic is finite (not NaN/Inf) for a valid 2-mode signal."""
    signal, fs = _make_two_mode_signal()
    mode_220 = _make_mode_dict("220", 220.0, 12.0)
    mode_221 = _make_mode_dict("221", 332.0, 8.0)

    result = compute_model_comparison(signal, fs, mode_220, mode_221, ok_220=True, ok_221=True)

    assert result["valid_1mode"] is True
    assert result["valid_2mode"] is True
    assert result["delta_bic"] is not None
    assert math.isfinite(result["delta_bic"])
    assert math.isfinite(result["bic_1mode"])
    assert math.isfinite(result["bic_2mode"])
    assert math.isfinite(result["rss_1mode"])
    assert math.isfinite(result["rss_2mode"])
    assert isinstance(result["decision"]["two_mode_preferred"], bool)


def test_model_comparison_json_serializable() -> None:
    """Unit: result is JSON-serializable (strict) with no NaN/Inf leaking in."""
    signal, fs = _make_two_mode_signal()
    mode_220 = _make_mode_dict("220", 220.0, 12.0)
    mode_221 = _make_mode_dict("221", 332.0, 8.0)

    result = compute_model_comparison(signal, fs, mode_220, mode_221, ok_220=True, ok_221=True)
    serialized = json.dumps(result)  # must not raise
    reloaded = json.loads(serialized)
    assert reloaded["schema_version"] == "model_comparison_v1"


# ---------------------------------------------------------------------------
# Section 2 – ΔBIC prefers 2 modes on a clean bimodal signal
# ---------------------------------------------------------------------------

def test_delta_bic_prefers_two_modes_on_clean_bimodal_signal() -> None:
    """Unit: on a genuine 2-mode ringdown, delta_bic < threshold → two_mode_preferred=True."""
    signal, fs = _make_two_mode_signal(
        f220=220.0, tau220=0.08, f221=332.0, tau221=0.05, amp220=1.0, amp221=0.6,
    )
    # Point estimates close to the true frequencies used to generate the signal
    mode_220 = _make_mode_dict("220", 220.0, math.pi * 220.0 * 0.08)
    mode_221 = _make_mode_dict("221", 332.0, math.pi * 332.0 * 0.05)

    result = compute_model_comparison(signal, fs, mode_220, mode_221, ok_220=True, ok_221=True)

    assert result["valid_1mode"] is True
    assert result["valid_2mode"] is True
    assert result["delta_bic"] is not None
    # 2-mode fit should dramatically reduce RSS → delta_bic << -10
    threshold = result["thresholds"]["two_mode_preferred_delta_bic"]
    assert result["delta_bic"] < threshold, (
        f"Expected delta_bic < {threshold}, got {result['delta_bic']:.2f}. "
        "2-mode signal should strongly prefer 2-mode model."
    )
    assert result["decision"]["two_mode_preferred"] is True


def test_rss_2mode_le_rss_1mode_on_bimodal_signal() -> None:
    """Unit: adding mode 221 columns always reduces or maintains RSS (lstsq is optimal)."""
    signal, fs = _make_two_mode_signal()
    mode_220 = _make_mode_dict("220", 220.0, math.pi * 220.0 * 0.08)
    mode_221 = _make_mode_dict("221", 332.0, math.pi * 332.0 * 0.05)

    result = compute_model_comparison(signal, fs, mode_220, mode_221, ok_220=True, ok_221=True)

    assert result["rss_2mode"] is not None
    assert result["rss_1mode"] is not None
    assert result["rss_2mode"] <= result["rss_1mode"] + 1e-10  # lstsq guarantee


# ---------------------------------------------------------------------------
# Section 3 – Pure noise: no crash, deterministic
# ---------------------------------------------------------------------------

def test_model_comparison_pure_noise_no_crash() -> None:
    """Unit: on pure Gaussian noise, compute_model_comparison does not raise."""
    rng = np.random.default_rng(7)
    signal = rng.standard_normal(2048)
    fs = 4096.0
    mode_220 = _make_mode_dict("220", 200.0, 10.0)
    mode_221 = _make_mode_dict("221", 320.0, 7.0)

    result = compute_model_comparison(signal, fs, mode_220, mode_221, ok_220=True, ok_221=True)

    assert result["schema_version"] == "model_comparison_v1"
    assert isinstance(result["decision"]["two_mode_preferred"], bool)
    # delta_bic may be positive (BIC penalizes extra params on noise)
    if result["delta_bic"] is not None:
        assert math.isfinite(result["delta_bic"])


def test_model_comparison_pure_noise_is_deterministic() -> None:
    """Unit: same noise seed → identical delta_bic (reproducibility contract)."""
    rng = np.random.default_rng(99)
    signal = rng.standard_normal(2048)
    fs = 4096.0
    mode_220 = _make_mode_dict("220", 200.0, 10.0)
    mode_221 = _make_mode_dict("221", 320.0, 7.0)

    result_a = compute_model_comparison(signal, fs, mode_220, mode_221, ok_220=True, ok_221=True)
    result_b = compute_model_comparison(signal, fs, mode_220, mode_221, ok_220=True, ok_221=True)

    assert result_a["delta_bic"] == result_b["delta_bic"]
    assert result_a["decision"]["two_mode_preferred"] == result_b["decision"]["two_mode_preferred"]
    assert result_a["rss_1mode"] == result_b["rss_1mode"]
    assert result_a["rss_2mode"] == result_b["rss_2mode"]


def test_model_comparison_invalid_221_returns_null_delta_bic() -> None:
    """Unit: when ok_221=False, delta_bic=None and valid_2mode=False (no crash)."""
    signal, fs = _make_two_mode_signal()
    mode_220 = _make_mode_dict("220", 220.0, 12.0)
    mode_221 = {"label": "221", "ln_f": None, "ln_Q": None}  # invalid 221

    result = compute_model_comparison(signal, fs, mode_220, mode_221, ok_220=True, ok_221=False)

    assert result["valid_2mode"] is False
    assert result["delta_bic"] is None
    assert result["bic_2mode"] is None
    assert result["decision"]["two_mode_preferred"] is None
    assert result["schema_version"] == "model_comparison_v1"
    assert isinstance(result["trace"]["rss_floored_1mode"], bool)
    assert isinstance(result["trace"]["rss_floored_2mode"], bool)


def test_model_comparison_invalid_220_returns_all_null() -> None:
    """Unit: when ok_220=False, both valid flags are False and all BIC fields are None."""
    signal, fs = _make_two_mode_signal()
    mode_220 = {"label": "220", "ln_f": None, "ln_Q": None}
    mode_221 = _make_mode_dict("221", 332.0, 8.0)

    result = compute_model_comparison(signal, fs, mode_220, mode_221, ok_220=False, ok_221=True)

    assert result["valid_1mode"] is False
    assert result["valid_2mode"] is False
    assert result["delta_bic"] is None
    assert result["bic_1mode"] is None
    assert result["decision"]["two_mode_preferred"] is None
    assert isinstance(result["trace"]["rss_floored_1mode"], bool)
    assert isinstance(result["trace"]["rss_floored_2mode"], bool)


# ---------------------------------------------------------------------------
# Integration – CLI writes model_comparison.json (schema end-to-end)
# ---------------------------------------------------------------------------

def test_cli_writes_model_comparison_json(tmp_path: Path) -> None:
    """Integration: running s3b via CLI produces model_comparison.json with valid schema."""
    run_id = "joint_test_cli_001"
    tmp_runs = tmp_path / "runs_root"
    _write_minimal_s2_inputs(tmp_runs, run_id)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "mvp" / "s3b_multimode_estimates.py"),
        "--runs-root", str(tmp_runs),
        "--run-id", run_id,
        "--n-bootstrap", "8",
        "--seed", "42",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(tmp_runs)}
    env.pop("BASURIN_RUNS_ROOT", None)

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr

    comparison_path = (
        tmp_runs / run_id / "s3b_multimode_estimates" / "outputs" / "model_comparison.json"
    )
    assert comparison_path.exists(), "model_comparison.json must be written by CLI"

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "model_comparison_v1"
    assert isinstance(payload["n_samples"], int)
    assert payload["k_1mode"] == 4
    assert payload["k_2mode"] == 8
    if payload["delta_bic"] is None:
        assert payload["decision"]["two_mode_preferred"] is None
    else:
        assert isinstance(payload["decision"]["two_mode_preferred"], bool)
    assert isinstance(payload["valid_1mode"], bool)
    assert isinstance(payload["valid_2mode"], bool)
    if payload["delta_bic"] is not None:
        assert math.isfinite(payload["delta_bic"])


def test_cli_model_comparison_in_manifest(tmp_path: Path) -> None:
    """Integration: model_comparison.json SHA256 appears in manifest.json."""
    run_id = "joint_test_cli_002"
    tmp_runs = tmp_path / "runs_root"
    _write_minimal_s2_inputs(tmp_runs, run_id)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "mvp" / "s3b_multimode_estimates.py"),
        "--runs-root", str(tmp_runs),
        "--run-id", run_id,
        "--n-bootstrap", "8",
        "--seed", "43",
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    stage_dir = tmp_runs / run_id / "s3b_multimode_estimates"
    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))

    # manifest.artifacts must include both outputs
    artifacts = manifest.get("artifacts", {})
    artifact_paths = list(artifacts.values())
    assert any("model_comparison.json" in str(p) for p in artifact_paths), (
        f"model_comparison.json not found in manifest artifacts: {artifact_paths}"
    )
    assert any("multimode_estimates.json" in str(p) for p in artifact_paths)


def test_cli_multimode_estimates_has_quality_gates(tmp_path: Path) -> None:
    """Integration: multimode_estimates.json includes results.quality_gates.two_mode_preferred."""
    run_id = "joint_test_cli_003"
    tmp_runs = tmp_path / "runs_root"
    _write_minimal_s2_inputs(tmp_runs, run_id)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "mvp" / "s3b_multimode_estimates.py"),
        "--runs-root", str(tmp_runs),
        "--run-id", run_id,
        "--n-bootstrap", "8",
        "--seed", "44",
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    out_path = (
        tmp_runs / run_id / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "quality_gates" in payload["results"], (
        "results.quality_gates must be present in multimode_estimates.json"
    )
    assert "two_mode_preferred" in payload["results"]["quality_gates"]
    assert isinstance(payload["results"]["quality_gates"]["two_mode_preferred"], bool)


# ---------------------------------------------------------------------------
# Section 4 – Numerical edge cases (audit guards)
# ---------------------------------------------------------------------------

def test_model_comparison_rss_floor_finite_when_perfect_fit() -> None:
    """Unit: when signal is exactly the 1-mode template (rss_1mode→0), BIC stays finite.

    Verifies that the _RSS_FLOOR clamp prevents -inf from entering bic_1mode / delta_bic,
    and that the output is JSON-serialisable with no inf/nan leaking through.
    """
    fs = 4096.0
    t = np.arange(1229, dtype=float) / fs  # ~0.3 s at 4096 Hz
    f220, q220 = 220.0, 12.0
    tau220 = q220 / (math.pi * f220)
    # Signal that is *exactly* a single-mode template (rss_1mode will be machine-epsilon small)
    signal = np.exp(-t / tau220) * np.cos(2.0 * math.pi * f220 * t)

    mode_220 = _make_mode_dict("220", f220, q220)
    mode_221 = _make_mode_dict("221", 332.0, 8.0)

    result = compute_model_comparison(signal, fs, mode_220, mode_221, ok_220=True, ok_221=True)

    assert result["schema_version"] == "model_comparison_v1"
    assert result["rss_1mode"] is not None and result["rss_1mode"] >= 0.0

    if result["valid_bic_1mode"]:
        assert result["bic_1mode"] is not None
        assert math.isfinite(result["bic_1mode"]), (
            f"bic_1mode must be finite even when rss_1mode≈0, got {result['bic_1mode']}"
        )
    if result["delta_bic"] is not None:
        assert math.isfinite(result["delta_bic"]), (
            f"delta_bic must be finite even when rss_1mode≈0, got {result['delta_bic']}"
        )

    # No inf/nan must escape into JSON
    serialized = json.dumps(result, allow_nan=False)  # must not raise
    reloaded = json.loads(serialized)
    assert reloaded["schema_version"] == "model_comparison_v1"
    # conventions block must be present
    assert "conventions" in reloaded
    assert reloaded["conventions"]["delta_bic_definition"] == "bic_2mode - bic_1mode"
    assert reloaded["conventions"]["bic_formula"] == "k*ln(n) + n*ln(rss/n)"
    assert reloaded["conventions"]["tie_break_rule"] == "|ΔBIC|<eps => ΔBIC=0.0 and prefer 1-mode"
    assert isinstance(reloaded["trace"]["rss_floored_1mode"], bool)
    assert isinstance(reloaded["trace"]["rss_floored_2mode"], bool)


def test_model_comparison_invalid_when_n_too_small_for_k() -> None:
    """Unit: when n <= k_2mode + _N_MIN_BIC_MARGIN, valid_bic_2mode=False and delta_bic=None.

    k_2mode=8, _N_MIN_BIC_MARGIN=2 → need n > 10; we use n=9 (too small for 2-mode BIC).
    The function must not crash and must return a well-formed, JSON-serialisable dict.
    """
    fs = 4096.0
    n = 9  # k_2mode + margin = 10; n=9 is below that threshold
    signal = np.random.default_rng(0).standard_normal(n)
    mode_220 = _make_mode_dict("220", 220.0, 12.0)
    mode_221 = _make_mode_dict("221", 332.0, 8.0)

    result = compute_model_comparison(signal, fs, mode_220, mode_221, ok_220=True, ok_221=True)

    assert result["schema_version"] == "model_comparison_v1"
    # 2-mode BIC is not interpretable → delta_bic must be null
    assert result["delta_bic"] is None, (
        f"delta_bic must be None when n={n} <= k_2mode + margin, got {result['delta_bic']}"
    )
    assert result["valid_bic_2mode"] is False
    assert result["decision"]["two_mode_preferred"] is None

    # JSON-serialisable even in this degenerate case
    serialized = json.dumps(result, allow_nan=False)  # must not raise
    reloaded = json.loads(serialized)
    assert reloaded["schema_version"] == "model_comparison_v1"
    assert reloaded["valid_bic_2mode"] is False
    assert isinstance(reloaded["trace"]["rss_floored_1mode"], bool)
    assert isinstance(reloaded["trace"]["rss_floored_2mode"], bool)
