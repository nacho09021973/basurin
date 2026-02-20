from __future__ import annotations

import json
from pathlib import Path

import pytest
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.oracles.oracle_v1_plateau import (
    FAIL_MISSING_FIELD,
    FAIL_SIGMA_MISSING,
    SIGMA_FLOOR,
    WindowMetricsParseError,
    compute_sha256,
    load_window_metrics_from_subrun,
    load_window_metrics_from_subruns,
    sigma_from_iqr,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def test_parses_canonical_subrun_outputs(tmp_path: Path) -> None:
    subrun = tmp_path / "run__t0ms0002"
    _write(
        subrun / "s2_ringdown_window" / "outputs" / "window_meta.json",
        {"duration_s": 0.1, "n_samples": 512, "t0_offset_ms": 2},
    )
    _write(
        subrun / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        {
            "combined": {"f_hz": 251.0, "tau_s": 0.004},
            "combined_uncertainty": {"sigma_f_hz": 1e-20, "sigma_tau_s": 1e-6},
        },
    )
    _write(
        subrun / "s4c_kerr_consistency" / "outputs" / "kerr_consistency.json",
        {"cond_number": 10.0, "delta_bic": 4.2, "p_ljungbox": 0.3, "chi2_coh": 1.2},
    )

    metrics = load_window_metrics_from_subrun(subrun)

    assert metrics.t0 == 2.0
    assert metrics.f_sigma == SIGMA_FLOOR
    assert metrics.tau_sigma == 1e-6
    assert metrics.chi2_coh == 1.2
    assert metrics.valid_fraction is None


def test_missing_sigma_raises_typed_error(tmp_path: Path) -> None:
    subrun = tmp_path / "run__t0ms0000"
    _write(subrun / "s2_ringdown_window" / "outputs" / "window_meta.json", {"duration_s": 0.1, "n_samples": 256, "t0_offset_ms": 0})
    _write(
        subrun / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        {"combined": {"f_hz": 250.0, "tau_s": 0.004}},
    )
    _write(
        subrun / "s4c_kerr_consistency" / "outputs" / "kerr_consistency.json",
        {"cond_number": 7.0, "delta_bic": 3.0, "p_ljungbox": 0.5},
    )

    with pytest.raises(WindowMetricsParseError) as exc:
        load_window_metrics_from_subrun(subrun)

    assert exc.value.code == FAIL_SIGMA_MISSING


def test_subruns_are_sorted_by_t0_and_hash_is_stable(tmp_path: Path) -> None:
    subrun_a = tmp_path / "run__t0ms0004"
    subrun_b = tmp_path / "run__t0ms0001"

    for root, t0 in ((subrun_a, 4), (subrun_b, 1)):
        _write(root / "s2_ringdown_window" / "outputs" / "window_meta.json", {"duration_s": 0.1, "n_samples": 128, "t0_offset_ms": t0})
        _write(
            root / "s3_ringdown_estimates" / "outputs" / "estimates.json",
            {
                "combined": {"f_hz": 250.0, "tau_s": 0.004},
                "combined_uncertainty": {"sigma_f_hz": 0.3, "sigma_tau_s": 0.001},
            },
        )
        _write(root / "s4c_kerr_consistency" / "outputs" / "kerr_consistency.json", {"cond_number": 1.0, "delta_bic": 2.0, "p_ljungbox": 0.4})

    out = load_window_metrics_from_subruns([subrun_a, subrun_b])
    assert [item.t0 for item in out] == [1.0, 4.0]

    h = compute_sha256(subrun_a / "s3_ringdown_estimates" / "outputs" / "estimates.json")
    assert len(h) == 64
    assert sigma_from_iqr(1.349) == pytest.approx(1.0)


def test_missing_required_field_raises_typed_error(tmp_path: Path) -> None:
    subrun = tmp_path / "run__t0ms0000"
    _write(subrun / "s3_ringdown_estimates" / "outputs" / "estimates.json", {})

    with pytest.raises(WindowMetricsParseError) as exc:
        load_window_metrics_from_subrun(subrun)

    assert exc.value.code == FAIL_MISSING_FIELD
