from __future__ import annotations

import json
from pathlib import Path

import pytest

from mvp.oracles.t0_input_schema import WindowSummaryV1Error, map_sweep_point_to_window_summary_v1


def _load_fixture() -> dict:
    fixture = Path("tests/fixtures/t0_sweep_full_results.min.json")
    return json.loads(fixture.read_text(encoding="utf-8"))


def test_map_sweep_point_to_window_summary_v1_ok(tmp_path: Path) -> None:
    payload = _load_fixture()
    subruns_root = tmp_path / "subruns"
    payload["subruns_root"] = str(subruns_root)

    subrun = subruns_root / "segment__t0ms0000"
    (subrun / "s2_ringdown_window" / "outputs").mkdir(parents=True, exist_ok=True)
    (subrun / "s3b_multimode_estimates" / "outputs").mkdir(parents=True, exist_ok=True)
    (subrun / "s3_ringdown_estimates" / "outputs").mkdir(parents=True, exist_ok=True)

    (subrun / "s2_ringdown_window" / "outputs" / "window_meta.json").write_text(
        json.dumps({"duration_s": 0.12}), encoding="utf-8"
    )
    (subrun / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json").write_text(
        json.dumps(
            {
                "modes": [
                    {
                        "label": "220",
                        "ln_f": 4.6,
                        "ln_Q": 2.1,
                        "Sigma": [[0.04, 0.0], [0.0, 0.09]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (subrun / "s3_ringdown_estimates" / "outputs" / "estimates.json").write_text(
        json.dumps({"combined": {"snr_peak": 11.5}, "combined_uncertainty": {"cov_logf_logQ": 0.0, "r": 0.0}}),
        encoding="utf-8",
    )

    summary = map_sweep_point_to_window_summary_v1(payload, payload["points"][0])

    assert summary["t0_ms"] == 0
    assert summary["T_s"] == 0.12
    assert summary["theta"] == {"ln_f_220": 4.6, "ln_Q_220": 2.1}
    assert summary["sigma_theta"] == {"sigma_ln_f_220": 0.2, "sigma_ln_Q_220": 0.3}
    assert summary["snr"] == 11.5


def test_map_sweep_point_to_window_summary_v1_missing_file_error(tmp_path: Path) -> None:
    payload = _load_fixture()
    subruns_root = tmp_path / "subruns"
    payload["subruns_root"] = str(subruns_root)

    subrun = subruns_root / "segment__t0ms0000"
    (subrun / "s2_ringdown_window" / "outputs").mkdir(parents=True, exist_ok=True)
    (subrun / "s2_ringdown_window" / "outputs" / "window_meta.json").write_text(json.dumps({"duration_s": 0.12}), encoding="utf-8")

    with pytest.raises(WindowSummaryV1Error) as excinfo:
        map_sweep_point_to_window_summary_v1(payload, payload["points"][0])

    msg = str(excinfo.value)
    assert "modes.220" in msg
    assert "multimode_estimates.json" in msg
