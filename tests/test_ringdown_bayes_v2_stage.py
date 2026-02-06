from __future__ import annotations

import json
import math
from pathlib import Path

from basurin_io import sha256_file
from stages.ringdown_bayes_v2_stage import main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_inputs(run_dir: Path, t: list[float], y: list[float]) -> None:
    _write_json(run_dir / "RUN_VALID" / "outputs" / "run_valid.json", {"overall_verdict": "PASS"})
    inference_report = {
        "schema_version": "test",
        "created_utc": "2030-01-01T00:00:00+00:00",
        "whitened": True,
        "per_detector": {
            "H1": {
                "status": "OK",
                "window": {
                    "t": t,
                    "y": y,
                },
            }
        },
    }
    _write_json(
        run_dir / "ringdown_real_inference_v0" / "outputs" / "inference_report.json",
        inference_report,
    )


def test_contract_pass_requires_finite_bf(tmp_path: Path) -> None:
    run_id = "2042-01-01__ringdown_bayes_v2_contract"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    t = [i * 0.0005 for i in range(128)]
    y = [0.3 * math.exp(-ti / 0.03) * math.cos(2.0 * math.pi * 120.0 * ti + 0.2) for ti in t]
    _write_inputs(run_dir, t, y)

    rc = main(["--run", run_id, "--out-root", str(runs_root)])
    assert rc == 0

    out_path = run_dir / "ringdown_bayes_v2" / "outputs" / "bayes_v2.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["verdict"] in {"PASS", "INSPECT"}
    assert math.isfinite(payload["results"]["global"]["logBF_10"])
    assert payload["inputs"]
    assert all("sha256" in item and item["sha256"] for item in payload["inputs"])


def test_determinism_same_input_same_sha256(tmp_path: Path) -> None:
    run_id = "2042-01-01__ringdown_bayes_v2_determinism"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    t = [i * 0.0005 for i in range(128)]
    y = [0.25 * math.exp(-ti / 0.02) * math.cos(2.0 * math.pi * 140.0 * ti + 0.4) for ti in t]
    _write_inputs(run_dir, t, y)

    rc1 = main(["--run", run_id, "--out-root", str(runs_root)])
    assert rc1 == 0
    out_path = run_dir / "ringdown_bayes_v2" / "outputs" / "bayes_v2.json"
    h1 = sha256_file(out_path)

    rc2 = main(["--run", run_id, "--out-root", str(runs_root)])
    assert rc2 == 0
    h2 = sha256_file(out_path)

    assert h1 == h2


def test_sanity_no_signal_bf_near_zero_or_negative(tmp_path: Path) -> None:
    run_id = "2042-01-01__ringdown_bayes_v2_no_signal"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    t = [i * 0.0005 for i in range(128)]
    # deterministic small pseudo-noise pattern around zero (no random)
    pattern = [0.0, 0.002, -0.0015, 0.001, -0.0005, 0.0015, -0.002, 0.0005]
    y = [pattern[i % len(pattern)] for i in range(len(t))]
    _write_inputs(run_dir, t, y)

    rc = main(["--run", run_id, "--out-root", str(runs_root)])
    assert rc == 0

    payload = json.loads(
        (run_dir / "ringdown_bayes_v2" / "outputs" / "bayes_v2.json").read_text(encoding="utf-8")
    )
    logbf = payload["results"]["global"]["logBF_10"]
    assert math.isfinite(logbf)
    assert logbf < 1.0
