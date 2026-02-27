from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mvp import s2_ringdown_window as s2


def test_resolve_t0_gps_accepts_alias_by_canonical_prefix(tmp_path: Path) -> None:
    catalog_path = tmp_path / "window_catalog_v1.json"
    catalog_path.write_text(json.dumps({"GW190521": {"t0_gps": 1242442967.4}}), encoding="utf-8")

    t0_gps, t0_source, lookup_key = s2._resolve_t0_gps("GW190521_030229", catalog_path)

    assert t0_gps == 1242442967.4
    assert t0_source == str(catalog_path)
    assert lookup_key == "GW190521"


def test_main_persists_lookup_key_in_window_meta_for_alias(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    run_id = "alias_s2"
    run_dir = runs_root / run_id
    (run_dir / "RUN_VALID").mkdir(parents=True, exist_ok=True)
    (run_dir / "RUN_VALID" / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

    s1_outputs = run_dir / "s1_fetch_strain" / "outputs"
    s1_outputs.mkdir(parents=True, exist_ok=True)
    fs = 1024.0
    gps_start = 1242442960.0
    n_samples = 20000
    np.savez(
        s1_outputs / "strain.npz",
        H1=np.zeros(n_samples, dtype=np.float64),
        L1=np.zeros(n_samples, dtype=np.float64),
        gps_start=np.float64(gps_start),
        sample_rate_hz=np.float64(fs),
    )

    catalog_path = tmp_path / "window_catalog_v1.json"
    catalog_path.write_text(json.dumps({"GW190521": {"t0_gps": 1242442967.4}}), encoding="utf-8")

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))
    monkeypatch.setattr(
        "sys.argv",
        [
            "s2_ringdown_window.py",
            "--run-id",
            run_id,
            "--event-id",
            "GW190521_030229",
            "--window-catalog",
            str(catalog_path),
        ],
    )

    assert s2.main() == 0

    meta_path = run_dir / "s2_ringdown_window" / "outputs" / "window_meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["event_id"] == "GW190521_030229"
    assert payload["event_id_lookup_key"] == "GW190521"
    assert payload["t0_source"] == str(catalog_path)
