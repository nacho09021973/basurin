from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp import s1_fetch_strain


def _write_candidate(
    runs_root: Path,
    run_id: str,
    *,
    source: str,
    event_id: str = "GW150914",
    detectors: list[str] | None = None,
    duration_s: float = 32.0,
) -> None:
    detectors = detectors or ["H1", "L1"]
    out = runs_root / run_id / "s1_fetch_strain" / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    (out / "strain.npz").write_bytes(b"placeholder npz for selector test")
    prov = {
        "event_id": event_id,
        "detectors": detectors,
        "duration_s": duration_s,
        "source": source,
    }
    (out / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")


def test_reuse_selector_skips_synthetic_by_default(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    _write_candidate(runs_root, "run_synth", source="synthetic")
    _write_candidate(runs_root, "run_real", source="gwosc")

    current_outputs = runs_root / "run_current" / "s1_fetch_strain" / "outputs"
    current_outputs.mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(run_id="run_current", outputs_dir=current_outputs)

    monkeypatch.setattr(s1_fetch_strain, "resolve_out_root", lambda _: runs_root)
    monkeypatch.setattr(s1_fetch_strain, "_try_reuse", lambda *args, **kwargs: True)

    reused = s1_fetch_strain._try_reuse_from_other_runs(
        ctx,
        event_id="GW150914",
        detectors=["H1", "L1"],
        duration_s=32.0,
    )

    assert reused is True
    dst_prov = json.loads((current_outputs / "provenance.json").read_text(encoding="utf-8"))
    assert dst_prov["reused_from_run_id"] == "run_real"
    assert dst_prov["source"] == "gwosc"


def test_reuse_selector_can_allow_synthetic(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    _write_candidate(runs_root, "run_only_synth", source="synthetic")

    current_outputs = runs_root / "run_current" / "s1_fetch_strain" / "outputs"
    current_outputs.mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(run_id="run_current", outputs_dir=current_outputs)

    monkeypatch.setattr(s1_fetch_strain, "resolve_out_root", lambda _: runs_root)
    monkeypatch.setattr(s1_fetch_strain, "_try_reuse", lambda *args, **kwargs: True)

    reused = s1_fetch_strain._try_reuse_from_other_runs(
        ctx,
        event_id="GW150914",
        detectors=["H1", "L1"],
        duration_s=32.0,
        allow_synthetic_reuse=True,
    )

    assert reused is True
    dst_prov = json.loads((current_outputs / "provenance.json").read_text(encoding="utf-8"))
    assert dst_prov["reused_from_run_id"] == "run_only_synth"
    assert dst_prov["source"] == "synthetic"
