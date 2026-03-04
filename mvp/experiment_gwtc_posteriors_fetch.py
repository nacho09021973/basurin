#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import sha256_file
from mvp import contracts

STAGE = "experiment_gwtc_posteriors_fetch"


def _read_events(path: Path) -> list[str]:
    items: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ev = line.strip()
        if not ev or ev.startswith("#"):
            continue
        items.append(ev)
    if not items:
        raise ValueError(f"events list is empty: {path}")
    return items


def _default_events_file(run_dir: Path) -> Path:
    base = run_dir / "external_inputs" / "gwtc_posteriors"
    pilot = base / "pilot_events.txt"
    required = base / "required_events.txt"
    return pilot if pilot.exists() else required


def _parse_and_validate(path: Path, event_id: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in posterior JSON: {path}")
    if data.get("event_id") != event_id:
        raise ValueError(f"event_id mismatch in {path}: expected={event_id!r} got={data.get('event_id')!r}")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"samples must be a non-empty list in {path}")
    required = ("m1_source", "m2_source", "chi1", "chi2")
    first = samples[0]
    if not isinstance(first, dict) or any(k not in first for k in required):
        raise ValueError(f"samples[0] missing required keys {required} in {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate/download GWTC IMR posterior samples under external_inputs")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--events-file", default=None)
    ap.add_argument("--source", choices=["manual", "gwtc_public"], default="manual")
    ap.add_argument("--format", choices=["json", "hdf5"], default="json")
    args = ap.parse_args()

    ctx = contracts.init_stage(
        args.run_id,
        STAGE,
        params={"events_file": args.events_file, "source": args.source, "format": args.format},
    )

    external_dir = ctx.run_dir / "external_inputs" / "gwtc_posteriors"
    events_file = Path(args.events_file) if args.events_file else _default_events_file(ctx.run_dir)

    if not events_file.exists():
        contracts.abort(
            ctx,
            (
                f"Missing events file. expected={events_file}; "
                f"regen_cmd='printf ""GW150914\n"" > {external_dir / 'pilot_events.txt'}'; "
                f"candidates={sorted(str(p) for p in external_dir.glob('*events*.txt'))}"
            ),
        )

    events = _read_events(events_file)

    if args.format != "json":
        contracts.abort(ctx, "Only --format=json is supported in deterministic contract-first mode")

    if args.source == "gwtc_public":
        contracts.abort(
            ctx,
            "--source=gwtc_public not implemented in this environment; use --source=manual with canonical JSON files",
        )

    missing: list[str] = []
    input_map: dict[str, Path] = {"events_file": events_file}
    for event_id in events:
        posterior_path = external_dir / f"{event_id}.json"
        input_map[f"posterior_{event_id}"] = posterior_path
        if not posterior_path.exists():
            missing.append(event_id)

    if missing:
        missing_paths = [str(external_dir / f"{ev}.json") for ev in missing]
        candidates = sorted(str(p.name) for p in external_dir.glob("*.json"))
        contracts.abort(
            ctx,
            (
                f"Missing required IMR posterior JSON. expected_paths={missing_paths}; "
                f"regen_cmd='python -m mvp.experiment_gwtc_posteriors_fetch --run-id {args.run_id} --source manual --format json'; "
                f"missing_events={missing}; candidates={candidates}"
            ),
        )

    contracts.check_inputs(ctx, input_map)

    for event_id in events:
        _parse_and_validate(external_dir / f"{event_id}.json", event_id)

    out_path = ctx.outputs_dir / "validated_posteriors.json"
    payload = {
        "run_id": args.run_id,
        "events_file": str(events_file),
        "source": args.source,
        "format": args.format,
        "events": events,
        "posterior_hashes": {
            ev: sha256_file(external_dir / f"{ev}.json")
            for ev in events
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    contracts.finalize(
        ctx,
        artifacts={"validated_posteriors": out_path},
        results={"event_count": len(events)},
    )
    contracts.log_stage_paths(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
