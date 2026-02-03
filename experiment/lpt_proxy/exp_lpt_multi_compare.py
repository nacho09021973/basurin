#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_run_valid_verdict(run_dir: Path) -> Tuple[str, Path]:
    preferred = run_dir / "RUN_VALID" / "verdict.json"
    legacy = run_dir / "RUN_VALID" / "outputs" / "run_valid.json"
    if preferred.exists():
        j = read_json(preferred)
        return str(j.get("verdict", "")).upper(), preferred
    if legacy.exists():
        j = read_json(legacy)
        return str(j.get("verdict", j.get("overall_verdict", ""))).upper(), legacy
    return "MISSING", preferred


def main() -> int:
    ap = argparse.ArgumentParser(description="BASURIN LPT multi-compare (proxy -> L_eff)")
    ap.add_argument("--run", required=True, help="run_id (must have RUN_VALID PASS)")
    ap.add_argument("--input", required=True, help="path to experiment input.json")
    ap.add_argument("--proxy-json", required=True, help="path to comparison_proxy_obs.json (proxy reference)")
    args = ap.parse_args()

    run_id = args.run
    run_dir = Path("runs") / run_id
    if not run_dir.exists():
        raise SystemExit(f"ERROR: run not found: {run_dir}")

    verdict, verdict_path = load_run_valid_verdict(run_dir)
    if verdict != "PASS":
        raise SystemExit(f"ABORT: RUN_VALID != PASS (got {verdict}) at {verdict_path}")

    in_path = Path(args.input)
    proxy_path = Path(args.proxy_json)
    if not in_path.exists():
        raise SystemExit(f"ERROR: missing input: {in_path}")
    if not proxy_path.exists():
        raise SystemExit(f"ERROR: missing proxy-json: {proxy_path}")

    inp = read_json(in_path)
    proxy = read_json(proxy_path)

    best = (proxy.get("proxy", {}) or {}).get("best_candidate", {}) or {}
    T_proxy = float(best.get("T_proxy"))
    delta_uv = float(best.get("delta_uv"))

    sources_raw = inp.get("sources", [])
    if not isinstance(sources_raw, list) or len(sources_raw) == 0:
        raise SystemExit("ERROR: input.json must contain non-empty list: sources")

    rows: List[Dict[str, Any]] = []
    for s in sorted(sources_raw, key=lambda d: (float(d["period_s"]), str(d["name"]))):
        name = str(s.get("name", "")).strip()
        period_s = float(s.get("period_s"))
        if not name:
            raise SystemExit("ERROR: source.name is required")
        if not (period_s > 0):
            raise SystemExit(f"ERROR: source.period_s must be > 0 (got {period_s}) for {name}")
        L_eff = period_s / T_proxy
        rows.append(
            {
                "name": name,
                "period_s": period_s,
                "period_min": period_s / 60.0,
                "L_eff_s": L_eff,
                "T_proxy": T_proxy,
                "delta_uv": delta_uv,
            }
        )

    out_dir = run_dir / "experiment" / "lpt_multi_compare"
    outputs_dir = out_dir / "outputs"

    created = datetime.now(timezone.utc).isoformat()
    result = {
        "schema_version": 1,
        "experiment": "lpt_multi_compare",
        "created": created,
        "run": run_id,
        "proxy_reference": {
            "path": str(proxy_path.as_posix()),
            "sha256": sha256_file(proxy_path),
            "T_proxy": T_proxy,
            "delta_uv": delta_uv,
        },
        "inputs": {
            "input_json": {"path": str(in_path.as_posix()), "sha256": sha256_file(in_path)},
            "RUN_VALID": {"path": str(verdict_path.as_posix()), "sha256": sha256_file(verdict_path)},
        },
        "rows": rows,
    }
    write_json(outputs_dir / "lpt_multi_comparison.json", result)

    manifest = {
        "schema_version": 1,
        "stage": "experiment/lpt_multi_compare",
        "run": run_id,
        "created": created,
        "inputs": result["inputs"]
        | {"proxy_json": {"path": result["proxy_reference"]["path"], "sha256": result["proxy_reference"]["sha256"]}},
        "outputs": {
            "lpt_multi_comparison.json": {
                "path": str((outputs_dir / "lpt_multi_comparison.json").relative_to(run_dir).as_posix()),
                "sha256": sha256_file(outputs_dir / "lpt_multi_comparison.json"),
            }
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    stage_summary = {
        "schema_version": 1,
        "stage": "experiment/lpt_multi_compare",
        "run": run_id,
        "created": created,
        "results": {
            "overall_verdict": "PASS",
            "n_sources": len(rows),
            "T_proxy": T_proxy,
            "delta_uv": delta_uv,
        },
    }
    write_json(out_dir / "stage_summary.json", stage_summary)

    print("PASS: wrote", (outputs_dir / "lpt_multi_comparison.json").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
