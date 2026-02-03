#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# --- BASURIN import bootstrap (no depende de PYTHONPATH) ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[1], _here.parents[2], _here.parents[3]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break
# -----------------------------------------------------------

import argparse
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from basurin_io import (
    ensure_stage_dirs,
    resolve_out_root,
    require_run_valid,
    validate_run_id,
    write_manifest,
    write_stage_summary,
    sha256_file,
)


EXIT_CONTRACT_FAIL = 2
STAGE_NAME_DEFAULT = "experiment/ringdown/STAGE_RINGDOWN_OBSERVABLES_V1"
CANDIDATES = (
    "obs_Q_hat",
    "obs_log_tau_hat",
    "obs_log_f_hat",
    "obs_f_tau_hat",
)


def abort_contract(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _resolve_exp01_cases(run_dir: Path) -> Path:
    preferred = (
        run_dir
        / "experiment"
        / "ringdown"
        / "EXP_RINGDOWN_01_injection_recovery"
        / "outputs"
        / "recovery_cases.jsonl"
    )
    legacy = (
        run_dir
        / "experiment"
        / "ringdown_01_injection_recovery"
        / "outputs"
        / "recovery_cases.jsonl"
    )
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    abort_contract(f"recovery_cases.jsonl missing at {preferred} (legacy {legacy})")


def _compute_observables(row: Dict[str, Any]) -> Optional[Dict[str, float]]:
    estimate = row.get("estimate") or {}
    f_hat = estimate.get("f_220_hat")
    tau_hat = estimate.get("tau_220_hat")
    q_hat = estimate.get("Q_220_hat")

    if f_hat is None or tau_hat is None:
        return None

    f_hat = float(f_hat)
    tau_hat = float(tau_hat)
    if not np.isfinite(f_hat) or not np.isfinite(tau_hat):
        return None
    if f_hat <= 0 or tau_hat <= 0:
        return None

    if q_hat is None:
        q_hat = float(np.pi * f_hat * tau_hat)
    else:
        q_hat = float(q_hat)

    if not np.isfinite(q_hat) or q_hat <= 0:
        return None

    obs = {
        "obs_Q_hat": q_hat,
        "obs_log_tau_hat": float(np.log(tau_hat)),
        "obs_log_f_hat": float(np.log(f_hat)),
        "obs_f_tau_hat": float(f_hat * tau_hat),
    }
    if any((not np.isfinite(v)) for v in obs.values()):
        return None
    return obs


def main() -> int:
    ap = argparse.ArgumentParser(description="STAGE_RINGDOWN_OBSERVABLES_V1 (deterministic, contract-first)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage-name", default=STAGE_NAME_DEFAULT)
    ap.add_argument(
        "--cases-jsonl",
        default=None,
        help="optional path under runs/<run_id>/... to recovery_cases.jsonl",
    )
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    validate_run_id(args.run, out_root)
    run_dir = (out_root / args.run).resolve()

    try:
        require_run_valid(out_root, args.run)
    except Exception as exc:
        abort_contract(str(exc))

    if args.cases_jsonl:
        cases_path = Path(args.cases_jsonl)
        if not cases_path.is_absolute():
            cases_path = (Path.cwd() / cases_path).resolve()
        try:
            cases_path.relative_to(run_dir)
        except ValueError:
            abort_contract(f"cases-jsonl must live under {run_dir}, got {cases_path}")
        if not cases_path.exists():
            abort_contract(f"cases-jsonl not found at {cases_path}")
    else:
        cases_path = _resolve_exp01_cases(run_dir)

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, args.stage_name, base_dir=out_root)

    observables_path = outputs_dir / "observables.jsonl"
    contract_path = outputs_dir / "contract_verdict.json"

    n_total = 0
    n_kept = 0
    n_skipped = 0

    with open(observables_path, "w", encoding="utf-8") as out:
        for row in _read_jsonl(cases_path):
            n_total += 1
            if str(row.get("status", "")).upper() != "OK":
                n_skipped += 1
                continue
            case_id = str(row.get("case_id") or "")
            obs = _compute_observables(row)
            if obs is None:
                n_skipped += 1
                continue
            n_kept += 1
            out.write(json.dumps({"case_id": case_id, "observables": obs}) + "\n")

    min_cases = 1
    verdict = "PASS" if n_kept >= min_cases else "FAIL"

    inputs = {
        "recovery_cases": {
            "path": str(cases_path.relative_to(run_dir)),
            "sha256": sha256_file(cases_path),
        }
    }

    contracts = {
        "OBS_V1_MIN_CASES": {
            "verdict": verdict,
            "min_cases": min_cases,
            "n_cases": n_kept,
        }
    }

    contract_payload = {
        "verdict": verdict,
        "contracts": contracts,
        "violations": [] if verdict == "PASS" else ["OBS_V1_MIN_CASES"],
        "inputs": inputs,
        "outputs": {
            "observables": "outputs/observables.jsonl",
            "contract_verdict": "outputs/contract_verdict.json",
        },
    }

    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    stage_summary = {
        "stage": stage_dir.name,
        "run": args.run,
        "inputs": inputs,
        "parameters": {
            "candidates": list(CANDIDATES),
        },
        "results": {
            "overall_verdict": verdict,
            "n_total": n_total,
            "n_kept": n_kept,
            "n_skipped": n_skipped,
        },
    }
    summary_written = write_stage_summary(stage_dir, stage_summary)
    write_manifest(
        stage_dir,
        {
            "observables": observables_path,
            "contract": contract_path,
            "stage_summary": summary_written,
        },
        extra={"version": "1"},
    )

    if verdict != "PASS":
        abort_contract("STAGE_RINGDOWN_OBSERVABLES_V1 FAIL: insufficient observables")

    print("OK: STAGE_RINGDOWN_OBSERVABLES_V1 PASS")
    print(f"  outputs: {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
