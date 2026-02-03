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
STAGE_NAME = "experiment/ringdown/EXP_RINGDOWN_03__observable_minimality"
OBS_STAGE_NAME = "experiment/ringdown/STAGE_RINGDOWN_OBSERVABLES_V1"
METRIC_NAME = "effective_rank"
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


def _resolve_exp02_cases(run_dir: Path) -> Path:
    preferred = (
        run_dir
        / "experiment"
        / "ringdown"
        / "EXP_RINGDOWN_02_recovery_robustness"
        / "outputs"
        / "per_case.jsonl"
    )
    legacy = (
        run_dir
        / "experiment"
        / "ringdown_02_recovery_robustness"
        / "outputs"
        / "per_case.jsonl"
    )
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    abort_contract(f"per_case.jsonl missing at {preferred} (legacy {legacy})")


def _resolve_observables(run_dir: Path, obs_arg: Optional[str]) -> Path:
    if obs_arg:
        obs_path = Path(obs_arg)
        if not obs_path.is_absolute():
            obs_path = (Path.cwd() / obs_path).resolve()
        try:
            obs_path.relative_to(run_dir)
        except ValueError:
            abort_contract(f"observables-jsonl must live under {run_dir}, got {obs_path}")
        if not obs_path.exists():
            abort_contract(f"observables-jsonl not found at {obs_path}")
        return obs_path

    default_path = (
        run_dir
        / OBS_STAGE_NAME
        / "outputs"
        / "observables.jsonl"
    )
    if not default_path.exists():
        abort_contract(
            "observables.jsonl missing. Run stage_ringdown_observables_v1.py or pass --observables-jsonl"
        )
    return default_path


def _effective_rank(matrix: np.ndarray) -> float:
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] < 2:
        return float("nan")
    mat = mat - np.mean(mat, axis=0, keepdims=True)
    if not np.all(np.isfinite(mat)):
        return float("nan")
    try:
        _, s, _ = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")
    total = float(np.sum(s))
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    p = s / total
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def _load_baseline_exp01(path: Path) -> Dict[str, Tuple[float, float]]:
    baseline: Dict[str, Tuple[float, float]] = {}
    for row in _read_jsonl(path):
        if str(row.get("status", "")).upper() != "OK":
            continue
        case_id = str(row.get("case_id") or "")
        estimate = row.get("estimate") or {}
        f_hat = estimate.get("f_220_hat")
        tau_hat = estimate.get("tau_220_hat")
        if f_hat is None or tau_hat is None:
            continue
        f_hat = float(f_hat)
        tau_hat = float(tau_hat)
        if not (np.isfinite(f_hat) and np.isfinite(tau_hat)):
            continue
        baseline[case_id] = (f_hat, tau_hat)
    return baseline


def _load_baseline_exp02(path: Path) -> Dict[str, Tuple[float, float]]:
    baseline: Dict[str, Tuple[float, float]] = {}
    for row in _read_jsonl(path):
        case_id = str(row.get("case_id") or "")
        payload = row.get("baseline") or row.get("baseline_exp01") or {}
        f_hat = payload.get("f_220_hat")
        tau_hat = payload.get("tau_220_hat")
        if f_hat is None or tau_hat is None:
            continue
        f_hat = float(f_hat)
        tau_hat = float(tau_hat)
        if not (np.isfinite(f_hat) and np.isfinite(tau_hat)):
            continue
        baseline[case_id] = (f_hat, tau_hat)
    return baseline


def _load_observables(path: Path) -> Dict[str, Dict[str, float]]:
    obs_map: Dict[str, Dict[str, float]] = {}
    for row in _read_jsonl(path):
        case_id = str(row.get("case_id") or "")
        obs = row.get("observables") or {}
        values: Dict[str, float] = {}
        for key in CANDIDATES:
            if key not in obs:
                values = {}
                break
            val = float(obs[key])
            if not np.isfinite(val):
                values = {}
                break
            values[key] = val
        if values:
            obs_map[case_id] = values
    return obs_map


def main() -> int:
    ap = argparse.ArgumentParser(description="EXP_RINGDOWN_03 observable minimality (contract-first)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", choices=["exp01", "exp02"], default="exp01")
    ap.add_argument("--observables-jsonl", default=None)
    ap.add_argument("--min-gain", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    validate_run_id(args.run, out_root)
    run_dir = (out_root / args.run).resolve()

    try:
        require_run_valid(out_root, args.run)
    except Exception as exc:
        abort_contract(str(exc))

    if args.base == "exp02":
        base_path = _resolve_exp02_cases(run_dir)
        baseline = _load_baseline_exp02(base_path)
    else:
        base_path = _resolve_exp01_cases(run_dir)
        baseline = _load_baseline_exp01(base_path)

    if not baseline:
        abort_contract(f"baseline dataset empty: {base_path}")

    observables_path = _resolve_observables(run_dir, args.observables_jsonl)
    obs_map = _load_observables(observables_path)

    case_ids = [cid for cid in baseline.keys() if cid in obs_map]
    if len(case_ids) < 2:
        abort_contract("insufficient cases with observables to evaluate identifiability")

    base_matrix = np.array([baseline[cid] for cid in case_ids], dtype=float)
    baseline_value = _effective_rank(base_matrix)

    if not np.isfinite(baseline_value):
        abort_contract("baseline effective_rank is not finite")

    ablations_path = None
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)
    identifiability_path = outputs_dir / "identifiability_report.json"
    ablations_path = outputs_dir / "ablations.jsonl"
    contract_path = outputs_dir / "contract_verdict.json"

    rows: List[Dict[str, Any]] = []
    rankings: List[Dict[str, Any]] = []

    for candidate in CANDIDATES:
        values = np.array([obs_map[cid][candidate] for cid in case_ids], dtype=float)
        mat = np.column_stack([base_matrix, values])
        metric_value = _effective_rank(mat)
        gain = metric_value - baseline_value if np.isfinite(metric_value) else float("nan")
        passes_gain = bool(np.isfinite(gain) and gain >= float(args.min_gain))
        row = {
            "candidate_id": candidate,
            "metric_value": float(metric_value),
            "gain_vs_baseline": float(gain),
            "passes_gain": passes_gain,
            "chosen": False,
        }
        rows.append(row)

    for row in rows:
        rankings.append(
            {
                "observable_id": row["candidate_id"],
                "value": row["metric_value"],
                "gain": row["gain_vs_baseline"],
            }
        )

    def _rank_key(item: Dict[str, Any]) -> Tuple[float, int]:
        gain = item["gain"]
        idx = list(CANDIDATES).index(item["observable_id"])
        if not np.isfinite(gain):
            gain = -1.0e30
        return (-gain, idx)

    rankings.sort(key=_rank_key)
    ranking_topk = rankings[: max(1, int(args.topk))]

    chosen = None
    for candidate in CANDIDATES:
        row = next(r for r in rows if r["candidate_id"] == candidate)
        if row["passes_gain"]:
            chosen = candidate
            row["chosen"] = True
            break

    if chosen is None:
        for row in rows:
            row["chosen"] = False

    best = ranking_topk[0] if ranking_topk else None
    best_gain = float(best["gain"]) if best else float("nan")
    best_value = float(best["value"]) if best else float("nan")
    best_observable_id = str(best["observable_id"]) if best else ""

    with open(ablations_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    identifiability_report = {
        "schema_version": "exp_ringdown_03_identifiability_v1",
        "run_id": args.run,
        "base_dataset": {
            "path": str(base_path.relative_to(run_dir)),
            "sha256": sha256_file(base_path),
        },
        "observables_source": {
            "path": str(observables_path.relative_to(run_dir)),
            "sha256": sha256_file(observables_path),
        },
        "metric": {
            "name": METRIC_NAME,
            "baseline": float(baseline_value),
            "best": float(best_value),
            "gain": float(best_gain),
        },
        "best_observable_id": chosen or "",
        "ranking_topk": ranking_topk,
        "notes": ["deterministic sweep", "no theory-as-signal"],
    }

    with open(identifiability_path, "w", encoding="utf-8") as f:
        json.dump(identifiability_report, f, indent=2, sort_keys=True)
        f.write("\n")

    inputs = {
        "base_dataset": {
            "path": str(base_path.relative_to(run_dir)),
            "sha256": sha256_file(base_path),
        },
        "observables_source": {
            "path": str(observables_path.relative_to(run_dir)),
            "sha256": sha256_file(observables_path),
        },
    }

    identifiability_pass = bool(np.isfinite(best_gain) and best_gain >= float(args.min_gain))
    minimality_pass = bool(chosen is not None)

    contracts = {
        "R03_IDENTIFIABILITY_GAIN": {
            "verdict": "PASS" if identifiability_pass else "FAIL",
            "metric": METRIC_NAME,
            "min_gain": float(args.min_gain),
            "gain": float(best_gain),
            "bounds": {
                "metric_name": METRIC_NAME,
                "min_gain": float(args.min_gain),
                "baseline_value": float(baseline_value),
                "best_value": float(best_value),
            },
            "best_observable_id": best_observable_id,
            "ranking_topk": ranking_topk,
        },
        "R03_MINIMALITY": {
            "verdict": "PASS" if minimality_pass else "FAIL",
            "rule": "first_in_order_meeting_min_gain",
            "chosen": chosen or "",
            "candidates": list(CANDIDATES),
        },
    }

    verdict = "PASS" if identifiability_pass and minimality_pass else "FAIL"

    violations: List[str] = []
    if not identifiability_pass:
        violations.append("R03_IDENTIFIABILITY_GAIN")
    if not minimality_pass:
        violations.append("R03_MINIMALITY")

    contract_payload = {
        "verdict": verdict,
        "contracts": contracts,
        "violations": violations,
        "inputs": inputs,
        "outputs": {
            "identifiability_report": "outputs/identifiability_report.json",
            "ablations": "outputs/ablations.jsonl",
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
            "base": args.base,
            "min_gain": float(args.min_gain),
            "topk": int(args.topk),
            "metric": METRIC_NAME,
        },
        "results": {
            "overall_verdict": verdict,
            "best_observable_id": chosen or "",
            "gain": float(best_gain),
        },
    }
    summary_written = write_stage_summary(stage_dir, stage_summary)
    write_manifest(
        stage_dir,
        {
            "identifiability_report": identifiability_path,
            "ablations": ablations_path,
            "contract": contract_path,
            "stage_summary": summary_written,
        },
        extra={"version": "1"},
    )

    if verdict != "PASS":
        abort_contract("EXP_RINGDOWN_03 FAIL: identifiability or minimality did not pass")

    print("OK: EXP_RINGDOWN_03 PASS")
    print(f"  outputs: {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
