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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from basurin_io import (
    ensure_stage_dirs,
    resolve_out_root,
    require_run_valid,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)


EXIT_CONTRACT_FAIL = 2
STAGE_NAME = "experiment/ringdown/EXP_RINGDOWN_04__psd_validity"
PSD_NPERSEG = 1024
PSD_OVERLAP = 0.5
PSD_FLOOR = 1.0e-12
DEFAULT_DT = 1.0 / 4096.0
MIN_CASES = 1


def abort_contract(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_relative(run_dir: Path, rel: str) -> Path:
    candidate = Path(rel)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (run_dir / "ringdown_synth" / "outputs" / candidate).resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"path {rel} escapes run_dir {run_dir}") from exc
    return resolved


def _resolve_strain_path(
    event: Dict[str, Any],
    run_dir: Path,
    cases_dir: Path,
    fallback_paths: List[Path],
    index: int,
) -> Tuple[Optional[Path], str]:
    if "path" in event:
        candidate = _resolve_relative(run_dir, str(event["path"]))
        if candidate.exists():
            return candidate, "path"
    for key in ("strain_npz",):
        if key in event:
            candidate = _resolve_relative(run_dir, str(event[key]))
            if candidate.exists():
                return candidate, key
    paths = event.get("paths") or {}
    if isinstance(paths, dict) and "strain_npz" in paths:
        candidate = _resolve_relative(run_dir, str(paths["strain_npz"]))
        if candidate.exists():
            return candidate, "paths.strain_npz"
    outputs = event.get("outputs") or {}
    if isinstance(outputs, dict) and "strain" in outputs:
        candidate = _resolve_relative(run_dir, str(outputs["strain"]))
        if candidate.exists():
            return candidate, "outputs.strain"
    case_id = event.get("case_id") or event.get("id")
    if case_id:
        candidate = (cases_dir / str(case_id) / "strain.npz").resolve()
        if candidate.exists():
            return candidate, "case_id"
    if fallback_paths and 0 <= index < len(fallback_paths):
        candidate = fallback_paths[index]
        if candidate.exists():
            return candidate, "glob_index"
    return None, "missing"


def _load_strain(path: Path) -> Tuple[np.ndarray, float]:
    with np.load(path) as data:
        if "strain" in data:
            strain = np.asarray(data["strain"], dtype=float)
        elif "h" in data:
            strain = np.asarray(data["h"], dtype=float)
        else:
            raise ValueError("strain key missing (expected 'strain' or 'h')")
        if "t" in data:
            t = np.asarray(data["t"], dtype=float)
            if t.ndim != 1 or len(t) < 2:
                raise ValueError("t array invalid")
            dt = float(np.median(np.diff(t)))
        elif "dt" in data:
            dt = float(data["dt"])
        else:
            dt = float(DEFAULT_DT)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt invalid")
    strain = np.asarray(strain, dtype=float).reshape(-1)
    if strain.size < 8 or not np.all(np.isfinite(strain)):
        raise ValueError("strain invalid")
    return strain, dt


def _welch_psd(strain: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, int, int]:
    n = len(strain)
    nperseg = min(int(PSD_NPERSEG), n)
    if nperseg < 8:
        raise ValueError("nperseg too small for PSD")
    step = max(1, int(nperseg * (1.0 - PSD_OVERLAP)))
    window = np.hanning(nperseg)
    window_norm = float(np.sum(window**2))
    if window_norm <= 0:
        raise ValueError("window normalization invalid")
    segments = []
    for start in range(0, n - nperseg + 1, step):
        segment = strain[start : start + nperseg]
        if segment.shape[0] != nperseg:
            continue
        seg = segment * window
        fft = np.fft.rfft(seg)
        power = (np.abs(fft) ** 2) * (dt / window_norm)
        if power.size > 2:
            power[1:-1] *= 2.0
        segments.append(power)
    if not segments:
        raise ValueError("no PSD segments")
    psd = np.mean(np.vstack(segments), axis=0)
    freqs = np.fft.rfftfreq(nperseg, dt)
    return psd.astype(float), freqs.astype(float), len(segments), nperseg


def _whiten_check(strain: np.ndarray, dt: float, psd: np.ndarray, freqs: np.ndarray) -> bool:
    n = len(strain)
    fft_full = np.fft.rfft(strain)
    freqs_full = np.fft.rfftfreq(n, dt)
    psd_interp = np.interp(freqs_full, freqs, psd, left=psd[0], right=psd[-1])
    if not np.all(np.isfinite(psd_interp)):
        return False
    if np.any(psd_interp <= PSD_FLOOR):
        return False
    whitened = np.fft.irfft(fft_full / np.sqrt(psd_interp), n=n)
    return bool(np.all(np.isfinite(whitened)))


def main() -> int:
    ap = argparse.ArgumentParser(description="EXP_RINGDOWN_04 PSD validity (contract-first)")
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    validate_run_id(args.run, out_root)
    run_dir = (out_root / args.run).resolve()

    try:
        require_run_valid(out_root, args.run)
    except Exception as exc:
        abort_contract(str(exc))

    events_list_path = (
        run_dir / "ringdown_synth" / "outputs" / "synthetic_events_list.json"
    )
    if not events_list_path.exists():
        abort_contract(f"missing synthetic_events_list.json at {events_list_path}")

    events_list = _read_json(events_list_path)
    if not isinstance(events_list, list):
        abort_contract("synthetic_events_list.json must be a list")
    if not events_list:
        abort_contract("synthetic_events_list.json is empty")

    cases_dir = run_dir / "ringdown_synth" / "outputs" / "cases"
    fallback_paths = sorted(cases_dir.glob("*/strain.npz"))

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)
    per_case_path = outputs_dir / "per_case_psd.jsonl"
    diagnostics_path = outputs_dir / "psd_diagnostics.json"
    contract_path = outputs_dir / "contract_verdict.json"

    counts: Dict[str, int] = {}
    n_total = len(events_list)
    n_effective = 0
    n_psd_bad = 0
    n_whiten_bad = 0
    n_rows = 0

    with open(per_case_path, "w", encoding="utf-8") as f:
        for idx, event in enumerate(events_list):
            event = event if isinstance(event, dict) else {}
            case_id = str(event.get("case_id") or event.get("id") or f"case_{idx:04d}")
            strain_path, resolution = _resolve_strain_path(
                event, run_dir, cases_dir, fallback_paths, idx
            )
            row: Dict[str, Any] = {
                "case_id": case_id,
                "status": "SKIP",
                "reason": "",
                "resolution": resolution,
                "path": None,
                "min_psd": None,
                "max_psd": None,
                "frac_bad": None,
                "n_bad": None,
                "n_bins": None,
                "n_segments": None,
                "nperseg": None,
                "dt": None,
                "n_samples": None,
            }

            if strain_path is None:
                row["reason"] = "missing_strain"
                counts[row["reason"]] = counts.get(row["reason"], 0) + 1
                f.write(json.dumps(row) + "\n")
                n_rows += 1
                continue

            row["path"] = str(strain_path.relative_to(run_dir))

            try:
                strain, dt = _load_strain(strain_path)
                psd, freqs, n_segments, nperseg = _welch_psd(strain, dt)
            except Exception as exc:
                row["reason"] = f"load_fail:{exc}"
                counts[row["reason"]] = counts.get(row["reason"], 0) + 1
                f.write(json.dumps(row) + "\n")
                n_rows += 1
                continue

            finite = np.all(np.isfinite(psd))
            positive = np.all(psd > 0)
            bad_mask = ~np.isfinite(psd) | (psd <= PSD_FLOOR)
            n_bad = int(np.sum(bad_mask))
            frac_bad = float(n_bad) / float(psd.size)

            row.update(
                {
                    "min_psd": float(np.min(psd)),
                    "max_psd": float(np.max(psd)),
                    "frac_bad": float(frac_bad),
                    "n_bad": int(n_bad),
                    "n_bins": int(psd.size),
                    "n_segments": int(n_segments),
                    "nperseg": int(nperseg),
                    "dt": float(dt),
                    "n_samples": int(len(strain)),
                }
            )

            psd_ok = bool(finite and positive and n_bad == 0)
            whiten_ok = False
            if psd_ok:
                whiten_ok = _whiten_check(strain, dt, psd, freqs)

            if not psd_ok:
                row["reason"] = "psd_invalid"
                n_psd_bad += 1
                counts[row["reason"]] = counts.get(row["reason"], 0) + 1
            elif not whiten_ok:
                row["reason"] = "whitening_invalid"
                n_whiten_bad += 1
                counts[row["reason"]] = counts.get(row["reason"], 0) + 1
            else:
                row["status"] = "OK"
                row["reason"] = "ok"
                n_effective += 1
                counts[row["reason"]] = counts.get(row["reason"], 0) + 1

            f.write(json.dumps(row) + "\n")
            n_rows += 1

    diagnostics = {
        "schema_version": "exp_ringdown_04_psd_validity_v1",
        "run_id": args.run,
        "n_total": int(n_total),
        "n_effective": int(n_effective),
        "counts": counts,
        "bounds": {
            "psd_floor": float(PSD_FLOOR),
            "min_cases": int(MIN_CASES),
            "nperseg": int(PSD_NPERSEG),
            "overlap": float(PSD_OVERLAP),
        },
        "inputs": {
            "synthetic_events_list": {
                "path": str(events_list_path.relative_to(run_dir)),
                "sha256": sha256_file(events_list_path),
            }
        },
    }

    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, sort_keys=True)
        f.write("\n")

    psd_contract_pass = bool(n_psd_bad == 0)
    whiten_contract_pass = bool(n_whiten_bad == 0)
    diagnostics_complete_pass = bool(n_rows == n_total)
    coverage_pass = bool(n_effective >= MIN_CASES)

    contracts = {
        "R04_PSD_WELL_CONDITIONED": {
            "verdict": "PASS" if psd_contract_pass else "FAIL",
            "bounds": {"psd_floor": float(PSD_FLOOR)},
            "counts": {"n_bad_psd": int(n_psd_bad), "n_total": int(n_total)},
        },
        "R04_WHITENING_FINITE": {
            "verdict": "PASS" if whiten_contract_pass else "FAIL",
            "counts": {"n_bad_whitening": int(n_whiten_bad), "n_total": int(n_total)},
        },
        "R04_DIAGNOSTICS_COMPLETE": {
            "verdict": "PASS" if diagnostics_complete_pass else "FAIL",
            "counts": {"n_rows": int(n_rows), "n_total": int(n_total)},
        },
        "R04_COVERAGE": {
            "verdict": "PASS" if coverage_pass else "FAIL",
            "min_cases": int(MIN_CASES),
            "n_effective": int(n_effective),
        },
    }

    violations = [key for key, value in contracts.items() if value["verdict"] != "PASS"]
    verdict = "PASS" if not violations else "FAIL"

    contract_payload = {
        "verdict": verdict,
        "contracts": contracts,
        "violations": violations,
        "inputs": {
            "synthetic_events_list": {
                "path": str(events_list_path.relative_to(run_dir)),
                "sha256": sha256_file(events_list_path),
            }
        },
        "outputs": {
            "psd_diagnostics": "outputs/psd_diagnostics.json",
            "per_case_psd": "outputs/per_case_psd.jsonl",
            "contract_verdict": "outputs/contract_verdict.json",
        },
    }

    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    stage_summary = {
        "stage": stage_dir.name,
        "run": args.run,
        "inputs": contract_payload["inputs"],
        "parameters": {
            "psd_floor": float(PSD_FLOOR),
            "nperseg": int(PSD_NPERSEG),
            "overlap": float(PSD_OVERLAP),
            "min_cases": int(MIN_CASES),
        },
        "outputs": contract_payload["outputs"],
        "results": {"overall_verdict": verdict, "n_effective": int(n_effective)},
    }
    summary_written = write_stage_summary(stage_dir, stage_summary)
    write_manifest(
        stage_dir,
        {
            "psd_diagnostics": diagnostics_path,
            "per_case_psd": per_case_path,
            "contract": contract_path,
            "stage_summary": summary_written,
        },
        extra={"version": "1"},
    )

    if verdict != "PASS":
        abort_contract("EXP_RINGDOWN_04 FAIL: PSD/whitening validity did not pass")

    print("OK: EXP_RINGDOWN_04 PASS")
    print(f"  outputs: {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
