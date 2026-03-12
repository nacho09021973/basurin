#!/usr/bin/env python3
"""Sweep multimode pipeline over multiple analysis bands for one event."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import resolve_out_root, sha256_file, validate_run_id, write_json_atomic, write_manifest, write_stage_summary

EXPERIMENT_STAGE = "experiment/band_sweep_multimode"
RESULTS_JSON = "band_sweep_results.json"
SUMMARY_CSV = "band_sweep_summary.csv"
RECOMMENDATION_JSON = "recommendation.json"
ABORT_STAGE_RE = re.compile(r"\[pipeline\] ABORT: ([^ ]+) failed")
BAND_RE = re.compile(r"^\s*(?P<low>[0-9]+(?:\.[0-9]+)?)\s*-\s*(?P<high>[0-9]+(?:\.[0-9]+)?)\s*$")


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_event_id(event_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in event_id)


def _band_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(round(float(value))))
    token = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return token.replace(".", "p")


def _parse_bands(raw: str) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        match = BAND_RE.match(token)
        if not match:
            raise ValueError(f"invalid band token: {token!r} (expected LOW-HIGH)")
        low = float(match.group("low"))
        high = float(match.group("high"))
        if not math.isfinite(low) or not math.isfinite(high) or low < 0.0 or high <= low:
            raise ValueError(f"invalid band range: {token!r}")
        pair = (low, high)
        if pair not in seen:
            bands.append(pair)
            seen.add(pair)
    if not bands:
        raise ValueError("no valid bands parsed from --bands")
    return bands


def _init_parent_run_valid(out_root: Path, run_id: str) -> None:
    verdict = out_root / run_id / "RUN_VALID" / "verdict.json"
    if verdict.exists():
        return
    verdict.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(verdict, {"verdict": "PASS"})


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return path


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    return payload if isinstance(payload, dict) else None


def _truncate_text(text: str, *, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + '...[truncated]'


def _extract_abort_stage(stdout: str, stderr: str) -> str | None:
    combined = "\n".join(part for part in (stdout, stderr) if part)

    match = ABORT_STAGE_RE.search(combined)
    return match.group(1) if match else None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _edge_lock_side(f_hz: float | None, band_low: float, band_high: float) -> str | None:
    if f_hz is None:
        return None
    width = max(band_high - band_low, 1.0)
    tol = max(2.0, 0.01 * width)
    if abs(f_hz - band_low) <= tol:
        return 'LOW'
    if abs(f_hz - band_high) <= tol:
        return 'HIGH'
    return None


def _classify_band_result(row: dict[str, Any]) -> tuple[str, list[str], int]:
    reasons: list[str] = []
    n_final = _safe_int(row.get('support_region_n_final')) or 0
    n_accepted = _safe_int(row.get('s4g_n_geometries_accepted')) or 0
    edge_lock_side = row.get('edge_lock_side')
    tau_floor_locked = bool(row.get('tau_floor_locked'))
    q_too_low = bool(row.get('q_too_low'))
    if row.get('pipeline_status') != 'PASS':
        stage = row.get('error_stage') or 'unknown_stage'
        return 'FAILED_SUBRUN', [f'pipeline_failed_at={stage}'], -1
    if n_final > 0:
        reasons.append(f'support_region_n_final={n_final}')
        return 'FOUND_NONEMPTY_SUPPORT_REGION', reasons, 50 + n_final
    if n_accepted > 0:
        reasons.append(f's4g_n_geometries_accepted={n_accepted}')
        reasons.append(f'support_region_status={row.get("support_region_status")}')
        return 'MODE220_SUPPORT_FOUND_BUT_FILTERED_LATER', reasons, 30 + n_accepted
    if edge_lock_side is not None and tau_floor_locked:
        reasons.append(f'edge_lock_side={edge_lock_side}')
        reasons.append('tau_floor_locked=true')
        if q_too_low:
            reasons.append('q_too_low=true')
        return 'LIKELY_EDGE_LOCKED_220', reasons, 5
    if edge_lock_side is not None:
        reasons.append(f'edge_lock_side={edge_lock_side}')
        return 'EDGE_LOCKED_220', reasons, 4
    if tau_floor_locked:
        reasons.append('tau_floor_locked=true')
        if q_too_low:
            reasons.append('q_too_low=true')
        return 'TAU_FLOOR_LOCKED_220', reasons, 3
    if q_too_low:
        reasons.append('q_too_low=true')
        return 'UNPHYSICAL_LOW_Q_220', reasons, 2
    reasons.append(f'support_region_status={row.get("support_region_status")}')
    reasons.append(f'multimode_viability_class={row.get("multimode_viability_class")}')
    return 'NO_SUPPORT_REGION_ACROSS_FILTERS', reasons, 1


def _subrun_id(event_id: str, band_low: float, band_high: float, index: int) -> str:
    safe_event = _safe_event_id(event_id)
    return f"{safe_event}__band{index:02d}_{_band_token(band_low)}_{_band_token(band_high)}"


def _build_cmd(args: argparse.Namespace, *, subrun_id: str, band_low: float, band_high: float) -> list[str]:
    cmd = [
        sys.executable,
        '-m',
        'mvp.pipeline',
        'multimode',
        '--event-id',
        args.event_id,
        '--run-id',
        subrun_id,
        '--duration-s',
        str(float(args.duration_s)),
        '--dt-start-s',
        str(float(args.dt_start_s)),
        '--window-duration-s',
        str(float(args.window_duration_s)),
        '--band-low',
        str(float(band_low)),
        '--band-high',
        str(float(band_high)),
        '--s3b-method',
        args.s3b_method,
        '--s3b-n-bootstrap',
        str(int(args.s3b_n_bootstrap)),
        '--s3b-seed',
        str(int(args.s3b_seed)),
    ]
    if args.atlas_default:
        cmd.append('--atlas-default')
    elif args.atlas_path:
        cmd.extend(['--atlas-path', args.atlas_path])
    if args.offline:
        cmd.append('--offline')
    if args.reuse_strain:
        cmd.append('--reuse-strain')
    if args.with_t0_sweep:
        cmd.append('--with-t0-sweep')
    if args.stage_timeout_s is not None:
        cmd.extend(['--stage-timeout-s', str(float(args.stage_timeout_s))])
    return cmd


def _summarize_subrun(*, subrun_dir: Path, subrun_id: str, event_id: str, band_low: float, band_high: float, proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    timeline = _read_json_if_exists(subrun_dir / 'pipeline_timeline.json') or {}
    estimates = _read_json_if_exists(subrun_dir / 's3_ringdown_estimates' / 'outputs' / 'estimates.json') or {}
    mode220 = _read_json_if_exists(subrun_dir / 's4g_mode220_geometry_filter' / 'outputs' / 'mode220_filter.json') or {}
    multimode = timeline.get('multimode_results', {}) if isinstance(timeline.get('multimode_results', {}), dict) else {}
    combined = estimates.get('combined', {}) if isinstance(estimates.get('combined', {}), dict) else {}
    uncertainty = estimates.get('combined_uncertainty', {}) if isinstance(estimates.get('combined_uncertainty', {}), dict) else {}

    f_hz = _safe_float(combined.get('f_hz'))
    tau_s = _safe_float(combined.get('tau_s'))
    q_val = _safe_float(combined.get('Q'))
    edge_side = _edge_lock_side(f_hz, band_low, band_high)
    tau_floor_locked = tau_s is not None and tau_s <= 1.05e-4
    q_too_low = q_val is not None and q_val < 1.0

    row: dict[str, Any] = {
        'event_id': event_id,
        'subrun_id': subrun_id,
        'subrun_dir': str(subrun_dir),
        'band_low_hz': float(band_low),
        'band_high_hz': float(band_high),
        'band_label': f'{_band_token(band_low)}-{_band_token(band_high)}',
        'pipeline_status': 'PASS' if proc.returncode == 0 else 'FAILED_SUBRUN',
        'returncode': int(proc.returncode),
        'error_stage': _extract_abort_stage(proc.stdout or '', proc.stderr or ''),
        's3_f_hz': f_hz,
        's3_tau_s': tau_s,
        's3_Q': q_val,
        's3_sigma_f_hz': _safe_float(uncertainty.get('sigma_f_hz')),
        's3_sigma_tau_s': _safe_float(uncertainty.get('sigma_tau_s')),
        's4g_verdict': mode220.get('verdict'),
        's4g_n_geometries_accepted': _safe_int(mode220.get('n_geometries_accepted')),
        'multimode_viability_class': multimode.get('multimode_viability_class'),
        'fallback_path': multimode.get('fallback_path'),
        'support_region_status': multimode.get('support_region_status'),
        'support_region_n_final': _safe_int(multimode.get('support_region_n_final')),
        'downstream_status_class': multimode.get('downstream_status_class'),
        'edge_lock_side': edge_side,
        'tau_floor_locked': bool(tau_floor_locked),
        'q_too_low': bool(q_too_low),
        'stdout_tail': _truncate_text(proc.stdout or ''),
        'stderr_tail': _truncate_text(proc.stderr or ''),
    }
    conclusion_class, reasons, score = _classify_band_result(row)
    row['conclusion_class'] = conclusion_class
    row['conclusion_reasons'] = reasons
    row['selection_score'] = score
    return row


def _build_recommendation(results: list[dict[str, Any]], *, event_id: str, args: argparse.Namespace) -> dict[str, Any]:
    completed = [row for row in results if row.get('pipeline_status') == 'PASS']
    best = max(results, key=lambda row: (int(row.get('selection_score') or -999), -(int(row.get('returncode') or 999)))) if results else None
    n_nonempty = sum(1 for row in completed if (_safe_int(row.get('support_region_n_final')) or 0) > 0)
    n_mode220_nonempty = sum(1 for row in completed if (_safe_int(row.get('s4g_n_geometries_accepted')) or 0) > 0)
    n_edge_locked = sum(1 for row in completed if row.get('conclusion_class') in {'LIKELY_EDGE_LOCKED_220', 'EDGE_LOCKED_220', 'TAU_FLOOR_LOCKED_220', 'UNPHYSICAL_LOW_Q_220'})

    if not results:
        overall = 'NO_BANDS_EXECUTED'
        next_action = 'provide at least one valid band'
    elif n_nonempty > 0:
        overall = 'FOUND_SUPPORT_REGION'
        next_action = 'use the recommended band for a larger cohort before changing other knobs'
    elif n_mode220_nonempty > 0:
        overall = 'MODE220_REGION_FOUND_BUT_FILTERED_LATER'
        next_action = 'inspect s4j/s4k for the recommended band before changing the band grid'
    elif completed and n_edge_locked == len(completed):
        overall = 'LIKELY_EDGE_LOCKED_OR_TAU_FLOOR_LOCKED'
        next_action = 'shift the analysis band upward and/or enable --with-t0-sweep before launching a larger cohort'
    elif completed:
        overall = 'NO_SUPPORT_REGION_ACROSS_SWEEP'
        next_action = 'try a different dt-start or t0 sweep; the current band sweep still yields empty support regions'
    else:
        overall = 'ALL_SUBRUNS_FAILED'
        next_action = 'inspect subrun stderr and failing stage before extending the sweep'

    return {
        'schema_name': 'band_sweep_multimode_recommendation',
        'schema_version': 'v1',
        'created_utc': _utc_now_z(),
        'event_id': event_id,
        'host_run_id': args.run_id,
        'experiment_stage': EXPERIMENT_STAGE,
        'overall_conclusion': overall,
        'n_bands': len(results),
        'n_completed': len(completed),
        'n_nonempty_support': n_nonempty,
        'n_mode220_nonempty': n_mode220_nonempty,
        'n_edge_locked_like': n_edge_locked,
        'recommended_band': None if best is None else {
            'subrun_id': best.get('subrun_id'),
            'band_low_hz': best.get('band_low_hz'),
            'band_high_hz': best.get('band_high_hz'),
            'conclusion_class': best.get('conclusion_class'),
            'conclusion_reasons': best.get('conclusion_reasons'),
            'support_region_n_final': best.get('support_region_n_final'),
            's4g_n_geometries_accepted': best.get('s4g_n_geometries_accepted'),
            'multimode_viability_class': best.get('multimode_viability_class'),
            'fallback_path': best.get('fallback_path'),
            'downstream_status_class': best.get('downstream_status_class'),
        },
        'next_action': next_action,
    }


def _print_stage_paths(*, out_root: Path, stage_dir: Path, outputs_dir: Path) -> None:
    print(f'OUT_ROOT={out_root}')
    print(f'STAGE_DIR={stage_dir}')
    print(f'OUTPUTS_DIR={outputs_dir}')
    print(f'STAGE_SUMMARY={stage_dir / "stage_summary.json"}')
    print(f'MANIFEST={stage_dir / "manifest.json"}')


def run_band_sweep(args: argparse.Namespace) -> int:
    out_root = resolve_out_root('runs')
    validate_run_id(args.run_id, out_root)
    _init_parent_run_valid(out_root, args.run_id)

    host_run_dir = out_root / args.run_id
    stage_dir = host_run_dir / EXPERIMENT_STAGE
    outputs_dir = stage_dir / 'outputs'
    runsroot = stage_dir / 'runsroot'
    outputs_dir.mkdir(parents=True, exist_ok=True)
    runsroot.mkdir(parents=True, exist_ok=True)

    bands = _parse_bands(args.bands)
    env = os.environ.copy()
    env['BASURIN_RUNS_ROOT'] = str(runsroot)

    results: list[dict[str, Any]] = []
    for index, (band_low, band_high) in enumerate(bands, start=1):
        subrun_id = _subrun_id(args.event_id, band_low, band_high, index)
        validate_run_id(subrun_id, runsroot)
        cmd = _build_cmd(args, subrun_id=subrun_id, band_low=band_low, band_high=band_high)
        print('[band_sweep_multimode] $', ' '.join(shlex.quote(part) for part in cmd), flush=True)
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_here.parents[1]),
        )
        subrun_dir = runsroot / subrun_id
        row = _summarize_subrun(
            subrun_dir=subrun_dir,
            subrun_id=subrun_id,
            event_id=args.event_id,
            band_low=band_low,
            band_high=band_high,
            proc=proc,
        )
        results.append(row)
        print(
            '[band_sweep_multimode] '
            f'band={row["band_label"]} pipeline_status={row["pipeline_status"]} '
            f'conclusion={row["conclusion_class"]} support_region_n_final={row.get("support_region_n_final")}',
            flush=True,
        )

    results_payload = {
        'schema_name': 'band_sweep_multimode_results',
        'schema_version': 'v1',
        'created_utc': _utc_now_z(),
        'event_id': args.event_id,
        'host_run_id': args.run_id,
        'experiment_stage': EXPERIMENT_STAGE,
        'bands_hz': [{'low': low, 'high': high} for low, high in bands],
        'subruns_root': str(runsroot),
        'params': {
            'duration_s': float(args.duration_s),
            'dt_start_s': float(args.dt_start_s),
            'window_duration_s': float(args.window_duration_s),
            'atlas_path': args.atlas_path,
            'atlas_default': bool(args.atlas_default),
            'offline': bool(args.offline),
            'reuse_strain': bool(args.reuse_strain),
            'with_t0_sweep': bool(args.with_t0_sweep),
            's3b_method': args.s3b_method,
            's3b_n_bootstrap': int(args.s3b_n_bootstrap),
            's3b_seed': int(args.s3b_seed),
            'stage_timeout_s': None if args.stage_timeout_s is None else float(args.stage_timeout_s),
        },
        'n_bands': len(results),
        'n_completed': sum(1 for row in results if row.get('pipeline_status') == 'PASS'),
        'results': results,
    }
    recommendation = _build_recommendation(results, event_id=args.event_id, args=args)

    results_path = outputs_dir / RESULTS_JSON
    summary_csv_path = outputs_dir / SUMMARY_CSV
    recommendation_path = outputs_dir / RECOMMENDATION_JSON
    write_json_atomic(results_path, results_payload)
    fieldnames = [
        'event_id',
        'subrun_id',
        'band_low_hz',
        'band_high_hz',
        'pipeline_status',
        'returncode',
        'error_stage',
        's3_f_hz',
        's3_tau_s',
        's3_Q',
        's3_sigma_f_hz',
        's3_sigma_tau_s',
        's4g_verdict',
        's4g_n_geometries_accepted',
        'multimode_viability_class',
        'fallback_path',
        'support_region_status',
        'support_region_n_final',
        'downstream_status_class',
        'edge_lock_side',
        'tau_floor_locked',
        'q_too_low',
        'conclusion_class',
        'selection_score',
    ]
    _write_csv_atomic(summary_csv_path, results, fieldnames)
    write_json_atomic(recommendation_path, recommendation)

    summary = {
        'stage': EXPERIMENT_STAGE,
        'run_id': args.run_id,
        'verdict': 'PASS',
        'results': {
            'event_id': args.event_id,
            'n_bands': len(results),
            'n_completed': results_payload['n_completed'],
            'overall_conclusion': recommendation['overall_conclusion'],
            'recommended_band': recommendation['recommended_band'],
            'results_sha256': sha256_file(results_path),
            'summary_csv_sha256': sha256_file(summary_csv_path),
            'recommendation_sha256': sha256_file(recommendation_path),
        },
    }
    write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {
            'band_sweep_results': results_path,
            'band_sweep_summary_csv': summary_csv_path,
            'recommendation': recommendation_path,
        },
        extra={
            'schema_name': 'band_sweep_multimode_manifest',
            'subruns_root': str(runsroot),
        },
    )

    print('[band_sweep_multimode] overall_conclusion=' + str(recommendation['overall_conclusion']))
    if recommendation.get('recommended_band') is not None:
        band = recommendation['recommended_band']
        print(
            '[band_sweep_multimode] recommended_band='
            f"{band.get('band_low_hz')}-{band.get('band_high_hz')} "
            f"conclusion={band.get('conclusion_class')}",
        )
    _print_stage_paths(out_root=out_root, stage_dir=stage_dir, outputs_dir=outputs_dir)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Experiment: sweep multimode pipeline across analysis bands for one event')
    ap.add_argument('--run-id', required=True, help='Host run id for experiment outputs')
    ap.add_argument('--event-id', required=True)
    ap.add_argument('--bands', required=True, help='Comma-separated LOW-HIGH bands in Hz, e.g. 150-400,400-800,800-1200')
    ap.add_argument('--duration-s', type=float, default=32.0)
    ap.add_argument('--dt-start-s', type=float, default=0.003)
    ap.add_argument('--window-duration-s', type=float, default=0.06)
    ap.add_argument('--atlas-path', default=None)
    ap.add_argument('--atlas-default', action='store_true', default=False)
    ap.add_argument('--offline', action='store_true', default=False)
    ap.add_argument('--reuse-strain', action='store_true', default=False)
    ap.add_argument('--with-t0-sweep', action='store_true', default=False)
    ap.add_argument('--s3b-method', choices=['hilbert_peakband', 'spectral_two_pass'], default='hilbert_peakband')
    ap.add_argument('--s3b-n-bootstrap', type=int, default=200)
    ap.add_argument('--s3b-seed', type=int, default=12345)
    ap.add_argument('--stage-timeout-s', type=float, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if bool(args.atlas_default) == bool(args.atlas_path):
        raise SystemExit('choose exactly one of --atlas-default or --atlas-path')
    try:
        return run_band_sweep(args)
    except ValueError as exc:
        print(f'[band_sweep_multimode] ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
