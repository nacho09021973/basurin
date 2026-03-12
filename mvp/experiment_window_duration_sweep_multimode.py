#!/usr/bin/env python3
"""Sweep multimode pipeline over multiple window_duration_s values for one event."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import resolve_out_root, sha256_file, validate_run_id, write_json_atomic, write_manifest, write_stage_summary
from mvp.experiment_band_sweep_multimode import (
    _classify_band_result,
    _edge_lock_side,
    _extract_abort_stage,
    _init_parent_run_valid,
    _print_stage_paths,
    _read_json_if_exists,
    _safe_event_id,
    _safe_float,
    _safe_int,
    _truncate_text,
    _write_csv_atomic,
    _band_token,
)

EXPERIMENT_STAGE = "experiment/window_duration_sweep_multimode"
RESULTS_JSON = "window_duration_sweep_results.json"
SUMMARY_CSV = "window_duration_sweep_summary.csv"
RECOMMENDATION_JSON = "recommendation.json"


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_window_grid(raw: str) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()
    for chunk in raw.split(','):
        token = chunk.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0:
            raise ValueError(f'invalid window_duration_s: {token!r} (must be > 0)')
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError('no valid window_duration_s values parsed from --window-duration-grid-s')
    return values


def _window_token(value: float) -> str:
    token = f"{float(value):.6f}".rstrip('0').rstrip('.')
    if token == '':
        token = '0'
    return token.replace('.', 'p')


def _subrun_id(event_id: str, window_duration_s: float, band_low: float, band_high: float, index: int) -> str:
    safe_event = _safe_event_id(event_id)
    return f"{safe_event}__win{index:02d}_{_window_token(window_duration_s)}s_band{_band_token(band_low)}_{_band_token(band_high)}"


def _build_cmd(args: argparse.Namespace, *, subrun_id: str, window_duration_s: float) -> list[str]:
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
        str(float(window_duration_s)),
        '--band-low',
        str(float(args.band_low)),
        '--band-high',
        str(float(args.band_high)),
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


def _summarize_subrun(*, subrun_dir: Path, subrun_id: str, event_id: str, dt_start_s: float, window_duration_s: float, band_low: float, band_high: float, proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
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
        'dt_start_s': float(dt_start_s),
        'window_duration_s': float(window_duration_s),
        'window_duration_label': _window_token(window_duration_s),
        'band_low_hz': float(band_low),
        'band_high_hz': float(band_high),
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
        overall = 'NO_WINDOW_DURATION_VALUES_EXECUTED'
        next_action = 'provide at least one valid window_duration_s value'
    elif n_nonempty > 0:
        overall = 'FOUND_SUPPORT_REGION'
        next_action = 'use the recommended window_duration_s for follow-up runs before changing the estimator'
    elif n_mode220_nonempty > 0:
        overall = 'MODE220_REGION_FOUND_BUT_FILTERED_LATER'
        next_action = 'inspect s4j/s4k for the recommended window before changing the band'
    elif completed and n_edge_locked == len(completed):
        overall = 'LIKELY_EDGE_LOCKED_OR_TAU_FLOOR_LOCKED'
        next_action = 'window-duration sweep did not rescue the run; the estimator likely needs attention'
    elif completed:
        overall = 'NO_SUPPORT_REGION_ACROSS_SWEEP'
        next_action = 'window-duration sweep completed without support region; compare with another band before scaling to cohort'
    else:
        overall = 'ALL_SUBRUNS_FAILED'
        next_action = 'inspect subrun stderr and failing stage before extending the window-duration sweep'

    return {
        'schema_name': 'window_duration_sweep_multimode_recommendation',
        'schema_version': 'v1',
        'created_utc': _utc_now_z(),
        'event_id': event_id,
        'host_run_id': args.run_id,
        'experiment_stage': EXPERIMENT_STAGE,
        'band_hz': [float(args.band_low), float(args.band_high)],
        'dt_start_s': float(args.dt_start_s),
        'overall_conclusion': overall,
        'n_window_values': len(results),
        'n_completed': len(completed),
        'n_nonempty_support': n_nonempty,
        'n_mode220_nonempty': n_mode220_nonempty,
        'n_edge_locked_like': n_edge_locked,
        'recommended_window_duration': None if best is None else {
            'subrun_id': best.get('subrun_id'),
            'window_duration_s': best.get('window_duration_s'),
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


def run_window_duration_sweep(args: argparse.Namespace) -> int:
    out_root = resolve_out_root('runs')
    validate_run_id(args.run_id, out_root)
    _init_parent_run_valid(out_root, args.run_id)

    host_run_dir = out_root / args.run_id
    stage_dir = host_run_dir / EXPERIMENT_STAGE
    outputs_dir = stage_dir / 'outputs'
    runsroot = stage_dir / 'runsroot'
    outputs_dir.mkdir(parents=True, exist_ok=True)
    runsroot.mkdir(parents=True, exist_ok=True)

    window_values = _parse_window_grid(args.window_duration_grid_s)
    env = os.environ.copy()
    env['BASURIN_RUNS_ROOT'] = str(runsroot)

    results: list[dict[str, Any]] = []
    for index, window_duration_s in enumerate(window_values, start=1):
        subrun_id = _subrun_id(args.event_id, window_duration_s, args.band_low, args.band_high, index)
        validate_run_id(subrun_id, runsroot)
        cmd = _build_cmd(args, subrun_id=subrun_id, window_duration_s=window_duration_s)
        print('[window_duration_sweep_multimode] $', ' '.join(shlex.quote(part) for part in cmd), flush=True)
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
            dt_start_s=args.dt_start_s,
            window_duration_s=window_duration_s,
            band_low=args.band_low,
            band_high=args.band_high,
            proc=proc,
        )
        results.append(row)
        print(
            '[window_duration_sweep_multimode] '
            f'window_duration_s={row["window_duration_s"]} pipeline_status={row["pipeline_status"]} '
            f'conclusion={row["conclusion_class"]} support_region_n_final={row.get("support_region_n_final")}',
            flush=True,
        )

    results_payload = {
        'schema_name': 'window_duration_sweep_multimode_results',
        'schema_version': 'v1',
        'created_utc': _utc_now_z(),
        'event_id': args.event_id,
        'host_run_id': args.run_id,
        'experiment_stage': EXPERIMENT_STAGE,
        'band_hz': [float(args.band_low), float(args.band_high)],
        'dt_start_s': float(args.dt_start_s),
        'window_duration_grid_s': [float(v) for v in window_values],
        'subruns_root': str(runsroot),
        'params': {
            'duration_s': float(args.duration_s),
            'dt_start_s': float(args.dt_start_s),
            'band_low': float(args.band_low),
            'band_high': float(args.band_high),
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
        'n_window_values': len(results),
        'n_completed': sum(1 for row in results if row.get('pipeline_status') == 'PASS'),
        'results': results,
    }
    recommendation = _build_recommendation(results, event_id=args.event_id, args=args)

    results_path = outputs_dir / RESULTS_JSON
    summary_csv_path = outputs_dir / SUMMARY_CSV
    recommendation_path = outputs_dir / RECOMMENDATION_JSON
    write_json_atomic(results_path, results_payload)
    fieldnames = [
        'event_id', 'subrun_id', 'dt_start_s', 'window_duration_s', 'band_low_hz', 'band_high_hz', 'pipeline_status', 'returncode', 'error_stage',
        's3_f_hz', 's3_tau_s', 's3_Q', 's3_sigma_f_hz', 's3_sigma_tau_s', 's4g_verdict', 's4g_n_geometries_accepted',
        'multimode_viability_class', 'fallback_path', 'support_region_status', 'support_region_n_final', 'downstream_status_class',
        'edge_lock_side', 'tau_floor_locked', 'q_too_low', 'conclusion_class', 'selection_score',
    ]
    _write_csv_atomic(summary_csv_path, results, fieldnames)
    write_json_atomic(recommendation_path, recommendation)

    summary = {
        'stage': EXPERIMENT_STAGE,
        'run_id': args.run_id,
        'verdict': 'PASS',
        'results': {
            'event_id': args.event_id,
            'band_hz': [float(args.band_low), float(args.band_high)],
            'dt_start_s': float(args.dt_start_s),
            'n_window_values': len(results),
            'n_completed': results_payload['n_completed'],
            'overall_conclusion': recommendation['overall_conclusion'],
            'recommended_window_duration': recommendation['recommended_window_duration'],
            'results_sha256': sha256_file(results_path),
            'summary_csv_sha256': sha256_file(summary_csv_path),
            'recommendation_sha256': sha256_file(recommendation_path),
        },
    }
    write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {
            'window_duration_sweep_results': results_path,
            'window_duration_sweep_summary_csv': summary_csv_path,
            'recommendation': recommendation_path,
        },
        extra={
            'schema_name': 'window_duration_sweep_multimode_manifest',
            'subruns_root': str(runsroot),
        },
    )

    print('[window_duration_sweep_multimode] overall_conclusion=' + str(recommendation['overall_conclusion']))
    if recommendation.get('recommended_window_duration') is not None:
        selected = recommendation['recommended_window_duration']
        print(
            '[window_duration_sweep_multimode] recommended_window_duration=' +
            f"{selected.get('window_duration_s')} conclusion={selected.get('conclusion_class')}",
        )
    _print_stage_paths(out_root=out_root, stage_dir=stage_dir, outputs_dir=outputs_dir)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Experiment: sweep multimode pipeline across window_duration_s values for one event')
    ap.add_argument('--run-id', required=True, help='Host run id for experiment outputs')
    ap.add_argument('--event-id', required=True)
    ap.add_argument('--window-duration-grid-s', required=True, help='Comma-separated window_duration_s values in seconds, e.g. 0.02,0.04,0.06,0.08')
    ap.add_argument('--band-low', type=float, required=True)
    ap.add_argument('--band-high', type=float, required=True)
    ap.add_argument('--duration-s', type=float, default=32.0)
    ap.add_argument('--dt-start-s', type=float, default=0.003)
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
        return run_window_duration_sweep(args)
    except ValueError as exc:
        print(f'[window_duration_sweep_multimode] ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
