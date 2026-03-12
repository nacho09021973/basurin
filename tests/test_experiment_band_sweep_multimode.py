from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp import experiment_band_sweep_multimode as exp


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def test_band_sweep_writes_outputs_and_recommends_best_band(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / 'runs'
    monkeypatch.setenv('BASURIN_RUNS_ROOT', str(runs_root))
    atlas = tmp_path / 'atlas.json'
    atlas.write_text('{}', encoding='utf-8')

    def fake_run(cmd: list[str], *, check: bool, capture_output: bool, text: bool, env: dict[str, str], cwd: str):
        assert check is False
        assert capture_output is True
        assert text is True
        assert Path(env['BASURIN_RUNS_ROOT']).parts[-1] == 'runsroot'
        run_id = cmd[cmd.index('--run-id') + 1]
        band_low = float(cmd[cmd.index('--band-low') + 1])
        band_high = float(cmd[cmd.index('--band-high') + 1])
        subrun_dir = Path(env['BASURIN_RUNS_ROOT']) / run_id
        _write_json(subrun_dir / 'RUN_VALID' / 'verdict.json', {'verdict': 'PASS'})
        if (band_low, band_high) == (150.0, 400.0):
            _write_json(
                subrun_dir / 'pipeline_timeline.json',
                {
                    'event_id': 'GWTEST',
                    'multimode_results': {
                        'multimode_viability_class': 'SINGLEMODE_ONLY',
                        'fallback_path': '220_HAWKING',
                        'support_region_status': 'NO_COMMON_REGION',
                        'support_region_n_final': 0,
                        'downstream_status_class': 'NO_SUPPORT_REGION',
                    },
                },
            )
            _write_json(
                subrun_dir / 's3_ringdown_estimates' / 'outputs' / 'estimates.json',
                {
                    'combined': {'f_hz': 150.0, 'tau_s': 1.0e-4, 'Q': 0.05},
                    'combined_uncertainty': {'sigma_f_hz': 11.77, 'sigma_tau_s': 0.0},
                },
            )
            _write_json(
                subrun_dir / 's4g_mode220_geometry_filter' / 'outputs' / 'mode220_filter.json',
                {'verdict': 'NO_COMMON_GEOMETRIES', 'n_geometries_accepted': 0},
            )
        else:
            _write_json(
                subrun_dir / 'pipeline_timeline.json',
                {
                    'event_id': 'GWTEST',
                    'multimode_results': {
                        'multimode_viability_class': 'MULTIMODE_OK',
                        'fallback_path': None,
                        'support_region_status': 'SUPPORT_REGION_AVAILABLE',
                        'support_region_n_final': 12,
                        'downstream_status_class': 'MULTIMODE_USABLE',
                    },
                },
            )
            _write_json(
                subrun_dir / 's3_ringdown_estimates' / 'outputs' / 'estimates.json',
                {
                    'combined': {'f_hz': 980.0, 'tau_s': 0.003, 'Q': 15.0},
                    'combined_uncertainty': {'sigma_f_hz': 8.0, 'sigma_tau_s': 2.0e-4},
                },
            )
            _write_json(
                subrun_dir / 's4g_mode220_geometry_filter' / 'outputs' / 'mode220_filter.json',
                {'verdict': 'PASS', 'n_geometries_accepted': 40},
            )

        return exp.subprocess.CompletedProcess(cmd, 0, stdout='ok', stderr='')

    monkeypatch.setattr(exp.subprocess, 'run', fake_run)

    rc = exp.main([
        '--run-id', 'band_sweep_host',
        '--event-id', 'GWTEST',
        '--bands', '150-400,800-1200',
        '--atlas-path', str(atlas),
        '--offline',
    ])
    assert rc == 0

    stage_dir = runs_root / 'band_sweep_host' / 'experiment' / 'band_sweep_multimode'
    results = json.loads((stage_dir / 'outputs' / 'band_sweep_results.json').read_text(encoding='utf-8'))
    recommendation = json.loads((stage_dir / 'outputs' / 'recommendation.json').read_text(encoding='utf-8'))
    summary = json.loads((stage_dir / 'stage_summary.json').read_text(encoding='utf-8'))

    assert len(results['results']) == 2
    assert recommendation['overall_conclusion'] == 'FOUND_SUPPORT_REGION'
    assert recommendation['recommended_band']['band_low_hz'] == 800.0
    assert recommendation['recommended_band']['band_high_hz'] == 1200.0
    assert summary['results']['n_bands'] == 2
    assert summary['results']['overall_conclusion'] == 'FOUND_SUPPORT_REGION'
    assert (stage_dir / 'runsroot' / 'GWTEST__band01_150_400').exists()
    assert (stage_dir / 'runsroot' / 'GWTEST__band02_800_1200').exists()


def test_band_sweep_records_failed_subrun_and_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / 'runs'
    monkeypatch.setenv('BASURIN_RUNS_ROOT', str(runs_root))
    atlas = tmp_path / 'atlas.json'
    atlas.write_text('{}', encoding='utf-8')

    def fake_run(cmd: list[str], *, check: bool, capture_output: bool, text: bool, env: dict[str, str], cwd: str):
        return exp.subprocess.CompletedProcess(
            cmd,
            2,
            stdout='[pipeline] ABORT: s2_ringdown_window failed (exit=2) after 00:00\n',
            stderr='',
        )

    monkeypatch.setattr(exp.subprocess, 'run', fake_run)

    rc = exp.main([
        '--run-id', 'band_sweep_fail',
        '--event-id', 'GWFAIL',
        '--bands', '150-400',
        '--atlas-path', str(atlas),
        '--offline',
    ])
    assert rc == 0

    stage_dir = runs_root / 'band_sweep_fail' / 'experiment' / 'band_sweep_multimode'
    results = json.loads((stage_dir / 'outputs' / 'band_sweep_results.json').read_text(encoding='utf-8'))
    recommendation = json.loads((stage_dir / 'outputs' / 'recommendation.json').read_text(encoding='utf-8'))

    row = results['results'][0]
    assert row['pipeline_status'] == 'FAILED_SUBRUN'
    assert row['error_stage'] == 's2_ringdown_window'
    assert row['conclusion_class'] == 'FAILED_SUBRUN'
    assert recommendation['overall_conclusion'] == 'ALL_SUBRUNS_FAILED'
