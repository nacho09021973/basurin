"""s4f_area_observation — Canonical stage: build a per-event area observation
artifact for downstream Hawking-area filtering.

This stage does not attempt a strict inspiral-spin reconstruction.  Instead it
constructs a conservative lower bound on the pre-merger total horizon area from
source-frame component masses only:

    A_initial_lower = 8π (m1_source^2 + m2_source^2)

For each geometry that survives the explicit common-intersection branch, it also
computes a proxy final horizon area from atlas metadata (M_solar, chi):

    A_final = 8π M_final^2 (1 + sqrt(1 - chi^2))

The resulting ``area_obs.json`` is the canonical input for ``s4j``.  If source
masses or atlas metadata are unavailable, the stage still emits a deterministic
artifact with empty ``area_data`` so downstream semantics remain explicit.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / 'basurin_io.py').exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths
from mvp.golden_geometry_spec import VERDICT_PASS, _utc_now_iso
from mvp.gwtc_events import GWTC_CITATION, get_event
from mvp.s4g_mode220_geometry_filter import load_atlas_entries

STAGE = 's4f_area_observation'
S4I_OUTPUT_REL = 's4i_common_geometry_intersection/outputs/common_intersection.json'
RUN_PROVENANCE_REL = 'run_provenance.json'
OUTPUT_FILE = 'area_obs.json'
POLICY = 'mass_only_lower_bound_v1'
OBS_STATUS_AVAILABLE = 'AREA_DATA_AVAILABLE'
OBS_STATUS_NO_COMMON = 'NO_COMMON_GEOMETRIES'
OBS_STATUS_MISSING_SOURCE_MASSES = 'MISSING_SOURCE_MASSES'
OBS_STATUS_MISSING_ATLAS_INFO = 'MISSING_ATLAS_METADATA'
OBS_STATUS_MISSING_EVENT_ID = 'MISSING_EVENT_ID'
VERDICT_NO_AREA_OBSERVATION = 'NO_AREA_OBSERVATION'


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Expected JSON object at {path}, got {type(payload).__name__}')
    return payload


def _extract_event_id(run_provenance: dict[str, Any]) -> str | None:
    invocation = run_provenance.get('invocation')
    if not isinstance(invocation, dict):
        return None
    event_id = invocation.get('event_id')
    if not isinstance(event_id, str) or not event_id.strip():
        return None
    return event_id.strip()


def _extract_common_geometry_ids(payload: dict[str, Any]) -> list[str]:
    value = payload.get('common_geometry_ids')
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _as_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _initial_area_lower_bound(m1_source_msun: float, m2_source_msun: float) -> float:
    return 8.0 * math.pi * (m1_source_msun * m1_source_msun + m2_source_msun * m2_source_msun)


def _kerr_horizon_area_proxy(m_final_msun: float, chi: float) -> float:
    chi_abs = abs(float(chi))
    chi_abs = min(chi_abs, 0.999999999999)
    return 8.0 * math.pi * (m_final_msun * m_final_msun) * (1.0 + math.sqrt(max(0.0, 1.0 - chi_abs * chi_abs)))


def _extract_final_area(entry: dict[str, Any]) -> float | None:
    meta = entry.get('metadata')
    if not isinstance(meta, dict):
        return None
    m_final = _as_positive_float(meta.get('M_solar'))
    try:
        chi = float(meta.get('chi'))
    except (TypeError, ValueError):
        return None
    if m_final is None or not math.isfinite(chi):
        return None
    return _kerr_horizon_area_proxy(m_final, chi)


def _index_atlas_entries(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        gid = entry.get('geometry_id')
        if isinstance(gid, str) and gid:
            out[gid] = entry
    return out


def _relative_to_run(run_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f'MVP {STAGE}: build canonical area observation for s4j')
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--atlas-path', required=True)
    args = ap.parse_args(argv)

    ctx = init_stage(
        args.run_id,
        STAGE,
        params={
            'atlas_path': args.atlas_path,
            'policy': POLICY,
        },
    )

    s4i_path = ctx.run_dir / S4I_OUTPUT_REL
    run_provenance_path = ctx.run_dir / RUN_PROVENANCE_REL
    atlas_path = Path(args.atlas_path)
    if not atlas_path.is_absolute():
        atlas_path = (Path.cwd() / atlas_path).resolve()

    if not atlas_path.exists():
        abort(
            ctx,
            f'Atlas not found. expected: {atlas_path}. '
            'Command to regenerate upstream: provide --atlas-path <ATLAS_PATH>. Candidates detected: <none>',
        )

    try:
        check_inputs(
            ctx,
            {
                's4i_common_intersection': s4i_path,
                'run_provenance': run_provenance_path,
                'atlas': atlas_path,
            },
        )

        s4i_payload = _load_json_object(s4i_path)
        run_provenance = _load_json_object(run_provenance_path)
        event_id = _extract_event_id(run_provenance)
        common_geometry_ids = _extract_common_geometry_ids(s4i_payload)

        observation_status = OBS_STATUS_AVAILABLE
        event_catalog = get_event(event_id) if event_id is not None else None
        m1_source = _as_positive_float((event_catalog or {}).get('m1_source'))
        m2_source = _as_positive_float((event_catalog or {}).get('m2_source'))
        initial_area_lower = (
            _initial_area_lower_bound(m1_source, m2_source)
            if m1_source is not None and m2_source is not None
            else None
        )

        atlas_entries = load_atlas_entries(atlas_path)
        atlas_by_gid = _index_atlas_entries(atlas_entries)
        area_data: dict[str, dict[str, float]] = {}
        n_missing_geometry_ids = 0
        n_missing_final_area_metadata = 0

        if event_id is None:
            observation_status = OBS_STATUS_MISSING_EVENT_ID
        elif not common_geometry_ids:
            observation_status = OBS_STATUS_NO_COMMON
        elif initial_area_lower is None:
            observation_status = OBS_STATUS_MISSING_SOURCE_MASSES
        else:
            for gid in common_geometry_ids:
                entry = atlas_by_gid.get(gid)
                if entry is None:
                    n_missing_geometry_ids += 1
                    continue
                area_final = _extract_final_area(entry)
                if area_final is None:
                    n_missing_final_area_metadata += 1
                    continue
                area_data[gid] = {
                    'area_final': float(area_final),
                    'area_initial': float(initial_area_lower),
                }
            if not area_data:
                observation_status = OBS_STATUS_MISSING_ATLAS_INFO

        payload: dict[str, Any] = {
            'schema_name': 'hawking_area_observation',
            'schema_version': 'v1',
            'created_utc': _utc_now_iso(),
            'run_id': args.run_id,
            'event_id': event_id,
            'stage': STAGE,
            'policy': POLICY,
            'area_kind': 'lower_bound',
            'catalog_citation': GWTC_CITATION,
            'observation_status': observation_status,
            'n_common_input': len(common_geometry_ids),
            'n_area_entries': len(area_data),
            'n_missing_geometry_ids': n_missing_geometry_ids,
            'n_missing_final_area_metadata': n_missing_final_area_metadata,
            'initial_total_area_lower_bound': initial_area_lower,
            'assumptions': {
                'm1_source_msun': m1_source,
                'm2_source_msun': m2_source,
                'spin_information': 'absent_for_initial_area_lower_bound',
                'initial_area_definition': '8*pi*(m1_source^2 + m2_source^2)',
                'final_area_definition': '8*pi*M_final^2*(1 + sqrt(1 - chi^2))',
                'note': 'Mass-only lower-bound Hawking proxy; inspiral component spins are unavailable in the canonical cohort catalog.',
            },
            'area_data': area_data,
            'sources': {
                's4i_common_intersection': _relative_to_run(ctx.run_dir, s4i_path),
                'run_provenance': _relative_to_run(ctx.run_dir, run_provenance_path),
                'atlas_path': str(atlas_path),
            },
            'verdict': VERDICT_PASS if area_data else VERDICT_NO_AREA_OBSERVATION,
        }

        out_path = ctx.outputs_dir / OUTPUT_FILE
        write_json_atomic(out_path, payload)
        finalize(
            ctx,
            artifacts={'area_obs': out_path},
            verdict=VERDICT_PASS,
            results={
                'event_id': event_id,
                'policy': POLICY,
                'observation_status': observation_status,
                'n_common_input': len(common_geometry_ids),
                'n_area_entries': len(area_data),
                'n_missing_geometry_ids': n_missing_geometry_ids,
                'n_missing_final_area_metadata': n_missing_final_area_metadata,
                'source_masses_present': bool(m1_source is not None and m2_source is not None),
            },
        )
        log_stage_paths(ctx)
        print(
            f'[{STAGE}] event_id={event_id} observation_status={observation_status} '
            f'n_common_input={len(common_geometry_ids)} n_area_entries={len(area_data)}'
        )
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
