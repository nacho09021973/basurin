#!/usr/bin/env python3
"""MVP Pipeline Orchestrator: end-to-end ringdown analysis.

Single-event:
    python mvp/pipeline.py single --event-id GW150914 --atlas-path atlas.json \
        [--run-id custom_run_id] [--synthetic]

Multi-event:
    python mvp/pipeline.py multi --events GW150914,GW151226 --atlas-path atlas.json \
        [--min-coverage 1.0] [--synthetic]

What it does:
    1. Creates runs/<run_id>/RUN_VALID/verdict.json
    2. Runs s1_fetch_strain (download or synthetic)
    3. Runs s2_ringdown_window (crop ringdown)
    4. Runs s3_ringdown_estimates (f, tau, Q)
    5. Runs s4_geometry_filter (atlas compatibility)
    6. [multi] Runs s5_aggregate (intersection across events)

Abort semantics:
    If ANY stage exits != 0, the pipeline stops immediately.
    The run's stage_summary.json will have verdict=FAIL.
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from importlib import metadata as importlib_metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- BASURIN import bootstrap ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break

from basurin_io import resolve_out_root, sha256_file, write_json_atomic

MVP_DIR = Path(__file__).resolve().parent
DEFAULT_ATLAS_PATH = Path("docs/ringdown/atlas/atlas_berti_v2.json")
FAMILY_STAGE_MAP: dict[str, tuple[str, str]] = {
    "GR_KERR_BH": ("s8a_family_gr_kerr.py", "s8a_family_gr_kerr"),
    "BNS_REMNANT": ("s8b_family_bns.py", "s8b_family_bns"),
    "LOW_MASS_BH_POSTMERGER": ("s8c_family_low_mass_bh_postmerger.py", "s8c_family_low_mass_bh_postmerger"),
}
KERR_LIKE_FAMILIES = {"GR_KERR_BH", "LOW_MASS_BH_POSTMERGER"}


def _autodetect_losc_hdf5_mappings(event_id: str) -> list[str]:
    """Return local LOSC mappings for the best available detector pair.

    Selection is deterministic per detector:
      1) largest file size
      2) lexicographically largest file name (full path)

    Preferred detector pairs:
      1) H1+L1
      2) H1+V1
      3) L1+V1
    """
    losc_root = Path(os.environ.get("BASURIN_LOSC_ROOT", "data/losc"))
    event_root = losc_root / event_id

    def _pick(det: str) -> Path | None:
        candidates = [
            *event_root.glob(f"*{det}*.h5"),
            *event_root.glob(f"*{det}*.hdf5"),
        ]
        files = [p for p in candidates if p.is_file()]
        if not files:
            return None
        return sorted(files, key=lambda p: (p.stat().st_size, p.name), reverse=True)[0]

    picked = {det: _pick(det) for det in ("H1", "L1", "V1")}
    for pair in (("H1", "L1"), ("H1", "V1"), ("L1", "V1")):
        if all(picked[det] is not None for det in pair):
            return [f"{det}={picked[det].resolve().as_posix()}" for det in pair]
    return []


def _build_s1_fetch_args(
    run_id: str,
    event_id: str,
    duration_s: float,
    synthetic: bool,
    reuse_strain: bool,
    local_hdf5: list[str] | None,
    offline: bool,
) -> list[str]:
    args = ["--run", run_id, "--event-id", event_id, "--duration-s", str(duration_s)]
    if synthetic:
        args.append("--synthetic")
    if reuse_strain:
        args.append("--reuse-if-present")
    effective_local_hdf5 = list(local_hdf5 or [])
    if not effective_local_hdf5:
        effective_local_hdf5 = _autodetect_losc_hdf5_mappings(event_id)
    for mapping in effective_local_hdf5:
        args.extend(["--local-hdf5", mapping])
    if offline:
        args.append("--offline")
    return args


def _build_s0_oracle_args(
    run_id: str,
    event_id: str,
    local_hdf5: list[str] | None,
    offline: bool,
) -> list[str]:
    args = ["--run", run_id, "--event-id", event_id]
    effective_local_hdf5 = list(local_hdf5 or [])
    if offline:
        args.append("--require-offline")
        if not effective_local_hdf5:
            effective_local_hdf5 = _autodetect_losc_hdf5_mappings(event_id)
    for mapping in effective_local_hdf5:
        args.extend(["--local-hdf5", mapping])
    return args


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _generate_run_id(event_id: str) -> str:
    return f"mvp_{event_id}_{_ts()}"


def _require_nonempty_event_id(event_id: str, arg_name: str = "--event-id") -> str:
    normalized = (event_id or "").strip()
    if not normalized:
        raise SystemExit(f"ERROR: {arg_name} cannot be empty")
    return normalized


def _create_run_valid(out_root: Path, run_id: str) -> None:
    rv_dir = out_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(rv_dir / "verdict.json", {
        "verdict": "PASS",
        "created": datetime.now(timezone.utc).isoformat(),
        "note": "MVP pipeline initialization",
    })
    print(f"[pipeline] RUN_VALID created for {run_id}")


def _set_run_valid_verdict(out_root: Path, run_id: str, verdict: str, reason: str) -> None:
    rv_dir = out_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(rv_dir / "verdict.json", {
        "verdict": verdict,
        "created": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "stage": "pipeline",
    })


def _write_timeline(out_root: Path, run_id: str, timeline: dict[str, Any]) -> None:
    write_json_atomic(out_root / run_id / "pipeline_timeline.json", timeline)


def _write_run_provenance(
    out_root: Path,
    run_id: str,
    *,
    mode: str,
    event_id: str | None = None,
    events: list[str] | None = None,
    atlas_path: str | None = None,
    estimator: str | None = None,
    key_params: dict[str, Any] | None = None,
) -> None:
    repo_root = MVP_DIR.parent

    def _git_run(cmd: list[str]) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=repo_root,
                check=False,
            )
        except Exception:
            return None

    git_sha = "UNKNOWN"
    sha_proc = _git_run(["git", "rev-parse", "HEAD"])
    if sha_proc is not None and sha_proc.returncode == 0:
        maybe_sha = sha_proc.stdout.strip()
        if maybe_sha:
            git_sha = maybe_sha

    git_dirty = False
    dirty_proc = _git_run(["git", "status", "--porcelain"])
    if dirty_proc is not None and dirty_proc.returncode == 0:
        git_dirty = bool(dirty_proc.stdout.strip())

    git_branch = "UNKNOWN"
    branch_proc = _git_run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch_proc is not None and branch_proc.returncode == 0:
        maybe_branch = branch_proc.stdout.strip()
        if maybe_branch:
            git_branch = maybe_branch

    atlas_sha: str | None = None
    if atlas_path is not None:
        atlas_file = Path(atlas_path)
        if atlas_file.is_file():
            atlas_sha = sha256_file(atlas_file)

    deps: dict[str, str | None] = {}
    for pkg in ("numpy", "scipy", "qnm"):
        try:
            deps[pkg] = importlib_metadata.version(pkg)
        except Exception:
            deps[pkg] = None

    payload = {
        "schema_version": "run_provenance_v1",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "git_branch": git_branch,
            "python_version": sys.version,
            "platform": platform.platform(),
            "basurin_runs_root": os.environ.get("BASURIN_RUNS_ROOT", ""),
            "basurin_losc_root": os.environ.get("BASURIN_LOSC_ROOT", ""),
        },
        "invocation": {
            "argv": list(sys.argv),
            "mode": mode,
            "event_id": event_id,
            "events": events,
            "atlas_path": str(atlas_path) if atlas_path is not None else None,
            "atlas_sha256": atlas_sha,
            "estimator": estimator,
            "key_params": key_params or {},
        },
        "dependencies": deps,
    }
    write_json_atomic(out_root / run_id / "run_provenance.json", payload)


def _heartbeat(label: str, t0: float, stop: threading.Event, interval: float = 5.0) -> None:
    """Print elapsed time every *interval* seconds while a stage runs."""
    while not stop.wait(interval):
        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        print(f"[pipeline] {label} ... elapsed {mins:02d}:{secs:02d}", flush=True)


def _run_stage(
    script: str,
    args: list[str],
    label: str,
    out_root: Path,
    run_id: str,
    timeline: dict[str, Any],
    stage_timeout_s: float | None = None,
) -> int:
    module = f"mvp.{Path(script).stem}"
    cmd = [sys.executable, "-m", module] + args
    stage_started = datetime.now(timezone.utc).isoformat()
    stage_t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"[pipeline] Stage: {label}")
    print(f"[pipeline] Command: {' '.join(cmd)}")
    if stage_timeout_s is not None:
        print(f"[pipeline] Timeout: {stage_timeout_s:.0f}s")
    print(f"{'=' * 60}", flush=True)

    stop_evt = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(label, stage_t0, stop_evt), daemon=True)
    hb.start()

    timed_out = False
    try:
        proc = subprocess.Popen(cmd)
        try:
            proc.wait(timeout=stage_timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
    finally:
        stop_evt.set()
        hb.join(timeout=2)

    returncode = proc.returncode
    elapsed = time.time() - stage_t0
    mins, secs = divmod(int(elapsed), 60)

    stage_entry = {
        "stage": label,
        "script": script,
        "command": cmd,
        "started_utc": stage_started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": elapsed,
        "returncode": returncode,
        "timed_out": timed_out,
    }
    timeline["stages"].append(stage_entry)
    _write_timeline(out_root, run_id, timeline)

    if timed_out:
        print(
            f"[pipeline] TIMEOUT: {label} killed after {mins:02d}:{secs:02d} "
            f"(limit={stage_timeout_s:.0f}s)",
            file=sys.stderr, flush=True,
        )
    elif returncode != 0:
        print(
            f"[pipeline] ABORT: {label} failed (exit={returncode}) after {mins:02d}:{secs:02d}",
            file=sys.stderr, flush=True,
        )
    else:
        print(f"[pipeline] OK: {label} completed in {mins:02d}:{secs:02d}", flush=True)

    return returncode if not timed_out else 124  # 124 = timeout convention


def _run_preflight_viability(
    out_root: Path,
    run_id: str,
    event_id: str,
    dt_start_s: float,
    window_duration_s: float,
    timeline: dict[str, Any],
) -> dict[str, Any] | None:
    """Run Fisher-based preflight viability check (pure computation, no subprocess).

    Emits preflight_viability.json in runs/<run_id>/preflight_viability/outputs/.
    Returns the preflight result dict, or None if event is not in catalog.
    """
    try:
        from mvp.gwtc_events import get_event
        from mvp.preflight_viability import preflight_viability
    except ImportError:
        print("[pipeline] WARNING: preflight_viability not available, skipping", flush=True)
        return None

    event_params = get_event(event_id)
    if event_params is None:
        print(f"[pipeline] preflight: event {event_id} not in GWTC catalog, skipping preflight", flush=True)
        return None

    m_final = event_params.get("m_final_msun")
    chi_final = event_params.get("chi_final")
    if m_final is None or chi_final is None:
        print(
            f"[pipeline] preflight: event {event_id} has partial catalog entry "
            f"(m_final={m_final}, chi_final={chi_final}), skipping preflight",
            flush=True,
        )
        return None

    rho_ringdown = event_params.get("snr_network", 0.0) * 0.33
    result = preflight_viability(
        event_id=event_id,
        m_final_msun=m_final,
        chi_final=chi_final,
        rho_total=rho_ringdown,
        t0_s=dt_start_s,
        T_s=window_duration_s,
    )

    preflight_dir = out_root / run_id / "preflight_viability" / "outputs"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(preflight_dir / "preflight_viability.json", result)

    verdict = result.get("overall_verdict", "UNKNOWN")
    print(f"[pipeline] preflight viability: {verdict} (event={event_id})", flush=True)
    mode_220 = result.get("modes", {}).get("220", {})
    if mode_220:
        print(
            f"[pipeline] preflight 220: eta={mode_220.get('eta', 0):.6f}, "
            f"rho_eff={mode_220.get('rho_eff', 0):.3f}, "
            f"Q*rho={mode_220.get('Q_x_rho_eff', 0):.3f}, "
            f"rel_iqr_pred={mode_220.get('rel_iqr_predicted', 0):.3f}, "
            f"t0_max={mode_220.get('t0_max_s', 0)*1000:.1f}ms",
            flush=True,
        )
    timeline["preflight_viability"] = {
        "verdict": verdict,
        "t0_max_220_ms": (result.get("recommended_config") or {}).get("t0_max_220_s", 0) * 1000,
    }
    _write_timeline(out_root, run_id, timeline)
    return result


def _run_optional_experiment_t0_sweep(
    out_root: Path,
    run_id: str,
    timeline: dict[str, Any],
    stage_timeout_s: float | None,
) -> float | None:
    def _parse_best_t0_ms(path: Path) -> float | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        best = (
            payload.get("summary", {})
            .get("best_point", {})
            .get("t0_ms")
        )
        if isinstance(best, bool):
            return None
        if isinstance(best, (int, float)):
            best_f = float(best)
            if math.isfinite(best_f) and best_f >= 0.0:
                return best_f
        return None

    label = "experiment_t0_sweep"
    script = "mvp/experiment_t0_sweep.py"
    script_path = MVP_DIR / "experiment_t0_sweep.py"

    stage_started = datetime.now(timezone.utc).isoformat()

    if not script_path.exists():
        timeline["stages"].append({
            "stage": label,
            "label": label,
            "script": script,
            "started_utc": stage_started,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": 0.0,
            "returncode": None,
            "timed_out": False,
            "status": "SKIPPED",
            "message": "missing script",
            "best_effort": True,
            "selected_t0_ms": None,
        })
        _write_timeline(out_root, run_id, timeline)
        print("[pipeline] WARNING: experiment_t0_sweep.py not found, skipping best-effort experiment", flush=True)
        return None

    cmd = [sys.executable, "-m", "mvp.experiment_t0_sweep", "--run", run_id]
    stage_t0 = time.time()
    status = "OK"
    message = "completed"
    timed_out = False
    selected_t0_ms: float | None = None

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=stage_timeout_s,
            check=False,
        )
        rc = proc.returncode
        if rc != 0:
            status = "FAILED"
            stderr = (proc.stderr or "").strip()
            if len(stderr) > 240:
                stderr = stderr[:240] + "..."
            message = f"exit={rc}: {stderr}" if stderr else f"exit={rc}"
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = 124
        status = "FAILED"
        stderr = (exc.stderr or "")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr = stderr.strip()
        if len(stderr) > 240:
            stderr = stderr[:240] + "..."
        message = f"timeout after {stage_timeout_s:.0f}s"
        if stderr:
            message = f"{message}: {stderr}"

    if status == "OK":
        results_path = (
            out_root
            / run_id
            / "experiment"
            / "t0_sweep"
            / "outputs"
            / "t0_sweep_results.json"
        )
        selected_t0_ms = _parse_best_t0_ms(results_path)

    timeline["stages"].append({
        "stage": label,
        "label": label,
        "script": script,
        "started_utc": stage_started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": float(time.time() - stage_t0),
        "returncode": rc,
        "timed_out": timed_out,
        "status": status,
        "message": message,
        "best_effort": True,
        "selected_t0_ms": selected_t0_ms,
    })
    _write_timeline(out_root, run_id, timeline)

    if status != "OK":
        print(f"[pipeline] WARNING: {label} failed (exit={rc}), continuing", flush=True)
    elif selected_t0_ms is not None:
        print(f"[pipeline] {label}: selected_t0_ms={selected_t0_ms}", flush=True)

    return selected_t0_ms


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _as_finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected finite float for {field}, got {value!r}")
    coerced = float(value)
    if not math.isfinite(coerced):
        raise ValueError(f"Expected finite float for {field}, got {value!r}")
    return coerced


def _stabilize_sigma(value: Any, *, scale: float, field: str) -> float:
    sigma = abs(_as_finite_float(value, field=field))
    floor = max(abs(scale) * 1e-6, 1e-12)
    return max(sigma, floor)


def _find_mode_payload(multimode_payload: dict[str, Any], label: str) -> dict[str, Any] | None:
    modes = multimode_payload.get("modes")
    if not isinstance(modes, list):
        return None
    for item in modes:
        if isinstance(item, dict) and str(item.get("label")) == label:
            return item
    return None


def _reasons_indicate_221_unavailable(reasons: list[str]) -> bool:
    reason_blob = " ".join(str(reason).lower() for reason in reasons)
    return "mode_221_ok=false" in reason_blob or "overtone posterior not usable" in reason_blob


def _build_mode220_obs_payload(run_dir: Path) -> dict[str, float]:
    estimates_path = run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    estimates = _load_json_object(estimates_path)
    combined = estimates.get("combined")
    combined_uncertainty = estimates.get("combined_uncertainty")
    if not isinstance(combined, dict):
        raise ValueError(f"Invalid estimates schema in {estimates_path}: missing combined object")
    if not isinstance(combined_uncertainty, dict):
        raise ValueError(
            f"Invalid estimates schema in {estimates_path}: missing combined_uncertainty object"
        )

    obs_f_hz = _as_finite_float(combined.get("f_hz"), field="combined.f_hz")
    obs_tau_s = _as_finite_float(combined.get("tau_s"), field="combined.tau_s")
    sigma_f_hz = _stabilize_sigma(
        combined.get("sigma_f_hz", combined_uncertainty.get("sigma_f_hz")),
        scale=obs_f_hz,
        field="combined_uncertainty.sigma_f_hz",
    )
    sigma_tau_s = _stabilize_sigma(
        combined.get("sigma_tau_s", combined_uncertainty.get("sigma_tau_s")),
        scale=obs_tau_s,
        field="combined_uncertainty.sigma_tau_s",
    )
    return {
        "obs_f_hz": obs_f_hz,
        "obs_tau_s": obs_tau_s,
        "sigma_f_hz": sigma_f_hz,
        "sigma_tau_s": sigma_tau_s,
    }


def _build_mode221_obs_payload(run_dir: Path) -> dict[str, float] | None:
    s3b_summary_path = run_dir / "s3b_multimode_estimates" / "stage_summary.json"
    multimode_path = run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"

    s3b_summary = _load_json_object(s3b_summary_path)
    multimode_payload = _load_json_object(multimode_path)

    viability = s3b_summary.get("multimode_viability")
    reasons: list[str] = []
    if isinstance(viability, dict):
        raw_reasons = viability.get("reasons")
        if isinstance(raw_reasons, list):
            reasons = [str(reason) for reason in raw_reasons]
    if _reasons_indicate_221_unavailable(reasons):
        return None

    mode_221 = _find_mode_payload(multimode_payload, "221")
    if mode_221 is None:
        return None

    try:
        ln_f = _as_finite_float(mode_221.get("ln_f"), field="modes[221].ln_f")
        ln_q = _as_finite_float(mode_221.get("ln_Q"), field="modes[221].ln_Q")
    except ValueError:
        return None

    sigma = mode_221.get("Sigma")
    if not (
        isinstance(sigma, list)
        and len(sigma) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in sigma)
    ):
        return None

    try:
        var_lnf = _as_finite_float(sigma[0][0], field="modes[221].Sigma[0][0]")
        cov_lnf_lnq = _as_finite_float(sigma[0][1], field="modes[221].Sigma[0][1]")
        var_lnq = _as_finite_float(sigma[1][1], field="modes[221].Sigma[1][1]")
    except ValueError:
        return None
    if var_lnf < 0.0 or var_lnq < 0.0:
        return None

    obs_f_hz = math.exp(ln_f)
    q_221 = math.exp(ln_q)
    obs_tau_s = q_221 / (math.pi * obs_f_hz)
    sigma_lnf = math.sqrt(var_lnf)
    sigma_ln_tau = math.sqrt(max(var_lnq + var_lnf - (2.0 * cov_lnf_lnq), 0.0))
    sigma_f_hz = max(obs_f_hz * sigma_lnf, obs_f_hz * 1e-6, 1e-12)
    sigma_tau_s = max(obs_tau_s * sigma_ln_tau, obs_tau_s * 1e-6, 1e-12)
    return {
        "obs_f_hz": obs_f_hz,
        "obs_tau_s": obs_tau_s,
        "sigma_f_hz": sigma_f_hz,
        "sigma_tau_s": sigma_tau_s,
    }


def _prepare_explicit_support_region_inputs(out_root: Path, run_id: str) -> dict[str, Any]:
    run_dir = out_root / run_id
    mode220_input_path = run_dir / "s4g_mode220_geometry_filter" / "inputs" / "mode220_obs.json"
    mode221_input_path = run_dir / "s4h_mode221_geometry_filter" / "inputs" / "mode221_obs.json"

    mode220_payload = _build_mode220_obs_payload(run_dir)
    mode220_input_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(mode220_input_path, mode220_payload)

    mode221_payload = _build_mode221_obs_payload(run_dir)
    if mode221_payload is None:
        mode221_input_path.unlink(missing_ok=True)
    else:
        mode221_input_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(mode221_input_path, mode221_payload)

    print(
        f"[pipeline] explicit support-region inputs ready: "
        f"mode220=present mode221={'present' if mode221_payload is not None else 'skipped'}",
        flush=True,
    )
    return {
        "mode220_input": str(mode220_input_path),
        "mode221_input": str(mode221_input_path) if mode221_payload is not None else None,
    }


def _parse_multimode_results(out_root: Path, run_id: str) -> dict[str, Any]:
    results: dict[str, Any] = {
        "kerr_consistent": None,
        "chi_best": None,
        "d2_min": None,
        "extraction_quality": None,
        "s4c_status": None,
        "kerr_from_multimode_status": None,
        "multimode_viability_class": None,
        "multimode_viability_reasons": [],
        "program_classification": None,
        "fallback_classification": None,
        "fallback_path": None,
        "primary_family": None,
        "families_to_run": [],
        "family_assessments": {},
        "downstream_status_class": None,
        "downstream_status_reasons": [],
        "support_region_status": None,
        "support_region_n_final": None,
        "support_region_analysis_path": None,
    }

    def _coerce_reason_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(v) for v in value]

    s4c_path = out_root / run_id / "s4c_kerr_consistency" / "outputs" / "kerr_consistency.json"
    try:
        payload = json.loads(s4c_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[pipeline] WARNING: cannot parse {s4c_path}: {exc}", flush=True)
    else:
        for key in ("consistent_kerr_95", "kerr_consistent", "consistent"):
            if key in payload:
                results["kerr_consistent"] = payload[key]
                break
        for key in ("chi_best_fit", "chi_best"):
            if key in payload:
                results["chi_best"] = payload[key]
                break
        if "d2_min" in payload:
            results["d2_min"] = payload["d2_min"]
        if payload.get("status") is not None:
            results["s4c_status"] = payload.get("status")
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        source_class = source.get("multimode_viability_class")
        if isinstance(source_class, str):
            results["multimode_viability_class"] = source_class
            results["multimode_viability_reasons"] = _coerce_reason_list(source.get("multimode_viability_reasons"))

    s3b_path = out_root / run_id / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    try:
        payload_s3b = json.loads(s3b_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[pipeline] WARNING: cannot parse {s3b_path}: {exc}", flush=True)
    else:
        if isinstance(payload_s3b.get("results"), dict):
            results["extraction_quality"] = payload_s3b["results"].get("verdict")
        elif payload_s3b.get("extraction_quality") is not None:
            results["extraction_quality"] = payload_s3b.get("extraction_quality")

    s3b_summary_path = out_root / run_id / "s3b_multimode_estimates" / "stage_summary.json"
    if s3b_summary_path.exists():
        try:
            payload_s3b_summary = json.loads(s3b_summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[pipeline] WARNING: cannot parse {s3b_summary_path}: {exc}", flush=True)
        else:
            viability = payload_s3b_summary.get("multimode_viability")
            if isinstance(viability, dict):
                viability_class = viability.get("class")
                if isinstance(viability_class, str):
                    results["multimode_viability_class"] = viability_class
                    results["multimode_viability_reasons"] = _coerce_reason_list(viability.get("reasons"))

    s4d_path = out_root / run_id / "s4d_kerr_from_multimode" / "outputs" / "kerr_from_multimode.json"
    if s4d_path.exists():
        try:
            payload_s4d = json.loads(s4d_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[pipeline] WARNING: cannot parse {s4d_path}: {exc}", flush=True)
        else:
            if payload_s4d.get("status") is not None:
                results["kerr_from_multimode_status"] = payload_s4d.get("status")
            viability = payload_s4d.get("multimode_viability")
            if isinstance(viability, dict):
                viability_class = viability.get("class")
                if isinstance(viability_class, str):
                    results["multimode_viability_class"] = viability_class
                    results["multimode_viability_reasons"] = _coerce_reason_list(viability.get("reasons"))
            fallback = payload_s4d.get("multimode_fallback")
            if isinstance(fallback, dict):
                classification = fallback.get("classification")
                if isinstance(classification, str):
                    results["fallback_classification"] = classification
                fallback_path = fallback.get("fallback_path")
                if isinstance(fallback_path, str):
                    results["fallback_path"] = fallback_path
                program_classification = fallback.get("program_classification")
                if isinstance(program_classification, str):
                    results["program_classification"] = program_classification

    s4k_path = out_root / run_id / "s4k_event_support_region" / "outputs" / "event_support_region.json"
    if s4k_path.exists():
        try:
            payload_s4k = json.loads(s4k_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[pipeline] WARNING: cannot parse {s4k_path}: {exc}", flush=True)
        else:
            downstream = payload_s4k.get("downstream_status")
            if isinstance(downstream, dict):
                downstream_class = downstream.get("class")
                if isinstance(downstream_class, str):
                    results["downstream_status_class"] = downstream_class
                results["downstream_status_reasons"] = _coerce_reason_list(downstream.get("reasons"))
            support_region_status = payload_s4k.get("support_region_status")
            if isinstance(support_region_status, str):
                results["support_region_status"] = support_region_status
            n_final_geometries = payload_s4k.get("n_final_geometries")
            if n_final_geometries is not None:
                results["support_region_n_final"] = n_final_geometries
            analysis_path = payload_s4k.get("analysis_path")
            if isinstance(analysis_path, str):
                results["support_region_analysis_path"] = analysis_path

    router_path = out_root / run_id / "s8_family_router" / "outputs" / "family_router.json"
    if router_path.exists():
        try:
            payload_router = json.loads(router_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[pipeline] WARNING: cannot parse {router_path}: {exc}", flush=True)
        else:
            primary_family = payload_router.get("primary_family")
            if isinstance(primary_family, str):
                results["primary_family"] = primary_family
            families_to_run = payload_router.get("families_to_run")
            if isinstance(families_to_run, list):
                results["families_to_run"] = [str(v) for v in families_to_run]
            program_classification = payload_router.get("program_classification")
            if isinstance(program_classification, str):
                results["program_classification"] = program_classification
            fallback_classification = payload_router.get("fallback_classification")
            if isinstance(fallback_classification, str):
                results["fallback_classification"] = fallback_classification
            fallback_path = payload_router.get("fallback_path")
            if isinstance(fallback_path, str):
                results["fallback_path"] = fallback_path

    ratio_path = out_root / run_id / "s4e_kerr_ratio_filter" / "outputs" / "ratio_filter_result.json"
    if ratio_path.exists():
        try:
            payload_ratio = json.loads(ratio_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[pipeline] WARNING: cannot parse {ratio_path}: {exc}", flush=True)
        else:
            consistency = payload_ratio.get("kerr_consistency")
            diagnostics = payload_ratio.get("diagnostics")
            filtering = payload_ratio.get("filtering")
            if isinstance(consistency, dict):
                results["ratio_rf_consistent"] = consistency.get("Rf_consistent")
            if isinstance(diagnostics, dict):
                results["ratio_informativity_class"] = diagnostics.get("informativity_class")
            if isinstance(filtering, dict):
                results["ratio_n_compatible"] = filtering.get("n_ratio_compatible")

    family_output_map = {
        "GR_KERR_BH": out_root / run_id / "s8a_family_gr_kerr" / "outputs" / "gr_kerr_family.json",
        "BNS_REMNANT": out_root / run_id / "s8b_family_bns" / "outputs" / "bns_family.json",
        "LOW_MASS_BH_POSTMERGER": out_root / run_id / "s8c_family_low_mass_bh_postmerger" / "outputs" / "low_mass_bh_family.json",
    }
    for family, path in family_output_map.items():
        if not path.exists():
            continue
        try:
            payload_family = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[pipeline] WARNING: cannot parse {path}: {exc}", flush=True)
            continue
        results["family_assessments"][family] = {
            "status": payload_family.get("status"),
            "assessment": payload_family.get("assessment"),
            "reason": payload_family.get("reason"),
            "model_status": payload_family.get("model_status"),
        }

    return results


def _load_family_routes(out_root: Path, run_id: str) -> list[str]:
    router_path = out_root / run_id / "s8_family_router" / "outputs" / "family_router.json"
    payload = json.loads(router_path.read_text(encoding="utf-8"))
    families = payload.get("families_to_run")
    if not isinstance(families, list):
        raise ValueError(f"Invalid family router payload in {router_path}: missing families_to_run list")
    out: list[str] = []
    for raw in families:
        family = str(raw)
        if family in FAMILY_STAGE_MAP and family not in out:
            out.append(family)
    if not out:
        raise ValueError(f"Family router selected no supported families in {router_path}")
    return out


def run_single_event(
    event_id: str,
    atlas_path: str,
    run_id: str | None = None,
    synthetic: bool = False,
    duration_s: float = 32.0,
    dt_start_s: float = 0.003,
    window_duration_s: float = 0.06,
    band_low: float = 150.0,
    band_high: float = 400.0,
    epsilon: float = 0.3,
    stage_timeout_s: float | None = None,
    reuse_strain: bool = False,
    with_t0_sweep: bool = False,
    local_hdf5: list[str] | None = None,
    offline: bool = False,
    offline_s2: bool = False,
    t0_catalog: str | None = None,
    estimator: str = "dual",
    psd_path: str | None = None,
) -> tuple[int, str]:
    """Run full pipeline for a single event. Returns (exit_code, run_id)."""
    event_id = _require_nonempty_event_id(event_id, "--event-id")
    out_root = resolve_out_root("runs")

    if run_id is None:
        run_id = _generate_run_id(event_id)

    print(f"\n[pipeline] Starting single-event pipeline")
    print(f"[pipeline] event_id={event_id}, run_id={run_id}")
    print(f"[pipeline] atlas={atlas_path}, synthetic={synthetic}")

    _create_run_valid(out_root, run_id)

    run_started_utc = datetime.now(timezone.utc).isoformat()
    _write_run_provenance(
        out_root,
        run_id,
        mode="single",
        event_id=event_id,
        events=None,
        atlas_path=atlas_path,
        estimator=estimator,
        key_params={
            "epsilon": epsilon,
            "band_low": band_low,
            "band_high": band_high,
            "dt_start_s": dt_start_s,
            "window_duration_s": window_duration_s,
        },
    )

    timeline: dict[str, Any] = {
        "schema_version": "mvp_pipeline_timeline_v1",
        "run_id": run_id,
        "mode": "single",
        "started_utc": run_started_utc,
        "ended_utc": None,
        "event_id": event_id,
        "atlas_path": atlas_path,
        "synthetic": synthetic,
        "estimator": estimator,
        "stages": [],
    }
    _write_timeline(out_root, run_id, timeline)

    # Stage 0: Oracle precheck (deterministic/offline-first)
    s0_args = _build_s0_oracle_args(
        run_id=run_id,
        event_id=event_id,
        local_hdf5=local_hdf5,
        offline=offline,
    )
    rc = _run_stage("s0_oracle_mvp.py", s0_args, "s0_oracle_mvp", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        _set_run_valid_verdict(out_root, run_id, "FAIL", "s0_oracle_mvp precheck failed")
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    # Stage 1: Fetch strain
    s1_args = _build_s1_fetch_args(
        run_id=run_id,
        event_id=event_id,
        duration_s=duration_s,
        synthetic=synthetic,
        reuse_strain=reuse_strain,
        local_hdf5=local_hdf5,
        offline=offline,
    )
    rc = _run_stage("s1_fetch_strain.py", s1_args, "s1_fetch_strain", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    # Stage 2: Ringdown window
    s2_args = [
        "--run", run_id, "--event-id", event_id,
        "--dt-start-s", str(dt_start_s),
        "--duration-s", str(window_duration_s),
    ]
    if offline:
        s2_args.append("--offline")
    if offline_s2:
        s2_args.append("--offline")
    if t0_catalog:
        s2_args.extend(["--window-catalog", str(t0_catalog)])
    rc = _run_stage("s2_ringdown_window.py", s2_args, "s2_ringdown_window", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        _set_run_valid_verdict(out_root, run_id, "FAIL", f"s2_ringdown_window failed: exit={rc}")
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    # Stage 3: Ringdown estimates — select estimator
    s3_args = ["--run", run_id, "--band-low", str(band_low), "--band-high", str(band_high)]
    estimates_path_override = None

    if estimator == "hilbert":
        s3_args = [*s3_args, "--method", "hilbert_envelope"]
        rc = _run_stage(
            "s3_ringdown_estimates.py",
            s3_args + ["--method", "hilbert_envelope"],
            "s3_ringdown_estimates",
            out_root, run_id, timeline, stage_timeout_s,
        )
        if rc != 0:
            timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
            _write_timeline(out_root, run_id, timeline)
            return rc, run_id

    elif estimator == "spectral":
        s3_args = [*s3_args, "--method", "spectral_lorentzian"]
        rc = _run_stage(
            "s3_ringdown_estimates.py",
            s3_args + ["--method", "spectral_lorentzian"],
            "s3_ringdown_estimates",
            out_root, run_id, timeline, stage_timeout_s,
        )
        if rc != 0:
            timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
            _write_timeline(out_root, run_id, timeline)
            return rc, run_id
        # estimates_path_override remains None: canonical path is s3_ringdown_estimates

    elif estimator == "dual":
        # Run both Hilbert and spectral, then dual-method gate
        rc = _run_stage(
            "s3_ringdown_estimates.py",
            s3_args + ["--method", "hilbert_envelope"],
            "s3_ringdown_estimates",
            out_root, run_id, timeline, stage_timeout_s,
        )
        if rc != 0:
            timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
            _write_timeline(out_root, run_id, timeline)
            return rc, run_id

        s3_spectral_args = list(s3_args)
        if psd_path:
            s3_spectral_args.extend(["--psd-path", psd_path])
        rc = _run_stage(
            "s3_spectral_estimates.py", s3_spectral_args, "s3_spectral_estimates",
            out_root, run_id, timeline, stage_timeout_s,
        )
        if rc != 0:
            timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
            _write_timeline(out_root, run_id, timeline)
            return rc, run_id

        rc = _run_stage(
            "experiment_dual_method.py",
            ["--run", run_id, "--band-low", str(band_low), "--band-high", str(band_high)],
            "experiment_dual_method",
            out_root, run_id, timeline, stage_timeout_s,
        )
        if rc != 0:
            print(f"[pipeline] WARNING: dual method gate failed (exit={rc}), continuing",
                  file=sys.stderr, flush=True)

        # Determine which estimates to use based on dual method recommendation
        dual_path = (out_root / run_id / "experiment" / "DUAL_METHOD_V1"
                     / "dual_method_comparison.json")
        recommendation = "spectral"
        if dual_path.exists():
            try:
                with open(dual_path, "r", encoding="utf-8") as f:
                    dual = json.load(f)
                recommendation = dual.get("recommendation", "spectral")
            except Exception:
                pass

        if recommendation == "spectral":
            estimates_path_override = (
                "s3_spectral_estimates/outputs/spectral_estimates.json"
            )
        timeline["dual_method_recommendation"] = recommendation

    else:
        print(f"[pipeline] ERROR: unknown estimator '{estimator}'", file=sys.stderr)
        return 2, run_id

    if with_t0_sweep:
        _run_optional_experiment_t0_sweep(out_root, run_id, timeline, stage_timeout_s)

    # Stage 4: Geometry filter
    s4_args = ["--run", run_id, "--atlas-path", atlas_path, "--epsilon", str(epsilon)]
    if estimates_path_override is not None:
        s4_args.extend(["--estimates-path", estimates_path_override])
    rc = _run_stage("s4_geometry_filter.py", s4_args, "s4_geometry_filter", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    s6_args: list[str] = ["--run", run_id]
    rc = _run_stage("s6_information_geometry.py", s6_args, "s6_information_geometry", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        _set_run_valid_verdict(out_root, run_id, "FAIL", "s6_information_geometry failed")
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    rc = _run_stage(
        "s6b_information_geometry_ranked.py",
        s6_args,
        "s6b_information_geometry_ranked",
        out_root,
        run_id,
        timeline,
        stage_timeout_s,
    )
    if rc != 0:
        _set_run_valid_verdict(out_root, run_id, "FAIL", "s6b_information_geometry_ranked failed")
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
    _write_timeline(out_root, run_id, timeline)

    print(f"\n[pipeline] Single-event pipeline COMPLETE: run_id={run_id}")
    return 0, run_id


def run_multimode_event(
    event_id: str,
    atlas_path: str,
    run_id: str | None = None,
    synthetic: bool = False,
    duration_s: float = 32.0,
    dt_start_s: float = 0.003,
    window_duration_s: float = 0.06,
    band_low: float = 150.0,
    band_high: float = 400.0,
    epsilon: float = 0.3,
    stage_timeout_s: float | None = None,
    reuse_strain: bool = False,
    with_t0_sweep: bool = False,
    local_hdf5: list[str] | None = None,
    s3b_n_bootstrap: int = 200,
    s3b_seed: int = 12345,
    s3b_method: str = "hilbert_peakband",
    estimator: str = "dual",
    offline: bool = False,
    psd_path: str | None = None,
) -> tuple[int, str]:
    event_id = _require_nonempty_event_id(event_id, "--event-id")
    out_root = resolve_out_root("runs")

    if run_id is None:
        run_id = _generate_run_id(event_id)

    print("\n[pipeline] Starting multimode pipeline")
    print(f"[pipeline] event_id={event_id}, run_id={run_id}")
    print(f"[pipeline] atlas={atlas_path}, synthetic={synthetic}")

    _create_run_valid(out_root, run_id)
    run_started_utc = datetime.now(timezone.utc).isoformat()
    _write_run_provenance(
        out_root,
        run_id,
        mode="multimode",
        event_id=event_id,
        events=None,
        atlas_path=atlas_path,
        estimator=estimator,
        key_params={
            "epsilon": epsilon,
            "band_low": band_low,
            "band_high": band_high,
            "dt_start_s": dt_start_s,
            "window_duration_s": window_duration_s,
        },
    )
    timeline: dict[str, Any] = {
        "schema_version": "mvp_pipeline_timeline_v1",
        "run_id": run_id,
        "mode": "multimode",
        "started_utc": run_started_utc,
        "ended_utc": None,
        "event_id": event_id,
        "atlas_path": atlas_path,
        "synthetic": synthetic,
        "stages": [],
        "multimode_results": {
            "kerr_consistent": None,
            "chi_best": None,
            "d2_min": None,
            "extraction_quality": None,
            "s4c_status": None,
            "kerr_from_multimode_status": None,
            "multimode_viability_class": None,
            "multimode_viability_reasons": [],
            "program_classification": None,
            "fallback_classification": None,
            "fallback_path": None,
            "primary_family": None,
            "families_to_run": [],
            "ratio_rf_consistent": None,
            "ratio_informativity_class": None,
            "ratio_n_compatible": None,
            "family_assessments": {},
            "downstream_status_class": None,
            "downstream_status_reasons": [],
            "support_region_status": None,
            "support_region_n_final": None,
            "support_region_analysis_path": None,
        },
    }
    _write_timeline(out_root, run_id, timeline)

    s0_args = _build_s0_oracle_args(
        run_id=run_id,
        event_id=event_id,
        local_hdf5=local_hdf5,
        offline=offline,
    )
    rc = _run_stage("s0_oracle_mvp.py", s0_args, "s0_oracle_mvp", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        _set_run_valid_verdict(out_root, run_id, "FAIL", "s0_oracle_mvp precheck failed")
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    s1_args = _build_s1_fetch_args(
        run_id=run_id,
        event_id=event_id,
        duration_s=duration_s,
        synthetic=synthetic,
        reuse_strain=reuse_strain,
        local_hdf5=local_hdf5,
        offline=offline,
    )
    rc = _run_stage("s1_fetch_strain.py", s1_args, "s1_fetch_strain", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    s2_args = [
        "--run", run_id, "--event-id", event_id,
        "--dt-start-s", str(dt_start_s),
        "--duration-s", str(window_duration_s),
    ]
    if offline:
        s2_args.append("--offline")
    rc = _run_stage("s2_ringdown_window.py", s2_args, "s2_ringdown_window", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        _set_run_valid_verdict(out_root, run_id, "FAIL", f"s2_ringdown_window failed: exit={rc}")
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    # Preflight viability check (Fisher-based, pure computation)
    preflight_result = _run_preflight_viability(
        out_root, run_id, event_id, dt_start_s, window_duration_s, timeline,
    )

    selected_t0_ms: float | None = None
    if with_t0_sweep:
        selected_t0_ms = _run_optional_experiment_t0_sweep(
            out_root, run_id, timeline, stage_timeout_s
        )
        if selected_t0_ms is not None and selected_t0_ms > 0.0:
            selected_dt_start_s = dt_start_s + (selected_t0_ms / 1000.0)
            # Validate selected t0 against preflight domain
            if preflight_result is not None:
                t0_max_s = (preflight_result.get("recommended_config") or {}).get("t0_max_220_s", None)
                if t0_max_s is not None and selected_dt_start_s > t0_max_s:
                    print(
                        f"[pipeline] WARNING: t0_sweep selected dt_start_s={selected_dt_start_s:.4f} "
                        f"exceeds preflight t0_max={t0_max_s:.4f}s for mode 220; "
                        f"run may be non-informative",
                        flush=True,
                    )
            s2_selected_args = [
                "--run", run_id, "--event-id", event_id,
                "--dt-start-s", str(selected_dt_start_s),
                "--duration-s", str(window_duration_s),
            ]
            if offline:
                s2_selected_args.append("--offline")
            rc = _run_stage(
                "s2_ringdown_window.py",
                s2_selected_args,
                "s2_ringdown_window",
                out_root,
                run_id,
                timeline,
                stage_timeout_s,
            )
            if rc != 0:
                _set_run_valid_verdict(
                    out_root,
                    run_id,
                    "FAIL",
                    f"s2_ringdown_window(selected_t0_ms={selected_t0_ms}) failed: exit={rc}",
                )
                timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
                _write_timeline(out_root, run_id, timeline)
                return rc, run_id

    # Stage 3: Ringdown estimates — select estimator
    s3_args = ["--run", run_id, "--band-low", str(band_low), "--band-high", str(band_high)]
    estimates_path_override = None

    if estimator == "hilbert":
        rc = _run_stage(
            "s3_ringdown_estimates.py",
            s3_args + ["--method", "hilbert_envelope"],
            "s3_ringdown_estimates",
            out_root, run_id, timeline, stage_timeout_s,
        )
        if rc != 0:
            timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
            _write_timeline(out_root, run_id, timeline)
            return rc, run_id

    elif estimator == "spectral":
        rc = _run_stage(
            "s3_ringdown_estimates.py",
            s3_args + ["--method", "spectral_lorentzian"],
            "s3_ringdown_estimates",
            out_root, run_id, timeline, stage_timeout_s,
        )
        if rc != 0:
            timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
            _write_timeline(out_root, run_id, timeline)
            return rc, run_id

    elif estimator == "dual":
        rc = _run_stage(
            "s3_ringdown_estimates.py",
            s3_args + ["--method", "hilbert_envelope"],
            "s3_ringdown_estimates",
            out_root, run_id, timeline, stage_timeout_s,
        )
        if rc != 0:
            timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
            _write_timeline(out_root, run_id, timeline)
            return rc, run_id

        s3_spectral_args = list(s3_args)
        if psd_path:
            s3_spectral_args.extend(["--psd-path", psd_path])
        rc = _run_stage(
            "s3_spectral_estimates.py",
            s3_spectral_args,
            "s3_spectral_estimates",
            out_root, run_id, timeline, stage_timeout_s,
        )
        if rc != 0:
            timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
            _write_timeline(out_root, run_id, timeline)
            return rc, run_id

        rc = _run_stage(
            "experiment_dual_method.py",
            ["--run", run_id, "--band-low", str(band_low), "--band-high", str(band_high)],
            "experiment_dual_method",
            out_root, run_id, timeline, stage_timeout_s,
        )
        if rc != 0:
            print(f"[pipeline] WARNING: dual method gate failed (exit={rc}), continuing", file=sys.stderr, flush=True)

        dual_path = out_root / run_id / "experiment" / "DUAL_METHOD_V1" / "dual_method_comparison.json"
        recommendation = "spectral"
        if dual_path.exists():
            try:
                with open(dual_path, "r", encoding="utf-8") as f:
                    dual = json.load(f)
                recommendation = dual.get("recommendation", "spectral")
            except Exception:
                pass

        if recommendation == "spectral":
            estimates_path_override = "s3_spectral_estimates/outputs/spectral_estimates.json"
        timeline["dual_method_recommendation"] = recommendation

    else:
        print(f"[pipeline] ERROR: unknown estimator '{estimator}'", file=sys.stderr)
        return 2, run_id

    s3_estimates_relpath = Path(
        estimates_path_override or "s3_ringdown_estimates/outputs/estimates.json"
    )
    s3_estimates_path = out_root / run_id / s3_estimates_relpath
    s3b_args = [
        "--run-id", run_id,
        "--s3-estimates", str(s3_estimates_path),
        "--n-bootstrap", str(s3b_n_bootstrap),
        "--seed", str(s3b_seed),
        "--method", s3b_method,
    ]
    if psd_path:
        s3b_args.extend(["--psd-path", psd_path])
    rc = _run_stage("s3b_multimode_estimates.py", s3b_args, "s3b_multimode_estimates", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    try:
        _prepare_explicit_support_region_inputs(out_root, run_id)
    except Exception as exc:
        _set_run_valid_verdict(
            out_root,
            run_id,
            "FAIL",
            f"explicit support-region input preparation failed: {exc}",
        )
        print(
            f"[pipeline] ABORT: cannot prepare explicit support-region inputs: {exc}",
            file=sys.stderr,
            flush=True,
        )
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return 2, run_id

    explicit_stage_specs = [
        ("s4g_mode220_geometry_filter.py", ["--run-id", run_id, "--atlas-path", atlas_path], "s4g_mode220_geometry_filter"),
        ("s4h_mode221_geometry_filter.py", ["--run-id", run_id, "--atlas-path", atlas_path], "s4h_mode221_geometry_filter"),
        ("s4i_common_geometry_intersection.py", ["--run-id", run_id], "s4i_common_geometry_intersection"),
        ("s4f_area_observation.py", ["--run-id", run_id, "--atlas-path", atlas_path], "s4f_area_observation"),
        ("s4j_hawking_area_filter.py", ["--run-id", run_id], "s4j_hawking_area_filter"),
    ]
    for script, args, label in explicit_stage_specs:
        rc = _run_stage(script, args, label, out_root, run_id, timeline, stage_timeout_s)
        if rc != 0:
            timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
            _write_timeline(out_root, run_id, timeline)
            return rc, run_id

    s4_args = ["--run", run_id, "--atlas-path", atlas_path, "--epsilon", str(epsilon)]
    rc = _run_stage("s4_geometry_filter.py", s4_args, "s4_geometry_filter", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    s4c_args = ["--run-id", run_id, "--atlas-path", atlas_path]
    rc = _run_stage("s4c_kerr_consistency.py", s4c_args, "s4c_kerr_consistency", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    # Phase B: Kerr inference from multimode (canonical; does not replace s4/s4c)
    s4d_args = ["--run-id", run_id]
    rc = _run_stage("s4d_kerr_from_multimode.py", s4d_args, "s4d_kerr_from_multimode", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    s4k_args = ["--run-id", run_id]
    rc = _run_stage("s4k_event_support_region.py", s4k_args, "s4k_event_support_region", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    s7_args = ["--run-id", run_id]
    rc = _run_stage("s7_beyond_kerr_deviation_score.py", s7_args, "s7_beyond_kerr_deviation_score", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    s8_router_args = ["--run-id", run_id]
    rc = _run_stage("s8_family_router.py", s8_router_args, "s8_family_router", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    try:
        families_to_run = _load_family_routes(out_root, run_id)
    except Exception as exc:
        print(f"[pipeline] ABORT: cannot load family routes: {exc}", file=sys.stderr, flush=True)
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return 2, run_id

    if any(family in KERR_LIKE_FAMILIES for family in families_to_run):
        s4e_args = ["--run-id", run_id]
        rc = _run_stage("s4e_kerr_ratio_filter.py", s4e_args, "s4e_kerr_ratio_filter", out_root, run_id, timeline, stage_timeout_s)
        if rc != 0:
            timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
            _write_timeline(out_root, run_id, timeline)
            return rc, run_id

    for family in families_to_run:
        script, label = FAMILY_STAGE_MAP[family]
        rc = _run_stage(script, ["--run-id", run_id], label, out_root, run_id, timeline, stage_timeout_s)
        if rc != 0:
            timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
            _write_timeline(out_root, run_id, timeline)
            return rc, run_id

    timeline["multimode_results"] = _parse_multimode_results(out_root, run_id)
    timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
    _write_timeline(out_root, run_id, timeline)
    return 0, run_id


def run_multi_event(
    events: list[str],
    atlas_path: str,
    agg_run_id: str | None = None,
    min_coverage: float = 1.0,
    catalog_path: str | None = None,
    abort_on_event_fail: bool = True,
    **kwargs: Any,
) -> tuple[int, str]:
    """Run pipeline for multiple events, then aggregate.

    Args:
        abort_on_event_fail: If False (batch mode), continue on event failure.
        catalog_path: Optional path to GWTC catalog JSON for deviation analysis.
    """
    if agg_run_id is None:
        agg_run_id = f"mvp_aggregate_{_ts()}"

    out_root = resolve_out_root("runs")
    estimator = kwargs.get("estimator", "dual")

    events = [_require_nonempty_event_id(e, "--events") for e in events]

    print(f"\n[pipeline] Starting multi-event pipeline")
    print(f"[pipeline] events={events}")
    print(f"[pipeline] aggregate_run={agg_run_id}")
    print(f"[pipeline] estimator={estimator}")

    _create_run_valid(out_root, agg_run_id)
    run_started_utc = datetime.now(timezone.utc).isoformat()
    _write_run_provenance(
        out_root,
        agg_run_id,
        mode="multi",
        event_id=None,
        events=events,
        atlas_path=atlas_path,
        estimator=estimator,
        key_params={
            "epsilon": kwargs.get("epsilon", 0.3),
            "band_low": kwargs.get("band_low", 150.0),
            "band_high": kwargs.get("band_high", 400.0),
            "dt_start_s": kwargs.get("dt_start_s", 0.003),
            "window_duration_s": kwargs.get("window_duration_s", 0.06),
        },
    )

    timeline: dict[str, Any] = {
        "schema_version": "mvp_pipeline_timeline_v1",
        "run_id": agg_run_id,
        "mode": "multi",
        "started_utc": run_started_utc,
        "ended_utc": None,
        "events": events,
        "atlas_path": atlas_path,
        "synthetic": bool(kwargs.get("synthetic", False)),
        "estimator": estimator,
        "stages": [],
    }
    _write_timeline(out_root, agg_run_id, timeline)

    per_event_runs: list[str] = []
    failed_events: list[str] = []

    for event_id in events:
        rc, run_id = run_single_event(event_id=event_id, atlas_path=atlas_path, **kwargs)
        if rc != 0:
            if abort_on_event_fail:
                print(f"[pipeline] ABORT: event {event_id} failed", file=sys.stderr)
                timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
                _write_timeline(out_root, agg_run_id, timeline)
                return rc, agg_run_id
            else:
                print(f"[pipeline] WARNING: event {event_id} failed (exit={rc}), continuing",
                      file=sys.stderr, flush=True)
                failed_events.append(event_id)
                continue
        per_event_runs.append(run_id)

    if not per_event_runs:
        print("[pipeline] ERROR: no events succeeded", file=sys.stderr)
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, agg_run_id, timeline)
        return 2, agg_run_id

    if failed_events:
        print(f"[pipeline] WARNING: {len(failed_events)} events failed: {failed_events}",
              flush=True)
        timeline["failed_events"] = failed_events

    # Stage 5: Aggregate
    s5_args = [
        "--out-run", agg_run_id,
        "--source-runs", ",".join(per_event_runs),
        "--min-coverage", str(min_coverage),
    ]
    if catalog_path:
        s5_args.extend(["--catalog-path", catalog_path])
    rc = _run_stage("s5_aggregate.py", s5_args, "s5_aggregate", out_root, agg_run_id, timeline, kwargs.get("stage_timeout_s"))
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, agg_run_id, timeline)
        return rc, agg_run_id

    timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
    _write_timeline(out_root, agg_run_id, timeline)

    print(f"\n[pipeline] Multi-event pipeline COMPLETE: agg_run={agg_run_id}")
    print(f"[pipeline] Per-event runs: {per_event_runs}")
    return 0, agg_run_id


def _resolve_atlas_path(atlas_path: str | None, atlas_default: bool) -> str:
    if atlas_path:
        return atlas_path

    if atlas_default:
        if not DEFAULT_ATLAS_PATH.exists():
            print(
                "Atlas not found. Generate it by running: "
                "python mvp/generate_atlas_from_fits.py",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return str(DEFAULT_ATLAS_PATH)

    raise SystemExit("--atlas-path is required unless --atlas-default is set")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="MVP Pipeline Orchestrator")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Single event
    sp_single = sub.add_parser("single", help="Run pipeline for one event")
    sp_single.add_argument("--event-id", required=True)
    sp_single.add_argument("--atlas-path", default=None)
    sp_single.add_argument("--atlas-default", action="store_true", default=False)
    sp_single.add_argument("--run-id", default=None)
    sp_single.add_argument("--synthetic", action="store_true")
    sp_single.add_argument("--duration-s", type=float, default=32.0)
    sp_single.add_argument("--dt-start-s", type=float, default=0.003)
    sp_single.add_argument("--window-duration-s", type=float, default=0.06)
    sp_single.add_argument("--band-low", type=float, default=150.0)
    sp_single.add_argument("--band-high", type=float, default=400.0)
    sp_single.add_argument("--epsilon", type=float, default=0.3)
    sp_single.add_argument(
        "--stage-timeout-s", type=float, default=None,
        help="Kill a stage if it exceeds this many seconds (default: no limit)",
    )
    sp_single.add_argument(
        "--reuse-strain", action="store_true", default=False,
        help="Skip s1 download if outputs already exist and params match",
    )
    sp_single.add_argument("--with-t0-sweep", action="store_true", default=False)
    sp_single.add_argument("--psd-path", default=None, help="Path to measured_psd.json for spectral whitening")
    sp_single.add_argument(
        "--local-hdf5",
        action="append",
        default=[],
        metavar="DET=PATH",
        help="Forward local HDF5 detector mapping(s) to s1_fetch_strain (repeatable)",
    )
    sp_single.add_argument("--offline", action="store_true", default=False)
    sp_single.add_argument(
        "--offline-s2",
        action="store_true",
        default=False,
        help="Pass --offline only to s2_ringdown_window",
    )
    sp_single.add_argument(
        "--window-catalog",
        default=None,
        metavar="PATH",
        help="Preferred alias: pass PATH as --window-catalog to s2_ringdown_window",
    )
    sp_single.add_argument(
        "--t0-catalog",
        default=None,
        metavar="PATH",
        help="Alias supported for compatibility; pass PATH as --window-catalog to s2_ringdown_window",
    )
    sp_single.add_argument(
        "--estimator", choices=["hilbert", "spectral", "dual"], default="dual",
        help="Estimator to use for s3: dual (default), spectral, or hilbert (legacy)",
    )
    sp_single.add_argument(
        "--psd-path", default=None, metavar="PATH",
        help="Path to measured_psd.json; enables whitening in s3_spectral_estimates and s3b_multimode_estimates",
    )

    # Multi event
    sp_multi = sub.add_parser("multi", help="Run pipeline for multiple events + aggregate")
    sp_multi.add_argument("--events", required=True, help="Comma-separated event IDs")
    sp_multi.add_argument("--atlas-path", default=None)
    sp_multi.add_argument("--atlas-default", action="store_true", default=False)
    sp_multi.add_argument("--agg-run-id", default=None)
    sp_multi.add_argument("--min-coverage", type=float, default=1.0)
    sp_multi.add_argument("--synthetic", action="store_true")
    sp_multi.add_argument("--duration-s", type=float, default=32.0)
    sp_multi.add_argument("--dt-start-s", type=float, default=0.003)
    sp_multi.add_argument("--window-duration-s", type=float, default=0.06)
    sp_multi.add_argument("--band-low", type=float, default=150.0)
    sp_multi.add_argument("--band-high", type=float, default=400.0)
    sp_multi.add_argument("--epsilon", type=float, default=0.3)
    sp_multi.add_argument(
        "--stage-timeout-s", type=float, default=None,
        help="Kill a stage if it exceeds this many seconds (default: no limit)",
    )
    sp_multi.add_argument(
        "--reuse-strain", action="store_true", default=False,
        help="Skip s1 download if outputs already exist and params match",
    )
    sp_multi.add_argument("--with-t0-sweep", action="store_true", default=False)
    sp_multi.add_argument(
        "--local-hdf5",
        action="append",
        default=[],
        metavar="DET=PATH",
        help="Forward local HDF5 detector mapping(s) to per-event s1_fetch_strain (repeatable)",
    )
    sp_multi.add_argument("--offline", action="store_true", default=False)
    sp_multi.add_argument(
        "--estimator", choices=["hilbert", "spectral", "dual"], default="dual",
        help="Estimator for s3 (dual/spectral/hilbert)",
    )
    sp_multi.add_argument(
        "--catalog-path", default=None,
        help="Optional GWTC catalog JSON for deviation analysis in s5",
    )
    sp_multi.add_argument("--psd-path", default=None, help="Path to measured_psd.json for spectral whitening")

    # Single event multimode
    sp_multimode = sub.add_parser("multimode", help="Run single-event multimode pipeline")
    sp_multimode.add_argument("--event-id", required=True)
    sp_multimode.add_argument("--atlas-path", default=None)
    sp_multimode.add_argument("--atlas-default", action="store_true", default=False)
    sp_multimode.add_argument("--run-id", default=None)
    sp_multimode.add_argument("--synthetic", action="store_true")
    sp_multimode.add_argument("--duration-s", type=float, default=32.0)
    sp_multimode.add_argument("--dt-start-s", type=float, default=0.003)
    sp_multimode.add_argument("--window-duration-s", type=float, default=0.06)
    sp_multimode.add_argument("--band-low", type=float, default=150.0)
    sp_multimode.add_argument("--band-high", type=float, default=400.0)
    sp_multimode.add_argument("--epsilon", type=float, default=0.3)
    sp_multimode.add_argument(
        "--stage-timeout-s", type=float, default=None,
        help="Kill a stage if it exceeds this many seconds (default: no limit)",
    )
    sp_multimode.add_argument(
        "--reuse-strain", action="store_true", default=False,
        help="Skip s1 download if outputs already exist and params match",
    )
    sp_multimode.add_argument("--with-t0-sweep", action="store_true", default=False)
    sp_multimode.add_argument(
        "--estimator", choices=["hilbert", "spectral", "dual"], default="dual",
        help="Estimator for s3 (dual/spectral/hilbert)",
    )
    sp_multimode.add_argument("--s3b-n-bootstrap", type=int, default=200)
    sp_multimode.add_argument("--s3b-seed", type=int, default=12345)
    sp_multimode.add_argument(
        "--s3b-method",
        choices=["hilbert_peakband", "spectral_two_pass"],
        default="hilbert_peakband",
    )
    sp_multimode.add_argument(
        "--local-hdf5",
        action="append",
        default=[],
        metavar="DET=PATH",
        help="Forward local HDF5 detector mapping(s) to s1_fetch_strain (repeatable)",
    )
    sp_multimode.add_argument("--offline", action="store_true", default=False)
    sp_multimode.add_argument("--psd-path", default=None, help="Path to measured_psd.json for spectral whitening")

    # Batch: multi-event GWTC pipeline (continue-on-failure mode)
    sp_batch = sub.add_parser(
        "batch",
        help="Run pipeline for multiple events in batch mode (continue on failure)",
    )
    sp_batch.add_argument("--events", required=True,
                          help="Comma-separated event IDs (e.g. GW150914,GW151226)")
    sp_batch.add_argument("--atlas-path", default=None)
    sp_batch.add_argument("--atlas-default", action="store_true", default=False)
    sp_batch.add_argument("--agg-run-id", default=None)
    sp_batch.add_argument("--min-coverage", type=float, default=1.0)
    sp_batch.add_argument("--synthetic", action="store_true")
    sp_batch.add_argument("--duration-s", type=float, default=32.0)
    sp_batch.add_argument("--dt-start-s", type=float, default=0.003)
    sp_batch.add_argument("--window-duration-s", type=float, default=0.06)
    sp_batch.add_argument("--band-low", type=float, default=150.0)
    sp_batch.add_argument("--band-high", type=float, default=400.0)
    sp_batch.add_argument("--epsilon", type=float, default=0.3)
    sp_batch.add_argument("--stage-timeout-s", type=float, default=None)
    sp_batch.add_argument("--reuse-strain", action="store_true", default=False)
    sp_batch.add_argument(
        "--estimator", choices=["hilbert", "spectral", "dual"], default="dual",
        help="Estimator for s3 (dual/spectral/hilbert)",
    )
    sp_batch.add_argument(
        "--catalog-path", default=None,
        help="Optional GWTC catalog JSON for deviation analysis in s5",
    )
    sp_batch.add_argument("--offline", action="store_true", default=False)
    sp_batch.add_argument("--psd-path", default=None, help="Path to measured_psd.json for spectral whitening")

    args = parser.parse_args()

    if args.mode == "single":
        window_catalog = args.window_catalog if args.window_catalog is not None else args.t0_catalog
        atlas_path = _resolve_atlas_path(args.atlas_path, args.atlas_default)
        event_id = _require_nonempty_event_id(args.event_id, "--event-id")
        rc, run_id = run_single_event(
            event_id=event_id,
            atlas_path=atlas_path,
            run_id=args.run_id,
            synthetic=args.synthetic,
            duration_s=args.duration_s,
            dt_start_s=args.dt_start_s,
            window_duration_s=args.window_duration_s,
            band_low=args.band_low,
            band_high=args.band_high,
            epsilon=args.epsilon,
            stage_timeout_s=args.stage_timeout_s,
            reuse_strain=args.reuse_strain,
            with_t0_sweep=args.with_t0_sweep,
            local_hdf5=args.local_hdf5,
            offline=args.offline,
            offline_s2=args.offline_s2,
            t0_catalog=window_catalog,
            estimator=args.estimator,
            psd_path=args.psd_path,
        )
        return rc

    elif args.mode == "multi":
        atlas_path = _resolve_atlas_path(args.atlas_path, args.atlas_default)
        events = [_require_nonempty_event_id(e, "--events") for e in args.events.split(",") if e.strip()]
        rc, agg_run_id = run_multi_event(
            events=events,
            atlas_path=atlas_path,
            agg_run_id=args.agg_run_id,
            min_coverage=args.min_coverage,
            synthetic=args.synthetic,
            duration_s=args.duration_s,
            dt_start_s=args.dt_start_s,
            window_duration_s=args.window_duration_s,
            band_low=args.band_low,
            band_high=args.band_high,
            epsilon=args.epsilon,
            stage_timeout_s=args.stage_timeout_s,
            reuse_strain=args.reuse_strain,
            with_t0_sweep=args.with_t0_sweep,
            local_hdf5=args.local_hdf5,
            offline=args.offline,
            estimator=args.estimator,
            catalog_path=args.catalog_path,
            psd_path=args.psd_path,
        )
        return rc

    elif args.mode == "batch":
        atlas_path = _resolve_atlas_path(args.atlas_path, args.atlas_default)
        events = [_require_nonempty_event_id(e, "--events") for e in args.events.split(",") if e.strip()]
        rc, agg_run_id = run_multi_event(
            events=events,
            atlas_path=atlas_path,
            agg_run_id=args.agg_run_id,
            min_coverage=args.min_coverage,
            synthetic=args.synthetic,
            duration_s=args.duration_s,
            dt_start_s=args.dt_start_s,
            window_duration_s=args.window_duration_s,
            band_low=args.band_low,
            band_high=args.band_high,
            epsilon=args.epsilon,
            stage_timeout_s=args.stage_timeout_s,
            reuse_strain=args.reuse_strain,
            offline=args.offline,
            estimator=args.estimator,
            catalog_path=args.catalog_path,
            abort_on_event_fail=False,
            psd_path=args.psd_path,
        )
        return rc

    elif args.mode == "multimode":
        atlas_path = _resolve_atlas_path(args.atlas_path, args.atlas_default)
        event_id = _require_nonempty_event_id(args.event_id, "--event-id")
        rc, run_id = run_multimode_event(
            event_id=event_id,
            atlas_path=atlas_path,
            run_id=args.run_id,
            synthetic=args.synthetic,
            duration_s=args.duration_s,
            dt_start_s=args.dt_start_s,
            window_duration_s=args.window_duration_s,
            band_low=args.band_low,
            band_high=args.band_high,
            epsilon=args.epsilon,
            stage_timeout_s=args.stage_timeout_s,
            reuse_strain=args.reuse_strain,
            with_t0_sweep=args.with_t0_sweep,
            s3b_n_bootstrap=args.s3b_n_bootstrap,
            s3b_seed=args.s3b_seed,
            s3b_method=args.s3b_method,
            estimator=args.estimator,
            local_hdf5=args.local_hdf5,
            offline=args.offline,
            psd_path=args.psd_path,
        )
        return rc

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
