#!/usr/bin/env python3
"""Post-hoc, auditable check for a candidate Kerr (2,2,1) overtone."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import (
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)
from mvp import contracts
from mvp.experiment_t0_sweep_221 import _parse_grid as _parse_t0_grid
from mvp.experiment_t0_sweep_221 import _run_one_point as _run_t0_point
from mvp.kerr_qnm_fits import kerr_qnm

EXPERIMENT_NAME = "qnm_221_literature_check"
EXPERIMENT_STAGE = f"experiment/{EXPERIMENT_NAME}"
MODE_LABEL = "221"
MODE_TUPLE = (2, 2, 1)
MODEL_METRIC_NAME = "delta_bic"
ALLOWED_VERDICTS = (
    "KERR_COMPATIBLE",
    "WEAK_EVIDENCE",
    "REJECTED",
    "INSUFFICIENT_DATA",
)
GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_WEAK = "WEAK"
GATE_NOT_AVAILABLE = "NOT_AVAILABLE"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Auditable post-hoc 221 literature/Kerr compatibility check")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--event-id", default=None)
    ap.add_argument("--t0-grid-ms", default="0,2,4,6,8")
    ap.add_argument("--f-tol-rel", type=float, default=0.10)
    ap.add_argument("--tau-tol-rel", type=float, default=0.25)
    ap.add_argument("--t0-stability-f-rel-tol", type=float, default=0.10)
    ap.add_argument("--t0-stability-tau-rel-tol", type=float, default=0.25)
    ap.add_argument("--stable-t0-min-fraction", type=float, default=0.60)
    ap.add_argument("--stable-t0-reject-fraction", type=float, default=0.25)
    return ap.parse_args(argv)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _rel_to(base: Path, path: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _record_input(run_dir: Path, records: list[dict[str, str]], label: str, path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    records.append(
        {
            "label": label,
            "path": _rel_to(run_dir, path),
            "sha256": sha256_file(path),
        }
    )


def _find_first_event_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        raw = payload.get("event_id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        invocation = payload.get("invocation")
        if isinstance(invocation, dict):
            raw = invocation.get("event_id")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        for value in payload.values():
            found = _find_first_event_id(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_first_event_id(value)
            if found:
                return found
    return None


def _resolve_event_id(run_dir: Path, explicit_event_id: str | None, input_records: list[dict[str, str]]) -> tuple[str | None, str | None, list[str]]:
    if explicit_event_id is not None and explicit_event_id.strip():
        return explicit_event_id.strip(), "cli", []

    candidates = [
        run_dir / "run_provenance.json",
        run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        run_dir / "s1_fetch_strain" / "outputs" / "provenance.json",
    ]
    found_paths: list[str] = []
    for path in candidates:
        found_paths.append(_rel_to(run_dir, path))
        if not path.exists():
            continue
        _record_input(run_dir, input_records, "event_id_source", path)
        try:
            payload = _load_json_object(path)
        except Exception:
            continue
        event_id = _find_first_event_id(payload)
        if event_id:
            return event_id, _rel_to(run_dir, path), found_paths
    return None, None, found_paths


def _find_mode_payload(multimode: dict[str, Any], label: str) -> dict[str, Any] | None:
    modes = multimode.get("modes")
    if not isinstance(modes, list):
        return None
    for mode in modes:
        if isinstance(mode, dict) and str(mode.get("label")) == label:
            return mode
    return None


def _extract_mode_estimate(multimode_path: Path) -> dict[str, Any]:
    result = {
        "status": GATE_NOT_AVAILABLE,
        "reason": f"missing required input: {multimode_path}",
        "f_hz": None,
        "tau_s": None,
        "sigma_f_hz": None,
        "sigma_tau_s": None,
        "valid_fraction": None,
        "selection_policy": None,
    }
    if not multimode_path.exists():
        return result

    try:
        payload = _load_json_object(multimode_path)
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result

    mode = _find_mode_payload(payload, MODE_LABEL)
    if mode is None:
        flags = payload.get("results", {}).get("quality_flags") if isinstance(payload.get("results"), dict) else []
        result["reason"] = f"mode {MODE_LABEL} missing; quality_flags={flags}"
        return result

    fit = mode.get("fit") if isinstance(mode.get("fit"), dict) else {}
    stability = fit.get("stability") if isinstance(fit.get("stability"), dict) else {}
    lnf_p10 = _safe_float(stability.get("lnf_p10"))
    lnf_p50 = _safe_float(stability.get("lnf_p50"))
    lnf_p90 = _safe_float(stability.get("lnf_p90"))
    lnq_p10 = _safe_float(stability.get("lnQ_p10"))
    lnq_p50 = _safe_float(stability.get("lnQ_p50"))
    lnq_p90 = _safe_float(stability.get("lnQ_p90"))
    valid_fraction = _safe_float(stability.get("valid_fraction"))

    if None not in (lnf_p10, lnf_p50, lnf_p90, lnq_p10, lnq_p50, lnq_p90):
        f_p10 = math.exp(float(lnf_p10))
        f_p50 = math.exp(float(lnf_p50))
        f_p90 = math.exp(float(lnf_p90))
        tau_p10 = math.exp(float(lnq_p10) - float(lnf_p10)) / math.pi
        tau_p50 = math.exp(float(lnq_p50) - float(lnf_p50)) / math.pi
        tau_p90 = math.exp(float(lnq_p90) - float(lnf_p90)) / math.pi
        return {
            "status": GATE_PASS,
            "reason": None,
            "f_hz": f_p50,
            "tau_s": tau_p50,
            "sigma_f_hz": max((f_p90 - f_p10) / 2.0, 0.0),
            "sigma_tau_s": max((tau_p90 - tau_p10) / 2.0, 0.0),
            "valid_fraction": valid_fraction,
            "selection_policy": "mode_221.fit.stability.p50",
        }

    ln_f = _safe_float(mode.get("ln_f"))
    ln_q = _safe_float(mode.get("ln_Q"))
    if ln_f is None or ln_q is None:
        result["reason"] = f"mode {MODE_LABEL} lacks usable ln_f/ln_Q estimates"
        return result

    f_hz = math.exp(ln_f)
    tau_s = math.exp(ln_q - ln_f) / math.pi
    sigma_f_hz: float | None = None
    sigma_tau_s: float | None = None
    sigma = mode.get("Sigma")
    if (
        isinstance(sigma, list)
        and len(sigma) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in sigma)
    ):
        var_lnf = _safe_float(sigma[0][0])
        cov_lnf_lnq = _safe_float(sigma[0][1])
        var_lnq = _safe_float(sigma[1][1])
        if None not in (var_lnf, cov_lnf_lnq, var_lnq) and float(var_lnf) >= 0.0 and float(var_lnq) >= 0.0:
            sigma_f_hz = max(f_hz * math.sqrt(float(var_lnf)), 0.0)
            sigma_ln_tau = max(float(var_lnq) + float(var_lnf) - (2.0 * float(cov_lnf_lnq)), 0.0)
            sigma_tau_s = max(tau_s * math.sqrt(sigma_ln_tau), 0.0)

    return {
        "status": GATE_PASS,
        "reason": None,
        "f_hz": f_hz,
        "tau_s": tau_s,
        "sigma_f_hz": sigma_f_hz,
        "sigma_tau_s": sigma_tau_s,
        "valid_fraction": valid_fraction,
        "selection_policy": "mode_221.point_estimate",
    }


def _load_kerr_remnant(kerr_path: Path) -> dict[str, Any]:
    result = {
        "status": GATE_NOT_AVAILABLE,
        "reason": f"missing required input: {kerr_path}",
        "M_final_Msun": None,
        "chi_final": None,
        "kerr_verdict": None,
        "domain_status": None,
        "payload_reason": None,
    }
    if not kerr_path.exists():
        return result
    try:
        payload = _load_json_object(kerr_path)
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result

    M_final = _safe_float(payload.get("M_final_Msun"))
    chi_final = _safe_float(payload.get("chi_final"))
    result["kerr_verdict"] = payload.get("verdict")
    result["domain_status"] = payload.get("domain_status")
    result["payload_reason"] = payload.get("reason")
    if M_final is None or chi_final is None or M_final <= 0.0 or not (0.0 <= chi_final < 1.0):
        result["reason"] = (
            f"non-usable Kerr remnant in {kerr_path}: "
            f"verdict={payload.get('verdict')!r} domain_status={payload.get('domain_status')!r} "
            f"reason={payload.get('reason')!r}"
        )
        return result
    result.update(
        {
            "status": GATE_PASS,
            "reason": None,
            "M_final_Msun": M_final,
            "chi_final": chi_final,
        }
    )
    return result


def _rel_err(measured: float | None, reference: float | None) -> float | None:
    if measured is None or reference is None:
        return None
    scale = max(abs(reference), 1.0e-12)
    return abs(float(measured) - float(reference)) / scale


def _load_model_selection(run_dir: Path, model_path: Path) -> dict[str, Any]:
    row = {
        "source_path": _rel_to(run_dir, model_path),
        "metric_name": MODEL_METRIC_NAME,
        "metric_value": None,
        "metric_threshold": None,
        "two_mode_preferred": None,
        "status": GATE_NOT_AVAILABLE,
        "reason": f"missing required input: {model_path}",
        "bic_1mode": None,
        "bic_2mode": None,
        "rss_1mode": None,
        "rss_2mode": None,
    }
    if not model_path.exists():
        return row

    try:
        payload = _load_json_object(model_path)
    except Exception as exc:
        row["reason"] = f"{type(exc).__name__}: {exc}"
        return row

    row["metric_value"] = _safe_float(payload.get(MODEL_METRIC_NAME))
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    two_mode_preferred = decision.get("two_mode_preferred")
    row["metric_threshold"] = _safe_float(thresholds.get("two_mode_preferred_delta_bic"))
    row["two_mode_preferred"] = two_mode_preferred if isinstance(two_mode_preferred, bool) else None
    row["bic_1mode"] = _safe_float(payload.get("bic_1mode"))
    row["bic_2mode"] = _safe_float(payload.get("bic_2mode"))
    row["rss_1mode"] = _safe_float(payload.get("rss_1mode"))
    row["rss_2mode"] = _safe_float(payload.get("rss_2mode"))

    if row["metric_value"] is None or row["two_mode_preferred"] is None:
        row["status"] = GATE_NOT_AVAILABLE
        row["reason"] = (
            f"{MODEL_METRIC_NAME} or decision.two_mode_preferred not available "
            f"in {_rel_to(run_dir, model_path)}"
        )
        return row

    if bool(row["two_mode_preferred"]):
        row["status"] = GATE_PASS
        row["reason"] = (
            f"{MODEL_METRIC_NAME}={row['metric_value']:.6g} supports 220+221 "
            f"(threshold={row['metric_threshold']})"
        )
    else:
        row["status"] = GATE_WEAK
        row["reason"] = (
            f"{MODEL_METRIC_NAME}={row['metric_value']:.6g} does not support 220+221 "
            f"(threshold={row['metric_threshold']})"
        )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _sweep_t0_rows(
    *,
    out_root: Path,
    run_dir: Path,
    stage_dir: Path,
    event_id: str | None,
    t0_grid_ms: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    if event_id is None:
        return (
            {
                "status": GATE_NOT_AVAILABLE,
                "reason": "event_id unavailable; t0 stability gate was not evaluated",
                "stable_t0_fraction": None,
                "n_attempted": 0,
                "n_usable": 0,
                "n_stable": 0,
                "reference_f221_hz": None,
                "reference_tau221_s": None,
                "reference_policy": "median over usable t0 subruns",
            },
            rows,
            model_rows,
        )

    if not t0_grid_ms:
        return (
            {
                "status": GATE_NOT_AVAILABLE,
                "reason": "empty t0 grid; t0 stability gate was not evaluated",
                "stable_t0_fraction": None,
                "n_attempted": 0,
                "n_usable": 0,
                "n_stable": 0,
                "reference_f221_hz": None,
                "reference_tau221_s": None,
                "reference_policy": "median over usable t0 subruns",
            },
            rows,
            model_rows,
        )

    for t0_ms in t0_grid_ms:
        try:
            row = _run_t0_point(
                event_id=event_id,
                base_run_dir=run_dir,
                exp_stage_dir=stage_dir,
                t0_value=float(t0_ms),
                units="ms",
                mass_msun=1.0,
                python_exe=sys.executable,
                out_root=out_root,
            )
            subrun_path = out_root / str(row["subrun_path"]).rstrip("/")
            multimode_path = subrun_path / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
            model_path = subrun_path / "s3b_multimode_estimates" / "outputs" / "model_comparison.json"
            estimate = _extract_mode_estimate(multimode_path)
            model_row = _load_model_selection(run_dir, model_path)
            model_row.update(
                {
                    "scope": "t0_subrun",
                    "t0_ms": float(t0_ms),
                }
            )
            model_rows.append(model_row)
            rows.append(
                {
                    "event_id": event_id,
                    "t0_ms": float(t0_ms),
                    "status": GATE_PASS if estimate["status"] == GATE_PASS else GATE_WEAK,
                    "subrun_path": _rel_to(run_dir, subrun_path),
                    "has_221": estimate["status"] == GATE_PASS,
                    "f221_measured": estimate["f_hz"],
                    "tau221_measured": estimate["tau_s"],
                    "valid_fraction_221": estimate["valid_fraction"],
                    "t0_reference_f221_hz": None,
                    "t0_reference_tau221_s": None,
                    "rel_err_f_vs_t0_median": None,
                    "rel_err_tau_vs_t0_median": None,
                    "stable_vs_t0_median": False,
                    "model_metric_name": model_row["metric_name"],
                    "model_metric_value": model_row["metric_value"],
                    "two_mode_preferred": model_row["two_mode_preferred"],
                    "reason": estimate["reason"] or model_row["reason"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "event_id": event_id,
                    "t0_ms": float(t0_ms),
                    "status": "ERROR",
                    "subrun_path": "",
                    "has_221": False,
                    "f221_measured": None,
                    "tau221_measured": None,
                    "valid_fraction_221": None,
                    "t0_reference_f221_hz": None,
                    "t0_reference_tau221_s": None,
                    "rel_err_f_vs_t0_median": None,
                    "rel_err_tau_vs_t0_median": None,
                    "stable_vs_t0_median": False,
                    "model_metric_name": MODEL_METRIC_NAME,
                    "model_metric_value": None,
                    "two_mode_preferred": None,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    usable_rows = [row for row in rows if row["has_221"] and row["f221_measured"] is not None and row["tau221_measured"] is not None]
    if not usable_rows:
        return (
            {
                "status": GATE_NOT_AVAILABLE,
                "reason": "no usable 221 estimates were recovered across the requested t0 sweep",
                "stable_t0_fraction": None,
                "n_attempted": len(rows),
                "n_usable": 0,
                "n_stable": 0,
                "reference_f221_hz": None,
                "reference_tau221_s": None,
                "reference_policy": "median over usable t0 subruns",
            },
            rows,
            model_rows,
        )

    ref_f = float(statistics.median([float(row["f221_measured"]) for row in usable_rows]))
    ref_tau = float(statistics.median([float(row["tau221_measured"]) for row in usable_rows]))
    for row in rows:
        row["t0_reference_f221_hz"] = ref_f
        row["t0_reference_tau221_s"] = ref_tau
    return (
        {
            "status": GATE_PASS,
            "reason": None,
            "stable_t0_fraction": None,
            "n_attempted": len(rows),
            "n_usable": len(usable_rows),
            "n_stable": 0,
            "reference_f221_hz": ref_f,
            "reference_tau221_s": ref_tau,
            "reference_policy": "median over usable t0 subruns",
        },
        rows,
        model_rows,
    )


def _finalize_t0_gate(
    gate_b: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    f_rel_tol: float,
    tau_rel_tol: float,
    stable_min_fraction: float,
    stable_reject_fraction: float,
) -> dict[str, Any]:
    if gate_b["status"] == GATE_NOT_AVAILABLE:
        return gate_b

    n_stable = 0
    for row in rows:
        rel_f = _rel_err(_safe_float(row["f221_measured"]), gate_b["reference_f221_hz"])
        rel_tau = _rel_err(_safe_float(row["tau221_measured"]), gate_b["reference_tau221_s"])
        stable = (
            row["has_221"]
            and rel_f is not None
            and rel_tau is not None
            and rel_f <= f_rel_tol
            and rel_tau <= tau_rel_tol
        )
        row["rel_err_f_vs_t0_median"] = rel_f
        row["rel_err_tau_vs_t0_median"] = rel_tau
        row["stable_vs_t0_median"] = stable
        if stable:
            n_stable += 1

    attempted = int(gate_b["n_attempted"])
    stable_fraction = (float(n_stable) / float(attempted)) if attempted > 0 else None
    gate_b["n_stable"] = n_stable
    gate_b["stable_t0_fraction"] = stable_fraction
    gate_b["stability_f_rel_tol"] = f_rel_tol
    gate_b["stability_tau_rel_tol"] = tau_rel_tol

    if stable_fraction is None:
        gate_b["status"] = GATE_NOT_AVAILABLE
        gate_b["reason"] = "stable_t0_fraction unavailable"
    elif stable_fraction < stable_reject_fraction:
        gate_b["status"] = GATE_FAIL
        gate_b["reason"] = (
            f"stable_t0_fraction={stable_fraction:.6g} below reject threshold {stable_reject_fraction:.6g}"
        )
    elif stable_fraction < stable_min_fraction:
        gate_b["status"] = GATE_WEAK
        gate_b["reason"] = (
            f"stable_t0_fraction={stable_fraction:.6g} below support threshold {stable_min_fraction:.6g}"
        )
    else:
        gate_b["status"] = GATE_PASS
        gate_b["reason"] = (
            f"stable_t0_fraction={stable_fraction:.6g} meets support threshold {stable_min_fraction:.6g}"
        )
    return gate_b


def _verdict_from_gates(gate_a: dict[str, Any], gate_b: dict[str, Any], gate_c: dict[str, Any]) -> tuple[str, str]:
    if gate_a["status"] == GATE_NOT_AVAILABLE:
        return "INSUFFICIENT_DATA", f"Gate A not available: {gate_a['reason']}"
    if gate_a["status"] == GATE_FAIL:
        return "REJECTED", f"Gate A failed: {gate_a['reason']}"
    if gate_b["status"] == GATE_FAIL:
        return "REJECTED", f"Gate B failed: {gate_b['reason']}"
    if gate_b["status"] == GATE_PASS and gate_c["status"] == GATE_PASS:
        return "KERR_COMPATIBLE", "Gate A passes; Gate B is stable; Gate C supports 220+221 over 220-only"
    reasons = []
    if gate_b["status"] != GATE_PASS:
        reasons.append(f"Gate B {gate_b['status'].lower()}: {gate_b['reason']}")
    if gate_c["status"] != GATE_PASS:
        reasons.append(f"Gate C {gate_c['status'].lower()}: {gate_c['reason']}")
    if not reasons:
        reasons.append("Gate A passes but the evidence remains conditional/post-hoc")
    return "WEAK_EVIDENCE", "; ".join(reasons)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_root = resolve_out_root("runs")
    validate_run_id(args.run_id, out_root)
    require_run_valid(out_root, args.run_id)

    run_dir = out_root / args.run_id
    stage_dir = run_dir / "experiment" / EXPERIMENT_NAME
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    input_records: list[dict[str, str]] = []
    source_estimates_path = run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    kerr_extraction_path = run_dir / "s4d_kerr_from_multimode" / "outputs" / "kerr_extraction.json"
    base_model_path = run_dir / "s3b_multimode_estimates" / "outputs" / "model_comparison.json"
    _record_input(run_dir, input_records, "source_estimates", source_estimates_path)
    _record_input(run_dir, input_records, "kerr_extraction", kerr_extraction_path)
    _record_input(run_dir, input_records, "model_comparison", base_model_path)

    event_id, event_id_source, event_id_candidates = _resolve_event_id(run_dir, args.event_id, input_records)
    measured = _extract_mode_estimate(source_estimates_path)
    remnant = _load_kerr_remnant(kerr_extraction_path)
    base_model = _load_model_selection(run_dir, base_model_path)

    f221_kerr: float | None = None
    tau221_kerr: float | None = None
    rel_err_f: float | None = None
    rel_err_tau: float | None = None
    gate_a_reason: str
    gate_a_status: str
    if remnant["status"] != GATE_PASS:
        gate_a_status = GATE_NOT_AVAILABLE
        gate_a_reason = str(remnant["reason"])
    else:
        oracle = kerr_qnm(float(remnant["M_final_Msun"]), float(remnant["chi_final"]), MODE_TUPLE)
        f221_kerr = float(oracle.f_hz)
        tau221_kerr = float(oracle.tau_s)
        if measured["status"] != GATE_PASS:
            gate_a_status = GATE_NOT_AVAILABLE
            gate_a_reason = f"measured 221 unavailable: {measured['reason']}"
        else:
            rel_err_f = _rel_err(measured["f_hz"], f221_kerr)
            rel_err_tau = _rel_err(measured["tau_s"], tau221_kerr)
            if (
                rel_err_f is not None
                and rel_err_tau is not None
                and rel_err_f <= float(args.f_tol_rel)
                and rel_err_tau <= float(args.tau_tol_rel)
            ):
                gate_a_status = GATE_PASS
                gate_a_reason = (
                    f"rel_err_f={rel_err_f:.6g} <= {args.f_tol_rel:.6g} and "
                    f"rel_err_tau={rel_err_tau:.6g} <= {args.tau_tol_rel:.6g}"
                )
            else:
                gate_a_status = GATE_FAIL
                gate_a_reason = (
                    f"rel_err_f={rel_err_f} rel_err_tau={rel_err_tau} exceed "
                    f"tolerances ({args.f_tol_rel}, {args.tau_tol_rel})"
                )
    gate_a = {
        "status": gate_a_status,
        "reason": gate_a_reason,
        "f_tol_rel": float(args.f_tol_rel),
        "tau_tol_rel": float(args.tau_tol_rel),
    }

    t0_grid_ms = _parse_t0_grid(args.t0_grid_ms) if args.t0_grid_ms.strip() else []
    gate_b, t0_rows, t0_model_rows = _sweep_t0_rows(
        out_root=out_root,
        run_dir=run_dir,
        stage_dir=stage_dir,
        event_id=event_id,
        t0_grid_ms=t0_grid_ms,
    )
    gate_b = _finalize_t0_gate(
        gate_b,
        t0_rows,
        f_rel_tol=float(args.t0_stability_f_rel_tol),
        tau_rel_tol=float(args.t0_stability_tau_rel_tol),
        stable_min_fraction=float(args.stable_t0_min_fraction),
        stable_reject_fraction=float(args.stable_t0_reject_fraction),
    )

    gate_c = {
        "status": base_model["status"],
        "reason": base_model["reason"],
        "metric_name": base_model["metric_name"],
        "metric_value": base_model["metric_value"],
        "metric_threshold": base_model["metric_threshold"],
        "two_mode_preferred": base_model["two_mode_preferred"],
    }

    verdict, verdict_reason = _verdict_from_gates(gate_a, gate_b, gate_c)
    if verdict not in ALLOWED_VERDICTS:
        raise RuntimeError(f"Unexpected verdict {verdict!r}; allowed={ALLOWED_VERDICTS}")

    kerr_oracle_payload = {
        "run_id": args.run_id,
        "event_id": event_id,
        "mode": list(MODE_TUPLE),
        "mode_label": MODE_LABEL,
        "source_estimates_path": _rel_to(run_dir, source_estimates_path),
        "mf_source": f"{_rel_to(run_dir, kerr_extraction_path)}#M_final_Msun",
        "af_source": f"{_rel_to(run_dir, kerr_extraction_path)}#chi_final",
        "selection_policy": measured["selection_policy"],
        "f221_measured": measured["f_hz"],
        "tau221_measured": measured["tau_s"],
        "f221_kerr": f221_kerr,
        "tau221_kerr": tau221_kerr,
        "rel_err_f": rel_err_f,
        "rel_err_tau": rel_err_tau,
        "gate_a": gate_a,
    }

    summary_payload = {
        "run_id": args.run_id,
        "event_id": event_id,
        "event_id_source": event_id_source,
        "event_id_candidates": event_id_candidates,
        "source_estimates_path": _rel_to(run_dir, source_estimates_path),
        "mf_source": f"{_rel_to(run_dir, kerr_extraction_path)}#M_final_Msun",
        "af_source": f"{_rel_to(run_dir, kerr_extraction_path)}#chi_final",
        "selection_policy": {
            "primary_221": measured["selection_policy"],
            "t0_reference": gate_b.get("reference_policy"),
        },
        "f221_measured": measured["f_hz"],
        "tau221_measured": measured["tau_s"],
        "f221_kerr": f221_kerr,
        "tau221_kerr": tau221_kerr,
        "rel_err_f": rel_err_f,
        "rel_err_tau": rel_err_tau,
        "stable_t0_fraction": gate_b.get("stable_t0_fraction"),
        "model_selection_metric_name": MODEL_METRIC_NAME,
        "model_selection_metric_value": base_model["metric_value"],
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "gate_c": gate_c,
    }

    t0_fields = [
        "event_id",
        "t0_ms",
        "status",
        "subrun_path",
        "has_221",
        "f221_measured",
        "tau221_measured",
        "valid_fraction_221",
        "t0_reference_f221_hz",
        "t0_reference_tau221_s",
        "rel_err_f_vs_t0_median",
        "rel_err_tau_vs_t0_median",
        "stable_vs_t0_median",
        "model_metric_name",
        "model_metric_value",
        "two_mode_preferred",
        "reason",
    ]
    model_fields = [
        "scope",
        "t0_ms",
        "source_path",
        "metric_name",
        "metric_value",
        "metric_threshold",
        "two_mode_preferred",
        "status",
        "reason",
        "bic_1mode",
        "bic_2mode",
        "rss_1mode",
        "rss_2mode",
    ]
    model_rows = [
        {
            "scope": "base_run",
            "t0_ms": "",
            **base_model,
        },
        *t0_model_rows,
    ]

    kerr_oracle_path = outputs_dir / "kerr_oracle_221.json"
    t0_stability_path = outputs_dir / "t0_stability_221.csv"
    model_selection_path = outputs_dir / "model_selection_220_vs_220221.csv"
    summary_path = outputs_dir / "summary_221_validation.json"
    write_json_atomic(kerr_oracle_path, kerr_oracle_payload)
    _write_csv(t0_stability_path, t0_rows, t0_fields)
    _write_csv(model_selection_path, model_rows, model_fields)
    write_json_atomic(summary_path, summary_payload)

    output_artifacts = {
        "kerr_oracle_221": kerr_oracle_path,
        "t0_stability_221": t0_stability_path,
        "model_selection_220_vs_220221": model_selection_path,
        "summary_221_validation": summary_path,
    }
    outputs = [
        {
            "path": _rel_to(run_dir, path),
            "sha256": sha256_file(path),
        }
        for path in output_artifacts.values()
    ]
    stage_summary_payload = {
        "stage": EXPERIMENT_STAGE,
        "run": args.run_id,
        "created": utc_now_iso(),
        "command": " ".join(shlex.quote(x) for x in [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "parameters": {
            "event_id": args.event_id,
            "t0_grid_ms": t0_grid_ms,
            "f_tol_rel": float(args.f_tol_rel),
            "tau_tol_rel": float(args.tau_tol_rel),
            "t0_stability_f_rel_tol": float(args.t0_stability_f_rel_tol),
            "t0_stability_tau_rel_tol": float(args.t0_stability_tau_rel_tol),
            "stable_t0_min_fraction": float(args.stable_t0_min_fraction),
            "stable_t0_reject_fraction": float(args.stable_t0_reject_fraction),
        },
        "inputs": input_records,
        "outputs": outputs,
        "verdict": "PASS",
        "results": {
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "gate_a_status": gate_a["status"],
            "gate_b_status": gate_b["status"],
            "gate_c_status": gate_c["status"],
        },
    }
    stage_summary_path = write_stage_summary(stage_dir, stage_summary_payload)
    manifest_path = write_manifest(
        stage_dir,
        {
            **output_artifacts,
            "stage_summary": stage_summary_path,
        },
        extra={"inputs": input_records, "verdict": "PASS"},
    )

    contracts.log_stage_paths(
        SimpleNamespace(out_root=out_root, stage_dir=stage_dir, outputs_dir=outputs_dir)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
