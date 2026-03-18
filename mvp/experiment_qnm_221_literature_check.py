#!/usr/bin/env python3
"""Post-hoc, auditable, minimal validation for a candidate Kerr (2,2,1) mode."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import (
    ensure_stage_dirs,
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
from mvp.kerr_qnm_fits import kerr_qnm

EXPERIMENT_STAGE = "experiment/qnm_221_literature_check"
ALLOWED_VERDICTS = {
    "KERR_COMPATIBLE",
    "WEAK_EVIDENCE",
    "REJECTED",
    "INSUFFICIENT_DATA",
}
T0MS_RE = re.compile(r"__t0ms(\d+)")
EVENT_ID_RE = re.compile(r"(GW\d{6}(?:_\d{6})?)")
REMNANT_H5_SUFFIXES = {".h5", ".hdf5", ".nc"}
PREFERRED_POSTERIOR_DATASETS = ("C01:Mixed/posterior_samples",)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _rel_to(base: Path, path: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _tau_from_q(f_hz: float | None, q_val: float | None) -> float | None:
    if f_hz is None or q_val is None or f_hz <= 0.0 or q_val <= 0.0:
        return None
    return q_val / (math.pi * f_hz)


def _extract_mode_dict_candidates(obj: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        out.append(obj)
        for value in obj.values():
            out.extend(_extract_mode_dict_candidates(value))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(_extract_mode_dict_candidates(value))
    return out


def _upstream_221_block_reason(multimode: dict[str, Any]) -> str | None:
    results = multimode.get("results") if isinstance(multimode.get("results"), dict) else {}
    raw_flags = results.get("quality_flags")
    flags = [str(flag).strip() for flag in raw_flags if str(flag).strip()] if isinstance(raw_flags, list) else []
    flagged_221 = [flag for flag in flags if flag.startswith("221_")]
    if flagged_221:
        return f"upstream_221_quality_flags:{','.join(flagged_221)}"

    verdict = results.get("verdict")
    if isinstance(verdict, str):
        verdict = verdict.strip()
        if verdict and verdict != "PASS":
            return f"upstream_221_verdict:{verdict}"
    return None


def _extract_221_from_multimode(multimode: dict[str, Any], s3_estimates: dict[str, Any] | None) -> tuple[float | None, float | None, str]:
    """Minimal explicit policy for recovering the 221 estimate from s3b."""
    block_reason = _upstream_221_block_reason(multimode)
    if block_reason is not None:
        return None, None, block_reason

    for item in _extract_mode_dict_candidates(multimode):
        label = str(item.get("label", item.get("mode", ""))).strip()
        ell = item.get("l")
        emm = item.get("m")
        enn = item.get("n")
        is_221 = label in {"221", "(2,2,1)"} or (ell == 2 and emm == 2 and enn == 1)
        if not is_221:
            continue

        fit = item.get("fit") if isinstance(item.get("fit"), dict) else {}
        stability = fit.get("stability") if isinstance(fit.get("stability"), dict) else {}
        lnf_p50 = _safe_float(stability.get("lnf_p50"))
        lnq_p50 = _safe_float(stability.get("lnQ_p50"))
        if lnf_p50 is not None and lnq_p50 is not None:
            f221 = math.exp(lnf_p50)
            tau221 = math.exp(lnq_p50 - lnf_p50) / math.pi
            return f221, tau221, "mode_221.fit.stability.p50"

        f221 = _safe_float(
            item.get("f_hz")
            or item.get("freq_hz")
            or item.get("f221_hz")
            or item.get("frequency_hz")
        )
        tau221 = _safe_float(
            item.get("tau_s")
            or item.get("tau221_s")
            or item.get("decay_time_s")
        )
        q221 = _safe_float(item.get("Q") or item.get("q") or item.get("Q221"))
        if tau221 is None:
            tau221 = _tau_from_q(f221, q221)
        if f221 is not None:
            return f221, tau221, "explicit_mode_221"

        ln_f = _safe_float(item.get("ln_f"))
        ln_q = _safe_float(item.get("ln_Q"))
        if ln_f is not None and ln_q is not None:
            f221 = math.exp(ln_f)
            tau221 = math.exp(ln_q - ln_f) / math.pi
            return f221, tau221, "mode_221.point_estimate"

    joint_3d = multimode.get("joint_3d")
    if isinstance(joint_3d, dict) and joint_3d.get("valid") and s3_estimates is not None:
        point = joint_3d.get("point_estimate")
        if isinstance(point, list) and len(point) >= 3:
            ln_ratio = _safe_float(point[2])
            f220 = _safe_float(((s3_estimates.get("combined") or {}) if isinstance(s3_estimates.get("combined"), dict) else {}).get("f_hz"))
            if ln_ratio is not None and f220 is not None and f220 > 0.0:
                return f220 * math.exp(ln_ratio), None, "joint_3d_ln_ratio_times_s3_f220"

    return None, None, "not_found"


def _extract_model_selection_metric(model_comp: dict[str, Any]) -> tuple[str, float | None, bool | None, float | None, str]:
    decision = model_comp.get("decision") if isinstance(model_comp.get("decision"), dict) else {}
    thresholds = model_comp.get("thresholds") if isinstance(model_comp.get("thresholds"), dict) else {}
    two_mode_preferred = decision.get("two_mode_preferred")
    if not isinstance(two_mode_preferred, bool):
        raw = model_comp.get("two_mode_preferred")
        two_mode_preferred = raw if isinstance(raw, bool) else None

    if "delta_bic" in model_comp:
        return (
            "delta_bic",
            _safe_float(model_comp.get("delta_bic")),
            two_mode_preferred,
            _safe_float(thresholds.get("two_mode_preferred_delta_bic")),
            "from_model_comparison",
        )
    if "delta_aic" in model_comp:
        return "delta_aic", _safe_float(model_comp.get("delta_aic")), two_mode_preferred, None, "from_model_comparison"
    if "log_likelihood_gain" in model_comp:
        return "log_likelihood_gain", _safe_float(model_comp.get("log_likelihood_gain")), two_mode_preferred, None, "from_model_comparison"
    if "residual_norm_ratio" in model_comp:
        return "residual_norm_ratio", _safe_float(model_comp.get("residual_norm_ratio")), two_mode_preferred, None, "from_model_comparison"
    return "NOT_AVAILABLE", None, two_mode_preferred, None, "metric_missing"


def _discover_remnant_path(run_dir: Path, *, event_id: str | None) -> Path | None:
    candidates = [
        run_dir / "external_inputs" / "remnant_kerr.json",
        run_dir / "external_inputs" / "remnant.json",
        run_dir / "external_inputs" / "final_state.json",
        run_dir / "external_inputs" / "imr_remnant.json",
        run_dir / "annotations" / "remnant_kerr.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raw_dir = run_dir / "external_inputs" / "gwtc_posteriors" / "raw"
    if not raw_dir.exists():
        return None
    h5_candidates = sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in REMNANT_H5_SUFFIXES
    )
    if not h5_candidates:
        return None
    if event_id:
        exact = [path for path in h5_candidates if event_id in path.name]
        if exact:
            return exact[0]
        canonical = event_id.split("_", 1)[0]
        partial = [path for path in h5_candidates if canonical in path.name]
        if partial:
            return partial[0]
    return h5_candidates[0]


def _extract_mf_af(remnant: dict[str, Any]) -> tuple[float | None, float | None]:
    mf_keys = ("Mf", "mf", "final_mass", "mass_final", "remnant_mass")
    af_keys = ("af", "chi_f", "final_spin", "spin_final", "remnant_spin")
    mf = next((_safe_float(remnant.get(key)) for key in mf_keys if _safe_float(remnant.get(key)) is not None), None)
    af = next((_safe_float(remnant.get(key)) for key in af_keys if _safe_float(remnant.get(key)) is not None), None)
    return mf, af


def _extract_event_id(run_dir: Path, run_id: str, s3_estimates: dict[str, Any] | None) -> str | None:
    provenance_path = run_dir / "run_provenance.json"
    if provenance_path.exists():
        provenance = _load_json(provenance_path)
        invocation = provenance.get("invocation")
        if isinstance(invocation, dict):
            value = invocation.get("event_id")
            if isinstance(value, str) and value.strip():
                return value.strip()
        value = provenance.get("event_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(s3_estimates, dict):
        value = s3_estimates.get("event_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    match = EVENT_ID_RE.search(run_id)
    return match.group(1) if match else None


def _percentiles(values: Any) -> tuple[float | None, float | None, float | None]:
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(f"numpy_unavailable: {exc}") from exc

    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None, None
    p10, p50, p90 = np.percentile(arr, [10.0, 50.0, 90.0])
    return float(p10), float(p50), float(p90)


def _find_posterior_samples_dataset(path: Path) -> tuple[str | None, tuple[str, ...], str | None]:
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:
        return None, (), f"h5py_unavailable: {exc}"

    with h5py.File(str(path), "r") as h5:
        for preferred in PREFERRED_POSTERIOR_DATASETS:
            if preferred not in h5:
                continue
            ds = h5[preferred]
            names = tuple(ds.dtype.names or ())
            if "final_mass" in names and "final_spin" in names:
                return preferred, names, None

        dataset_paths: list[str] = []

        def _visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if not name.endswith("posterior_samples"):
                return
            names = tuple(obj.dtype.names or ())
            if "final_mass" in names and "final_spin" in names:
                dataset_paths.append(name)

        h5.visititems(_visit)
        if dataset_paths:
            chosen = sorted(dataset_paths)[0]
            names = tuple(h5[chosen].dtype.names or ())
            return chosen, names, None
    return None, (), "no_posterior_samples_with_final_mass_and_final_spin"


def _extract_mf_af_from_h5(path: Path) -> tuple[float | None, float | None, dict[str, Any]]:
    dataset_path, names, reason = _find_posterior_samples_dataset(path)
    meta: dict[str, Any] = {
        "source_format": "hdf5",
        "selection_policy": (
            "prefer C01:Mixed/posterior_samples; fallback to the first posterior_samples "
            "dataset with final_mass and final_spin; use p50 of each field"
        ),
        "dataset_path": dataset_path,
        "available_fields": list(names),
        "reason": reason,
    }
    if dataset_path is None:
        return None, None, meta

    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:
        meta["reason"] = f"h5py_unavailable: {exc}"
        return None, None, meta

    with h5py.File(str(path), "r") as h5:
        data = h5[dataset_path][()]

    try:
        mf_p10, mf_p50, mf_p90 = _percentiles(data["final_mass"])
        af_p10, af_p50, af_p90 = _percentiles(data["final_spin"])
    except RuntimeError as exc:
        meta["reason"] = str(exc)
        return None, None, meta

    meta.update(
        {
            "mf_p10": mf_p10,
            "mf_p50": mf_p50,
            "mf_p90": mf_p90,
            "af_p10": af_p10,
            "af_p50": af_p50,
            "af_p90": af_p90,
            "reason": None if mf_p50 is not None and af_p50 is not None else "non_finite_final_mass_or_spin",
        }
    )
    return mf_p50, af_p50, meta


def _load_remnant_estimate(path: Path) -> tuple[float | None, float | None, dict[str, Any]]:
    if path.suffix.lower() in REMNANT_H5_SUFFIXES:
        return _extract_mf_af_from_h5(path)

    remnant = _load_json(path)
    mf, af = _extract_mf_af(remnant)
    return (
        mf,
        af,
        {
            "source_format": "json",
            "selection_policy": "first non-null canonical key among Mf/mf/final_mass and af/chi_f/final_spin",
            "reason": None if mf is not None and af is not None else "missing_json_mf_or_af",
        },
    )


def _predict_kerr_221(mf: float, af: float) -> tuple[float, float, dict[str, Any]]:
    if mf <= 0.0:
        raise ValueError(f"invalid Mf={mf}")
    if not (0.0 <= af <= 0.999):
        raise ValueError(f"invalid af={af}")
    prediction = kerr_qnm(mf, af, (2, 2, 1))
    if prediction.f_hz <= 0.0 or prediction.tau_s <= 0.0:
        raise RuntimeError(f"invalid_kerr_qnm_prediction: {prediction}")
    return (
        float(prediction.f_hz),
        float(prediction.tau_s),
        {
            "tool": "mvp.kerr_qnm_fits.kerr_qnm",
            "mode": "221",
            "a_final": af,
            "M_final": mf,
            "Q": float(prediction.Q),
        },
    )


def _gate_a_kerr(
    *,
    f_meas: float | None,
    tau_meas: float | None,
    mf: float | None,
    af: float | None,
    f_rel_tol: float,
    tau_rel_tol: float,
) -> dict[str, Any]:
    out = {
        "name": "GATE_A_KERR_CONSISTENCY",
        "status": "NOT_AVAILABLE",
        "reason": None,
        "f221_kerr": None,
        "tau221_kerr": None,
        "rel_err_f": None,
        "rel_err_tau": None,
        "oracle_meta": None,
    }
    if f_meas is None or tau_meas is None:
        out["reason"] = "missing_measured_f_or_tau_221"
        return out
    if mf is None or af is None:
        out["reason"] = "missing_canonical_mf_af"
        return out

    try:
        f_kerr, tau_kerr, meta = _predict_kerr_221(mf, af)
    except Exception as exc:
        out["reason"] = str(exc)
        return out

    rel_err_f = abs(f_meas - f_kerr) / abs(f_kerr)
    rel_err_tau = abs(tau_meas - tau_kerr) / abs(tau_kerr)
    out.update(
        {
            "f221_kerr": f_kerr,
            "tau221_kerr": tau_kerr,
            "rel_err_f": rel_err_f,
            "rel_err_tau": rel_err_tau,
            "oracle_meta": meta,
        }
    )
    if rel_err_f <= f_rel_tol and rel_err_tau <= tau_rel_tol:
        out["status"] = "PASS"
        out["reason"] = "frequency_and_tau_within_tolerance"
    else:
        out["status"] = "FAIL"
        out["reason"] = "kerr_mismatch"
    return out


def _parse_t0_ms_from_run_id(run_id: str) -> float | None:
    match = T0MS_RE.search(run_id)
    if not match:
        return None
    return float(match.group(1))


def _iter_t0_subruns(run_dir: Path) -> list[Path]:
    exp_root = run_dir / "experiment"
    if not exp_root.exists():
        return []
    subruns: list[Path] = []
    for path in exp_root.rglob("*__t0ms*"):
        if not path.is_dir():
            continue
        if "t0_sweep_full" not in path.as_posix():
            continue
        if (path / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json").exists():
            subruns.append(path)
    return sorted(subruns)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _gate_b_t0_stability(
    *,
    run_dir: Path,
    outputs_dir: Path,
    primary_f221: float | None,
    primary_tau221: float | None,
    stability_rel_tol_f: float,
    stability_rel_tol_tau: float,
    min_stable_fraction: float,
) -> dict[str, Any]:
    csv_path = outputs_dir / "t0_stability_221.csv"
    rows: list[dict[str, Any]] = []

    for subrun_dir in _iter_t0_subruns(run_dir):
        s3_path = subrun_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
        s3b_path = subrun_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
        row = {
            "subrun_id": subrun_dir.name,
            "t0_ms": _parse_t0_ms_from_run_id(subrun_dir.name),
            "source_estimates_path": _rel_to(run_dir, s3b_path),
            "f221_measured": None,
            "tau221_measured": None,
            "stable_vs_primary": None,
            "reason": None,
        }
        s3_data = _load_json(s3_path) if s3_path.exists() else None
        s3b_data = _load_json(s3b_path)
        f221, tau221, why = _extract_221_from_multimode(s3b_data, s3_data)
        row["f221_measured"] = f221
        row["tau221_measured"] = tau221

        if primary_f221 is None or f221 is None:
            row["reason"] = f"cannot_compare:{why}"
            rows.append(row)
            continue

        f_ok = abs(f221 - primary_f221) / abs(primary_f221) <= stability_rel_tol_f
        tau_ok = True
        if primary_tau221 is not None and tau221 is not None:
            tau_ok = abs(tau221 - primary_tau221) / abs(primary_tau221) <= stability_rel_tol_tau

        row["stable_vs_primary"] = bool(f_ok and tau_ok)
        row["reason"] = why
        rows.append(row)

    fieldnames = [
        "subrun_id",
        "t0_ms",
        "source_estimates_path",
        "f221_measured",
        "tau221_measured",
        "stable_vs_primary",
        "reason",
    ]
    _write_csv(csv_path, rows, fieldnames)

    comparable = [row for row in rows if isinstance(row.get("stable_vs_primary"), bool)]
    if len(comparable) < 2:
        return {
            "name": "GATE_B_T0_STABILITY",
            "status": "NOT_AVAILABLE",
            "reason": "insufficient_t0_subruns_with_comparable_221",
            "stable_t0_fraction": None,
            "csv_path": str(csv_path),
        }

    stable_fraction = sum(1 for row in comparable if row["stable_vs_primary"]) / len(comparable)
    status = "PASS" if stable_fraction >= min_stable_fraction else "FAIL"
    reason = "stable_across_t0" if status == "PASS" else "unstable_across_t0"
    return {
        "name": "GATE_B_T0_STABILITY",
        "status": status,
        "reason": reason,
        "stable_t0_fraction": stable_fraction,
        "csv_path": str(csv_path),
    }


def _gate_c_model_selection(
    *,
    model_comp_path: Path | None,
    run_dir: Path,
    outputs_dir: Path,
) -> dict[str, Any]:
    csv_path = outputs_dir / "model_selection_220_vs_220221.csv"
    rows: list[dict[str, Any]] = []

    if model_comp_path is None or not model_comp_path.exists():
        rows.append(
            {
                "source_model_comparison_path": None,
                "metric_name": "NOT_AVAILABLE",
                "metric_value": None,
                "metric_threshold": None,
                "two_mode_preferred": None,
                "status": "NOT_AVAILABLE",
                "reason": "missing_model_comparison_json",
            }
        )
        _write_csv(
            csv_path,
            rows,
            [
                "source_model_comparison_path",
                "metric_name",
                "metric_value",
                "metric_threshold",
                "two_mode_preferred",
                "status",
                "reason",
            ],
        )
        return {
            "name": "GATE_C_MODEL_SELECTION",
            "status": "NOT_AVAILABLE",
            "reason": "missing_model_comparison_json",
            "metric_name": "NOT_AVAILABLE",
            "metric_value": None,
            "csv_path": str(csv_path),
        }

    data = _load_json(model_comp_path)
    metric_name, metric_value, two_mode_preferred, threshold, why = _extract_model_selection_metric(data)
    if two_mode_preferred is not None:
        status = "PASS" if two_mode_preferred else "FAIL"
        reason = "two_mode_preferred" if two_mode_preferred else "two_mode_not_preferred"
    elif metric_name == "delta_bic" and metric_value is not None and threshold is not None:
        status = "PASS" if metric_value < threshold else "FAIL"
        reason = "delta_bic_threshold_compare"
    elif metric_value is None:
        status = "NOT_AVAILABLE"
        reason = why
    else:
        status = "NOT_AVAILABLE"
        reason = "missing_two_mode_preferred"

    rows.append(
        {
            "source_model_comparison_path": _rel_to(run_dir, model_comp_path),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "metric_threshold": threshold,
            "two_mode_preferred": two_mode_preferred,
            "status": status,
            "reason": reason,
        }
    )
    _write_csv(
        csv_path,
        rows,
        [
            "source_model_comparison_path",
            "metric_name",
            "metric_value",
            "metric_threshold",
            "two_mode_preferred",
            "status",
            "reason",
        ],
    )
    return {
        "name": "GATE_C_MODEL_SELECTION",
        "status": status,
        "reason": reason,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "csv_path": str(csv_path),
    }


def _decide_verdict(
    *,
    gate_a: dict[str, Any],
    gate_b: dict[str, Any],
    gate_c: dict[str, Any],
    f221: float | None,
    tau221: float | None,
    extraction_policy: str,
) -> tuple[str, str]:
    if f221 is None:
        if extraction_policy.startswith("upstream_221_"):
            return "INSUFFICIENT_DATA", extraction_policy
        return "INSUFFICIENT_DATA", "missing_221_frequency_measurement"
    if f221 <= 0.0:
        return "REJECTED", "non_physical_221_frequency"
    if tau221 is not None and tau221 <= 0.0:
        return "REJECTED", "non_physical_221_tau"

    if gate_a["status"] == "NOT_AVAILABLE":
        return "INSUFFICIENT_DATA", f"gate_a_not_available:{gate_a['reason']}"
    if gate_a["status"] == "FAIL":
        return "REJECTED", "gate_a_kerr_consistency_failed"
    if gate_b["status"] == "FAIL":
        return "REJECTED", "gate_b_t0_instability"
    if gate_c["status"] == "FAIL":
        return "WEAK_EVIDENCE", "kerr_compatible_but_model_selection_not_supportive"
    if gate_b["status"] == "NOT_AVAILABLE" or gate_c["status"] == "NOT_AVAILABLE":
        return "WEAK_EVIDENCE", "kerr_compatible_but_missing_stability_or_model_selection_support"
    return "KERR_COMPATIBLE", "gate_a_pass_and_b_c_do_not_contradict"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Post-hoc literature check for QNM 221")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--source-estimates-path", default=None)
    ap.add_argument("--model-comparison-path", default=None)
    ap.add_argument("--remnant-json", default=None)
    ap.add_argument("--f-rel-tol", type=float, default=0.10)
    ap.add_argument("--tau-rel-tol", type=float, default=0.25)
    ap.add_argument("--stability-rel-tol-f", type=float, default=0.10)
    ap.add_argument("--stability-rel-tol-tau", type=float, default=0.25)
    ap.add_argument("--min-stable-fraction", type=float, default=0.60)
    args = ap.parse_args(argv)

    out_root = resolve_out_root("runs")
    validate_run_id(args.run_id, out_root)
    require_run_valid(out_root, args.run_id)

    stage_dir, outputs_dir = ensure_stage_dirs(args.run_id, EXPERIMENT_STAGE, out_root)
    run_dir = out_root / args.run_id

    source_estimates_path = (
        Path(args.source_estimates_path)
        if args.source_estimates_path
        else run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    )
    s3_estimates_path = run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    model_comparison_path = (
        Path(args.model_comparison_path)
        if args.model_comparison_path
        else run_dir / "s3b_multimode_estimates" / "outputs" / "model_comparison.json"
    )
    s3_data = _load_json(s3_estimates_path) if s3_estimates_path.exists() else None
    event_id = _extract_event_id(run_dir, args.run_id, s3_data)
    remnant_path = Path(args.remnant_json) if args.remnant_json else _discover_remnant_path(run_dir, event_id=event_id)

    input_records: list[dict[str, str]] = []
    for label, path in (
        ("source_estimates", source_estimates_path),
        ("s3_estimates", s3_estimates_path),
        ("model_comparison", model_comparison_path),
        ("remnant", remnant_path),
    ):
        if path is not None and path.exists():
            input_records.append(
                {
                    "label": label,
                    "path": _rel_to(run_dir, path),
                    "sha256": sha256_file(path),
                }
            )

    s3b_data = _load_json(source_estimates_path) if source_estimates_path.exists() else {}
    f221_measured, tau221_measured, extraction_policy = _extract_221_from_multimode(s3b_data, s3_data)

    mf = None
    af = None
    mf_source = None
    af_source = None
    remnant_meta: dict[str, Any] | None = None
    if remnant_path is not None and remnant_path.exists():
        mf_source = _rel_to(run_dir, remnant_path)
        af_source = _rel_to(run_dir, remnant_path)
        mf, af, remnant_meta = _load_remnant_estimate(remnant_path)

    gate_a = _gate_a_kerr(
        f_meas=f221_measured,
        tau_meas=tau221_measured,
        mf=mf,
        af=af,
        f_rel_tol=float(args.f_rel_tol),
        tau_rel_tol=float(args.tau_rel_tol),
    )
    gate_b = _gate_b_t0_stability(
        run_dir=run_dir,
        outputs_dir=outputs_dir,
        primary_f221=f221_measured,
        primary_tau221=tau221_measured,
        stability_rel_tol_f=float(args.stability_rel_tol_f),
        stability_rel_tol_tau=float(args.stability_rel_tol_tau),
        min_stable_fraction=float(args.min_stable_fraction),
    )
    gate_c = _gate_c_model_selection(
        model_comp_path=model_comparison_path if model_comparison_path.exists() else None,
        run_dir=run_dir,
        outputs_dir=outputs_dir,
    )
    verdict, verdict_reason = _decide_verdict(
        gate_a=gate_a,
        gate_b=gate_b,
        gate_c=gate_c,
        f221=f221_measured,
        tau221=tau221_measured,
        extraction_policy=extraction_policy,
    )
    if verdict not in ALLOWED_VERDICTS:
        raise RuntimeError(f"invalid verdict: {verdict}")

    kerr_oracle = {
        "schema_version": "qnm_221_kerr_oracle_v1",
        "run_id": args.run_id,
        "remnant_source_path": _rel_to(run_dir, remnant_path) if remnant_path is not None else None,
        "mf": mf,
        "af": af,
        "remnant_meta": remnant_meta,
        "f221_kerr": gate_a["f221_kerr"],
        "tau221_kerr": gate_a["tau221_kerr"],
        "status": gate_a["status"],
        "reason": gate_a["reason"],
        "oracle_meta": gate_a["oracle_meta"],
    }
    summary = {
        "schema_version": "qnm_221_validation_summary_v1",
        "created_utc": utc_now_iso(),
        "run_id": args.run_id,
        "source_estimates_path": _rel_to(run_dir, source_estimates_path),
        "source_estimates_sha256": sha256_file(source_estimates_path) if source_estimates_path.exists() else None,
        "selection_policy": (
            "primary measurement uses canonical s3b_multimode_estimates from the PASS run; "
            "t0 subruns are used only for Gate B."
        ),
        "remnant_selection_policy": None if remnant_meta is None else remnant_meta.get("selection_policy"),
        "remnant_extraction_reason": None if remnant_meta is None else remnant_meta.get("reason"),
        "mf_source": mf_source,
        "af_source": af_source,
        "f221_measured": f221_measured,
        "tau221_measured": tau221_measured,
        "f221_kerr": gate_a["f221_kerr"],
        "tau221_kerr": gate_a["tau221_kerr"],
        "rel_err_f": gate_a["rel_err_f"],
        "rel_err_tau": gate_a["rel_err_tau"],
        "stable_t0_fraction": gate_b["stable_t0_fraction"],
        "model_selection_metric_name": gate_c["metric_name"],
        "model_selection_metric_value": gate_c["metric_value"],
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "gates": {"A": gate_a, "B": gate_b, "C": gate_c},
        "extraction_policy": extraction_policy,
    }

    kerr_path = outputs_dir / "kerr_oracle_221.json"
    summary_path = outputs_dir / "summary_221_validation.json"
    write_json_atomic(kerr_path, kerr_oracle)
    write_json_atomic(summary_path, summary)

    stage_summary_payload = {
        "stage": EXPERIMENT_STAGE,
        "run": args.run_id,
        "created": utc_now_iso(),
        "parameters": {
            "source_estimates_path": args.source_estimates_path,
            "model_comparison_path": args.model_comparison_path,
            "remnant_json": args.remnant_json,
            "f_rel_tol": float(args.f_rel_tol),
            "tau_rel_tol": float(args.tau_rel_tol),
            "stability_rel_tol_f": float(args.stability_rel_tol_f),
            "stability_rel_tol_tau": float(args.stability_rel_tol_tau),
            "min_stable_fraction": float(args.min_stable_fraction),
        },
        "inputs": input_records,
        "outputs": [
            {"path": _rel_to(run_dir, kerr_path), "sha256": sha256_file(kerr_path)},
            {"path": _rel_to(run_dir, outputs_dir / "t0_stability_221.csv"), "sha256": sha256_file(outputs_dir / "t0_stability_221.csv")},
            {"path": _rel_to(run_dir, outputs_dir / "model_selection_220_vs_220221.csv"), "sha256": sha256_file(outputs_dir / "model_selection_220_vs_220221.csv")},
            {"path": _rel_to(run_dir, summary_path), "sha256": sha256_file(summary_path)},
        ],
        "verdict": "PASS",
        "results": {
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "f221_measured": f221_measured,
            "tau221_measured": tau221_measured,
            "f221_kerr": gate_a["f221_kerr"],
            "tau221_kerr": gate_a["tau221_kerr"],
            "stable_t0_fraction": gate_b["stable_t0_fraction"],
            "model_selection_metric_name": gate_c["metric_name"],
            "model_selection_metric_value": gate_c["metric_value"],
        },
    }
    stage_summary_path = write_stage_summary(stage_dir, stage_summary_payload)
    write_manifest(
        stage_dir,
        {
            "kerr_oracle_221": kerr_path,
            "t0_stability_221": outputs_dir / "t0_stability_221.csv",
            "model_selection_220_vs_220221": outputs_dir / "model_selection_220_vs_220221.csv",
            "summary_221_validation": summary_path,
            "stage_summary": stage_summary_path,
        },
        extra={"inputs": input_records, "verdict": "PASS"},
    )

    contracts.log_stage_paths(SimpleNamespace(out_root=out_root, stage_dir=stage_dir, outputs_dir=outputs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
