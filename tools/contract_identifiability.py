#!/usr/bin/env python3
"""BASURIN — Contract: IDENTIFIABILITY (trivalent verdict)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from basurin_io import (
    ensure_stage_dirs,
    get_run_dir,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

__version__ = "0.1.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Config:
    run: str
    input_path: str | None
    tau_deg: float
    tau_stab: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Contrato IDENTIFIABILITY: veredictos trivalentes")
    parser.add_argument("--run", required=True, type=str, help="Run ID")
    parser.add_argument("--in", dest="input_path", default=None, help="Ruta opcional al degeneracy_per_point.json")
    parser.add_argument("--tau-deg", type=float, default=100.0, help="Threshold degeneracy (default: 100.0)")
    parser.add_argument("--tau-stab", type=float, default=0.8, help="Threshold stability (default: 0.8)")
    args = parser.parse_args()
    return Config(
        run=args.run,
        input_path=args.input_path,
        tau_deg=float(args.tau_deg),
        tau_stab=float(args.tau_stab),
    )


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_input_path(run_dir: Path, input_path: str | None) -> Path:
    if input_path:
        candidate = Path(input_path)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return candidate

    canonical = run_dir / "bridge_f4_1_alignment" / "outputs" / "degeneracy_per_point.json"
    legacy = run_dir / "bridge_f4_1_alignment" / "degeneracy_per_point.json"
    if canonical.exists():
        return canonical
    if legacy.exists():
        return legacy
    return canonical


def _extract_per_point(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        per_point = payload.get("per_point")
        if isinstance(per_point, list):
            return [row for row in per_point if isinstance(row, dict)]
    return []


def _extract_values(rows: Iterable[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        for key in ("cond_local", "degeneracy_index"):
            if key in row:
                try:
                    values.append(float(row[key]))
                except (TypeError, ValueError):
                    continue
                break
    return values


def _extract_stability(payload: Any) -> tuple[float | None, str | None]:
    if not isinstance(payload, dict):
        return None, None
    for key in ("stability_score", "sigma_Delta_max", "sigma_delta_max"):
        if key in payload:
            try:
                return float(payload[key]), key
            except (TypeError, ValueError):
                return None, key
    return None, None


def main() -> int:
    cfg = parse_args()

    try:
        out_root = resolve_out_root("runs")
        validate_run_id(cfg.run, out_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    run_dir = get_run_dir(cfg.run, base_dir=out_root)
    stage_dir, outputs_dir = ensure_stage_dirs(cfg.run, "IDENTIFIABILITY", base_dir=out_root)
    input_path = _resolve_input_path(run_dir, cfg.input_path)

    errors: list[str] = []
    payload: Any | None = None
    if not input_path.exists():
        errors.append(f"input missing: {input_path}")
    else:
        try:
            payload = _read_json(input_path)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid json: {input_path} ({exc})")

    degeneracy_index = None
    degeneracy_source = "derived"
    summary_max = None
    summary_mean = None
    if payload is not None:
        if isinstance(payload, dict) and "degeneracy_index" in payload:
            try:
                degeneracy_index = float(payload["degeneracy_index"])
                degeneracy_source = "input"
            except (TypeError, ValueError):
                errors.append("degeneracy_index inválido")
        else:
            rows = _extract_per_point(payload)
            values = _extract_values(rows)
            if values:
                summary_max = max(values)
                summary_mean = sum(values) / len(values)
                degeneracy_index = summary_max
            else:
                errors.append("no se pudo derivar degeneracy_index")

    stability_score, stability_key = _extract_stability(payload)
    stability_status = "OK" if stability_score is not None else "UNKNOWN"

    if errors:
        verdict = "FAIL"
        pipeline_status = "ERROR_IO"
        scientific_status = "UNKNOWN"
    else:
        if degeneracy_index is None:
            verdict = "FAIL"
            pipeline_status = "ERROR_IO"
            scientific_status = "UNKNOWN"
        elif degeneracy_index >= cfg.tau_deg and (
            stability_score is None or stability_score >= cfg.tau_stab
        ):
            verdict = "UNDERDETERMINED"
            pipeline_status = "OK"
            scientific_status = "NOT_IDENTIFIABLE_UNDER_CURRENT_OBSERVABLES"
        else:
            verdict = "PASS"
            pipeline_status = "OK"
            scientific_status = "IDENTIFIABLE_UNDER_CURRENT_OBSERVABLES"

    report = {
        "stage": "IDENTIFIABILITY",
        "version": __version__,
        "created": utc_now_iso(),
        "run": cfg.run,
        "verdict": verdict,
        "pipeline_status": pipeline_status,
        "scientific_status": scientific_status,
        "thresholds": {
            "tau_deg": cfg.tau_deg,
            "tau_stab": cfg.tau_stab,
        },
        "evidence": {
            "degeneracy_index": {
                "value": degeneracy_index,
                "source": degeneracy_source,
                "summary": {
                    "max": summary_max,
                    "mean": summary_mean,
                },
            },
            "stability_score": {
                "value": stability_score,
                "status": stability_status,
                "source": stability_key,
            },
        },
        "inputs": {
            "degeneracy_per_point": str(input_path),
        },
    }
    if errors:
        report["errors"] = errors

    output_path = outputs_dir / "identifiability.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    stage_summary = {
        "stage": "IDENTIFIABILITY",
        "run": cfg.run,
        "version": __version__,
        "created": utc_now_iso(),
        "config": {
            "run": cfg.run,
            "input": str(input_path),
            "tau_deg": cfg.tau_deg,
            "tau_stab": cfg.tau_stab,
        },
        "hashes": {
            "outputs/identifiability.json": sha256_file(output_path),
        },
        "results": {
            "verdict": verdict,
        },
    }
    stage_summary_path = write_stage_summary(stage_dir, stage_summary)

    manifest_path = write_manifest(
        stage_dir,
        {
            "identifiability": output_path,
            "summary": stage_summary_path,
        },
    )
    manifest_payload = _read_json(manifest_path)
    if "manifest" not in manifest_payload.get("files", {}):
        manifest_payload.setdefault("files", {})["manifest"] = "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    return 0 if verdict != "FAIL" else 2


if __name__ == "__main__":
    sys.exit(main())
