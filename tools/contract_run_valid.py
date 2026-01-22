#!/usr/bin/env python3
"""BASURIN — Contract: RUN_VALID executive gate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from basurin_io import (
    ensure_stage_dirs,
    get_run_dir,
    sha256_file,
    write_manifest,
    write_stage_summary,
)


@dataclass
class CheckResult:
    id: str
    ok: bool
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrato ejecutivo RUN_VALID")
    parser.add_argument("--run", required=True, help="Run ID")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stage_summary_path(run_dir: Path, stage: str) -> Path:
    return run_dir / stage / "stage_summary.json"


def _manifest_path(run_dir: Path, stage: str) -> Path:
    return run_dir / stage / "manifest.json"


def _is_within_stage(stage_dir: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(stage_dir.resolve())
    except ValueError:
        return False
    return True


def _check_manifest_paths(run_dir: Path, stage: str) -> tuple[bool, str]:
    manifest_path = _manifest_path(run_dir, stage)
    if not manifest_path.exists():
        return False, f"manifest.json faltante en {stage}"
    manifest = _read_json(manifest_path)
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        return False, f"manifest.json inválido en {stage}"
    stage_dir = run_dir / stage
    for label, rel in files.items():
        candidate = stage_dir / rel
        if not candidate.exists():
            return False, f"{stage} manifest referencia {label} que no existe: {candidate}"
        if not _is_within_stage(stage_dir, candidate):
            return False, f"{stage} manifest referencia {label} fuera del stage: {candidate}"
    return True, "ok"


def _check_stage_summary_hash(summary_path: Path, rel_key: str, actual_path: Path) -> tuple[bool, str]:
    summary = _read_json(summary_path)
    hashes = summary.get("hashes", {})
    if rel_key not in hashes:
        return False, f"hashes sin clave {rel_key} en {summary_path}"
    if not actual_path.exists():
        return False, f"faltante {actual_path}"
    return True, "ok"


def _get_run_kind(summary_path: Path) -> str:
    if not summary_path.exists():
        return "geometry_pipeline"
    summary = _read_json(summary_path)
    config = summary.get("config", {})
    inputs = summary.get("inputs", {})
    return (
        config.get("run_kind")
        or inputs.get("run_kind")
        or summary.get("run_kind")
        or "geometry_pipeline"
    )


def _check_geometry_verdict(run_dir: Path) -> tuple[bool, str]:
    geometry_validation = run_dir / "geometry" / "outputs" / "validation.json"
    if geometry_validation.exists():
        payload = _read_json(geometry_validation)
        verdict = payload.get("verdict")
        if verdict and verdict != "PASS":
            return False, f"validation.json verdict {verdict}"
    geometry_contracts_summary = run_dir / "geometry_contracts" / "stage_summary.json"
    if geometry_contracts_summary.exists():
        payload = _read_json(geometry_contracts_summary)
        verdict = payload.get("verdict") or payload.get("results", {}).get("verdict")
        if verdict and verdict != "PASS":
            return False, f"geometry_contracts verdict {verdict}"
    return True, "ok"


def _require_h5_datasets(path: Path, datasets: Iterable[str]) -> tuple[bool, str]:
    try:
        import h5py
    except ImportError:
        return False, "h5py no disponible"

    with h5py.File(path, "r") as h5:
        for name in datasets:
            if name not in h5:
                return False, f"dataset faltante {name} en {path}"
    return True, "ok"


def main() -> int:
    args = parse_args()
    run_dir = get_run_dir(args.run)
    spectrum_summary = _stage_summary_path(run_dir, "spectrum")
    run_kind = _get_run_kind(spectrum_summary)

    checks: list[CheckResult] = []
    required = [
        ("spectrum", _manifest_path(run_dir, "spectrum")),
        ("spectrum", spectrum_summary),
        ("dictionary", _manifest_path(run_dir, "dictionary")),
        ("dictionary", _stage_summary_path(run_dir, "dictionary")),
    ]
    if run_kind != "spectrum_only":
        required.extend(
            [
                ("geometry", _manifest_path(run_dir, "geometry")),
                ("geometry", _stage_summary_path(run_dir, "geometry")),
            ]
        )
    missing = [str(path) for _, path in required if not path.exists()]
    if missing:
        checks.append(CheckResult("E1", False, f"faltan archivos: {missing}"))
    else:
        checks.append(CheckResult("E1", True, "ok"))

    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    spectrum_legacy = run_dir / "spectrum" / "spectrum.h5"
    if spectrum_path.exists():
        resolved_spectrum = spectrum_path
    elif spectrum_legacy.exists():
        resolved_spectrum = spectrum_legacy
    else:
        resolved_spectrum = spectrum_path
    dictionary_path = run_dir / "dictionary" / "outputs" / "dictionary.h5"

    if resolved_spectrum.exists() and dictionary_path.exists():
        checks.append(CheckResult("E2", True, "ok"))
    else:
        checks.append(
            CheckResult(
                "E2",
                False,
                f"spectrum o dictionary faltante: {resolved_spectrum}, {dictionary_path}",
            )
        )

    stage_manifest_ok = True
    reasons: list[str] = []
    manifest_targets = ["spectrum", "dictionary"]
    if run_kind != "spectrum_only":
        manifest_targets.insert(0, "geometry")
    for stage in manifest_targets:
        ok, reason = _check_manifest_paths(run_dir, stage)
        if not ok:
            stage_manifest_ok = False
            reasons.append(reason)
    checks.append(CheckResult("E3", stage_manifest_ok, "; ".join(reasons) if reasons else "ok"))

    dict_summary = _stage_summary_path(run_dir, "dictionary")
    e4_ok = True
    e4_reasons: list[str] = []
    if resolved_spectrum.exists() and spectrum_summary.exists():
        ok, reason = _check_stage_summary_hash(
            spectrum_summary,
            "outputs/spectrum.h5",
            run_dir / "spectrum" / "outputs" / "spectrum.h5",
        )
        if not ok:
            e4_ok = False
            e4_reasons.append(reason)
    else:
        e4_ok = False
        e4_reasons.append("spectrum o summary faltante")
    if dict_summary.exists():
        ok, reason = _check_stage_summary_hash(
            dict_summary,
            "outputs/dictionary.h5",
            dictionary_path,
        )
        if not ok:
            e4_ok = False
            e4_reasons.append(reason)
    else:
        e4_ok = False
        e4_reasons.append("dictionary summary faltante")
    checks.append(CheckResult("E4", e4_ok, "; ".join(e4_reasons)))

    geometry_path_value = None
    if spectrum_summary.exists():
        summary = _read_json(spectrum_summary)
        inputs = summary.get("inputs", {})
        config = summary.get("config", {})
        geometry_path_value = inputs.get("geometry_path")
        if run_kind == "spectrum_only":
            generator_script = inputs.get("generator_script") or inputs.get("generator")
            seed_value = inputs.get("generator_seed", config.get("seed"))
            if generator_script and seed_value is not None:
                checks.append(CheckResult("E5", True, "ok"))
            else:
                checks.append(
                    CheckResult(
                        "E5",
                        False,
                        "inputs.generator_script/generator o seed faltante para spectrum_only",
                    )
                )
        else:
            if inputs.get("geometry_path") and inputs.get("geometry_sha256"):
                checks.append(CheckResult("E5", True, "ok"))
            else:
                checks.append(CheckResult("E5", False, "inputs.geometry_path o geometry_sha256 faltante"))
    else:
        checks.append(CheckResult("E5", False, "stage_summary spectrum faltante"))

    e6_ok = True
    e6_reasons: list[str] = []
    if resolved_spectrum.exists():
        ok, reason = _require_h5_datasets(resolved_spectrum, ["M2", "delta_uv"])
        if not ok:
            e6_ok = False
            e6_reasons.append(reason)
    else:
        e6_ok = False
        e6_reasons.append("spectrum.h5 faltante")
    if run_kind == "spectrum_only":
        spectrum_validation = run_dir / "spectrum" / "outputs" / "validation.json"
        if not spectrum_validation.exists():
            e6_ok = False
            e6_reasons.append("validation.json faltante en spectrum")
    if dictionary_path.exists():
        ok, reason = _require_h5_datasets(dictionary_path, [])
        if not ok:
            e6_ok = False
            e6_reasons.append(reason)
    else:
        e6_ok = False
        e6_reasons.append("dictionary.h5 faltante")
    checks.append(CheckResult("E6", e6_ok, "; ".join(e6_reasons)))

    if run_kind == "spectrum_only":
        checks.append(CheckResult("E7", True, "skip: spectrum_only"))
    else:
        ok, reason = _check_geometry_verdict(run_dir)
        checks.append(CheckResult("E7", ok, reason))

    verdict = "PASS" if all(c.ok for c in checks) else "FAIL"
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, "RUN_VALID")
    run_valid_path = outputs_dir / "run_valid.json"
    advisories: dict[str, Any] = {}
    identifiability_path = run_dir / "IDENTIFIABILITY" / "outputs" / "identifiability.json"
    if identifiability_path.exists():
        identifiability_payload = _read_json(identifiability_path)
        advisories["identifiability"] = {
            "path": str(identifiability_path),
            "verdict": identifiability_payload.get("verdict"),
            "scientific_status": identifiability_payload.get("scientific_status"),
        }
    run_valid_payload = {
        "run": args.run,
        "verdict": verdict,
        "checks": [
            {"id": c.id, "ok": c.ok, "reason": c.reason} for c in checks
        ],
        "advisories": advisories,
        "inputs": {
            "geometry_path": geometry_path_value,
            "spectrum_path": str(resolved_spectrum),
            "dictionary_path": str(dictionary_path),
        },
    }

    with open(run_valid_path, "w", encoding="utf-8") as f:
        json.dump(run_valid_payload, f, indent=2)

    stage_summary = {
        "stage": "RUN_VALID",
        "run": args.run,
        "version": "1.0.0",
        "config": {"run": args.run},
        "hashes": {
            "outputs/run_valid.json": sha256_file(run_valid_path),
        },
        "results": {"verdict": verdict},
        "validation_summary": {
            "total": len(checks),
            "passed": sum(1 for c in checks if c.ok),
        },
    }
    stage_summary_path = write_stage_summary(stage_dir, stage_summary)

    manifest_path = write_manifest(
        stage_dir,
        {
            "run_valid": run_valid_path,
            "summary": stage_summary_path,
        },
    )
    manifest_payload = _read_json(manifest_path)
    if "manifest" not in manifest_payload.get("files", {}):
        manifest_payload.setdefault("files", {})["manifest"] = "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
