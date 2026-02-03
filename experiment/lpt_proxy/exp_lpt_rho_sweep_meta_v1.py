#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_run_valid_verdict(run_dir: Path) -> Tuple[str, Path]:
    preferred = run_dir / "RUN_VALID" / "verdict.json"
    legacy = run_dir / "RUN_VALID" / "outputs" / "run_valid.json"
    if preferred.exists():
        j = read_json(preferred)
        return str(j.get("verdict", "")).upper(), preferred
    if legacy.exists():
        j = read_json(legacy)
        return str(j.get("verdict", j.get("overall_verdict", ""))).upper(), legacy
    return "MISSING", preferred


def find_spectrum_h5(run_dir: Path) -> Optional[Path]:
    # Canonical preference: runs/<run>/spectrum/outputs/*.h5
    cand = list((run_dir / "spectrum" / "outputs").glob("*.h5"))
    if cand:
        return sorted(cand)[0]
    # Fallback: anything under spectrum/
    cand = list((run_dir / "spectrum").rglob("*.h5"))
    if cand:
        return sorted(cand)[0]
    return None


def read_rho_crit_attr(h5_path: Path) -> Optional[float]:
    try:
        with h5py.File(h5_path, "r") as h5:
            if "rho_crit" in h5.attrs:
                return float(h5.attrs["rho_crit"])
    except Exception:
        return None
    return None


def extract_validation_summary(v: Dict[str, Any]) -> Dict[str, Any]:
    # Estructura flexible: no asumimos schema rígido; extraemos campos típicos
    out: Dict[str, Any] = {}

    # C1/C2/C3 pueden vivir en distintos sitios. Intentamos varias rutas.
    candidates = []
    if isinstance(v.get("contracts"), dict):
        candidates.append(v["contracts"])
    if isinstance(v.get("results"), dict):
        candidates.append(v["results"])
    if isinstance(v.get("checks"), dict):
        candidates.append(v["checks"])

    # También puede haber lista
    if isinstance(v.get("contracts"), list):
        out["contracts_list"] = v["contracts"]

    # Scan heurístico (conservador)
    text = json.dumps(v)
    for key in ("C1", "C2", "C3"):
        if key in text:
            out[f"has_{key}"] = True

    # Campos comunes
    for k in ("overall_verdict", "verdict", "profile", "run_kind"):
        if k in v:
            out[k] = v[k]

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Meta sweep rho_crit (phase1): aggregate RUN_VALID + dictionary validation + rho_crit attr")
    ap.add_argument("--meta-run", required=True)
    ap.add_argument("--runs", required=True, help="comma-separated run_ids")
    args = ap.parse_args()

    meta_run = args.meta_run
    run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    if not run_ids:
        raise SystemExit("ERROR: empty --runs")

    created = datetime.now(timezone.utc).isoformat()
    meta_dir = Path("runs") / meta_run / "experiment" / "lpt_rho_sweep_meta_v1"
    outputs_dir = meta_dir / "outputs"

    rows: List[Dict[str, Any]] = []
    inputs: Dict[str, Any] = {}

    for rid in run_ids:
        rdir = Path("runs") / rid
        if not rdir.exists():
            raise SystemExit(f"ERROR: missing run dir: {rdir}")

        verdict, verdict_path = load_run_valid_verdict(rdir)
        if verdict != "PASS":
            raise SystemExit(f"ABORT: {rid} RUN_VALID != PASS (got {verdict}) at {verdict_path}")

        # dictionary validation
        val_path = rdir / "dictionary" / "outputs" / "validation.json"
        val_obj = None
        if val_path.exists():
            val_obj = read_json(val_path)

        # spectrum rho_crit
        h5_path = find_spectrum_h5(rdir)
        rho_crit = read_rho_crit_attr(h5_path) if h5_path else None

        row = {
            "run": rid,
            "RUN_VALID": {
                "path": str(verdict_path.as_posix()),
                "sha256": sha256_file(verdict_path),
            },
            "dictionary_validation": (
                {
                    "path": str(val_path.as_posix()),
                    "sha256": sha256_file(val_path),
                    "summary": extract_validation_summary(val_obj) if isinstance(val_obj, dict) else None,
                }
                if val_path.exists()
                else None
            ),
            "spectrum_h5": (
                {"path": str(h5_path.as_posix()), "sha256": sha256_file(h5_path), "rho_crit_attr": rho_crit}
                if h5_path
                else None
            ),
        }
        rows.append(row)

    # Deterministic ordering
    rows = sorted(rows, key=lambda r: r["run"])

    result = {
        "schema_version": 1,
        "experiment": "lpt_rho_sweep_meta_v1",
        "created": created,
        "meta_run": meta_run,
        "runs": [r["run"] for r in rows],
        "rows": rows,
    }

    write_json(outputs_dir / "rho_sweep_summary.json", result)

    manifest = {
        "schema_version": 1,
        "stage": "experiment/lpt_rho_sweep_meta_v1",
        "run": meta_run,
        "created": created,
        "inputs": {"runs": result["runs"]},
        "outputs": {
            "rho_sweep_summary.json": {
                "path": str((outputs_dir / "rho_sweep_summary.json").relative_to(Path("runs") / meta_run).as_posix()),
                "sha256": sha256_file(outputs_dir / "rho_sweep_summary.json"),
            }
        },
    }
    write_json(meta_dir / "manifest.json", manifest)

    stage_summary = {
        "schema_version": 1,
        "stage": "experiment/lpt_rho_sweep_meta_v1",
        "run": meta_run,
        "created": created,
        "results": {"overall_verdict": "PASS", "n_runs": len(rows)},
    }
    write_json(meta_dir / "stage_summary.json", stage_summary)

    print("PASS: wrote", (outputs_dir / "rho_sweep_summary.json").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
