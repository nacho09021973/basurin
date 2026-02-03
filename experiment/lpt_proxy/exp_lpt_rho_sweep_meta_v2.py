#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def find_spectrum_h5(run_dir: Path) -> Path:
    cand = sorted((run_dir / "spectrum" / "outputs").glob("*.h5"))
    if cand:
        return cand[0]
    cand = sorted((run_dir / "spectrum").rglob("*.h5"))
    if cand:
        return cand[0]
    raise FileNotFoundError(f"no spectrum h5 under {run_dir/'spectrum'}")


def extract_proxy_from_spectrum(h5_path: Path) -> Dict[str, Any]:
    """
    Contract (observed):
      datasets: delta_uv (n_delta), m2L2 (n_delta), M2 (n_delta,n_modes)
      attrs: rho_crit, family, n_delta, n_modes, ...
    Rule:
      delta* = argmin m2L2
      T_proxy = 2*pi/sqrt(m2L2(delta*))  (requires m2L2>0)
    """
    with h5py.File(h5_path, "r") as h5:
        delta_uv = h5["delta_uv"][:]
        m2L2 = h5["m2L2"][:]
        if delta_uv.shape != m2L2.shape:
            raise ValueError(f"shape mismatch: delta_uv {delta_uv.shape} vs m2L2 {m2L2.shape}")

        i = int(m2L2.argmin())
        m2 = float(m2L2[i])
        delta_star = float(delta_uv[i])

        if not (m2 > 0.0):
            return {
                "status": "OUT_OF_DOMAIN",
                "reason": f"m2L2_min<=0 (m2L2_min={m2})",
                "delta_star": delta_star,
                "m2L2_min": m2,
            }

        T_proxy = 2.0 * math.pi / math.sqrt(m2)

        attrs = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in h5.attrs.items()}
        # Keep only a small stable subset
        keep = {}
        for k in (
            "rho_crit", "family", "n_delta", "n_modes", "noise_rel", "seed",
            "alpha_min", "alpha_max", "map_mode", "profiles", "rho0", "L", "d",
            "symmetron_normalize_by_A0"
        ):
            if k in attrs:
                keep[k] = attrs[k]

        return {
            "status": "OK",
            "delta_star": delta_star,
            "m2L2_min": m2,
            "T_proxy": float(T_proxy),
            "attrs": keep,
        }


def _scan_for_c3(v: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Heurística conservadora: buscar en el JSON claves típicas.
    No asumimos schema, solo intentamos extraer números si están.
    """
    text = json.dumps(v)
    if "C3" not in text and "c3" not in text:
        return None

    out: Dict[str, Any] = {"present": True}
    # Búsqueda de campos frecuentes (si existen)
    for key in ("C3a", "c3a", "C3b", "c3b", "failure_mode", "verdict", "overall_verdict"):
        if key in v:
            out[key] = v[key]

    # Si hay estructura "contracts": {"C3": {...}}
    contracts = v.get("contracts")
    if isinstance(contracts, dict):
        c3 = contracts.get("C3")
        if isinstance(c3, dict):
            out["C3"] = c3

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Meta sweep rho_crit v2: add spectrum-derived delta* and T_proxy")
    ap.add_argument("--meta-run", required=True)
    ap.add_argument("--runs", required=True, help="comma-separated run_ids")
    args = ap.parse_args()

    meta_run = args.meta_run
    run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    if not run_ids:
        raise SystemExit("ERROR: empty --runs")

    created = datetime.now(timezone.utc).isoformat()
    meta_dir = Path("runs") / meta_run / "experiment" / "lpt_rho_sweep_meta_v2"
    outputs_dir = meta_dir / "outputs"

    rows: List[Dict[str, Any]] = []

    for rid in sorted(run_ids):
        rdir = Path("runs") / rid
        if not rdir.exists():
            raise SystemExit(f"ERROR: missing run dir: {rdir}")

        verdict, verdict_path = load_run_valid_verdict(rdir)
        if verdict != "PASS":
            raise SystemExit(f"ABORT: {rid} RUN_VALID != PASS (got {verdict}) at {verdict_path}")

        h5_path = find_spectrum_h5(rdir)
        proxy = extract_proxy_from_spectrum(h5_path)

        val_path = rdir / "dictionary" / "outputs" / "validation.json"
        val_obj = read_json(val_path) if val_path.exists() else None

        row = {
            "run": rid,
            "RUN_VALID": {"path": str(verdict_path.as_posix()), "sha256": sha256_file(verdict_path)},
            "spectrum_h5": {"path": str(h5_path.as_posix()), "sha256": sha256_file(h5_path)},
            "proxy": proxy,
            "dictionary_validation": (
                {"path": str(val_path.as_posix()), "sha256": sha256_file(val_path), "c3": _scan_for_c3(val_obj) if isinstance(val_obj, dict) else None}
                if val_path.exists()
                else None
            ),
        }
        rows.append(row)

    # Optional ranking (only OK rows)
    ok = [r for r in rows if r["proxy"].get("status") == "OK"]
    ok_sorted = sorted(ok, key=lambda r: float(r["proxy"]["T_proxy"]), reverse=True)
    ranking = [{"run": r["run"], "rho_crit": r["proxy"]["attrs"].get("rho_crit"), "delta_star": r["proxy"]["delta_star"], "T_proxy": r["proxy"]["T_proxy"]} for r in ok_sorted]

    result = {
        "schema_version": 1,
        "experiment": "lpt_rho_sweep_meta_v2",
        "created": created,
        "meta_run": meta_run,
        "runs": [r["run"] for r in rows],
        "ranking_by_T_proxy_desc": ranking,
        "rows": rows,
    }

    write_json(outputs_dir / "rho_sweep_summary_v2.json", result)

    manifest = {
        "schema_version": 1,
        "stage": "experiment/lpt_rho_sweep_meta_v2",
        "run": meta_run,
        "created": created,
        "inputs": {"runs": result["runs"]},
        "outputs": {
            "rho_sweep_summary_v2.json": {
                "path": str((outputs_dir / "rho_sweep_summary_v2.json").relative_to(Path("runs") / meta_run).as_posix()),
                "sha256": sha256_file(outputs_dir / "rho_sweep_summary_v2.json"),
            }
        },
    }
    write_json(meta_dir / "manifest.json", manifest)

    stage_summary = {
        "schema_version": 1,
        "stage": "experiment/lpt_rho_sweep_meta_v2",
        "run": meta_run,
        "created": created,
        "results": {
            "overall_verdict": "PASS",
            "n_runs": len(rows),
            "n_ok_proxy": len(ok),
        },
    }
    write_json(meta_dir / "stage_summary.json", stage_summary)

    print("PASS: wrote", (outputs_dir / "rho_sweep_summary_v2.json").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
