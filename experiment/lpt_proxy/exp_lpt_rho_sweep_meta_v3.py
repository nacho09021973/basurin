#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def extract_proxy_from_spectrum_M2(h5_path: Path) -> Dict[str, Any]:
    """
    Contract observed:
      datasets: delta_uv (n_delta), M2 (n_delta,n_modes)
      attrs: rho_crit, family, ...
    Rule v3:
      m_eff(delta) = M2[delta,0]  (modo fundamental)
      choose delta* = argmin m_eff over m_eff>eps
      T_proxy = 2*pi/sqrt(m_eff(delta*))
    """
    eps = 1e-12
    with h5py.File(h5_path, "r") as h5:
        delta_uv = h5["delta_uv"][:]
        M2 = h5["M2"][:]
        if M2.shape[0] != delta_uv.shape[0]:
            raise ValueError(f"shape mismatch: M2 {M2.shape} vs delta_uv {delta_uv.shape}")

        m_eff = M2[:, 0]
        mask = m_eff > eps
        if not mask.any():
            return {
                "status": "OUT_OF_DOMAIN",
                "reason": "no M2[:,0] values > eps",
                "M2_mode0_min": float(m_eff.min()),
                "M2_mode0_max": float(m_eff.max()),
            }

        i_rel = int((m_eff[mask]).argmin())
        i = int((mask.nonzero()[0])[i_rel])

        m2 = float(m_eff[i])
        delta_star = float(delta_uv[i])
        T_proxy = 2.0 * math.pi / math.sqrt(m2)

        attrs = dict(h5.attrs.items())
        keep = {}
        for k in (
            "rho_crit", "family", "n_delta", "n_modes", "noise_rel", "seed",
            "alpha_min", "alpha_max", "map_mode", "profiles", "rho0", "L", "d",
            "symmetron_normalize_by_A0"
        ):
            if k in attrs:
                v = attrs[k]
                try:
                    keep[k] = float(v)
                except Exception:
                    keep[k] = v

        return {
            "status": "OK",
            "delta_star": delta_star,
            "m_eff": m2,
            "T_proxy": float(T_proxy),
            "attrs": keep,
            "definition": "m_eff = M2[:,0] (fundamental mode)",
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="Meta sweep rho_crit v3: proxy from M2[:,0]")
    ap.add_argument("--meta-run", required=True)
    ap.add_argument("--runs", required=True, help="comma-separated run_ids")
    args = ap.parse_args()

    meta_run = args.meta_run
    run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    if not run_ids:
        raise SystemExit("ERROR: empty --runs")

    created = datetime.now(timezone.utc).isoformat()
    meta_dir = Path("runs") / meta_run / "experiment" / "lpt_rho_sweep_meta_v3"
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
        proxy = extract_proxy_from_spectrum_M2(h5_path)

        rows.append(
            {
                "run": rid,
                "RUN_VALID": {"path": str(verdict_path.as_posix()), "sha256": sha256_file(verdict_path)},
                "spectrum_h5": {"path": str(h5_path.as_posix()), "sha256": sha256_file(h5_path)},
                "proxy": proxy,
            }
        )

    ok = [r for r in rows if r["proxy"].get("status") == "OK"]
    ok_sorted = sorted(ok, key=lambda r: float(r["proxy"]["T_proxy"]), reverse=True)
    ranking = [
        {
            "run": r["run"],
            "rho_crit": r["proxy"]["attrs"].get("rho_crit"),
            "delta_star": r["proxy"]["delta_star"],
            "T_proxy": r["proxy"]["T_proxy"],
            "m_eff": r["proxy"]["m_eff"],
        }
        for r in ok_sorted
    ]

    result = {
        "schema_version": 1,
        "experiment": "lpt_rho_sweep_meta_v3",
        "created": created,
        "meta_run": meta_run,
        "runs": [r["run"] for r in rows],
        "ranking_by_T_proxy_desc": ranking,
        "rows": rows,
    }

    write_json(outputs_dir / "rho_sweep_summary_v3.json", result)

    manifest = {
        "schema_version": 1,
        "stage": "experiment/lpt_rho_sweep_meta_v3",
        "run": meta_run,
        "created": created,
        "inputs": {"runs": result["runs"]},
        "outputs": {
            "rho_sweep_summary_v3.json": {
                "path": str((outputs_dir / "rho_sweep_summary_v3.json").relative_to(Path("runs") / meta_run).as_posix()),
                "sha256": sha256_file(outputs_dir / "rho_sweep_summary_v3.json"),
            }
        },
    }
    write_json(meta_dir / "manifest.json", manifest)

    stage_summary = {
        "schema_version": 1,
        "stage": "experiment/lpt_rho_sweep_meta_v3",
        "run": meta_run,
        "created": created,
        "results": {"overall_verdict": "PASS", "n_runs": len(rows), "n_ok_proxy": len(ok)},
    }
    write_json(meta_dir / "stage_summary.json", stage_summary)

    print("PASS: wrote", (outputs_dir / "rho_sweep_summary_v3.json").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
