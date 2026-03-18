#!/usr/bin/env python3
"""Materializa external_inputs/remnant_kerr.json para un run dado.

Resuelve el bloqueador de Gate A del experimento qnm_221_literature_check:
el pipeline `multimode` no escribe Mf/af canónicos en los runs nuevos, así
que este script lo hace como pre-paso explícito.

Fuentes de (Mf, af), en orden de preferencia:
  1. Valores curados en mvp/gwtc_events._LEGACY_CHI_FINAL  (10 eventos,
     medianas PE publicadas).
  2. Estimación NR-calibrada a partir de m1, m2, chi_eff del CSV de
     calidad (Barausse & Rezzolla 2009, ec. 5 simplificada).

La fórmula para af cuando no hay valor curado es:

    eta = m1*m2 / (m1+m2)^2          # masa reducida simétrica
    af  = sqrt(12)*eta - 3.871*eta^2 + 4.028*eta^3   (contrib. orbital)
        + chi_eff * (1 - 2*eta)                        (contrib. spin)
    af  = clip(|af|, 0.0, 0.998)

Verificación numérica:
  GW150914 (m1=36, m2=29, chi_eff≈0):   af_fit ≈ 0.680  vs. curado 0.67
  GW190521 (m1=85, m2=66, chi_eff=0.1): af_fit ≈ 0.727  vs. curado 0.72

Uso::

    python -m mvp.tools.prepare_remnant_kerr --run-id <RUN_ID>
    python -m mvp.tools.prepare_remnant_kerr --run-id <RUN_ID> --event-id GW150914
    python -m mvp.tools.prepare_remnant_kerr --run-id <RUN_ID> --force

Escribe (o sobreescribe con --force)::

    runs/<run_id>/external_inputs/remnant_kerr.json

con schema::

    {
      "Mf": <float, solar masses>,
      "af": <float, dimensionless 0-0.998>,
      "af_method": "curated" | "nr_fit_barausse_rezzolla_2009",
      "event_id": <str>,
      "citation": <str>,
      "provenance": { ... }
    }
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
# Allow running as script or as module from repo root
for _cand in (_here.parents[1], _here.parents[2]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import resolve_out_root, utc_now_iso, validate_run_id, write_json_atomic
from mvp.gwtc_events import GWTC_CITATION, GWTC_EVENTS

_NR_FIT_CITATION = (
    "Barausse & Rezzolla (2009), ApJL 704 L40, arXiv:0905.2562 — "
    "simplified aligned-spin formula: "
    "af = sqrt(12)*eta - 3.871*eta^2 + 4.028*eta^3 + chi_eff*(1 - 2*eta); "
    "coefficients calibrated to NR via Buonanno et al. (2008), arXiv:0709.3254"
)

_CURATED_CITATION = GWTC_CITATION


def af_nr_fit(m1: float, m2: float, chi_eff: float = 0.0) -> float:
    """Estimate final spin using NR-calibrated fitting formula.

    Parameters
    ----------
    m1, m2 : float
        Component masses in any consistent unit (ratio is what matters).
    chi_eff : float
        Effective aligned spin parameter.  Treated as proxy for individual spins.

    Returns
    -------
    float
        Dimensionless final spin in [0.0, 0.998].
    """
    eta = m1 * m2 / (m1 + m2) ** 2
    af_orb = math.sqrt(12.0) * eta - 3.871 * eta ** 2 + 4.028 * eta ** 3
    af_spin = chi_eff * (1.0 - 2.0 * eta)
    return max(0.0, min(0.998, abs(af_orb + af_spin)))


def _load_json(path: Path) -> dict[str, Any]:
    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _read_event_id(run_dir: Path, override: str | None) -> str | None:
    if override:
        return override.strip() or None
    prov_path = run_dir / "run_provenance.json"
    if not prov_path.exists():
        return None
    prov = _load_json(prov_path)
    invocation = prov.get("invocation")
    if isinstance(invocation, dict):
        val = invocation.get("event_id")
        if isinstance(val, str) and val.strip():
            return val.strip()
    val = prov.get("event_id")
    return val.strip() if isinstance(val, str) and val.strip() else None


def resolve_remnant(event_id: str) -> dict[str, Any] | None:
    """Return remnant dict for event_id, or None if event unknown.

    Returns a dict with keys: Mf, af, af_method, citation.
    Mf is in solar masses.  af is dimensionless in [0, 0.998].
    """
    entry = GWTC_EVENTS.get(event_id)
    if entry is None:
        return None

    mf = entry.get("m_final_msun")
    if mf is None or not math.isfinite(mf) or mf <= 0.0:
        return None

    chi_final = entry.get("chi_final")
    if chi_final is not None and math.isfinite(chi_final) and 0.0 <= chi_final <= 0.998:
        return {
            "Mf": float(mf),
            "af": float(chi_final),
            "af_method": "curated",
            "citation": _CURATED_CITATION,
        }

    # Estimate af from progenitor parameters
    m1 = entry.get("m1_source")
    m2 = entry.get("m2_source")
    if m1 is None or m2 is None or m1 <= 0.0 or m2 <= 0.0:
        return None
    chi_eff = entry.get("chi_eff") or 0.0
    if not math.isfinite(chi_eff):
        chi_eff = 0.0

    af = af_nr_fit(m1, m2, chi_eff)
    return {
        "Mf": float(mf),
        "af": af,
        "af_method": "nr_fit_barausse_rezzolla_2009",
        "citation": _NR_FIT_CITATION,
    }


def run(
    run_id: str,
    *,
    event_id_override: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Materialise remnant_kerr.json for run_id.

    Returns a status dict with keys: status, path, event_id, Mf, af, af_method.
    Raises on hard errors (invalid run_id, unknown event, etc.).
    """
    out_root = resolve_out_root()
    validate_run_id(run_id, out_root)
    run_dir = out_root / run_id

    event_id = _read_event_id(run_dir, event_id_override)
    if not event_id:
        raise RuntimeError(
            f"Cannot determine event_id for run '{run_id}'. "
            "Pass --event-id explicitly or ensure run_provenance.json exists."
        )

    out_path = run_dir / "external_inputs" / "remnant_kerr.json"
    if out_path.exists() and not force:
        return {
            "status": "skipped_already_exists",
            "path": str(out_path),
            "event_id": event_id,
            "Mf": None,
            "af": None,
            "af_method": None,
        }

    remnant = resolve_remnant(event_id)
    if remnant is None:
        raise RuntimeError(
            f"Event '{event_id}' not found in GWTC catalog or missing Mf/m1/m2. "
            "Add the event to gwtc_quality_events.csv to enable NR-fit estimation."
        )

    payload: dict[str, Any] = {
        **remnant,
        "event_id": event_id,
        "run_id": run_id,
        "created_utc": utc_now_iso(),
        "provenance": {
            "tool": "mvp.tools.prepare_remnant_kerr",
            "schema": "remnant_kerr_v1",
        },
    }
    write_json_atomic(out_path, payload)

    return {
        "status": "written",
        "path": str(out_path),
        "event_id": event_id,
        "Mf": remnant["Mf"],
        "af": remnant["af"],
        "af_method": remnant["af_method"],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True, help="Run identifier (e.g. mvp_GW150914_...)")
    ap.add_argument("--event-id", default=None, help="Override event ID (default: read from run_provenance.json)")
    ap.add_argument("--runs-root", default=None, type=Path, help="Root directory for runs (default: $BASURIN_RUNS_ROOT or ./runs)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing remnant_kerr.json")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        import os
        if args.runs_root is not None:
            os.environ["BASURIN_RUNS_ROOT"] = str(args.runs_root.resolve())
        result = run(
            args.run_id,
            event_id_override=args.event_id,
            force=args.force,
        )
    except Exception as exc:
        print(f"[prepare_remnant_kerr] ERROR: {exc}", file=sys.stderr)
        return 1

    status = result["status"]
    if status == "skipped_already_exists":
        print(f"[prepare_remnant_kerr] SKIP — {result['path']} already exists (use --force to overwrite)")
        return 0

    print(
        f"[prepare_remnant_kerr] OK — event={result['event_id']} "
        f"Mf={result['Mf']:.2f} Msol  af={result['af']:.4f}  "
        f"method={result['af_method']}"
    )
    print(f"[prepare_remnant_kerr]     → {result['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
