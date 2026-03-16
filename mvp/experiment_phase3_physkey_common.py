#!/usr/bin/env python3
"""Non-canonical experiment: phase3 physical-key population intersection between batch 220 and batch 221.

Computes, reproducibly and contract-first, the physical population intersection
between two offline batches using a phys_key projection:
    phys_key = (family, provenance, M_solar, chi)

provenance_rule: metadata.source if present else metadata.ref
    with fallback to row-level source / ref.

family and provenance are normalised (lowercase + strip).
M_solar and chi are rounded to ROUND_M / ROUND_CHI decimals for the key.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
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
)
from mvp import contracts

SCHEMA_VERSION = "experiment_phase3_physkey_common_v1"
DEFAULT_OUT_NAME = "phase3_physkey_common"
DEFAULT_BATCH_RESULTS = "experiment/offline_batch/outputs/results.csv"
DEFAULT_S4_COMPATIBLE = "s4_geometry_filter/outputs/compatible_set.json"

ROUND_M = 6
ROUND_CHI = 6

# Type alias
PhysKey = tuple[str, str, float, float]


# ---------------------------------------------------------------------------
# Small helpers (local copies – avoids transversal refactor)
# ---------------------------------------------------------------------------


def _canonical(s: str) -> str:
    return s.strip().lower().replace("-", "_")


def _find_column(columns: list[str], *candidates: str) -> str | None:
    canon = {_canonical(c): c for c in columns}
    for cand in candidates:
        hit = canon.get(_canonical(cand))
        if hit:
            return hit
    return None


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Physical key extraction
# ---------------------------------------------------------------------------


def _phys_key_rounded(row: dict[str, Any]) -> PhysKey:
    """Extract physical identity tuple with normalisation and rounding.

    Uses metadata.* with fallback to row-level fields.
    Raises ValueError with an actionable message if any required field is absent.
    """
    md = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}

    fam = str(md.get("family", row.get("family", ""))).strip().lower()
    src = str(
        md.get(
            "source",
            md.get("ref", row.get("source", row.get("ref", ""))),
        )
    ).strip().lower()
    m = _as_float(md.get("M_solar", row.get("M_solar")))
    chi = _as_float(md.get("chi", row.get("chi")))

    if not fam:
        raise ValueError(
            f"Geometría sin family – no se puede construir phys_key. "
            f"geometry_id={row.get('geometry_id')!r}. "
            f"Campos disponibles: {sorted(row.keys())}"
        )
    if not src:
        raise ValueError(
            f"Geometría sin provenance (metadata.source / metadata.ref / row.source / row.ref) – "
            f"no se puede construir phys_key. "
            f"geometry_id={row.get('geometry_id')!r}. "
            f"Campos disponibles: {sorted(row.keys())}"
        )
    if m is None:
        raise ValueError(
            f"Geometría sin M_solar – no se puede construir phys_key. "
            f"geometry_id={row.get('geometry_id')!r}. "
            f"Campos disponibles: {sorted(row.keys())}"
        )
    if chi is None:
        raise ValueError(
            f"Geometría sin chi – no se puede construir phys_key. "
            f"geometry_id={row.get('geometry_id')!r}. "
            f"Campos disponibles: {sorted(row.keys())}"
        )

    return (fam, src, round(float(m), ROUND_M), round(float(chi), ROUND_CHI))


def _extract_phys_keys(compatible_set_path: Path) -> set[PhysKey]:
    """Load compatible_set.json and return the set of rounded phys_keys.

    Raises ValueError (with explicit message) on schema problems or missing fields.
    """
    data = json.loads(compatible_set_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"compatible_set.json debe ser un dict, encontrado {type(data).__name__}: "
            f"{compatible_set_path}"
        )
    geoms = data.get("compatible_geometries", [])
    if not isinstance(geoms, list):
        raise ValueError(
            f"compatible_geometries debe ser una lista en {compatible_set_path}"
        )

    keys: set[PhysKey] = set()
    for i, row in enumerate(geoms):
        if not isinstance(row, dict):
            raise ValueError(
                f"compatible_geometries[{i}] no es un dict en {compatible_set_path}: {row!r}"
            )
        key = _phys_key_rounded(row)  # raises ValueError if required fields missing
        keys.add(key)
    return keys


# ---------------------------------------------------------------------------
# results.csv loading with PASS + compatible_set existence filter
# ---------------------------------------------------------------------------


def _load_event_to_subrun_filtered(
    results_csv: Path,
    out_root: Path,
    s4_compatible_relpath: str,
) -> dict[str, str]:
    """Return event→subrun mapping, keeping only:
    - rows where status == PASS (if the column exists)
    - rows where runs/<subrun>/<s4_compatible_relpath> exists on disk.

    Raises FileNotFoundError if results_csv is absent.
    Raises ValueError if column detection fails.
    """
    if not results_csv.exists():
        raise FileNotFoundError(
            f"results.csv faltante: {results_csv}. "
            "Regenerar con: python -m mvp.experiment_offline_batch --batch-run-id <BATCH_RUN_ID>"
        )
    with results_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ValueError(f"results.csv sin encabezados: {results_csv}")
        ev_col = _find_column(fields, "event_id", "event", "event_name")
        subrun_col = _find_column(fields, "subrun_id", "run_id", "child_run_id")
        status_col = _find_column(fields, "status")
        if not ev_col or not subrun_col:
            raise ValueError(
                "No se pudieron detectar columnas event/subrun en results.csv. "
                f"ruta esperada exacta: {results_csv}. columnas disponibles: {fields}."
            )
        mapping: dict[str, str] = {}
        for row in reader:
            ev = str(row.get(ev_col, "")).strip()
            sub = str(row.get(subrun_col, "")).strip()
            if not ev or not sub:
                continue
            # Skip FAIL rows when status column present
            if status_col:
                status = str(row.get(status_col, "")).strip()
                if status != "PASS":
                    continue
            # Skip if compatible_set.json missing for this subrun
            compat_path = out_root / sub / s4_compatible_relpath
            if not compat_path.exists():
                continue
            mapping[ev] = sub
    return mapping


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _phys_key_to_str(key: PhysKey) -> str:
    """Stable string representation of a phys_key tuple."""
    return f"{key[0]}|{key[1]}|{key[2]}|{key[3]}"


def _keys_to_sorted_list(keys: set[PhysKey]) -> list[str]:
    return sorted(_phys_key_to_str(k) for k in keys)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(
    run_id: str,
    batch_220: str,
    batch_221: str,
    out_name: str = DEFAULT_OUT_NAME,
    batch_results_relpath: str = DEFAULT_BATCH_RESULTS,
    s4_compatible_relpath: str = DEFAULT_S4_COMPATIBLE,
) -> dict[str, Any]:
    out_root = resolve_out_root("runs")
    validate_run_id(run_id, out_root)
    require_run_valid(out_root, run_id)
    require_run_valid(out_root, batch_220)
    require_run_valid(out_root, batch_221)

    run_dir = out_root / run_id

    results_220_path = out_root / batch_220 / batch_results_relpath
    results_221_path = out_root / batch_221 / batch_results_relpath

    # Load filtered mappings (PASS + compatible_set.json present)
    map220 = _load_event_to_subrun_filtered(results_220_path, out_root, s4_compatible_relpath)
    map221 = _load_event_to_subrun_filtered(results_221_path, out_root, s4_compatible_relpath)

    valid_events_220 = set(map220.keys())
    valid_events_221 = set(map221.keys())
    common_events = sorted(valid_events_220 & valid_events_221)

    # Per-event computation + global accumulation
    K220_global: set[PhysKey] = set()
    K221_global: set[PhysKey] = set()
    per_event_rows: list[dict[str, Any]] = []
    empty_intersection_events: list[str] = []
    non_subset_cases: list[str] = []

    for event in common_events:
        sub220 = map220[event]
        sub221 = map221[event]
        path220 = out_root / sub220 / s4_compatible_relpath
        path221 = out_root / sub221 / s4_compatible_relpath

        # _extract_phys_keys raises ValueError on schema/field errors → propagate (abort)
        k220 = _extract_phys_keys(path220)
        k221 = _extract_phys_keys(path221)
        k_inter = k220 & k221

        K220_global |= k220
        K221_global |= k221

        if not k_inter:
            empty_intersection_events.append(event)
        if not k220.issubset(k221):
            non_subset_cases.append(event)

        per_event_rows.append(
            {
                "event_id": event,
                "k220": len(k220),
                "k221": len(k221),
                "k_inter": len(k_inter),
                "run_id_220": sub220,
                "run_id_221": sub221,
            }
        )

    K220_inter_K221 = K220_global & K221_global

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "batch_220": batch_220,
        "batch_221": batch_221,
        "phys_key": ["family", "provenance", "M_solar", "chi"],
        "provenance_rule": "metadata.source if present else metadata.ref",
        "round_M": ROUND_M,
        "round_chi": ROUND_CHI,
        "n_events_valid_220": len(valid_events_220),
        "n_events_valid_221": len(valid_events_221),
        "n_common_events": len(common_events),
        "K220": _keys_to_sorted_list(K220_global),
        "K221": _keys_to_sorted_list(K221_global),
        "K220_inter_K221": _keys_to_sorted_list(K220_inter_K221),
        "empty_intersection_events": sorted(empty_intersection_events),
        "n_empty_intersection_events": len(empty_intersection_events),
        "non_subset_cases": sorted(non_subset_cases),
        "n_non_subset_cases": len(non_subset_cases),
    }

    # Atomic write under runs/<run_id>/experiment/<out_name>/
    exp_dir = run_dir / "experiment" / out_name
    with tempfile.TemporaryDirectory(prefix=f"{out_name}_", dir=run_dir) as tmpdir:
        tmp_stage = Path(tmpdir) / "stage"
        tmp_outputs = tmp_stage / "outputs"
        tmp_outputs.mkdir(parents=True, exist_ok=True)

        # 1. summary_physkey_common.json
        out_summary = tmp_outputs / "summary_physkey_common.json"
        write_json_atomic(out_summary, summary)

        # 2. per_event_physkey_intersection.csv
        out_csv = tmp_outputs / "per_event_physkey_intersection.csv"
        csv_fields = ["event_id", "k220", "k221", "k_inter", "run_id_220", "run_id_221"]
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(per_event_rows)

        outputs = [out_summary, out_csv]
        output_records = [
            {"path": str(p.relative_to(tmp_stage)), "sha256": sha256_file(p)}
            for p in outputs
        ]

        # 3. manifest.json
        manifest_payload: dict[str, Any] = {
            "schema_version": "mvp_manifest_v1",
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "artifacts": output_records,
            "inputs": [
                {"path": str(results_220_path), "sha256": sha256_file(results_220_path)},
                {"path": str(results_221_path), "sha256": sha256_file(results_221_path)},
            ],
        }
        write_json_atomic(tmp_stage / "manifest.json", manifest_payload)

        # 4. stage_summary.json
        stage_summary: dict[str, Any] = {
            "status": "PASS",
            "stage": f"experiment/{out_name}",
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "inputs": manifest_payload["inputs"],
            "outputs": output_records,
            "metrics": {
                "n_common_events": len(common_events),
                "n_K220": len(K220_global),
                "n_K221": len(K221_global),
                "n_K220_inter_K221": len(K220_inter_K221),
                "n_empty_intersection_events": len(empty_intersection_events),
                "n_non_subset_cases": len(non_subset_cases),
            },
        }
        write_json_atomic(tmp_stage / "stage_summary.json", stage_summary)

        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        exp_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_stage), str(exp_dir))

    contracts.log_stage_paths(
        SimpleNamespace(
            out_root=out_root,
            stage_dir=exp_dir,
            outputs_dir=exp_dir / "outputs",
        )
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Phase-3 physical-key population intersection between "
            "two offline batches (220 and 221)."
        )
    )
    ap.add_argument("--run-id", required=True, help="Analysis run ID")
    ap.add_argument("--batch-220", required=True, help="Batch run ID for mode-220")
    ap.add_argument("--batch-221", required=True, help="Batch run ID for mode-221")
    ap.add_argument(
        "--out-name",
        default=DEFAULT_OUT_NAME,
        help=f"Output experiment name (default: {DEFAULT_OUT_NAME})",
    )
    ap.add_argument(
        "--batch-results-relpath",
        default=DEFAULT_BATCH_RESULTS,
        help=f"Relative path to results.csv inside each batch run (default: {DEFAULT_BATCH_RESULTS})",
    )
    ap.add_argument(
        "--s4-compatible-relpath",
        default=DEFAULT_S4_COMPATIBLE,
        help=(
            f"Relative path to compatible_set.json inside each sub-run "
            f"(default: {DEFAULT_S4_COMPATIBLE})"
        ),
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    run_experiment(
        run_id=args.run_id,
        batch_220=args.batch_220,
        batch_221=args.batch_221,
        out_name=args.out_name,
        batch_results_relpath=args.batch_results_relpath,
        s4_compatible_relpath=args.s4_compatible_relpath,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
