#!/usr/bin/env python3
"""
select_sentinel_events.py
─────────────────────────
Selecciona 10 eventos centinela para el A/B de topología s3b
(rigid_spectral_split vs shared_band_early_taper).

Criterio de estratificación (congelado):
  Bloque A  — 4 eventos: top-4 SNR de la cohorte.
  Bloque B  — 3 eventos: cuartil Q3 de SNR, diversidad en Mf_source.
  Bloque C  — 3 eventos: cuartil Q2 de SNR (controles negativos).

Ejes: SNR, Mf_source, af (chi_eff como fallback si af=null), len(detectors).

Uso:
  python select_sentinel_events.py \
      --catalog /path/to/gwtc_events_t0.json \
      --cohort-dir /path/to/runs/<prepare_run_id>/outputs/events/ \
      [--output sentinel_selection.json]

El script NO modifica nada. Solo emite la selección con justificación.
"""

import argparse
import json
import sys
from pathlib import Path


def load_catalog(path: Path) -> dict:
    """Carga catálogo canónico. Devuelve dict {event_name: {...}}."""
    with open(path) as f:
        data = json.load(f)
    # Soporta tanto lista como dict
    if isinstance(data, list):
        return {e["event"]: e for e in data}
    elif isinstance(data, dict):
        # Si es dict directo {event_name: {fields...}}
        # o si tiene una clave wrapper
        first_val = next(iter(data.values()))
        if isinstance(first_val, dict):
            return data
    raise ValueError(f"Formato de catálogo no reconocido en {path}")


def discover_cohort_events(cohort_dir: Path) -> list[str]:
    """
    Descubre los eventos de la cohorte desde el directorio de prepare_events.
    Intenta varias estrategias:
      1. Leer manifest.json si existe
      2. Leer event_list.json si existe
      3. Listar subdirectorios como nombres de evento
    """
    manifest = cohort_dir / "manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            m = json.load(f)
        # Busca lista de eventos en claves comunes
        for key in ("events", "event_list", "event_names"):
            if key in m and isinstance(m[key], list):
                return m[key]

    event_list = cohort_dir / "event_list.json"
    if event_list.exists():
        with open(event_list) as f:
            return json.load(f)

    # Fallback: subdirectorios
    dirs = sorted([d.name for d in cohort_dir.iterdir() if d.is_dir()])
    if dirs:
        return dirs

    raise FileNotFoundError(
        f"No pude descubrir eventos de cohorte en {cohort_dir}. "
        "Pasa --cohort-events con lista explícita si la estructura es diferente."
    )


def get_spin(event_data: dict) -> tuple[float | None, str]:
    """Devuelve (valor_spin, fuente) usando af primario, chi_eff fallback."""
    af = event_data.get("af")
    if af is not None:
        try:
            return float(af), "af"
        except (TypeError, ValueError):
            pass
    chi = event_data.get("chi_eff")
    if chi is not None:
        try:
            return float(chi), "chi_eff"
        except (TypeError, ValueError):
            pass
    return None, "none"


def get_n_detectors(event_data: dict) -> int:
    """Cuenta detectores del evento."""
    det = event_data.get("detectors", [])
    if isinstance(det, str):
        return len(det.split(","))
    if isinstance(det, list):
        return len(det)
    return 0


def select_sentinels(catalog: dict, cohort_events: list[str]) -> dict:
    """
    Aplica criterio de estratificación y devuelve selección con justificación.
    """
    # Filtrar catálogo a cohorte
    cohort = []
    missing = []
    for ev in cohort_events:
        if ev in catalog:
            entry = catalog[ev].copy()
            entry["event"] = ev
            cohort.append(entry)
        else:
            missing.append(ev)

    if missing:
        print(f"WARN: {len(missing)} eventos de cohorte no encontrados en catálogo: "
              f"{missing[:5]}{'...' if len(missing)>5 else ''}", file=sys.stderr)

    if len(cohort) < 10:
        raise ValueError(f"Solo {len(cohort)} eventos en la intersección cohorte∩catálogo. "
                         "Necesito al menos 10.")

    # Anotar campos derivados
    for e in cohort:
        e["_snr"] = float(e.get("SNR", 0) or 0)
        e["_mf"] = float(e.get("Mf_source", 0) or 0)
        spin_val, spin_src = get_spin(e)
        e["_spin"] = spin_val if spin_val is not None else 0.0
        e["_spin_source"] = spin_src
        e["_n_det"] = get_n_detectors(e)

    # Ordenar por SNR descendente
    cohort.sort(key=lambda e: e["_snr"], reverse=True)

    n = len(cohort)
    q3_start = n // 4          # top 25% boundary
    q2_start = n // 2          # median boundary
    q2_end = 3 * n // 4        # Q2 lower boundary

    # ── Bloque A: top-4 SNR ──
    block_a = cohort[:4]

    # ── Bloque B: Q3 de SNR (posiciones q3_start..q2_start), diversidad Mf ──
    q3_pool = cohort[q3_start:q2_start]
    if len(q3_pool) < 3:
        # Fallback: ampliar al rango 4..q2_start
        q3_pool = cohort[4:q2_start]

    # Ordenar Q3 por Mf y tomar bajo/medio/alto
    q3_sorted_mf = sorted(q3_pool, key=lambda e: e["_mf"])
    if len(q3_sorted_mf) >= 3:
        block_b = [
            q3_sorted_mf[0],                          # Mf bajo
            q3_sorted_mf[len(q3_sorted_mf) // 2],     # Mf medio
            q3_sorted_mf[-1],                          # Mf alto
        ]
    else:
        block_b = q3_sorted_mf[:3]

    # Evitar duplicados con bloque A
    block_a_names = {e["event"] for e in block_a}
    block_b = [e for e in block_b if e["event"] not in block_a_names]
    # Rellenar si hay duplicados
    for e in q3_sorted_mf:
        if len(block_b) >= 3:
            break
        if e["event"] not in block_a_names and e not in block_b:
            block_b.append(e)
    block_b = block_b[:3]

    # ── Bloque C: Q2 de SNR (posiciones q2_start..q2_end), controles negativos ──
    used = block_a_names | {e["event"] for e in block_b}
    q2_pool = [e for e in cohort[q2_start:q2_end] if e["event"] not in used]
    if len(q2_pool) < 3:
        q2_pool = [e for e in cohort[q2_start:] if e["event"] not in used]

    # Tomar 3 uniformemente espaciados
    if len(q2_pool) >= 3:
        step = len(q2_pool) // 3
        block_c = [q2_pool[0], q2_pool[step], q2_pool[2 * step]]
    else:
        block_c = q2_pool[:3]

    # ── Compilar resultado ──
    selection = []
    for block_label, block, rationale in [
        ("A_top_snr", block_a,
         "Top SNR — máxima oportunidad física para 221. "
         "Si ni estos mejoran con shared_band_early_taper, señal fuerte de problema más profundo."),
        ("B_q3_mf_diverse", block_b,
         "Q3 SNR con diversidad en Mf — cubre eje de frecuencia del overtone "
         "(Mf bajo → f_221 alto, Mf alto → f_221 bajo)."),
        ("C_q2_control", block_c,
         "Q2 SNR — controles negativos. Si alguno mejora con nueva topología, "
         "señal muy fuerte de que rigid_spectral_split destruía señal."),
    ]:
        for i, e in enumerate(block):
            selection.append({
                "event": e["event"],
                "block": block_label,
                "block_rationale": rationale,
                "rank_in_block": i + 1,
                "SNR": e["_snr"],
                "Mf_source": e["_mf"],
                "spin": e["_spin"],
                "spin_source": e["_spin_source"],
                "n_detectors": e["_n_det"],
                "snr_rank_in_cohort": next(
                    j + 1 for j, c in enumerate(cohort) if c["event"] == e["event"]
                ),
            })

    return {
        "selection_criteria": {
            "method": "stratified_snr_mf_spin",
            "cohort_size": n,
            "n_selected": len(selection),
            "blocks": {
                "A_top_snr": {"n": len(block_a), "criterion": "top-4 SNR"},
                "B_q3_mf_diverse": {"n": len(block_b), "criterion": "Q3 SNR, Mf diversity"},
                "C_q2_control": {"n": len(block_c), "criterion": "Q2 SNR, negative controls"},
            },
            "spin_axis": "af primary, chi_eff fallback",
            "notes": [
                "Thresholds congelados: no se tocan cv/lnQ/lnf gates.",
                "Solo varía mode_221_topology: rigid_spectral_split → shared_band_early_taper.",
                "Objetivo: falsar hipótesis B+G (arquitectura sesgada contra 221).",
            ],
        },
        "cohort_summary": {
            "n_total": n,
            "snr_range": [cohort[-1]["_snr"], cohort[0]["_snr"]],
            "mf_range": [
                min(e["_mf"] for e in cohort if e["_mf"] > 0),
                max(e["_mf"] for e in cohort),
            ],
            "af_available": sum(1 for e in cohort if e["_spin_source"] == "af"),
            "af_null_fallback_chi_eff": sum(1 for e in cohort if e["_spin_source"] == "chi_eff"),
            "spin_unavailable": sum(1 for e in cohort if e["_spin_source"] == "none"),
        },
        "selected_events": selection,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Selecciona 10 eventos centinela para A/B de topología s3b."
    )
    parser.add_argument(
        "--catalog", required=True, type=Path,
        help="Ruta a gwtc_events_t0.json"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cohort-dir", type=Path,
        help="Directorio de prepare_events/outputs/events/ del run"
    )
    group.add_argument(
        "--cohort-events", type=Path,
        help="Archivo JSON con lista de nombres de eventos"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Archivo de salida JSON (default: stdout)"
    )

    args = parser.parse_args()

    # Cargar catálogo
    catalog = load_catalog(args.catalog)
    print(f"Catálogo: {len(catalog)} eventos", file=sys.stderr)

    # Descubrir cohorte
    if args.cohort_events:
        with open(args.cohort_events) as f:
            cohort_events = json.load(f)
    else:
        cohort_events = discover_cohort_events(args.cohort_dir)
    print(f"Cohorte: {len(cohort_events)} eventos", file=sys.stderr)

    # Seleccionar
    result = select_sentinels(catalog, cohort_events)

    # Emitir
    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Selección escrita en {args.output}", file=sys.stderr)
    else:
        print(output_json)

    # Resumen legible
    print("\n── Resumen de selección ──", file=sys.stderr)
    for ev in result["selected_events"]:
        print(
            f"  {ev['block']:20s}  {ev['event']:30s}  "
            f"SNR={ev['SNR']:6.1f}  Mf={ev['Mf_source']:6.1f}  "
            f"spin={ev['spin']:.3f}({ev['spin_source']:7s})  "
            f"det={ev['n_detectors']}  rank={ev['snr_rank_in_cohort']}/{result['cohort_summary']['n_total']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
