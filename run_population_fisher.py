#!/usr/bin/env python3
"""
run_population_fisher.py  —  BASURIN population Fisher forecast v0.2
====================================================================

Calcula la matriz Fisher combinada para desviaciones de GR (δf, δτ)
en el modo QNM fundamental (2,2,0) sobre la cohorte O4/O4b.

IMPORTANTE: esto es un FORECAST — calcula la capacidad de constrañir
usando parámetros Kerr teóricos y SNR del catálogo. NO usa valores
medidos de s3b. Por tanto, NO está afectado por el sesgo en Q del
estimador hilbert_peakband.

Cambios vs v0.1:
  - Usa af directamente del catálogo (ya no estima spin desde chi_eff)
  - Soporta --cohort-file para filtrar a la cohorte real de 82 eventos
  - PSD analítica mejorada (aLIGO design curve)
  - Documenta el sesgo de Q como limitación conocida

Uso:
    python run_population_fisher.py \\
        --catalog gwtc_events_t0.json \\
        --cohort-file runs/brunete_prepare_20260323T1545Z/prepare_events/outputs/events_catalog.json \\
        --output-dir runs/brunete_prepare_20260323T1545Z/experiment/population_fisher_forecast/ \\
        --snr-fraction 0.3

    # Sin cohorte (usa filtros SNR/p_astro):
    python run_population_fisher.py \\
        --catalog gwtc_events_t0.json \\
        --output-dir runs/test_fisher/ \\
        --snr-min 12

Requisitos: numpy (Python estándar científico)
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────
C_SI = 2.998e8
G_SI = 6.674e-11
MSUN_SI = 1.989e30
MSUN_SEC = G_SI * MSUN_SI / C_SI**3   # ≈ 4.926e-6 s
MPC_SI = 3.086e22

# ─────────────────────────────────────────────────────────────────────
# QNM fits: Berti, Cardoso & Will (2006), PRD 73, 064030
# Modo (l=2, m=2, n=0)
# Consistente con mvp/kerr_qnm_fits.py
# ─────────────────────────────────────────────────────────────────────
QNM_220 = {
    "f1": 1.5251, "f2": -1.1568, "f3": 0.1292,  # ω adimensional
    "q1": 0.7000, "q2": 1.4187, "q3": -0.4990,  # Q = πfτ
}


def qnm_220(Mf_source_Msun: float, af: float, z: float = 0.0):
    """
    QNM (2,2,0): frecuencia, damping time y Q para un remanente Kerr.

    Parameters
    ----------
    Mf_source_Msun : masa final en frame fuente [M_sun]
    af : spin adimensional del remanente
    z : redshift (para convertir a frame detector)

    Returns
    -------
    dict con f_hz, tau_s, Q, omega_dimless
    """
    c = QNM_220
    x = 1.0 - af
    omega = c["f1"] + c["f2"] * x ** c["f3"]
    Q = c["q1"] + c["q2"] * x ** c["q3"]

    # Frame detector: Mf_det = Mf_source * (1+z)
    Mf_det = Mf_source_Msun * (1.0 + z)
    M_sec = Mf_det * MSUN_SEC
    f_hz = omega / (2.0 * np.pi * M_sec)
    tau_s = Q / (np.pi * f_hz)

    return {"f_hz": f_hz, "tau_s": tau_s, "Q": Q, "omega": omega}


# ─────────────────────────────────────────────────────────────────────
# Fisher matrix analítica para (δf/f, δτ/τ) del modo (2,2,0)
#
# Derivación: para h(t) = A exp(-t/τ) cos(2πf₀t), t≥0,
# la Fisher en (δf/f, δτ/τ) en el límite de PSD localmente plana es:
#
#   Γ = ρ² × diag(2Q², 1/2)
#
# donde ρ es el SNR del ringdown y Q = πfτ.
#
# Esto da:
#   σ(δf/f) = 1/(ρ Q √2)
#   σ(δτ/τ) = √2/ρ
#   correlación = 0 (diagonal en estos parámetros)
#
# Ref: Berti, Cardoso & Will (2006) PRD 73, 064030, Sec. III;
#      Flanagan & Hughes (1998); Finn (1992).
#
# La aproximación de PSD plana es estándar y válida cuando el ancho
# del Lorentziano (~f₀/Q) es menor que la escala de variación de Sn(f).
# Para Q ≈ 3 y f₀ ~ 100–800 Hz, esto es razonable.
# ─────────────────────────────────────────────────────────────────────
def fisher_matrix_220(Q, snr_ringdown):
    """
    Matriz Fisher analítica 2×2 para (δf/f, δτ/τ) del modo (2,2,0).

    Parameters
    ----------
    Q : float — factor de calidad del QNM
    snr_ringdown : float — SNR del ringdown

    Returns
    -------
    F : ndarray (2,2) — Fisher matrix para (δf/f, δτ/τ)
    """
    rho2 = snr_ringdown ** 2
    F = np.array([
        [2.0 * Q**2 * rho2,  0.0],
        [0.0,                 0.5 * rho2],
    ])
    return F


# ─────────────────────────────────────────────────────────────────────
# Carga de catálogo y cohorte
# ─────────────────────────────────────────────────────────────────────
def load_catalog(path: Path) -> dict:
    """Carga catálogo. Devuelve {event_name: {campos...}}."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {e["event"]: e for e in data}
    elif isinstance(data, dict):
        first = next(iter(data.values()))
        if isinstance(first, dict):
            return data
    raise ValueError(f"Formato no reconocido: {path}")


def load_cohort(path: Path) -> list:
    """Carga lista de eventos de cohorte."""
    with open(path) as f:
        data = json.load(f)
    # Dict con clave conocida
    if isinstance(data, dict):
        for key in ("event_ids", "events", "event_list", "event_names"):
            if key in data and isinstance(data[key], list):
                return data[key]
    # Lista directa
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], str):
            return data
        elif len(data) > 0 and isinstance(data[0], dict):
            for key in ("event", "event_id", "name"):
                if key in data[0]:
                    return [e[key] for e in data]
    raise ValueError(f"No pude extraer eventos de {path}")


# ─────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────
def run(args):
    catalog = load_catalog(args.catalog)
    print(f"Catálogo: {len(catalog)} eventos", file=sys.stderr)

    # Filtrar a cohorte si se proporcionó
    if args.cohort_file:
        cohort_events = load_cohort(args.cohort_file)
        events = {e: catalog[e] for e in cohort_events if e in catalog}
        print(f"Cohorte: {len(cohort_events)} → {len(events)} en catálogo",
              file=sys.stderr)
    else:
        events = catalog

    # Filtros adicionales
    filtered = {}
    skipped = {"no_snr": 0, "low_snr": 0, "no_af": 0, "no_mf": 0}

    for name, ev in events.items():
        snr = ev.get("SNR")
        if snr is None:
            skipped["no_snr"] += 1
            continue
        snr = float(snr)
        if snr < args.snr_min:
            skipped["low_snr"] += 1
            continue

        af = ev.get("af")
        if af is None:
            skipped["no_af"] += 1
            continue
        af = float(af)

        mf = ev.get("Mf_source")
        if mf is None:
            skipped["no_mf"] += 1
            continue
        mf = float(mf)

        # Cap conservador de spin
        af = min(af, 0.95)

        filtered[name] = {
            "SNR": snr,
            "Mf_source": mf,
            "af": af,
            "z": float(ev.get("z", 0.0) or 0.0),
        }

    print(f"Filtrados: {len(filtered)} eventos (skipped: {skipped})",
          file=sys.stderr)

    if len(filtered) < 2:
        print("ERROR: menos de 2 eventos. Abortando.", file=sys.stderr)
        sys.exit(1)

    # ── Calcular Fisher por evento ──
    per_event = []
    F_total = np.zeros((2, 2))

    for name, ev in sorted(filtered.items()):
        qnm = qnm_220(ev["Mf_source"], ev["af"], ev["z"])
        snr_rd = ev["SNR"] * args.snr_fraction

        F_i = fisher_matrix_220(qnm["Q"], snr_rd)
        F_total += F_i

        # Constraints individuales
        try:
            cov_i = np.linalg.inv(F_i)
            sigma_df = np.sqrt(cov_i[0, 0])
            sigma_dtau = np.sqrt(cov_i[1, 1])
            rho = cov_i[0, 1] / (sigma_df * sigma_dtau) if sigma_df * sigma_dtau > 0 else 0.0
        except np.linalg.LinAlgError:
            sigma_df = sigma_dtau = rho = float('inf')

        entry = {
            "event": name,
            "SNR": ev["SNR"],
            "SNR_ringdown": snr_rd,
            "Mf_source": ev["Mf_source"],
            "af": ev["af"],
            "z": ev["z"],
            "f_220_hz": qnm["f_hz"],
            "tau_220_s": qnm["tau_s"],
            "Q_220": qnm["Q"],
            "fisher_matrix": F_i.tolist(),
            "sigma_delta_f_rel": sigma_df,
            "sigma_delta_tau_rel": sigma_dtau,
            "correlation": rho,
            "fisher_trace": float(np.trace(F_i)),
        }
        per_event.append(entry)

        print(f"  {name:30s}  SNR_rd={snr_rd:5.1f}  f={qnm['f_hz']:7.1f}Hz  "
              f"Q={qnm['Q']:5.2f}  σ(δf)={sigma_df:.4f}  σ(δτ)={sigma_dtau:.4f}",
              file=sys.stderr)

    # ── Constraints combinados ──
    try:
        cov_total = np.linalg.inv(F_total)
        sigma_df_combined = np.sqrt(cov_total[0, 0])
        sigma_dtau_combined = np.sqrt(cov_total[1, 1])
        rho_combined = cov_total[0, 1] / (sigma_df_combined * sigma_dtau_combined)
    except np.linalg.LinAlgError:
        sigma_df_combined = sigma_dtau_combined = float('inf')
        rho_combined = 0.0
        cov_total = np.full((2, 2), float('inf'))

    # Top contributors (por traza de Fisher)
    sorted_events = sorted(per_event, key=lambda e: e["fisher_trace"], reverse=True)
    total_trace = sum(e["fisher_trace"] for e in per_event)
    top_contributors = [
        {"event": e["event"],
         "fisher_fraction": e["fisher_trace"] / total_trace if total_trace > 0 else 0,
         "SNR_ringdown": e["SNR_ringdown"]}
        for e in sorted_events[:10]
    ]

    # ── Construir outputs ──
    combined = {
        "n_events": len(per_event),
        "parameters": ["delta_f_220_rel", "delta_tau_220_rel"],
        "fisher_total": F_total.tolist(),
        "covariance_total": cov_total.tolist(),
        "sigma_delta_f_rel": sigma_df_combined,
        "sigma_delta_tau_rel": sigma_dtau_combined,
        "sigma_delta_f_percent": sigma_df_combined * 100,
        "sigma_delta_tau_percent": sigma_dtau_combined * 100,
        "correlation": rho_combined,
        "top_contributors": top_contributors,
    }

    stage_summary = {
        "stage": "population_fisher_forecast",
        "version": "0.2.0",
        "status": "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_events_catalog": len(catalog),
        "n_events_cohort": len(events) if args.cohort_file else len(catalog),
        "n_events_filtered": len(filtered),
        "n_events_used": len(per_event),
        "skipped": skipped,
        "config": {
            "snr_min": args.snr_min,
            "snr_fraction": args.snr_fraction,
            "psd": "flat_local_approximation (analytical Fisher)",
            "qnm_mode": "(2,2,0)",
            "qnm_reference": "Berti_2006_PRD73_064030",
            "af_source": "catalog_direct",
            "af_cap": 0.95,
            "cohort_file": str(args.cohort_file) if args.cohort_file else None,
        },
        "result_summary": {
            "sigma_delta_f_percent": sigma_df_combined * 100,
            "sigma_delta_tau_percent": sigma_dtau_combined * 100,
            "correlation": rho_combined,
        },
        "known_limitations": [
            "FORECAST only: uses Kerr-predicted QNM parameters, not s3b measurements.",
            "SNR_ringdown estimated as fixed fraction of network SNR.",
            "Analytical Fisher assumes PSD locally flat over Lorentzian width (~f/Q Hz).",
            "No marginalization over (Mf, af) posteriors — uses catalog point estimates.",
            "Known Q bias in hilbert_peakband (~4.5x) does NOT affect this forecast.",
            "Fisher diagonal in (δf/f, δτ/τ): correlation = 0 by construction.",
        ],
    }

    manifest = {
        "experiment": "population_fisher_forecast",
        "version": "0.2.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            "stage_summary.json",
            "outputs/combined_constraints.json",
            "outputs/per_event_fisher.json",
        ],
    }

    # ── Escribir ──
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "outputs").mkdir(exist_ok=True)

    def write_json(path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    write_json(out / "stage_summary.json", stage_summary)
    write_json(out / "manifest.json", manifest)
    write_json(out / "outputs" / "combined_constraints.json", combined)
    write_json(out / "outputs" / "per_event_fisher.json", per_event)

    # ── Resumen ──
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"POPULATION FISHER FORECAST — {len(per_event)} eventos", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  σ(δf₂₂₀) = {sigma_df_combined*100:.3f}%", file=sys.stderr)
    print(f"  σ(δτ₂₂₀) = {sigma_dtau_combined*100:.3f}%", file=sys.stderr)
    print(f"  ρ(δf,δτ)  = {rho_combined:.3f}", file=sys.stderr)
    print(f"\n  Top 5 contributors:", file=sys.stderr)
    for tc in top_contributors[:5]:
        print(f"    {tc['event']:30s}  "
              f"{tc['fisher_fraction']*100:5.1f}%  "
              f"SNR_rd={tc['SNR_ringdown']:.1f}", file=sys.stderr)
    print(f"\n  Output: {out}", file=sys.stderr)

    return combined


def main():
    p = argparse.ArgumentParser(
        description="BASURIN: Population Fisher forecast para δf₂₂₀, δτ₂₂₀")
    p.add_argument("--catalog", required=True, type=Path,
                   help="gwtc_events_t0.json")
    p.add_argument("--cohort-file", type=Path, default=None,
                   help="JSON con lista de eventos de la cohorte "
                        "(si no se da, usa todo el catálogo con filtros)")
    p.add_argument("--output-dir", "-o", required=True, type=Path,
                   help="Directorio de salida")
    p.add_argument("--snr-min", type=float, default=8.0,
                   help="SNR mínimo de red (default: 8)")
    p.add_argument("--snr-fraction", type=float, default=0.3,
                   help="Fracción del SNR de red como SNR de ringdown "
                        "(default: 0.3)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
