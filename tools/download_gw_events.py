#!/usr/bin/env python3
"""
download_gw_events.py
=====================
Descarga eventos de ondas gravitacionales del catálogo GWTC (GWOSC),
filtrando por calidad (SNR, p_astro, FAR) y guardando metadatos + strain data.

Pensado para análisis de ringdown con BASURIN.

Uso:
    python tools/download_gw_events.py                      # Solo metadata (CSV/JSON)
    python tools/download_gw_events.py --download           # Metadata + descarga strain
    python tools/download_gw_events.py --download --losc-root data/losc
    python tools/download_gw_events.py --download --sr 16384  # Strain a 16kHz

Requisitos:
    pip install gwosc requests pandas
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests


# =============================================================================
# Configuración por defecto - criterios de calidad para ringdown
# =============================================================================
DEFAULT_SNR_MIN = 8.0       # SNR red mínimo (paper usa pastro>0.5 ~ SNR>8)
DEFAULT_PASTRO_MIN = 0.9    # p_astro mínimo (0.9 = alta confianza; 0.5 = catálogo completo)
DEFAULT_FAR_MAX = 1.0       # FAR máximo en yr^-1 (None = sin filtro)
DEFAULT_CATALOGS = [        # Catálogos "confident" de GWTC
    "GWTC-1-confident",
    "GWTC-2.1-confident",
    "GWTC-3-confident",
]

# Eventos con problemas conocidos de calidad de datos o posteriors poco fiables
# (ver GWTC-3 paper Sec. V y Table XVI)
KNOWN_PROBLEMATIC = {
    "GW200308_173609": "Posteriors dominadas por prior, SNR~4.7, modo de baja likelihood",
    "GW200322_091133": "Posteriors dominadas por prior, SNR~4.5, modo de baja likelihood",
    "GW200208_222617": "Multimodal, pastro=0.70, solo 1 pipeline",
}

# Eventos con glitches que requirieron mitigación (BayesWave o gwsubtract)
# No necesariamente hay que excluirlos, pero es útil saberlo
GLITCH_MITIGATED = {
    "GW191105_143521": "Virgo: BayesWave deglitching",
    "GW191109_010717": "H1+L1: BayesWave deglitching",
    "GW191113_071753": "H1: BayesWave deglitching",
    "GW191127_050227": "H1: BayesWave deglitching",
    "GW191219_163120": "H1+L1: BayesWave deglitching",
    "GW200105_162426": "L1: BayesWave deglitching",
    "GW200115_042309": "L1: BayesWave deglitching",
    "GW200129_065458": "L1: linear subtraction (gwsubtract)",
}

GWOSC_API_BASE = "https://gwosc.org"


def fetch_catalog_events(catalog_name):
    """Obtiene todos los eventos de un catálogo con sus parámetros completos."""
    url = f"{GWOSC_API_BASE}/eventapi/jsonfull/{catalog_name}/"
    print(f"  Consultando {catalog_name}...")
    
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("events", {})
    except requests.exceptions.RequestException as e:
        print(f"  [WARN] Error consultando {catalog_name}: {e}")
        return {}


def parse_event(event_name_versioned, event_data):
    """Extrae los parámetros relevantes de un evento."""
    
    # El nombre del evento en el JSON incluye la versión: "GW150914-v3"
    # Extraemos el nombre limpio
    parts = event_name_versioned.rsplit("-v", 1)
    name = parts[0]
    version = int(parts[1]) if len(parts) > 1 else 1
    
    # Normalizar nombre (GWOSC usa tanto _ como espacio)
    common_name = event_data.get("commonName", name)
    
    info = {
        "event": common_name,
        "version": version,
        "catalog": event_data.get("catalog.shortName", ""),
        "GPS": event_data.get("GPS", None),
        "snr": event_data.get("network_matched_filter_snr", None),
        "p_astro": event_data.get("p_astro", None),
        "far_yr": event_data.get("far", None),
        "m1_source": event_data.get("mass_1_source", None),
        "m1_source_lower": event_data.get("mass_1_source_lower", None),
        "m1_source_upper": event_data.get("mass_1_source_upper", None),
        "m2_source": event_data.get("mass_2_source", None),
        "m2_source_lower": event_data.get("mass_2_source_lower", None),
        "m2_source_upper": event_data.get("mass_2_source_upper", None),
        "M_total_source": event_data.get("total_mass_source", None),
        "chirp_mass_source": event_data.get("chirp_mass_source", None),
        "chi_eff": event_data.get("chi_eff", None),
        "chi_eff_lower": event_data.get("chi_eff_lower", None),
        "chi_eff_upper": event_data.get("chi_eff_upper", None),
        "final_mass_source": event_data.get("final_mass_source", None),
        "final_spin": event_data.get("final_mass_source_lower", None),  # placeholder
        "luminosity_distance": event_data.get("luminosity_distance", None),
        "redshift": event_data.get("redshift", None),
        "reference": event_data.get("reference", ""),
        "jsonurl": event_data.get("jsonurl", ""),
    }
    
    # Extraer información de strain disponible
    strain_list = event_data.get("strain", [])
    detectors_with_data = set()
    for s in strain_list:
        det = s.get("detector", "")
        if det:
            detectors_with_data.add(det)
    info["detectors"] = ",".join(sorted(detectors_with_data))
    info["n_strain_files"] = len(strain_list)
    
    # Flags de calidad
    info["is_problematic"] = common_name in KNOWN_PROBLEMATIC
    info["glitch_mitigated"] = common_name in GLITCH_MITIGATED
    if common_name in KNOWN_PROBLEMATIC:
        info["quality_note"] = KNOWN_PROBLEMATIC[common_name]
    elif common_name in GLITCH_MITIGATED:
        info["quality_note"] = f"Glitch mitigado: {GLITCH_MITIGATED[common_name]}"
    else:
        info["quality_note"] = ""
    
    return info


def apply_quality_filters(df, snr_min, pastro_min, far_max, exclude_problematic):
    """Aplica filtros de calidad al DataFrame de eventos."""
    
    n_initial = len(df)
    
    # Filtro SNR
    if snr_min is not None:
        mask_snr = df["snr"].notna() & (df["snr"] >= snr_min)
        df = df[mask_snr]
        print(f"  SNR >= {snr_min}: {len(df)}/{n_initial} eventos")
    
    # Filtro p_astro
    if pastro_min is not None:
        mask_pastro = df["p_astro"].notna() & (df["p_astro"] >= pastro_min)
        df = df[mask_pastro]
        print(f"  p_astro >= {pastro_min}: {len(df)}/{n_initial} eventos")
    
    # Filtro FAR
    if far_max is not None:
        mask_far = df["far_yr"].notna() & (df["far_yr"] <= far_max)
        df = df[mask_far]
        print(f"  FAR <= {far_max} yr^-1: {len(df)}/{n_initial} eventos")
    
    # Excluir eventos problemáticos
    if exclude_problematic:
        mask_ok = ~df["is_problematic"]
        n_excluded = (~mask_ok).sum()
        df = df[mask_ok]
        if n_excluded > 0:
            print(f"  Excluidos {n_excluded} eventos problemáticos")
    
    return df


def _normalize_event_id(event_name: str) -> str:
    """Map event names to canonical folder naming used under data/losc."""
    return event_name.strip().replace(" ", "_")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_sha256_sums(event_dir: Path) -> None:
    h5_files = sorted(
        [
            p for p in event_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}
        ],
        key=lambda p: p.name,
    )
    if not h5_files:
        return
    sums_path = event_dir / "SHA256SUMS.txt"
    lines = [f"{_sha256_file(p)}  {p.name}\n" for p in h5_files]
    sums_path.write_text("".join(lines), encoding="utf-8")


def download_strain_for_event(
    event_row,
    losc_root,
    sample_rate=4096,
    duration=4096,
    fmt="hdf5",
):
    """Descarga strain para un evento en data/losc/<EVENT_ID>/."""
    from gwosc.locate import get_event_urls
    
    event_name = event_row["event"]
    event_id = _normalize_event_id(event_name)
    event_dir = Path(losc_root) / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    
    for det in ["H1", "L1"]:
        try:
            urls = get_event_urls(
                event_name, 
                detector=det, 
                format=fmt,
                duration=duration,
                sample_rate=sample_rate,
            )
        except Exception:
            urls = []
        
        for url in urls:
            fname = url.split("/")[-1]
            fpath = event_dir / fname
            
            if fpath.exists():
                print(f"    [SKIP] {fname} (ya existe)")
                downloaded.append(str(fpath))
                continue
            
            print(f"    Descargando {fname}...")
            try:
                r = requests.get(url, stream=True, timeout=120)
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                downloaded_size = 0
                start_time = time.time()
                
                with open(fpath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        elapsed = time.time() - start_time
                        speed = downloaded_size / elapsed if elapsed > 0 else 0
                        
                        if total_size > 0:
                            pct = downloaded_size / total_size * 100
                            bar_len = 30
                            filled = int(bar_len * downloaded_size / total_size)
                            bar = "█" * filled + "░" * (bar_len - filled)
                            size_mb = downloaded_size / 1e6
                            total_mb = total_size / 1e6
                            speed_mb = speed / 1e6
                            eta = (total_size - downloaded_size) / speed if speed > 0 else 0
                            print(f"\r    {bar} {pct:5.1f}%  "
                                  f"{size_mb:.1f}/{total_mb:.1f} MB  "
                                  f"{speed_mb:.2f} MB/s  "
                                  f"ETA {int(eta)}s   ", end="", flush=True)
                        else:
                            size_mb = downloaded_size / 1e6
                            speed_mb = speed / 1e6
                            print(f"\r    {size_mb:.1f} MB  "
                                  f"{speed_mb:.2f} MB/s   ", end="", flush=True)
                
                print()  # nueva línea al terminar
                downloaded.append(str(fpath))
            except Exception as e:
                print(f"\n    [ERROR] {fname}: {e}")

    _write_sha256_sums(event_dir)
    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Descarga eventos GW de calidad del catálogo GWTC (GWOSC)"
    )
    parser.add_argument("--snr-min", type=float, default=DEFAULT_SNR_MIN,
                        help=f"SNR mínimo (default: {DEFAULT_SNR_MIN})")
    parser.add_argument("--pastro-min", type=float, default=DEFAULT_PASTRO_MIN,
                        help=f"p_astro mínimo (default: {DEFAULT_PASTRO_MIN})")
    parser.add_argument("--far-max", type=float, default=DEFAULT_FAR_MAX,
                        help=f"FAR máximo en yr^-1 (default: {DEFAULT_FAR_MAX})")
    parser.add_argument("--no-exclude-problematic", action="store_true",
                        help="No excluir eventos con posteriors poco fiables")
    parser.add_argument("--download", action="store_true",
                        help="Descargar archivos de strain")
    parser.add_argument("--sr", type=int, default=4096, choices=[4096, 16384],
                        help="Sample rate para descarga (default: 4096)")
    parser.add_argument("--duration", type=int, default=4096, choices=[32, 4096],
                        help="Duración de los archivos: 32s o 4096s (default: 4096)")
    parser.add_argument("--format", default="hdf5", choices=["hdf5", "gwf"],
                        help="Formato de strain (default: hdf5)")
    parser.add_argument("--output-dir", default="gw_events",
                        help="Directorio de metadatos (CSV/JSON) (default: gw_events)")
    parser.add_argument("--losc-root", default="data/losc",
                        help="Destino canónico de strain para pipeline (default: data/losc)")
    parser.add_argument("--catalogs", nargs="+", default=DEFAULT_CATALOGS,
                        help="Catálogos a consultar")
    parser.add_argument("--all-events", action="store_true",
                        help="Incluir TODOS los eventos (pastro>0.5, sin filtros extra)")
    
    args = parser.parse_args()
    
    if args.all_events:
        args.snr_min = 0
        args.pastro_min = 0.5
        args.far_max = None
        args.no_exclude_problematic = True
    
    # =========================================================================
    # 1. Obtener eventos de GWOSC
    # =========================================================================
    print("=" * 60)
    print("DESCARGA DE EVENTOS GW - GWOSC/GWTC")
    print("=" * 60)
    print(f"\nCatálogos: {', '.join(args.catalogs)}")
    print(f"Filtros: SNR >= {args.snr_min}, p_astro >= {args.pastro_min}, "
          f"FAR <= {args.far_max} yr^-1")
    print()
    
    all_events = {}
    for cat in args.catalogs:
        events = fetch_catalog_events(cat)
        print(f"    -> {len(events)} eventos encontrados")
        all_events.update(events)
        time.sleep(0.5)  # cortesía con el servidor
    
    print(f"\nTotal eventos únicos (todas las versiones): {len(all_events)}")
    
    # =========================================================================
    # 2. Parsear y construir DataFrame
    # =========================================================================
    rows = []
    for ev_name, ev_data in all_events.items():
        row = parse_event(ev_name, ev_data)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Quedarnos con la versión más reciente de cada evento
    df = df.sort_values("version", ascending=False).drop_duplicates(
        subset=["event"], keep="first"
    ).sort_values("snr", ascending=False, na_position="last")
    
    print(f"Eventos únicos (última versión): {len(df)}")
    
    # =========================================================================
    # 3. Aplicar filtros de calidad
    # =========================================================================
    print(f"\nAplicando filtros de calidad:")
    df_filtered = apply_quality_filters(
        df.copy(),
        snr_min=args.snr_min,
        pastro_min=args.pastro_min,
        far_max=args.far_max,
        exclude_problematic=not args.no_exclude_problematic,
    )
    
    df_filtered = df_filtered.sort_values("snr", ascending=False)
    
    print(f"\n{'=' * 60}")
    print(f"RESULTADO: {len(df_filtered)} eventos de alta calidad")
    print(f"{'=' * 60}")
    
    # =========================================================================
    # 4. Mostrar resumen
    # =========================================================================
    cols_display = ["event", "GPS", "snr", "p_astro", "far_yr", 
                    "m1_source", "m2_source", "chi_eff", "luminosity_distance",
                    "detectors", "glitch_mitigated"]
    
    print("\nEventos seleccionados (ordenados por SNR):\n")
    print(df_filtered[cols_display].to_string(index=False, max_rows=100))
    
    # =========================================================================
    # 5. Guardar CSV con metadatos
    # =========================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    losc_root = Path(args.losc_root)
    losc_root.mkdir(parents=True, exist_ok=True)
    
    csv_all = output_dir / "gwtc_all_events.csv"
    csv_filtered = output_dir / "gwtc_quality_events.csv"
    
    df.to_csv(csv_all, index=False)
    df_filtered.to_csv(csv_filtered, index=False)
    
    print(f"\nCSV guardados:")
    print(f"  Todos los eventos:      {csv_all}")
    print(f"  Eventos filtrados:      {csv_filtered}")
    
    # Guardar también un JSON limpio con los t0 (GPS times) para BASURIN
    t0_data = {}
    for _, row in df_filtered.iterrows():
        t0_data[row["event"]] = {
            "GPS": row["GPS"],
            "snr": row["snr"],
            "p_astro": row["p_astro"],
            "m1": row["m1_source"],
            "m2": row["m2_source"],
            "M_total": row["M_total_source"],
            "chirp_mass": row["chirp_mass_source"],
            "chi_eff": row["chi_eff"],
            "Mf": row["final_mass_source"],
            "DL_Mpc": row["luminosity_distance"],
            "z": row["redshift"],
            "detectors": row["detectors"],
            "glitch_mitigated": row["glitch_mitigated"],
        }
    
    json_path = output_dir / "gwtc_events_t0.json"
    with open(json_path, "w") as f:
        json.dump(t0_data, f, indent=2, default=str)
    print(f"  JSON con GPS/t0:        {json_path}")
    
    # =========================================================================
    # 6. Descargar strain data (opcional)
    # =========================================================================
    if args.download:
        print(f"\n{'=' * 60}")
        print(f"DESCARGANDO STRAIN DATA")
        print(f"  LOSC root:  {losc_root}")
        print(f"  Sample rate: {args.sr} Hz")
        print(f"  Duración:    {args.duration} s")
        print(f"  Formato:     {args.format}")
        print(f"{'=' * 60}\n")

        for i, (_, row) in enumerate(df_filtered.iterrows()):
            event = row["event"]
            print(f"[{i+1}/{len(df_filtered)}] {event} (SNR={row['snr']:.1f}, "
                  f"GPS={row['GPS']})...")
            
            downloaded = download_strain_for_event(
                row, losc_root,
                sample_rate=args.sr,
                duration=args.duration,
                fmt=args.format,
            )
            
            if downloaded:
                print(f"    -> {len(downloaded)} archivos descargados")
            else:
                print(f"    -> Sin archivos disponibles con estos parámetros")
            
            time.sleep(0.3)
        
        print(f"\nStrain data guardado en: {losc_root}/<EVENT_ID>/")
    
    print("\n¡Listo!")
    return df_filtered


if __name__ == "__main__":
    main()
