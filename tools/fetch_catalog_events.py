#!/usr/bin/env python3
"""
tools/fetch_catalog_events.py — BASURIN
========================================
Descarga los mejores eventos BBH del catálogo GWOSC para análisis de ringdown.

Criterios de selección (contract-first):
  1. Catálogos: GWTC-1, GWTC-2, GWTC-2.1, GWTC-3 (no GWTC-4.0)
  2. Solo BBH: descarta BNS (m2 < 3 M☉) y eventos marginal/retracted
  3. SNR de red ≥ SNR_MIN (default 12.0)
  4. Ordenación: final_mass_source DESC, luego SNR DESC
  5. Resolución: 4096 Hz (suficiente para QNMs de 60–500 Hz)
  6. Top-N eventos (default 100)

Salidas (bajo data/losc/):
  data/losc/<EVENT_ID>/
      <detector>_<event>_4KHz.hdf5
  data/losc/catalog_manifest.json   ← inventario auditable con SHA256

Uso básico:
  python tools/fetch_catalog_events.py --dry-run        # ver lista sin descargar
  python tools/fetch_catalog_events.py                  # descargar top-100
  python tools/fetch_catalog_events.py --top 50 --snr-min 15

Anti-redundancia: si el HDF5 ya existe y el SHA256 coincide, se omite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constantes de catálogos aceptados (GWTC-1/2/2.1/3)
# ---------------------------------------------------------------------------
ACCEPTED_CATALOGS = {
    "GWTC-1-confident",
    "GWTC-2",
    "GWTC-2.1-confident",
    "GWTC-3-confident",
}

GWOSC_EVENTAPI = "https://gwosc.org/eventapi/json/GWTC/"
GWOSC_DETAIL   = "https://gwosc.org/eventapi/json/event/{event_name}/"

DETECTORS = ["H1", "L1"]   # V1 tiene menos cobertura; se añade si existe
SAMPLE_RATE = 4096          # Hz

# Masa mínima del objeto secundario para considerar BBH (M☉)
# < 3 M☉ → probable estrella de neutrones → ringdown no limpio
M2_BBH_MIN = 3.0

# ---------------------------------------------------------------------------
# Helpers de red (sin dependencia de gwpy para esta fase)
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = 30) -> dict:
    """GET JSON desde url usando urllib (sin dependencias extra)."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "BASURIN/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _download_file(url: str, dest: Path, *, chunk: int = 65536) -> str:
    """Descarga url → dest y devuelve SHA256 hex. Muestra progreso."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "BASURIN/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = r.headers.get("Content-Length")
        total_mb = f"{int(total)/1e6:.1f} MB" if total else "? MB"
        downloaded = 0
        with open(tmp, "wb") as fh:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                h.update(buf)
                downloaded += len(buf)
                pct = f"{downloaded/int(total)*100:.0f}%" if total else f"{downloaded/1e6:.1f}MB"
                print(f"\r    [{pct} / {total_mb}]", end="", flush=True)
    print()
    tmp.rename(dest)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(65536)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Lógica de selección
# ---------------------------------------------------------------------------

def fetch_gwtc_events() -> list[dict]:
    """Descarga el catálogo GWTC completo desde la API de GWOSC."""
    print(f"[API] GET {GWOSC_EVENTAPI}")
    data = _get(GWOSC_EVENTAPI)
    events_raw = data.get("events", {})
    print(f"[API] {len(events_raw)} entradas en catálogo crudo")
    return list(events_raw.values())


def is_bbh(ev: dict) -> bool:
    """True si el evento es BBH (ambas masas ≥ M2_BBH_MIN M☉)."""
    m1 = ev.get("mass_1_source")
    m2 = ev.get("mass_2_source")
    # Si no hay datos de masa, conservamos (puede ser BBH sin PE publicado)
    if m1 is None or m2 is None:
        return True
    return float(m2) >= M2_BBH_MIN


def is_accepted_catalog(ev: dict) -> bool:
    cat = ev.get("catalog.shortName", "")
    return cat in ACCEPTED_CATALOGS


def get_snr(ev: dict) -> float:
    v = ev.get("network_matched_filter_snr")
    return float(v) if v is not None else 0.0


def get_final_mass(ev: dict) -> float:
    # final_mass_source o estimación desde m1+m2 (borra ~5% radiado)
    v = ev.get("final_mass_source")
    if v is not None:
        return float(v)
    m1 = ev.get("mass_1_source")
    m2 = ev.get("mass_2_source")
    if m1 and m2:
        return 0.95 * (float(m1) + float(m2))
    return 0.0


def select_events(all_events: list[dict], snr_min: float, top_n: int) -> list[dict]:
    """Filtra y ordena los eventos según los criterios de ringdown."""
    filtered = []
    stats = {"total": len(all_events), "bad_catalog": 0, "not_bbh": 0, "low_snr": 0, "ok": 0}

    for ev in all_events:
        if not is_accepted_catalog(ev):
            stats["bad_catalog"] += 1
            continue
        if not is_bbh(ev):
            stats["not_bbh"] += 1
            continue
        if get_snr(ev) < snr_min:
            stats["low_snr"] += 1
            continue
        stats["ok"] += 1
        filtered.append(ev)

    print(f"[FILTER] total={stats['total']} | "
          f"catálogo_excluido={stats['bad_catalog']} | "
          f"no_BBH={stats['not_bbh']} | "
          f"SNR<{snr_min}={stats['low_snr']} | "
          f"candidatos={stats['ok']}")

    # Ordenar: masa_final DESC, luego SNR DESC
    filtered.sort(key=lambda e: (get_final_mass(e), get_snr(e)), reverse=True)

    return filtered[:top_n]


# ---------------------------------------------------------------------------
# Descarga de HDF5
# ---------------------------------------------------------------------------

def get_strain_urls(event_name: str, detectors: list[str], sample_rate: int) -> dict[str, str]:
    """
    Consulta el detalle del evento y devuelve {detector: url} para la
    resolución pedida (4096 Hz).
    """
    url = GWOSC_DETAIL.format(event_name=event_name)
    try:
        detail = _get(url)
    except Exception as e:
        print(f"    [WARN] No se pudo obtener detalle de {event_name}: {e}")
        return {}

    # El detalle tiene versiones; tomar la primera disponible
    versions = detail.get("events", {})
    if not versions:
        return {}
    # Tomar la versión más reciente (última key ordenada)
    latest = sorted(versions.keys())[-1]
    ev_detail = versions[latest]

    strain_list = ev_detail.get("strain", [])
    result = {}
    for s in strain_list:
        det = s.get("detector", "")
        sr = s.get("sampling_rate", 0)
        url_s = s.get("url", "")
        if det in detectors and int(sr) == sample_rate and url_s.endswith(".hdf5"):
            if det not in result:   # tomar el primero (duración más larga)
                result[det] = url_s
    return result


def download_event(
    ev: dict,
    losc_root: Path,
    detectors: list[str],
    sample_rate: int,
    dry_run: bool,
) -> dict:
    """
    Descarga los HDF5 de un evento. Devuelve un dict de auditoría.
    Respeta idempotencia: si el fichero existe y SHA256 coincide, omite.
    """
    name = ev.get("commonName") or ev.get("GPS", "unknown")
    event_dir = losc_root / str(name)
    record = {
        "event_id": name,
        "catalog": ev.get("catalog.shortName"),
        "snr": get_snr(ev),
        "final_mass_source": get_final_mass(ev),
        "mass_1_source": ev.get("mass_1_source"),
        "mass_2_source": ev.get("mass_2_source"),
        "files": {},
        "status": "pending",
    }

    print(f"\n[EVENT] {name}  SNR={record['snr']:.1f}  "
          f"M_f≈{record['final_mass_source']:.1f} M☉  "
          f"cat={record['catalog']}")

    if dry_run:
        record["status"] = "dry_run"
        return record

    # Obtener URLs
    urls = get_strain_urls(name, detectors, sample_rate)
    if not urls:
        print(f"    [WARN] No se encontraron URLs de strain para {name}")
        record["status"] = "no_urls"
        return record

    for det, url in urls.items():
        fname = event_dir / f"{det}_{name}_{sample_rate // 1000}KHz.hdf5"
        sha_file = event_dir / f"{det}_{name}_{sample_rate // 1000}KHz.sha256"

        # Idempotencia
        if fname.exists() and sha_file.exists():
            expected_sha = sha_file.read_text().strip()
            actual_sha = _sha256_file(fname)
            if expected_sha == actual_sha:
                print(f"    [SKIP] {det}: ya existe y SHA256 OK ({actual_sha[:12]}...)")
                record["files"][det] = {
                    "path": str(fname),
                    "sha256": actual_sha,
                    "status": "cached",
                }
                continue
            else:
                print(f"    [WARN] {det}: SHA256 no coincide, re-descargando")

        print(f"    [DL]  {det}: {url}")
        try:
            sha = _download_file(url, fname)
            sha_file.write_text(sha)
            record["files"][det] = {
                "path": str(fname),
                "sha256": sha,
                "status": "downloaded",
                "url": url,
            }
            print(f"    [OK]  {det}: {fname.name}  sha256={sha[:12]}...")
        except Exception as e:
            print(f"    [ERR] {det}: {e}")
            record["files"][det] = {"status": "error", "error": str(e)}

    ok = all(v.get("status") in ("downloaded", "cached") for v in record["files"].values())
    record["status"] = "ok" if ok else "partial"
    return record


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(losc_root: Path, records: list[dict]) -> Path:
    manifest = {
        "schema": "basurin_catalog_manifest_v1",
        "created": datetime.now(timezone.utc).isoformat(),
        "criteria": {
            "catalogs": sorted(ACCEPTED_CATALOGS),
            "bbh_only": True,
            "m2_bbh_min_solar_masses": M2_BBH_MIN,
            "detectors": DETECTORS,
            "sample_rate_hz": SAMPLE_RATE,
        },
        "events": records,
    }
    path = losc_root / "catalog_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[MANIFEST] → {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Selecciona y descarga los mejores eventos BBH de GWOSC para BASURIN."
    )
    p.add_argument("--top", type=int, default=100,
                   help="Número máximo de eventos a descargar (default: 100)")
    p.add_argument("--snr-min", type=float, default=12.0,
                   help="SNR de red mínimo (default: 12.0)")
    p.add_argument("--losc-root", type=Path, default=Path("data/losc"),
                   help="Directorio raíz para los HDF5 (default: data/losc)")
    p.add_argument("--dry-run", action="store_true",
                   help="Solo mostrar lista de eventos sin descargar")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Segundos de espera entre descargas (default: 1.0, cortesía GWOSC)")
    p.add_argument("--detectors", default="H1,L1",
                   help="Detectores a descargar, CSV (default: H1,L1)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    detectors = [d.strip() for d in args.detectors.split(",")]

    print("=" * 70)
    print("BASURIN — fetch_catalog_events")
    print(f"  top={args.top}  snr_min={args.snr_min}  "
          f"detectors={detectors}  rate={SAMPLE_RATE}Hz")
    print(f"  losc_root={args.losc_root.resolve()}")
    print(f"  dry_run={args.dry_run}")
    print("=" * 70)

    # 1) Obtener catálogo
    all_events = fetch_gwtc_events()

    # 2) Filtrar y ordenar
    selected = select_events(all_events, snr_min=args.snr_min, top_n=args.top)
    print(f"\n[SELECT] {len(selected)} eventos seleccionados (top {args.top})\n")

    # 3) Tabla resumen
    print(f"{'#':>3}  {'EVENT_ID':<24}  {'SNR':>6}  {'M_f(M☉)':>9}  {'CAT'}")
    print("-" * 70)
    for i, ev in enumerate(selected, 1):
        name = ev.get("commonName") or "?"
        print(f"{i:>3}  {name:<24}  {get_snr(ev):>6.1f}  "
              f"{get_final_mass(ev):>9.1f}  {ev.get('catalog.shortName','?')}")

    if args.dry_run:
        print("\n[DRY-RUN] Sin descargas. Usa sin --dry-run para descargar.")
        write_manifest(args.losc_root, [
            {"event_id": ev.get("commonName"), "snr": get_snr(ev),
             "final_mass_source": get_final_mass(ev),
             "catalog": ev.get("catalog.shortName"), "status": "dry_run"}
            for ev in selected
        ])
        return 0

    # 4) Descargar
    args.losc_root.mkdir(parents=True, exist_ok=True)
    records = []
    for i, ev in enumerate(selected, 1):
        name = ev.get("commonName") or "?"
        print(f"\n[{i}/{len(selected)}]", end="")
        rec = download_event(ev, args.losc_root, detectors, SAMPLE_RATE, dry_run=False)
        records.append(rec)
        if i < len(selected):
            time.sleep(args.delay)

    # 5) Manifest
    write_manifest(args.losc_root, records)

    # 6) Resumen final
    ok = sum(1 for r in records if r["status"] == "ok")
    cached = sum(1 for r in records if any(
        f.get("status") == "cached" for f in r.get("files", {}).values()
    ))
    err = sum(1 for r in records if r["status"] not in ("ok", "dry_run"))
    print(f"\n[DONE]  OK={ok}  cached={cached}  errores={err}")
    print(f"[LOSC]  {args.losc_root.resolve()}")
    print(f"[MANIFEST]  {args.losc_root.resolve() / 'catalog_manifest.json'}")

    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
