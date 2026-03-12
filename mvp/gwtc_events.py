"""GWTC catalog events: source masses, remnant mass, optional final spin, and network SNR.

Primary source for event-level metadata is the local quality catalog
``gwtc_quality_events.csv`` at repository root.  This provides broad cohort
coverage for ``m1_source``, ``m2_source``, ``final_mass_source``, and ``snr``.

Only a small subset of legacy events has curated ``chi_final`` values in this
module.  For all other events the catalog entry is partial: callers may use the
available source/remnant masses and ``snr_network`` directly, but must tolerate
``chi_final is None``.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_QUALITY_CSV = _REPO_ROOT / "gwtc_quality_events.csv"

# Curated legacy chi_final medians used by preflight/Kerr-centered helpers.
# Keep these explicit until we have a trustworthy cohort-wide remnant-spin source.
_LEGACY_CHI_FINAL: dict[str, float] = {
    "GW150914": 0.67,
    "GW151226": 0.74,
    "GW170104": 0.64,
    "GW170608": 0.69,
    "GW170729": 0.81,
    "GW170809": 0.70,
    "GW170814": 0.72,
    "GW170818": 0.67,
    "GW170823": 0.71,
    "GW190521": 0.72,
}

# fmt: off
_LEGACY_EVENTS: dict[str, dict[str, float | None]] = {
    # Event        M_final (M_sun)   chi_final    SNR_network
    "GW150914": {"m_final_msun": 62.2,  "chi_final": 0.67, "snr_network": 24.4},
    "GW151226": {"m_final_msun": 20.8,  "chi_final": 0.74, "snr_network": 13.0},
    "GW170104": {"m_final_msun": 48.7,  "chi_final": 0.64, "snr_network": 13.0},
    "GW170608": {"m_final_msun": 18.0,  "chi_final": 0.69, "snr_network": 14.9},
    "GW170729": {"m_final_msun": 79.5,  "chi_final": 0.81, "snr_network": 10.8},
    "GW170809": {"m_final_msun": 56.4,  "chi_final": 0.70, "snr_network": 12.4},
    "GW170814": {"m_final_msun": 53.2,  "chi_final": 0.72, "snr_network": 15.9},
    "GW170818": {"m_final_msun": 59.4,  "chi_final": 0.67, "snr_network": 11.3},
    "GW170823": {"m_final_msun": 65.6,  "chi_final": 0.71, "snr_network": 11.5},
    "GW190521": {"m_final_msun": 142.0, "chi_final": 0.72, "snr_network": 14.7},
}
# fmt: on


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if parsed == parsed else None


def _load_quality_catalog() -> dict[str, dict[str, float | None]]:
    catalog: dict[str, dict[str, float | None]] = {k: dict(v) for k, v in _LEGACY_EVENTS.items()}
    if not _QUALITY_CSV.exists():
        return catalog

    with _QUALITY_CSV.open(encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        for row in rows:
            event_id = (row.get("event") or "").strip()
            if not event_id:
                continue
            m1_source = _as_float(row.get("m1_source"))
            m2_source = _as_float(row.get("m2_source"))
            m_final = _as_float(row.get("final_mass_source"))
            snr_network = _as_float(row.get("snr"))
            if m1_source is None and m2_source is None and m_final is None and snr_network is None:
                continue

            entry = dict(catalog.get(event_id, {}))
            if m1_source is not None:
                entry["m1_source"] = m1_source
            if m2_source is not None:
                entry["m2_source"] = m2_source
            if m_final is not None:
                entry["m_final_msun"] = m_final
            if snr_network is not None:
                entry["snr_network"] = snr_network
            if event_id in _LEGACY_CHI_FINAL:
                entry["chi_final"] = _LEGACY_CHI_FINAL[event_id]
            else:
                entry.setdefault("chi_final", None)
            catalog[event_id] = entry

    return catalog


GWTC_EVENTS: dict[str, dict[str, float | None]] = _load_quality_catalog()

# Citation string for use in JSON provenance fields
GWTC_CITATION = (
    "gwtc_quality_events.csv (local quality catalog for m1_source, m2_source, final_mass_source, and snr); "
    "legacy curated chi_final medians for GWTC-1 + GW190521."
)


def get_event(event_id: str) -> dict[str, float | None] | None:
    """Return catalog entry for event_id, or None if not found."""
    entry = GWTC_EVENTS.get(event_id)
    return dict(entry) if entry is not None else None


def list_events() -> list[str]:
    """Return sorted list of all known event IDs."""
    return sorted(GWTC_EVENTS.keys())
