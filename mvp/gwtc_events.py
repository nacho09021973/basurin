"""GWTC catalog events: final mass, final spin, and network SNR.

Static reference data for confirmed BBH merger events from GWTC-1/2/3.

Source:
    GWTC-1: Abbott et al. (2019), PRX 9, 031040 (arXiv:1811.12907)
    GWTC-2: Abbott et al. (2021), PRX 11, 021053 (arXiv:2010.14527)
    GWTC-3: Abbott et al. (2023), PRX 13, 041039 (arXiv:2111.03606)

Values are posterior medians from the LVC parameter estimation release.
Uncertainties in the catalog posteriors are NOT propagated here — this is
a first-pass catalog for diagnostic purposes. See docstring of
compute_deviation_distribution() in s5_aggregate.py for caveats.
"""
from __future__ import annotations

from typing import Any

# fmt: off
GWTC_EVENTS: dict[str, dict[str, float]] = {
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

# Citation string for use in JSON provenance fields
GWTC_CITATION = (
    "GWTC-1 (arXiv:1811.12907), GWTC-2 (arXiv:2010.14527), "
    "GWTC-3 (arXiv:2111.03606); values are posterior medians."
)


def get_event(event_id: str) -> dict[str, float] | None:
    """Return catalog entry for event_id, or None if not found."""
    return GWTC_EVENTS.get(event_id)


def list_events() -> list[str]:
    """Return sorted list of all event IDs in the catalog."""
    return sorted(GWTC_EVENTS.keys())
