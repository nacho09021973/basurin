from pathlib import Path

from mvp.s2_ringdown_window import _resolve_t0_gps


def test_gw170814_has_resolvable_t0_gps_from_event_metadata() -> None:
    t0_gps, source = _resolve_t0_gps("GW170814", Path("docs/ringdown/window_catalog_v1.json"))

    assert int(t0_gps) == 1186741861
    assert source.endswith("docs/ringdown/event_metadata/GW170814_metadata.json")
