from pathlib import Path

from mvp.s4_geometry_filter import _load_atlas


def test_s4_load_atlas_accepts_v2_format_smoke():
    atlas_path = Path("docs/ringdown/atlas/atlas_real_v2_s4.json")
    entries = _load_atlas(atlas_path)

    assert entries
    assert any(e.get("geometry_id") == "bK_GR_a0.80_df+0.00_dQ+0.00" for e in entries)
