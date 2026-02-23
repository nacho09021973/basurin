import json
from pathlib import Path


def _load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_atlas_real_v2_contains_gr_points_for_required_spins():
    atlas = _load("docs/ringdown/atlas/atlas_real_v2_s4.json")
    entries = atlas["entries"]

    gr_entries = [
        e
        for e in entries
        if (e.get("metadata") or {}).get("delta_f_frac") == 0.0
        and (e.get("metadata") or {}).get("delta_Q_frac") == 0.0
    ]
    assert gr_entries, "atlas_real_v2_s4 debe incluir al menos un punto GR (df=0,dQ=0)"

    spins_with_gr = {float((e.get("metadata") or {}).get("spin")) for e in gr_entries}
    assert 0.80 in spins_with_gr
    assert 0.95 in spins_with_gr


def test_atlas_real_v2_structure_compatible_with_v1():
    v1 = _load("docs/ringdown/atlas/atlas_real_v1_s4.json")
    v2 = _load("docs/ringdown/atlas/atlas_real_v2_s4.json")

    assert set(v1.keys()) == set(v2.keys()) == {"entries"}
    assert isinstance(v2["entries"], list) and v2["entries"]

    sample = v2["entries"][0]
    assert "geometry_id" in sample
    assert "metadata" in sample
    assert isinstance(sample["metadata"], dict)
