import json
from pathlib import Path


def _load_fixture(name: str) -> dict:
    fixture_path = Path(__file__).parent / "fixtures" / name
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _rank_shift_stats(curvature_payload: dict) -> tuple[int, int]:
    """Compute reranking shifts using s6 formula: rank_delta = rank_flat - rank_conformal."""
    deltas = [
        int(g["rank_flat"]) - int(g["rank_conformal"])
        for g in curvature_payload.get("reranked_geometries", [])
    ]
    if not deltas:
        return 0, 0
    n_moved = sum(1 for d in deltas if d != 0)
    max_abs_delta_rank = max(abs(d) for d in deltas)
    return n_moved, max_abs_delta_rank


def test_s6_rerank_contrast_demo_phase3():
    # Regeneración de fixtures (copiar desde runs reales):
    # cp runs/<RUN_GW150914>/s6_information_geometry/outputs/curvature.json \
    #   tests/fixtures/s6_curvature_GW150914.json
    # cp runs/<RUN_GW170814>/s6_information_geometry/outputs/curvature.json \
    #   tests/fixtures/s6_curvature_GW170814.json
    gw150914 = _load_fixture("s6_curvature_GW150914.json")
    gw170814 = _load_fixture("s6_curvature_GW170814.json")

    n_moved_150914, max_abs_150914 = _rank_shift_stats(gw150914)
    n_moved_170814, max_abs_170814 = _rank_shift_stats(gw170814)

    assert n_moved_150914 > 0
    assert max_abs_150914 >= 10

    assert n_moved_170814 == 0
    assert max_abs_170814 == 0
