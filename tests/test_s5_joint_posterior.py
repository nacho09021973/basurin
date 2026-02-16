from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mvp.s5_aggregate import aggregate_compatible_sets


def _source_data(include_c_in_event2: bool = True) -> list[dict[str, object]]:
    event2_rows = [
        {"geometry_id": "A", "d2": 1.0},
        {"geometry_id": "B", "d2": 1.0},
    ]
    if include_c_in_event2:
        event2_rows.append({"geometry_id": "C", "d2": 16.0})

    return [
        {
            "run_id": "runA",
            "event_id": "event1",
            "metric": "mahalanobis_log",
            "threshold_d2": 5.9915,
            "ranked_all": [
                {"geometry_id": "A", "d2": 1.0},
                {"geometry_id": "B", "d2": 4.0},
                {"geometry_id": "C", "d2": 9.0},
            ],
        },
        {
            "run_id": "runB",
            "event_id": "event2",
            "metric": "mahalanobis_log",
            "threshold_d2": 5.9915,
            "ranked_all": event2_rows,
        },
    ]


def test_joint_posterior_prefers_minimum_d2_sum() -> None:
    result = aggregate_compatible_sets(_source_data(), min_coverage=1.0, top_k=50)

    ranked = result["joint_posterior"]["joint_ranked_all"]
    by_id = {row["geometry_id"]: row for row in ranked}

    assert by_id["A"]["d2_sum"] == pytest.approx(2.0)
    assert by_id["B"]["d2_sum"] == pytest.approx(5.0)
    assert by_id["C"]["d2_sum"] == pytest.approx(25.0)
    assert result["joint_posterior"]["best_entry_id"] == "A"

    assert by_id["A"]["posterior_weight_joint"] > by_id["B"]["posterior_weight_joint"]
    assert by_id["B"]["posterior_weight_joint"] > by_id["C"]["posterior_weight_joint"]
    assert sum(row["posterior_weight_joint"] for row in ranked) == pytest.approx(1.0, abs=1e-9)


def test_min_coverage_filters_partial_geometries() -> None:
    source_data = _source_data(include_c_in_event2=False)
    result = aggregate_compatible_sets(source_data, min_coverage=1.0, top_k=50)
    relaxed = aggregate_compatible_sets(source_data, min_coverage=0.5, top_k=50)

    ranked_ids = [row["geometry_id"] for row in result["joint_posterior"]["joint_ranked_all"]]
    assert "C" not in ranked_ids

    row_c = next(
        row for row in relaxed["joint_posterior"]["joint_ranked_all"] if row["geometry_id"] == "C"
    )
    assert row_c["coverage"] == pytest.approx(0.5)

    assert result["coverage_histogram"]["1"] >= 1


def test_support_count_respects_per_event_thresholds() -> None:
    result = aggregate_compatible_sets(_source_data(), min_coverage=1.0, top_k=50)

    by_id = {row["geometry_id"]: row for row in result["joint_posterior"]["joint_ranked_all"]}
    assert by_id["A"]["support_count"] == 2
    assert by_id["B"]["support_count"] == 2
    assert by_id["C"]["support_count"] == 0
    assert by_id["C"]["support_fraction"] == pytest.approx(0.0)

    assert math.isclose(result["joint_posterior"]["log_likelihood_rel_best"], -1.0)
