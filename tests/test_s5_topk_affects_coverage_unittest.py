from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mvp.s5_aggregate import aggregate_compatible_sets


def _build_source_data() -> list[dict[str, object]]:
    return [
        {
            "run_id": "run1",
            "event_id": "event1",
            "metric": "mahalanobis_log",
            "ranked_all": [
                {"geometry_id": "A", "d2": 1.0},
                {"geometry_id": "X", "d2": 2.0},
                {"geometry_id": "P", "d2": 3.0},
            ],
        },
        {
            "run_id": "run2",
            "event_id": "event2",
            "metric": "mahalanobis_log",
            "ranked_all": [
                {"geometry_id": "B", "d2": 1.0},
                {"geometry_id": "X", "d2": 2.0},
                {"geometry_id": "Q", "d2": 3.0},
            ],
        },
        {
            "run_id": "run3",
            "event_id": "event3",
            "metric": "mahalanobis_log",
            "ranked_all": [
                {"geometry_id": "C", "d2": 1.0},
                {"geometry_id": "X", "d2": 2.0},
                {"geometry_id": "R", "d2": 3.0},
            ],
        },
    ]


def test_topk_changes_coverage_histogram_common_count() -> None:
    source_data = _build_source_data()

    top1 = aggregate_compatible_sets(source_data, min_coverage=1.0, top_k=1)
    top3 = aggregate_compatible_sets(source_data, min_coverage=1.0, top_k=3)

    assert top1["coverage_histogram"].get("3", 0) == 0
    assert top3["coverage_histogram"].get("3", 0) == 1
