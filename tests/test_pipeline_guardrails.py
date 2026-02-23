from __future__ import annotations

import pytest

from mvp import pipeline


def test_require_nonempty_event_id_rejects_blank() -> None:
    with pytest.raises(SystemExit, match="--event-id cannot be empty"):
        pipeline._require_nonempty_event_id("   ", "--event-id")


def test_require_nonempty_event_id_accepts_trimmed() -> None:
    assert pipeline._require_nonempty_event_id(" GW150914 ", "--event-id") == "GW150914"
