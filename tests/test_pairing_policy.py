import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiment" / "bridge" / "pairing.py"
SPEC = importlib.util.spec_from_file_location("pairing", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
pair_frames = MODULE.pair_frames


def test_order_ignores_id_columns():
    ids_x = [1, 2, 3]
    ids_y = [3, 2, 1]
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[30.0], [20.0], [10.0]])

    ids, Xp, Yp, info = pair_frames(ids_x, X, ids_y, Y, "order")

    assert info["paired_by"] == "order"
    assert ids == ["idx_0", "idx_1", "idx_2"]
    assert np.array_equal(Xp, X)
    assert np.array_equal(Yp, Y)


def test_id_join_independent_of_row_order():
    ids_x = [1, 2, 3]
    ids_y = [3, 1, 2]
    X = np.array([[10.0], [20.0], [30.0]])
    Y = np.array([[300.0], [100.0], [200.0]])

    ids, Xp, Yp, info = pair_frames(ids_x, X, ids_y, Y, "id")

    assert info["paired_by"] == "id"
    assert ids == ["1", "2", "3"]
    assert np.array_equal(Xp, np.array([[10.0], [20.0], [30.0]]))
    assert np.array_equal(Yp, np.array([[100.0], [200.0], [300.0]]))


def test_order_requires_same_length():
    ids_x = [1, 2]
    ids_y = [1, 2, 3]
    X = np.array([[1.0], [2.0]])
    Y = np.array([[10.0], [20.0], [30.0]])

    with pytest.raises(ValueError, match="order requiere mismo N"):
        pair_frames(ids_x, X, ids_y, Y, "order")
