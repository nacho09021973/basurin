from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def pair_frames(
    ids_x: Optional[List[Any]],
    X: np.ndarray,
    ids_y: Optional[List[Any]],
    Y: np.ndarray,
    pairing_policy: str,
) -> Tuple[List[Any], np.ndarray, np.ndarray, Dict[str, Any]]:
    """Devuelve ids comunes y matrices alineadas.

    Nota: id join only when pairing_policy==id; order ignores ids.
    """
    info: Dict[str, Any] = {"pairing_policy": pairing_policy}

    if pairing_policy == "order":
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"--pairing-policy order requiere mismo N. N_X={X.shape[0]} vs N_Y={Y.shape[0]}"
            )
        ids = [f"idx_{i}" for i in range(X.shape[0])]
        pair_map = []
        for i in range(X.shape[0]):
            atlas_id = str(ids_x[i]) if ids_x is not None else None
            event_id = str(ids_y[i]) if ids_y is not None else None
            pair_map.append({"atlas_id": atlas_id, "event_id": event_id, "row_i": i})
        info.update({"paired_by": "order", "n_common": len(ids), "pair_map": pair_map})
        return ids, X, Y, info

    if pairing_policy != "id":
        raise ValueError(f"pairing_policy desconocida: {pairing_policy}")

    if ids_x is None or ids_y is None:
        raise ValueError(
            "No hay ids en uno o ambos datasets. Debes proporcionar ids con --pairing-policy id."
        )

    ix: Dict[str, int] = {}
    for idx, item in enumerate(ids_x):
        key = str(item)
        if key not in ix:
            ix[key] = idx
    iy: Dict[str, int] = {}
    for idx, item in enumerate(ids_y):
        key = str(item)
        if key not in iy:
            iy[key] = idx
    common = sorted(set(ix.keys()).intersection(set(iy.keys())))
    if not common:
        raise ValueError("Intersección por id vacía; no se pueden parear X y Y.")
    Xp = np.vstack([X[ix[c]] for c in common])
    Yp = np.vstack([Y[iy[c]] for c in common])
    pair_map = [{"atlas_id": c, "event_id": c, "row_i": i} for i, c in enumerate(common)]
    info.update({"paired_by": "id", "n_common": len(common), "pair_map": pair_map})
    return common, Xp, Yp, info
