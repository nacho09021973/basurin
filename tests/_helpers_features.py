from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_minimal_canonical_features(
    run_dir: Path,
    n: int,
    dx: int,
    dy: int,
    feature_key: str,
    seed: int,
) -> None:
    outputs_dir = run_dir / "features" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dx))
    Y = rng.normal(size=(n, dy))

    np.save(outputs_dir / "X.npy", X)
    np.save(outputs_dir / "Y.npy", Y)

    payload = {
        "schema_version": "1",
        "feature_key": feature_key,
        "ids": [f"id_{i}" for i in range(n)],
        "X_path": "X.npy",
        "Y_path": "Y.npy",
        "shapes": {"X": [n, dx], "Y": [n, dy]},
        "meta": {
            "feature_key": feature_key,
            "seed": seed,
            "generated_by": "tests._helpers_features",
        },
    }
    wrapped_payload = {
        "metadata": {
            "schema_version": "1.0",
            "stage": "features",
            "run": run_dir.name,
            "created_utc": "2024-01-01T00:00:00Z",
            "source": {},
            "config": {},
            "conventions": {},
        },
        "features": payload,
    }
    (outputs_dir / "features.json").write_text(
        json.dumps(wrapped_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
