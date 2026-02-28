"""Helpers de validación de schemas canónicos del MVP."""
from __future__ import annotations

from typing import Any


REQUIRED_KEYS = (
    "schema_version",
    "observables",
    "metric",
    "metric_params",
    "epsilon",
    "n_atlas",
    "n_compatible",
    "compatible_geometries",
    "ranked_all",
    "bits_excluded",
    "bits_kl",
    "likelihood_stats",
)

MAHALANOBIS_KEYS = (
    "threshold_d2",
    "d2_min",
    "distance",
    "covariance_logspace",
)


def validate_compatible_set(
    data: dict[str, Any], *, strict_mahalanobis: bool = False
) -> tuple[bool, list[str]]:
    """Valida el schema canónico de compatible_set.json.

    Retorna `(ok, errors)` con orden determinista.
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        return False, [f"data must be dict, got {type(data).__name__}"]

    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"missing required key: {key}")

    if data.get("schema_version") != "mvp_compatible_set_v1":
        errors.append("schema_version must be 'mvp_compatible_set_v1'")

    observables = data.get("observables")
    if not isinstance(observables, dict):
        errors.append("observables must be a dict")
    else:
        for key in ("f_hz", "Q"):
            val = observables.get(key)
            if not isinstance(val, (int, float)):
                errors.append(f"observables.{key} must be numeric")

    if not isinstance(data.get("metric"), str):
        errors.append("metric must be a string")

    if not isinstance(data.get("metric_params"), dict):
        errors.append("metric_params must be a dict")

    if not isinstance(data.get("epsilon"), (int, float)):
        errors.append("epsilon must be numeric")

    for key in ("n_atlas", "n_compatible"):
        if not isinstance(data.get(key), int):
            errors.append(f"{key} must be int")

    for key in ("compatible_geometries", "ranked_all"):
        if not isinstance(data.get(key), list):
            errors.append(f"{key} must be a list")

    for key in ("bits_excluded", "bits_kl"):
        if not isinstance(data.get(key), (int, float)):
            errors.append(f"{key} must be numeric")

    likelihood_stats = data.get("likelihood_stats")
    if likelihood_stats is not None and not isinstance(likelihood_stats, dict):
        errors.append("likelihood_stats must be dict or null")

    if strict_mahalanobis:
        for key in MAHALANOBIS_KEYS:
            if key not in data:
                errors.append(f"missing mahalanobis key: {key}")

    return len(errors) == 0, errors

