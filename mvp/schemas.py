"""Helpers de validación/normalización de schemas canónicos del MVP."""
from __future__ import annotations

from typing import Any


class SchemaError(ValueError):
    """Raised when payloads do not satisfy schema contracts."""


COMPATIBLE_SET_SCHEMA_VERSION = "mvp_compatible_set_v1"
COMPATIBLE_SET_MIN_KEYS = {"schema_version", "event_id", "compatible_geometries"}

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


def normalize_schema_version(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized copy with canonical string schema_version when supported."""
    if not isinstance(payload, dict):
        raise SchemaError(f"payload must be dict, got {type(payload).__name__}")

    normalized = dict(payload)
    if kind == "compatible_set" and normalized.get("schema_version") == 1:
        normalized["schema_version"] = COMPATIBLE_SET_SCHEMA_VERSION
    return normalized


def extract_compatible_geometry_ids(payload: dict[str, Any]) -> set[str]:
    """Extract compatible geometry IDs for canonical and legacy compatible_set payloads."""
    normalized = normalize_schema_version("compatible_set", payload)

    compatible_geometries = normalized.get("compatible_geometries")
    if isinstance(compatible_geometries, list):
        out: set[str] = set()
        for row in compatible_geometries:
            if not isinstance(row, dict) or row.get("compatible") is not True:
                continue
            geometry_id = row.get("geometry_id") if "geometry_id" in row else row.get("id")
            if isinstance(geometry_id, str) and geometry_id:
                out.add(geometry_id)
        return out

    compatible_entries = normalized.get("compatible_entries")
    if isinstance(compatible_entries, list):
        out = set()
        for row in compatible_entries:
            if isinstance(row, str) and row:
                out.add(row)
                continue
            if isinstance(row, dict):
                geometry_id = row.get("id") if "id" in row else row.get("geometry_id")
                if isinstance(geometry_id, str) and geometry_id:
                    out.add(geometry_id)
        return out

    compatible_ids = normalized.get("compatible_ids")
    if isinstance(compatible_ids, list):
        return {str(x) for x in compatible_ids}

    raise SchemaError(
        "missing supported keys: expected canonical schema or legacy keys "
        "('compatible_geometries', 'compatible_entries', 'compatible_ids')"
    )


def validate(kind: str, payload: dict[str, Any]) -> list[str]:
    """Validate payload by kind and raise SchemaError on unsupported kinds."""
    if kind == "compatible_set":
        errors = _validate_compatible_set_contract(payload)
    else:
        raise SchemaError(f"unsupported schema kind: {kind!r}")
    return errors


def _validate_compatible_set_contract(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    normalized = normalize_schema_version("compatible_set", payload)

    missing_min = sorted(COMPATIBLE_SET_MIN_KEYS.difference(normalized.keys()))
    if missing_min:
        errors.append(f"missing required keys: {', '.join(missing_min)}")

    if normalized.get("schema_version") != COMPATIBLE_SET_SCHEMA_VERSION:
        errors.append(f"schema_version must be '{COMPATIBLE_SET_SCHEMA_VERSION}'")

    event_id = normalized.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        errors.append("event_id must be a non-empty string")

    compatible_geometries = normalized.get("compatible_geometries")
    if not isinstance(compatible_geometries, list):
        errors.append("compatible_geometries must be a list")
    else:
        for row in compatible_geometries:
            if not isinstance(row, dict):
                errors.append("compatible_geometries entries must be objects")
                continue
            if "compatible" in row and not isinstance(row.get("compatible"), bool):
                errors.append("compatible_geometries[*].compatible must be bool when present")
            if row.get("compatible") is not True:
                continue
            geometry_id = row.get("geometry_id") if "geometry_id" in row else row.get("id")
            if not isinstance(geometry_id, str) or not geometry_id:
                errors.append("compatible_geometries[*] compatible rows must include non-empty geometry_id/id")

    return errors


def validate_compatible_set(
    data: dict[str, Any], *, strict_mahalanobis: bool = False,
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

    if data.get("schema_version") != COMPATIBLE_SET_SCHEMA_VERSION:
        errors.append(f"schema_version must be '{COMPATIBLE_SET_SCHEMA_VERSION}'")

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
