"""golden_geometry_spec — Shared specification for "golden geometries" analysis.

This module defines the canonical constants, verdict strings, pure helpers, and
payload builders that all stages and experiments dealing with golden geometries
MUST use.  No IO, no subprocess, no numpy/scipy.

Definitions (operational)
--------------------------
same geometry
    Two detections refer to the same geometry if and only if they share the same
    ``geometry_id`` string.  Geometry IDs come from the atlas; this module does
    NOT invent or validate them.

compatible geometry (mode 220 / mode 221)
    A geometry is *compatible* with an observed ringdown mode if the chi-squared
    distance in (f, tau) space between the observed values and the atlas
    prediction falls below a threshold.  See ``chi2_mode`` and
    ``passes_mode_threshold``.

common intersection
    The set of geometry_ids that pass *both* the mode-220 and mode-221
    compatibility tests for a given event/scenario.

area law
    The black-hole area theorem requires the post-merger area to be ≥ the
    pre-merger total area.  ``delta_area`` returns the (signed) surplus; a
    non-negative value means the law is satisfied.

golden geometry (per event)
    A geometry that passes the mode-220 compatibility test, the mode-221
    compatibility test (when 221 data are available), and the area law check.

exact global intersection
    The set of geometry_ids that are golden in *every* event of a population.

geometry most supported
    The geometry_id with the highest support count across events (may not appear
    in the exact global intersection).

robust unique (per event or per scenario)
    A single geometry dominates: it is the sole survivor in at least
    ``ROBUST_UNIQUE_MIN_SUPPORT_FRACTION`` of the valid scenarios for that event.

Unicality caveat
----------------
A cardinality of 1 does NOT by itself imply physical truth.  It implies
uniqueness conditioned on the atlas and on the chosen thresholds.  Any claim of
"the geometry" is subject to atlas completeness and threshold choice.
"""
from __future__ import annotations

import datetime
from typing import Any

# ---------------------------------------------------------------------------
# A. Specification version
# ---------------------------------------------------------------------------

GOLDEN_GEOMETRY_SPEC_VERSION: str = "v1"

# ---------------------------------------------------------------------------
# B. Mode labels
# ---------------------------------------------------------------------------

MODE_220: str = "220"
MODE_221: str = "221"

# ---------------------------------------------------------------------------
# C. Default numerical thresholds
# ---------------------------------------------------------------------------

# chi-squared cut-offs for a 2-parameter (f, tau) Gaussian test.
# These correspond to the 90th and 99th percentiles of chi²(2 d.o.f.).
# Callers that want a different confidence level should pass their own
# threshold; these are the shared defaults so that stages agree.
DEFAULT_MODE_CHI2_THRESHOLD_90: float = 4.605   # chi²(2) at 90 % CL
DEFAULT_MODE_CHI2_THRESHOLD_99: float = 9.210   # chi²(2) at 99 % CL

# Area-law tolerance: by default no deficit is allowed (exact non-negativity).
DEFAULT_AREA_TOLERANCE: float = 0.0

# Fraction of valid scenarios in which a single geometry must appear as the
# singleton to be declared "robustly unique".
ROBUST_UNIQUE_MIN_SUPPORT_FRACTION: float = 0.80

# ---------------------------------------------------------------------------
# D. Verdict strings — centralised so stages never hardcode raw strings
# ---------------------------------------------------------------------------

VERDICT_PASS: str = "PASS"
VERDICT_REJECT: str = "REJECT"

VERDICT_SKIPPED_221_UNAVAILABLE: str = "SKIPPED_221_UNAVAILABLE"
VERDICT_NO_COMMON_GEOMETRIES: str = "NO_COMMON_GEOMETRIES"
VERDICT_NO_GOLDEN_GEOMETRIES: str = "NO_GOLDEN_GEOMETRIES"

VERDICT_EXACT_GLOBAL_GEOMETRY_FOUND: str = "EXACT_GLOBAL_GEOMETRY_FOUND"
VERDICT_NO_EXACT_GLOBAL_GEOMETRY: str = "NO_EXACT_GLOBAL_GEOMETRY"

VERDICT_NO_DATA: str = "NO_DATA"
VERDICT_ROBUST_UNIQUE: str = "ROBUST_UNIQUE"
VERDICT_UNSTABLE_UNIQUE: str = "UNSTABLE_UNIQUE"
VERDICT_NOT_UNIQUE: str = "NOT_UNIQUE"

# ---------------------------------------------------------------------------
# E. Pure physics / score helpers
# ---------------------------------------------------------------------------


def chi2_mode(
    obs_f: float,
    obs_tau: float,
    pred_f: float,
    pred_tau: float,
    sigma_f: float,
    sigma_tau: float,
) -> float:
    """Return the chi-squared distance for one ringdown mode.

    chi² = ((obs_f - pred_f) / sigma_f)² + ((obs_tau - pred_tau) / sigma_tau)²

    This is a 2-degree-of-freedom test on (frequency, damping time).

    Callers are responsible for supplying sigma values that are strictly
    positive and finite.  No floor is applied here; the spec does not hide
    degenerate inputs.

    Parameters
    ----------
    obs_f, obs_tau   : observed frequency (Hz) and damping time (s).
    pred_f, pred_tau : atlas-predicted frequency and damping time.
    sigma_f, sigma_tau : uncertainties (same units as obs).  Must be > 0.
    """
    r_f = (obs_f - pred_f) / sigma_f
    r_tau = (obs_tau - pred_tau) / sigma_tau
    return r_f * r_f + r_tau * r_tau


def passes_mode_threshold(chi2: float, threshold: float) -> bool:
    """Return True if chi2 is strictly below the threshold.

    Equality is treated as rejection so that the boundary does not count as
    compatible (conservative choice; stages may override by passing a
    threshold derived from a looser CL).
    """
    return chi2 < threshold


def delta_area(final_area: float, initial_total_area: float) -> float:
    """Return the signed area surplus: A_final - A_initial.

    A positive value means the area theorem is satisfied.
    A negative value means a deficit (violation within measurement errors).

    No normalisation is applied; units are those of the caller.
    """
    return final_area - initial_total_area


def passes_area_law(delta_area_value: float, tolerance: float) -> bool:
    """Return True if the area surplus meets or exceeds -tolerance.

    With the default tolerance of 0.0 this requires delta_area_value >= 0.
    A positive tolerance relaxes the cut to allow small deficits (e.g. from
    measurement uncertainty).
    """
    return delta_area_value >= -tolerance


# ---------------------------------------------------------------------------
# F. Set / consensus helpers
# ---------------------------------------------------------------------------


def exact_intersection_geometry_ids(list_of_id_iterables: list) -> list:
    """Return geometry_ids present in every non-empty iterable.

    "Same geometry" is defined by exact string equality of geometry_id.

    Parameters
    ----------
    list_of_id_iterables : sequence of iterables of geometry_id strings.
        Empty inner iterables contribute nothing to the intersection
        (i.e. if any inner iterable is empty the result is empty).

    Returns
    -------
    Sorted list of geometry_id strings that appear in every inner iterable.
    An empty list if any inner iterable is empty or the outer list is empty.

    This is the *exact global intersection* definition: a geometry must
    survive in ALL sub-sets, not just most of them.
    """
    if not list_of_id_iterables:
        return []
    sets = [set(ids) for ids in list_of_id_iterables]
    common = sets[0]
    for s in sets[1:]:
        common = common & s
    return sorted(common)


def support_count_geometry_ids(list_of_id_iterables: list) -> dict:
    """Return a mapping geometry_id -> count of inner iterables that contain it.

    Counts how many sub-lists each geometry appears in (not total occurrences
    within a single sub-list).  This is the *support count* underlying
    "geometry most supported" and "robust unique" logic.
    """
    counts: dict[str, int] = {}
    for ids in list_of_id_iterables:
        for gid in set(ids):  # deduplicate within each inner iterable
            counts[gid] = counts.get(gid, 0) + 1
    return counts


def rank_geometries_by_support(list_of_id_iterables: list) -> list:
    """Return geometry rows ordered by descending support count.

    Each row is a dict: {"geometry_id": str, "support_count": int}.

    Ties are broken by lexicographic geometry_id (ascending) for
    deterministic output across runs.

    Difference from exact_intersection_geometry_ids
    -----------------------------------------------
    * exact_intersection requires presence in ALL sub-lists.
    * rank_geometries_by_support counts presence in each sub-list and
      orders by that count; geometries not in all sub-lists still appear.
    """
    counts = support_count_geometry_ids(list_of_id_iterables)
    rows = [{"geometry_id": gid, "support_count": cnt} for gid, cnt in counts.items()]
    rows.sort(key=lambda r: (-r["support_count"], r["geometry_id"]))
    return rows


def singleton_geometry_id(ids: list) -> "str | None":
    """Return the single element if ids has exactly one element, else None.

    Used to test whether a given scenario produced a unique geometry after
    all filters have been applied.
    """
    if len(ids) == 1:
        return str(ids[0])
    return None


def robust_unique_verdict(
    singleton_ids: list,
    n_valid_scenarios: int,
    min_support_fraction: float = ROBUST_UNIQUE_MIN_SUPPORT_FRACTION,
) -> dict:
    """Determine the robustness verdict for a single-event geometry uniqueness claim.

    Parameters
    ----------
    singleton_ids : list of geometry_id strings or None values.
        Each entry corresponds to one scenario.  A non-None entry means that
        scenario produced exactly one surviving geometry (the entry value).
        A None entry means the scenario produced 0 or >1 surviving geometries.
    n_valid_scenarios : int
        Total count of scenarios considered valid (including those that gave
        0 or >1 survivors).
    min_support_fraction : float
        Fraction of valid scenarios in which a single geometry must dominate
        to qualify as ROBUST_UNIQUE.  Default: ROBUST_UNIQUE_MIN_SUPPORT_FRACTION.

    Returns
    -------
    dict with keys:
        robustness_verdict : one of VERDICT_NO_DATA, VERDICT_NOT_UNIQUE,
                             VERDICT_UNSTABLE_UNIQUE, VERDICT_ROBUST_UNIQUE.
        robust_unique_geometry_id : str or None.
        support_fraction : float or None.
        singleton_counts : dict mapping geometry_id -> count of scenarios
                           in which it was the singleton.

    Logic
    -----
    1. n_valid_scenarios == 0 → NO_DATA.
    2. No non-None singleton_ids → NOT_UNIQUE (no scenario produced a unique
       survivor).
    3. Non-None singleton_ids from more than one distinct geometry →
       UNSTABLE_UNIQUE (different scenarios disagree on which geometry survives).
    4. A single geometry appears as singleton in ≥ min_support_fraction of
       n_valid_scenarios → ROBUST_UNIQUE.
    5. Otherwise → UNSTABLE_UNIQUE.
    """
    if n_valid_scenarios == 0:
        return {
            "robustness_verdict": VERDICT_NO_DATA,
            "robust_unique_geometry_id": None,
            "support_fraction": None,
            "singleton_counts": {},
        }

    # Count how many times each geometry appeared as the unique survivor.
    singleton_counts: dict[str, int] = {}
    for gid in singleton_ids:
        if gid is not None:
            singleton_counts[gid] = singleton_counts.get(gid, 0) + 1

    if not singleton_counts:
        # No scenario ever produced a unique survivor.
        return {
            "robustness_verdict": VERDICT_NOT_UNIQUE,
            "robust_unique_geometry_id": None,
            "support_fraction": None,
            "singleton_counts": singleton_counts,
        }

    distinct_geometries = list(singleton_counts.keys())

    if len(distinct_geometries) > 1:
        # Multiple geometries compete; the claim is unstable.
        return {
            "robustness_verdict": VERDICT_UNSTABLE_UNIQUE,
            "robust_unique_geometry_id": None,
            "support_fraction": None,
            "singleton_counts": singleton_counts,
        }

    # Exactly one distinct geometry appeared as singleton.
    sole_geometry = distinct_geometries[0]
    count = singleton_counts[sole_geometry]
    fraction = float(count) / float(n_valid_scenarios)

    if fraction >= min_support_fraction:
        verdict_str = VERDICT_ROBUST_UNIQUE
    else:
        verdict_str = VERDICT_UNSTABLE_UNIQUE

    return {
        "robustness_verdict": verdict_str,
        "robust_unique_geometry_id": sole_geometry if verdict_str == VERDICT_ROBUST_UNIQUE else None,
        "support_fraction": fraction,
        "singleton_counts": singleton_counts,
    }


# ---------------------------------------------------------------------------
# G. Payload builders — avoid schema divergence across stages
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return current UTC time as an ISO 8601 string (seconds precision, Z suffix)."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_mode_filter_payload(
    *,
    run_id: str,
    stage: str,
    mode: str,
    geometry_ids: list,
    chi2_threshold: float,
    verdict: str,
    created_utc: "str | None" = None,
    **extra: Any,
) -> dict:
    """Build the canonical payload for a single-mode geometry filter result.

    Parameters
    ----------
    run_id       : pipeline run identifier.
    stage        : name of the stage producing this payload.
    mode         : mode label, e.g. MODE_220 or MODE_221.
    geometry_ids : list of geometry_ids that passed the filter.
    chi2_threshold : threshold used; one of DEFAULT_MODE_CHI2_THRESHOLD_*.
    verdict      : one of the VERDICT_* constants.
    created_utc  : ISO 8601 string; defaults to current UTC.
    **extra      : additional fields merged into the payload dict.
    """
    payload: dict[str, Any] = {
        "schema_name": "golden_geometry_mode_filter",
        "schema_version": GOLDEN_GEOMETRY_SPEC_VERSION,
        "created_utc": created_utc or _utc_now_iso(),
        "run_id": run_id,
        "stage": stage,
        "mode": mode,
        "chi2_threshold": chi2_threshold,
        "geometry_ids": list(geometry_ids),
        "n_passed": len(geometry_ids),
        "verdict": verdict,
    }
    payload.update(extra)
    return payload


def build_common_geometries_payload(
    *,
    run_id: str,
    stage: str,
    common_geometry_ids: list,
    verdict: str,
    created_utc: "str | None" = None,
    **extra: Any,
) -> dict:
    """Build the canonical payload for the mode-220 ∩ mode-221 intersection result.

    Parameters
    ----------
    run_id               : pipeline run identifier.
    stage                : name of the stage producing this payload.
    common_geometry_ids  : geometry_ids in the intersection (may be empty).
    verdict              : VERDICT_NO_COMMON_GEOMETRIES or a passing verdict.
    created_utc          : ISO 8601 string; defaults to current UTC.
    **extra              : additional fields merged into the payload dict.
    """
    payload: dict[str, Any] = {
        "schema_name": "golden_geometry_common",
        "schema_version": GOLDEN_GEOMETRY_SPEC_VERSION,
        "created_utc": created_utc or _utc_now_iso(),
        "run_id": run_id,
        "stage": stage,
        "common_geometry_ids": list(common_geometry_ids),
        "n_common": len(common_geometry_ids),
        "verdict": verdict,
    }
    payload.update(extra)
    return payload


def build_golden_geometries_payload(
    *,
    run_id: str,
    stage: str,
    golden_geometry_ids: list,
    verdict: str,
    created_utc: "str | None" = None,
    **extra: Any,
) -> dict:
    """Build the canonical payload for the per-event golden geometry result.

    "Golden geometries" are those that pass both mode filters and the area law.

    Parameters
    ----------
    run_id               : pipeline run identifier.
    stage                : name of the stage producing this payload.
    golden_geometry_ids  : geometry_ids classified as golden for this event.
    verdict              : e.g. VERDICT_NO_GOLDEN_GEOMETRIES, VERDICT_PASS, etc.
    created_utc          : ISO 8601 string; defaults to current UTC.
    **extra              : additional fields merged into the payload dict.
    """
    payload: dict[str, Any] = {
        "schema_name": "golden_geometry_per_event",
        "schema_version": GOLDEN_GEOMETRY_SPEC_VERSION,
        "created_utc": created_utc or _utc_now_iso(),
        "run_id": run_id,
        "stage": stage,
        "golden_geometry_ids": list(golden_geometry_ids),
        "n_golden": len(golden_geometry_ids),
        "verdict": verdict,
    }
    payload.update(extra)
    return payload


def build_population_consensus_payload(
    *,
    experiment_run_id: str,
    experiment_name: str,
    exact_global_geometry_ids: list,
    ranked_by_support: list,
    verdict: str,
    created_utc: "str | None" = None,
    **extra: Any,
) -> dict:
    """Build the canonical payload for the population-level geometry consensus.

    Distinguishes between:
    * exact global intersection — geometry_ids present in ALL events.
    * ranked by support        — ordered list from rank_geometries_by_support.

    Parameters
    ----------
    experiment_run_id        : run identifier for the experiment.
    experiment_name          : name of the experiment producing this payload.
    exact_global_geometry_ids : geometry_ids in the exact global intersection.
    ranked_by_support        : output of rank_geometries_by_support.
    verdict                  : VERDICT_EXACT_GLOBAL_GEOMETRY_FOUND or
                               VERDICT_NO_EXACT_GLOBAL_GEOMETRY.
    created_utc              : ISO 8601 string; defaults to current UTC.
    **extra                  : additional fields merged into the payload dict.
    """
    payload: dict[str, Any] = {
        "schema_name": "golden_geometry_population_consensus",
        "schema_version": GOLDEN_GEOMETRY_SPEC_VERSION,
        "created_utc": created_utc or _utc_now_iso(),
        "experiment_run_id": experiment_run_id,
        "experiment_name": experiment_name,
        "exact_global_geometry_ids": list(exact_global_geometry_ids),
        "n_exact_global": len(exact_global_geometry_ids),
        "ranked_by_support": list(ranked_by_support),
        "verdict": verdict,
    }
    payload.update(extra)
    return payload


def build_single_event_robustness_payload(
    *,
    run_id: str,
    stage: str,
    robustness_result: dict,
    n_valid_scenarios: int,
    min_support_fraction: float = ROBUST_UNIQUE_MIN_SUPPORT_FRACTION,
    created_utc: "str | None" = None,
    **extra: Any,
) -> dict:
    """Build the canonical payload for single-event robust-uniqueness assessment.

    Parameters
    ----------
    run_id               : pipeline run identifier.
    stage                : name of the stage producing this payload.
    robustness_result    : dict returned by ``robust_unique_verdict``.
    n_valid_scenarios    : number of valid scenarios used.
    min_support_fraction : threshold applied; stored for audit.
    created_utc          : ISO 8601 string; defaults to current UTC.
    **extra              : additional fields merged into the payload dict.
    """
    payload: dict[str, Any] = {
        "schema_name": "golden_geometry_single_event_robustness",
        "schema_version": GOLDEN_GEOMETRY_SPEC_VERSION,
        "created_utc": created_utc or _utc_now_iso(),
        "run_id": run_id,
        "stage": stage,
        "n_valid_scenarios": n_valid_scenarios,
        "min_support_fraction": min_support_fraction,
        **robustness_result,
    }
    payload.update(extra)
    return payload
