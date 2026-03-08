# Golden Geometry Specification — `mvp/golden_geometry_spec.py`

Version: **v1** (`GOLDEN_GEOMETRY_SPEC_VERSION`)

This document describes the operational definitions, formulas, and verdict
logic frozen in `mvp/golden_geometry_spec.py`.  All stages and experiments
that work with "golden geometries" MUST import from this module rather than
inventing their own thresholds or verdict strings.

---

## 1. Definitions

### 1.1 Same geometry

Two results refer to the **same geometry** if and only if they share the same
`geometry_id` string.  Geometry IDs originate from the atlas; this spec module
does not invent or validate them.

### 1.2 Compatible geometry (mode 220 / mode 221)

A geometry entry from the atlas is *compatible* with an observed ringdown mode
if the chi-squared distance between the observed `(f, tau)` and the atlas
prediction falls **strictly below** the threshold:

```
chi² = ((f_obs - f_pred) / σ_f)² + ((τ_obs - τ_pred) / σ_τ)²
```

This is a **2-degree-of-freedom** test.  Standard thresholds:

| Confidence level | chi² cut-off | Constant |
|-----------------|-------------|----------|
| 90 % | 4.605 | `DEFAULT_MODE_CHI2_THRESHOLD_90` |
| 99 % | 9.210 | `DEFAULT_MODE_CHI2_THRESHOLD_99` |

Equality is treated as rejection (strict inequality).  Callers that require a
different confidence level should pass their own threshold; the constants above
are shared defaults so that all stages agree by default.

### 1.3 Common intersection

The **common intersection** for one event is the set of `geometry_id`s that
pass *both* the mode-220 and mode-221 compatibility tests.

When mode-221 data are unavailable the verdict is
`VERDICT_SKIPPED_221_UNAVAILABLE`; the single-mode result is reported
separately.

### 1.4 Area law

The black-hole area theorem requires the post-merger horizon area to be ≥ the
pre-merger total area.  The signed surplus is:

```
delta_area = A_final - A_initial_total
```

A positive value satisfies the law.  With the default tolerance of 0 (`DEFAULT_AREA_TOLERANCE = 0.0`) the cut is `delta_area >= 0`.  A positive
tolerance `t` relaxes this to `delta_area >= -t`.

### 1.5 Golden geometry (per event)

A geometry is **golden** for a given event if it:

1. Passes the mode-220 compatibility test, AND
2. Passes the mode-221 compatibility test (or 221 is unavailable and the stage
   policy allows skipping), AND
3. Satisfies the area law.

### 1.6 Exact global intersection vs geometry most supported

| Concept | Definition |
|---------|------------|
| **Exact global intersection** | Set of `geometry_id`s that are golden in **every** event of the population. |
| **Geometry most supported** | The `geometry_id` with the highest *support count* (number of events in which it is golden). It may not appear in the exact global intersection. |
| **Robust unique** | See §1.7. |

Both are computed by helpers in this module:
- `exact_intersection_geometry_ids` → exact global intersection.
- `rank_geometries_by_support` → ordered list by support count.

### 1.7 Robust unique (per event)

A geometry is **robustly unique** for a single event if:

- It is the *sole survivor* (singleton) of all filters in at least
  `ROBUST_UNIQUE_MIN_SUPPORT_FRACTION` (default **0.80**) of the valid
  scenarios for that event.

Verdict logic (`robust_unique_verdict`):

| Condition | Verdict |
|-----------|---------|
| `n_valid_scenarios == 0` | `VERDICT_NO_DATA` |
| No scenario produced a single survivor | `VERDICT_NOT_UNIQUE` |
| Different scenarios nominate different singletons | `VERDICT_UNSTABLE_UNIQUE` |
| One geometry is singleton in ≥ 80 % of valid scenarios | `VERDICT_ROBUST_UNIQUE` |
| One geometry is singleton but below 80 % | `VERDICT_UNSTABLE_UNIQUE` |

---

## 2. Formulas

### chi² for one mode

```python
chi2 = ((obs_f - pred_f) / sigma_f) ** 2 + ((obs_tau - pred_tau) / sigma_tau) ** 2
```

No sigma floor is applied inside the spec.  Callers must supply valid
(positive, finite) sigma values.

### Area surplus

```python
delta_area = A_final - A_initial_total
passes = delta_area >= -tolerance   # tolerance defaults to 0.0
```

### Support fraction

```python
support_fraction = count_of_singleton_scenarios / n_valid_scenarios
robust = support_fraction >= ROBUST_UNIQUE_MIN_SUPPORT_FRACTION
```

---

## 3. Verdict strings (centralised constants)

```python
VERDICT_PASS                         = "PASS"
VERDICT_REJECT                       = "REJECT"
VERDICT_SKIPPED_221_UNAVAILABLE      = "SKIPPED_221_UNAVAILABLE"
VERDICT_NO_COMMON_GEOMETRIES         = "NO_COMMON_GEOMETRIES"
VERDICT_NO_GOLDEN_GEOMETRIES         = "NO_GOLDEN_GEOMETRIES"
VERDICT_EXACT_GLOBAL_GEOMETRY_FOUND  = "EXACT_GLOBAL_GEOMETRY_FOUND"
VERDICT_NO_EXACT_GLOBAL_GEOMETRY     = "NO_EXACT_GLOBAL_GEOMETRY"
VERDICT_NO_DATA                      = "NO_DATA"
VERDICT_ROBUST_UNIQUE                = "ROBUST_UNIQUE"
VERDICT_UNSTABLE_UNIQUE              = "UNSTABLE_UNIQUE"
VERDICT_NOT_UNIQUE                   = "NOT_UNIQUE"
```

---

## 4. Minimal usage examples

```python
from mvp.golden_geometry_spec import (
    chi2_mode, passes_mode_threshold, DEFAULT_MODE_CHI2_THRESHOLD_90,
    delta_area, passes_area_law,
    exact_intersection_geometry_ids, rank_geometries_by_support,
    robust_unique_verdict,
    VERDICT_ROBUST_UNIQUE,
)

# --- per-mode filter ---
c2 = chi2_mode(obs_f=250.0, obs_tau=0.01,
               pred_f=248.0, pred_tau=0.0102,
               sigma_f=5.0, sigma_tau=0.001)
ok = passes_mode_threshold(c2, DEFAULT_MODE_CHI2_THRESHOLD_90)

# --- area law ---
surplus = delta_area(final_area=95.0, initial_total_area=90.0)
area_ok = passes_area_law(surplus, tolerance=0.0)

# --- population intersection ---
per_event_golden = [
    ["geom_A", "geom_B"],
    ["geom_A", "geom_C"],
    ["geom_A"],
]
global_ids = exact_intersection_geometry_ids(per_event_golden)  # ["geom_A"]
ranked = rank_geometries_by_support(per_event_golden)

# --- single-event robustness ---
singletons = ["geom_A", "geom_A", None, "geom_A", "geom_A"]
result = robust_unique_verdict(singletons, n_valid_scenarios=5)
assert result["robustness_verdict"] == VERDICT_ROBUST_UNIQUE
```

---

## 5. Physical interpretation caveat

> **A cardinality of 1 does NOT by itself imply physical truth.**
>
> It implies **uniqueness conditioned on the atlas and on the chosen
> thresholds**.  Any claim of "the golden geometry" is therefore subject to:
>
> - Atlas completeness: the true geometry must be represented in the atlas.
> - Threshold choice: a more permissive chi² threshold will generally enlarge
>   the surviving set; a stricter threshold may empty it.
> - Measurement errors: uncertainties on `(f, tau)` and on the areas propagate
>   into the filter outcomes.
>
> The spec fixes the thresholds and vocabulary so that all stages agree on what
> they are testing.  It does not adjudicate the underlying physics.

---

## 6. Field naming for extractors

Stage-specific extractors that consume `multimode_estimates.json` or the atlas
must inspect the schema at runtime; this module does **not** hardcode field
names from those schemas.  Where field names are referenced in docstrings they
are labelled as examples, not as normative schema.

Schema names and versions emitted by the payload builders:

| Builder | `schema_name` |
|---------|--------------|
| `build_mode_filter_payload` | `golden_geometry_mode_filter` |
| `build_common_geometries_payload` | `golden_geometry_common` |
| `build_golden_geometries_payload` | `golden_geometry_per_event` |
| `build_population_consensus_payload` | `golden_geometry_population_consensus` |
| `build_single_event_robustness_payload` | `golden_geometry_single_event_robustness` |

All payloads include `schema_version`, `created_utc`, and either `run_id` +
`stage` or `experiment_run_id` + `experiment_name`.
