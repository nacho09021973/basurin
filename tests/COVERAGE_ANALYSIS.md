# Test Coverage Analysis

## Current State

**66 tests passing** across 7 test files (~1,950 lines of test code).
**Test-to-source ratio:** ~0.75 (tests are 75% the size of production code).

| Test File | Tests | What It Covers |
|---|---|---|
| `test_mvp_contracts.py` | 13 | Contract registry, init/check/finalize/abort/enforce |
| `test_mvp_stages.py` | 13 | End-to-end stage subprocess tests (s1–s5), integration, reuse |
| `test_s3_uncertainty_schema.py` | 6 | Uncertainty propagation schema, determinism |
| `test_s4_mahalanobis.py` | 16 | Mahalanobis distance, chi² threshold, singular covariance, backward compat |
| `test_s6_information_geometry.py` | 11 | Flat-space regression, curvature, conformal distance, CLI contract |
| `test_eps_sweep.py` | 7 | Epsilon sweep artifacts, monotonicity, schema, error cases |
| `test_atlas_real_v1_hash.py` | 1 | Golden SHA-256 of atlas file |

## Gap 1: `basurin_io.py` Has No Unit Tests (219 lines)

This module is the I/O foundation of the entire pipeline. Every stage depends on it. While it gets exercised *indirectly* through integration tests, there are no isolated unit tests. Functions that should be tested directly:

- **`validate_run_id()`** — Only tested indirectly via `init_stage`. No direct tests for edge cases:
  - Empty string
  - Exactly 128 characters (boundary)
  - 129 characters (should fail)
  - Characters like `@`, `/`, `..` (path traversal), spaces, unicode
- **`resolve_out_root()`** — No test for the `BASURIN_RUNS_ROOT` env var path vs. default `cwd/runs` path. No test that the directory is actually created.
- **`require_run_valid()`** — No test for missing file, wrong verdict value, or malformed JSON.
- **`sha256_file()`** — No test against a known hash. Critical for provenance integrity.
- **`write_json_atomic()`** — No test for atomicity guarantees, Path coercion, or nested Path objects in data.
- **`_coerce_paths()`** — No test for recursive Path-to-str conversion in nested dicts/lists.
- **`write_manifest()`** — No test that it produces the correct schema with hashes.
- **`write_stage_summary()`** — Trivial wrapper, but still untested directly.

**Priority: HIGH** — A regression in any of these functions would silently corrupt pipeline outputs.

## Gap 2: `mvp/pipeline.py` Has No Tests (388 lines)

The pipeline orchestrator is the most complex untested module. It contains:

- **`run_single_event()`** — Orchestrates s1→s4. No test that:
  - It creates `RUN_VALID` before stages
  - It stops on first stage failure (abort semantics)
  - It writes `pipeline_timeline.json` with correct schema
  - Timeline records correct start/end timestamps and durations
  - It returns the correct exit code on success/failure
- **`run_multi_event()`** — Orchestrates multiple events + s5 aggregate. No test that:
  - Each event gets its own run_id
  - Failure of any event aborts the whole multi-event pipeline
  - Aggregation is only attempted after all events succeed
  - `agg_run_id` is properly generated when not provided
- **`_run_stage()` internal** — No test for:
  - Stage timeout handling (kill + exit code 124)
  - Heartbeat thread lifecycle (starts and stops cleanly)
  - Timeline entry schema correctness
- **`_generate_run_id()`** — No test for format
- **CLI argument parsing** — No test for `main()` with `single`/`multi` subcommands

**Priority: HIGH** — The orchestrator controls pipeline correctness. A bug here (e.g., wrong argument forwarding, timeline not written on failure) would be caught only in manual runs.

## Gap 3: `s5_aggregate.py` — `aggregate_compatible_sets()` Has Minimal Unit Tests

The `test_mvp_stages.py::TestS5Aggregate` class has only 1 test (intersection of two events). Missing:

- **Edge case: zero events** — `aggregate_compatible_sets([])` should return empty result.
- **Edge case: single event** — All geometries should be "common" with coverage 1.0.
- **Partial coverage** — `min_coverage=0.5` with 4 events should include geometries seen in >= 2 events.
- **No overlap** — Two events with completely disjoint compatible sets should produce empty intersection at `min_coverage=1.0`.
- **Coverage histogram correctness** — Verify the histogram matches actual counts.
- **Metadata preservation** — Verify that metadata from the first occurrence is retained.
- **NaN distances** — Verify handling when some `distance` values are missing or NaN.

**Priority: MEDIUM** — The function is simple but multi-event analysis is a core use case.

## Gap 4: `s2_ringdown_window.py` — `_resolve_t0_gps()` Is Untested

The `_resolve_t0_gps()` function has three lookup paths:
1. Window catalog file with `windows[].t0_ref.value_gps`
2. Event metadata file with `t_coalescence_gps`, `t0_ref_gps`, or `GPS` keys
3. `RuntimeError` if neither source has data

None of these paths are tested in isolation. The stage tests only exercise path 2 (metadata file) indirectly. Missing:

- Test catalog lookup (path 1)
- Test fallback from missing catalog to metadata file
- Test `RuntimeError` when event is unknown
- Test priority order of metadata keys (`t_coalescence_gps` vs `GPS`)

**Priority: MEDIUM** — Incorrect t0 resolution would silently produce wrong ringdown windows.

## Gap 5: Error and Abort Path Coverage Is Incomplete

While some abort paths are tested (empty atlas, out-of-range window, singular covariance), many are not:

- **s2**: No test for `fs <= 0`, `duration_s <= 0`, non-1D strain, NaN/Inf strain
- **s3**: No test for what happens when all detectors fail estimation (e.g., flat noise with no ringdown signal)
- **s4**: No test for atlas entries with missing `f_hz` or `Q` fields
- **s5**: No test for `--source-runs ""` (empty) or source run with missing `compatible_set.json`
- **s6**: No test for empty compatible geometries list, or geometries with `f_hz <= 0`

**Priority: MEDIUM** — These are defensive checks that prevent garbage-in-garbage-out.

## Gap 6: `mvp/tools/generate_real_atlas.py` Has No Tests (257 lines)

This module has testable pure functions:

- **`omega_to_physical()`** — Converts dimensionless QNM frequency to physical units. Should be tested against published values (e.g., GW150914 expected f ~ 251 Hz, Q ~ 4).
- **`generate_beyond_kerr_entries()`** — Parametric deviation logic. Should verify:
  - Correct fractional shifts are applied
  - `df=0, dq=0` is skipped (no Kerr duplicate)
  - `phi_atlas` coordinates are correct `[log(f), log(Q)]`
- **`generate_kerr_atlas()`** — Depends on `qnm` package (external), so harder to test without it. But the function's output structure can be validated.

**Priority: LOW** — The golden hash test (`test_atlas_real_v1_hash.py`) provides a coarse regression guard, but it won't localize which function broke if the hash changes.

## Gap 7: No Coverage Measurement Infrastructure

There is no:
- `pytest-cov` configuration
- `.coveragerc` file
- CI/CD pipeline running tests automatically
- Pre-commit hooks for test execution

Without line-level coverage data, this analysis is based on code reading. Actual branch coverage may be lower than estimated.

**Priority: MEDIUM** — Adding `pytest-cov` would provide concrete numbers and catch regressions.

## Gap 8: No `conftest.py` With Shared Fixtures

Multiple test files duplicate the same helper functions:
- `_create_run_valid()` — duplicated in 5 test files
- `_run_stage()` — duplicated in 4 test files
- `_assert_stage_contract()` — duplicated in 2 test files
- `_create_s2_outputs()` / `_create_synthetic_strain()` — similar but slightly different

A `conftest.py` with shared fixtures would reduce duplication and make tests easier to maintain.

**Priority: LOW** — Not a coverage gap, but a maintainability issue that makes writing new tests harder.

## Recommended Priority Order

1. **`basurin_io.py` unit tests** — Foundation module, high impact, easy to test (pure functions)
2. **`pipeline.py` orchestrator tests** — Complex logic with no coverage, high risk
3. **`s5_aggregate` unit tests for `aggregate_compatible_sets()`** — Important function with only 1 integration test
4. **`s2_ringdown_window._resolve_t0_gps()` unit tests** — Three code paths, zero tests
5. **Error/abort path tests** — Cover the defensive checks across stages
6. **`pytest-cov` setup** — Get concrete coverage numbers
7. **`conftest.py` refactoring** — Shared fixtures for maintainability
8. **`generate_real_atlas.py` function tests** — Lower priority due to golden hash guard
