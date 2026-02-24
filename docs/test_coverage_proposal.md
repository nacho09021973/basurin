# Test Coverage Improvement Proposal

**Date:** 2026-02-24
**Branch:** `claude/analyze-test-coverage-wHuq2`
**Scope:** `/home/user/basurin` — full codebase audit

---

## Current State

The codebase has **79 test files** (~499 test functions) across `tests/` and `tests/integration/`.
Coverage is uneven: the spectral estimator, oracle, contracts, and experiment scripts are
well-tested, while several critical modules have partial or zero coverage.

---

## Gap 1 — `basurin_io.py`: Core I/O functions lack direct unit tests [HIGH]

Only `write_json_atomic` is exercised directly (3 symlink-guard tests). Every other public
function is hit only indirectly through subprocess-level integration tests, which means a
regression would surface as a confusing pipeline failure rather than a clear unit-test failure.

| Function | Gap |
|---|---|
| `validate_run_id()` | No edge cases: empty string, exactly-128-char boundary, 129-char (should fail), path-traversal chars (`/`, `..`), unicode |
| `sha256_file()` | Never asserted against a known hash — critical for provenance integrity |
| `require_run_valid()` | No test for missing file, wrong verdict (`"FAIL"`), or malformed JSON |
| `resolve_out_root()` | No test comparing `BASURIN_RUNS_ROOT` env-var path vs. `cwd/runs` default |
| `write_manifest()` | No test that the output schema contains correct hashes for listed files |
| `_coerce_paths()` | No test for recursive `Path → str` conversion in nested dicts/lists |

**Suggested test file:** `tests/test_basurin_io_unit.py`

Example cases to add:

```python
def test_validate_run_id_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        validate_run_id("", Path("/tmp"))

def test_validate_run_id_at_boundary_128():
    validate_run_id("a" * 128, Path("/tmp"))  # should not raise

def test_validate_run_id_too_long_129():
    with pytest.raises(ValueError, match="too long"):
        validate_run_id("a" * 129, Path("/tmp"))

def test_validate_run_id_path_traversal():
    with pytest.raises(ValueError):
        validate_run_id("../escape", Path("/tmp"))

def test_sha256_file_known_hash(tmp_path):
    f = tmp_path / "test.json"
    f.write_bytes(b'{"a":1}')
    assert sha256_file(f) == hashlib.sha256(b'{"a":1}').hexdigest()

def test_require_run_valid_wrong_verdict(tmp_path):
    rv = tmp_path / "RUN_VALID" / "verdict.json"
    rv.parent.mkdir()
    rv.write_text('{"verdict":"FAIL"}')
    with pytest.raises(SystemExit):
        require_run_valid(tmp_path.parent, tmp_path.name)
```

---

## Gap 2 — `mvp/pipeline.py`: Orchestrator logic is almost entirely untested [HIGH]

`test_pipeline_guardrails.py` has only 2 tests (`_require_nonempty_event_id`). The 1028-line
orchestrator contains multiple complex functions with zero test coverage:

| Function | Lines | Gap |
|---|---|---|
| `run_single_event()` | 334–540 | No test for abort-on-first-failure, `pipeline_timeline.json` schema, `RUN_VALID` creation order |
| `run_multimode_event()` | 541–669 | 128 lines of multimode orchestration logic with no tests |
| `run_multi_event()` | 670–764 | No test that one-event failure aborts the rest, or that aggregation requires all events to pass |
| `_run_stage()` | 141–210 | No test for timeout handling (kill + exit code 124) or heartbeat thread lifecycle |
| `_generate_run_id()` | 96–99 | No test for output format |

**Suggested test file:** `tests/test_pipeline_orchestrator.py`

Key scenarios to add:

```python
def test_run_single_event_creates_run_valid_before_stages(tmp_path, monkeypatch):
    # Verify RUN_VALID is written before any stage subprocess is launched

def test_run_single_event_aborts_on_stage_failure(tmp_path, monkeypatch):
    # First stage fails → subsequent stages are never called

def test_pipeline_timeline_written_on_success(tmp_path, monkeypatch):
    # pipeline_timeline.json exists with correct keys: stages, start, end, duration_s

def test_run_multi_event_aborts_if_any_event_fails(tmp_path, monkeypatch):
    # One of two events fails → aggregate stage is never invoked

def test_generate_run_id_format():
    run_id = _generate_run_id("GW150914")
    assert run_id.startswith("GW150914_")
    assert re.match(r"^[A-Za-z0-9._-]+$", run_id)
```

---

## Gap 3 — `mvp/extract_psd.py`: Completely untested [HIGH]

`extract_psd()` (180 lines, 1 public function) has **zero references anywhere in the test
suite**. It is the only stage-level module with no test at all — not even a CLI smoke test.

**Suggested test file:** `tests/test_extract_psd.py`

```python
def test_extract_psd_output_shape_and_positivity(tmp_path):
    # Synthetic white noise → PSD values all positive, freq array correct length

def test_extract_psd_flat_noise_has_expected_level():
    # RMS-normalised white noise → PSD near expected theoretical level

def test_extract_psd_zero_duration_raises():
    with pytest.raises(ValueError):
        extract_psd(np.zeros(0), fs=4096.0, ...)

def test_extract_psd_nan_in_strain_raises():
    strain = np.full(4096, np.nan)
    with pytest.raises(ValueError):
        extract_psd(strain, fs=4096.0, ...)
```

---

## Gap 4 — `bootstrap_ringdown_observables()` in `s3_ringdown_estimates.py`: Untested [MEDIUM]

`estimate_ringdown_observables` (Hilbert) and `estimate_ringdown_spectral` (Lorentzian) are
thoroughly tested. The third estimator, `bootstrap_ringdown_observables()` (lines 132–224),
has **zero coverage** — no reference appears anywhere in the test suite.

**Suggested addition to** `tests/test_s3_spectral_lorentzian.py`:

```python
class TestBootstrapEstimator:
    def test_output_schema_keys(self):
        result = bootstrap_ringdown_observables(SYNTHETIC_STRAIN, FS, BAND)
        for key in ("f_hz", "sigma_f_hz", "Q", "sigma_Q", "cov_logf_logQ"):
            assert key in result

    def test_bootstrap_converges_on_synthetic_ringdown(self):
        result = bootstrap_ringdown_observables(SYNTHETIC_STRAIN, FS, BAND)
        assert abs(result["f_hz"] - F_TRUE) / F_TRUE < 0.05  # within 5%

    def test_bootstrap_n_zero_raises_cleanly(self):
        with pytest.raises(ValueError):
            bootstrap_ringdown_observables(SYNTHETIC_STRAIN, FS, BAND, n_bootstrap=0)
```

---

## Gap 5 — `mvp/s4c_kerr_consistency.py`: Business logic functions untested [MEDIUM]

`_mode_row()` and `_infer_censoring()` are pure functions determining whether multimode results
are censored. No unit tests exist for them.

**Suggested test file:** `tests/test_s4c_kerr_consistency.py`

```python
def test_infer_censoring_hard_violation():
    multimode = {"mode_221": {"f_hz": 9999.0, "Q": 100.0}}  # far from Kerr
    censored, reason, weight = _infer_censoring(multimode, mode_221=...)
    assert censored is True
    assert "hard" in reason.lower()

def test_infer_censoring_no_violation():
    multimode = {"mode_221": {"f_hz": 251.0, "Q": 4.0}}  # near Kerr
    censored, reason, weight = _infer_censoring(multimode, mode_221=...)
    assert censored is False

def test_mode_row_returns_none_for_missing_label():
    multimode = {"mode_221": {"f_hz": 251.0}}
    assert _mode_row(multimode, "mode_330") is None
```

---

## Gap 6 — `s2_ringdown_window._resolve_t0_gps()`: Three code paths, none tested in isolation [MEDIUM]

The function resolves the ringdown start time via three separate lookup paths:
1. Window catalog file (`windows[].t0_ref.value_gps`)
2. Event metadata file (`t_coalescence_gps` → `t0_ref_gps` → `GPS`, in priority order)
3. `RuntimeError` if neither source yields a value

Path 2 is exercised only indirectly. Paths 1 and 3 are never exercised. An incorrect t0
silently produces wrong ringdown windows.

**Suggested addition to** `tests/test_s2_t0_gps_catalog_unittest.py`:

```python
def test_resolve_t0_gps_catalog_path(tmp_path):
    # Create window catalog with t0_ref.value_gps → assert correct value returned

def test_resolve_t0_gps_metadata_coalescence_key(tmp_path):
    # Metadata has t_coalescence_gps → that value wins over GPS key

def test_resolve_t0_gps_metadata_gps_fallback(tmp_path):
    # Metadata has only GPS key → that value is returned

def test_resolve_t0_gps_unknown_event_raises(tmp_path):
    with pytest.raises(RuntimeError, match="unknown"):
        _resolve_t0_gps("GW_DOES_NOT_EXIST", ...)
```

---

## Gap 7 — Error/abort path coverage is incomplete across pipeline stages [MEDIUM]

Defensive guards exist in several stages but are never triggered by tests:

| Stage | Untested abort condition |
|---|---|
| `s2_ringdown_window` | `fs <= 0`, `duration_s <= 0`, non-1D strain array, NaN/Inf in strain |
| `s3_ringdown_estimates` | All detectors fail estimation (flat noise, no ringdown signal) |
| `s4_geometry_filter` | Atlas entry missing both `f_hz`/`Q` and `phi_atlas`; epsilon threshold edge values |
| `s5_aggregate` | `--source-runs ""` (empty), source run directory missing `compatible_set.json` |
| `s6_information_geometry` | Empty compatible-geometries list; `f_hz <= 0` entries |

These tests can be written as short subprocess invocations or direct function calls with
synthetic bad inputs, and they guard against "garbage-in, garbage-out" scenarios.

---

## Gap 8 — `mvp/path_utils.py` and `mvp/gwtc_events.py`: No direct unit tests [LOW]

- **`path_utils.py`** (33 lines): `resolve_run_scoped_input()` is called by `s4_geometry_filter`
  but never tested directly. Missing: path traversal attempt (`../../etc/passwd`), non-existent
  override path, `None` override (should return default).

- **`gwtc_events.py`**: `test_pipeline_estimator.py` covers basic catalog lookups, but does not
  test `get_event()` with an alias, or the returned dict structure when fields are missing.

---

## Gap 9 — No test coverage measurement infrastructure [LOW]

There is no `pytest-cov`, `.coveragerc`, or CI step reporting coverage. All gaps identified
above are based on code reading; actual branch coverage may be lower than estimated.

**Suggested addition to `requirements.txt`:**
```
pytest-cov>=4.0
```

**Suggested `setup.cfg` or `pyproject.toml` section:**
```ini
[tool:pytest]
addopts = --cov=mvp --cov=basurin_io --cov-report=term-missing:skip-covered --cov-fail-under=70
```

This would enforce a minimum coverage floor and make regressions immediately visible.

---

## Gap 10 — `_create_run_valid()` duplicated across ~15 test files [LOW]

The helper is copy-pasted in the majority of test files. Moving it (and `_create_s3_estimates`,
`_run_stage`, `_assert_stage_contract`) into `tests/conftest.py` as shared fixtures would reduce
duplication and make adding new tests easier.

---

## Recommended Priority Order

| # | Area | Effort | Impact |
|---|---|---|---|
| 1 | `basurin_io.py` unit tests | Low — pure functions, no I/O setup needed | High — foundation of all pipeline I/O |
| 2 | `pipeline.py` orchestrator tests | Medium — requires monkeypatching subprocess | High — controls pipeline correctness and abort semantics |
| 3 | `extract_psd.py` tests | Low — numpy-only, no subprocess | High — only module with zero tests |
| 4 | `bootstrap_ringdown_observables()` | Low — extend existing test file | Medium — third estimator code path |
| 5 | `s4c_kerr_consistency` function tests | Low — pure functions | Medium — censoring logic affects population results |
| 6 | `s2._resolve_t0_gps()` path tests | Low — file-fixture based | Medium — wrong t0 produces wrong ringdown windows |
| 7 | Error/abort paths across stages | Medium — one bad-input test per stage | Medium — defensive guards preventing garbage output |
| 8 | `pytest-cov` setup | Very low — two config lines | Medium — enables ongoing monitoring and CI enforcement |
| 9 | `conftest.py` fixture consolidation | Low — mechanical refactor | Low — maintainability |
| 10 | `path_utils` / `gwtc_events` unit tests | Low | Low — already partially covered |
