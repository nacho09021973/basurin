# BRUNETE / `legaxy` Audit Plan

## Keep-set criterion

`brunete/keep_set.txt` contains the exact tracked files that must remain outside `legaxy/` so BRUNETE keeps working under the current codebase.

Conservative inclusion rule:

- Keep every tracked file under `brunete/`.
- Keep every tracked test that directly exercises `brunete/`.
- Keep every tracked file imported directly or indirectly by code under `brunete/`.
- Keep every tracked file reached at runtime by `brunete/brunete_run_batch.py` through `mvp/pipeline.py`, including subprocess-invoked stages.
- Keep every tracked catalog, atlas, schema, or static asset resolved by default paths from those execution paths.
- Keep installation/runtime basics required by those files: `basurin_io.py`, `requirements.txt`, `setup.cfg`.
- When in doubt, keep.

## Move-set criterion

`brunete/move_to_legaxy_candidates.txt` is the exact complement of `git ls-files` with respect to `brunete/keep_set.txt`, excluding the explicitly deferred ambiguous BRUNETE-named paths.

Excluded ambiguous paths for now:

- `docs/BRUNETE_INCONSISTENCIES.md`
- `docs/BRUNETE_INVENTORY.md`
- `docs/BRUNETE_S6C.md`
- `docs/BRUNETE_TEXT_FIXES.md`
- `docs/estudio_brunete.md`
- `docs/estudio_brunete_completo.md`
- `docs/metodo_brunete.md`
- `mvp/brunete/__init__.py`
- `mvp/brunete/core.py`
- `mvp/s6c_brunete_psd_curvature.py`
- `tests/test_brunete_core.py`
- `tests/test_s6c_brunete_psd_curvature_integration.py`

## Known risks

- `brunete/brunete_run_batch.py` does not only depend on `brunete/`; it calls `mvp/pipeline.py`, which launches multiple `mvp.*` stages by subprocess. Missing one stage file can break execution even if imports still resolve.
- `brunete/brunete_bounded_analysis.py` imports `mvp.s1_fetch_strain`, `mvp.s3_ringdown_estimates`, and `mvp.s3b_multimode_estimates` directly.
- Default runtime paths matter: moving `docs/ringdown/atlas/atlas_berti_v2.json`, `gwtc_events_t0.json`, `gwtc_quality_events.csv`, `mvp/assets/*.json`, or `docs/ringdown/event_metadata/**` can break offline execution without changing imports.
- `mvp/s3_spectral_estimates.py` and `mvp/experiment_dual_method.py` are part of the `dual` estimator path. The first is in the keep-set because it is runtime-relevant; the second is not fatal today because the pipeline treats it as best-effort.
- `AGENTS.md` states a different canonical repo root than the actual git root returned by `git rev-parse --show-toplevel`. Path-sensitive validation should use the real git root before any movement.
- The 12 ambiguous BRUNETE-named files are intentionally deferred. They may be safe to move later, but they are not included in the current move list.

## Post-move validation commands

Run these after any future move to `legaxy/`:

```bash
git rev-parse --show-toplevel
python - <<'PY'
import brunete.brunete_prepare_events
import brunete.brunete_list_events
import brunete.brunete_audit_cohort_authority
import brunete.brunete_run_batch
import brunete.brunete_classify_geometries
import brunete.brunete_bounded_analysis
import brunete.experiment.b5a
import brunete.experiment.b5b
import brunete.experiment.b5c
import brunete.experiment.b5e
import brunete.experiment.b5f
import brunete.experiment.b5h
import brunete.experiment.b5z
PY
pytest tests/brunete tests/test_brunete_experiment.py -q
BASURIN_RUNS_ROOT="$(mktemp -d)" pytest tests/brunete/test_prepare_events.py tests/brunete/test_run_batch.py tests/brunete/test_classify_geometries.py tests/test_brunete_experiment.py -q
python -m brunete.brunete_prepare_events --help
python -m brunete.brunete_run_batch --help
python -m brunete.brunete_classify_geometries --help
python -m brunete.brunete_bounded_analysis --help
```
