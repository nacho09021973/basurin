# Auditoría de Fragmentación Estructural — BASURIN

**Fecha:** 2026-03-08
**Auditor:** Revisión arquitectónica automatizada (contract-first)
**Commit base:** `main` (HEAD al momento de la auditoría)
**Alcance:** Repositorio completo — mvp/, malda/, tools/, tests/, raíz

---

# 1. Resumen ejecutivo

**Diagnóstico neto: TENSIONADO — con focos de fragmentación controlable.**

El repositorio NO está en estado crítico de fragmentación incontrolada, pero presenta tensión estructural en tres ejes:

1. **Proliferación de entrypoints CLI:** 47 `def main()` con `if __name__` en mvp/ + malda/, de los cuales ~12 son variantes o wrappers sobre la misma responsabilidad funcional.
2. **Familia s4 con 8 variantes** que cubren dos workflows distintos (MVP legacy y golden geometries) sin interfaz unificada ni flag de dispatch.
3. **Familia t0_sweep con 3 implementaciones** (`experiment_t0_sweep.py`, `experiment_t0_sweep_full.py`, `experiment_t0_sweep_221.py`) donde la primera está declarada como DEV pero coexiste con la canónica.

**Señal principal de riesgo:** La superficie de mantenimiento crece linealmente con cada variante, pero el contrato centralizado (`contracts.py`) no cubre uniformemente todas las variantes — 5 stages en mvp/ no usan `init_stage/finalize` (s4g, s4h, s4i, s4j, s5_event_row), creando una brecha de gobernanza.

**Magnitud estimada:**
- **211 ficheros .py** en total (424 ficheros totales).
- **~97 ficheros de test** (46% del código Python son tests — ratio saludable pero con redundancia interna).
- **~20 scripts de experimento** en mvp/ — 9 sin integración contractual completa.
- **19 stages malda/** — pipeline independiente, autocontenido, sin conflicto con mvp/.
- **Estimación de poda segura:** 8-12 scripts pueden consolidarse, deprecarse o reubicarse sin romper el pipeline canónico.

---

# 2. Inventario estructural

| Categoría | Nº scripts | Rutas representativas | Observaciones |
|-----------|-----------|----------------------|---------------|
| **canonical_stage (MVP core)** | 16 | `mvp/s0_oracle_mvp.py`, `mvp/s1_fetch_strain.py`, `mvp/s2_ringdown_window.py`, `mvp/s3_ringdown_estimates.py`, `mvp/s3b_multimode_estimates.py`, `mvp/s4_geometry_filter.py`, `mvp/s4d_kerr_from_multimode.py`, `mvp/s5_aggregate.py`, `mvp/s6_information_geometry.py`, `mvp/s7_beyond_kerr_deviation_score.py` | Todos con contrato en `contracts.py`; todos con `init_stage/finalize` |
| **canonical_stage (golden geo)** | 4 | `mvp/s4g_mode220_geometry_filter.py`, `mvp/s4h_mode221_geometry_filter.py`, `mvp/s4i_common_geometry_intersection.py`, `mvp/s4j_hawking_area_filter.py` | Workflow paralelo; NO usan `init_stage/finalize`; usan `basurin_io` directo |
| **canonical_stage (secundario)** | 7 | `mvp/s3_spectral_estimates.py`, `mvp/s4b_spectral_curvature.py`, `mvp/s4c_kerr_consistency.py`, `mvp/s6b_information_geometry_3d.py`, `mvp/s6b_information_geometry_ranked.py`, `mvp/s6c_brunete_psd_curvature.py`, `mvp/s6c_population_geometry.py` | Stages complementarios; la mayoría con contratos |
| **experiment (canonical)** | 11 | `mvp/experiment_area_theorem_deltaA.py`, `mvp/experiment_eps_sweep.py`, `mvp/experiment_ex3_golden_sweep.py`, `mvp/experiment_ex4_spectral_exclusion_map.py`, `mvp/experiment_ex8_area_consistency.py`, `mvp/experiment_t0_sweep_full.py`, `mvp/experiment/ex2_ranking.py`, `mvp/experiment/geometry_evidence_vs_gr.py` | Con integración contractual completa |
| **experiment (no-canonical)** | 9 | `mvp/experiment_t0_sweep.py`, `mvp/experiment_t0_sweep_221.py`, `mvp/experiment_dual_method.py`, `mvp/experiment_injection_suite.py`, `mvp/experiment_oracle_t0_ringdown.py`, `mvp/experiment_population_kerr.py`, `mvp/experiment_single_event_golden_robustness.py`, `mvp/experiment_t6_rd_weighted.py`, `mvp/experiment/legacy_top_level/analyze_geometry_support.py` | Contratos parciales o ausentes |
| **helper/library** | 12 | `mvp/contracts.py`, `mvp/schemas.py`, `mvp/distance_metrics.py`, `mvp/kerr_qnm_fits.py`, `mvp/golden_geometry_spec.py`, `mvp/path_utils.py`, `mvp/oracle_t0_math.py`, `mvp/oracle_t0_selection.py`, `mvp/multimode_viability.py`, `mvp/preflight_viability.py`, `mvp/gwtc_events.py`, `mvp/brunete/core.py` | Librerías internas; sin CLI; importadas por stages |
| **cli_entrypoint (orquestación)** | 2 | `mvp/pipeline.py`, `tools/run_experiment_batch.py` | Orquestadores de pipeline y batch |
| **cli_entrypoint (utilidad)** | 5 | `mvp/extract_psd.py`, `mvp/find_artifact.py`, `mvp/generate_atlas_from_fits.py`, `mvp/s5_event_row.py`, `mvp/s6_multi_event_table.py` | Herramientas auxiliares con CLI |
| **tools (data fetch)** | 5 | `tools/fetch_catalog_events.py`, `tools/fetch_losc_event.py`, `tools/fetch_losc_batch.sh`, `tools/gwosc_sync_losc.py`, `tools/losc_precheck.py` | Descarga y verificación de datos GWOSC |
| **tools (CI)** | 1 | `tools/ci/check_theory_doc.py` | Linter de documentación teórica |
| **malda pipeline** | 20 | `malda/00_load_ligo_data.py` ... `malda/09_real_data_and_dictionary_contracts.py`, `malda/stage_utils.py` | Pipeline holográfico independiente; ~17k LOC; autocontenido |
| **test** | 97 | `tests/test_*.py`, `tests/unit/test_*.py`, `tests/integration/test_*.py` | ~23k LOC; ratio tests/source ~0.86 |
| **config/data** | 7 | `config/external_oracle_gw150914.yaml`, `gwtc_events_t0.json`, `gwtc_quality_events.csv`, atlas JSONs, schema JSON | Datos de referencia y configuración |
| **deprecated_candidate** | 3 | `mvp/experiment_t0_sweep.py`, `mvp/s6_geometry_table.py`, `mvp/experiment/legacy_top_level/analyze_geometry_support.py` | Ver hallazgos H-01, H-06, H-07 |
| **IO framework** | 2 | `basurin_io.py`, `conftest.py` | Framework IO determinista y configuración pytest |
| **scratch** | 1 | `scratch/events_10.txt` | Lista de eventos para pruebas rápidas |
| **download (raíz)** | 1 | `download_gw_events.py` | Script de descarga top-level; posiblemente redundante con tools/ |

---

# 3. Hallazgos trazables

## H-01: Familia t0_sweep con 3 implementaciones
- **Tipo:** duplicación funcional / entrypoint redundante
- **Rutas afectadas:**
  - `mvp/experiment_t0_sweep.py` (386 LOC) — marcado "DEV/INTEGRATION"
  - `mvp/experiment_t0_sweep_full.py` (1918 LOC) — canónico, con subruns aislados
  - `mvp/experiment_t0_sweep_221.py` (411 LOC) — modo 221 específico
- **Evidencia:** `experiment_t0_sweep.py` contiene banner explícito "DEV/INTEGRATION" y no usa `init_stage`. Su funcionalidad es un subconjunto de `experiment_t0_sweep_full.py`. `experiment_t0_sweep_221.py` cubre un caso de uso específico (modo 221) que podría parametrizarse dentro de `_full`.
- **Impacto:** 3 entrypoints que confunden sobre cuál usar; mantenimiento triple para cambios en lógica de sweep.
- **Severidad:** **alta**

## H-02: Brecha de gobernanza contractual en golden geometry pipeline
- **Tipo:** validación dispersa / inconsistencia contractual
- **Rutas afectadas:**
  - `mvp/s4g_mode220_geometry_filter.py` — NO usa `init_stage/finalize`
  - `mvp/s4h_mode221_geometry_filter.py` — NO usa `init_stage/finalize`
  - `mvp/s4i_common_geometry_intersection.py` — NO usa `init_stage/finalize`
  - `mvp/s4j_hawking_area_filter.py` — NO usa `init_stage/finalize`
- **Evidencia:** Estos 4 stages usan `basurin_io` directamente (resolve_out_root, write_stage_summary, write_manifest) sin pasar por el patrón centralizado de `contracts.py`. Sin embargo, s4g y s4h NO están declarados en `CONTRACTS` dict de `contracts.py` (solo `s4g_mode220_geometry_filter` aparece, pero s4h/s4i/s4j faltan).
- **Impacto:** Los tests de completitud contractual (`test_contracts_completeness.py`) no cubren estos stages. Las funciones `check_inputs`, `abort` con tracing y `enforce_outputs` no se aplican.
- **Severidad:** **alta**

## H-03: Duplicación semántica en tests de compatible_set schema
- **Tipo:** test redundante
- **Rutas afectadas:**
  - `tests/test_schema_compatible_set.py` — tests de `validate_compatible_set` (mvp.schemas)
  - `tests/test_compatible_set_schema.py` — tests de `validate` + `extract_compatible_geometry_ids` (mvp.schemas)
- **Evidencia:** Ambos ficheros prueban funciones del mismo módulo (`mvp/schemas.py`) sobre el mismo concepto (compatible_set schema). El primero usa `validate_compatible_set` (función legacy); el segundo usa `validate` (función canónica). Son dos APIs del mismo módulo sobre el mismo payload type.
- **Impacto:** Mantenimiento doble; confusión sobre cuál es la API canónica.
- **Severidad:** **media**

## H-04: Tests fragmentados para t0_sweep_full (5 ficheros)
- **Tipo:** test redundante / violación regla de consolidación AGENTS.md
- **Rutas afectadas:**
  - `tests/test_experiment_t0_sweep_full_diagnose_unittest.py`
  - `tests/test_experiment_t0_sweep_full_finalize_golden.py`
  - `tests/test_experiment_t0_sweep_full_inventory_unittest.py`
  - `tests/test_experiment_t0_sweep_full_oracle_from_results.py`
  - `tests/test_experiment_t0_sweep_full_paths.py`
  - `tests/integration/test_experiment_t0_sweep_full_integration_unittest.py`
- **Evidencia:** AGENTS.md §Regla de consolidación indica "1 fichero por experimento". Para `experiment_t0_sweep_full.py` hay 6 ficheros de test (5 unit + 1 integration). Juntos suman ~2400 LOC.
- **Impacto:** Superficie de test inflada; los cambios en t0_sweep_full requieren revisar 6 ficheros. Potenciales assertions duplicadas entre ficheros.
- **Severidad:** **media**

## H-05: Tests fragmentados para pipeline (5 ficheros)
- **Tipo:** test redundante / violación regla de consolidación AGENTS.md
- **Rutas afectadas:**
  - `tests/test_pipeline_orchestrator.py`
  - `tests/test_pipeline_cli_local_hdf5.py`
  - `tests/test_pipeline_estimator.py`
  - `tests/test_pipeline_guardrails.py`
  - `tests/test_pipeline_calls_s6_information_geometry.py`
  - `tests/test_pipeline_s1_local_hdf5_autodetect.py`
- **Evidencia:** AGENTS.md §Regla de consolidación indica "Cambios parser/CLI → 1 fichero (usar/ampliar `tests/test_pipeline_cli_*.py`)". Sin embargo, existen 6 ficheros de test distintos para `mvp/pipeline.py`.
- **Impacto:** Mismo que H-04.
- **Severidad:** **media**

## H-06: `mvp/s6_geometry_table.py` — utilidad huérfana
- **Tipo:** obsolescencia potencial
- **Rutas afectadas:**
  - `mvp/s6_geometry_table.py` (374 LOC)
- **Evidencia:** Este script extrae tablas TSV/JSONL desde `s3b_multimode_estimates/stage_summary.json`, pero NO está declarado en `CONTRACTS` dict. Su funcionalidad de tabulación se solapa parcialmente con `mvp/s6_multi_event_table.py` (que tabula desde s5_event_row) y con la funcionalidad de `mvp/s5_aggregate.py` (que consume los mismos inputs). No tiene stage_summary ni manifest propio bajo el patrón contractual. El test correspondiente (`tests/test_geometry_table_unittest.py`) es ligero.
- **Impacto:** Script fuera del flujo contractual; confusión con `s6_multi_event_table.py` (nombre similar, responsabilidad diferente pero solapada).
- **Severidad:** **media**

## H-07: `mvp/experiment/legacy_top_level/analyze_geometry_support.py` — legacy activo
- **Tipo:** experimento filtrado al core / obsolescencia
- **Rutas afectadas:**
  - `mvp/experiment/legacy_top_level/analyze_geometry_support.py` (426 LOC)
  - `tests/test_experiment_geometry_support_unittest.py`
  - `tests/integration/test_experiment_geometry_support_smoke.py`
- **Evidencia:** Ubicado en `legacy_top_level/` indicando intención de deprecación, pero tiene 2 ficheros de test activos. Sin integración contractual. Su funcionalidad de análisis de soporte de geometría está parcialmente cubierta por `mvp/experiment/geometry_evidence_vs_gr.py` (evidencia vs GR) y `mvp/experiment/ex2_ranking.py` (ranking por potencia discriminativa).
- **Impacto:** Mantenimiento de código legacy con tests activos; confusión sobre si es canónico.
- **Severidad:** **baja**

## H-08: `download_gw_events.py` en raíz — redundancia con tools/
- **Tipo:** entrypoint redundante / dispersión
- **Rutas afectadas:**
  - `download_gw_events.py` (raíz, 17413 LOC — fichero grande)
  - `tools/fetch_catalog_events.py` (14575 LOC)
  - `tools/fetch_losc_event.py` (8099 LOC)
  - `tools/gwosc_sync_losc.py` (6587 LOC)
- **Evidencia:** `download_gw_events.py` en raíz realiza descarga de eventos GWTC. `tools/fetch_catalog_events.py` y `tools/fetch_losc_event.py` hacen funcionalidad similar. La ubicación en raíz rompe la convención de que los scripts de herramientas vivan en `tools/`.
- **Impacto:** Confusión sobre cuál usar; script grande en raíz contaminando el namespace.
- **Severidad:** **media**

## H-09: `mvp/s4c_kerr_consistency.py` — aggregator trivial sin computación nueva
- **Tipo:** entrypoint que podría ser lógica de orquestación
- **Rutas afectadas:**
  - `mvp/s4c_kerr_consistency.py` (168 LOC)
- **Evidencia:** Este stage NO realiza computación original. Lee outputs de s3b y s4, y emite un veredicto booleano `kerr_consistent = bool(n_compatible > 0)`. Podría ser una función en `pipeline.py` o un step dentro de `s4d_kerr_from_multimode.py`.
- **Impacto:** Un stage completo (con directorio, manifest, stage_summary) para una operación condicional de 3 líneas. Incrementa la cuenta de stages sin añadir capacidad analítica.
- **Severidad:** **baja** (funcional, pero inflacionario)

## H-10: Tests de contratos dispersos (4 ficheros)
- **Tipo:** test redundante
- **Rutas afectadas:**
  - `tests/test_mvp_contracts.py`
  - `tests/test_contracts_completeness.py`
  - `tests/test_contracts_fase4.py`
  - `tests/test_contracts_fase5.py`
  - `tests/test_contract_runtime_alignment.py`
  - `tests/test_dynamic_contracts.py`
- **Evidencia:** 6 ficheros de test para `mvp/contracts.py` (un solo módulo). La regla AGENTS.md permite "1 fichero por stage" para contratos. Estos ficheros podrían consolidarse en 2: `test_contracts_core.py` (completitud + runtime) y `test_contracts_phase.py` (fases 4-5).
- **Impacto:** Fragmentación de tests; potencial duplicación de fixtures y helpers.
- **Severidad:** **baja**

## H-11: Tests de oracle fragmentados (6 ficheros)
- **Tipo:** test redundante
- **Rutas afectadas:**
  - `tests/test_oracle_mvp_governance.py`
  - `tests/test_oracle_t0_input_schema.py`
  - `tests/test_oracle_t0_math.py`
  - `tests/test_oracle_t0_selection.py`
  - `tests/test_oracle_t0_v1_2_golden.py`
  - `tests/test_oracle_v1_gate_reasons.py`
  - `tests/test_oracle_v1_plateau_logic.py`
  - `tests/test_oracle_v1_plateau_parsing.py`
- **Evidencia:** 8 ficheros de test para el subsistema oracle (3-4 módulos fuente). Los módulos fuente son `mvp/oracle_t0_math.py`, `mvp/oracle_t0_selection.py`, `mvp/oracles/oracle_v1_plateau.py`, `mvp/oracles/t0_input_schema.py`. La ratio 8:4 (tests:fuentes) indica fragmentación excesiva, especialmente `test_oracle_v1_plateau_logic.py` vs `test_oracle_v1_plateau_parsing.py` que prueban el mismo módulo.
- **Severidad:** **baja**

## H-12: `mvp/experiment_dual_method.py` — gate masquerading de experimento
- **Tipo:** experimento filtrado al core
- **Rutas afectadas:**
  - `mvp/experiment_dual_method.py` (268 LOC)
- **Evidencia:** Este script compara Hilbert vs Lorentzian y emite un veredicto CONSISTENT/TENSION/INCONSISTENT. Usa manifest custom (no `init_stage`), y lanza subproceso `s3_spectral_estimates` si falta. Funcionalmente es un **gate de calidad**, no un experimento. Su lógica de decisión es operacional (recomienda método preferido), no exploratoria.
- **Impacto:** Clasificación incorrecta — debería ser un stage de QC o una función de `pipeline.py`.
- **Severidad:** **baja**

## H-13: Tests de s5 fragmentados (6 ficheros)
- **Tipo:** test redundante
- **Rutas afectadas:**
  - `tests/test_s5_aggregate_extractors.py`
  - `tests/test_s5_aggregate_ranked_integration.py`
  - `tests/test_s5_event_row.py`
  - `tests/test_s5_joint_posterior.py`
  - `tests/test_s5_paths_relative_unittest.py`
  - `tests/test_s5_topk_affects_coverage_unittest.py`
- **Evidencia:** 6 ficheros para 2 módulos fuente (`s5_aggregate.py`, `s5_event_row.py`). Consolidables en 2 ficheros.
- **Severidad:** **baja**

---

# 4. Core canónico propuesto

## Scripts soberanos del pipeline end-to-end (CONSERVAR INTACTOS)

**Pipeline MVP core (single-event):**
1. `mvp/pipeline.py` — orquestador
2. `mvp/s0_oracle_mvp.py` — oracle pre-pipeline
3. `mvp/s1_fetch_strain.py` — descarga/generación de strain
4. `mvp/s2_ringdown_window.py` — ventana ringdown
5. `mvp/s3_ringdown_estimates.py` — estimaciones (f, τ, Q)
6. `mvp/s3b_multimode_estimates.py` — estimaciones multimode (220, 221)
7. `mvp/s4_geometry_filter.py` — filtro de geometría (atlas)
8. `mvp/s4d_kerr_from_multimode.py` — inversión Kerr desde multimode
9. `mvp/s5_aggregate.py` — agregación multi-evento
10. `mvp/s6_information_geometry.py` — métrica conformal y curvatura
11. `mvp/s7_beyond_kerr_deviation_score.py` — score beyond-Kerr

**Pipeline golden geometries:**
12. `mvp/s4g_mode220_geometry_filter.py` — filtro modo 220
13. `mvp/s4h_mode221_geometry_filter.py` — filtro modo 221
14. `mvp/s4i_common_geometry_intersection.py` — intersección
15. `mvp/s4j_hawking_area_filter.py` — filtro área Hawking

**Stages complementarios (conservar):**
16. `mvp/s3_spectral_estimates.py` — estimaciones espectrales (método alternativo)
17. `mvp/s4b_spectral_curvature.py` — curvatura espectral (diagnóstico ortogonal)
18. `mvp/s6b_information_geometry_3d.py` — geometría 3D con censura
19. `mvp/s6b_information_geometry_ranked.py` — export compacto para agregación
20. `mvp/s6c_brunete_psd_curvature.py` — curvatura PSD (BRUNETE)
21. `mvp/s6c_population_geometry.py` — geometría poblacional

**Librerías internas (conservar):**
22. `mvp/contracts.py` — SSOT contractual
23. `mvp/schemas.py` — validación de schemas
24. `mvp/distance_metrics.py` — métricas de distancia
25. `mvp/kerr_qnm_fits.py` — fits QNM Kerr
26. `mvp/golden_geometry_spec.py` — spec golden geometries
27. `mvp/path_utils.py` — resolución de rutas
28. `mvp/oracle_t0_math.py` — matemática oracle t0
29. `mvp/oracle_t0_selection.py` — selección oracle t0
30. `mvp/oracles/oracle_v1_plateau.py` — algoritmo plateau oracle v1
31. `mvp/oracles/t0_input_schema.py` — schema input oracle
32. `mvp/brunete/core.py` — core BRUNETE
33. `mvp/preflight_viability.py` — viabilidad preflight
34. `mvp/gwtc_events.py` — catálogo GWTC
35. `basurin_io.py` — framework IO determinista

**Experimentos canónicos (conservar):**
36. `mvp/experiment_t0_sweep_full.py` — sweep t0 canónico
37. `mvp/experiment_eps_sweep.py` — sweep epsilon
38. `mvp/experiment_ex3_golden_sweep.py` — sweep golden events
39. `mvp/experiment_ex4_spectral_exclusion_map.py` — mapa exclusión espectral
40. `mvp/experiment_ex8_area_consistency.py` — consistencia de área
41. `mvp/experiment_area_theorem_deltaA.py` — teorema de área
42. `mvp/experiment_gwtc_posteriors_fetch.py` — fetch posteriors
43. `mvp/experiment_offline_batch.py` — batch offline
44. `mvp/experiment/ex2_ranking.py` — ranking de eventos
45. `mvp/experiment/geometry_evidence_vs_gr.py` — evidencia vs GR

## Scripts que deberían pasar a deprecated_candidate

| Script | Motivo | Sustituto |
|--------|--------|-----------|
| `mvp/experiment_t0_sweep.py` | Marcado "DEV/INTEGRATION"; subconjunto funcional de `_full.py` | `mvp/experiment_t0_sweep_full.py` |
| `mvp/experiment/legacy_top_level/analyze_geometry_support.py` | Autodeclarado legacy; cubierto por `ex2_ranking.py` + `geometry_evidence_vs_gr.py` | `mvp/experiment/ex2_ranking.py` |
| `mvp/s6_geometry_table.py` | No contractual; tabulación redundante con `s6_multi_event_table.py` + `s5_aggregate.py` | `mvp/s6_multi_event_table.py` |

## Scripts que deberían vivir solo en experiment/

| Script | Motivo |
|--------|--------|
| `mvp/experiment_single_event_golden_robustness.py` | Sin contrato; generación de escenarios in-memory; no produce stage canónico |
| `mvp/experiment_population_kerr.py` | Contrato minimal; 3 fases custom; spawning de s7 |
| `mvp/experiment_t0_sweep_221.py` | Modo 221 específico; caso candidato a parametrizar en `_full.py --mode 221` |
| `mvp/experiment_t6_rd_weighted.py` | Contrato minimal; complejo multi-source join; candidato a absorción en `experiment_area_theorem_deltaA.py` |
| `mvp/experiment_injection_suite.py` | Standalone; sin contrato; herramienta de validación, no pipeline |
| `mvp/experiment_oracle_t0_ringdown.py` | Post-procesador de `_full`; contrato parcial |

## Scripts que deberían absorberse como librería interna

| Script | Destino propuesto | Motivo |
|--------|-------------------|--------|
| `mvp/s4c_kerr_consistency.py` | Función en `mvp/contracts.py` o `mvp/s4d_kerr_from_multimode.py` | 168 LOC de lógica condicional sin computación original |
| `mvp/s5_event_row.py` | Función en `mvp/s5_aggregate.py` o `mvp/s6_multi_event_table.py` | 276 LOC; tracing de punto único; sin contrato |
| `mvp/experiment_dual_method.py` | Gate en `mvp/pipeline.py` o función en `mvp/s3_ringdown_estimates.py` | Gate de QC, no experimento |
| `mvp/find_artifact.py` | Función en `basurin_io.py` | Utilidad de búsqueda de artefactos |

---

# 5. Plan mínimo de consolidación

## Fase 1: Deprecar t0_sweep DEV (riesgo: bajo)
- **Objetivo:** Eliminar entrypoint confuso `experiment_t0_sweep.py`.
- **Scripts afectados:** `mvp/experiment_t0_sweep.py`, `tests/integration/test_experiment_t0_sweep_integration.py`
- **Cambio mínimo:**
  1. Añadir banner `DEPRECATED` con fecha y redirect a `experiment_t0_sweep_full.py`.
  2. Mover a `mvp/experiment/deprecated/experiment_t0_sweep.py`.
  3. Actualizar import en test de integración.
- **Riesgo:** Bajo — ya marcado como DEV/INTEGRATION.
- **Test de regresión:** Ejecutar `tests/integration/test_experiment_t0_sweep_full_integration_unittest.py` para confirmar que `_full` cubre los mismos escenarios.

## Fase 2: Integrar golden geometry pipeline en contracts.py (riesgo: medio)
- **Objetivo:** Cerrar brecha de gobernanza H-02.
- **Scripts afectados:** `mvp/s4g_mode220_geometry_filter.py`, `mvp/s4h_mode221_geometry_filter.py`, `mvp/s4i_common_geometry_intersection.py`, `mvp/s4j_hawking_area_filter.py`, `mvp/contracts.py`
- **Cambio mínimo:**
  1. Añadir contratos `s4h_mode221_geometry_filter`, `s4i_common_geometry_intersection`, `s4j_hawking_area_filter` al dict `CONTRACTS` de `contracts.py`.
  2. Migrar los 4 scripts a usar `init_stage/check_inputs/finalize` en lugar de `basurin_io` directo.
  3. Mantener la lógica de negocio intacta.
- **Riesgo:** Medio — requiere verificar que `init_stage` no rompa la semántica de "obs_path opcional" en s4h (graceful skip).
- **Test de regresión:** Ejecutar `tests/test_s4g_mode220_geometry_filter.py`, tests existentes de golden geometries, y verificar que los outputs son bit-a-bit idénticos.

## Fase 3: Consolidar tests de t0_sweep_full (riesgo: bajo)
- **Objetivo:** Reducir de 5 a 2 ficheros de test unit para t0_sweep_full.
- **Scripts afectados:** Los 5 ficheros `tests/test_experiment_t0_sweep_full_*.py`
- **Cambio mínimo:**
  1. Consolidar `_diagnose_unittest.py`, `_finalize_golden.py`, `_oracle_from_results.py` en un solo `tests/test_experiment_t0_sweep_full_unit.py`.
  2. Mantener `_inventory_unittest.py` y `_paths.py` si cubren aspectos ortogonales, o absorber también.
  3. No cambiar assertions; solo mover funciones de test.
- **Riesgo:** Bajo — refactoring puramente organizacional.
- **Test de regresión:** `pytest tests/test_experiment_t0_sweep_full_*.py` antes y después; misma cuenta de tests PASS.

## Fase 4: Consolidar tests de pipeline (riesgo: bajo)
- **Objetivo:** Reducir de 6 a 2 ficheros de test para `pipeline.py`.
- **Scripts afectados:** Los 6 ficheros `tests/test_pipeline_*.py`
- **Cambio mínimo:** Consolidar en `tests/test_pipeline_orchestrator.py` y `tests/test_pipeline_cli.py`.
- **Riesgo:** Bajo.
- **Test de regresión:** Misma cuenta de tests PASS.

## Fase 5: Mover `download_gw_events.py` a tools/ (riesgo: bajo)
- **Objetivo:** Limpiar raíz del repo.
- **Scripts afectados:** `download_gw_events.py` → `tools/download_gw_events.py`
- **Cambio mínimo:** `git mv` + actualizar referencias.
- **Riesgo:** Bajo — verificar que no haya paths hardcodeados.
- **Test de regresión:** Grep por `download_gw_events` en todo el repo.

## Fase 6: Mover `analyze_geometry_support.py` a deprecated (riesgo: bajo)
- **Objetivo:** Aclarar status legacy.
- **Scripts afectados:** `mvp/experiment/legacy_top_level/analyze_geometry_support.py`
- **Cambio mínimo:** Añadir DEPRECATED header; mover tests a `tests/deprecated/`.
- **Riesgo:** Bajo.
- **Test de regresión:** Ninguno — legacy.

## Fase 7: Fusionar tests de compatible_set schema (riesgo: bajo)
- **Objetivo:** Eliminar duplicación H-03.
- **Scripts afectados:** `tests/test_schema_compatible_set.py`, `tests/test_compatible_set_schema.py`
- **Cambio mínimo:** Consolidar en `tests/test_compatible_set_schema.py` (nombre más canónico). Mover tests de `validate_compatible_set` al mismo fichero.
- **Riesgo:** Bajo.
- **Test de regresión:** Misma cuenta de tests PASS.

---

# 6. Top 10 scripts a revisar primero

| Prioridad | Ruta | Motivo | Acción recomendada |
|-----------|------|--------|-------------------|
| 1 | `mvp/experiment_t0_sweep.py` | DEV/INTEGRATION sin contrato; supersedido por `_full` | **Deprecar** y mover a experiment/deprecated/ |
| 2 | `mvp/s4g_mode220_geometry_filter.py` | No usa `init_stage/finalize`; brecha contractual | **Migrar** a patrón contractual centralizado |
| 3 | `mvp/s4h_mode221_geometry_filter.py` | Idem H-02 | **Migrar** a patrón contractual centralizado |
| 4 | `mvp/s4i_common_geometry_intersection.py` | Idem H-02 | **Migrar** a patrón contractual centralizado |
| 5 | `mvp/s4j_hawking_area_filter.py` | Idem H-02 | **Migrar** a patrón contractual centralizado |
| 6 | `mvp/s6_geometry_table.py` | Utilidad huérfana; no contractual; solapada | **Deprecar** o documentar como utility no-stage |
| 7 | `download_gw_events.py` | En raíz; redundancia con tools/ | **Mover** a tools/ |
| 8 | `mvp/experiment/legacy_top_level/analyze_geometry_support.py` | Legacy con tests activos | **Deprecar** formalmente |
| 9 | `tests/test_schema_compatible_set.py` + `tests/test_compatible_set_schema.py` | Duplicación semántica | **Fusionar** en un solo fichero |
| 10 | `mvp/s4c_kerr_consistency.py` | Aggregator trivial (168 LOC); candidato a función | **Evaluar** absorción en s4d o pipeline |

---

# 7. Criterio de aceptación

La consolidación se considerará exitosa cuando se verifiquen las siguientes métricas observables:

| Métrica | Estado actual | Objetivo post-consolidación |
|---------|--------------|----------------------------|
| Entrypoints `def main()` en mvp/ | 47 | ≤ 42 (reducción de ~5 deprecated/absorbidos) |
| Scripts sin integración contractual (`init_stage/finalize`) en mvp/ stages | 5 (s4g, s4h, s4i, s4j, s5_event_row) | ≤ 1 (solo s5_event_row si se mantiene como utility) |
| Stages declarados en `CONTRACTS` dict vs stages reales | Faltan s4h, s4i, s4j | 100% cobertura |
| Ficheros de test por módulo (máximo) | 6 (t0_sweep_full), 6 (pipeline) | ≤ 3 por módulo |
| Scripts marcados DEV/INTEGRATION en mvp/ | 1 (experiment_t0_sweep.py) | 0 |
| Scripts en `legacy_top_level/` con tests activos | 1 | 0 (deprecar o mover tests) |
| Scripts en raíz que deberían estar en tools/ | 1 (download_gw_events.py) | 0 |
| Pipeline canónico end-to-end PASS | PASS | PASS (sin regresión) |
| Todos los tests existentes PASS | PASS | PASS (misma cuenta, reorganizados) |
| Contracts completeness audit clean | Gaps en s4h/s4i/s4j | Clean |

---

# 8. NO HACER

1. **NO reescribir el pipeline MVP core.** Los stages s0-s7 están bien diseñados, con contratos centralizados y IO determinista. El problema es periférico, no nuclear.

2. **NO fusionar s4g/s4h en un único "mode-parameterized filter".** Aunque tienen código similar, son stages con outputs independientes bajo directorios distintos (`s4g_*/outputs/`, `s4h_*/outputs/`). Fusionarlos rompería la convención de stage_dir y requeriría reestructurar `s4i_common_geometry_intersection.py`.

3. **NO eliminar el pipeline malda/.** Es un subsistema independiente con su propia lógica (holográfico), su propio `stage_utils.py`, y ~17k LOC. No está en conflicto con mvp/. No genera fragmentación — es un pipeline paralelo con propósito distinto.

4. **NO eliminar los 20 scripts de experimento.** Los experimentos canónicos (con contrato) son parte del flujo de validación científica. Solo deprecar los marcados como DEV o los que han sido supersedidos.

5. **NO consolidar tests de forma agresiva.** La ratio ~0.86 tests/source es saludable. Solo consolidar donde hay fragmentación clara (>3 ficheros por módulo) sin reducir cobertura.

6. **NO tocar los atlas JSON** (`docs/ringdown/atlas/*.json`). Son datos de referencia inmutables con hashes en tests golden.

7. **NO mover los event_metadata JSON** (`docs/ringdown/event_metadata/*.json`). Son datos de referencia usados por el pipeline.

8. **NO eliminar `scratch/events_10.txt`.** Es un fichero de conveniencia para pruebas rápidas; cuesta 90 bytes.

9. **NO refactorizar `mvp/contracts.py` para soportar dos "flavors" de contract** (MVP vs golden). Mejor migrar golden geometry pipeline al mismo patrón.

10. **NO hacer cambios que requieran recalcular golden tests** (`tests/golden/`) salvo que sea consecuencia directa de un fix. Los golden snapshots son la última línea de defensa contra regresiones silenciosas.

---

# Apéndice: Conteo detallado

## Ficheros Python por directorio

| Directorio | Ficheros .py | LOC total |
|------------|-------------|-----------|
| `mvp/` (raíz) | 48 | ~26,700 |
| `mvp/experiment/` | 3 | ~900 |
| `mvp/oracles/` | 2 | ~350 |
| `mvp/brunete/` | 2 | ~200 |
| `mvp/tools/` | 1 | ~250 |
| `mvp/schemas/` | 1 | ~50 |
| `malda/` | 20 | ~16,850 |
| `tests/` (raíz) | 87 | ~20,500 |
| `tests/unit/` | 2 | ~300 |
| `tests/integration/` | 5 | ~700 |
| `tools/` | 5 | ~3,700 |
| raíz | 3 | ~25,500 |
| **Total** | **211** | **~96,000** |

## Entrypoints CLI (`def main()` + `if __name__`)

- **mvp/ stages:** 28 entrypoints
- **mvp/ experiments:** 17 entrypoints
- **mvp/ utilities:** 2 entrypoints
- **malda/:** 19 entrypoints
- **tools/:** 5 entrypoints (+1 .sh)
- **raíz:** 1 entrypoint
- **Total:** 72 entrypoints CLI
