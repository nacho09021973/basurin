# Plan de remediación ejecutable (post-audit) — BASURIN

Fecha: 2026-03-09
Alcance: validación en repositorio real + plan mínimo seguro (contract-first, sin reescritura masiva)

## 1. Verificación del audit previo

| ID | veredicto | rutas exactas | evidencia breve |
|---|---|---|---|
| H-01 | parcialmente confirmado | `mvp/experiment_t0_sweep.py`, `mvp/experiment_t0_sweep_full.py`, `mvp/experiment_t0_sweep_221.py`, `mvp/pipeline.py` | Siguen existiendo 3 entrypoints, pero `experiment_t0_sweep.py` ya es wrapper explícitamente deprecado que delega a `_full`; `pipeline.py` aún invoca ese wrapper en modo best-effort. |
| H-02 | corregido (audit desactualizado) | `mvp/s4g_mode220_geometry_filter.py`, `mvp/s4h_mode221_geometry_filter.py`, `mvp/s4i_common_geometry_intersection.py`, `mvp/s4j_hawking_area_filter.py`, `mvp/contracts.py` | Los 4 stages ya usan `init_stage/check_inputs/finalize/abort/log_stage_paths` y están registrados en `CONTRACTS` con `check_run_valid=True`. |
| H-03 | confirmado | `tests/test_schema_compatible_set.py`, `tests/test_compatible_set_schema.py`, `mvp/schemas.py` | Doble superficie de tests para el mismo concepto (`compatible_set`) vía API legacy/canónica. |
| H-04 | confirmado | `tests/test_experiment_t0_sweep_full_*.py`, `tests/integration/test_experiment_t0_sweep_full_integration_unittest.py` | El experimento `_full` mantiene 6 ficheros de test fragmentados (5 unit + 1 integration). |
| H-05 | confirmado | `tests/test_pipeline_*.py` | `pipeline.py` está cubierto por múltiples ficheros heterogéneos; falta consolidación temática mínima. |
| H-06 | confirmado | `mvp/s6_geometry_table.py`, `mvp/s6_multi_event_table.py` | `s6_geometry_table.py` sigue fuera del carril contractual canónico y solapa función tabular. |
| H-07 | confirmado | `mvp/experiment/legacy_top_level/analyze_geometry_support.py`, `tests/test_experiment_geometry_support_unittest.py`, `tests/integration/test_experiment_geometry_support_smoke.py` | Código marcado legacy aún activo con tests vivos. |
| H-08 | confirmado | `download_gw_events.py`, `tools/fetch_catalog_events.py`, `tools/fetch_losc_event.py`, `tools/gwosc_sync_losc.py` | Sigue existiendo script voluminoso en raíz con responsabilidad solapada a `tools/`. |
| H-09 | parcialmente confirmado | `mvp/s4c_kerr_consistency.py`, `mvp/pipeline.py`, `mvp/s5_aggregate.py`, `mvp/experiment_t0_sweep_full.py` | Es stage liviano, pero actualmente sí tiene valor operativo como artefacto de gate consumido por varios downstreams; absorción inmediata elevaría riesgo. |
| H-10 | confirmado | `tests/test_mvp_contracts.py`, `tests/test_contracts_completeness.py`, `tests/test_contracts_fase4.py`, `tests/test_contracts_fase5.py`, `tests/test_contract_runtime_alignment.py`, `tests/test_dynamic_contracts.py` | Test suite contractual muy repartida para un SSOT único (`mvp/contracts.py`). |
| H-11 | confirmado | `tests/test_oracle_*.py`, `tests/test_oracle_v1_*.py` | Fragmentación alta del subsistema oracle. |
| H-12 | parcialmente confirmado | `mvp/experiment_dual_method.py`, `mvp/s3_spectral_estimates.py` | Es más gate/QC que experimento, pero puede mantenerse en `experiment/` como wrapper no soberano sin romper contrato canónico. |
| H-13 | confirmado | `tests/test_s5_*.py` | Cobertura de s5 dividida en 6 ficheros para 2 módulos. |

## 2. Tabla de remediación priorizada

| prioridad | ID hallazgo | ruta exacta | categoría | problema contractual o estructural | acción mínima propuesta | destino soberano | test de regresión requerido | riesgo | esfuerzo |
|---|---|---|---|---|---|---|---|---|---|
| P0 | H-02 | `mvp/s4g_mode220_geometry_filter.py` | contrato/gobernanza | Output canónico dual (`geometries_220.json` + alias) crea ambigüedad para downstream | Declarar `mode220_filter.json` como output soberano; mantener `geometries_220.json` solo como alias de compatibilidad con deprecación explícita | `s4g_mode220_geometry_filter/outputs/mode220_filter.json` | test de contrato que verifique outputs+hashes+manifest y preferencia del nombre soberano | bajo | bajo |
| P0 | H-02 | `mvp/s4i_common_geometry_intersection.py` | contrato/gobernanza | Admite fallback a output legacy de s4g, perpetuando deuda | Mantener fallback 1 release con warning y fecha de retiro; luego exigir solo output soberano | `s4i_common_geometry_intersection` leyendo `mode220_filter.json` | test de compatibilidad temporal (nuevo+legacy) y test futuro (solo nuevo) | bajo | bajo |
| P1 | H-01 | `mvp/pipeline.py` | unificación entrypoint | `pipeline` llama al wrapper deprecado (`experiment_t0_sweep.py`) en lugar de `_full` | Cambiar invocación best-effort a `mvp.experiment_t0_sweep_full --phase run` y mantener parser de resultado estable | `mvp.experiment_t0_sweep_full` | test de pipeline opcional t0_sweep verificando misma selección `best_point.t0_ms` | medio | bajo |
| P1 | H-01 | `mvp/experiment_t0_sweep.py` | deprecación controlada | Wrapper aún soberano para CLI legacy implícita | Mantener wrapper pero con deprecación con fecha/semver y salida estable; documentar EOL; evitar borrado brusco | `_full` como única implementación soberana | test CLI de compatibilidad (`--run`/`--run-id`) + snapshot payload legacy | bajo | bajo |
| P1 | H-01 | `mvp/experiment_t0_sweep_221.py` | variante funcional | Variante 221 está separada del troncal `_full` | Extraer núcleo común (grid/subrun/contract writer) en helper compartido; conservar CLI 221 como wrapper funcional | `_full` + helper común reutilizable | test determinista comparando filas/campos comunes entre `_221` y `_full` en modo equivalente | medio | medio |
| P2 | H-03 | `tests/test_schema_compatible_set.py`, `tests/test_compatible_set_schema.py` | deuda de tests | Duplicación semántica | Consolidar en un único fichero, preservando casos legacy como sub-sección | `tests/test_compatible_set_schema.py` | pytest de módulo consolidado | bajo | bajo |
| P2 | H-04 | `tests/test_experiment_t0_sweep_full_*.py` | deuda de tests | Fragmentación excesiva para un experimento | Consolidar unit tests por temática dentro de 1 fichero principal + 1 integration | `tests/test_experiment_t0_sweep_full_unittest.py` (o equivalente existente ampliado) | corrida completa de tests t0_sweep_full antes/después con mismo número PASS | medio | medio |
| P3 | H-05 | `tests/test_pipeline_*.py` | deuda de tests | Fragmentación de cobertura pipeline | Consolidar gradualmente por categorías (CLI/orquestación/guardrails), sin big-bang | `tests/test_pipeline_cli_local_hdf5.py` + 1 soporte | pytest subconjunto pipeline | bajo | medio |
| P3 | H-06 | `mvp/s6_geometry_table.py` | legado utilitario | utilidad fuera de contratos canónicos | Reclasificar como `tools/` o `experiment/derived` + aviso “no-stage” y sin gate downstream | `tools/` (no stage) | test smoke de CLI + test no escritura fuera de `BASURIN_RUNS_ROOT` | bajo | bajo |
| P3 | H-07 | `mvp/experiment/legacy_top_level/analyze_geometry_support.py` | legacy | legacy activo con tests | Marcar wrapper deprecated + redirigir a `mvp/experiment/geometry_evidence_vs_gr.py` cuando aplique | `mvp/experiment/geometry_evidence_vs_gr.py` | smoke test legacy con warning y código 0 | bajo | bajo |
| P3 | H-08 | `download_gw_events.py` | layout/higiene | script raíz fuera de convención | Crear wrapper en raíz que delegue a `tools/download_gw_events.py`; mover implementación sin romper CLI | `tools/download_gw_events.py` | test import/CLI path estable | bajo | bajo |
| P3 | H-09 | `mvp/s4c_kerr_consistency.py` | diseño stage | stage ligero pero consumido downstream | No absorber ahora; primero endurecer contrato y documentar rol gate | stage actual se mantiene temporalmente | test contractual + wiring en pipeline | bajo | bajo |
| P4 | H-10 | `tests/test_*contracts*.py` | deuda de tests | dispersión contratos | consolidación incremental por fase sin perder granularidad | 2-3 ficheros consolidados | pytest contratos | bajo | medio |
| P4 | H-11 | `tests/test_oracle*.py` | deuda de tests | dispersión oracle | consolidar por módulo real (`math`, `selection`, `plateau`, `schema`) | 4 ficheros máximo | pytest oracle | bajo | medio |
| P4 | H-12 | `mvp/experiment_dual_method.py` | clasificación | experimento opera como gate | mantener como wrapper no soberano; extraer función de decisión reusables | helper en `mvp/` + CLI mínimo | test unit de decisión + smoke wrapper | bajo | medio |
| P4 | H-13 | `tests/test_s5_*.py` | deuda de tests | fragmentación s5 | consolidar en 2 ficheros (aggregate/event_row) | 2 ficheros | pytest s5 | bajo | medio |

## 3. Remediación detallada de H-02

### Diagnóstico trazable
- El hallazgo original ya no describe el estado actual: los stages `s4g/s4h/s4i/s4j` sí están integrados al lifecycle contractual (`init_stage`, `check_inputs`, `finalize`, `abort`, `log_stage_paths`) y están en `mvp/contracts.py`.
- La deuda remanente ya no es “bypass de contracts.py”, sino **ambigüedad de output soberano** en `s4g` + fallback legacy en `s4i`.

### Rutas exactas afectadas
- `mvp/s4g_mode220_geometry_filter.py`
- `mvp/s4i_common_geometry_intersection.py`
- `mvp/contracts.py`
- tests relacionados: `tests/test_s4g_mode220_geometry_filter.py`

### Dónde se rompe el SSOT contractual (estado actual)
- SSOT de contrato existe, pero `s4g` publica output histórico (`geometries_220.json`) y alias (`mode220_filter.json`), mientras `s4i` permite fallback legacy. Eso mantiene dos “verdades” de naming para el mismo artefacto funcional.

### Cambio mínimo recomendado
1. Declarar formalmente en doc+tests que `mode220_filter.json` es el artefacto soberano de `s4g`.
2. Mantener escritura de `geometries_220.json` solo por compatibilidad transitoria.
3. Añadir warning explícito en `s4i` cuando consume el legado (`geometries_220.json`) con fecha de retiro.
4. En fase posterior corta, eliminar fallback y exigir sólo `mode220_filter.json`.

### Fases de migración seguras
- Fase A (compat): soberano + warning legacy.
- Fase B (enforcement): quitar fallback legacy en `s4i`.

### Tests concretos
- Unit: `s4g` emite ambos archivos con mismo contenido lógico y hashes en manifest.
- Integration-lite: `s4i` usa output soberano por defecto; fallback legacy activa warning.
- Golden: snapshot normalizado de `mode220_filter.json` para evitar drift silencioso.

### Criterio de cierre
- No quedan lecturas downstream obligatorias de `geometries_220.json`.
- `s4i` falla con error contractual claro si falta `mode220_filter.json`.
- Tests contractuales de golden pipeline verdes.

## 4. Remediación detallada de H-01

### Inventario de implementaciones t0_sweep
- `mvp/experiment_t0_sweep_full.py`: implementación troncal (subruns, fases run/finalize/diagnose, inventario/oracle).
- `mvp/experiment_t0_sweep.py`: wrapper legacy deprecado que delega a `_full`.
- `mvp/experiment_t0_sweep_221.py`: variante específica de modo 221 para campañas concretas.

### Implementación soberana propuesta
- **Soberana:** `mvp/experiment_t0_sweep_full.py`.
- Motivo: ya concentra lógica principal, gating y topología de subruns; el wrapper legacy y la variante 221 son bordes de interfaz.

### Qué deprecar / absorber
- `experiment_t0_sweep.py`: mantener temporalmente por compatibilidad CLI; deprecación explícita con sunset.
- `experiment_t0_sweep_221.py`: no borrar; convertir progresivamente en wrapper del núcleo común de `_full` (o helper compartido), preservando su CLI especializada.
- `pipeline.py`: dejar de invocar wrapper legacy y llamar directo a `_full`.

### Cambio mínimo recomendado
1. `pipeline.py` -> reemplazar comando opcional por `_full`.
2. `experiment_t0_sweep.py` -> solo compatibilidad + warning fuerte + documentación EOL.
3. Factorizar utilidades comunes `_full`/`_221` (parsing de grid, ejecución de subrun, escritura contractual) sin alterar outputs.

### Tests concretos
- Test de regresión de `pipeline --with-t0-sweep`: mismo `selected_t0_ms` antes/después.
- Test CLI wrapper legacy: sigue generando `experiment/t0_sweep/outputs/t0_sweep_results.json` con schema legacy.
- Test comparativo `_221` vs helper común: mismas columnas y orden determinista para una grid fija.

### Criterio de cierre
- `pipeline.py` ya no depende de `experiment_t0_sweep.py`.
- `_full` es único backend soberano de sweep general.
- Wrapper legacy queda aislado, explícitamente deprecado y no bloquea evolución.

## 5. Secuencia de ejecución recomendada

1. **Objetivo:** cerrar H-02 real (naming soberano).
   - Archivos: `mvp/s4g_mode220_geometry_filter.py`, `mvp/s4i_common_geometry_intersection.py`, `tests/test_s4g_mode220_geometry_filter.py`.
   - Precondición: contratos golden existentes en verde.
   - Cambio: soberanía `mode220_filter.json` + warning de fallback legacy.
   - Validación: pytest de s4g/s4i + contrato manifest/hash.

2. **Objetivo:** desacoplar pipeline del wrapper legacy (H-01).
   - Archivos: `mvp/pipeline.py`, tests pipeline t0.
   - Precondición: `_full` estable.
   - Cambio: invocar `mvp.experiment_t0_sweep_full --phase run`.
   - Validación: mismo `selected_t0_ms`, no regresión CLI.

3. **Objetivo:** blindar deprecación de `experiment_t0_sweep.py`.
   - Archivos: `mvp/experiment_t0_sweep.py`, README/doc experimento, test CLI.
   - Precondición: paso 2 mergeado.
   - Cambio: deprecación explícita con sunset.
   - Validación: compatibilidad de salida legacy + warning.

4. **Objetivo:** reducir deuda de tests mayor impacto (H-03/H-04).
   - Archivos: tests de `compatible_set` y `t0_sweep_full`.
   - Precondición: pasos 1-3 estables.
   - Cambio: consolidación incremental sin cambiar semántica.
   - Validación: mismo número de tests funcionales PASS.

5. **Objetivo:** backlog periférico H-05..H-13.
   - Archivos: tests y wrappers legacy.
   - Precondición: core contractual estabilizado.
   - Cambio: higiene estructural por lotes pequeños.
   - Validación: smoke + contract tests.

## 6. Riesgos y supuestos

- Riesgo real: consumidores externos puedan leer aún `geometries_220.json`; por eso se requiere fase de transición con warning.
- Riesgo real: `pipeline` depende de semántica específica del payload legacy `t0_sweep_results.json`; hay que conservar parser o mapear exactamente al payload de `_full`.
- Supuesto: no se cambia formato científico de resultados, sólo gobernanza/interfaz.
- Supuesto: `BASURIN_RUNS_ROOT` seguirá siendo la raíz efectiva en CI para validar no-escape de IO.

## 7. Qué NO tocar todavía

- No reescribir `mvp/experiment_t0_sweep_full.py` completo (alto riesgo por tamaño/superficie).
- No absorber `mvp/s4c_kerr_consistency.py` en otro stage en esta fase.
- No mover masivamente tests de oracle/contracts en el mismo PR que cambie ejecución de pipeline.
- No alterar el pipeline canónico s0-s7 salvo el punto mínimo de invocación opcional de t0_sweep.

## 8. Backlog de PRs mínimos

### PR-1
- **Título:** `golden-geometry: declarar output soberano mode220_filter y transición legacy en s4i`
- **Archivos exactos:**
  - `mvp/s4g_mode220_geometry_filter.py`
  - `mvp/s4i_common_geometry_intersection.py`
  - `tests/test_s4g_mode220_geometry_filter.py`
- **Objetivo:** cerrar H-02 real (ambigüedad de output, no bypass contractual).
- **Riesgo:** bajo.
- **Tests:** unit+integration-lite s4g/s4i, manifest/hash checks.
- **Dependencia:** ninguna.

### PR-2
- **Título:** `pipeline: usar experiment_t0_sweep_full como backend opcional`
- **Archivos exactos:**
  - `mvp/pipeline.py`
  - `tests/test_pipeline_orchestrator.py`
  - `tests/test_pipeline_guardrails.py`
- **Objetivo:** cerrar parte crítica de H-01 (interfaz soberana).
- **Riesgo:** medio (wiring de CLI).
- **Tests:** tests pipeline relacionados con `--with-t0-sweep`.
- **Dependencia:** PR-1 no requerida.

### PR-3
- **Título:** `t0_sweep legacy wrapper: deprecación explícita y contrato de compatibilidad`
- **Archivos exactos:**
  - `mvp/experiment_t0_sweep.py`
  - `tests/integration/test_experiment_t0_sweep_integration.py`
  - `README.md` (sección deprecaciones)
- **Objetivo:** mantener compatibilidad sin soberanía funcional del wrapper.
- **Riesgo:** bajo.
- **Tests:** integration wrapper + snapshot payload legacy.
- **Dependencia:** PR-2.

### PR-4
- **Título:** `t0_sweep tests: consolidación mínima compatible_set + t0_sweep_full`
- **Archivos exactos:**
  - `tests/test_schema_compatible_set.py`
  - `tests/test_compatible_set_schema.py`
  - `tests/test_experiment_t0_sweep_full_*.py` (consolidación incremental)
- **Objetivo:** reducir deuda H-03/H-04 sin tocar lógica científica.
- **Riesgo:** medio (movimiento de tests).
- **Tests:** pytest subset + comparación de conteo PASS.
- **Dependencia:** PR-2 recomendado.

### PR-5
- **Título:** `legacy/peripheral hygiene: s6_geometry_table y analyze_geometry_support como wrappers no soberanos`
- **Archivos exactos:**
  - `mvp/s6_geometry_table.py`
  - `mvp/experiment/legacy_top_level/analyze_geometry_support.py`
  - tests smoke correspondientes
- **Objetivo:** cerrar H-06/H-07 con impacto mínimo.
- **Riesgo:** bajo.
- **Tests:** smoke CLIs + no escritura fuera de runs root.
- **Dependencia:** ninguna fuerte.
