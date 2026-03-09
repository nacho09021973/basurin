# Plan de Remediacion — BASURIN Audit Findings H-01..H-13

**Fecha:** 2026-03-08
**Base:** `fragmentation_audit_2026-03-08.md` (13 hallazgos)
**Commit base:** `main` HEAD (89ee24e)
**Metodologia:** Reinspeccion contract-first sobre codigo real, no sobre resumen del audit

---

# 1. Verificacion del audit previo

| ID | Veredicto | Rutas exactas | Evidencia breve |
|----|-----------|---------------|-----------------|
| H-01 | **Confirmado** | `mvp/experiment_t0_sweep.py`, `mvp/experiment_t0_sweep_full.py`, `mvp/experiment_t0_sweep_221.py` | Las 3 rutas existen. `_sweep.py` tiene banner DEV/INTEGRATION explicito (linea 46-51). `_full.py` es canonico con subruns aislados. `_221.py` es mode-specific. Ninguno de los 3 usa `from mvp.contracts import init_stage`. |
| H-02 | **Confirmado + AMPLIADO con bug critico** | `mvp/s4g_mode220_geometry_filter.py`, `mvp/s4h_mode221_geometry_filter.py`, `mvp/s4i_common_geometry_intersection.py`, `mvp/s4j_hawking_area_filter.py` | Ninguno usa `init_stage/finalize/abort` de contracts.py. Solo s4g tiene entrada en `CONTRACTS` dict; s4h, s4i, s4j **faltan**. **Bug nuevo descubierto:** s4g escribe `"accepted_geometry_ids"` (linea 233) pero s4i lee `s4g_data.get("geometry_ids", [])` (linea 104) — **key mismatch = lista vacia siempre**. Ademas s4i espera fichero `mode220_filter.json` (S4G_OUTPUT_REL, linea 47) pero s4g escribe `geometries_220.json` (OUTPUT_FILE, linea 60) — **file-not-found garantizado**. |
| H-03 | **Confirmado** | `tests/test_schema_compatible_set.py`, `tests/test_compatible_set_schema.py` | Ambos testean `mvp/schemas.py` sobre compatible_set. El primero usa `validate_compatible_set`, el segundo usa `validate`. |
| H-04 | **Confirmado** | `tests/test_experiment_t0_sweep_full_diagnose_unittest.py`, `tests/test_experiment_t0_sweep_full_finalize_golden.py`, `tests/test_experiment_t0_sweep_full_inventory_unittest.py`, `tests/test_experiment_t0_sweep_full_oracle_from_results.py`, `tests/test_experiment_t0_sweep_full_paths.py` | 5 ficheros unit + 1 integration para un solo experimento. |
| H-05 | **Confirmado** | `tests/test_pipeline_orchestrator.py`, `tests/test_pipeline_cli_local_hdf5.py`, `tests/test_pipeline_estimator.py`, `tests/test_pipeline_guardrails.py`, `tests/test_pipeline_calls_s6_information_geometry.py`, `tests/test_pipeline_s1_local_hdf5_autodetect.py` | 6 ficheros de test para pipeline.py. |
| H-06 | **Confirmado** | `mvp/s6_geometry_table.py` | Sin entrada en CONTRACTS. No usa init_stage/finalize. Tabulacion solapada con s6_multi_event_table.py. |
| H-07 | **Confirmado** | `mvp/experiment/legacy_top_level/analyze_geometry_support.py` | En directorio legacy_top_level/ con 2 ficheros de test activos. |
| H-08 | **Confirmado** | `download_gw_events.py` (raiz) | 17413 bytes en raiz. tools/ tiene scripts de descarga independientes. |
| H-09 | **Confirmado** | `mvp/s4c_kerr_consistency.py` | 168 LOC. Solo lee s3b + s4 y emite booleano. Usa init_stage/finalize (contractual), pero computacionalmente trivial. |
| H-10 | **Confirmado** | `tests/test_mvp_contracts.py`, `tests/test_contracts_completeness.py`, `tests/test_contracts_fase4.py`, `tests/test_contracts_fase5.py`, `tests/test_contract_runtime_alignment.py`, `tests/test_dynamic_contracts.py` | 6 ficheros de test para contracts.py. |
| H-11 | **Confirmado** | `tests/test_oracle_mvp_governance.py`, `tests/test_oracle_t0_input_schema.py`, `tests/test_oracle_t0_math.py`, `tests/test_oracle_t0_selection.py`, `tests/test_oracle_t0_v1_2_golden.py`, `tests/test_oracle_v1_gate_reasons.py`, `tests/test_oracle_v1_plateau_logic.py`, `tests/test_oracle_v1_plateau_parsing.py` | 8 ficheros de test para 4 modulos oracle. |
| H-12 | **Confirmado** | `mvp/experiment_dual_method.py` | 268 LOC. Gate de QC disfrazado de experimento. Sin init_stage, usa manifest custom. |
| H-13 | **Confirmado** | `tests/test_s5_aggregate_extractors.py`, `tests/test_s5_aggregate_ranked_integration.py`, `tests/test_s5_event_row.py`, `tests/test_s5_joint_posterior.py`, `tests/test_s5_paths_relative_unittest.py`, `tests/test_s5_topk_affects_coverage_unittest.py` | 6 ficheros para 2 modulos fuente (s5_aggregate.py, s5_event_row.py). |

### Hallazgo nuevo descubierto durante reinspeccion

| ID | Tipo | Rutas | Descripcion |
|----|------|-------|-------------|
| **H-02a** | Bug de integridad de datos | `mvp/s4g_mode220_geometry_filter.py:233`, `mvp/s4i_common_geometry_intersection.py:104` | **Key mismatch:** s4g escribe `"accepted_geometry_ids"`, s4i lee `"geometry_ids"` → lista vacia siempre. |
| **H-02b** | Bug file-not-found | `mvp/s4g_mode220_geometry_filter.py:60`, `mvp/s4i_common_geometry_intersection.py:47` | **Filename mismatch:** s4g escribe `geometries_220.json`, s4i espera `mode220_filter.json` → s4i falla siempre con "mode-220 filter output not found". |
| **H-02c** | Contract output mismatch | `mvp/contracts.py:361`, `mvp/s4g_mode220_geometry_filter.py:60` | Contract declara `outputs/geometries_220.json` pero docstring de s4g dice `mode220_filter.json`. El codigo (linea 60) escribe `geometries_220.json`, consistente con el contract pero inconsistente con la cadena s4i downstream. |

---

# 2. Tabla de remediacion priorizada

| Prio | ID | Ruta exacta | Categoria | Problema | Accion minima | Destino soberano | Test regresion | Riesgo | Esfuerzo |
|------|----|-------------|-----------|----------|---------------|------------------|----------------|--------|----------|
| **P0** | H-02b | `mvp/s4i_common_geometry_intersection.py:47` | Bug | s4i espera `mode220_filter.json` pero s4g escribe `geometries_220.json` — pipeline roto | Cambiar `S4G_OUTPUT_REL` en s4i a `"s4g_mode220_geometry_filter/outputs/geometries_220.json"` | s4i lee el fichero correcto | `test_s4g_mode220_geometry_filter.py` + nuevo test de integracion s4g→s4i filename match | Bajo | Bajo |
| **P0** | H-02a | `mvp/s4i_common_geometry_intersection.py:104` | Bug | s4i lee key `"geometry_ids"` pero s4g escribe `"accepted_geometry_ids"` | Cambiar linea 104 de s4i: `s4g_data.get("accepted_geometry_ids", s4g_data.get("geometry_ids", []))` — o normalizar key en s4g a `"geometry_ids"` | Llave consistente entre s4g y s4i | Nuevo test: escribir s4g output, leer con s4i, verificar ids no vacios | Bajo | Bajo |
| **P1** | H-02 | `mvp/s4h_mode221_geometry_filter.py`, `mvp/s4i_common_geometry_intersection.py`, `mvp/s4j_hawking_area_filter.py` | Contractual | s4h/s4i/s4j no tienen entrada en `CONTRACTS` dict | Anadir 3 StageContract entries en `mvp/contracts.py` | `CONTRACTS["s4h_*"]`, `CONTRACTS["s4i_*"]`, `CONTRACTS["s4j_*"]` | `test_contracts_completeness.py` pasa limpio; test nuevo verifica 4/4 golden stages en CONTRACTS | Bajo | Bajo |
| **P1** | H-02 | `mvp/s4g_mode220_geometry_filter.py`, `mvp/s4h_mode221_geometry_filter.py`, `mvp/s4i_common_geometry_intersection.py`, `mvp/s4j_hawking_area_filter.py` | Contractual | 4 stages no usan `init_stage/finalize/abort` | Migrar `main()` de cada uno a usar `init_stage → check_inputs → finalize` manteniendo logica de negocio intacta | Pasos contractuales centralizados | Tests existentes de s4g + nuevos tests de lifecycle (init crea stage_dir, abort produce FAIL summary) | Medio | Medio |
| **P2** | H-01 | `mvp/experiment_t0_sweep.py` | Duplicacion | Script DEV/INTEGRATION que confunde sobre cual usar | Deprecar: anadir header DEPRECATED + redirect a `experiment_t0_sweep_full.py`. Mover a `mvp/experiment/deprecated/` | `mvp/experiment_t0_sweep_full.py` es el unico soberano para sweeps t0 | `test_experiment_t0_sweep_full_integration_unittest.py` cubre escenarios equivalentes | Bajo | Bajo |
| **P2** | H-01 | `mvp/experiment_t0_sweep_221.py` | Duplicacion parcial | Mode-221 sweep como script separado; candidato a parametrizacion en `_full.py` | Fase 1: documentar como variante legitima (modo 221 aislado). Fase 2 (opcional): parametrizar `_full.py --mode 221` | Si se parametriza: `_full.py` con flag `--mode`. Si no: mantener como variante documentada | Test de equivalencia: ejecutar ambos con mismo grid, comparar structure de outputs | Medio | Medio |
| **P3** | H-03 | `tests/test_schema_compatible_set.py` | Test duplicado | Duplicacion semantica con `tests/test_compatible_set_schema.py` | Fusionar en `tests/test_compatible_set_schema.py`; eliminar `test_schema_compatible_set.py` | `tests/test_compatible_set_schema.py` | Misma cuenta de tests PASS pre/post | Bajo | Bajo |
| **P3** | H-06 | `mvp/s6_geometry_table.py` | Obsolescencia | No contractual; solapado con s6_multi_event_table.py | Anadir header DEPRECATED. Documentar que s6_multi_event_table.py es la alternativa | `mvp/s6_multi_event_table.py` | Test existente `test_geometry_table_unittest.py` sigue pasando | Bajo | Bajo |
| **P3** | H-07 | `mvp/experiment/legacy_top_level/analyze_geometry_support.py` | Legacy | Legacy con tests activos en mainline | Anadir header DEPRECATED con fecha y redirect a `ex2_ranking.py` + `geometry_evidence_vs_gr.py` | `mvp/experiment/ex2_ranking.py`, `mvp/experiment/geometry_evidence_vs_gr.py` | Tests legacy siguen pasando; no se borran aun | Bajo | Bajo |
| **P3** | H-08 | `download_gw_events.py` (raiz) | Ubicacion | Script grande en raiz, rompe convencion tools/ | `git mv download_gw_events.py tools/download_gw_events.py`. Actualizar .gitignore y README si hay refs | `tools/download_gw_events.py` | `grep -r download_gw_events` muestra 0 refs rotas | Bajo | Bajo |
| **P4** | H-04 | `tests/test_experiment_t0_sweep_full_*.py` (5 ficheros) | Test fragmentation | 5+1 ficheros para un experimento | Consolidar en 2: `test_experiment_t0_sweep_full_unit.py` + integration existente | 2 ficheros max | Misma cuenta de tests PASS | Bajo | Medio |
| **P4** | H-05 | `tests/test_pipeline_*.py` (6 ficheros) | Test fragmentation | 6 ficheros para pipeline.py | Consolidar en 2: `test_pipeline_orchestrator.py` + `test_pipeline_cli.py` | 2 ficheros max | Misma cuenta de tests PASS | Bajo | Medio |
| **P4** | H-09 | `mvp/s4c_kerr_consistency.py` | Inflacion | Stage trivial (gate booleano) con full lifecycle para 3 lineas de logica | No tocar en esta fase. Ya es contractual. Evaluar absorcion en s4d en fase posterior | Mantener como esta | N/A | Bajo | N/A |
| **P4** | H-10 | `tests/test_*contracts*.py` (6 ficheros) | Test fragmentation | 6 ficheros para contracts.py | Consolidar en 2-3 ficheros en fase posterior | 2-3 ficheros max | Misma cuenta | Bajo | Medio |
| **P4** | H-11 | `tests/test_oracle_*.py` (8 ficheros) | Test fragmentation | 8 ficheros para 4 modulos oracle | Consolidar en 3-4 en fase posterior | 3-4 ficheros max | Misma cuenta | Bajo | Medio |
| **P4** | H-12 | `mvp/experiment_dual_method.py` | Clasificacion | Gate de QC disfrazado de experimento | Documentar como QC gate en fase posterior. No mover aun | Mantener | N/A | Bajo | N/A |
| **P4** | H-13 | `tests/test_s5_*.py` (6 ficheros) | Test fragmentation | 6 ficheros para 2 modulos s5 | Consolidar en 2 en fase posterior | 2 ficheros max | Misma cuenta | Bajo | Medio |

---

# 3. Remediacion detallada de H-02

## Diagnostico trazable

El pipeline golden geometry consta de 4 stages ejecutados secuencialmente:

```
s4g_mode220_geometry_filter
  → s4h_mode221_geometry_filter  (paralelo/independiente de s4g)
    → s4i_common_geometry_intersection  (consume s4g + s4h)
      → s4j_hawking_area_filter  (consume s4i)
```

### 3.1 Brecha contractual: ausencia de `init_stage/finalize/abort`

**Rutas exactas afectadas:**
- `mvp/s4g_mode220_geometry_filter.py` — importa `basurin_io` directamente (lineas 39-47), nunca importa `mvp.contracts`
- `mvp/s4h_mode221_geometry_filter.py` — idem (lineas 39-46)
- `mvp/s4i_common_geometry_intersection.py` — idem (lineas 31-38)
- `mvp/s4j_hawking_area_filter.py` — idem (lineas 44-51)

**Donde se rompe el SSOT contractual:**

1. **Sin check_inputs contractual:** Los 4 stages verifican existencia de ficheros manualmente (`if not obs_path.exists(): return 2`). No pasan por `check_inputs()` que ademas:
   - Computa SHA256 de inputs y los registra en `ctx.inputs_record`
   - Verifica upstream stages via `required_inputs_glob`
   - Emite abort con tracing completo si falta algo

2. **Sin abort contractual:** Cuando fallan, imprimen a stderr y retornan `2`. No escriben `stage_summary.json` con `verdict: FAIL` ni `manifest.json` de fallo — dejando el stage en estado indeterminado (sin summary ≠ FAIL; simplemente no existe).

3. **Sin finalize contractual:** Escriben manifest y summary manualmente llamando `write_manifest()` y `write_stage_summary()` de basurin_io, lo cual produce output correcto en happy path pero:
   - No ejecuta `enforce_outputs` para verificar que los produced_outputs declarados existen
   - No emite la marca `PASS_WITHOUT_OUTPUTS` si no se produjeron outputs
   - No invoca `log_stage_paths()` de contracts.py (aunque los 4 imprimen las 5 variables manualmente, identicas)

4. **Declaracion incompleta en CONTRACTS dict:**
   - `s4g_mode220_geometry_filter`: **presente** en `contracts.py:352-365` pero con anomalias:
     - `required_inputs: ["s4g_mode220_geometry_filter/inputs/mode220_obs.json"]` — apunta a si mismo como upstream, lo cual es semanticamente correcto (input local al stage) pero inusual
     - `upstream_stages: []` — no declara dependencia de ningun upstream
     - `produced_outputs: ["outputs/geometries_220.json"]` — correcto vs codigo
   - `s4h_mode221_geometry_filter`: **AUSENTE**
   - `s4i_common_geometry_intersection`: **AUSENTE**
   - `s4j_hawking_area_filter`: **AUSENTE**

5. **Bugs de integracion entre stages (H-02a, H-02b):**
   - `s4g` escribe fichero `geometries_220.json` pero `s4i` espera `mode220_filter.json` (H-02b)
   - `s4g` escribe key `"accepted_geometry_ids"` pero `s4i` lee `"geometry_ids"` (H-02a)
   - Estos bugs hacen que el pipeline s4g→s4i este **roto en produccion** si se ejecuta secuencialmente

### 3.2 Cambio minimo recomendado

**Fase A: Bugs criticos (P0, dia 1)**

1. **Fix H-02b (filename mismatch):**
   - Ruta: `mvp/s4i_common_geometry_intersection.py:47`
   - Cambio: `S4G_OUTPUT_REL = "s4g_mode220_geometry_filter/outputs/geometries_220.json"`
   - Alternativa rechazada: cambiar s4g para que escriba `mode220_filter.json` — romperia el contrato ya declarado en contracts.py y tests existentes de s4g

2. **Fix H-02a (key mismatch):**
   - Opcion A (preferida): normalizar s4g para que escriba `"geometry_ids"` (consistente con s4h). Cambio en `mvp/s4g_mode220_geometry_filter.py:233`: cambiar `"accepted_geometry_ids"` → `"geometry_ids"`
   - Opcion B: cambiar s4i para leer ambos keys con fallback. Rechazada por no resolver la inconsistencia de schema.
   - Verificar que ningun otro consumer lee `"accepted_geometry_ids"` de s4g:
     - `experiment_single_event_golden_robustness.py` — no lee s4g outputs directamente (usa funciones puras in-memory)
     - `tests/test_s4g_mode220_geometry_filter.py` — verificar si testea esta key

**Fase B: Contratos SSOT (P1, dia 2-3)**

3. **Anadir contratos faltantes a `mvp/contracts.py`:**

```python
"s4h_mode221_geometry_filter": StageContract(
    name="s4h_mode221_geometry_filter",
    required_inputs=[],  # mode221_obs.json es opcional (skip si ausente)
    external_inputs=["atlas"],
    produced_outputs=[
        "outputs/mode221_filter.json",
    ],
    upstream_stages=["s4g_mode220_geometry_filter"],  # secuencial, no hard dependency
    check_run_valid=True,
),
"s4i_common_geometry_intersection": StageContract(
    name="s4i_common_geometry_intersection",
    required_inputs=[
        "s4g_mode220_geometry_filter/outputs/geometries_220.json",
    ],
    dynamic_inputs=[
        "s4h_mode221_geometry_filter/outputs/mode221_filter.json",  # optional
    ],
    produced_outputs=[
        "outputs/common_intersection.json",
    ],
    upstream_stages=["s4g_mode220_geometry_filter"],
    check_run_valid=True,
),
"s4j_hawking_area_filter": StageContract(
    name="s4j_hawking_area_filter",
    required_inputs=[
        "s4i_common_geometry_intersection/outputs/common_intersection.json",
    ],
    produced_outputs=[
        "outputs/hawking_area_filter.json",
    ],
    upstream_stages=["s4i_common_geometry_intersection"],
    check_run_valid=True,
),
```

4. **Corregir contrato existente de s4g:**
   - Anadir `external_inputs=["atlas"]` (actualmente falta — s4g recibe `--atlas-path`)
   - Verificar que `produced_outputs` dice `"outputs/geometries_220.json"` (ya correcto)

**Fase C: Migracion a lifecycle contractual (P1, dia 4-6)**

5. **Migrar `main()` de s4g/s4h/s4i/s4j** para usar `init_stage → check_inputs → finalize`:
   - Reemplazar bloques manuales de `resolve_out_root / validate_run_id / require_run_valid / mkdir` por `init_stage(run_id, STAGE)`
   - Reemplazar verificaciones manuales de inputs por `check_inputs(ctx, paths)`
   - Reemplazar `write_stage_summary + write_manifest` manuales por `finalize(ctx, artifacts, results=...)`
   - Reemplazar prints manuales de ERROR + return 2 por `abort(ctx, reason)`
   - Reemplazar prints manuales de OUT_ROOT/STAGE_DIR/... por `log_stage_paths(ctx)`
   - **Excepcion para s4h:** La semantica de "skip si obs no existe" requiere cuidado especial. El contrato de s4h NO debe tener `mode221_obs.json` como `required_inputs` (es opcional). La logica de skip debe ejecutarse DESPUES de `init_stage` pero ANTES de `check_inputs`, emitiendo un `finalize` con `verdict="SKIPPED"`.

### 3.3 Fases de migracion seguras

| Fase | Objetivo | Ficheros | Pre-condicion | Validacion |
|------|----------|----------|---------------|------------|
| A.1 | Fix filename s4g→s4i | `s4i_common_geometry_intersection.py` | Ninguna | Test: crear fixture s4g output como `geometries_220.json`, ejecutar s4i, verificar que lee correctamente |
| A.2 | Fix key mismatch s4g→s4i | `s4g_mode220_geometry_filter.py` o `s4i_common_geometry_intersection.py` | A.1 completado | Test: verificar que s4i obtiene ids no vacios de s4g output |
| B.1 | Anadir 3 contratos a CONTRACTS | `contracts.py` | A.1+A.2 | `test_contracts_completeness.py` pasa; `audit_contract_completeness()` no lista s4h/s4i/s4j |
| B.2 | Corregir contrato s4g | `contracts.py` | B.1 | `CONTRACTS["s4g_mode220_geometry_filter"].external_inputs == ["atlas"]` |
| C.1 | Migrar s4g a lifecycle | `s4g_mode220_geometry_filter.py` | B.1+B.2 | Test existente `test_s4g_mode220_geometry_filter.py` pasa; output bit-a-bit identico |
| C.2 | Migrar s4h a lifecycle | `s4h_mode221_geometry_filter.py` | B.1 | Nuevo test: happy path + skip path producen mismos outputs |
| C.3 | Migrar s4i a lifecycle | `s4i_common_geometry_intersection.py` | A.1+A.2+B.1 | Nuevo test de integracion s4g→s4i→verificar ids |
| C.4 | Migrar s4j a lifecycle | `s4j_hawking_area_filter.py` | B.1 | Nuevo test: happy path + empty-area-data path |

### 3.4 Tests concretos requeridos

1. **test_s4g_s4i_filename_contract**: Verificar que `S4G_OUTPUT_REL` en s4i apunta al `OUTPUT_FILE` real de s4g
2. **test_s4g_s4i_key_contract**: Escribir output s4g, leer con logica de s4i, verificar ids extraidos correctamente
3. **test_golden_stages_in_contracts**: Verificar que `s4g`, `s4h`, `s4i`, `s4j` estan en `CONTRACTS` dict
4. **test_s4h_skip_produces_summary**: Ejecutar s4h sin obs file, verificar que produce `stage_summary.json` con `verdict=SKIPPED`
5. **test_golden_pipeline_e2e_lifecycle**: Fixture que ejecuta s4g→s4h→s4i→s4j con init_stage/finalize, verificar que cada stage produce manifest+summary

### 3.5 Criterio de cierre

- [ ] `audit_contract_completeness()` retorna lista vacia (o solo stages intencionalmente sin inputs)
- [ ] Los 4 stages golden usan `from mvp.contracts import init_stage, check_inputs, finalize, abort`
- [ ] s4i lee correctamente output de s4g (filename correcto, key correcto)
- [ ] `test_contracts_completeness.py` pasa
- [ ] Nuevo test `test_golden_pipeline_e2e_lifecycle` pasa
- [ ] Los tests existentes de s4g no regresan
- [ ] `experiment_single_event_golden_robustness.py` sigue funcionando (usa funciones puras, no afectado por cambios de lifecycle)

---

# 4. Remediacion detallada de H-01

## 4.1 Inventario de implementaciones t0_sweep

| # | Ruta exacta | LOC | Usa contracts.py | Escribe bajo | Proposito | Estado |
|---|-------------|-----|------------------|-------------|-----------|--------|
| 1 | `mvp/experiment_t0_sweep.py` | 386 | NO (banner explicito lineas 46-51: "no contract-first, no inventory/finalize") | `runs/<run_id>/experiment/t0_sweep/` | DEV/INTEGRATION: sweep ligero sobre s2 outputs existentes. Sin subruns aislados, sin inventario, sin finalize contractual | **DEV legacy** |
| 2 | `mvp/experiment_t0_sweep_full.py` | 1918 | NO (no importa contracts.py; usa basurin_io directamente) | `runs/<run_id>/experiment/t0_sweep_full_seed<seed>/` con subruns aislados | Canonico: sweep completo con subruns aislados (s3 → s3b → s4c por punto del grid). Inventario de subruns. Oracle v1 plateau | **Canonico (soberano)** |
| 3 | `mvp/experiment_t0_sweep_221.py` | 411 | NO (usa basurin_io directo, sin init_stage) | `runs/<run_id>/experiment/t0_sweep_221/` con subruns via subprocess | Mode-221 especifico: sweep t0 para auditar aparicion del modo 221. Ejecuta subruns s2+s3b por punto del grid | **Variante mode-specific** |

### Relaciones entre las 3 implementaciones

- `_sweep.py` es un **subconjunto funcional** de `_full.py`: ambos barren t0, pero `_sweep.py` no aisla subruns, no tiene inventario, no tiene finalize contractual.
- `_sweep_221.py` tiene un **proposito ortogonal**: no busca el t0 optimo para el modo fundamental, sino que audita la aparicion/desaparicion del modo 221 en funcion de t0. Su logica de negocio (evaluar modo 221) es distinta de `_full.py` (evaluar modo 220 + oracle).
- `tools/run_experiment_batch.py` (lineas 178, 211) importa y orquesta `experiment_t0_sweep_full` — confirma que `_full.py` es el canonico para batch.
- `mvp/experiment_oracle_t0_ringdown.py` (linea 23, 35) consume outputs de `_full.py` — confirma cadena de dependencia.

### Consumers que importan directamente de cada uno:

| Script | Importa de | Que importa |
|--------|-----------|-------------|
| `tools/run_experiment_batch.py` | `experiment_t0_sweep_full` | `_new_subrun_trace`, `_write_subrun_trace` |
| `mvp/experiment_oracle_t0_ringdown.py` | N/A (lee ficheros) | Lee `t0_sweep_full_results.json` |
| `tests/test_experiment_t0_sweep_full_*.py` (5 ficheros) | `experiment_t0_sweep_full` | Varias funciones internas |
| `tests/test_experiment_t0_sweep_221.py` | `experiment_t0_sweep_221` | `_parse_grid` |
| `tests/test_eps_sweep.py` | N/A (testea eps_sweep, no t0) | N/A |

Ningun modulo importa funciones de `experiment_t0_sweep.py` (el DEV).

## 4.2 Implementacion soberana

**`mvp/experiment_t0_sweep_full.py` debe quedar como soberano** porque:

1. Es el unico con subruns aislados (requisito de reproducibilidad)
2. Es el unico con inventario y tracing de subruns
3. Es consumido por `tools/run_experiment_batch.py` (orquestador batch) y `experiment_oracle_t0_ringdown.py` (post-procesador)
4. Tiene 5 ficheros de test unit + 1 integration
5. Es 5x mas grande en LOC precisamente porque implementa la gobernanza completa

## 4.3 Clasificacion de las otras dos

### `experiment_t0_sweep.py` — DEPRECAR

- **Tipo:** Duplicado semantico (subconjunto de `_full.py`)
- **Evidencia:** Banner explicito "DEV/INTEGRATION TOOL: no contract-first, no inventory/finalize, no subrun isolation" (lineas 46-51)
- **Ningun modulo lo importa** (verificado con grep)
- **Riesgo de deprecacion:** Bajo. El unico test referenciado es `tests/test_experiment_t0_sweep_full_paths.py` que testea `_full`, no `_sweep`.

### `experiment_t0_sweep_221.py` — VARIANTE LEGITIMA (mantener, documentar)

- **Tipo:** Variante mode-specific con proposito distinto
- **Diferencia funcional real:** `_full.py` ejecuta s3→s3b→s4c (Kerr consistency) por punto. `_221.py` ejecuta s2→s3b por punto y evalua especificamente la aparicion del modo 221. Son workflows distintos.
- **No es parametrizable trivialmente** en `_full.py` porque:
  - `_full.py` asume que s2 ya esta ejecutado (toma s2 outputs como input)
  - `_221.py` re-ejecuta s2 por cada punto del grid (el t0 shift afecta la ventana ringdown)
  - Los outputs tienen estructura diferente (221 evalua frecuencias de modo, full evalua oracle plateau)
- **Recomendacion:** Mantener como variante documentada. No fusionar. Evaluar en fase posterior si ambos podrian compartir infraestructura de subruns.

## 4.4 Cambio minimo recomendado

1. **Deprecar `mvp/experiment_t0_sweep.py`:**
   - Anadir al principio del fichero (despues del docstring):
     ```python
     import warnings
     warnings.warn(
         "experiment_t0_sweep.py is DEPRECATED (2026-03-08). "
         "Use experiment_t0_sweep_full.py instead.",
         DeprecationWarning, stacklevel=2,
     )
     ```
   - Actualizar docstring para incluir `DEPRECATED: use experiment_t0_sweep_full.py`
   - NO mover el fichero (para no romper imports hipoteticos de usuarios)
   - NO borrar (podria haber scripts de usuario que lo referencien)

2. **Documentar `experiment_t0_sweep_221.py` como variante:**
   - Anadir al docstring: "VARIANTE mode-221: workflow ortogonal a experiment_t0_sweep_full.py"
   - Verificar que sus outputs escriben bajo `experiment/t0_sweep_221/` (ya correcto)

3. **No migrar ningun t0_sweep a contracts.py en esta fase:**
   - Los 3 son experimentos, no stages canonicos
   - La migracion a lifecycle contractual seria un refactor grande en `_full.py` (1918 LOC) con alto riesgo
   - Priorizar H-02 (golden geometry) que son stages canonicos genuinos

## 4.5 Tests concretos

1. **test_t0_sweep_deprecation_warning**: Importar `mvp.experiment_t0_sweep` y verificar que emite `DeprecationWarning`
2. **test_t0_sweep_full_covers_dev_scenarios**: Verificar que `_full.py` produce outputs equivalentes a `_sweep.py` para un grid simple (test de equivalencia funcional)

## 4.6 Criterio de cierre

- [ ] `experiment_t0_sweep.py` emite DeprecationWarning al importarse
- [ ] Docstring de `experiment_t0_sweep.py` dice DEPRECATED
- [ ] Docstring de `experiment_t0_sweep_221.py` documenta que es variante mode-221
- [ ] Ningun nuevo test o script importa de `experiment_t0_sweep.py`
- [ ] Tests existentes siguen pasando

---

# 5. Secuencia de ejecucion recomendada

## Paso 1: Fix bugs criticos s4g→s4i (P0)
- **Objetivo:** Restaurar integridad del pipeline golden geometry
- **Archivos afectados:**
  - `mvp/s4i_common_geometry_intersection.py` — cambiar `S4G_OUTPUT_REL` y key de lectura
  - `mvp/s4g_mode220_geometry_filter.py` — normalizar key `"accepted_geometry_ids"` → `"geometry_ids"` (si se elige esta opcion)
- **Precondicion:** Ninguna
- **Cambio:** 2-3 lineas
- **Validacion:** Nuevo test de integracion s4g→s4i con fixture de output real

## Paso 2: Anadir contratos faltantes a CONTRACTS dict (P1)
- **Objetivo:** SSOT contractual completo para golden geometry pipeline
- **Archivos afectados:** `mvp/contracts.py`
- **Precondicion:** Paso 1 completado (filenames y keys correctos para definir contratos precisos)
- **Cambio:** ~45 lineas nuevas en CONTRACTS dict (3 StageContracts + fix s4g external_inputs)
- **Validacion:** `test_contracts_completeness.py` pasa limpio; nuevo test verifica 4/4 golden stages

## Paso 3: Migrar s4g a lifecycle contractual (P1)
- **Objetivo:** Primer stage golden con init_stage/finalize
- **Archivos afectados:** `mvp/s4g_mode220_geometry_filter.py`
- **Precondicion:** Paso 2 (contrato declarado)
- **Cambio:** Refactor de `main()` para usar `init_stage → check_inputs → finalize`. ~40 lineas cambiadas, logica de negocio intacta
- **Validacion:** `test_s4g_mode220_geometry_filter.py` pasa; output identico

## Paso 4: Migrar s4h a lifecycle contractual (P1)
- **Objetivo:** Stage 221 con semantica de skip contractual
- **Archivos afectados:** `mvp/s4h_mode221_geometry_filter.py`
- **Precondicion:** Paso 2
- **Cambio:** Similar a paso 3, con atencion especial a la ruta de skip (obs file ausente → finalize con verdict=SKIPPED)
- **Validacion:** Nuevo test happy path + skip path

## Paso 5: Migrar s4i a lifecycle contractual (P1)
- **Objetivo:** Stage interseccion con lifecycle
- **Archivos afectados:** `mvp/s4i_common_geometry_intersection.py`
- **Precondicion:** Pasos 1 + 2 (bugs fixeados, contrato definido)
- **Cambio:** Refactor de main()
- **Validacion:** Test e2e s4g→s4i

## Paso 6: Migrar s4j a lifecycle contractual (P1)
- **Objetivo:** Ultimo stage golden con lifecycle
- **Archivos afectados:** `mvp/s4j_hawking_area_filter.py`
- **Precondicion:** Paso 2
- **Cambio:** Refactor de main()
- **Validacion:** Test e2e completo s4g→s4h→s4i→s4j

## Paso 7: Deprecar experiment_t0_sweep.py (P2)
- **Objetivo:** Eliminar confusion sobre cual t0_sweep usar
- **Archivos afectados:** `mvp/experiment_t0_sweep.py`
- **Precondicion:** Ninguna (independiente de pasos 1-6)
- **Cambio:** Anadir DeprecationWarning + actualizar docstring
- **Validacion:** Test de warning; tests existentes pasan

## Paso 8: Documentar experiment_t0_sweep_221.py (P2)
- **Objetivo:** Aclarar status de variante mode-221
- **Archivos afectados:** `mvp/experiment_t0_sweep_221.py`
- **Precondicion:** Ninguna
- **Cambio:** Solo docstring
- **Validacion:** N/A

## Paso 9: Remediaciones P3 (deprecaciones menores)
- **Objetivo:** Limpiar H-03, H-06, H-07, H-08
- **Archivos afectados:** tests duplicados, s6_geometry_table.py, legacy analysis, download_gw_events.py
- **Precondicion:** Pasos 1-8 completados
- **Cambio:** Headers DEPRECATED, fusiones de test, git mv
- **Validacion:** Misma cuenta de tests PASS

## Paso 10: Consolidacion de tests P4 (fase posterior)
- **Objetivo:** Reducir fragmentacion de tests (H-04, H-05, H-10, H-11, H-13)
- **Archivos afectados:** ~30 ficheros de test
- **Precondicion:** Pasos 1-9 completados; pipeline estable
- **Cambio:** Reorganizacion sin cambio de assertions
- **Validacion:** Misma cuenta de tests PASS

---

# 6. Riesgos y supuestos

## Riesgos reales

| # | Riesgo | Probabilidad | Impacto | Mitigacion |
|---|--------|-------------|---------|-----------|
| R1 | La migracion de s4h a lifecycle rompe la semantica de skip (obs file ausente no debe ser fatal) | Media | Alto | Implementar skip como `finalize(ctx, artifacts, verdict="SKIPPED")` DESPUES de init_stage, no como abort. Testear explicitamente. |
| R2 | Cambiar `"accepted_geometry_ids"` → `"geometry_ids"` en s4g rompe algun consumer no detectado | Baja | Medio | Grep exhaustivo por `"accepted_geometry_ids"` antes de cambiar. Si hay consumer, usar fallback dual temporalmente. |
| R3 | La consolidacion de tests (P4) introduce regresiones por assertions perdidas en el merge | Media | Medio | Comparar cuenta de tests PASS antes/despues. No cambiar assertions. |
| R4 | `experiment_single_event_golden_robustness.py` depende implicitamente del formato de output de s4g | Baja | Bajo | Verificado: usa funciones puras in-memory (filter_mode220, filter_mode221), no lee ficheros de stages. No afectado. |

## Supuestos

| # | Supuesto | Impacto si falso |
|---|----------|-----------------|
| S1 | El pipeline golden geometry (s4g→s4j) no se ha ejecutado exitosamente end-to-end en produccion (dado los bugs H-02a/H-02b) | Si se ha ejecutado exitosamente, hay algun workaround no documentado o un orchestrador que pasa datos de otra forma. Verificar con el equipo. |
| S2 | `experiment_t0_sweep.py` no tiene usuarios externos activos | Si tiene usuarios, la deprecation warning les dara tiempo para migrar antes de borrar el fichero. |
| S3 | Los tests existentes de s4g prueban la estructura del output pero no la integracion con s4i | Si testean integracion, el fix de H-02b podria hacer fallar un test que esperaba el filename antiguo. Verificar antes de patchear. |
| S4 | `_full.py` (1918 LOC) no se migrara a contracts.py en esta fase porque el riesgo es alto | Si el equipo quiere migrarlo, hacerlo en fase independiente con tests de equivalencia exhaustivos. |

---

# 7. Que NO tocar todavia

| Ruta/Subsistema | Razon |
|-----------------|-------|
| `mvp/pipeline.py` | Orquestador core. Sin bugs. No integrar golden geometry aqui hasta que s4g-s4j esten contractualizados. |
| `mvp/s0_oracle_mvp.py` .. `mvp/s7_beyond_kerr_deviation_score.py` (core MVP) | Core bien disenado, contractual, sin hallazgos. No tocar. |
| `malda/` (todo el directorio) | Pipeline independiente, autocontenido, sin conflicto con mvp/. No genera fragmentacion. |
| `mvp/experiment_t0_sweep_full.py` | Canonico. 1918 LOC. La migracion a contracts.py es alto riesgo/bajo beneficio en esta fase. |
| `mvp/s4c_kerr_consistency.py` | Trivial pero contractual. No absorber en s4d hasta fase posterior. |
| `mvp/experiment_dual_method.py` | Gate de QC. Funciona. Reclasificacion puede esperar. |
| `mvp/schemas.py` | Libreria de validacion. Sin hallazgos. No consolidar APIs legacy/canonical hasta cerrar H-03 (tests). |
| `basurin_io.py` | Framework IO. Bien disenado. No modificar. |
| `config/`, `docs/ringdown/`, atlas JSONs | Datos de referencia inmutables. No tocar. |
| `tests/golden/` (si existe), golden snapshots en tests | Ultima linea de defensa contra regresiones. No recalcular salvo que sea consecuencia directa de un fix. |
| Tests de contratos, oracle, pipeline (H-04, H-05, H-10, H-11, H-13) | Consolidacion de tests es P4. Hacerla prematuramente introduce riesgo sin beneficio funcional. |
| `mvp/experiment_population_kerr.py`, `mvp/experiment_injection_suite.py`, `mvp/experiment_t6_rd_weighted.py` | Experimentos sin contrato pero funcionales. No son stages canonicos. Documentar en fase posterior. |
| `tools/` (todo el directorio) | Herramientas de descarga/CI. Sin hallazgos criticos. |
| `scratch/` | 90 bytes de conveniencia. No merece atencion. |
