# Backlog contract-first (mínimos cambios) para `informe_multimode_viability_v3`

## Requirements

| Nivel | Requisito | Trazabilidad informe | Impacto contractual mínimo |
|---|---|---|---|
| MUST | Separar responsabilidades: `multimode_viability` clasifica informatividad y **no** emite claim de nueva física. | §1(1), §7.2 | Mantener bloque de viabilidad como gate independiente en `stage_summary` de `s4d_kerr_from_multimode`. |
| MUST | Implementar clasificaciones exclusivas `MULTIMODE_OK` / `SINGLEMODE_ONLY` / `RINGDOWN_NONINFORMATIVE`. | §1(1), §6, §7.2 | Campo obligatorio `multimode_viability.class` + razones/flags/umbrales. |
| MUST | `science_evidence` debe existir siempre, incluso cuando no se evalúa (`NOT_EVALUATED`). | §1(4), §9.1, §11.1 | Campo obligatorio `science_evidence` con `status` y `reason_if_skipped`. |
| MUST | `science_evidence` solo se evalúa si `multimode_viability.class == MULTIMODE_OK` **y** `systematics_gate.verdict_final == PASS`. | §9.1 | Regla de gating explícita en summary y tests de no-evaluación. |
| MUST | `systematics_gate` determinista con checks `t0_plateau`, `psd_sanity`, `estimator_resolution`. | §1(5), §8.2, §8.3 | Nuevo bloque obligatorio `systematics_gate` con checks y umbrales usados. |
| MUST | Override humano (`annotations/systematics_override.json`) solo puede degradar (nunca promover). | §1(6), §8.5 | Registrar `verdict_auto`, `verdict_final`, referencia y hash del override. |
| MUST | Umbrales usados deben quedar explícitos en artefactos. | §1(8), §7.2, §8.4 | Persistir `thresholds_used` en viabilidad y sistemáticas. |
| MUST | Contrato final debe incluir rutas de evidencia upstream usadas por sistemáticas (`t0_sweep_ref`, `psd_ref`) y slots futuros `null`. | §11.1 | Añadir referencias de inputs y `future_slots` en `science_evidence`. |
| SHOULD | Usar `delta_bic` como señal de severidad, no gate único ni evidencia bayesiana. | §1(7), §7.2 | Métrica opcional en `multimode_viability.metrics` con flag severo. |
| SHOULD | Publicar `outputs/kerr_ratio_reference.json` como golden determinista. | §11.2 | Declarar output adicional de `s4d_kerr_from_multimode` (si no existe ya en runtime). |
| SHOULD | Incluir diagnóstico de consistencia Kerr por solape de intervalo observado con banda Kerr. | §5.1–§5.3, §7.2 | Guardar `kerr_consistent`, fracciones de solape e informatividad en métricas. |
| SHOULD | Tratar derivada baja a spins pequeños como diagnóstico (no gate duro). | §10 (F6) | Campo diagnóstico opcional, sin afectar clase final. |
| MAY | Preparar slots futuros `delta_f_221`, `delta_tau_221`, `log_bayes_factor` en `null`. | §4.3, §9.3, §11.1 | Reserva de esquema sin cálculo downstream. |
| MAY | Permitir checks `NA` cuando no hay insumos de sistemáticas y emitir `NOT_AVAILABLE`. | §8.3 | Evitar abort duro por ausencia de t0/PSD, pero bloquear `science_evidence`. |

## Minimal diffs plan (mvp/contracts.py)

1. **Actualizar contrato de `s4d_kerr_from_multimode`** para declarar un output nuevo `outputs/kerr_ratio_reference.json` (golden file) además de los dos outputs actuales. No crear stage nuevo.
2. **Mantener upstreams mínimos de `s4d_kerr_from_multimode`** en `s3b_multimode_estimates/{multimode_estimates.json,stage_summary.json}` y añadir solo como *referencias en summary* (no dependencia rígida en `contracts.py`) los insumos de `t0` y `psd` para respetar disponibilidad parcial.
3. **No endurecer dependencias cross-stage en contrato** para `oracle_t0_ringdown` o `s6c_brunete_psd_curvature`; registrarlas como `inputs refs` en `stage_summary` y resolverlas runtime como opcionales (permite `NOT_AVAILABLE`).
4. **Extender esquema esperado de summary (en validación de stage)** con tres bloques obligatorios: `multimode_viability`, `systematics_gate`, `science_evidence` + `annotations`.
5. **Exigir hashes SHA256 en manifest/summary** de outputs nuevos (`kerr_ratio_reference.json`) y de override humano cuando exista (hash por referencia de archivo de anotación).

## Orden de implementación (6 pasos)

### Paso 1 — Congelar especificación de esquema (`s4d` summary)
**Acceptance criteria**
- Existe especificación explícita (keys obligatorias/tipos) para `multimode_viability`, `systematics_gate`, `science_evidence`, `annotations`.
- `science_evidence` se define presente siempre.

**Regression tests**
- Unit de contrato: valida presencia/ausencia de claves obligatorias en `stage_summary` de `s4d`.
- Golden de esquema JSON normalizado.

### Paso 2 — Ajustar contrato de outputs en `mvp/contracts.py`
**Acceptance criteria**
- `CONTRACTS["s4d_kerr_from_multimode"].produced_outputs` incluye `outputs/kerr_ratio_reference.json`.
- No se crean stages nuevos.

**Regression tests**
- Unit test de contrato de stage (`s4d`) verificando outputs esperados.

### Paso 3 — Implementar gate de viabilidad puro y determinista
**Acceptance criteria**
- Función pura devuelve una sola clase de viabilidad y `thresholds_used`.
- Mismos inputs producen salida idéntica byte a byte.

**Regression tests**
- Set de 6 tests de clasificación (casos good/broad/bad220/inconsistent/spin-floor/determinism).

### Paso 4 — Implementar `systematics_gate` con combinación auto+humano
**Acceptance criteria**
- `verdict_auto` ∈ {PASS, FAIL, NOT_AVAILABLE}.
- Regla "override solo degrada" se cumple en todas las combinaciones.

**Regression tests**
- Tabla de verdad completa para `_combine_verdicts`.
- Tests con checks NA parciales.

### Paso 5 — Implementar `science_evidence` (H1 mínima `delta_Rf`)
**Acceptance criteria**
- `status=EVALUATED` solo cuando pasan viabilidad y sistemáticas.
- En cualquier otro caso `status=NOT_EVALUATED` + `reason_if_skipped` no vacío.
- `future_slots` presentes en `null`.

**Regression tests**
- Tests de gating cruzado (viabilidad/systematics).
- Test de `contains_zero` coherente con intervalo de `delta_Rf`.

### Paso 6 — Integración contract-first de IO y gobernanza
**Acceptance criteria**
- El stage sigue escribiendo únicamente bajo `runs/<run_id>/...` y registra `manifest/stage_summary` con hashes.
- Si faltan inputs obligatorios del stage, aborta con mensaje accionable; si faltan inputs opcionales de sistemáticas, degrada a `NOT_AVAILABLE`.

**Regression tests**
- Integration-lite con `BASURIN_RUNS_ROOT` temporal verificando cero escrituras fuera de raíz.
- Test de abort por upstream obligatorio faltante y test warning-only para insumo opcional faltante.

## Riesgos de gobernanza (abort vs warning-only)

### Debe abortar el run (fail-fast)
- Falta de inputs obligatorios del contrato de stage (`s4d`): `multimode_estimates.json` o `stage_summary.json` de `s3b`.
- Error de serialización de artefactos contractuales (`manifest.json` / `stage_summary.json`) o hash inconsistente.
- `RUN_VALID != PASS` cuando aplica gating upstream del pipeline.

### Debe ser warning-only (sin claim científico)
- Ausencia de insumos opcionales para `systematics_gate` (p.ej. no hay referencia de `t0` o `psd`): `verdict_auto=NOT_AVAILABLE`, `science_evidence=NOT_EVALUATED`.
- Presencia de override humano redundante (`PASS` sobre `PASS`): se registra y no cambia veredicto.
- Señales diagnósticas no críticas (p.ej. degeneración de spin bajo/F6): quedan en métricas, no abortan stage.
