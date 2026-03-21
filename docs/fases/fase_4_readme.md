# Fase 4 — E5: área del horizonte y entropía en soporte físico común

## Objetivo

Construir derivados físicos de horizonte (`A`, `S=A/4`) sobre una malla física discreta compartida entre modos 220 y 221.

Esta fase corresponde al experimento E5 y debe interpretarse como:

- consistencia de derivados `A,S` en soporte físico común
- no como inferencia bayesiana final
- no como prueba independiente de termodinámica de agujeros negros

---

## Definición conceptual

Según la documentación existente, E5 se construye así:

1. partir de compatibles por evento
2. proyectar a una clave física común entre modos
3. construir una intersección física `220∩221`
4. calcular derivados:
   - `A = 8π M^2 (1 + sqrt(1-chi^2))`
   - `S = A/4`
5. resumir con cuantiles y métricas agregadas

---

## Definición de soporte común válida hoy

No usar `geometry_id`.

Usar:

- `phys_key = (family, provenance, M_solar, chi)`
- `provenance = metadata.source if existe; si no, metadata.ref`
- `M_solar = round(metadata.M_solar, 6)`
- `chi = round(metadata.chi, 6)`

---

## Entradas canónicas

- batch 220 canónico
- batch 221 canónico
- `compatible_set.json` de runs por evento en `PASS`
- gate de fase 3 basado en intersección física

---

## Estado actual (2026-03-16, referencia histórica)

### Artefactos históricos obsoletos

- `analysis_area_entropy_20260304T162929Z`
- `analysis_area_weighted_20260304T163248Z`
- `analysis_area_llrel_20260304T163551Z`
- `analysis_area_theorem_20260304T173747Z`

Motivo:

- dependían de batch runs antiguos ya retirados a cuarentena

---

## Estado canónico auditado (2026-03-20)

E5 ha sido ejecutada canónicamente sobre los batches del 2026-03-20.

### Corrida canónica

- **Batch 220:** `batch_with_t0_220_eps2500_20260320T113433Z`
- **Batch 221:** `batch_with_t0_221_eps2500_20260320T114207Z`
- **Host run:** `mvp_GW150914_221info_20260320T105521Z`

### Métricas observadas (artefactos auditados)

- `PASS_220 = 38`, `PASS_221 = 38`, `COMMON_PASS = 38`
- `n_empty_intersection_events = 0`
- `n_events_with_nonempty_hawking = 38`
- `n_rows_hawking_fail = 0`
- Para los 38 eventos: `n_k221 == n_k_common` y `n_k_common == n_k_hawking`

En términos de soporte:

```
K_common = K220 ∩ K221 = K221
K_hawking = K_common
```

### Conclusión soportada por los artefactos auditados

En la configuración canónica actual (epsilon=2500, cohorte de 38 eventos):

- El **221 es el discriminante efectivo** del soporte respecto a 220: `K_common = K221 ⊂ K220`.
- El filtro de Hawking **valida compatibilidad física** del soporte común, pero **no lo reduce adicionalmente**: `K_hawking = K_common`.

### Alcance y limitaciones explícitas

Estas conclusiones son **específicas de la configuración canónica actual** y NO deben generalizarse:

- No implica que "221 nunca discrimina espectralmente" bajo otras configuraciones o valores de epsilon.
- No implica que "Hawking nunca discrimina" en general.
- **"Compatible con Hawking"** (ninguna fila falla el test de área) **≠ "discriminado por Hawking"** (Hawking reduce el soporte).

En esta corrida, el soporte `K_common` ya resultante de `220 ∩ 221` es íntegramente compatible con la restricción de Hawking; por eso Hawking no añade recorte. Este resultado depende del epsilon=2500 y de la cohorte offline disponible. Una cohorte más amplia o un epsilon distinto podría producir resultados distintos.

### Artefactos canónicos de la corrida 2026-03-20

- `runs/mvp_GW150914_221info_20260320T105521Z/experiment/phase4_hawking_area_common_support/outputs/hawking_area_summary.json`
- `runs/mvp_GW150914_221info_20260320T105521Z/experiment/phase4_hawking_area_common_support/outputs/per_event_common_support.csv`

---

## Gate de salida

La fase 4 solo puede cerrarse cuando exista una implementación reproducible que:

- consuma los batch runs canónicos actuales
- use la proyección `phys_key` correcta
- excluya eventos/filas no válidos
- escriba artefactos bajo `runs/<analysis_run_id>/experiment/...`
- documente explícitamente el soporte común usado

A fecha 2026-03-20, E5 **está reruneada canónicamente** sobre los batches `eps2500` del 2026-03-20.

---

## Indicadores que hay que documentar siempre

- `batch_220`
- `batch_221`
- `K220`
- `K221`
- `K220_inter_K221`
- `n_empty_intersection_events`
- `n_non_subset_cases`
- definición exacta de `phys_key`
- número de eventos realmente usados

---

## Fallos típicos

- reutilizar outputs históricos de E5 sin revisar sus upstreams
- asumir que el soporte común se decide por `geometry_id`
- confundir verificación post-hoc de artefactos con receta de construcción
- interpretar `A,S` como observables primarios del ajuste

---

## Dependencia hacia fases posteriores

Cualquier fase posterior que consuma E5 debe declarar explícitamente:

- qué batches usa
- qué definición de soporte común usa
- si asume o no relaciones de inclusión entre 220 y 221
