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

## Estado actual (2026-03-16)

### Artefactos históricos obsoletos

- `analysis_area_entropy_20260304T162929Z`
- `analysis_area_weighted_20260304T163248Z`
- `analysis_area_llrel_20260304T163551Z`
- `analysis_area_theorem_20260304T173747Z`

Motivo:

- dependían de batch runs antiguos ya retirados a cuarentena

### Veredicto actual

E5 **es rehacible**, porque con los batches nuevos:

- `K220_inter_K221 = 800 > 0`
- `n_empty_intersection_events = 0`

Pero E5 **no debe reinterpretarse con la semántica histórica sin revisión**, porque:

- la hipótesis antigua `K220 ⊂ K221` ya no se cumple
- ahora el soporte común parece alinearse con `K221 ⊂ K220`

---

## Gate de salida

La fase 4 solo puede cerrarse cuando exista una implementación reproducible que:

- consuma los batch runs canónicos actuales
- use la proyección `phys_key` correcta
- excluya eventos/filas no válidos
- escriba artefactos bajo `runs/<analysis_run_id>/experiment/...`
- documente explícitamente el soporte común usado

A día de hoy, E5 está **habilitada**, pero **todavía no está reruneada canónicamente**.

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
