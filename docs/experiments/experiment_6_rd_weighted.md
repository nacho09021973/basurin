# Experimento 6 (T6) — RD-weighted (incl. variante fullrank)

## 1) Propósito

T6 extiende el experimento base de “area theorem / ΔA-only” para construir un observable **RD-weighted** (ringdown-weighted) que:
- **Pondera** cada geometría compatible por evidencia relativa `w ∝ exp(delta_lnL)` (con estabilización numérica).
- Reporta **ESS (Effective Sample Size)** para auditar si la ponderación es informativa o degenerada.

Qué problema resuelve:
- Evita “sobre-interpretar” candidatos marginales donde la intersección multimodo es vacía o la masa efectiva de pesos colapsa.
- Separa explícitamente:
  - **claims** (cuando hay soporte + ESS suficiente),
  - **stress tests** (cuando hay `AF_EMPTY`, ESS bajo o intersección vacía bajo `phys_key` exacto).

> Gobernanza BASURIN: este experimento no modifica stages canónicos ni contracts; solo consume artefactos ya emitidos bajo `runs/<run_id>/...` y escribe sus outputs bajo `runs/<T6_RUN_ID>/experiment/<OUT_NAME>/...`.

---

## 2) Inputs (rutas exactas)

### 2.1 Run T6 base (ΔA-only)
Fuente de verdad por evento (ΔA-only):

- `runs/<T6_RUN_ID>/experiment/area_theorem/outputs/per_event_spinmag.csv`

### 2.2 Batches ringdown (offline_batch)
Resultados batch por modo (contienen `event_id` y el `run_id` del subrun por evento):

- `runs/<BATCH_220>/experiment/offline_batch/outputs/results.csv`
- `runs/<BATCH_221>/experiment/offline_batch/outputs/results.csv`

> Ver guía batch/t0: `docs/readme_experiment_4.md` (rutas, auditoría y semántica de `results.csv`).

### 2.3 Evidencia por evento (subruns mapeados desde results.csv)
Para cada fila de `results.csv`, se resuelve el `run_id` del evento y se leen:

- `runs/<SUBRUN>/s4_geometry_filter/outputs/compatible_set.json`
- `runs/<SUBRUN>/s4_geometry_filter/outputs/ranked_all_full.json` (si existe)
  - fallback 1: `runs/<SUBRUN>/s4_geometry_filter/outputs/ranked_all_full.csv` (si existe)
  - fallback 2: campo embebido `ranked_all` dentro de `compatible_set.json` (top-N según `ranked_all_limit`, top50 por defecto)

---

## 3) Cómo se construye RD-weighted (join exacto)

### 3.1 Join de pesos (geometría)
Se enriquece cada geometría compatible con un peso procedente del ranking (si existe):

- Join key:
  - `ranked_all_full[*].geometry_id -> compatible_geometries[*].geometry_id`
  - fallback: `compatible_set.ranked_all[*].geometry_id -> compatible_geometries[*].geometry_id`

### 3.2 Intersección multi-modo (igualdad exacta por phys_key)
Definimos el identificador físico exacto:

- `phys_key = (family, source, M_solar, chi)`  *(igualdad exacta, sin tolerancias)*

Construcción:
1. Para cada modo (220 / 221), se mapea cada geometría compatible a su `phys_key`.
2. Se toma la **intersección exacta** de `phys_key` entre modos.

### 3.3 Reducción por phys_key (max_delta_per_phys)
Si múltiples geometrías mapean al mismo `phys_key`, se reduce por:

- `max_delta_per_phys`: conservar el máximo `delta_lnL` por `phys_key` (por modo).

### 3.4 Combinación 220 + 221
Para cada `phys_key` en la intersección:

- `delta_combined = delta220 + delta221`

### 3.5 Pesos (estabilidad numérica)
Para evitar overflow:

- `w_i ∝ exp(delta_combined_i - max(delta_combined))`

Normalización (si se requiere) por `sum(w)`.

### 3.6 ESS (Effective Sample Size)
Auditoría de degeneración de pesos:

- `ESS = (sum_i w_i)^2 / sum_i (w_i^2)`

Interpretación:
- ESS ~ N: pesos casi uniformes (poca “información” diferencial).
- ESS ~ 1: colapso en un único `phys_key` (alta concentración; revisar estabilidad).

---

## 4) Outputs (rutas exactas y esquema)

### 4.1 CSV por evento
- `runs/<T6_RUN_ID>/experiment/<OUT_NAME>/outputs/per_event_spinmag_rd_weighted.csv`

Columnas mínimas:
- `event_id`
- `status`
- `p_violate`
- `dA_p10,dA_p50,dA_p90`
- `n_mc`
- `af_rd_p10,af_rd_p50,af_rd_p90` *(si aplica)*
- `ess_rd` *(si aplica)*

### 4.2 Summary JSON (auditabilidad)
- `runs/<T6_RUN_ID>/experiment/<OUT_NAME>/outputs/summary.json`

Esquema mínimo:
- `per_event`: array de objetos con:
  - `event_id`
  - `status`
  - `policy`
  - `ess`
  - `n_compat_220`, `n_compat_221`
  - `n_ranked_all_total_220`, `n_ranked_all_total_221`
  - `n_intersection`
  - `n_support_phys`, `n_used`
  - `weight_source_220`, `weight_source_221`
  - `join_policy`
  - `weight_reduce`

### 4.3 Manifiestos (hashes)
Como cualquier experimento gobernado:
- `runs/<T6_RUN_ID>/experiment/<OUT_NAME>/manifest.json`
- `runs/<T6_RUN_ID>/experiment/<OUT_NAME>/stage_summary.json`

Ambos deben incluir hashes SHA256 de outputs.

---

## 5) Política de calidad (claims vs stress tests)

### 5.1 Policy recomendada
- `CONSERVATIVE_SKIP` cuando:
  - `status == AF_EMPTY` (intersección 220∩221 vacía bajo `phys_key` exacto), o
  - `ess` bajo un umbral operativo (p.ej. ESS < 5 o ESS << n_support_phys), o
  - faltan weights (no existe ranked_all y no hay fallback aceptado)

Semántica:
- En `CONSERVATIVE_SKIP` **no se hace claim** de `Af_RD` combinado; se reporta como **stress test**.

### 5.2 Ejemplo auditado: GW200322_091133 (caso publicable)
Caso observado:
- Incluso con `ranked_all_full` (N grande), la intersección exacta por `phys_key` resulta vacía → `AF_EMPTY`.
- Esto es una propiedad del **match exacto multimodo** (no del recorte top50).

Ejemplo de bloque (campos clave; no pegar archivos completos):

```json
{
  "event_id": "GW200322_091133",
  "status": "AF_EMPTY",
  "policy": "CONSERVATIVE_SKIP",
  "ess": 0.0,
  "n_compat_220": 156,
  "n_compat_221": 730,
  "n_intersection": 0,
  "join_policy": "ranked_all.geometry_id -> compatible_geometries.geometry_id; reduce=max_delta_per_phys; combine=delta220+delta221"
}
```

Interpretación:

- `n_compat_220/221` altos no implican compatibilidad multimodo si `phys_key` exacto no intersecta.
- Este evento debe tratarse como stress test salvo que se justifique una relajación explícita del `phys_key` (fuera del alcance de T6 doc).

---

## 6) Comandos reproducibles (mínimo)

Ejecución (ejemplo):

```bash
python -m mvp.experiment_t6_rd_weighted \
  --run-id <T6_RUN_ID> \
  --in-per-event runs/<T6_RUN_ID>/experiment/area_theorem/outputs/per_event_spinmag.csv \
  --out-name <OUT_NAME> \
  --batch-220 <BATCH_220> \
  --batch-221 <BATCH_221>
```

Extraer GW200322 del summary:

```bash
jq -c '.per_event[] | select(.event_id=="GW200322_091133")' \
  runs/<T6_RUN_ID>/experiment/<OUT_NAME>/outputs/summary.json
```

---

## 7) Troubleshooting (concreto)

- `RUN_VALID verdict not found`

  - Diagnóstico: el run/batch/subrun no tiene `runs/<run_id>/RUN_VALID/verdict.json`.
  - Acción: bootstrap del run (o corregir `BASURIN_RUNS_ROOT` si era un subrun). Ver `docs/readme_rutas.md`.

- `AF_EMPTY`

  - Diagnóstico: intersección multimodo `phys_key` exacta vacía → no existe `Af_RD` combinado.
  - Acción: reportar como stress test (`policy=CONSERVATIVE_SKIP`). No inventar `Af_RD`.

- `Atlas not found`

  - Ver sección “Atlas (geometrías)” en `docs/readme_rutas.md` (rutas canónicas y comandos `find`).
