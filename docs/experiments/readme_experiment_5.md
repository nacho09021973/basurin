# Experimento 5 (E5): Área de horizonte y entropía `S=A/4` desde ringdown multimodo (220/221)

## 1) Propósito y alcance

E5 documenta un flujo **contract-first y auditable** para:

1. generar batches offline-first para modos `(2,2,0)` y `(2,2,1)` con `eps=2500`,
2. comparar su compatibilidad física por geometría,
3. construir la intersección física `220∩221`,
4. derivar área de horizonte `A` y entropía de Bekenstein-Hawking `S=A/4`,
5. rankear eventos por “amplitud 221” y “severidad 220”,
6. evaluar ponderaciones y su poder de discriminación.

**Qué sí mide E5:** consistencia interna de compatibles en el atlas y estabilidad de métricas derivadas de `(M,chi)` bajo soporte común discreto.

**Qué NO mide E5:**

- No es una inferencia bayesiana completa de parámetros astrofísicos.
- No es un test independiente de Hawking.
- No prueba teoría más allá de las hipótesis de modelo usadas para construir compatibles (Kerr/GR y familias del atlas).

---

## 2) Inputs y pre-requisitos

### 2.1 Catálogo `t0` y lista de eventos con `t0`

- `runs/audit_gwosc_t0_20260304T115440Z/experiment/losc_quality/t0_catalog_gwosc_v2.json`
- `runs/prep_events_with_t0_20260304T153418Z/external_inputs/events_with_t0.txt`

### 2.2 Atlas y filtro de modo

- Atlas: `--atlas-default`.
- Modo 220: `--mode-filter 220`.
- Modo 221: `--mode-filter 221`.

### 2.3 Política de ejecución

- Offline-first con catálogo `t0` explícito.
- IO determinista bajo `runs/<run_id>/...`.
- Auditoría por `manifest.json`, `stage_summary.json` y `outputs/*` en cada experimento/stage.

---

## 3) Ejecución (comandos exactos, `eps=2500`)

> Estos comandos reconstruyen los batches canónicos de E5 con IDs explícitos.

### 3.1 Batch 220

```bash
python -m mvp.experiment_offline_batch \
  --batch-run-id batch_with_t0_220_eps2500_fixlen_20260304T160054Z \
  --window-catalog runs/audit_gwosc_t0_20260304T115440Z/experiment/losc_quality/t0_catalog_gwosc_v2.json \
  --events-file runs/prep_events_with_t0_20260304T153418Z/external_inputs/events_with_t0.txt \
  --atlas-default \
  --mode-filter 220 \
  --eps 2500
```

Output principal:

- `runs/batch_with_t0_220_eps2500_fixlen_20260304T160054Z/experiment/offline_batch/outputs/results.csv`

Resumen canónico observado:

- `events=44`
- `n_zero=2` (`GW190521`, `GW191113_071753`)
- `total compatibles (sum)=6552`
- `unique phys_keys total=123`
- Conteo bruto `(family,source)`: `kerr/berti_2009_fit=5166`, `dcs=588`, `edgb=462`, `kerr_newman=336`
- `unique phys_keys by (family,source)`: solo `kerr/berti_2009_fit=123`

### 3.2 Batch 221

```bash
python -m mvp.experiment_offline_batch \
  --batch-run-id batch_with_t0_221_eps2500_fixlen_20260304T160617Z \
  --window-catalog runs/audit_gwosc_t0_20260304T115440Z/experiment/losc_quality/t0_catalog_gwosc_v2.json \
  --events-file runs/prep_events_with_t0_20260304T153418Z/external_inputs/events_with_t0.txt \
  --atlas-default \
  --mode-filter 221 \
  --eps 2500
```

Output principal:

- `runs/batch_with_t0_221_eps2500_fixlen_20260304T160617Z/experiment/offline_batch/outputs/results.csv`

Resumen canónico observado:

- `events=44`
- `n_zero=0`
- `total compatibles (sum)=31398`
- `unique phys_keys total=730`
- Todo en `kerr/berti_2009_fit`

---

## 4) Auditoría de runs

Para cada run, revisar en este orden:

1. `runs/<run_id>/RUN_VALID/verdict.json`
2. `runs/<run_id>/experiment/<name>/stage_summary.json`
3. `runs/<run_id>/experiment/<name>/manifest.json`
4. `runs/<run_id>/experiment/<name>/outputs/*`

Detección explícita de eventos con `n_compatible=0` (batch 220):

```bash
python - <<'PY'
import csv
p='runs/batch_with_t0_220_eps2500_fixlen_20260304T160054Z/experiment/offline_batch/outputs/results.csv'
with open(p,newline='') as f:
    rows=list(csv.DictReader(f))
zero=[r['event_id'] for r in rows if int(r['n_compatible'])==0]
print('n_zero=',len(zero))
print('zero_events=',zero)
PY
```

Esperado: `n_zero=2` con `GW190521` y `GW191113_071753`.

---

## 5) Intersección física `220∩221`

### 5.1 Por qué `geometry_id` no intersecta

Una intersección literal por `geometry_id` entre batches 220 y 221 da 0 porque el identificador incluye el modo (`n`) y, por construcción, un batch está filtrado en `l2m2n0` y el otro en `l2m2n1`.

### 5.2 Definición de intersección física

Se define:

- `phys_key = (family, source, M_solar, chi)`

La comparación física entre modos usa `phys_key`, no `geometry_id`.

### 5.3 Resultado global y outliers

Run de análisis:

- `runs/analysis_common_220_221_20260304T161639Z/experiment/common_geometries/outputs/common_summary.json`

Resultados reportados:

- `|K220|=123`
- `|K221|=730`
- `|K220∩K221|=123`
- `empty intersection events`: `GW190521`, `GW191113_071753`
- En el resto: `K220 ⊂ K221` con `non_subset_cases=0`

Outliers de severidad 220 (`d2_min`):

- `GW191113_071753`: `d2_min(220)=5577.320318 > 2500`, `n_compatible(220)=0`, mientras `n_compatible(221)=91`
- `GW190521`: `d2_min(220)=2944.802322 > 2500`, `n_compatible(220)=0`, mientras `n_compatible(221)=647`

---

## 6) Área del horizonte y entropía

### 6.1 Fórmulas (unidades geométricas, `G=c=1`)

Para cada geometría compatible con parámetros `(M,chi)`:

- `A = 8π M^2 (1 + sqrt(1-chi^2))`
- `S = A/4`

`A` y `S` son **derivados** de `(M,chi)`; no son observables primarios del ajuste.

### 6.2 Cómo se computa en E5

1. Sobre compatibles por evento.
2. Sobre la intersección física `220∩221` para mantener soporte común.
3. Con resúmenes tipo cuantiles y métricas agregadas.

Run principal:

- `runs/analysis_area_entropy_20260304T162929Z/experiment/area_entropy/outputs/area_entropy_summary.json`

Sanity check documentado (evento `GW170817`, intersección):

- `A_p50≈245943.2435`
- `S_p50≈61485.81088`
- Se verifica `S=A/4`.

### 6.3 Nota de interpretación

Esto **no** es un test de Hawking independiente; es una transformación algebraica bajo hipótesis de Kerr/GR y geometrías compatibles del atlas.

---

## 7) Rankings

Run de ranking:

- `runs/analysis_rank_221_amp_220_sev_20260304T162208Z/experiment/common_geometries/outputs/`

Artefactos:

- `ranking_events.csv`
- `ranking_by_k221.txt`
- `ranking_by_d2min220.txt`

Definiciones:

- **Amplitud 221**: ranking por `k221` (proxy de presencia/intensidad relativa de 221 en el análisis de compatibilidad).
- **Severidad 220**: ranking por `d2_min(220)` (más alto = más lejos del umbral de compatibilidad en 220).

Observación clave: los dos outliers más severos en 220 son precisamente los dos con intersección vacía.

---

## 8) Ponderación y limitaciones

Runs de ponderación:

- `runs/analysis_area_weighted_20260304T163248Z/experiment/area_entropy/outputs/area_weighted_intersection.json`
- `runs/analysis_area_llrel_20260304T163551Z/experiment/area_entropy/outputs/area_llrel_weighted_220.json`
- `runs/analysis_area_EA_20260304T164052Z/experiment/area_entropy/outputs/area_EA_sigma_220.csv`

Resultados:

1. `posterior_weight` existe y varía dentro de evento.
2. Cuantiles ponderados (incluyendo `exp(log_likelihood_rel-max)`) resultan prácticamente idénticos a los no ponderados.
3. Razón: soporte discreto común (misma malla/intersección) + cuantización, por lo que la reponderación no mueve percentiles de forma visible.
4. `EA/sigmaA` ponderados sí cambian, pero con spans minúsculos:
   - `EA span ≈ 0.002587`
   - `sigmaA span ≈ 0.0003296`

Conclusión: discriminación extremadamente débil (orden relativo ~`1e-9`).

---

## 9) Artefactos producidos (rutas exactas bajo `runs/`)

- `runs/audit_gwosc_t0_20260304T115440Z/experiment/losc_quality/t0_catalog_gwosc_v2.json`
- `runs/prep_events_with_t0_20260304T153418Z/external_inputs/events_with_t0.txt`
- `runs/batch_with_t0_220_eps2500_fixlen_20260304T160054Z/experiment/offline_batch/outputs/results.csv`
- `runs/batch_with_t0_221_eps2500_fixlen_20260304T160617Z/experiment/offline_batch/outputs/results.csv`
- `runs/analysis_common_220_221_20260304T161639Z/experiment/common_geometries/outputs/common_summary.json`
- `runs/analysis_rank_221_amp_220_sev_20260304T162208Z/experiment/common_geometries/outputs/ranking_events.csv`
- `runs/analysis_rank_221_amp_220_sev_20260304T162208Z/experiment/common_geometries/outputs/ranking_by_k221.txt`
- `runs/analysis_rank_221_amp_220_sev_20260304T162208Z/experiment/common_geometries/outputs/ranking_by_d2min220.txt`
- `runs/analysis_area_entropy_20260304T162929Z/experiment/area_entropy/outputs/area_entropy_summary.json`
- `runs/analysis_area_weighted_20260304T163248Z/experiment/area_entropy/outputs/area_weighted_intersection.json`
- `runs/analysis_area_llrel_20260304T163551Z/experiment/area_entropy/outputs/area_llrel_weighted_220.json`
- `runs/analysis_area_EA_20260304T164052Z/experiment/area_entropy/outputs/area_EA_sigma_220.csv`

---

## 10) Checklist de reproducibilidad (comandos de verificación)

### 10.1 Conteos básicos de compatibles

```bash
python - <<'PY'
import csv
p='runs/batch_with_t0_220_eps2500_fixlen_20260304T160054Z/experiment/offline_batch/outputs/results.csv'
with open(p,newline='') as f:
    rows=list(csv.DictReader(f))
print('events=',len(rows))
print('sum_n_compatible=',sum(int(r['n_compatible']) for r in rows))
print('n_zero=',sum(1 for r in rows if int(r['n_compatible'])==0))
PY
```

```bash
python - <<'PY'
import csv
p='runs/batch_with_t0_221_eps2500_fixlen_20260304T160617Z/experiment/offline_batch/outputs/results.csv'
with open(p,newline='') as f:
    rows=list(csv.DictReader(f))
print('events=',len(rows))
print('sum_n_compatible=',sum(int(r['n_compatible']) for r in rows))
print('n_zero=',sum(1 for r in rows if int(r['n_compatible'])==0))
PY
```

### 10.2 Intersección física global

```bash
python - <<'PY'
import json
p='runs/analysis_common_220_221_20260304T161639Z/experiment/common_geometries/outputs/common_summary.json'
with open(p) as f:
    x=json.load(f)
print('|K220|=',x['global']['k220_unique_phys'])
print('|K221|=',x['global']['k221_unique_phys'])
print('|K220∩K221|=',x['global']['k_intersection_phys'])
print('non_subset_cases=',x['global']['non_subset_cases'])
print('empty_intersection_events=',x['global']['empty_intersection_events'])
PY
```

### 10.3 Outliers por `d2_min(220)` y compatibilidad 221

```bash
python - <<'PY'
import csv
p='runs/analysis_rank_221_amp_220_sev_20260304T162208Z/experiment/common_geometries/outputs/ranking_events.csv'
keys=['event_id','d2_min_220','n_compatible_220','n_compatible_221']
with open(p,newline='') as f:
    rows=list(csv.DictReader(f))
for ev in ['GW191113_071753','GW190521']:
    r=next(z for z in rows if z['event_id']==ev)
    print({k:r[k] for k in keys})
PY
```

### 10.4 Verificación `S=A/4`

```bash
python - <<'PY'
import json
p='runs/analysis_area_entropy_20260304T162929Z/experiment/area_entropy/outputs/area_entropy_summary.json'
with open(p) as f:
    x=json.load(f)
r=x['events']['GW170817']
print('A_p50=',r['A_p50'])
print('S_p50=',r['S_p50'])
print('S_minus_A_over_4=',r['S_p50']-r['A_p50']/4.0)
PY
```

### 10.5 Ponderación: spans de `EA` y `sigmaA`

```bash
python - <<'PY'
import csv
p='runs/analysis_area_EA_20260304T164052Z/experiment/area_entropy/outputs/area_EA_sigma_220.csv'
with open(p,newline='') as f:
    rows=list(csv.DictReader(f))
EA=[float(r['EA']) for r in rows]
SA=[float(r['sigmaA']) for r in rows]
print('EA_span=',max(EA)-min(EA))
print('sigmaA_span=',max(SA)-min(SA))
PY
```

---

## Nota final de auditoría

E5 queda completamente trazado en artefactos bajo `runs/...` y su lectura se puede automatizar con los comandos de esta guía. Los resultados deben interpretarse como consistencia de derivados `A,S` en una malla física discreta compartida entre modos, no como inferencia bayesiana final ni como prueba independiente de termodinámica de agujeros negros.
