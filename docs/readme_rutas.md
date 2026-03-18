# BASURIN — mapa de rutas (versión anti-pérdida de tiempo para IA)

Objetivo: que una IA (o humano) no pierda 5–6 horas diarias por confundir `RUN_ID`, `SUBRUN_ID`, `RUNS_ROOT` y árboles de experimentos.

## Raiz canonica del checkout

- La unica raiz canonica de trabajo es `/home/adnac/basurin/work/basurin`.
- `/home/adnac/basurin/work` no debe tratarse como repo de desarrollo ni como CWD valido para Git, tests o ejecucion del pipeline.
- La carpeta contenedora `/home/adnac/basurin/work` debe contener solo `basurin/`.
- Antes de usar rutas relativas o ejecutar comandos de control, verificar la raiz con `git rev-parse --show-toplevel`.

## Índice: qué busco → ruta exacta (30 segundos)

- **Gating canónico (run válido para downstream):**
  - `runs/<RUN_ID>/RUN_VALID/verdict.json`
- **H5 efectivamente usados por s1 + trazabilidad:**
  - `runs/<RUN_ID>/s1_fetch_strain/inputs/H1.h5`
  - `runs/<RUN_ID>/s1_fetch_strain/inputs/L1.h5`
  - `runs/<RUN_ID>/s1_fetch_strain/outputs/provenance.json`
- **Ventana ringdown (s2):**
  - `runs/<RUN_ID>/s2_ringdown_window/outputs/`
  - `runs/<RUN_ID>/s2_ringdown_window/outputs/window_meta.json` (si existe)
- **Estimaciones clave (s3):**
  - `runs/<RUN_ID>/s3_ringdown_estimates/outputs/estimates.json`
- **Multimode (s3b):**
  - `runs/<RUN_ID>/s3b_multimode_estimates/outputs/`
- **Filtro geométrico (s4):**
  - `runs/<RUN_ID>/s4_geometry_filter/outputs/`
- **Artefacto consolidado por evento (rama golden geometry explícita):**
  - `runs/<RUN_ID>/s4k_event_support_region/outputs/event_support_region.json`
- **Curvatura/diagnóstico (s6/s6b):**
  - `runs/<RUN_ID>/s6*/outputs/curvature*.json`
  - `runs/<RUN_ID>/s6*/outputs/metric_diagnostics*.json`
- **Rutas canónicas (external inputs vs outputs de stages):**
  - `data/losc/<EVENT_ID>/`
  - `runs/<run_id>/external_inputs/...`
  - `runs/<run_id>/<stage>/outputs/`
  - `runs/<run_id>/experiment/<name>/`
- **Fase 5 experimental (autoridad diaria de selección):**
  - `runs/prep_fase5_catalog_20260318T170928Z/outputs/canonical_run_ids_strict_real_52.txt`
  - `runs/prep_fase5_catalog_20260318T170928Z/outputs/canonical_event_run_map_strict_real_52.tsv`
  - `runs/prep_fase5_catalog_20260318T170928Z/outputs/event_run_selection_latest_strict_real_pass_52.json`
- **MALDA sobre un run gobernado:**
  - `runs/<RUN_ID>/experiment/malda_feature_table/outputs/event_features.csv`
  - `runs/<RUN_ID>/experiment/malda_discovery/outputs/discovery_summary.json`
  - `runs/<RUN_ID>/experiment/malda_formula_validation/outputs/formula_validation.json`
- **Ejemplo NetCDF contra release externo:**
  - `runs/ext_220_210_20260227T090000Z/external_inputs/siegel_220_210/Users/RichardFineMan/Downloads/data_release/220_210/<file>.nc`

---

# Atlas (geometrías) — localización inmediata (<60s)

## Rutas canónicas dentro del repo (no inventar “atlas.json”)
Estos ficheros viven versionados en el repo (preferidos por gobernanza). Lista explícita:

- `docs/ringdown/atlas/atlas_real_v2_s4.json`  *(recomendado para `s4_geometry_filter`)*
- `docs/ringdown/atlas/atlas_berti_v2_s4.json`
- `docs/ringdown/atlas/atlas_real_v1_s4.json`
- `docs/ringdown/atlas/atlas_berti_v2.json`
- `docs/ringdown/atlas/atlas_real_v1.json`
- `mvp/test_atlas_fixture.json` *(solo tests; no usar en runs reales)*

**Regla:** no existe (ni debe sugerirse) `atlas.json` en raíz. Si ves docs/ejemplos con `--atlas-path atlas.json`, trátalo como anti-ejemplo heredado.

## Qué atlas usar dónde (regla operativa)
- `s4_geometry_filter`: preferir `docs/ringdown/atlas/*_s4.json`
- tests: `mvp/test_atlas_fixture.json`
- batch/offline (`experiment_offline_batch`): documentar el atlas efectivo usado por esa CLI:
  - hoy está documentado como `docs/ringdown/atlas/atlas_berti_v2.json` en `docs/readme_experiment_4.md`
  - migración a `*_s4.json` solo cuando se haga explícita (no asumir)

## Descubrimiento (copy/paste)
Si dudas de rutas o estás en un checkout distinto:

```bash
find . -maxdepth 6 -type f \( -name "atlas*.json" -o -name "*atlas*.json" \) | sort
```

Comprobación rápida (estructura JSON):

```bash
python -c 'import json; p="docs/ringdown/atlas/atlas_real_v2_s4.json"; print("OK", p, "top_keys=", list(json.load(open(p)).keys())[:10])'
```

## Ejemplo de uso en pipeline (single-event)

```bash
python mvp/pipeline.py single \
  --event-id GW191113_071753 \
  --atlas-path docs/ringdown/atlas/atlas_real_v2_s4.json \
  --run-id <RUN_ID>
```

Si vas con defaults:

- usa `--atlas-default` (si está soportado por tu CLI) en lugar de inventar rutas.

### Comando universal para encontrar outputs sin pensar

```bash
RUN_ID="mvp_GW150914_..."
find "runs/$RUN_ID" -type f \
  \( -name 'verdict.json' \
  -o -name 'H1.h5' -o -name 'L1.h5' \
  -o -name 'provenance.json' \
  -o -name 'window_meta.json' \
  -o -name 'estimates.json' \
  -o -name 'curvature*.json' \
  -o -name 'metric_diagnostics*.json' \)
```

---

# HDF5 (LOSC/GWOSC) en 10 segundos (para que s1 no aborte)

Ruta canónica (input externo *read-only*):

`data/losc/<EVENT_ID>/`

Ruta real desde la raíz del repo:

`./data/losc/<EVENT_ID>/`

En este checkout actual:

`/home/ignac/work/basurin/data/losc/<EVENT_ID>/`

Precheck canónico:

```bash
python tools/losc_precheck.py --event-id "$EVENT_ID" --losc-root data/losc
```

Decisión rápida A/B/C:

- **Caso A (mount/symlink roto o mal apuntado)**: `data/losc` no apunta a la caché real.
  - Reapunta `data/losc` con la estrategia estándar del equipo (symlink o bind mount).
- **Caso A2 (caché real en `gw_events/strain`)**: los HDF5 existen, pero la vista canónica sigue vacía.
  - No cambies el pipeline para leer `gw_events/strain`.
  - Expón cada evento bajo `data/losc/<EVENT_ID>/` con symlink al directorio plano existente:
```bash
ln -sfn ../../gw_events/strain/"$EVENT_ID" "data/losc/$EVENT_ID"
python tools/losc_precheck.py --event-id "$EVENT_ID" --losc-root data/losc
```
- **Caso B (naming)**: hay `.h5/.hdf5`, pero no casan con H1/L1.
  - Crea symlinks casables `H1.h5` y `L1.h5` dentro del evento, sin renombrar originales:
- **Caso C (carpeta inexistente o vacía)**: `data/losc/<EVENT_ID>/` no existe o no tiene HDF5 válidos.
  - Pobla primero `data/losc/<EVENT_ID>/` con los descargadores canónicos del repo.
  - Repite `tools/losc_precheck.py`.
  - Solo después corre `s1_fetch_strain`.

```bash
ln -sf "<archivo_real_H1>.h5" "data/losc/$EVENT_ID/H1.h5"
ln -sf "<archivo_real_L1>.h5" "data/losc/$EVENT_ID/L1.h5"
```

Descarga canónica para Caso C:

```bash
python tools/fetch_losc_event.py --event-id "$EVENT_ID" --out-root data/losc
python tools/losc_precheck.py --event-id "$EVENT_ID" --losc-root data/losc
```

Si hay varios eventos incompletos, usa batch en lugar de repetir one-offs:

```bash
bash tools/fetch_losc_batch.sh /tmp/events_missing.txt
```

Roles recomendados de los scripts:

- `tools/losc_precheck.py`: verificacion read-only de visibilidad y naming.
- `tools/fetch_losc_event.py`: bootstrap puntual de un evento.
- `tools/fetch_losc_batch.sh`: completar/cohesionar una cohorte de eventos faltantes o rezagados.
- `tools/download_gw_events.py` y `tools/fetch_catalog_events.py`: bootstrap amplio de catálogo/cohorte, no la herramienta por defecto para reparar huecos puntuales en una caché ya existente.

**Solo después del precheck PASS**, continúa offline con `s1` (ejemplo corto):

```bash
python mvp/s1_fetch_strain.py --run <run_id> --event-id <EVENT_ID> --detectors H1,L1 --hdf5-root data/losc --reuse-if-present
```

Procedimiento completo de bootstrap/descarga/poblado: ver `README.md` en la sección "Descarga manual rápida de strain (GWOSC) para modo offline".

**Nota de gobernanza**: `data/losc/...` es input externo. El árbol auditable del run empieza en `runs/<RUN_ID>/...`.

**Nota MALDA**: `malda/10_build_event_feature_table.py` lee del catálogo local del repo, no de `data/losc/...`, pero sigue colgándose de un `run_id` BASURIN para escribir sus outputs bajo `runs/<RUN_ID>/experiment/...`.


## 0) Regla de oro (léela primero)

Un stage **siempre resuelve rutas como**:

`<RUNS_ROOT>/<run_id>/...`

Si eso no coincide con el árbol real donde está `RUN_VALID/verdict.json`, el stage falla.

---

## Namespace experimental: código fuente vs artefactos

**Esta distinción es crítica para no confundir rutas de código con rutas de artefactos.**

| Tipo | Ruta | Descripción |
|------|------|-------------|
| Código fuente experimental | `mvp/experiment/` | Paquete Python con módulos E5-A a E5-H, E5-Z y `base_contract.py`. Reside en el árbol del repositorio. |
| Artefactos experimentales | `runs/<run_id>/experiment/<nombre>/` | Outputs producidos por ejecución de módulos E5. Gobernados por `RUN_VALID`. |

> **Nota de migración:** el paquete top-level `./experiment` fue retirado. El namespace canónico vigente del código experimental es `mvp/experiment`. Imports legacy `from experiment...` o rutas `experiment/...` (sin prefijo `mvp/`) son obsoletos.

Ejemplos de rutas de código fuente experimental (Fase 5):

```text
mvp/experiment/__init__.py
mvp/experiment/base_contract.py
mvp/experiment/e5a_multi_event_aggregation.py
mvp/experiment/e5b_jackknife.py
mvp/experiment/e5c_ranking.py
mvp/experiment/e5d_bridge_malda.py
mvp/experiment/e5e_query.py
mvp/experiment/e5f_verdict_aggregation.py
mvp/experiment/e5h_blind_prediction.py
mvp/experiment/e5z_gpr_emulator.py
mvp/experiment/sandbox/          # E5-G: aislamiento total
```

Ejemplos de artefactos experimentales Fase 5 (bajo run gobernado):

```text
runs/<run_id>/experiment/e5a_multi_event_aggregation/outputs/aggregation_result.json
runs/<run_id>/experiment/e5b_jackknife/outputs/stability_per_geometry.json
runs/<run_id>/experiment/e5c_ranking/outputs/ranked_geometries.json
runs/<run_id>/experiment/e5d_bridge_malda/outputs/malda_input_payload.json
runs/<run_id>/experiment/e5e_query/outputs/query_<hash>.json
runs/<run_id>/experiment/e5f_verdict_aggregation/outputs/population_verdict.json
runs/<run_id>/experiment/e5h_blind_prediction/outputs/prediction_summary.json
runs/<run_id>/experiment/e5z_gpr_emulator/outputs/predicted_minima.json
```

---

## Fase 5 experimental: autoridad operativa diaria

- Referencia gobernante de preparación: `runs/prep_fase5_catalog_20260318T170928Z/`.
- Referencia de catálogo conservadora (`54` eventos): `runs/prep_fase5_catalog_20260318T170928Z/outputs/eligible_events_conservative.json` y `runs/prep_fase5_catalog_20260318T170928Z/outputs/eligible_events_conservative.txt`.
- Base materializada de trabajo actual (`52` runs canónicos `strict-real`): `runs/prep_fase5_catalog_20260318T170928Z/outputs/canonical_run_ids_strict_real_52.txt` y `runs/prep_fase5_catalog_20260318T170928Z/outputs/canonical_event_run_map_strict_real_52.tsv`.
- Regla de selección gobernante por evento: `runs/prep_fase5_catalog_20260318T170928Z/outputs/event_run_selection_latest_strict_real_pass_52.json`.
- Universo permitido: `run_id` que contiene `_real_`.
- Exclusiones obligatorias: `_real_offline_` y `_real_offline_rescue_`.
- Criterio de elección: entre candidatos válidos con `RUN_VALID=PASS`, seleccionar el más reciente por timestamp UTC embebido en el sufijo del `run_id`.
- Eventos excluidos de la base `strict-real` actual por falta de candidato válido: `GW170817` y `GW200115_042309`.
- Catálogos divergentes retirados a cuarentena y sin autoridad operativa: `quarantine/phase5_catalog_ambiguity_20260318/gwtc_quality_events.repo_root.csv`, `quarantine/phase5_catalog_ambiguity_20260318/gwtc_quality_events.gw_events.csv`, `quarantine/phase5_catalog_ambiguity_20260318/gwtc_quality_events.data_losc.csv`.
- Para arrancar E5-A/E5-B/E5-C/E5-F, usar `canonical_run_ids_strict_real_52.txt` y `canonical_event_run_map_strict_real_52.tsv`; no seleccionar runs desde listas históricas dispersas ni desde catálogos locales previos.
- Nada downstream debe ejecutarse si `RUN_VALID != PASS`.

---

## Rutas canónicas

- `data/losc/<EVENT_ID>/`: caché local *read-only* de HDF5 (external input). No es generado por el pipeline.
- `gw_events/strain/<EVENT_ID>/`: caché cruda/histórica opcional. Si existe, debe exponerse bajo `data/losc/<EVENT_ID>/` mediante symlink o bind mount antes de ejecutar el pipeline.
- `runs/<run_id>/external_inputs/...`: anclaje determinista de releases externos (por ejemplo, `siegel_220_210.tar.gz`) con hash verificable para trazabilidad.
- `runs/<run_id>/<stage>/outputs/`: artefactos producidos por stages. Deben convivir con `manifest.json` y `stage_summary.json`, incluyendo hashes SHA256.
- `runs/<run_id>/experiment/<name>/`: espacio de artefactos para experimentos; no debe mutar artefactos canónicos de stages ya emitidos. El código fuente de estos experimentos vive en `mvp/experiment/`.

### MALDA: rutas exactas y orden estricto

Flujo soportado hoy:

1. `malda/10_build_event_feature_table.py`
2. `malda/11_kan_pysr_discovery.py`
3. `malda/12_validate_formula_candidates.py`

Contrato:

- `step 10` escribe en `runs/<RUN_ID>/experiment/malda_feature_table/`.
- `step 11` escribe en `runs/<RUN_ID>/experiment/malda_discovery/`.
- `step 12` escribe en `runs/<RUN_ID>/experiment/malda_formula_validation/`.
- `step 12` exige que exista `runs/<RUN_ID>/RUN_VALID/verdict.json` con `PASS`.
- Si quieres gobernanza estricta, no uses un `RUN_ID` "huérfano" creado solo para MALDA; reutiliza un run canónico ya válido o crea primero ese run con el pipeline principal.

Rutas de trabajo:

- `runs/<RUN_ID>/experiment/malda_feature_table/outputs/event_features.csv`
- `runs/<RUN_ID>/experiment/malda_discovery/outputs/discovery_summary.json`
- `runs/<RUN_ID>/experiment/malda_formula_validation/outputs/formula_validation.json`

Ejemplo estricto:

```bash
export BASURIN_RUNS_ROOT=/home/ignac/work/basurin/runs

python malda/10_build_event_feature_table.py \
  --run-id synth_family_router_smoke \
  --bbh-only

python malda/11_kan_pysr_discovery.py \
  --run-id synth_family_router_smoke \
  --feature-policy claim_grade_symmetric \
  --targets E_rad_frac,af,F_220_dimless,f_ratio_221_220 \
  --heartbeat-seconds 10 \
  --bbh-only

python malda/12_validate_formula_candidates.py \
  --run-id synth_family_router_smoke \
  --targets E_rad_frac,af,F_220_dimless,f_ratio_221_220 \
  --bootstrap-samples 200
```

Anti-pérdida-de-tiempo:

- `data/losc/...` no crea `RUN_VALID`.
- `event_features.csv` no sustituye `RUN_VALID`.
- Si `runs/<RUN_ID>/RUN_VALID/verdict.json` no existe, `step 12` debe fallar.

### Golden geometry explícita: artefacto canónico por evento

`python -m mvp.pipeline multimode` ya materializa por defecto los inputs observacionales de `s4g/s4h`, ejecuta `s4g -> s4h -> s4i -> s4f -> s4j -> s4k` y degrada a `MODE220_NO_AREA_CONSTRAINT` cuando `221` no es usable o `s4f_area_observation` no produce un `area_obs.json` efectivo. Si `s4j` sí recibe una restriccion de area efectiva desde `s4f`, el path queda como `MODE220_PLUS_HAWKING`.

Si ejecutas la rama explícita `s4g -> s4h -> s4i -> s4j`, el artefacto downstream recomendado ya no es mirar cuatro JSON por separado, sino:

- `runs/<RUN_ID>/s4k_event_support_region/outputs/event_support_region.json`

Ese artefacto consolida:

- región compatible `220`
- región compatible `221`
- intersección común
- filtro de Hawking
- `multimode_viability` desde `s3b`
- `domain_status` desde `s4d` cuando exista
- `downstream_status` como semántica conservadora para consumo downstream

Regla práctica:

- `s4k_event_support_region` no re-ejecuta física; solo consolida artefactos ya emitidos.
- Si falta cualquiera de `s4g`, `s4h`, `s4i` o `s4j`, el stage debe fallar por contrato.
- `downstream_status.class` resume la legibilidad downstream del artefacto: `MULTIMODE_USABLE`, `GEOMETRY_PRESENT_BUT_NONINFORMATIVE`, `OUT_OF_DOMAIN` o `NO_SUPPORT_REGION`.
- `s5_aggregate` consume este artefacto cuando existe y, si hay suficientes eventos `MULTIMODE_USABLE`, usa `s4k_event_support_region` como base preferente para `multimode_conditioned_population`.

### Experimento de barrido de bandas multimodo

Ruta canónica del experimento:

```text
runs/<RUN_ID>/experiment/band_sweep_multimode/
```

Artefactos principales:

- `runs/<RUN_ID>/experiment/band_sweep_multimode/outputs/band_sweep_results.json`
- `runs/<RUN_ID>/experiment/band_sweep_multimode/outputs/band_sweep_summary.csv`
- `runs/<RUN_ID>/experiment/band_sweep_multimode/outputs/recommendation.json`

Subruns aislados por banda:

```text
runs/<RUN_ID>/experiment/band_sweep_multimode/runsroot/<SUBRUN_ID>/
```

Cada `<SUBRUN_ID>` es un run completo de `python -m mvp.pipeline multimode` con un par `band_low/band_high` distinto. El objetivo no es mutar stages canónicos aguas arriba, sino producir un diagnóstico operativo sobre si la banda actual deja `220` edge-locked, si `s4g` acepta alguna geometría y si aparece una región final no vacía.

### Experimento de barrido de `dt_start_s`

Ruta canónica del experimento:

```text
runs/<RUN_ID>/experiment/dt_start_sweep_multimode/
```

Artefactos principales:

- `runs/<RUN_ID>/experiment/dt_start_sweep_multimode/outputs/dt_start_sweep_results.json`
- `runs/<RUN_ID>/experiment/dt_start_sweep_multimode/outputs/dt_start_sweep_summary.csv`
- `runs/<RUN_ID>/experiment/dt_start_sweep_multimode/outputs/recommendation.json`

Subruns aislados por `dt_start_s`:

```text
runs/<RUN_ID>/experiment/dt_start_sweep_multimode/runsroot/<SUBRUN_ID>/
```

Cada `<SUBRUN_ID>` es un run completo de `python -m mvp.pipeline multimode` sobre una banda fija y un `dt_start_s` distinto. El objetivo es separar si el colapso del `220` viene del borde de banda o del arranque temporal del recorte.

### Experimento de barrido de `window_duration_s`

Ruta canónica del experimento:

```text
runs/<RUN_ID>/experiment/window_duration_sweep_multimode/
```

Artefactos principales:

- `runs/<RUN_ID>/experiment/window_duration_sweep_multimode/outputs/window_duration_sweep_results.json`
- `runs/<RUN_ID>/experiment/window_duration_sweep_multimode/outputs/window_duration_sweep_summary.csv`
- `runs/<RUN_ID>/experiment/window_duration_sweep_multimode/outputs/recommendation.json`

Subruns aislados por `window_duration_s`:

```text
runs/<RUN_ID>/experiment/window_duration_sweep_multimode/runsroot/<SUBRUN_ID>/
```

Cada `<SUBRUN_ID>` es un run completo de `python -m mvp.pipeline multimode` sobre banda fija, `dt_start_s` fijo y una duración de ventana distinta. El objetivo es ver si el colapso del `220` viene de una ventana demasiado corta/larga o del estimador mismo.

### Rutas de auditoría LOSC/t0 y batch offline

> En CLI de pipeline/batch, usa `--window-catalog` para s2. Alias soportado: `--t0-catalog`.

- `runs/<audit>/experiment/losc_quality/losc_event_quality.csv`
- `runs/<audit>/experiment/losc_quality/approved_events.txt`
- `runs/<run_id>/experiment/losc_quality/t0_catalog_gwosc_v2.json` *(ruta canónica)*
- `runs/<audit>/experiment/losc_quality/gwosc_ready_events.txt`
- `runs/<prep_run_id>/external_inputs/events_with_t0.txt` *(lista derivada para batch offline)*
- `runs/<batch>/experiment/offline_batch/results.csv`

Propósito de `t0_catalog_gwosc_v2.json`: catálogo `event_id -> t0_gps` para experimentos offline/batch (por ejemplo `experiment_offline_batch --t0-catalog ...`), tratado como input externo gobernado dentro de `runs/`.

Gobernanza: tanto el catálogo t0 como artefactos derivados (por ejemplo `events_with_t0.txt`) deben vivir bajo `runs/<run_id>/...`; está prohibido escribir fuera del árbol de runs auditable.

Nota: `RUN_VALID` es un directorio; el veredicto canónico siempre vive en `RUN_VALID/verdict.json`.

Verificación mínima (contract-first, tratable/auditable):

```bash
RUN_ID="<run_id>"
STAGE="<stage>"

test -d "data/losc/<EVENT_ID>"
test -d "runs/$RUN_ID/external_inputs"
ls -l "runs/$RUN_ID/$STAGE/manifest.json" "runs/$RUN_ID/$STAGE/stage_summary.json"
sha256sum runs/$RUN_ID/external_inputs/**/* 2>/dev/null || true
```

---

## 1) Resolución de `RUNS_ROOT` (orden exacto)

1. Si existe `BASURIN_RUNS_ROOT` → `RUNS_ROOT=$BASURIN_RUNS_ROOT`
2. Si no existe → `RUNS_ROOT=<cwd>/runs`

Diagnóstico inmediato:

```bash
python -c "import os; print('BASURIN_RUNS_ROOT=', os.environ.get('BASURIN_RUNS_ROOT'))"
pwd
```

---

## 2) Tipos de run y rutas correctas

### A) Run principal

Ruta base:

```text
runs/<RUN_ID>/
```

Gating obligatorio:

```text
runs/<RUN_ID>/RUN_VALID/verdict.json
```

### B) Subrun de experimento (`t0_sweep_full`)

Ruta base típica:

```text
runs/<RUN_ID>/experiment/t0_sweep_full/runs/<SUBRUN_ID>/
```

Gating del subrun:

```text
runs/<RUN_ID>/experiment/t0_sweep_full/runs/<SUBRUN_ID>/RUN_VALID/verdict.json
```

> Error clásico: ejecutar un stage con `--run-id <SUBRUN_ID>` sin redefinir `RUNS_ROOT`; entonces busca en `<cwd>/runs/<SUBRUN_ID>/...` y falla.

---

## 3) Cómo ejecutar un stage sobre subrun (sin romper nada)

```bash
RUN_ID="mvp_GW150914_real_local_20260217T115536Z"
SUBRUN_ID="${RUN_ID}__t0ms0000"
SUBRUNS_ROOT="runs/$RUN_ID/experiment/t0_sweep_full/runs"

# 1) validar gating real
ls -l "$SUBRUNS_ROOT/$SUBRUN_ID/RUN_VALID/verdict.json"

# 2) ejecutar stage con RUNS_ROOT correcto
BASURIN_RUNS_ROOT="$SUBRUNS_ROOT" \
python mvp/s3b_multimode_estimates.py \
  --run-id "$SUBRUN_ID" \
  --n-bootstrap 600 \
  --seed 12345
```

---

## 4) Checklist de 30 segundos antes de lanzar cualquier stage

1. ¿`run_id` corresponde a un run principal o a un subrun?
2. ¿`RUN_VALID/verdict.json` existe exactamente en `<RUNS_ROOT>/<run_id>/RUN_VALID/verdict.json`?
3. ¿`RUNS_ROOT` efectivo es el que crees (`BASURIN_RUNS_ROOT` vs `<cwd>/runs`)?
4. Si es experimento, ¿estás en `.../t0_sweep_full/runs/<SUBRUN_ID>` y no en rutas inventadas?

Comando rápido:

```bash
test -f "<RUNS_ROOT>/<run_id>/RUN_VALID/verdict.json" \
  && echo "OK gating" || echo "ERROR gating"
```

---

## 5) Seed sweep: patrón que más confunde

Con separación por seed puede aparecer:

```text
runs/<BASE_RUN>/experiment/t0_sweep_full_seed<seed>/runsroot/<BASE_RUN>/experiment/t0_sweep_full/runs/<SUBRUN_ID>/...
```

Procedimiento correcto:

1. Validar que exista `runsroot`.
2. Tratar `runsroot` como `scan_root` para inventario/agregados.
3. No inferir por prefijos de nombre; recorrer árbol real.

Snippet:

```bash
BASE_RUN="mvp_GW150914_nofetch_realfix_20260218T150000Z"
seed=101
RUNSROOT="runs/$BASE_RUN/experiment/t0_sweep_full_seed${seed}/runsroot"

test -d "$RUNSROOT" && echo "OK RUNSROOT" || echo "ERROR RUNSROOT"
```

---

## 6) Dónde mirar artefactos agregados (fuente de verdad)

Bajo run principal:

```text
runs/<RUN_ID>/experiment/derived/geometry_table.tsv
runs/<RUN_ID>/experiment/derived/sweep_inventory.json
```

No mezclar con outputs internos de cada subrun al evaluar estado global.

---

## 7) HDF5 externos (LOSC/GWOSC): ubicación canónica

Inputs externos de solo lectura:

```text
data/losc/<EVENT_ID>/
```

Patrones de nombre esperados:

- `*H1*.hdf5` o `*H1*.h5`
- `*L1*.hdf5` o `*L1*.h5`

Precheck obligatorio (mismo bloque canónico):

```bash
EVENT_ID=GW150914
echo "data/losc -> $(readlink -f data/losc 2>/dev/null || echo '(no symlink)')"
test -d "data/losc/$EVENT_ID" || { echo "ERROR: falta data/losc/$EVENT_ID (cache no montada/visible)"; exit 2; }
echo "H1/L1 matches:"
ls -1 "data/losc/$EVENT_ID" | egrep -i 'H1.*\.(h5|hdf5)$|L1.*\.(h5|hdf5)$' || echo "ERROR: hay ficheros pero no casan con H1/L1"
echo "total h5/hdf5:"; find "data/losc/$EVENT_ID" -maxdepth 1 -type f \( -iname '*.h5' -o -iname '*.hdf5' \) | wc -l
```

Si falla: resolver primero **Caso A (mount/symlink)** o **Caso B (nombres con symlinks H1.h5/L1.h5)** y repetir el precheck.

Ejemplo mínimo (s1 exige rutas explícitas si no hay fetch/caché):

```bash
python mvp/s1_fetch_strain.py --run <run_id> --event-id GW150914 --detectors H1,L1 \
  --local-hdf5 "H1=data/losc/GW150914/H-H1_...hdf5" \
  --local-hdf5 "L1=data/losc/GW150914/L-L1_...hdf5"
```

---

## 8) Reglas de gobernanza que NO se negocian

- No relajar `RUN_VALID`: si falta `verdict.json`, debe fallar.
- Todo output de stages vive bajo `<RUNS_ROOT>/<run_id>/...`.
- `data/losc/...` es input externo, no árbol de auditoría del run.

---

## 9) Playbook ultra-corto para IA (copiar/pegar mental)

1. Identifica `run_id` real que quieres procesar.
2. Calcula `RUNS_ROOT` real donde vive su `RUN_VALID/verdict.json`.
3. Exporta `BASURIN_RUNS_ROOT` solo si el run no está bajo `<cwd>/runs`.
4. Ejecuta stage.
5. Verifica `manifest.json` y `stage_summary.json` en la ruta efectiva.

Chequeo final:

```bash
BASE="<RUNS_ROOT>/<run_id>/<stage_name>"
ls -l "$BASE/manifest.json" "$BASE/stage_summary.json"
```

---

## Ejemplo completo: comparar contra un data release externo (NetCDF)

Patrón general (sin crear stage nuevo):

1. Guardar/anclar el tarball externo fuera del pipeline o en `runs/<run_id>/external_inputs/...` y registrar hash (`sha256sum`).
2. Extraer un `.nc` concreto bajo `runs/<run_id>/external_inputs/...`.
3. Inspeccionar cabecera con `ncdump -h` para confirmar grupos/variables.
4. Extraer percentiles con `python3-netcdf4` leyendo `group='posterior'`, variables `M` y `chi`, aplanando `chain×draw`, y calculando `p10/p50/p90`.

Ruta real usada como ejemplo:

`runs/ext_220_210_20260227T090000Z/external_inputs/siegel_220_210/Users/RichardFineMan/Downloads/data_release/220_210/<file>.nc`

Comandos:

```bash
RUN_ID="ext_220_210_20260227T090000Z"
BASE="runs/$RUN_ID/external_inputs/siegel_220_210/Users/RichardFineMan/Downloads/data_release/220_210"
NC_FILE="$BASE/<file>.nc"

# 1) Integridad del release anclado
sha256sum "runs/$RUN_ID/external_inputs/siegel_220_210"/*.tar.gz

# 2) Confirmar archivo y cabecera NetCDF
test -f "$NC_FILE"
ncdump -h "$NC_FILE" | sed -n '1,120p'

# 3) Percentiles p10/p50/p90 para posterior.M y posterior.chi
python3 - <<'PY'
import numpy as np
from netCDF4 import Dataset

nc_path = "runs/ext_220_210_20260227T090000Z/external_inputs/siegel_220_210/Users/RichardFineMan/Downloads/data_release/220_210/<file>.nc"
with Dataset(nc_path, "r") as ds:
    g = ds.groups["posterior"]
    for name in ("M", "chi"):
        arr = np.array(g.variables[name][:]).reshape(-1)
        p10, p50, p90 = np.percentile(arr, [10, 50, 90])
        print(f"{name}: p10={p10:.6g}, p50={p50:.6g}, p90={p90:.6g}")
PY
```

Checklist corto de verificación:

- `test -f "$NC_FILE"` debe pasar.
- `ncdump -h` debe mostrar `group: posterior` con variables `M` y `chi`.
- Registrar `sha256sum` del tarball y, si aplica, del `.nc` extraído para mantener artefactos auditables.

Si este archivo se sigue, se eliminan casi todos los errores de “ruta equivocada”.


## 10) Oráculo t0 v1.2: rutas de outputs

Una vez ejecutado:

```bash
python mvp/experiment_oracle_t0_ringdown.py --run-id <RUN_ID>
```

los outputs quedan en el run base:

```text
runs/<RUN_ID>/experiment/oracle_t0_ringdown/outputs/oracle_report.json
runs/<RUN_ID>/experiment/oracle_t0_ringdown/stage_summary.json
runs/<RUN_ID>/experiment/oracle_t0_ringdown/manifest.json
```

Input requerido por el oráculo (debe existir antes):

```text
runs/<RUN_ID>/experiment/t0_sweep_full_seed<seed>/outputs/t0_sweep_full_results.json
```

Si falta ese directorio/JSON, el oráculo imprime la ruta esperada exacta y el comando para regenerar el sweep (`phase=run`)
