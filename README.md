# BASURIN

BASURIN es un pipeline contract-first para el analisis reproducible de ringdown en ondas gravitacionales. Su objetivo cientifico no es limitarse a una estimacion puntual Kerr del remanente, sino reconstruir, por evento y luego a nivel poblacional, regiones de geometria compatibles con los datos, preservando trazabilidad, gating semantico y auditoria de artefactos.

El repositorio prioriza:

- reproducibilidad de IO y artefactos;
- interpretacion conservadora de resultados multimodo;
- separacion estricta entre pipeline canonico y experimentos;
- semantica auditable sobre "hay informacion", "el dominio aplica" y "se puede interpretar".

## Lectura rapida operativa

- Toda salida canonica debe vivir bajo `runs/<run_id>/...` o bajo el `BASURIN_RUNS_ROOT` efectivo.
- Un run solo existe para downstream si `runs/<run_id>/RUN_VALID/verdict.json` existe y su `verdict` es `PASS`.
- Cada stage/experimento canonico debe producir `manifest.json`, `stage_summary.json` y `outputs/*`, con hashes SHA256 auditables.
- La resolucion de la raiz de salida debe hacerse con `basurin_io.resolve_out_root("runs")`.
- El documento de rutas operativo sigue siendo [`docs/readme_rutas.md`](docs/readme_rutas.md).

## Leyenda de estado

- **Implementado**: existe entrypoint/contrato/artefacto en el repositorio actual.
- **Validado por tests**: existe semantica fijada por tests unitarios o de integracion/regresion.
- **Planificado**: objetivo cientifico o artefacto aun no cerrado como superficie canonica definitiva.

## 1. Vision general del proyecto

### 1.1 Que es BASURIN

BASURIN es un framework MVP para analisis de ringdown con:

- ejecucion por etapas;
- contratos de artefactos por stage;
- hashes SHA256 y trazabilidad en disco;
- gobernanza por run mediante `RUN_VALID`;
- modos de ejecucion `single`, `multimode`, `multi` y `batch`.

### 1.2 Que problema cientifico aborda

El problema cientifico central es inferir, a partir de observables de ringdown, que geometrias del atlas teorico permanecen compatibles con un evento y como esa informacion debe agregarse sin sobreinterpretar ni el multimodo ni la consistencia con Kerr/GR cuando los datos no lo sostienen.

### 1.3 Por que el objetivo final es geometrico y no solo Kerr puntual

En BASURIN, una estimacion puntual Kerr puede ser util como diagnostico condicionado, pero no debe confundirse con la salida cientifica final. El objetivo mas fuerte del proyecto es:

- localizar regiones compatibles del atlas para un evento;
- distinguir cuando el modo 221 aporta informacion util y cuando no;
- conservar flags de calidad, viabilidad y dominio;
- evitar que la ausencia de informatividad multimodo se convierta en una narrativa espuria de soporte a Kerr o a GR.

En ese sentido, BASURIN es mas cercano a un sistema de cribado geometrico auditable que a un sustituto de una inferencia bayesiana completa de colaboracion.

Por tanto, BASURIN no debe documentarse ni interpretarse como reemplazo de pipelines bayesianos oficiales de inferencia final, sino como una capa reproducible de cribado geometrico, gating y priorizacion de hipotesis.

## 2. Objetivo cientifico

### 2.1 Inferencia por evento

El objetivo cientifico por evento es el siguiente:

1. Inferir la region compatible para el modo 220.
2. Inferir la region compatible para el modo 221 cuando la extraccion del overtone sea usable.
3. Construir la interseccion multimodo entre 220 y 221 cuando el gate de viabilidad multimodo lo permita.
4. Filtrar esa interseccion con restricciones geometrico-fisicas adicionales, incluyendo el criterio/area de Hawking y el atlas geometrico canonico.
5. Devolver como salida cientifica un conjunto o region de geometrias remanentes compatibles, no una unica solucion puntual.

Principio conservador:

- si el multimodo no es viable, el pipeline debe degradar a inferencia monomodo;
- si el caso esta fuera de dominio, debe declararlo como tal;
- no debe fabricarse un remanente Kerr ni aparentar una confirmacion fuerte de GR cuando la cadena de gates no la autoriza.

#### 2.1.1 Estado actual de implementacion

La implementacion actual contiene dos rutas complementarias:

- **Ruta canonica actual de soporte geometrico monomodo**: `s4_geometry_filter` consume `s3_ringdown_estimates` y produce `compatible_set.json` y `ranked_all_full.json`.
- **Ruta explicita de region geometrica por modos**: `s4g_mode220_geometry_filter`, `s4h_mode221_geometry_filter`, `s4i_common_geometry_intersection` y `s4j_hawking_area_filter` modelan explicitamente 220, 221, la interseccion comun y el filtrado por area.

La segunda ruta ya esta implementada y registrada en contratos, pero no constituye aun el artefacto canonico unificado por evento consumido por todo el pipeline multimodo. Ese cierre pertenece todavia al roadmap cientifico.

### 2.2 Inferencia poblacional

El objetivo poblacional posterior es estudiar geometrias usando muchos eventos validos, no acumular "best fits" puntuales.

En terminos conceptuales, una inferencia poblacional correcta debe:

- consumir por evento regiones de soporte geometrico, no solo puntos;
- preservar flags de calidad, viabilidad y dominio de cada evento;
- excluir de evidencia geometrica positiva a casos `SINGLEMODE_ONLY`, `RINGDOWN_NONINFORMATIVE` u `OUT_OF_DOMAIN`;
- estudiar que familias geometricas sobreviven de manera recurrente;
- estudiar que regiones del atlas concentran soporte;
- estudiar que zonas quedan progresivamente excluidas a medida que crece la muestra.

#### 2.2.1 Estado actual de implementacion

Hoy existen piezas parciales de agregacion:

- `s5_aggregate` agrega `compatible_set` y metadatos de viabilidad entre runs;
- `s6c_population_geometry` produce un resumen poblacional descriptivo a partir de `aggregate.json`;
- `experiment_population_kerr.py` es un experimento no canonico sobre scores `s7`.

Lo que aun no existe como superficie cerrada y canonica es un stage poblacional que consuma exclusivamente un artefacto por evento que represente formalmente la region geometrica compatible completa.

## 3. Principios de gobernanza y reproducibilidad

### 3.1 IO determinista y raiz de salida

Reglas duras:

- La IO canonica solo puede escribirse bajo `runs/<run_id>/...`.
- Si `BASURIN_RUNS_ROOT` esta definido, esa es la raiz efectiva.
- La resolucion debe hacerse con `basurin_io.resolve_out_root("runs")`.
- Esta prohibido hardcodear rutas absolutas del tipo `/home/...`.

Rutas canonicas:

- stage canonico: `<RUNS_ROOT>/<run_id>/<stage>/...`
- experimento canonico: `<RUNS_ROOT>/<run_id>/experiment/<name>/...`

### 3.2 Gating por RUN_VALID y abort semantics

`RUN_VALID` es la unica puerta canonica hacia downstream.

- Si `RUN_VALID != PASS`, no debe correr ningun stage downstream.
- Si falla un contrato o un stage retorna error, el run no existe a efectos downstream.
- La semantica es fail-fast: el pipeline aborta y no debe inventar continuaciones parciales.

### 3.3 Contrato de artefactos

Cada stage/experimento canonico debe producir bajo su directorio:

- `manifest.json`
- `stage_summary.json`
- `outputs/*`

Y debe reflejar hashes SHA256 de los outputs en manifest/summary cuando aplica. Si un dataset intermedio no esta canonizado por contrato/manifest/summary, no forma parte de la superficie canonica del pipeline.

### 3.4 Trazabilidad y logging obligatorio

Todo entrypoint que escriba outputs debe emitir al final, como minimo:

- `OUT_ROOT=...`
- `STAGE_DIR=...`
- `OUTPUTS_DIR=...`
- `STAGE_SUMMARY=...`
- `MANIFEST=...`

La implementacion canonica usa `contracts.log_stage_paths(ctx)` para ello.

### 3.5 Datos externos y politica offline-first

- `data/losc/<EVENT_ID>/` es input externo read-only.
- Desde la raiz del repo, la ubicacion operativa es `./data/losc/<EVENT_ID>/`.
- En este checkout concreto, eso resuelve bajo `/home/ignac/work/basurin/data/losc/<EVENT_ID>/`.
- `s1_fetch_strain` copia los HDF5 efectivamente usados a `runs/<run_id>/s1_fetch_strain/inputs/{H1,L1}.h5`.
- La trazabilidad de esos inputs queda en `runs/<run_id>/s1_fetch_strain/outputs/provenance.json`.
- Cuando un input externo se ancla dentro del run, debe quedar bajo `external_inputs/` o equivalente y hasheado.

Regla practica:

- `data/losc/...` no pertenece al arbol auditable del run.
- `data/losc/...` no sustituye `RUN_VALID`.
- Los experimentos MALDA y los stages canonicos solo pueden escribir bajo `runs/<run_id>/...`.

### 3.6 Canonico vs experimento

- Lo canonico vive en `mvp/` y `mvp/contracts.py`.
- Los experimentos viven bajo `runs/<run_id>/experiment/<name>/...`.
- Un experimento no debe mutar artefactos canonicos ya emitidos.
- La exploracion sin contrato cientifico suficientemente cerrado debe quedarse en `experiment/`.

### 3.7 Excepcion documentada

`mvp/extract_psd.py` (`psd_extract`) es un helper de preparacion, no un stage canonico del pipeline principal.

- `s6c_brunete_psd_curvature` soporta fallback runtime de PSD via `external_inputs/psd_model.json` cuando no existe `psd/measured_psd.json`.
- Por contrato, `psd/measured_psd.json` no debe modelarse como dependencia rigida cuando ese fallback esta habilitado.

### 3.8 Disciplina oracle-first

El repositorio mantiene una disciplina "oracle-first": la superficie canonica debe poder relacionarse con un baseline externo reproducible o con un criterio explicito de PASS/FAIL. La exploracion sin oracle o sin criterio de aceptacion debe permanecer fuera del pipeline canonico o declararse expresamente como experimento.

## 4. Arquitectura del pipeline

### 4.1 Stages principales existentes

| Stage | Rol conceptual | Gates semanticos o notas de interpretacion |
| --- | --- | --- |
| `s1_fetch_strain` | Obtiene strain real o sintetico y fija la trazabilidad de adquisicion. | Es el primer stage despues de crear `RUN_VALID`. Copia inputs efectivos y emite `provenance.json`. |
| `s2_ringdown_window` | Resuelve `t0` y recorta la ventana de ringdown. | Puede operar en modo offline/online; su error invalida el run downstream. |
| `s3_ringdown_estimates` | Estima observables de ringdown del modo dominante (`f`, `tau`, `Q`) y sus incertidumbres. | Alimenta el soporte geometrico monomodo actual. |
| `s3b_multimode_estimates` | Extrae resumen multimodo `220+221`, compara modelos 1-modo/2-modos y escribe `multimode_viability`, `systematics_gate` y `science_evidence`. | Distingue `MULTIMODE_OK`, `SINGLEMODE_ONLY` y `RINGDOWN_NONINFORMATIVE`; la evidencia formal puede quedar `NOT_EVALUATED`. |
| `s4_geometry_filter` | Filtra el atlas a partir de `s3` y produce `compatible_set.json` y ranking completo. | Es la pieza canonica actual de soporte geometrico monomodo. No equivale por si sola a una reconstruccion multimodo completa. |
| `s4d_kerr_from_multimode` | Realiza una inversion Kerr condicionada a partir de `220+221`. | Si `multimode_viability != MULTIMODE_OK`, emite `SKIPPED_MULTIMODE_GATE`; si el evento cae fuera de dominio, emite `SKIPPED_OUT_OF_DOMAIN`. No reemplaza `s4_geometry_filter`. |
| `s5_aggregate` | Agrega soporte geometrico entre runs y conserva metadatos de viabilidad multimodo. | Debe consumir runs validos; contabiliza clases de viabilidad y usa fallbacks explicitamente anotados. Si existe `s4k_event_support_region`, prefiere `downstream_status=MULTIMODE_USABLE` como base explícita del agregado multimodo condicionado. |
| `s7_beyond_kerr_deviation_score` | Evalua el residual del modo 221 respecto de la prediccion Kerr inducida por `s4d`. | Es un score condicional, no una confirmacion independiente de GR; usa `independence_class = NON_INDEPENDENT`. |
| `s8a_family_gr_kerr` | Evalua la familia `GR_KERR_BH` combinando router, ratio filter, soporte geometrico y score `s7`. | No permite reclamar soporte GR Kerr solo con `s7`; requiere soporte monomodo/geometrico explicito y puede degradar a `INCONCLUSIVE`. |
| `s8b_family_bns` | Evalua la familia `BNS_REMNANT` con un atlas fenomenologico de post-merger. | No es un solver EOS. Puede terminar en `SUPPORTED`, `DISFAVORED` o `INCONCLUSIVE`; fuera de dominio propaga `OUT_OF_DOMAIN`. |
| `s8c_family_low_mass_bh_postmerger` | Evalua una rama Kerr restringida para remanentes BH de baja masa. | Requiere inversion Kerr valida y consistencia con el prior restringido; fuera de dominio queda `INCONCLUSIVE`. |

### 4.2 Stages geometricos explicitos por modo

Estos stages ya existen y representan mas directamente el flujo cientifico "region 220 -> region 221 -> interseccion -> Hawking":

| Stage | Rol conceptual | Estado en la orquestacion actual |
| --- | --- | --- |
| `s4g_mode220_geometry_filter` | Filtra el atlas con el observable del modo 220. | Implementado y contractual; no forma parte del `pipeline multimode` por defecto. |
| `s4h_mode221_geometry_filter` | Filtra el atlas con el observable del modo 221; si falta input de 221, emite `SKIPPED_221_UNAVAILABLE`. | Implementado y contractual. |
| `s4i_common_geometry_intersection` | Calcula la interseccion de geometrias comunes entre 220 y 221. | Implementado y contractual. |
| `s4j_hawking_area_filter` | Aplica el filtro de area/Hawking sobre la interseccion comun. | Implementado y contractual. |
| `s4k_event_support_region` | Consolida `220`, `221`, interseccion, Hawking, `multimode_viability` y `domain_status` en un unico artefacto por evento. | Implementado y contractual; añade `downstream_status` conservador (`MULTIMODE_USABLE`, `GEOMETRY_PRESENT_BUT_NONINFORMATIVE`, `OUT_OF_DOMAIN`, `NO_SUPPORT_REGION`) para consumo downstream. |

### 4.3 Gates auxiliares que protegen la interpretacion

Ademas de los stages listados arriba, la interpretacion multimodo actual depende de tres piezas auxiliares:

- `s4c_kerr_consistency`: resume consistencia Kerr y se salta con `SKIPPED_MULTIMODE_GATE` cuando `s3b` no autoriza multimodo.
- `s4e_kerr_ratio_filter`: comprueba la consistencia del cociente `221/220` respecto de la banda Kerr de referencia.
- `s8_family_router`: decide que familias fisicas deben evaluarse; puede enrutar varias familias en orden de prioridad, sin forzar una sola interpretacion cuando el metadata es debil.

### 4.4 Otros stages existentes

El repositorio contiene ademas capas de geometria de la informacion y curvatura (`s6*`) y varios experimentos. Son utiles como diagnostico o exploracion, pero no sustituyen el flujo cientifico principal descrito arriba.

## 5. Semantica conservadora y no sobreinterpretacion

Las siguientes reglas deben leerse como parte del contrato cientifico del proyecto:

- `SINGLEMODE_ONLY` significa que el evento puede seguir siendo tratado como monomodo, pero no autoriza inferencia multimodo fuerte.
- `RINGDOWN_NONINFORMATIVE` significa que incluso el soporte del modo dominante es insuficiente para sostener inferencias downstream fuertes.
- `SKIPPED_MULTIMODE_GATE` en `s4d` o `s4c` impide fabricar un remanente Kerr a partir de `220+221`.
- `s7_beyond_kerr_deviation_score` es un chequeo condicional y no independiente. Incluso cuando devuelve `GR_CONSISTENT`, la salida sigue etiquetada como `NON_INDEPENDENT`.
- `SKIPPED_S4D_GATE` en `s7` impide aparentar evidencia independiente a favor de GR cuando `s4d` no produjo una inversion interpretable.
- `OUT_OF_DOMAIN` o `SKIPPED_OUT_OF_DOMAIN` significa que la banda de analisis no tiene solapamiento fisicamente util con la envolvente relevante del modelo/familia. No debe traducirse en `DISFAVORED` fuerte fuera de dominio.
- `INCONCLUSIVE` es un resultado cientificamente valido cuando los datos, el dominio o la cadena de gates no permiten ir mas lejos.
- En `s3b`, `science_evidence` solo puede quedar `EVALUATED` si `multimode_viability == MULTIMODE_OK` y el gate de sistematicas no bloquea la interpretacion. La ausencia de evaluacion es una salida explicita, no un fallo silencioso.

## 6. Estado actual y validacion

Esta seccion describe semantica de software validada por tests y regresiones. No debe leerse como validacion fisica completa del programa cientifico.

### 6.1 Semantica validada por tests

Se ha verificado por tests que:

- una cadena cross-stage tipo `GW170817` puede quedar en `SINGLEMODE_ONLY` en `s3b_multimode_estimates` cuando el modo 221 no es utilizable;
- esa condicion se propaga a `SKIPPED_MULTIMODE_GATE` en `s4d_kerr_from_multimode` y `s4c_kerr_consistency`, evitando fabricar una inversion Kerr multimodo;
- `s7_beyond_kerr_deviation_score` convierte esa situacion en `SKIPPED_S4D_GATE` y mantiene `independence_class = NON_INDEPENDENT`, sin producir chi-cuadrados o epsilons espurios;
- la narrativa Kerr/GR no se propaga downstream como si fuera evidencia fuerte cuando faltan gates o soporte geometrico adicional;
- para contextos fuera de dominio, `s4d` puede emitir `SKIPPED_OUT_OF_DOMAIN`, `s7` propaga `SKIPPED_OUT_OF_DOMAIN`, y las familias `BNS_REMNANT` y `LOW_MASS_BH_POSTMERGER` permanecen `INCONCLUSIVE` en lugar de convertirse artificialmente en evidencia positiva o negativa fuerte;
- `s8a_family_gr_kerr` no acepta un `GR_CONSISTENT` de `s7` como soporte suficiente por si solo: exige soporte monomodo/geometrico explicito y puede degradar a `INCONCLUSIVE`;
- `s5_aggregate` conserva cuentas y razones de `multimode_viability` por evento y no trata la ausencia de `s3b` como evidencia multimodo positiva.

Tests representativos de esta semantica:

- `tests/test_multimode_wiring_unittest.py`
- `tests/test_s7_beyond_kerr_deviation_score.py`
- `tests/test_s8_family_router.py`
- `tests/test_s5_aggregate_ranked_integration.py`

En el estado local inspeccionado para esta reescritura, ese subconjunto pasa (`52 passed`).

### 6.2 Que esta implementado pero no debe sobreafirmarse

Implementado hoy no significa que ya exista una superficie cientifica definitivamente cerrada. En particular:

- la ruta explicita `220 -> 221 -> interseccion -> Hawking` existe en stages separados, pero el artefacto canonico unico por evento todavia no esta fijado como salida unificada del pipeline multimodo completo;
- la ruta explicita `220 -> 221 -> interseccion -> Hawking` ya puede consolidarse en `s4k_event_support_region` y ahora se orquesta por defecto dentro de `python -m mvp.pipeline multimode`, degradando conservadoramente a `220 + Hawking` cuando `221` no es usable;
- existen handlers de familia y agregacion poblacional, pero la inferencia poblacional final basada en regiones canonicas por evento sigue siendo objetivo de diseno, no conclusion cerrada;
- `experiment_population_kerr.py` existe, pero es un experimento no canonico y no debe confundirse con el stage poblacional definitivo que consumira artefactos geometricos por evento.

## 7. Uso y ejemplos

Los ejemplos siguientes usan CLIs verificadas en el repositorio actual. Ajuste atlas, catalogos y datasets externos de acuerdo con [`docs/readme_rutas.md`](docs/readme_rutas.md).

### 7.1 Quickstart minimo

Smoke test sintetico de un evento:

```bash
python -m mvp.pipeline single \
  --event-id GW150914 \
  --atlas-default \
  --synthetic
```

Pipeline multimodo sintetico:

```bash
python -m mvp.pipeline multimode \
  --event-id GW150914 \
  --atlas-default \
  --synthetic
```

Agregacion multi-evento sintetica:

```bash
python -m mvp.pipeline multi \
  --events GW150914,GW151226 \
  --atlas-default \
  --synthetic \
  --min-coverage 1.0
```

### 7.2 Ejecucion offline-first recomendada

Antes de ejecutar datos reales, verifique la visibilidad de HDF5 locales:

```bash
python tools/losc_precheck.py --event-id GW150914 --losc-root data/losc
```

Pipeline single consumiendo catalogo de `t0` ya auditado:

```bash
python -m mvp.pipeline single \
  --event-id GW150914 \
  --atlas-default \
  --offline-s2 \
  --window-catalog "runs/<AUDIT_RUN>/experiment/losc_quality/t0_catalog_gwosc_v2.json"
```

Batch offline-first:

```bash
python -m mvp.experiment_offline_batch \
  --batch-run-id <BATCH_RUN_ID> \
  --events-file "runs/<PREP_RUN>/external_inputs/events_with_t0.txt" \
  --t0-catalog "runs/<AUDIT_RUN>/experiment/losc_quality/t0_catalog_gwosc_v2.json"
```

### 7.3 Sobre los stages geometricos explicitos

`s4g/s4h/s4i/s4j` tienen CLI propia y forman la rama mas directamente alineada con el objetivo "region compatible por modo + interseccion + Hawking". `python -m mvp.pipeline multimode` ahora materializa por defecto los inputs observacionales de `s4g/s4h`, ejecuta esa rama explícita y consolida `s4k_event_support_region`; si `221` no es usable, la ruta degrada de forma conservadora a `220 + Hawking` en lugar de abortar la región geométrica por evento.

### 7.3b Experimento de barrido de bandas multimodo

Cuando `s3_ringdown_estimates` parece clavarse en el borde de banda o en el floor de `tau`, no hace falta probar manualmente una banda cada vez. `experiment_band_sweep_multimode.py` lanza subruns aislados de `python -m mvp.pipeline multimode` para un mismo evento y resume por banda:

- `f_hz`, `tau_s` y `Q` estimados por `s3`
- `n_geometries_accepted` en `s4g_mode220_geometry_filter`
- `support_region_status` y `support_region_n_final` del top-level
- diagnóstico automático (`LIKELY_EDGE_LOCKED_220`, `FOUND_NONEMPTY_SUPPORT_REGION`, etc.)
- una recomendación final en `recommendation.json`

Ejemplo:

```bash
python -m mvp.experiment_band_sweep_multimode \
  --run-id band_sweep_GW190521 \
  --event-id GW190521_074359 \
  --bands 150-400,400-800,800-1200,1200-1600 \
  --atlas-default \
  --offline
```

Artefactos emitidos:

- `runs/<run_id>/experiment/band_sweep_multimode/outputs/band_sweep_results.json`
- `runs/<run_id>/experiment/band_sweep_multimode/outputs/band_sweep_summary.csv`
- `runs/<run_id>/experiment/band_sweep_multimode/outputs/recommendation.json`
- subruns completos bajo `runs/<run_id>/experiment/band_sweep_multimode/runsroot/<subrun_id>/...`

### 7.4 Flujo MALDA estricto sobre runs gobernados

Los entrypoints MALDA actuales relevantes para discovery simbolico son:

- `malda/10_build_event_feature_table.py`
- `malda/11_kan_pysr_discovery.py`
- `malda/12_validate_formula_candidates.py`

Contrato operativo:

- `10_build_event_feature_table.py` lee del catalogo local del repo (`gwtc_quality_events.csv`, `gwtc_events_t0.json`) y no necesita outputs canonicos upstream.
- Aun asi, si vas a colgar MALDA de BASURIN con gobernanza estricta, debes ejecutarlo sobre un `run_id` que ya tenga `runs/<run_id>/RUN_VALID/verdict.json` con `PASS`.
- `11_kan_pysr_discovery.py` consume el `event_features.csv` emitido por `step 10`.
- `12_validate_formula_candidates.py` consume `event_features.csv` + `discovery_summary.json` y exige `RUN_VALID == PASS`.
- La secuencia valida es siempre `10 -> 11 -> 12` dentro del mismo `run_id`.

Rutas emitidas por MALDA bajo un run gobernado:

- `runs/<run_id>/experiment/malda_feature_table/outputs/event_features.csv`
- `runs/<run_id>/experiment/malda_discovery/outputs/discovery_summary.json`
- `runs/<run_id>/experiment/malda_formula_validation/outputs/formula_validation.json`

Ejemplo estricto reutilizando un run ya gobernado:

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

Regla de interpretacion:

- Si un `run_id` no tiene `RUN_VALID/verdict.json`, MALDA discovery puede haberse ejecutado como experimento local, pero `step 12` debe abortar por contrato.
- Si quieres gobernanza estricta, no "repares" un run a mano creando `RUN_VALID`; cuelga MALDA de un run canonico ya valido o crea primero ese run con el pipeline canónico.

## 8. Rutas y artefactos clave

### 8.1 Donde mirar primero al auditar o depurar

1. `runs/<run_id>/RUN_VALID/verdict.json`
2. `runs/<run_id>/pipeline_timeline.json`
3. `runs/<run_id>/<stage>/stage_summary.json`
4. `runs/<run_id>/<stage>/manifest.json`
5. `runs/<run_id>/s1_fetch_strain/outputs/provenance.json`
6. `runs/<run_id>/s4k_event_support_region/outputs/event_support_region.json` cuando exista la rama golden geometry explicita

### 8.2 Rutas practicas

- mapa operativo de rutas: [`docs/readme_rutas.md`](docs/readme_rutas.md)
- flujo de ejecucion y validaciones: [`docs/request_flow.md`](docs/request_flow.md)
- semantica multimodo y canal de evidencia: [`docs/multimode_viability_and_evidence.md`](docs/multimode_viability_and_evidence.md)
- artefacto consolidado por evento de la rama golden geometry: `runs/<run_id>/s4k_event_support_region/outputs/event_support_region.json` con `downstream_status.class` para gating conservador downstream

### 8.3 Atlas y metadatos

- Los atlas versionados viven bajo `docs/ringdown/atlas/`.
- La clasificacion usada por `malda/10_build_event_feature_table.py` sale del catalogo (`m1_source`, `m2_source`), sin un arbol de metadatos versionados bajo `docs/ringdown/`.
- Cuando se pase un atlas explicito por CLI, debe ser un fichero versionado y trazable del repositorio o una ruta externa igualmente auditable.

Nota prudente:

- la direccion de gobernanza sigue siendo mover todo consumo efectivo de inputs externos hacia artefactos explicitamente anclados y hasheados dentro del run.

## 9. Hoja de ruta cientifica

Los siguientes puntos son planificacion explicita, no capacidades cerradas hoy:

1. Formalizar el artefacto canonico por evento que represente la region geometrica compatible completa, incluyendo 220, 221 cuando aplique, interseccion multimodo, filtros de area y metadata de dominio/viabilidad.
2. Construir un stage poblacional canonico que consuma solo runs validos y artefactos canonicos por evento.
3. Estudiar ocupacion del atlas, recurrencia de familias geometricas y exclusiones poblacionales sobre regiones de soporte, no sobre best fits puntuales.
4. Mantener separacion estricta entre logica por evento y agregacion poblacional para impedir que un evento no informativo contamine una conclusion poblacional como si fuera evidencia geometrica positiva.
5. Integrar de manera mas cerrada la rama explicita `s4g/s4h/s4i/s4j` con la superficie canonica del pipeline cuando el artefacto por evento quede fijado.

## 10. Contribucion y disciplina de cambios

Quien modifique BASURIN debe preservar estas reglas:

- no relajar `RUN_VALID` ni el gating downstream sin una justificacion explicita;
- no escribir fuera de `runs/<run_id>/...`;
- no introducir artefactos intermedios no canonizados como si fueran parte del pipeline;
- acompanar cambios en stages o contratos con tests y/o regresiones proporcionales;
- documentar cualquier cambio de rutas, contratos o semantica de veredictos.

En resumen: BASURIN debe fallar de forma explicita, degradar de forma conservadora y dejar evidencia reproducible de lo que hizo y de lo que deliberadamente no interpreto.
