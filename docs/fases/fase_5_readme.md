# Fase 5 — Catálogo de Alternativas Experimentales

## Namespace canónico

Los módulos de Fase 5 viven en el paquete **`mvp/experiment/`** del repositorio.

> **Nota de migración:** el paquete top-level `./experiment` fue retirado. El namespace vigente es `mvp/experiment`. Cualquier referencia a `from experiment...` o `experiment/...` como ruta de código fuente es legacy y debe corregirse a `mvp.experiment...` / `mvp/experiment/...`.

## Objetivo

Proveer un catálogo de **nueve módulos experimentales** (E5-A a E5-H, E5-Z) que operan como *lectores certificados* del pipeline canónico. Cada alternativa explora una pregunta científica o metodológica que **no puede ni debe** vivir en el core, pero que produce artefactos auditables, reproducibles y con gobernanza verificada.

Principio rector: **toda alternativa es un lector certificado**. Escribe solo en su namespace. El pipeline canónico es inmutable para ella.

---

## Arquitectura

```
mvp/experiment/
├── __init__.py
├── base_contract.py                  # Contrato universal de entrada (RUN_VALID gate)
├── e5a_multi_event_aggregation.py    # Intersección/unión/Jaccard multi-evento
├── e5b_jackknife.py                  # Estabilidad leave-one-out
├── e5c_ranking.py                    # Ranking determinista + Lorenz/Gini
├── e5d_bridge_malda.py               # Bridge a framework bayesiano externo
│   └── e5d_bridge_malda/
│       └── external_input/           # Schema externo (DEBE preexistir)
├── e5e_query.py                      # Motor de consulta reproducible (read-only)
├── e5f_verdict_aggregation.py        # Agregación de veredictos (evidencia 2º orden)
├── e5h_blind_prediction.py           # Predicción ciega cross-evento
├── e5z_gpr_emulator.py               # Emulador continuo GPR (arma secreta)
└── sandbox/                          # Espacio libre sin contrato (E5-G)
```

Todos los outputs van bajo:

```
runs/<run_id>/experiment/<nombre_experimento>/
```

Ningún módulo escribe fuera de su namespace. CI lo verifica.

---

## Contrato de entrada universal

Definido en `mvp/experiment/base_contract.py`. Todas las alternativas comparten:

```python
REQUIRED_CANONICAL_GATES = {
    "compatible_set": "s4_geometry_filter/compatible_set.json",
    "stage_summary":  "stage_summary.json",
    "verdict":        "verdict.json",
    "estimates":      "s3b_multimode_estimates/estimates.json",
}
```

**Invariante primario:** ninguna alternativa puede consumir un `run_id` cuyo `stage_summary.json` no tenga `run_valid: "PASS"`. Violación → excepción `GovernanceViolation`.

```python
from mvp.experiment.base_contract import validate_and_load_run

run_dir, summary = validate_and_load_run("<RUN_ID_CANONICO>")
# Si run_valid != PASS → GovernanceViolation
```

---

## Base operativa diaria gobernante

- El código fuente de los módulos E5 vive en `mvp/experiment/`; la autoridad operativa diaria vive en artefactos bajo `runs/prep_fase5_catalog_20260318T170928Z/`.
- La cohorte conservadora de catálogo es de `54` eventos y vive en `runs/prep_fase5_catalog_20260318T170928Z/outputs/eligible_events_conservative.json` y `runs/prep_fase5_catalog_20260318T170928Z/outputs/eligible_events_conservative.txt`. Es referencia de catálogo, no lista diaria de runs.
- La base materializada de trabajo actual para Fase 5 es de `52` runs canónicos `strict-real` y debe consumirse desde `runs/prep_fase5_catalog_20260318T170928Z/outputs/canonical_run_ids_strict_real_52.txt` y `runs/prep_fase5_catalog_20260318T170928Z/outputs/canonical_event_run_map_strict_real_52.tsv`.
- La selección gobernante por evento está documentada en `runs/prep_fase5_catalog_20260318T170928Z/outputs/event_run_selection_latest_strict_real_pass_52.json`: universo permitido `run_id` que contiene `_real_`; exclusiones obligatorias `_real_offline_` y `_real_offline_rescue_`; entre candidatos válidos con `RUN_VALID=PASS`, se elige el más reciente por timestamp UTC embebido en el sufijo del `run_id`.
- Los eventos `GW170817` y `GW200115_042309` quedan excluidos de la base `strict-real` actual por no tener candidato válido.
- Las listas históricas dispersas y los catálogos previos no deben usarse como autoridad operativa. Los catálogos divergentes quedaron retirados a `quarantine/phase5_catalog_ambiguity_20260318/`.
- Para arrancar E5-A/E5-B/E5-C/E5-F en operación diaria, la entrada operativa debe salir de `canonical_run_ids_strict_real_52.txt` y `canonical_event_run_map_strict_real_52.tsv`. Los ejemplos CLI de esta página son sintácticos y no sustituyen esos artefactos.
- Nada downstream debe ejecutarse si `RUN_VALID != PASS`.

---

## Catálogo de alternativas

### E5-A — Agregación Multi-Evento

**Pregunta:** ¿qué geometrías sobreviven en *todos* los eventos? ¿qué porcentaje del atlas aparece en un único evento?

**Consume:** `compatible_set.json` × N runs (todos `RUN_VALID=PASS`).

**Produce:**

| Artefacto | Contenido |
|-----------|-----------|
| `aggregation_result.json` | Intersección, unión, frecuencia por geometry_id |
| `jaccard_matrix.json` | Similitud par-a-par entre eventos |
| `persistence_histogram.json` | Distribución de apariciones por familia |
| `manifest.json` | SHA-256 de todos los inputs |

**Entrada operativa diaria:** cargar `--run-ids` desde `runs/prep_fase5_catalog_20260318T170928Z/outputs/canonical_run_ids_strict_real_52.txt` y usar `runs/prep_fase5_catalog_20260318T170928Z/outputs/canonical_event_run_map_strict_real_52.tsv` como traza evento → run.

**Uso:**

```bash
python mvp/experiment/e5a_multi_event_aggregation.py \
  --run-ids <RUN_ID_1> <RUN_ID_2> --dry-run
```

**Requisitos:** ≥ 2 runs con `RUN_VALID=PASS`.

**Criterio de promoción:** ≥ 5 eventos + calibración per-evento + permutation null model.

---

### E5-B — Auditoría de Estabilidad (Jackknife)

**Pregunta:** ¿cuánto cambia la intersección/unión al retirar un evento? ¿Qué evento es el más influyente?

**Consume:** `compatible_set.json` + `stage_summary.json` × N runs.

**Produce:**

| Artefacto | Contenido |
|-----------|-----------|
| `stability_per_geometry.json` | Varianza jackknife de presencia por geometry_id |
| `stability_certificate.json` | `STABLE` / `MODERATE` / `UNSTABLE` por geometría |
| `influence_ranking.json` | Eventos ordenados por influencia en intersección |

**Umbrales de estabilidad:**

- `STABLE`: varianza jackknife ≤ 0.05
- `MODERATE`: 0.05 < varianza ≤ 0.20
- `UNSTABLE`: varianza > 0.20
- `SINGLETON`: N ≤ 1 (no evaluable)

**Entrada operativa diaria:** cargar `--run-ids` desde `runs/prep_fase5_catalog_20260318T170928Z/outputs/canonical_run_ids_strict_real_52.txt` y usar `runs/prep_fase5_catalog_20260318T170928Z/outputs/canonical_event_run_map_strict_real_52.tsv` como traza evento → run.

**Uso:**

```bash
python mvp/experiment/e5b_jackknife.py \
  --run-ids <RUN_ID_1> <RUN_ID_2> <RUN_ID_3> --dry-run
```

**Requisitos:** ≥ 3 runs.

**Criterio de promoción:** Cuando exista `PopulationContract` y ≥ 5 eventos calibrados → fase 8 canónica.

---

### E5-C — Ranking de Geometrías por Score Compuesto

**Pregunta:** ¿cuál es el ranking determinista (NO bayesiano) del conjunto compatible?

**Score compuesto:**

```
score = w_maha × mahalanobis_norm + w_dlnL × delta_lnL_norm + w_sat × saturation_norm
```

Pesos por defecto: `(0.5, 0.4, 0.1)` — **actualmente arbitrarios**, pendientes de validación Fisher.

**Consume:** `compatible_set.json` + `estimates.json` + `verdict.json` (single run).

**Produce:**

| Artefacto | Contenido |
|-----------|-----------|
| `ranked_geometries.json` | Lista ordenada con score y componentes |
| `lorenz_curve.json` | Coordenadas para curva de Lorenz |
| `gini_coefficient.json` | Concentración del score (0=uniforme, 1=concentrado) |

**Invariante:** `ranked_geometries.json` solo contiene geometry_ids presentes en `compatible_set.json`. Ningún geometry_id nuevo introducido.

**Entrada operativa diaria:** el universo permitido de `run_id` está fijado por `runs/prep_fase5_catalog_20260318T170928Z/outputs/canonical_run_ids_strict_real_52.txt`; el `run_id` canónico por evento debe resolverse desde `runs/prep_fase5_catalog_20260318T170928Z/outputs/canonical_event_run_map_strict_real_52.tsv`.

**Uso:**

```bash
python mvp/experiment/e5c_ranking.py --run-id <RUN_ID_CANONICO> --weights 0.5 0.4 0.1
```

**Criterio de promoción:** Pesos justificados por Fisher information geometry (Método Brunete) + reproducibilidad en ≥ 3 eventos.

---

### E5-D — Bridge a MALDA (o Análogo Bayesiano Externo)

**Pregunta:** ¿cómo se compara BASURIN con un pipeline bayesiano (MALDA, pyRing) sobre los mismos datos?

**Consume:** `compatible_set.json` + `estimates.json` + `external_input/malda_schema.json` (DEBE preexistir).

**Produce:**

| Artefacto | Contenido |
|-----------|-----------|
| `malda_input_payload.json` | Traducción al formato externo |
| `translation_manifest.json` | Mapeo campo-a-campo + SHA-256 |
| `bridge_validation_report.json` | Campos traducidos vs. schema externo |

**Precondición obligatoria:**

```bash
[ -f mvp/experiment/e5d_bridge_malda/external_input/malda_schema.json ] || \
  echo "ERROR: bridge no ejecutable"
```

Si el schema no existe, el bridge opera en modo `DRAFT` y advierte.

**Criterio de promoción:** Nunca al core. Permanece como utilidad de comparación. Si el framework externo se integra, se convierte en adapter con contrato versionado.

---

### E5-E — Consulta Reproducible sin Stage Pesado

**Pregunta:** ¿cuántas geometrías edgb tienen d² < 5 en todos los eventos? ¿qué pasa si elevo el umbral de mahalanobis en 0.5?

Motor de consulta ligero, determinista e idempotente sobre snapshots congelados.

**Consume:** `compatible_set.json` × N (read-only) + `stage_summary.json`.

**Produce:**

| Artefacto | Contenido |
|-----------|-----------|
| `query_<hash>.json` | Resultado + inputs + SHA-256 verificación |
| `query_log.jsonl` | Registro de todas las consultas ejecutadas |

**Sintaxis de consulta:**

```
family == 'edgb' AND mahalanobis_d2 < 5.0
```

Operadores soportados: `==`, `!=`, `<`, `<=`, `>`, `>=`. Cláusulas conectadas por `AND`.

**Idempotencia verificada:** dos ejecuciones idénticas producen el mismo `query_id` (SHA-256 del query string) y los mismos resultados si los inputs no cambian.

**Uso:**

```bash
python mvp/experiment/e5e_query.py \
  --query "family == 'edgb' AND mahalanobis_d2 < 5.0" \
  --run-ids GW150914_v1 --dry-run
```

**Criterio de promoción:** No se promueve al core. Candidato a utilidad CLI independiente (`basurin-query`).

---

### E5-F — Agregación de Veredictos

**Pregunta:** ¿cuál es la distribución poblacional de veredictos (`SUPPORTED` / `REJECTED` / `INCONCLUSIVE`) por familia? Esto es *evidencia de segundo orden*: agrega conclusiones, no mediciones.

**Consume:** **Solo** `verdict.json` × N. Sin acceso a datos crudos ni geometrías.

**Produce:**

| Artefacto | Contenido |
|-----------|-----------|
| `population_verdict.json` | Distribución completa por familia y modo |
| `family_support_rate.json` | Fracción de eventos donde cada familia es compatible |
| `evidence_strength.json` | `WEAK` / `MODERATE` / `STRONG` por familia |

**Clasificación de evidencia (determinista, no bayesiana):**

| Tasa de soporte | Clasificación |
|-----------------|---------------|
| ≥ 80% | `STRONG` |
| ≥ 50% | `MODERATE` |
| < 50% | `WEAK` |

**Entrada operativa diaria:** cargar `--run-ids` desde `runs/prep_fase5_catalog_20260318T170928Z/outputs/canonical_run_ids_strict_real_52.txt` y usar `runs/prep_fase5_catalog_20260318T170928Z/outputs/canonical_event_run_map_strict_real_52.tsv` como traza evento → run.

**Uso:**

```bash
python mvp/experiment/e5f_verdict_aggregation.py \
  --run-ids <RUN_ID_1> <RUN_ID_2> <RUN_ID_3> --dry-run
```

**Impacto científico:** produce el resultado que aparece en la sección de resultados del paper de exclusión espectral: *"X% de eventos soportan familia Y"*.

**Criterio de promoción:** Cuando exista `VerdictCalibrationCertificate` para ≥ 5 eventos.

---

### E5-G — Sandbox Científico (aislamiento total)

**Pregunta:** espacio libre para hipótesis, visualizaciones, métricas nuevas sin compromiso de contrato.

**Consume:** copias locales de artefactos (nunca referencias directas).

```bash
cp runs/<run_id>/s4_geometry_filter/compatible_set.json \
   mvp/experiment/sandbox/<nombre>/input_snapshot/
```

**Produce:**

```
mvp/experiment/sandbox/<nombre_con_fecha>/
  input_snapshot/         # copias, no symlinks
  notebooks/              # Jupyter permitido aquí y solo aquí
  scratch/                # sin esquema requerido
  README.md               # hipótesis + resultado
  ISOLATION_MARKER        # archivo vacío: "esto no es un stage"
```

**Único contrato:** presencia de `ISOLATION_MARKER` + **cero imports** de `mvp/`.

```bash
# CI verifica:
grep -r "from mvp" mvp/experiment/sandbox/ && echo "VIOLATION"
grep -r "import mvp" mvp/experiment/sandbox/ && echo "VIOLATION"
```

**Criterio de promoción:** ningún output sale directamente al core. Camino: sandbox → formalizar como E5-X → evaluar promoción.

---

### E5-H — Predicción Ciega Cross-Evento (Leave-One-Out Prediction)

**Pregunta:** ¿puede BASURIN *predecir* las geometrías compatibles de un evento que no ha visto, usando solo los otros N-1?

Esto no es estabilidad (E5-B). Es **poder predictivo** — el estándar de oro en física.

**Consume:** `compatible_set.json` × N (todos `RUN_VALID=PASS`).

**Produce:**

| Artefacto | Contenido |
|-----------|-----------|
| `prediction_summary.json` | Recall, precision, F1 globales + headline publicable |
| `per_event_predictions.json` | Detalle por evento retenido |
| `per_family_predictions.json` | Métricas desagregadas por familia |

**Estrategias de predicción:**

| Estrategia | Definición |
|------------|-----------|
| `intersection` | Predecir solo las geometrías que aparecen en TODOS los N-1 eventos restantes |
| `majority` | Predecir las que aparecen en > 50% de los N-1 |
| `frequency_weighted` | Predecir las que aparecen en ≥ 2 de los N-1 |

**Headline de ejemplo:**

> "BASURIN predicted 78.3% of compatible geometries for GW170817 using only 4 other events (strategy: intersection, F1=0.82)"

**Uso:**

```bash
python mvp/experiment/e5h_blind_prediction.py \
  --run-ids GW150914_v1 GW170817_v1 GW190521_v1 \
  --strategy intersection --dry-run
```

**Requisitos:** ≥ 3 runs.

**Criterio de promoción:** cuando los resultados justifiquen un paper independiente de poder predictivo.

---

### E5-Z — Emulador Continuo de Superficie (Gaussian Process Reconstruction)

**Pregunta:** ¿existe un mínimo oculto entre los nodos de la malla del atlas? ¿dónde está exactamente el "valle de compatibilidad" continuo?

La cura contra el sesgo de discretización. Aplica Gaussian Process Regression sobre el espacio de parámetros físicos de cada familia (espín χ, constante de acoplamiento ζ) usando d² o Δ ln L evaluados en los nodos del atlas.

**El emulador no filtra; interpola.** Reconstruye la superficie continua, predice la coordenada matemática del mínimo real, y proporciona bandas de incertidumbre.

**Consume:** `compatible_set.json` (todas las geometrías, no solo compatibles) + `estimates.json`.

**Produce:**

| Artefacto | Contenido |
|-----------|-----------|
| `predicted_minima.json` | Mínimo continuo inferido por familia |
| `gpr_surface_<family>.json` | Malla fina N×N para ploteo 3D |
| `emulator_manifest.json` | Hiperparámetros del kernel (Matérn, RBF) + R² + hashes |
| `validation_residuals_<family>.json` | Error leave-one-out por geometría |

**Contrato de calidad (self-abort):**

```
Si R² < 0.90 → status: "SURFACE_UNLEARNABLE"
El emulador se aborta. No produce ficciones de baja calidad.
```

**Output killer para revisores:**

```json
{
  "no_hidden_minimum_confidence": {
    "confidence_no_hidden_minimum": 0.997,
    "confidence_level": "VERY_HIGH (3σ)",
    "interpretation": "99.7% confidence that no minimum deeper than 0.38
                       exists between atlas grid nodes for edgb"
  }
}
```

Cuando el referee diga *"su malla de dCS es gruesa, se les escapó una compatible"*, esta es la respuesta.

**Familias soportadas:**

| Familia | Parámetros | Dimensión |
|---------|-----------|-----------|
| `kerr` / `GR_Kerr` | χ (spin) | 1D |
| `edgb` | χ, ζ (coupling) | 2D |
| `dcs` | χ, ζ | 2D |
| `kerr_newman` | χ, q (charge) | 2D |

**Uso:**

```bash
# Emular una familia
python mvp/experiment/e5z_gpr_emulator.py --run-id GW150914_v1 --families kerr edgb

# Emular todas las familias conocidas
python mvp/experiment/e5z_gpr_emulator.py --run-id GW150914_v1

# Usar kernel RBF en vez de Matérn 5/2
python mvp/experiment/e5z_gpr_emulator.py --run-id GW150914_v1 --kernel RBF
```

**Nota epistemológica:** BASURIN core mide realidades numéricas (waveforms que existen). E5-Z crea ficciones estadísticas de alta precisión (waveforms interpoladas no integradas en NR). Mezclar medición directa con predicción estadística destruiría la inmutabilidad del pipeline. Por eso vive en Fase 5.

**Criterio de promoción:** nunca al core. Se promueve a librería hermana (`basurin-surrogate`) cuando inyecciones ciegas demuestren error < 5% respecto a simulación NR completa.

---

## Tabla resumen

| ID | Nombre | Consume | Produce | Aislamiento | Riesgo | Notoriedad |
|----|--------|---------|---------|-------------|--------|------------|
| E5-A | Multi-event aggregation | compatible_set × N | intersection/jaccard | Medio | Calibración previa | Media |
| E5-B | Jackknife stability | compatible_set × N | stability certificates | Medio | N mínimo | Baja |
| E5-C | Geometry ranking | compatible_set + estimates + verdict | ranked list + Gini | Bajo | Pesos arbitrarios | Media-Alta |
| E5-D | Bridge MALDA | compatible_set + estimates + **external** | payload traducido | Bajo | Schema externo | Alta |
| E5-E | Query reproducible | compatible_set × N (read-only) | query_cache | Muy bajo | Ninguno | Baja |
| E5-F | Verdict aggregation | verdict.json × N | population_verdict | Muy bajo | Veredictos no calibrados | **Muy Alta** |
| E5-G | Sandbox científico | Copias locales | Sin contrato | Total | Ninguno | — |
| E5-H | Blind prediction | compatible_set × N | recall/precision/F1 | Medio | N mínimo | **Muy Alta** |
| E5-Z | GPR emulator | compatible_set + estimates | surface + minima | Bajo | Ficción estadística | **Máxima** |

---

## Tests de gobernanza

Archivo: `tests/test_e5_governance.py` — **30 tests**.

Cubren:

| Categoría | Tests |
|-----------|-------|
| Base contract (RUN_VALID gate) | 4 |
| E5-E idempotencia + queries | 4 |
| E5-F agregación + rechazo | 2 |
| E5-A intersección + mínimos | 2 |
| E5-B jackknife + mínimos | 2 |
| E5-C ranking + invariante | 2 |
| E5-H predicción + perfección | 3 |
| E5-Z GPR (1D, 2D, noisy, hidden, multi) | 9 |
| Sandbox aislamiento | 2 |

Ejecutar:

```bash
python -m pytest tests/test_e5_governance.py -v
```

---

## Secuencia recomendada de ejecución

```
Semana 1:  E5-E (infraestructura, cero riesgo)
           └── Explorar sensibilidad de thresholds sobre runs actuales

Semana 2:  E5-F (resultados de paper)
           └── Tasas de soporte por familia → sección de resultados
           E5-Z (la figura espectacular)
           └── Superficies GPR → argumento contra discretización

Semana 3:  E5-H (poder predictivo)
           └── Headlines publicables de predicción ciega

Semana 4:  E5-C (Lorenz/Gini para figuras memorables)
           E5-D (validación contra bayesiano para referees)

Continuo:  E5-A + E5-B (prerequisites para claims poblacionales)
           E5-G (sandbox para hipótesis nuevas)
```

---

## Motivo global para mantener todo fuera del core

El pipeline canónico (s1→s8) mide realidades: waveforms que existen, distancias que se calculan, intersecciones que se computan. Cada stage tiene un contrato determinista y auditable.

La Fase 5 opera sobre las *conclusiones* de esas mediciones. Agregar, rankear, predecir, interpolar — son operaciones sobre artefactos, no sobre datos. Mezclar meta-análisis con medición directa destruiría la pureza epistemológica del pipeline.

La separación no es una limitación. Es la mayor fortaleza de BASURIN: la disciplina de saber qué es un dato y qué es una inferencia sobre datos.

---

## Inventario de archivos

| Archivo | Líneas | Dependencias externas |
|---------|--------|-----------------------|
| `base_contract.py` | 139 | — |
| `e5a_multi_event_aggregation.py` | 186 | — |
| `e5b_jackknife.py` | 178 | — |
| `e5c_ranking.py` | 238 | — |
| `e5d_bridge_malda.py` | 177 | — |
| `e5e_query.py` | 200 | — |
| `e5f_verdict_aggregation.py` | 199 | — |
| `e5h_blind_prediction.py` | 255 | — |
| `e5z_gpr_emulator.py` | 725 | `scikit-learn`, `scipy` |
| **Total** | **2,297** | |

---

## Notas de gobernanza

Las siguientes reglas se aplican a todos los módulos E5 sin excepción:

1. **Dependencia de `RUN_VALID`:** ningún módulo puede ejecutarse sobre un `run_id` cuyo `RUN_VALID/verdict.json` no tenga `verdict: PASS`. La excepción propaga `GovernanceViolation`; no existe modo "lax".

2. **Lectura sin mutación de artefactos canónicos:** los módulos E5 son lectores. Consumen artefactos ya emitidos por stages canónicos (`s3`, `s4`, `s4k`, etc.) pero no los reescriben, sobreescriben ni crean versiones paralelas en la ruta canónica.

3. **Escritura solo en el namespace experimental del run:** todos los outputs van bajo `runs/<run_id>/experiment/<nombre_experimento>/`. Nunca se escribe fuera de ese árbol. CI lo verifica.

4. **Separación código / artefactos:** el código fuente de los módulos E5 vive en `mvp/experiment/` (paquete del repositorio). Los artefactos producidos por ejecución viven en `runs/<run_id>/experiment/` (árbol de run). Estas dos rutas son distintas y no deben confundirse.

5. **Código experimental no es pipeline canónico:** `mvp/experiment/` es un espacio experimental gobernado, no el pipeline canónico. El pipeline canónico es `s1 → s8` en `mvp/`. Los módulos E5 operan sobre conclusiones de ese pipeline, no forman parte de él.

---

## Estado actual (2026-03-18)

- 9 módulos experimentales implementados en `mvp/experiment/`
- El paquete top-level `./experiment` fue retirado; el namespace canónico vigente es `mvp/experiment`
- 30/30 tests de gobernanza pasando (`tests/test_e5_governance.py`)
- Contrato universal de entrada (`mvp/experiment/base_contract.py`) operativo
- E5-Z validado con superficies 1D (Kerr) y 2D (EdGB) sintéticas
- Sandbox aislado verificado (cero imports de `mvp/`)
