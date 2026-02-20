# BASURIN

Framework MVP para análisis de *ringdown* (ondas gravitacionales) con ejecución por etapas, contratos de artefactos y trazabilidad de resultados en disco.

> Este README está optimizado para que tanto personas como asistentes de IA puedan entender el proyecto rápidamente y operar sin romper las reglas de IO/auditoría.

## TL;DR para una IA (quickstart operativo)

1. **Lee primero este archivo completo** para entender objetivos, etapas y convenciones.
2. **Consulta el mapa de rutas**: [`docs/readme_rutas.md`](docs/readme_rutas.md) (crítico para `RUNS_ROOT`, subruns y experimentos anidados).
3. Ejecuta con `python mvp/pipeline.py ...` y asume semántica *fail-fast* (si una etapa falla, el pipeline aborta).
4. No escribas fuera de `runs/<run_id>/...` (o del `BASURIN_RUNS_ROOT` efectivo).
5. Antes de proponer cambios, revisa tests en `tests/` y contratos en `mvp/contracts.py`.

---

## ¿Qué es BASURIN?

BASURIN implementa un pipeline reproducible para análisis de ringdown con estas propiedades:

- **Ejecución por etapas** (`s1`, `s2`, `s3`, `s4`, ...).
- **Artefactos explícitos por stage** (`manifest.json`, `stage_summary.json`, `outputs/...`).
- **Gobernanza por run** con `RUN_VALID/verdict.json`.
- **Modo evento único y multi-evento** desde un orquestador central.

## Arquitectura de alto nivel

- Orquestador principal: `mvp/pipeline.py`.
- Etapas del MVP (núcleo):
  - `mvp/s1_fetch_strain.py`
  - `mvp/s2_ringdown_window.py`
  - `mvp/s3_ringdown_estimates.py`
  - `mvp/s4_geometry_filter.py`
- Etapas/experimentos avanzados: `s3b`, `s4b`, `s4c`, `s5`, `s6`, `experiment_*`.

## Ejecución básica

### Prerrequisitos mínimos del entorno Python

Antes de ejecutar `pipeline.py` o cualquier stage del MVP, valida que estén disponibles
las dependencias base del flujo real (`numpy`) y del fetch de strain (`requests`, `gwpy`):

```bash
python - << 'PY'
mods = ["numpy", "requests", "gwpy"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(f"Faltan dependencias: {', '.join(missing)}")
print("OK: entorno listo para s1/s2/s3")
PY
```

Para modo local con `--local-hdf5`, añade además `h5py` a la verificación.

### 1) Single-event

```bash
python mvp/pipeline.py single \
  --event-id GW150914 \
  --atlas-path atlas.json
```

### 2) Multi-event

```bash
python mvp/pipeline.py multi \
  --events GW150914,GW151226 \
  --atlas-path atlas.json
```

## Experimentos (`t0_sweep_full`) e inventario por fases

- `mvp/experiment_t0_sweep_full.py`: **recomendado para escala/producción**, contract-first con fases `run`/`inventory`/`finalize`.
- `mvp/experiment_t0_sweep.py`: herramienta **DEV/INTEGRATION** para sanity-check rápido; no contract-first, no inventory/finalize, no aislamiento por subrun. **No usar para conteos oficiales**.

Cuándo usar cada uno:
- Usa `t0_sweep_full` cuando necesites reproducibilidad auditada, gobernanza por fases y barridos escalables.
- Usa `t0_sweep` simple solo para validar wiring/integración local rápida sobre un run ya preparado.

`experiment_t0_sweep_full.py` soporta fases explícitas con contrato estable:

- `--phase run`: ejecuta subruns del barrido y mantiene semántica fail-fast.
- `--phase inventory`: escaneo barato/determinista de completitud (sin atlas).
- `--phase finalize`: aplica gating final por faltantes (sin atlas).

### a) Sweep completo (`phase=run`, aquí sí va `--atlas-path`)

```bash
RUN_ID="mvp_GW150914_20260219T120000Z"
ATLAS_PATH="/ruta/al/atlas"

python mvp/experiment_t0_sweep_full.py \
  --run-id "$RUN_ID" \
  --phase run \
  --atlas-path "$ATLAS_PATH" \
  --t0-grid-ms 0,2,4,6,8 \
  --seed 101
```

### b) Inventario barato (`phase=inventory`, sin atlas)

```bash
RUN_ID="mvp_GW150914_20260219T120000Z"

python mvp/experiment_t0_sweep_full.py \
  --run-id "$RUN_ID" \
  --phase inventory \
  --inventory-seeds 101,202 \
  --t0-grid-ms 0,2,4,6,8
```

### c) Finalize estricto/tolerante (`phase=finalize`, sin atlas)

```bash
RUN_ID="mvp_GW150914_20260219T120000Z"

python mvp/experiment_t0_sweep_full.py \
  --run-id "$RUN_ID" \
  --phase finalize \
  --inventory-seeds 101,202 \
  --t0-grid-ms 0,2,4,6,8 \
  --max-missing-abs 0 \
  --max-missing-frac 0.0
```

Artefactos agregados a vigilar:

- `runs/<RUN_ID>/experiment/derived/geometry_table.tsv` (agregado de `s6_geometry_table.py`).
- `runs/<RUN_ID>/experiment/derived/sweep_inventory.json` (inventario/decisión de sweep).

Nota sobre subruns por seed: el experimento crea árboles por semilla para aislar trazabilidad y reintentos. Por eso, los agregados deben escanearse desde `scan_root` (global o por seed) y no por prefijos de nombre; además, se excluyen ancestros symlink para evitar duplicados/alias.

## Evitar descargas repetidas de GW150914/GW150904 (modo offline recomendado)

Si ya tienes los HDF5 completos en local (caso típico: repetir experimentos con el mismo evento),
**no hace falta volver a descargar desde GWOSC** en cada corrida.

Usa `s1_fetch_strain` en modo local:

```bash
python mvp/s1_fetch_strain.py \
  --run <run_id> \
  --event-id GW150914 \
  --detectors H1,L1 \
  --duration-s 32 \
  --local-hdf5 H1=/ruta/local/H-H1_GWOSC_*.h5 \
  --local-hdf5 L1=/ruta/local/L-L1_GWOSC_*.h5 \
  --reuse-if-present
```

Notas prácticas:

- `--local-hdf5` evita la descarga remota y fuerza lectura desde tus `.h5/.hdf5`.
- `--reuse-if-present` evita repetir trabajo si `outputs/strain.npz` + `provenance.json`
  ya coinciden con `event_id`, detectores y hashes.
- Al ejecutar en modo local, BASURIN deja copia auditable de los HDF5 usados en:
  - `runs/<run_id>/s1_fetch_strain/inputs/*.h5`
  y la referencia en:
  - `runs/<run_id>/s1_fetch_strain/outputs/provenance.json` (`local_inputs`, `local_input_sha256`).

En resumen: para experimentación iterativa con GW150914/GW150904, apunta siempre a tus
HDF5 locales y reutiliza artefactos para no saturar red ni perder tiempo.

## Semántica operacional importante

- El pipeline crea `RUN_VALID` al inicializar el run.
- Cada stage escribe su propio bloque de artefactos bajo el run activo.
- Si un stage retorna código distinto de cero, el pipeline **aborta inmediatamente**.
- Se mantiene trazabilidad temporal en `pipeline_timeline.json`.

## Layout esperado de artefactos

Ejemplo resumido:

```text
runs/<RUN_ID>/
  RUN_VALID/verdict.json
  s1_fetch_strain/{manifest.json,stage_summary.json,outputs/...}
  s2_ringdown_window/{manifest.json,stage_summary.json,outputs/...}
  s3_ringdown_estimates/{manifest.json,stage_summary.json,outputs/...}
  ...
```

## Documento clave de rutas (lectura obligatoria)

Cuando trabajes con subruns (por ejemplo en `experiment/t0_sweep_full`), es fácil romper la ejecución si no se entiende dónde vive realmente cada `run_id`.

➡️ **Lee este documento antes de depurar o ejecutar stages manuales:**

- [`docs/readme_rutas.md`](docs/readme_rutas.md)
- [`docs/request_flow.md`](docs/request_flow.md) (resumen del flujo de ejecución y validaciones por módulo).

Ese documento detalla:

- cómo se resuelve `BASURIN_RUNS_ROOT`,
- por qué ocurre el error típico de `RUN_VALID verdict not found`,
- y cómo ejecutar stages contra subruns de forma correcta.

## Convenciones para contribuir (humanos e IA)

- Mantener compatibilidad con contratos/artefactos existentes.
- Evitar cambios que relajen validaciones de `RUN_VALID` sin discusión explícita.
- Priorizar cambios pequeños y verificables.
- Incluir pruebas o validaciones mínimas al modificar lógica de pipeline/stages.
- Documentar decisiones de rutas/IO cuando afecten reproducibilidad.

## Reglas generales de BASURIN (para no volver a iterar a ciegas)

### 0) Principio soberano

**El pipeline existe para producir física reproducible, no para “tener pipeline”.**
Todo cambio debe cerrar un ciclo completo: **Inputs deterministas → Estimación → Oracle externo → Veredicto PASS/FAIL → Auditoría**.

---

## 1) “Oracle-first”: sin oracle externo no hay progreso

**Definición:** un *oracle* es un baseline independiente (paper/catálogo/tabla publicada) con valores y/o intervalos reproducibles y una regla cuantitativa de PASS/FAIL.

**Regla:**

* Cualquier nuevo análisis canónico debe declarar un oracle externo antes de añadir complejidad.
* Si no existe oracle, el trabajo debe ir a `experiment/` y **no** al pipeline canónico.

**Artefactos mínimos del oracle (versionados):**

* `docs/baselines/<oracle_id>.json` con:

  * referencia (paper/doi/arxiv),
  * parámetros objetivo (p.ej. f, tau, …),
  * intervalos o incertidumbres,
  * tolerancias (relativas/absolutas),
  * versión/fecha.

**Criterio de aceptación:**

* Existe un `physics_gate.json` (o equivalente) que da `PASS/FAIL` contra el oracle.
* En `FAIL`, el run se invalida (ver regla 3).

---

## 2) MVP científico mínimo (antes de fase 2)

**Regla:**

* El MVP científico de cualquier nuevo “tema” es: **1 caso + 1 observable + 1 oracle + PASS/FAIL**.
* Multi-evento, multimode, consistencia Kerr, geometría, agregación, sweeps masivos… son **fase 2** y quedan bloqueados hasta que el MVP pase.

**Anti-pattern prohibido:**

* “Añadir un stage nuevo” sin un test que demuestre que mejora o mantiene el oracle PASS.

---

## 3) RUN_VALID y abort semantics (gobernanza dura)

**Regla soberana:**

* `RUN_VALID` es la **única puerta** hacia downstream.
* Si `RUN_VALID != PASS` o falta `runs/<run_id>/RUN_VALID/verdict.json`, ningún stage downstream puede correr.
* Si un stage falla, el run **no existe** a efectos downstream (fail-fast).

**Criterio de aceptación:**

* Cada stage ejecuta `require_run_valid` (salvo excepciones explícitas).
* El pipeline corta en rc != 0 sin intentar “continuar”.

---

## 4) IO determinista: prohibido escribir fuera de `runs/<run_id>/`

**Regla:**

* Toda salida y todo artefacto intermedio debe vivir bajo `runs/<run_id>/...` (o `BASURIN_RUNS_ROOT`).
* **Prohibido** escribir en `docs/`, `./`, `/tmp`, home, etc., salvo:

  * documentación versionada (README/docs),
  * baselines versionados en `docs/baselines/`,
  * tests.

**Criterio de aceptación:**

* Cada stage produce:

  * `manifest.json` (SHA256 de inputs/outputs),
  * `stage_summary.json` (parámetros, veredictos, métricas),
  * `outputs/` (artefactos).

---

## 5) Datos externos: “offline-first” y inputs formalizados

**Regla:**

* Ningún stage puede depender de red “por defecto”.
* Los inputs externos deben formalizarse como:

  * `--local-*` / `--atlas-path` / `--external-inputs` (rutas explícitas),
  * o como `runs/<run_id>/external_inputs/...` con hashes.

**Política recomendada:**

* **Red sólo con opt-in** (`--allow-network`) y siempre cacheando con hash.
* Si `OFFLINE=1` y falta input local ⇒ FAIL temprano con mensaje claro.

**Criterio de aceptación:**

* Se puede ejecutar el pipeline completo en un entorno sin red si los inputs existen localmente.

---

## 6) Superficie canónica mínima: separación estricta canónico vs experimento

**Reglas:**

* Canónico:

  * vive en `mvp/` + `contracts.py`,
  * pequeño, estable, orientado a oracle.
* Experimentos:

  * viven bajo `runs/<run_id>/experiment/<name>/...`,
  * no mutan artefactos canónicos,
  * no cambian `RUN_VALID`.

**Anti-pattern prohibido:**

* “s3b/s4b/s4c…” sin deprecación clara y sin test oracle.
* Scripts sueltos sin contrato ni trazabilidad.

---

## 7) Sweeps y “1000 runs”: solo si hay función objetivo y logging auditable

**Regla:**

* Un sweep masivo sólo es aceptable si existe:

  * oracle PASS/FAIL o loss cuantitativa,
  * logging por subrun con seed/params,
  * resumen agregado (p.ej. `summary.jsonl` o `summary.csv`),
  * selección reproducible del “best”.

**Criterio de aceptación:**

* Dado el mismo seed y mismos inputs hash, el sweep produce el mismo “best” y el mismo veredicto.

---

## 8) Preflight de entorno (Stage 0) obligatorio

**Regla:**

* Antes de crear un run se valida:

  * Python version,
  * dependencias críticas importables,
  * presencia de inputs externos requeridos (si OFFLINE),
  * espacio en disco mínimo.

**Criterio de aceptación:**

* Si falla preflight, no se crea `runs/<run_id>/` (o se crea y queda `RUN_VALID=FAIL` con razón explícita).

---

## 9) Regla práctica de productividad (evitar 14 meses)

**Regla:**

* Cada PR debe mejorar al menos uno de:

  1. probabilidad de oracle PASS,
  2. determinismo/auditoría,
  3. reducción de superficie canónica.

Si no cumple, va a `experiment/` o se rechaza.

---

### Apéndice: Definition of Done (DoD) para cambios canónicos

Un cambio canónico está “DONE” si:

* `pytest -q` pasa,
* el oracle PASS/FAIL está implementado (o no se degrada),
* no se introducen escrituras fuera de `runs/<run_id>/`,
* todos los artefactos tienen `manifest.json` + `stage_summary.json`,
* la ejecución offline es posible con inputs locales.

---

## Diagnóstico de crecimiento en tests (actualización de gobernanza)

Correcto: con el criterio “desde hoy 12:00” y contando **altas** de `*.py`, son **3 scripts nuevos** y el conteo está bien hecho.

### Diagnóstico

Estamos pagando “deuda de gobernanza” (CLI + contratos) a base de **añadir tests unitarios**. Eso es saludable, pero el riesgo real es el señalado: **crecimiento de superficie** (más ficheros, más mantenimiento, más tiempo de CI) sin una estrategia de consolidación.

Los tres añadidos son de **tres áreas distintas** (experimento, pipeline/CLI, contrato s2). Eso suele ocurrir cuando no existe aún una convención fuerte de “dónde vive cada test” y “cuándo se amplía uno existente vs crear fichero nuevo”.

### Acción mínima (para no “engordar al monstruo”)

1. **Regla operativa**: por defecto, **no se crea fichero nuevo** si:
   - el test es del mismo stage/feature, o
   - es el mismo tipo (CLI help, contract, smoke determinista).
   En ese caso se **agrega al fichero existente**.
2. **Consolidación inmediata** (sin reescribir medio repo):
   - `tests/test_pipeline_cli_local_hdf5.py`: convertirlo en el hogar de *todo* lo relacionado con CLI de `mvp/pipeline.py` (help flags, atlas-default fail, passthrough, etc.).
   - `tests/test_s2_ringdown_window_contract_unittest.py`: mantenerlo como “contract tests” de s2, pero evitar duplicar utilidades; extraer helpers a `tests/_util_contract.py` si ya hay repetición (un único helper compartido vale más que tres copias).
   - `tests/test_experiment_t0_sweep_full_diagnose_unittest.py`: si esto es “experiment”, idealmente debería vivir bajo un patrón estable (p.ej. `tests/experiments/test_t0_sweep_full_*.py`) o convertirse en un **smoke test** más pequeño. Los experiments tienden a crecer sin control.
3. **Presupuesto de tests por cambio** (pragmático):
   - Cambios de parser/CLI: 1 fichero.
   - Cambios de contrato por stage: 1 fichero por stage.
   - Experimentos: 1 fichero por experimento (y preferir asserts de “manifiesto/gating”, no loops gigantes).

### Comandos útiles (para vigilar crecimiento)

Archivos añadidos hoy:

```bash
git log --since='today 12:00' --name-status --pretty=format: \
| awk '$1=="A" && $2 ~ /\.py$/ {print $2}' | sort -u
```

“Hotspots” (muchos ficheros tocados por commit):

```bash
git show --stat b08a7b7
```

### Riesgos / supuestos

- Si esos tests nuevos están evitando regresiones reales (lo parecen), no es “basura”: es **control de daños**.
- El peligro no es crear 3 ficheros una vez; es no poner una regla y acabar con 80 ficheros micro-especializados.

Si hace falta, se puede convertir esta guía en una convención concreta de estructura `tests/` (nombres + carpetas + helpers) con criterio de “cuándo crear fichero nuevo”.

## Dónde mirar primero al depurar

1. `runs/<run_id>/pipeline_timeline.json`
2. `runs/<run_id>/<stage>/stage_summary.json`
3. `runs/<run_id>/<stage>/manifest.json`
4. `docs/readme_rutas.md` para validar root efectivo y subruns.

Nota breve de shell: el comando `.` es `source` de bash; si aparece `-bash: .: filename argument required`, es un error de shell, no de BASURIN.

## Estado del proyecto

Este repositorio prioriza reproducibilidad y trazabilidad del MVP sobre ergonomía de packaging.
Si vas a automatizar tareas con IA, usa este README + `docs/readme_rutas.md` como contexto base mínimo.
