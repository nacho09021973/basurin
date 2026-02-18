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

## Dónde mirar primero al depurar

1. `runs/<run_id>/pipeline_timeline.json`
2. `runs/<run_id>/<stage>/stage_summary.json`
3. `runs/<run_id>/<stage>/manifest.json`
4. `docs/readme_rutas.md` para validar root efectivo y subruns.

## Estado del proyecto

Este repositorio prioriza reproducibilidad y trazabilidad del MVP sobre ergonomía de packaging.
Si vas a automatizar tareas con IA, usa este README + `docs/readme_rutas.md` como contexto base mínimo.
