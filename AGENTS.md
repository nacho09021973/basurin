# AGENTS.md — Instrucciones para Codex (BASURIN)

## Raiz canonica del repo
- La unica raiz canonica de trabajo es `/home/adnac/basurin/work/basurin`.
- `/home/adnac/basurin/work` no es un repo operable para desarrollo normal: no usarlo para `git`, edicion, tests ni ejecucion del pipeline.
- La carpeta contenedora `/home/adnac/basurin/work` debe contener solo `basurin/`.
- Antes de cambiar codigo o ejecutar comandos sensibles, verificar la raiz con `git rev-parse --show-toplevel`.

## Reglas duras de gobernanza
- IO determinista: prohibido escribir fuera de `runs/<run_id>/...` (o `BASURIN_RUNS_ROOT` si está definido).
- Resolución de raíz: usar `basurin_io.resolve_out_root("runs")` (respeta `BASURIN_RUNS_ROOT`).
- Gating: nada downstream si `RUN_VALID != PASS` (usar `require_run_valid` cuando aplique).
- Abort semantics: si un stage falla, el run “no existe” a efectos downstream.

## Cache LOSC canónica
- Antes de tocar catálogo/descargas de strain, leer `README.md` y `docs/readme_rutas.md` en las secciones de `data/losc`/offline-first.
- La unica vista canónica que puede consumir el pipeline es `data/losc/<EVENT_ID>/`.
- Si la caché real vive en `gw_events/strain/<EVENT_ID>/`, exponerla bajo `data/losc/<EVENT_ID>/` con symlink o bind mount; no redirigir el pipeline a `gw_events/strain`.
- No inventar descargadores ad hoc si ya existe tooling en `tools/`.
- Para visibilidad/naming usar `tools/losc_precheck.py`.
- Para un evento puntual usar `tools/fetch_losc_event.py`.
- Para varios eventos faltantes o rezagados usar `tools/fetch_losc_batch.sh`.
- `tools/download_gw_events.py` y `tools/fetch_catalog_events.py` se reservan para bootstrap/cohorte amplia, no para reparar huecos puntuales de una caché ya montada.

## Contrato de artefactos por stage/experimento
Cada stage/experimento debe producir bajo su directorio:
- `manifest.json`
- `stage_summary.json`
- `outputs/*`
y reflejar hashes SHA256 de outputs en manifest/summary.

### Excepción documentada (helper no-stage)
- `mvp/extract_psd.py` (`psd_extract`) es un helper de preparación, no un stage canónico del pipeline.
- `s6c_brunete_psd_curvature` soporta fallback runtime de PSD vía `external_inputs/psd_model.json` cuando no existe `psd/measured_psd.json`.
- Por contract-first SSOT, no modelar `psd/measured_psd.json` como dependencia rígida de stage cuando ese fallback esté habilitado.

## Rutas canónicas (no adivinar)
- Stage dir canónico: `<RUNS_ROOT>/<run_id>/<stage>/...`
- Experimento canónico: `<RUNS_ROOT>/<run_id>/experiment/<name>/...`
- Nunca escribir a rutas absolutas hardcodeadas (p.ej. `/home/...`).

## Logging obligatorio anti-pérdida-de-tiempo
Al final de cualquier entrypoint que escriba outputs, imprimir al menos:
- `OUT_ROOT=...`
- `STAGE_DIR=...`
- `OUTPUTS_DIR=...`
- `STAGE_SUMMARY=...`
- `MANIFEST=...`

## CLI anti-ambigüedad
- `--seed` (int) y `--seed-dir` (path) son argumentos distintos; nunca interpretar un entero como path.
- En errores por inputs faltantes, incluir:
  1) ruta esperada exacta
  2) comando exacto para regenerar upstream
  3) (si aplica) lista de candidatos detectados

## Logging obligatorio — helper canónico
Usar `contracts.log_stage_paths(ctx)` al final de cada entrypoint que escriba outputs.
Emite exactamente las cinco variables requeridas: `OUT_ROOT`, `STAGE_DIR`, `OUTPUTS_DIR`, `STAGE_SUMMARY`, `MANIFEST`.

## Tests obligatorios
- Unit tests: determinismo, bordes, sigma_floor/scale_floor cuando aplique.
- Integration-lite: ejecutar con `BASURIN_RUNS_ROOT` temporal y verificar que NO se escribe fuera.
- Golden tests: snapshots de outputs normalizados para evitar regresiones silenciosas.

## Regla de consolidación de tests (presupuesto por cambio)
Por defecto **no crear fichero nuevo** si el test es del mismo stage/feature o del mismo tipo
(CLI help, contract, smoke determinista): agregar al fichero existente.

Presupuesto máximo de ficheros nuevos por PR:
- Cambios parser/CLI → 1 fichero (usar/ampliar `tests/test_pipeline_cli_*.py`).
- Cambios de contrato por stage → 1 fichero por stage (`tests/test_<stage>_contract_*.py`).
- Experimentos → 1 fichero por experimento; preferir asserts de manifiesto/gating.

Helpers compartidos → `tests/_util_*.py` (no duplicar utilidades entre ficheros).
