# AGENTS.md — Instrucciones para Codex (BASURIN)

## Reglas duras de gobernanza
- IO determinista: prohibido escribir fuera de `runs/<run_id>/...` (o `BASURIN_RUNS_ROOT` si está definido).
- Resolución de raíz: usar `basurin_io.resolve_out_root("runs")` (respeta `BASURIN_RUNS_ROOT`).
- Gating: nada downstream si `RUN_VALID != PASS` (usar `require_run_valid` cuando aplique).
- Abort semantics: si un stage falla, el run “no existe” a efectos downstream.

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

## Tests obligatorios
- Unit tests: determinismo, bordes, sigma_floor/scale_floor cuando aplique.
- Integration-lite: ejecutar con `BASURIN_RUNS_ROOT` temporal y verificar que NO se escribe fuera.
- Golden tests: snapshots de outputs normalizados para evitar regresiones silenciosas.
