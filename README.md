# BASURIN

## Documento soberano (single source of truth)
- **BASURIN_README_SUPER.md**

> “Este README es un resumen ejecutivo no normativo. Si contradice BASURIN_README_SUPER.md, el soberano manda y este texto debe considerarse inválido.”

## Resumen ejecutivo (contract-first)
- Principio: “Si falla un contrato ejecutivo, el run no existe; abort; no hay resultados interesantes de runs inválidos.”
- IO determinista bajo `runs/<run_id>/`, prohibido escribir fuera, y estructura mínima de stage:
  - `runs/<run_id>/<stage_name>/{manifest.json, stage_summary.json, outputs/}`
- Contrato soberano: `RUN_VALID` debe ser `PASS` antes de downstream; si falla, abort.
- Stages canónicos (v1):
  - `geometry`
  - `spectrum` (produce `spectrum.h5`)
  - `dictionary` (no gobierna existencia)
  - `ringdown_synth` (único generador autorizado de sintéticos ringdown; índice canónico `synthetic_events.json`)
  - `RUN_INDEX` (inventario/auditoría; gated por `RUN_VALID==PASS`)
- Regla de experimentos:
  - viven en `runs/<run_id>/experiment/<name>/`
  - no generan sintéticos nuevos
  - no reconstruyen rutas “a mano”; consumen índices canónicos (p.ej. `synthetic_events.json`)
  - abort si `RUN_VALID != PASS`

Referencia normativa: lee y sigue BASURIN_README_SUPER.md; este README no sustituye contratos.
