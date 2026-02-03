# BASURIN

## Documento soberano (single source of truth)
- **BASURIN_README_SUPER.md**

> Este README es un **resumen ejecutivo no normativo**.  
> Si contradice `BASURIN_README_SUPER.md`, el soberano manda y este texto debe considerarse inválido.

Toda la documentación anterior se conserva en `docs/_historical/` solo con propósito forense/histórico.

---

## Resumen ejecutivo (contract-first)

### Principio no negociable
- **Si falla un contrato ejecutivo, el run no existe.**
- No existen “resultados interesantes” de runs inválidos. Si un contrato falla → abort.

### IO determinista
- Todo vive bajo `runs/<run_id>/`.
- **Prohibido** escribir fuera de `runs/<run_id>/`.
- Cada stage escribe como mínimo:

runs/<run_id>/<stage_name>/
├── manifest.json
├── stage_summary.json
└── outputs/


### Contrato soberano de existencia: `RUN_VALID`
- Antes de ejecutar cualquier downstream, debe cumplirse `runs/<run_id>/RUN_VALID/ == PASS`.
- Si `RUN_VALID` falla: abortar pipeline; run inválido por definición.

### Stages canónicos (v1)
- `geometry`
- `spectrum` (produce `spectrum.h5`)
- `dictionary` (no gobierna existencia)
- `ringdown_synth` (único generador autorizado de sintéticos ringdown; índice canónico `synthetic_events.json`)
- `RUN_INDEX` (inventario/auditoría; gated por `RUN_VALID==PASS`)

### Regla dura para experimentos
- Viven en `runs/<run_id>/experiment/<name>/`.
- No generan sintéticos nuevos.
- No reconstruyen rutas “a mano”: consumen índices canónicos (p.ej. `synthetic_events.json`).
- Abort si `RUN_VALID != PASS`.

---

## Referencia normativa
Lee y sigue **BASURIN_README_SUPER.md**. Este README no sustituye contratos ni especificación normativa.
