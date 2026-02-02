# BASURIN Ops: Entrypoints y rutas canónicas (no normativo)

Este documento es OPERATIVO. No redefine gobernanza. Para normativa soberana ver BASURIN_README_SUPER.md.

## Objetivo
Evitar “búsqueda manual” de:
- entrypoints ejecutables (stages/ y experiment/**/stage_*.py),
- rutas canónicas de inputs/outputs bajo runs/<run_id>/.

## Convenciones
- Todo artefacto debe vivir bajo: runs/<run_id>/
- Un run "existe" sólo si RUN_VALID == PASS.
- Los experimentos deben abortar downstream si falta un input canónico.

## Comando único de estado
- Comando:
  python tools/basurin_where.py --run "$RUN_ID" --ringdown-min
- Nota: si READY: NO, estás bloqueado upstream.

---

## Mapa mínimo: Ringdown pre-real (00..01)

### RUN_VALID (soberano)
- Entrypoint: experiment/run_valid/stage_run_valid.py
- Salida mínima:
  - runs/<run_id>/RUN_VALID/verdict.json
  - runs/<run_id>/RUN_VALID/stage_summary.json
  - runs/<run_id>/RUN_VALID/manifest.json

### ringdown_synth (evento sintético canónico)
- Entrypoint: stages/ringdown_synth_stage.py
- Inputs: (params CLI; determinista por seed)
- Salida mínima:
  - runs/<run_id>/ringdown_synth/outputs/synthetic_event.json
  - runs/<run_id>/ringdown_synth/stage_summary.json
  - runs/<run_id>/ringdown_synth/manifest.json

> Nota: en docs puede existir la convención "synthetic_events.json" (plural). Si el código consume "synthetic_event.json" (singular), eso manda.
  El tool basurin_where reporta el nombre requerido por los scripts actuales.

### EXP_RINGDOWN_00 stability sweep
- Entrypoint: experiment/ringdown/exp_ringdown_00_stability_sweep.py
- Inputs canónicos requeridos por el script:
  - runs/<run_id>/RUN_VALID/verdict.json (PASS)
  - runs/<run_id>/ringdown_synth/outputs/synthetic_event.json
- Outputs esperados bajo stage_dir (definido por ensure_stage_dirs):
  - .../outputs/sweep_plan.json
  - .../outputs/diagnostics.json
  - .../outputs/contract_verdict.json
  - .../outputs/per_case/case_*.json
  - .../stage_summary.json
  - .../manifest.json

### EXP_RINGDOWN_01 injection-recovery
- Entrypoint: experiment/ringdown/exp_ringdown_01_injection_recovery.py
- Config determinista:
  - experiment/ringdown/exp_ringdown_01_grid.json
- Inputs: RUN_VALID + ringdown_synth (o batch sintético si lo implementa)
- Outputs: contract_verdict + tabla truth/est + manifest + stage_summary

---

## Inventario de entrypoints en este repo (resumen)
- stages:
  - stages/ringdown_synth_stage.py
  - stages/stage_hsc_input.py
  - stages/stage_run_index.py
- experiment/run_valid:
  - experiment/run_valid/stage_run_valid.py
- ringdown:
  - experiment/ringdown/exp_ringdown_00_stability_sweep.py
  - experiment/ringdown/exp_ringdown_01_injection_recovery.py
  - experiment/ringdown/exp_ringdown_qnm_00_open_bc.py
  - experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py
