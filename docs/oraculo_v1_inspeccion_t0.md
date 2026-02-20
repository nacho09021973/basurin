# Oráculo v1 — inspección previa (sin implementación)

Resumen operativo del estado actual de `experiment_t0_sweep_full` para preparar la integración de selección de `t0`.

## Hallazgos clave

- El barrido gobernado de `t0` vive en `mvp/experiment_t0_sweep_full.py` y expone fases `run`, `inventory` y `finalize` vía `--phase`.
- `run` materializa subruns aislados por punto de grilla (`<run_id>__t0msNNNN`) y ejecuta `s2 -> s3 -> s3b -> s4c`.
- `inventory/finalize` no dependen de atlas; construyen y escriben `runs/<run_id>/experiment/derived/sweep_inventory.json`.
- No existe aún `oracle_report` dentro del finalize de `t0_sweep_full`; la salida de finalize es el inventario con `status` y `decision`.
- Los campos tipo `f_median`, `tau_median`, `delta_bic`, `p_ljungbox` no aparecen hoy en `stage_summary.json` del sweep/subruns.

## Rutas a integrar para Oráculo v1

- Entrada agregada del sweep: `runs/<run_id>/experiment/t0_sweep_full_seed<seed>/outputs/t0_sweep_full_results.json`.
- Evidencia por subrun: `runs/<run_id>/experiment/t0_sweep_full_seed<seed>/runs/<subrun_id>/{s2,s3,s3b,s4c}/...`.
- Inventario/finalize consolidado: `runs/<run_id>/experiment/derived/sweep_inventory.json`.
- Diagnóstico opcional pre-oráculo: `runs/<run_id>/experiment/derived/diagnose_report.json`.
- Trazas de ejecución: `runs/<run_id>/experiment/derived/run_trace.json` y `.../<subrun_id>/derived/subrun_trace.json`.
