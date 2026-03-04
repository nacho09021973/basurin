# Experimento 6 — Teorema del Área (ΔA)

## Objetivo
Calcular, por evento, la distribución de:

\[
\Delta A = A_f - (A_1 + A_2)
\]

con:
- \(A_f\): área final post-fusión inferida desde intersección física Kerr/Berti de los batches 220 y 221.
- \(A_1 + A_2\): área pre-fusión obtenida **exclusivamente** de posterior samples IMR (`m1_source`, `m2_source`, `chi1`, `chi2`).

> Guardrail contractual: `mvp/gwtc_events.py` **no sirve** para \(A_1 + A_2\) porque sólo expone medianas finales (`m_final_msun`, `chi_final`, `snr_network`) y no samples pre-fusión.

## Inputs requeridos
Bajo `runs/<analysis_run_id>/external_inputs/gwtc_posteriors/`:
- `required_events.txt` (o `pilot_events.txt`)
- `<EVENT_ID>.json` para cada evento requerido

Formato canónico de cada posterior JSON:

```json
{
  "event_id": "GW150914",
  "source": {
    "kind": "GWTC_public|manual",
    "citation": "...",
    "url_or_id": "..."
  },
  "samples": [
    {"m1_source": 36.0, "m2_source": 29.0, "chi1": 0.2, "chi2": 0.1}
  ]
}
```

## Paso A — Validar presencia de posteriors (manual)

```bash
python -m mvp.experiment_gwtc_posteriors_fetch \
  --run-id <analysis_run_id> \
  --source manual \
  --format json
```

Comportamiento:
- Si falta cualquier `<EVENT_ID>.json`, aborta deterministicamente con lista `missing_events=[...]`.
- Si todo está completo, escribe:
  - `runs/<run_id>/experiment_gwtc_posteriors_fetch/outputs/validated_posteriors.json`
  - `manifest.json`
  - `stage_summary.json`

## Paso B — Ejecutar cálculo ΔA

```bash
python -m mvp.experiment_area_theorem_deltaA \
  --run-id <analysis_run_id> \
  --batch220-run-id batch_with_t0_220_eps2500_fixlen_20260304T160054Z \
  --batch221-run-id batch_with_t0_221_eps2500_fixlen_20260304T160617Z \
  --mc-draws 4000 \
  --seed 7
```

Outputs esperados:
- `runs/<analysis_run_id>/experiment_area_theorem/outputs/per_event.csv`
- `runs/<analysis_run_id>/experiment_area_theorem/outputs/summary.json`
- `runs/<analysis_run_id>/experiment_area_theorem/manifest.json`
- `runs/<analysis_run_id>/experiment_area_theorem/stage_summary.json`

## Fórmula de área usada
En unidades geométricas:

\[
A(M,\chi) = 8\pi M^2\left(1 + \sqrt{1-\chi^2}\right)
\]

El experimento reporta por evento:
- `P_deltaA_lt_0`
- `deltaA_p10`, `deltaA_p50`, `deltaA_p90`
- tamaños de muestra usados (`n_draws`, `n_af_candidates`, `n_imr_samples`)

## Auditoría (manifest/stage_summary)
Ambos entrypoints escriben `manifest.json` y `stage_summary.json` con hashes SHA256 de artefactos producidos y de inputs consumidos; además imprimen:
- `OUT_ROOT`
- `STAGE_DIR`
- `OUTPUTS_DIR`
- `STAGE_SUMMARY`
- `MANIFEST`
