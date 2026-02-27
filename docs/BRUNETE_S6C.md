# BRUNETE S6C — Contrato operativo

## Propósito
`mvp/s6c_brunete_psd_curvature.py` calcula métricas BRUNETE por detector a partir de estimaciones ringdown y curvatura local de la PSD alrededor de `f_hz`:
- Derivadas log-PSD: `s1`, `kappa`.
- Escala geométrica: `sigma`.
- Indicador de contaminación instrumental: `chi_psd`.
- Curvaturas/escalares derivados: `K`, `R` (si hay `rho0>0`).
- Clasificación de regímenes: `regime_sigma`, `regime_chi_psd`.

Base matemática y significado físico de `sigma`/`chi`: ver `docs/metodo_brunete.md`.

## Inputs (rutas canónicas y schema)

### 1) Ringdown estimates (upstream canónico, requerido)
Ruta:
- `runs/<run_id>/s3_ringdown_estimates/outputs/estimates.json`

Campos mínimos usados por s6c:
- `event_id` (str)
- `per_detector` (objeto por detector)
  - `f_hz` (float)
  - `Q` (float)
  - `tau_s` (float)
  - `rho0` o `snr_peak` (float, opcional para `K/R`)

### 2) PSD input (preferencia upstream + fallback external)
Preferencia 1 (upstream real):
- `runs/<run_id>/psd/measured_psd.json`
- Se genera con: `python mvp/extract_psd.py --run <RUN_ID> [--detector H1]`

Fallback explícito:
- `runs/<run_id>/external_inputs/psd_model.json`

Schema mínimo aceptado de PSD:
- Opción A (global):
  - `frequencies_hz: [float, ...]`
  - `psd_values: [float, ...]`
- Opción B (por detector):
  - `models.<DETECTOR>.frequencies_hz: [float, ...]`
  - `models.<DETECTOR>.psd_values: [float, ...]`

> Estado contractual actual: la SSOT declara `psd/measured_psd.json` como input preferido; el stage mantiene fallback runtime a `external_inputs/psd_model.json`.

## Parámetros configurables (defaults)
- `--c-window` (default `30.0`): se usa como `half_window_hz` en el cálculo local de derivadas PSD.
- `--min-points` (default `7`): mínimo de bins en la ventana para ajuste local.
- `--sigma-switch` (default `0.1`): umbral numérico para régimen de expansión perturbativa/cerrada de `J0/J1` (control de `sigma`).
- `--chi-psd-threshold` (default `1.0`): umbral físico para clasificar `chi_psd` como `elevated`.
- `--mode` (default `220`).
- `--psd-path` (opcional): ruta explícita a PSD JSON.

## Outputs
Directorio canónico del stage:
- `runs/<run_id>/s6c_brunete_psd_curvature/`

Artefactos:
- `outputs/brunete_metrics.json`
- `outputs/psd_derivatives.json`
- `stage_summary.json`
- `manifest.json` (ownership contractual vía `contracts.finalize()`)

### Schema: `outputs/brunete_metrics.json`
```json
{
  "schema_version": "brunete_metrics_v1",
  "run_id": "<run_id>",
  "psd_input": "<path-relativa-al-run>",
  "metrics": [
    {
      "event_id": "string",
      "mode": "string",
      "detector": "string",
      "f_hz": 0.0,
      "Q": 0.0,
      "tau_s": 0.0,
      "rho0": 0.0,
      "s1": 0.0,
      "kappa": 0.0,
      "sigma": 0.0,
      "chi_psd": 0.0,
      "J0": 0.0,
      "J1": 0.0,
      "K": 0.0,
      "R": 0.0,
      "regime_sigma": "perturbative|closed_form|not_applicable",
      "regime_chi_psd": "low|elevated",
      "warnings": ["..."]
    }
  ]
}
```

### Schema: `outputs/psd_derivatives.json`
```json
{
  "schema_version": "psd_derivatives_v1",
  "run_id": "<run_id>",
  "derivatives": [
    {
      "event_id": "string",
      "mode": "string",
      "detector": "string",
      "method": "polyfit_log_psd_deg2",
      "half_window_hz": 30.0,
      "n_points": 7,
      "polyfit_coefficients": [0.0, 0.0, 0.0],
      "s1": 0.0,
      "kappa": 0.0
    }
  ]
}
```

### `stage_summary.json`
Incluye: `parameters`, `inputs` (con SHA256), `outputs` (con SHA256), `verdict`, contadores de regímenes y resultados (`n_rows`, `n_warnings`).

## Ejemplos de ejecución
CLI stage:
```bash
python mvp/s6c_brunete_psd_curvature.py \
  --run <RUN_ID> \
  --c-window 30.0 \
  --min-points 7 \
  --sigma-switch 0.1 \
  --chi-psd-threshold 1.0
```

Con PSD explícita:
```bash
python mvp/s6c_brunete_psd_curvature.py --run <RUN_ID> \
  --psd-path runs/<RUN_ID>/external_inputs/psd_model.json
```

Pipeline runner (si se ejecuta por stages independientes):
```bash
python mvp/pipeline.py --event-id <EVENT_ID> --atlas-path <ATLAS_JSON>
python mvp/s6c_brunete_psd_curvature.py --run <RUN_ID>
```

## Troubleshooting
- `RUN_VALID` faltante o `RUN_VALID != PASS`:
  - el stage aborta por gating contractual (`require_run_valid`).
- PSD faltante / schema inválido:
  - verificar primero `runs/<run_id>/psd/measured_psd.json`;
  - si no existe, proveer `runs/<run_id>/external_inputs/psd_model.json` con schema mínimo.
- Ventana insuficiente (`min_points`):
  - aumentar `--c-window` o disminuir `--min-points` según resolución de PSD.
- `sigma < 0` (no convergente):
  - esperado `regime_sigma="not_applicable"` y warning asociado (`J0_J1_status=...`).

## Referencias cruzadas
- `docs/metodo_brunete.md`
- `docs/BRUNETE_TEXT_FIXES.md`
