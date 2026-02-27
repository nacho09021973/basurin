# BRUNETE Inventory (contract-first)

> Inventario estático del repo (sin inferir outputs inexistentes y sin ejecutar pipeline).

## 1) Stages y contratos reales (SSOT)

## SSOT contractual
- **Fuente de verdad de contratos:** `mvp/contracts.py`.
- Registro canónico: `CONTRACTS: dict[str, StageContract]`.

## Stages declarados en `CONTRACTS`
- `s0_oracle_mvp`
- `s1_fetch_strain`
- `s2_ringdown_window`
- `s3_ringdown_estimates`
- `s4_geometry_filter`
- `s4_spectral_geometry_filter`
- `s5_aggregate`
- `s6_information_geometry`
- `s6b_information_geometry_3d`
- `s6b_information_geometry_ranked`
- `s6c_population_geometry`
- `s4b_spectral_curvature`
- `s3b_multimode_estimates`
- `s4c_kerr_consistency`
- `s4d_kerr_from_multimode`
- `s3_spectral_estimates`
- `experiment_geometry_evidence_vs_gr`

## Ubicación de piezas contractuales pedidas
- `require_run_valid(...)`: `basurin_io.py`.
  - Verifica existencia de `runs/<run_id>/RUN_VALID/verdict.json` (o `<BASURIN_RUNS_ROOT>/<run_id>/RUN_VALID/verdict.json` si env var está activa vía `resolve_out_root`).
  - Exige `"verdict": "PASS"`.
- `RUN_VALID` lifecycle en orquestación:
  - creación inicial: `_create_run_valid(...)` en `mvp/pipeline.py`
  - actualización de veredicto: `_set_run_valid_verdict(...)` en `mvp/pipeline.py`
- `finalize(...)`: `mvp/contracts.py`.
  - escribe `stage_summary.json` y `manifest.json` del stage.
- Hashing helpers:
  - `sha256_file(...)`: `basurin_io.py`
  - hashing en manifest: `write_manifest(...)` en `basurin_io.py`
  - hashing de inputs/outputs durante contrato: `check_inputs(...)` y `finalize(...)` en `mvp/contracts.py`.

---

## 2) Artefacto real de ringdown estimado (s3 o s3b)

## Ruta exacta bajo `runs/<run_id>/...`

### s3 (canónico single-mode)
- **Output contractual:**
  - `runs/<run_id>/s3_ringdown_estimates/outputs/estimates.json`
- Declarado en `CONTRACTS["s3_ringdown_estimates"].produced_outputs = ["outputs/estimates.json"]`.

### s3b (canónico multimodo)
- **Outputs contractuales:**
  - `runs/<run_id>/s3b_multimode_estimates/outputs/multimode_estimates.json`
  - `runs/<run_id>/s3b_multimode_estimates/outputs/model_comparison.json`
- Declarado en `CONTRACTS["s3b_multimode_estimates"].produced_outputs`.

## Schema real observado en código (sin inventar)

### s3 `estimates.json` (`schema_version: "mvp_estimates_v2"`)
Top-level keys construidas en `mvp/s3_ringdown_estimates.py`:
- `schema_version`
- `event_id`
- `method`
- `band_hz`
- `combined`
- `combined_uncertainty`
- `per_detector`
- `n_detectors_valid`
- opcionales: `t0_selected`, `t0_scan`, `bootstrap`

Campos relevantes solicitados:
- **event_id:** presente en top-level (`event_id`).
- **f, tau, Q:** presentes en `combined` (`f_hz`, `tau_s`, `Q`) y por detector dentro de `per_detector[DET]` cuando el detector es válido.
- **detector:** en s3 está representado por la **clave** del mapa `per_detector` (`H1`, `L1`, `V1`), no como campo explícito por fila.
- **snr/rho:** presente como `snr_peak` (no aparece campo `rho` en este artefacto).
- **mode:** no está en `s3`.

### s3b `multimode_estimates.json` (`schema_version: "multimode_estimates_v1"`)
Top-level keys construidas en `mvp/s3b_multimode_estimates.py`:
- `schema_version`
- `run_id`
- `source` (incluye `stage` y `window`)
- `modes_target`
- `results`
- `modes`

Campos relevantes solicitados:
- **mode:** presente en cada elemento de `modes` como lista (ej. `[2,2,0]`, `[2,2,1]`) y `label` (`"220"`, `"221"`).
- **f, Q/tau:** en s3b se guardan en log-espacio como `ln_f` y `ln_Q` por modo; no hay campo `tau_s` explícito en `multimode_estimates.json`.
- **event_id:** no aparece como key top-level garantizada en payload s3b (puede venir indirectamente en `source.window` si existe en `window_meta`).
- **detector:** no hay campo explícito de detector por modo.
- **snr/rho:** no hay campos `snr`/`rho` en `multimode_estimates.json`.

---

## 3) PSD canónica upstream

## Resultado
- **No existe actualmente un stage canónico en `CONTRACTS` que produzca PSD** (`produced_outputs` no contiene artefactos `psd*`).

## Lo que sí existe
- Helper opcional `mvp/extract_psd.py` (no stage formal):
  - Input: `runs/<run_id>/s1_fetch_strain/outputs/strain.npz`
  - Output: `runs/<run_id>/psd/measured_psd.json`
  - Manifest helper: `runs/<run_id>/psd/manifest.json`
  - Schema de `measured_psd.json` (`schema_version: "mvp_measured_psd_v1"`) incluye, entre otros:
    - `run_id`, `detector`, `method`, `fs`, `nperseg`, `nperseg_s`, `overlap`,
    - `n_samples`, `n_freq_bins`, `freq_resolution_hz`,
    - `frequencies_hz`, `psd_values`.

## Convención `external_inputs`
- Sí está documentada explícitamente la convención:
  - `runs/<run_id>/external_inputs/...`
- Referencia operativa: `docs/readme_rutas.md`.
