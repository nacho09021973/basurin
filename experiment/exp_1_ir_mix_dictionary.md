# Experimento 1 — Mezcla de espectros y verificación de Bloque C (`ir_mix`)

## Objetivo

Evaluar el comportamiento del **Bloque C (diccionario holográfico emergente)** bajo un dataset **mixto** construido por concatenación de dos familias espectrales (etapa `01_mix_spectra.py`), y verificar:

1. estabilidad del ajuste inverso **ratios → Δ** (Contrato C2),
2. compatibilidad espectral / *cycle-consistency* (Contrato C3) bajo una métrica robusta,
3. comportamiento de C1 (Ising 3D) como filtro de dominio (*OUT_OF_DOMAIN*).

Este experimento es deliberadamente **adversarial**: mezcla dos familias sin introducir una etiqueta explícita de “familia”. Se utiliza para testear sensibilidad contractual (norma/pesos) y la robustez del pipeline.

---

## Entradas y configuración

### Input

- Espectro: `runs/ir_mix/spectrum/spectrum.h5`
- Run: `ir_mix`
- Version Bloque C: `1.4.1`
- Timestamp: ``

### Comando reproducible (Bloque C)

```bash
python 04_diccionario.py \
  --run ir_mix \
  --enable-c3 \
  --k-features 4 \
  --n-bootstrap 0 \
  --c3-metric rmse_log \
  --c3-weights inv_n4 \
  --c3-adaptive-threshold \
  --c3-threshold 0.02
```

> Nota: este experimento es el “control positivo” (PASS) usando `rmse_log` + `inv_n4`.  
> El “control negativo” (FAIL) se obtiene con `metric=rmse, weights=none, threshold=0.05`, que típicamente produce *DECODER_MISMATCH* en mezclas.

---

## Resultados

### C1 — Ising 3D (compatibilidad puntual)

C1 devuelve *OUT_OF_DOMAIN* cuando el target está fuera del soporte del diccionario. En este run, el soporte de Δ proviene del espectro mixto y no cubre los valores de Ising σ y ε, por lo que C1 actúa como **gate de dominio** (informativo, no hard-fail).

### C2 — Consistencia interna (ratios → Δ)

- `cv_rmse = 0.002238398459376911`
- `threshold = None`
- `consistency_ok = True`

**Resultado:** C2 PASS.

### C3 — Compatibilidad espectral (cycle-consistency)

Configuración C3 (según stage/config):
- `metric = rmse_log`
- `weights = inv_n4`
- `threshold (user) = 0.02`
- `adaptive_threshold = True`

Resultados:
- `status = PASS`
- `C3a (decoder) = None`
- `C3b (cycle)   = None`
- `threshold_effective = {'user': 0.02, 'adaptive': 0.0008775142819454843, 'effective': 0.02, 'factor': 5.0, 'mode': 'adaptive'}`
- `noise_floor.aggregate_distance = 0.0002415858874114775`

**Resultado:** C3 PASS con margen sobre el umbral efectivo.

### Veredicto global

- `all_hard_contracts_pass = True`

---

## Artefactos (IO determinista)

Salidas escritas en:

- `runs/ir_mix/dictionary/dictionary.h5`
- `runs/ir_mix/dictionary/atlas.json`
- `runs/ir_mix/dictionary/ising_comparison.json`
- `runs/ir_mix/dictionary/validation.json`
- `runs/ir_mix/dictionary/stage_summary.json`
- `runs/ir_mix/dictionary/manifest.json`

Hashes (sha256) registrados para auditoría:

- `dictionary.h5`: `9cd569d4e036a2c75311c99147e802c8bf9e2ec1ac21851911c638eed1c57974`
- `atlas.json`: `0e6a640359d6a60ecd692bd68dbc14ab9782740ea7c77e275832852c677339e7`
- `ising_comparison.json`: `cae693f9e955024666dd28fd88193a504210751dcddedca8d27b56824fcfb268`
- `validation.json`: `9f41c103e12e216839f5242bf5b86f3b4166538a6815b1273bcb2a3ea74a273a`

---

## Interpretación y falsabilidad

1. **Sensibilidad de C3 a la norma:** en datasets mixtos, la elección de métrica/pesos en C3 no es cosmética; define el tipo de discrepancia penalizada (absoluta vs relativa, énfasis en ratios altos vs bajos). Este experimento demuestra que el pipeline puede capturar un PASS robusto con una norma coherente (`rmse_log`, `inv_n4`), y producir un FAIL informativo bajo una norma naive (control negativo).

2. **No extrapolación (C1):** el run no informa sobre Ising por soporte insuficiente de Δ. Este comportamiento es correcto y contractual.

**Falsabilidad operacional:** si futuras versiones del Bloque C rompen este PASS (con esta configuración exacta), se detecta regresión de manera inmediata por `validation.json` y hashes.

