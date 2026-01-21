# Experimento 03 — Sensibilidad de C3 a la métrica (rmse vs rmse_log) en `ir_mix`

**Proyecto:** BASURIN  
**Bloque:** C (diccionario holográfico inverso)  
**Versión del código:** v1.4.1  
**Run:** `ir_mix`  
**Generador del dataset mixto:** `02_mix_spectra.py` (concatena dos `spectrum.h5`)  

> Documento **mínimo, completo, reproducible y auditable**.  
> Regla de honestidad: **no se inventan números**. Todo valor numérico reportado debe provenir de los JSON de corrida.  
> Estado actual: los JSON de corrida **aún no han sido aportados en esta conversación**, por lo que los resultados se marcan como **N/A** y se deja explícito qué campo faltó.

---

## 1) Objetivo e hipótesis falsable

### Objetivo
Demostrar experimentalmente que el contrato **C3 (compatibilidad espectral con diagnóstico causal)** es **sensible a la elección de norma** y a la ponderación de ratios, de modo que **el mismo dataset mixto** (`ir_mix`) puede:

- **Fallar** con una configuración “naive” (control negativo).
- **Pasar** con una configuración robusta (control positivo).

### Hipótesis falsable (Exp 03)
Para el run `ir_mix`, si se evalúa C3 con:

- **Norma absoluta** (`--c3-metric rmse`) y **pesos uniformes** (`--c3-weights none`), entonces C3 **debe fallar** con `failure_mode = DECODER_MISMATCH` y valores **C3a≈C3b≈0.11** (según lo observado previamente).

Pero si se evalúa C3 con:

- **Norma logarítmica** (`--c3-metric rmse_log`) y **pesos `inv_n4`**, además de **umbral adaptativo**, entonces C3 **debe pasar** con **C3a≈C3b≈0.0043** (según lo observado previamente).

**Falsabilidad:** si cualquiera de estas dos predicciones no se cumple con las **mismas configuraciones**, el experimento “rompe” (ver §7).

---

## 2) Contexto y motivación técnica

- Estamos en **Bloque C v1.4.1** (`04_diccionario.py`).
- El run `ir_mix` es un dataset **mixto** creado por `02_mix_spectra.py`, que **concatena** dos espectros `outputs/spectrum.h5` compatibles (mismo `delta_uv` y mismo `n_modes`) y **duplica** `delta_uv` para etiquetar cada muestra (dos familias para los mismos Δ).

Motivación: en datasets mixtos (multi-familia), la comparación en **norma absoluta** puede penalizar de forma desproporcionada ratios de orden alto (típicamente con mayor escala), y un contrato C3 basado en **log-ratios** y pesos decrecientes puede capturar mejor la “compatibilidad estructural” del ciclo sin colapsar en penalizaciones por escala.

---

## 3) Inputs (rutas exactas bajo `runs/ir_mix/...`)

### Dataset de entrada
- `runs/ir_mix/spectrum/outputs/spectrum.h5`  *(input principal para `04_diccionario.py`)*

### Metadatos del dataset (mezcla)
- `runs/ir_mix/spectrum/stage_summary.json`
- `runs/ir_mix/spectrum/manifest.json`

> Nota: el experimento depende de que `ir_mix` exista y haya sido generado por `02_mix_spectra.py`. Este script documenta inputs y hashes en `runs/ir_mix/spectrum/stage_summary.json`.

---

## 4) Ejecución reproducible A — Control negativo (FAIL esperado)

### Propósito
Probar que la configuración naive (norma absoluta + pesos uniformes + umbral fijo) produce un **FAIL informativo** en C3 sobre el **mismo dataset mixto**.

### Comando
```bash
python 04_diccionario.py --run ir_mix \
  --enable-c3 \
  --k-features 4 \
  --n-bootstrap 0 \
  --c3-metric rmse \
  --c3-weights none \
  --c3-threshold 0.05
```

**Parámetros clave**
- `--k-features 4`: ratios `r_1..r_4`
- `--n-bootstrap 0`: ejecución determinista rápida (sin incertidumbre)
- `--c3-metric rmse`: norma absoluta en ratios
- `--c3-weights none`: pesos uniformes
- `--c3-threshold 0.05`: umbral fijo

### Resultado esperado (según observación previa)
- `C3.status = FAIL`
- `C3.failure_mode = DECODER_MISMATCH`
- `C3a.global ≈ 0.11`
- `C3b.global ≈ 0.11`

### Resultado reportado desde JSON (Corrida A)
Fuente obligatoria:
- `runs/ir_mix/dictionary/validation.json` *(Corrida A)*
- `runs/ir_mix/dictionary/stage_summary.json` *(Corrida A)*

| Campo | Valor |
|---|---:|
| C2.cv_rmse | N/A |
| C3.status | N/A |
| C3.failure_mode | N/A |
| C3a.global | N/A |
| C3b.global | N/A |
| threshold.user | N/A |
| threshold.adaptive | N/A |
| threshold.effective | N/A |
| noise_floor.median_sigma | N/A |
| notes / evidencia | N/A (JSON no aportado) |

---

## 5) Ejecución reproducible B — Control positivo (PASS esperado)

### Propósito
Probar que, con un contrato C3 más robusto (log-métrica + pesos `inv_n4` + umbral adaptativo), el mismo dataset `ir_mix` pasa C3.

### Comando
```bash
python 04_diccionario.py --run ir_mix \
  --enable-c3 \
  --k-features 4 \
  --n-bootstrap 0 \
  --c3-metric rmse_log \
  --c3-weights inv_n4 \
  --c3-adaptive-threshold \
  --c3-threshold 0.02
```

**Parámetros clave**
- `--c3-metric rmse_log`: penaliza discrepancias relativas (escala-invariante multiplicativa)
- `--c3-weights inv_n4`: de-emfatiza ratios altos
- `--c3-adaptive-threshold`: activa umbral efectivo `max(user, factor*noise_floor)`
- `--c3-threshold 0.02`: umbral base (piso)

### Resultado esperado (según observación previa)
- `C3.status = PASS`
- `C3a.global ≈ 0.0043`
- `C3b.global ≈ 0.0043`
- `C3.failure_mode = None`

### Resultado reportado desde JSON (Corrida B)
Fuente obligatoria:
- `runs/ir_mix/dictionary/validation.json` *(Corrida B)*
- `runs/ir_mix/dictionary/stage_summary.json` *(Corrida B)*

| Campo | Valor |
|---|---:|
| C2.cv_rmse | N/A |
| C3.status | N/A |
| C3.failure_mode | N/A |
| C3a.global | N/A |
| C3b.global | N/A |
| threshold.user | N/A |
| threshold.adaptive | N/A |
| threshold.effective | N/A |
| noise_floor.median_sigma | N/A |
| notes / evidencia | N/A (JSON no aportado) |

---

## 6) Resultados: tabla comparativa (A vs B)

> Requisito: reportar **C2 (cv_rmse), C3a, C3b, threshold efectivo, noise_floor, PASS/FAIL**.  
> Estado actual: **N/A** en todos los campos por falta de JSONs en esta conversación.

| Corrida | Configuración | C2.cv_rmse | C3a | C3b | threshold (effective) | noise_floor (median_sigma) | Estado |
|---|---|---:|---:|---:|---:|---:|---|
| A | rmse + none + thr=0.05 | N/A | N/A | N/A | N/A | N/A | N/A |
| B | rmse_log + inv_n4 + adaptive + thr=0.02 | N/A | N/A | N/A | N/A | N/A | N/A |

### Campos esperados en `validation.json`
Bajo `validation.json` (Bloque C v1.4.1) estos campos suelen estar en:
- `C2_consistency.cv_rmse`
- `C3_spectral.c3a_decoder.global`
- `C3_spectral.c3b_cycle.global`
- `C3_spectral.threshold.effective`
- `C3_spectral.noise_floor.median_sigma`
- `overall.C3_status` y/o `C3_spectral.status`

Si alguno no aparece, el documento lo marcará como **N/A** y quedará justificado.

---

## 7) Interpretación (por qué el experimento es informativo)

### 7.1 Norma absoluta vs norma logarítmica
- `rmse` mide diferencias **absolutas** en ratios `r_n`. Si algunos ratios tienen mayor escala o varianza, dominan el error global.
- `rmse_log` mide diferencias en `log(r)`, equivalente a penalizar discrepancias **multiplicativas/relativas**, lo que suele ser más estable cuando hay rangos dinámicos diferentes entre ratios.

Implicación: si el fallo en C3 con `rmse` desaparece con `rmse_log` sin cambiar el dataset, entonces el diagnóstico C3 era sensible a escala y no necesariamente a una incompatibilidad estructural.

### 7.2 Qué implica `inv_n4`
`inv_n4` asigna pesos `w_n ∝ 1/n^4` a ratios de orden alto. Efecto operacional:
- Reduce la contribución de ratios altos (`r_4`, etc.) al WRMSE global.
- Mitiga la dominancia numérica de componentes potencialmente más ruidosas o con mayor rango dinámico.
- En un dataset mixto, esto ayuda a que el diagnóstico enfatice compatibilidad en ratios bajos/medios, típicamente más estables.

### 7.3 Validez sin “inyectar teoría”
Este experimento no cambia:
- ni el dataset (`runs/ir_mix/spectrum/outputs/spectrum.h5`),
- ni el modelo inverso seleccionado por BIC,
- ni el mecanismo de entrenamiento.

Solo cambia el **contrato de evaluación post-hoc** (métrica, pesos, umbral).  
Esto es consistente con el principio BASURIN: **no inyectar teoría conocida durante el aprendizaje**, y usar contratos explícitos para diagnosticar falsablemente.

---

## 8) Artefactos y trazabilidad

### Outputs esperados del Bloque C (por corrida)
Bajo `runs/ir_mix/dictionary/`:

- `dictionary.h5`
- `atlas.json`
- `ising_comparison.json`
- `validation.json`
- `stage_summary.json`
- `manifest.json`

### Hashes (sha256)
**Fuente:** `runs/ir_mix/dictionary/stage_summary.json` → `hashes`.

| Artefacto | sha256 |
|---|---|
| dictionary.h5 | N/A |
| atlas.json | N/A |
| ising_comparison.json | N/A |
| validation.json | N/A |
| (otros) | N/A |

**Motivo de N/A:** los JSON no han sido proporcionados en esta conversación; este bloque se completará con los valores exactos al pegarlos.

---

## 9) Falsabilidad operacional (criterio de “rompe el experimento”)

El experimento se considera **roto** si, ejecutando exactamente los comandos de §4 y §5 sobre el mismo `runs/ir_mix/spectrum/outputs/spectrum.h5`:

1. **Control positivo** deja de pasar:  
   - `C3.status != PASS` **o** `C3a.global > threshold.effective` **o** `C3b.global > threshold.effective`.

2. **Control negativo** deja de fallar (falso negativo de sensibilidad):  
   - `C3.status == PASS` **o** `failure_mode` no corresponde al observado (p.ej., no aparece `DECODER_MISMATCH`).

3. Los resultados se vuelven ambiguos por cambios silenciosos en inputs:
   - hash de `runs/ir_mix/spectrum/outputs/spectrum.h5` cambia sin justificar (por ejemplo, regeneración del dataset).

En cualquiera de estos casos, el Exp 03 exige:
- registrar el cambio (hashes + stage_summary),
- y actualizar el documento con la causa (cambio de versión, parámetros, o dataset).

---

## 10) Pendiente para completar (inputs prometidos)

Para completar este documento con números exactos, se requieren los JSON (dos corridas):

- Corrida A (control negativo):
  - `runs/ir_mix/dictionary/stage_summary.json`
  - `runs/ir_mix/dictionary/validation.json`

- Corrida B (control positivo):
  - `runs/ir_mix/dictionary/stage_summary.json`
  - `runs/ir_mix/dictionary/validation.json`

Una vez aportados, este .md se actualizará sustituyendo los **N/A** por valores exactos y añadiendo los **sha256** de `stage_summary.json`.
