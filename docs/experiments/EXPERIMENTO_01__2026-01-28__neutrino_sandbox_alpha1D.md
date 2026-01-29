# EXPERIMENTO 01 — Neutrino sandbox (symmetron) → colapso a 1D en α
**Fecha:** 28-01-2026  
**Proyecto:** BASURIN  
**Estado:** CERRADO (resultado científico negativo/degenerado, pero válido)

---

## 1. Pregunta científica (contract-first)

**Pregunta:** ¿Puede inferirse estructura “bulk” (o geometría efectiva) a partir de un conjunto limitado de observables espectrales, **sin inyectar teoría conocida como señal de entrenamiento**?

**Hipótesis operacional:** si el inverso *datos → geometría* es identificable en este sandbox, entonces variaciones independientes en parámetros de la familia (α, δ, …) deben reflejarse como variaciones independientes (≥2D) en el observable disponible (espectro de masas al cuadrado **M²**).

---

## 2. Setup experimental

### 2.1 Familia / generador
- Sandbox “neutrino” (familia symmetron).
- Parámetros relevantes: **α** y **δ** (δ pretendía aportar una segunda dirección física).

> Nota epistemológica: el sandbox es sintético, pero la evaluación es estrictamente post-hoc. No se usa teoría conocida para “enseñar” el inverso.

### 2.2 Observable disponible
- `spectrum.h5` con datasets mínimos:
  - `M2` (matriz)  
  - `delta_uv`  
  - (opcional) `m2L2`, `z_grid`

Este es el “dato” que el inverso intenta explicar.

---

## 3. Evidencias (resultado cuantitativo)

### 3.1 Estructura del H5 inspeccionada
- keys: `['M2', 'delta_uv', 'm2L2', 'z_grid']`
- `M2.shape = (20, 5)`  *(20 muestras, 5 modos)*

### 3.2 Diagnóstico de identificabilidad (correlaciones)
En la inspección del primer modo y modos siguientes:

- **mode0**: `corr(M2, α) ≈ +1.000`  
- **mode2/3/4**: correlaciones ~ ±1 con α (señal de dependencia 1D)  
- **δ**: `corr(δ) ≈ 0.000` (o estadísticamente no informativo con el observable disponible)

También se observó:
- columnas/modos con **varianza efectiva ~ 0** → `corr = NaN` por degeneración numérica (constantes).

---

## 4. Conclusión científica (veredicto)

### 4.1 Veredicto
**El sandbox colapsa efectivamente a una variedad 1D gobernada por α** cuando el único observable es el espectro escalar `M²`.

- **δ no es identificable** con el espectro `M²` disponible.
- No hay estructura 2D recuperable en el inverso *datos → geometría* bajo este régimen de observables.

### 4.2 Interpretación (por qué esto es “buena ciencia”)
Esto **no es un fallo del pipeline**: es una **señal de subdeterminación real** del problema inverso incluso en un entorno sintético limpio y determinista.

En términos BASURIN:
- el experimento demuestra un **modo de fallo estructural**: *observable insuficiente ⇒ degeneración del inverso*.
- se obtiene un resultado falsable y trazable: “con este dato, δ no se puede recuperar”.

---

## 5. Artefactos y trazabilidad esperada (BASURIN IO)

> Este experimento debe vivir en `runs/<run_id>/...` para IO determinista.  
> La documentación (este fichero) vive en `docs/experiments/`.

En el run:
- `runs/<run_id>/spectrum/outputs/spectrum.h5`
- `runs/<run_id>/spectrum/manifest.json`
- `runs/<run_id>/spectrum/stage_summary.json`
- `runs/<run_id>/RUN_VALID/outputs/run_valid.json` (gobierna existencia)

---

## 6. Tests mínimos que previenen regresión (recomendados)

1) **Test de columnas constantes / NaN-corr**
- Detectar columnas con `std == 0` en `M2` y etiquetar explícitamente (no permitir NaN silencioso).

2) **Test de dimensión efectiva**
- SVD/PCA sobre `M2` (centrado):
  - `explained_var_1 > 0.98` ⇒ colapso 1D (criterio reproducible).
- Guardar diagnóstico en un JSON de auditoría.

3) **Test de no-identificabilidad de δ**
- Verificar `|corr(M2, δ)| < ε` con ε fijo (p.ej. 0.1) **en el dataset canónico del experimento**.

---

## 7. Próximo paso natural (EXPERIMENTO 02)

**Pregunta:** ¿Cuál es el **mínimo nuevo observable** que rompe el colapso 1D → 2D?

Candidatos “mínimos” y compatibles con BASURIN:
- **Espectro dual Dirichlet/Neumann** (añade canal observable sin cambiar el generador).
- **Múltiples sectores/modos** (p.ej. añadir un canal adicional o un observable derivado reproducible del mismo run).
- **Ruido estructurado controlado** como stress-test de identificabilidad.

---

## 8. Notas de gobernanza

- Si `RUN_VALID != PASS`, este experimento **no existe** y no se permiten conclusiones downstream.
- No se inventan datasets intermedios: cualquier nuevo observable debe formalizarse como artefacto canónico de un stage.

---

**Fin del EXPERIMENTO 01.**
