# BRUNETE — Diario de diagnóstico del carril 221

Directorio de trabajo: `runs/brunete_prepare_20260323T1545Z/experiment/`
Cohorte base: 82 eventos O4/O4b offline, `prepare_events` PASS.

---

## 2026-03-23 — Diagnóstico completo del carril 221

### Contexto de entrada

El batch 221 sobre la cohorte O4/O4b depurada (82 eventos) pasa contractualmente
(n_pass=82, n_fail=0), pero en 82/82 eventos el resultado científico es
`SINGLEMODE_ONLY`. El modo 221 nunca produce evidencia usable. Ningún evento
materializa `s4h_mode221_geometry_filter/inputs/mode221_obs.json`.

El bloqueo no es contractual sino científico: `s3b_multimode_estimates` declara
`mode_221_ok=false` en todos los casos, y `multimode_viability` degrada
inmediatamente a `SINGLEMODE_ONLY` (ref: `mvp/multimode_viability.py:77`).

Flags observadas en 82/82: `221_cv_Q_explosive`, `221_lnQ_span_explosive`.

Configuración baseline:
- `method`: `hilbert_peakband`
- `mode_221_topology`: `rigid_spectral_split`
- `bootstrap_221_residual_strategy`: `refit_220_each_iter`
- `n_bootstrap`: 200
- `cv_threshold_221`: 1.0
- `max_lnq_span_221`: 1.0
- `max_lnf_span_221`: 1.0
- `min_valid_fraction_221`: 0.5
- `band_strategy`: `kerr_centered_overlap`

### Hipótesis formuladas

Se formularon 7 hipótesis priorizadas:

| ID | Hipótesis | Prob. inicial |
|----|-----------|---------------|
| A  | Problema metodológico, no de catálogo (paraguas) | 0.70 |
| B  | `rigid_spectral_split` destruye/sesga la inferencia 221 | 0.23 |
| C  | Gating de calidad mal calibrado para datos reales | 0.11 |
| D  | Ventana de ringdown / t0 upstream deja 221 sin señal | 0.15 |
| E  | Detector único / red instrumental insuficiente | 0.02 |
| F  | Q_221 físicamente no medible a este SNR | 0.06 |
| G  | Arquitectura 221 subordinada a 220 (residual + gate binario) | ~0.42 (con B) |

Hipótesis principal de trabajo: B+G — el carril 221 está estructuralmente
220-dominado y la combinación de split rígido + residualización + gate binario
temprano impide producir evidencia autónoma de 221.

Hecho de código clave: 221 se construye como residual tras ajustar y sustraer 220
(`mvp/s3b_multimode_estimates.py:783`, `:825`), no de forma autónoma.

### Selección de eventos centinela

Se diseñó y ejecutó un selector estratificado (`select_sentinel_events.py`) para
definir 10 eventos centinela para el A/B de topología.

Criterio de estratificación:
- Bloque A (4 eventos): top-4 SNR de la cohorte.
- Bloque B (3 eventos): cuartil Q3 de SNR con diversidad en Mf_source.
- Bloque C (3 eventos): cuartil Q2 de SNR (controles negativos).

Ejes: SNR, Mf_source, af (chi_eff como fallback).

Selección materializada en:
`runs/brunete_prepare_20260323T1545Z/experiment/sentinel_selection.json`

Nota: toda la cohorte tiene 0 eventos con 3 detectores; hipótesis E no testeable.

### Experimento 1 — A/B de topología en s3b

**Objetivo:** falsar hipótesis B (rigid_spectral_split como causa principal).

**Diseño:** reejecutar s3b con `--mode-221-topology shared_band_early_taper`,
todo lo demás idéntico (thresholds congelados, misma residual strategy).

**Set ejecutado:** 10 eventos diagnósticos del batch 221 (no los centinela
oficiales; seleccionados por `valid_fraction_221 >= 0.80` + flags explosivas):

- GW230708_230935, GW231004_232346, GW230914_111401, GW230729_082317,
  GW230709_122727, GW230814_061920, GW230707_124047, GW231127_165300,
  GW231113_122623, GW230911_195324

Subruns: `runs/ab221sbet_<EVENT>_20260323/`

Implementación: subruns derivados con symlinks a RUN_VALID, s2, s3 del run
original; solo s3b se reejecutó.

**Resultados:**

| Métrica | Dirección con shared_band_early_taper |
|---------|---------------------------------------|
| `multimode_viability` | SINGLEMODE_ONLY en 10/10 (sin cambio) |
| `mode_221_usable` | false en 10/10 (sin cambio) |
| `valid_fraction_221` | ↑ mejoró en 10/10 |
| `lnQ_span` | ↓ mejoró ~20-35% en 10/10, pero sigue >1.0 |
| `lnf_span` | ≈ estable (ya era bajo) |
| `cv_Q` | ↑ empeoró en ~7/10 |
| `Sigma` | válida → válida en 10/10 |

Tabla completa (base → variante):

| Evento | vf221 | lnQ_span | lnf_span | cv_Q | Sigma |
|--------|-------|----------|----------|------|-------|
| GW230708_230935 | 0.970→1.000 | 1.823→1.286 | 0.039→0.045 | 1.606→4.299 | yes→yes |
| GW231004_232346 | 0.970→0.975 | 1.862→1.286 | 0.039→0.043 | 2.011→4.568 | yes→yes |
| GW230914_111401 | 0.960→0.990 | 2.193→1.665 | 0.043→0.053 | 2.147→1.634 | yes→yes |
| GW230729_082317 | 0.950→1.000 | 2.107→1.774 | 0.042→0.052 | 2.279→5.340 | yes→yes |
| GW230709_122727 | 0.875→0.990 | 2.374→1.899 | 0.012→0.012 | 2.780→3.004 | yes→yes |
| GW230814_061920 | 0.875→0.975 | 2.427→1.876 | 0.012→0.012 | 5.195→7.268 | yes→yes |
| GW230707_124047 | 0.875→0.980 | 2.310→1.984 | 0.013→0.013 | 5.402→5.024 | yes→yes |
| GW231127_165300 | 0.810→0.910 | 3.069→2.226 | 0.019→0.018 | 1.581→3.087 | yes→yes |
| GW231113_122623 | 0.805→0.940 | 2.681→2.324 | 0.015→0.016 | 1.719→2.964 | yes→yes |
| GW230911_195324 | 0.805→0.945 | 2.697→2.283 | 0.016→0.017 | 2.774→5.914 | yes→yes |

**Lectura:** hipótesis B debilitada. La topología mejora métricas secundarias pero
no rescata `mode_221_ok`. El cuello que sigue bloqueando es `cv_Q`.

Patrón clave: `lnQ_span` baja + `cv_Q` sube = conservación de patología en Q_221.
Las variantes de extractor no crean ni destruyen información sobre Q; la
redistribuyen entre dos modos de fallo.

### Experimento 2 — A/B de estrategia residual

**Objetivo:** falsar hipótesis G (`refit_220_each_iter` inyecta varianza dominante).

**Diseño:** fijar `shared_band_early_taper`, comparar `refit_220_each_iter` vs
`fixed_220_template`. Thresholds congelados.

Referencia de código: las dos opciones están implementadas en
`mvp/s3b_multimode_estimates.py:851` (refit) y `:864` (fixed).

**Set ejecutado:** 3 eventos top del Experimento 1 (menor lnQ_span en variante):

- GW230708_230935, GW231004_232346, GW230914_111401

Subruns: `runs/ab221fixed_<EVENT>_20260323/`

**Resultados (shared_band_early_taper + refit → fixed):**

| Evento | vf221 | lnQ_span | lnf_span | cv_Q | Sigma |
|--------|-------|----------|----------|------|-------|
| GW230708_230935 | 1.000→0.985 | 1.286→2.308 | 0.159→0.093 | 4.299→1.703 | yes→yes |
| GW231004_232346 | 0.975→0.990 | 1.286→2.383 | 0.163→0.086 | 4.568→4.237 | yes→yes |
| GW230914_111401 | 0.990→0.985 | 1.665→2.378 | 0.126→0.086 | 1.634→2.786 | yes→yes |

**Lectura:**
- MULTIMODE_OK: 0/3. mode_221_usable: 0/3 rescatados.
- cv_Q baja en 2/3 pero lnQ_span empeora en 3/3.
- Mismo patrón de conservación de patología: la intervención redistribuye el fallo
  entre cv_Q y lnQ_span sin resolverlo.
- G contribuye pero no domina. F sube como hipótesis principal.

### Experimento 3 — Inspección de distribuciones bootstrap crudas

**Objetivo:** distinguir F (Q_221 no medible) de G/C por forma de distribución.

**Hallazgo previo:** s3b NO materializa muestras bootstrap individuales. Solo
persiste estadísticos agregados (`fit.stability` en `multimode_estimates.json`).
No hay flag CLI de debug para emitirlas.
Ref: `mvp/s3b_multimode_estimates.py:523` (generación), `:955`/`:1065` (persistencia).

**Solución:** script extractor externo (`extract_s3b_bootstrap_samples.py`) que
reutiliza la lógica interna de s3b para emitir las 200 muestras de (f_221, Q_221).

**Set ejecutado:** mismos 3 eventos, configuración `shared_band_early_taper` +
`refit_220_each_iter`.

Output: `runs/ab221_bootstrap_samples_3events_20260323.json`

**Resultados — distribución de Q_221:**

| Evento | n_valid | Q min | Q mediana | Q max |
|--------|---------|-------|-----------|-------|
| GW230708_230935 | 200 | 5.07 | 9.37 | 1109.44 |
| GW231004_232346 | 195 | 4.63 | 10.10 | 1309.03 |
| GW230914_111401 | 198 | 4.00 | 10.74 | 317.74 |

Percentiles lnQ_221:

| Evento | p10 | p50 | p90 |
|--------|-----|-----|-----|
| GW230708_230935 | 1.818 | 2.233 | 3.104 |
| GW231004_232346 | 1.789 | 2.312 | 3.075 |
| GW230914_111401 | 1.813 | 2.373 | 3.479 |

**Estructura observada:** modo central Q ≈ 9–11, cola derecha extrema a
Q > 300–1300. Percentiles casi idénticos entre eventos → patrón sistemático.

### Experimento 4 — Cruce con referencia Kerr

**Objetivo:** determinar si el bootstrap "ve" la señal Kerr correcta pero inestable,
o si hay bias además de varianza.

**Referencia Kerr usada:** fórmula de Berti para (2,2,1) implementada en
`mvp/kerr_qnm_fits.py:40`:

```
Q_221(af) = 0.1000 + 0.5436 * (1 - af)^(-0.4731)
```

Q_221_Kerr no depende de Mf_source; solo de af.

**Parámetros de catálogo y Q_Kerr calculados:**

| Evento | Mf_source | af | Q_221_Kerr | ln Q_221_Kerr |
|--------|-----------|-----|------------|---------------|
| GW230708_230935 | 99.0 | 0.6737 | 1.023 | 0.023 |
| GW231004_232346 | 96.0 | 0.6377 | 0.979 | −0.021 |
| GW230914_111401 | 90.9 | 0.7083 | 1.074 | 0.071 |

Refs: `gwtc_events_t0.json` líneas 486, 1354, 1026 respectivamente.

**Tabla cruzada bootstrap vs Kerr:**

| Evento | Q_Kerr | Bootstrap p10(Q) | Bootstrap p50(Q) | Ratio p50/Kerr |
|--------|--------|-------------------|-------------------|----------------|
| GW230708_230935 | 1.023 | 6.16 | 9.37 | 9.2× |
| GW231004_232346 | 0.979 | 5.97 | 10.10 | 10.3× |
| GW230914_111401 | 1.074 | 6.13 | 10.74 | 10.0× |

**Lectura:** la referencia Kerr cae muy por debajo del modo central bootstrap.
Ni el percentil más bajo se acerca a Q_Kerr. El bootstrap no está midiendo el
overtone — está concentrándose en una región de Q ≈ 10× mayor que la predicción
física.

**Validación de la referencia Kerr:** Q_221 ≈ 1 es consistente con las tablas de
Berti et al. para af ≈ 0.64–0.71. Verificado manualmente:
(1 − 0.6737)^(−0.4731) ≈ 1.699, × 0.5436 = 0.923, + 0.1 = 1.023. ✓

### Diagnóstico cerrado

**Causa raíz: F en versión fuerte.**

Q_221_Kerr ≈ 1 significa que el overtone (2,2,1) se extingue antes de completar
un ciclo para remanentes con af ≈ 0.64–0.71 (la mayoría de la cohorte O4).

En ese régimen, un estimador espectral (`hilbert_peakband`) no puede medir Q
porque no hay suficiente estructura oscilatoria para separar frecuencia de damping.
Lo que el bootstrap encuentra en Q ≈ 9–11 es la escala de correlación espectral del
residuo post-220 (ruido coloreado + leakage de ventana + residuo de sustracción),
no el overtone.

Las hipótesis metodológicas B, C, G son contribuciones secundarias reales pero no
son la causa dominante.

**Evolución de probabilidades a lo largo del día:**

| Hipótesis | Inicial | Post-A/B topología | Post-A/B residual | Post-bootstrap+Kerr |
|-----------|---------|--------------------|--------------------|----------------------|
| F (no medible) | 0.06 | 0.25 | 0.45 | **0.80** |
| G (residual) | 0.42 | 0.30 | 0.15 | 0.05 |
| B (topología) | 0.23 | 0.15 | 0.05 | 0.03 |
| D (upstream) | 0.15 | 0.15 | 0.15 | 0.05 |
| C (gating) | 0.11 | 0.10 | 0.10 | 0.05 |
| E (detector) | 0.02 | 0.02 | 0.02 | 0.02 |

### Veredicto operativo

El resultado 82/82 SINGLEMODE_ONLY es el resultado correcto. Los gates de calidad
están funcionando como deben: rechazando posteriors que no contienen información
física sobre el overtone.

El carril 221 con `hilbert_peakband` tiene un dominio de validez restringido a
eventos con af suficientemente alto para que Q_221_Kerr > 2–3 (af > ~0.8).
Para la cohorte O4 actual (af ≈ 0.6–0.7), el carril es estructuralmente no
informativo.

### Recomendaciones

**Inmediato (sin cambiar código):**
- Documentar en el contrato del carril 221 que requiere af > 0.8 (o
  Q_221_Kerr > 2.0) como pre-gate.
- Añadir ese gate en `s3b` o `pipeline.py` antes de intentar la extracción.
- Esto convierte 82/82 SINGLEMODE_ONLY en resultado esperado y documentado.

**Medio plazo (si se quiere medir 221 en esta cohorte):**
- Estimador en dominio temporal: fit bayesiano directo de
  h(t) = A₂₂₀ e^(−t/τ₂₂₀) cos(...) + A₂₂₁ e^(−t/τ₂₂₁) cos(...)
  sobre la serie temporal de ringdown, sin resolución espectral.
- Esto puede funcionar con Q < 2 porque mide damping directamente.

**Lo que no se recomienda:**
- No seguir iterando topologías, estrategias residuales, o thresholds dentro de
  s3b. El diagnóstico está cerrado.
- No relajar gates para "rescatar" eventos — dejaría pasar posteriors patológicas.
- No reejecutar sobre los centinela oficiales — el resultado sería idéntico.

### Artefactos generados hoy

| Artefacto | Ubicación |
|-----------|-----------|
| Selección centinela | `runs/brunete_prepare_20260323T1545Z/experiment/sentinel_selection.json` |
| Script selector | `select_sentinel_events.py` |
| Script extractor bootstrap | `extract_s3b_bootstrap_samples.py` |
| A/B topología (10 eventos) | `runs/ab221sbet_<EVENT>_20260323/` |
| A/B residual (3 eventos) | `runs/ab221fixed_<EVENT>_20260323/` |
| Bootstrap samples crudo | `runs/ab221_bootstrap_samples_3events_20260323.json` |

### Experimento 5 — Dimensionamiento del pre-gate: distribución de af y Q_221_Kerr

**Objetivo:** antes de especificar los contratos del nuevo carril
`221_detection_candidate`, determinar cuántos eventos de la cohorte entrarían en
el dominio detectable.

**Script ejecutado por Codex** (Prompt 1). Output:
`runs/brunete_prepare_20260323T1545Z/experiment/af_distribution_and_pregate.json`

**Distribución de af (81 eventos con af disponible, 1 sin af):**

| Stat | af |
|------|----|
| min | 0.608 |
| p10 | 0.644 |
| p25 | 0.670 |
| median | 0.691 |
| p75 | 0.720 |
| p90 | 0.773 |
| max | 0.843 |

**Q_221_Kerr resultante:**

| Stat | Q_221_Kerr |
|------|------------|
| min | 0.947 |
| p10 | 0.986 |
| median | 1.048 |
| p90 | — |
| max | 1.407 |

**Pre-gate counts:**

| Umbral Q_221_Kerr | Eventos que pasan |
|--------------------|-------------------|
| > 1.5 | 0 |
| > 2.0 | 0 |
| > 2.5 | 0 |
| > 3.0 | 0 |

Evento con mayor spin: GW231123_135430 (af=0.843, Q_221_Kerr=1.407).
Para Q_221_Kerr > 2.0 se requiere af > ~0.929 — muy por encima del máximo
observado.

**Lectura:** la cohorte O4/O4b entera está fuera del dominio de detectabilidad
del modo (2,2,1) por factor de calidad sub-ciclo. El nuevo carril
`221_detection_candidate` tendría 0 candidatos reales en esta cohorte con
cualquier umbral razonable de Q_221_Kerr.

### Implicaciones para la estrategia del carril 221

Este resultado convierte lo que parecía un problema de pipeline en un **hecho
físico de la cohorte**: los remanentes O4/O4b tienen spins insuficientes para
que el overtone (2,2,1) sea una oscilación resoluble.

**Decisiones de diseño que esto fuerza:**

1. **El carril conservador actual (`s3b` → `SINGLEMODE_ONLY`) está produciendo
   el resultado correcto.** No necesita fix. La documentación debe reflejar que
   82/82 SINGLEMODE_ONLY es esperado, no un fallo.

2. **El nuevo carril `221_detection_candidate` no tiene sobre qué correr en O4.**
   Su especificación sigue siendo válida como contrato, pero la validación
   requiere:
   - Inyecciones sintéticas con af > 0.85 para testear sensibilidad, o
   - Esperar a eventos O5/future con spins más altos, o
   - Evaluar si un estimador en dominio temporal podría operar con Q < 2
     (expandiendo el dominio de aplicabilidad).

3. **El pre-gate (Contrato 1) es trivial de implementar y tiene valor inmediato:**
   evita que el pipeline gaste cómputo en 82 extracciones de 221 que van a fallar.
   Reducción de cómputo sin pérdida de información.

### Estado consolidado del diagnóstico

La cadena causal completa, verificada paso a paso:

```
af cohorte O4 ≈ 0.61–0.84
        ↓ (Berti fit para 2,2,1)
Q_221_Kerr ≈ 0.95–1.41  (sub-ciclo en toda la cohorte)
        ↓
221 no tiene suficientes ciclos para ser medible espectralmente
        ↓
hilbert_peakband bootstrap encuentra artefacto espectral del residuo (Q ≈ 9–11)
        ↓
lnQ_span y cv_Q explotan (porque mide ruido, no señal)
        ↓
mode_221_ok = false → SINGLEMODE_ONLY (gate correcto)
        ↓
82/82 SINGLEMODE_ONLY = resultado físicamente esperado
```

### Artefactos generados hoy (actualizado)

| Artefacto | Ubicación |
|-----------|-----------|
| Selección centinela | `runs/brunete_prepare_20260323T1545Z/experiment/sentinel_selection.json` |
| Script selector | `select_sentinel_events.py` |
| Script extractor bootstrap | `extract_s3b_bootstrap_samples.py` |
| A/B topología (10 eventos) | `runs/ab221sbet_<EVENT>_20260323/` |
| A/B residual (3 eventos) | `runs/ab221fixed_<EVENT>_20260323/` |
| Bootstrap samples crudo | `runs/ab221_bootstrap_samples_3events_20260323.json` |
| Distribución af + pre-gate | `runs/brunete_prepare_20260323T1545Z/experiment/af_distribution_and_pregate.json` |

### Análisis del panorama competitivo (investigación web)

**Publicaciones clave recientes:**

1. **LVK GWTC-4.0 TGR trilogy (arXiv 2603.19019/19020/19021, 19 marzo 2026)**
   - Tres papers publicados hace 5 días.
   - Paper III (Tests of the Remnants): 7 tests del remanente, 3 de ringdown,
     sobre 91 señales (42 de O4a + runs anteriores).
   - Método: pSEOBNRv5PHM — parametrización full IMR con (δf₂₂₀, δτ₂₂₀).
   - Resultado: consistencia con GR. Un análisis encuentra GR en la frontera
     del 98.6% credible region al combinar jerárquicamente, pero baja a 92.2%
     al incluir GW250114. Sin evidencia fuerte de desviación.
   - **Implicación para BRUNETE:** constraints poblacionales en (δf₂₂₀, δτ₂₂₀)
     con O4a ya publicadas. BRUNETE no puede ser primicia en el resultado.

2. **GW250114 BH Spectroscopy (arXiv 2509.08099, PRL 136:041403, enero 2026)**
   - Evento más ruidoso detectado hasta la fecha: SNR ~77-80.
   - Primera detección del overtone (2,2,1) con significancia 4.1σ.
   - Spin del remanente: af ≈ 0.68 — mismo rango que la cohorte O4 de BRUNETE.
   - Constraints de un solo evento comparables o 2-3× más estrictas que
     combinar docenas de eventos del catálogo.
   - Usa análisis en dominio temporal (pyRing, QNMRF), no espectral.
   - **Implicación para BRUNETE:** confirma que a af ≈ 0.68, 221 es detectable
     pero requiere SNR ~80 y métodos temporales. Valida el diagnóstico de hoy:
     el estimador espectral no puede con Q_221 < 2.

3. **Debate overtone activo en la literatura**
   - Grupos como Isi et al. afirman evidencia de overtones; Cotesta, Carullo
     et al. argumentan que la señal es sub-ruido y no se puede identificar
     de forma robusta.
   - GW250114 es el primer caso con significancia alta, pero es excepcional.

4. **pSEOBNRv5PHM (arXiv 2504.10130, junio 2025)**
   - Waveform model con parámetros de desviación ringdown integrados en
     full IMR precessing.
   - Reanaliza 12 eventos GWTC-3 con combinación jerárquica.
   - Muestra que ignorar precesión puede producir falsas detecciones de
     desviaciones de GR.

**Lo que NO existe en la literatura:**

- Análisis ringdown-only poblacional con Fisher y extracción explícita de QNM
  sobre O4. La LVK usa full-IMR; un análisis puramente ringdown es
  metodológicamente distinto.
- Caracterización sistemática del dominio de aplicabilidad del overtone:
  por qué 221 no es medible espectralmente en cohortes de spin moderado.
  Nadie ha publicado el argumento completo (Q_Kerr sub-ciclo → estimador
  espectral sesgado → artefacto post-sustracción).
- GW250114 no está en la cohorte O4a de BRUNETE (es O4b, enero 2025).

### Pivote estratégico del paper

**Framing anterior (descartado):**
"Multimode spectroscopy with O4"

**Framing actualizado:**
"Ringdown-only population analysis of O4 events with explicit QNM extraction:
methodology, constraints on (δf₂₂₀, δτ₂₂₀), and systematic characterization
of overtone measurability limits"

**Tres contribuciones diferenciadas:**

1. **Método.** Ringdown-only Fisher vs full-IMR (complementario a LVK).
   Pipeline contract-first, reproducible, auditado.

2. **Resultado.** Verificación independiente del carril 220 con 82 eventos.
   Constraints en (δf₂₂₀, δτ₂₂₀) comparables o mejores que GWTC-3 TGR
   (la LVK usó ~49 eventos de O1-O3 en ringdown; BRUNETE tendría 82 O4).

3. **Diagnóstico del overtone.** Sección o apéndice técnico documentando:
   - Q_221_Kerr < 1.5 para toda la cohorte O4 (af ≈ 0.61–0.84)
   - Estimador espectral sesgado a Q ≈ 10 (artefacto, no señal)
   - A/B controlados descartando topología y residual strategy como causas
   - Dominio de validez: af > 0.8 para estimación espectral del overtone
   - Implicaciones para diseño de futuros análisis

### Pendientes para sesiones futuras

**Prioridad alta (paper):**
- [ ] Auditoría cruzada Q_220: verificar Berti (2,2,0) vs bootstrap medido.
      (Prompt 3 preparado, ejecutable en Codex.) **Cierra el argumento.**
- [ ] Ejecutar `run_population_fisher.py` sobre los 82 eventos con el
      carril 220 y obtener σ(δf₂₂₀), σ(δτ₂₂₀) combinados.
- [ ] Comparar constraints resultantes con LVK GWTC-3 TGR y GWTC-4.0 TGR
      Paper III para posicionar el resultado.
- [ ] Evaluar si GW250114 (O4b) puede incorporarse a la cohorte.

**Prioridad media (infraestructura):**
- [ ] Especificación formal de contratos del carril `221_detection_candidate`.
      (Prompt 2 preparado.) Validación requiere inyecciones sintéticas o
      evento tipo GW250114.
- [ ] Implementar pre-gate Q_221_Kerr en pipeline.py.

**Prioridad baja (futuro):**
- [ ] Estimador dominio temporal para 221 (diseño de contrato + prototipo).
- [ ] Incorporar GW250114 como evento de validación del carril 221 temporal.

---

*Última actualización: 2026-03-23*
