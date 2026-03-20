# Auditoría técnica: ¿Por qué el modo 221 no sale usable en `s3b_multimode_estimates`?

**Fecha:** 2026-03-19  
**Pipeline:** BASURIN — Stage `s3b_multimode_estimates`  
**Artefacto de referencia:** `runs/mvp_GW250114_082203_221_20260319T130500Z/s3b_multimode_estimates/outputs/multimode_estimates.json`  
**Código fuente auditado:** `mvp/s3b_multimode_estimates.py` (1431 líneas)

---

## 1. Resumen ejecutivo

El 221 no sale usable porque el pipeline tiene **un fallo crítico de diseño en la estrategia de bandas** y **varios problemas amplificadores** en la cadena bootstrap → estadístico → gate. El efecto combinado hace que la extracción del 221 sea virtualmente imposible incluso en eventos donde la literatura lo detecta.

**Hallazgo principal (CRÍTICO):** La estrategia `kerr_centered_overlap` construye `band_221` como una expansión mínima de `band_220` (±1.5–3 Hz de padding). Esto ubica la banda del 221 alrededor de ~250 Hz para un evento típico, cuando f₂₂₁ ≈ 375–425 Hz. **El estimador está midiendo ruido residual de la sustracción del 220, no el overtone.**

**Hallazgos secundarios (SIGNIFICATIVOS):**

- El bootstrap re-estima 220 independientemente en cada iteración antes de sustraerlo, amplificando varianza en el residual.
- `cv_Q` se calcula sobre `exp(lnQ)`, donde distribuciones con spread moderado en log generan cv > 1 trivialmente.
- `max_lnq_span_221 = 1.0` está calibrado implícitamente para el 220 y es demasiado restrictivo para un overtone con Q intrínsecamente bajo.
- `p50` de bootstrap no es un estadístico adecuado cuando la distribución subyacente es multimodal o heavy-tail.

**Diagnóstico de nivel:** El problema es **primariamente metodológico** (bandas + bootstrap), con una componente de **parametrización/thresholds** que agrava. La componente científica (SNR insuficiente del 221) puede ser real pero es actualmente indistinguible del artefacto del pipeline.

---

## 2. Hipótesis más probables (ordenadas por impacto)

### H1. [CRÍTICO — Fallo de diseño confirmado] Banda 221 no contiene f₂₂₁

**Mecanismo:**

En `_resolve_mode_bands()` (líneas 426–479), cuando existe `f220_kerr_hz` (i.e., para cualquier evento conocido), se ejecuta la estrategia `kerr_centered_overlap`:

```python
# band_220 se centra en Kerr con half_width ≈ 15 Hz mínimo
band_220 = (max(band_low, kerr_low), min(band_high, kerr_high))

# band_221 se construye como band_220 ± padding minúsculo
pad_low_hz  = 0.1 * half_width_hz   # ≈ 1.5 Hz
pad_high_hz = 0.2 * half_width_hz   # ≈ 3.0 Hz
band_221 = (band_220[0] - pad_low_hz, band_220[1] + pad_high_hz)
```

Para GW150914 (f₂₂₀ ≈ 251 Hz, half_width ≈ 15 Hz):

- `band_220 ≈ (236, 266)` Hz
- `band_221 ≈ (234.5, 269)` Hz ← **centrada en el fundamental, no en el overtone**
- `f₂₂₁ ≈ 376–427 Hz` ← **fuera de la banda por ~110 Hz**

El estimador `_estimate_221_from_signal_in_bands()` (línea 531) sustrae la plantilla 220 y luego llama a `_estimate_observables_in_band(residual, fs, band=band_221)`. Como `band_221` cubre la región del 220, lo que se mide es ruido de sustracción imperfecta, no el overtone.

**Por qué afecta mucho más al 221 que al 220:** El 220 usa su propia banda Kerr centrada correctamente. El 221 hereda esa misma banda con padding cosmético.

**Clasificación:** Fallo de diseño confirmado por inspección de código. No requiere test empírico para confirmarse.

**Nota importante:** El fallback `default_split_60_40` (líneas 412–423) sí ubica `band_221` en la porción superior `(band_low + 0.6*(band_high-band_low), band_high)`, lo cual es razonable. Pero este fallback solo se usa cuando no hay `f220_kerr_hz` disponible, es decir, para eventos desconocidos. Paradójicamente, los eventos mejor caracterizados (con Kerr lookup) son los que usan la estrategia rota.


### H2. [SIGNIFICATIVO — Fallo de diseño probable] Bootstrap con sustracción 220 no-condicional

**Mecanismo:**

`_bootstrap_mode_log_samples()` (líneas 362–392) hace block bootstrap del signal completo. Para cada iteración:

1. Resamplea bloques del signal
2. Llama al estimador del 221, que internamente:
   a. Re-estima 220 en el signal resampled
   b. Construye plantilla 220 con esos parámetros
   c. Sustrae plantilla
   d. Estima 221 en el residual

El paso (a) produce estimaciones 220 ligeramente diferentes en cada iteración. La sustracción (c) deja residuales cualitativamente distintos. El paso (d) mide propiedades del residual, no del 221 real.

Esto infla la varianza bootstrap del 221 porque mezcla dos fuentes de variabilidad:

- Variabilidad real del 221 (señal física)
- Variabilidad del error de sustracción del 220 (artefacto de pipeline)

**Por qué afecta mucho más al 221:** El 220 se estima directamente de la señal. El 221 se estima sobre un residual que depende de una estimación previa ruidosa.

**Clasificación:** Fallo de diseño probable. Un estimador correcto fijaría los parámetros del 220 (p.ej., usando la estimación puntual del 220 para todas las iteraciones bootstrap) o haría bootstrap conjunto.


### H3. [SIGNIFICATIVO — Diseño cuestionable] `cv_Q` sobre `exp(lnQ)` es inherentemente inestable

**Mecanismo:**

En `evaluate_mode()` (líneas 706–717):

```python
q_vals = np.exp(samples[:, 1])   # transforma de lnQ a Q
cv_q = std(q_vals) / mean(q_vals)
```

Si `lnQ_span` (p90 - p10) ≈ 2.82 (valor observado para el 221), entonces Q varía en un factor `exp(2.82) ≈ 16.8`. La distribución de Q = exp(lnQ) es log-normal, donde cv = sqrt(exp(σ²) - 1). Para σ(lnQ) ≈ 1 (spread moderado), cv ≈ 1.3. Para σ(lnQ) ≈ 1.5, cv ≈ 3.

El threshold `cv_threshold_221 = 1.0` se dispara trivialmente cuando la distribución bootstrap tiene spread moderado en lnQ. Esto no detecta "inestabilidad física" sino la consecuencia aritmética de calcular cv sobre una variable log-normal.

**Observación:** `cv_Q = 2.47` (valor observado) es consistente con un lnQ_span de ~2.8 bajo distribución log-normal. No implica necesariamente una estimación patológica — implica que cv_Q es una métrica mal condicionada para este espacio.

**Nota:** `cv_Q` genera flag pero no gatea `ok` directamente (líneas 714–717: solo `flags.append`, no `ok = False`). Sin embargo, `lnQ_span_explosive` SÍ gatea `ok = False` (línea 702), y ambas se disparan por la misma causa raíz.

**Clasificación:** Diseño cuestionable. El cv en espacio lineal sobre una variable log-distribuida siempre será inestable.


### H4. [SIGNIFICATIVO — Threshold mal calibrado] `max_lnq_span_221 = 1.0` es demasiado restrictivo

**Mecanismo:**

El default `--max-lnq-span-221` es 1.0 (línea 1248), igual que `--max-lnf-span-221`. Esto implica que el rango intercuartil P90-P10 de lnQ no puede exceder 1.0, es decir, Q no puede variar más de un factor e ≈ 2.7 entre P10 y P90.

Para el 221 con Q típico ≈ 3–5 (vs Q₂₂₀ ≈ 10–15), la incertidumbre relativa es naturalmente mayor. Un Q₂₂₁ real de 4 con ±50% de incertidumbre (completamente razonable para un overtone débil) da Q ∈ (2, 6), lnQ ∈ (0.69, 1.79), lnQ_span = 1.1 — ya fuera de gate.

El threshold 1.0 para lnQ_span está implícitamente calibrado para la estabilidad del 220, no del 221.

**Comparación:** `max_lnq_span_220 = 3.0` (línea 1247), que es 3× más permisivo que el del 221. Esto es exactamente al revés de lo que la física sugiere: el modo más débil necesita mayor tolerancia, no menor.

**Clasificación:** Threshold mal calibrado. Probablemente un error de copiar defaults del 220 y ajustarlos en la dirección equivocada.


### H5. [MODERADO — Hipótesis plausible] Ventana temporal no adaptada para 221

**Mecanismo:**

El 221 tiene τ₂₂₁ << τ₂₂₀ (el overtone decae mucho más rápido). Una ventana de ringdown optimizada para capturar el 220 incluye muchas muestras donde el 221 ya ha desaparecido y solo queda ruido. Estas muestras diluyen la SNR del 221 y contribuyen ruido al bootstrap.

El stage usa un único `t0_offset` heredado de s3 (líneas 1319–1327), sin política diferenciada para el overtone. La literatura frecuentemente usa ventanas más cortas o adaptativas para la extracción del 221.

**Clasificación:** Hipótesis plausible que requiere test empírico.


### H6. [MODERADO — Hipótesis plausible] `p50` promedia regímenes cualitativamente distintos

**Mecanismo:**

Si la distribución bootstrap del 221 es bimodal (p.ej., ~60% de iteraciones capturan ruido con f ≈ f₂₂₀ y Q bajo, ~40% capturan algo cercano al overtone real), entonces p50 es un promedio entre dos poblaciones sin significado físico. Esto es especialmente probable dado H1 (banda incorrecta).

`valid_fraction = 0.84` (valor observado) indica que ~16% de iteraciones fallan completamente, pero no dice nada sobre la coherencia de las que sí producen un valor.

**Clasificación:** Hipótesis plausible, dependiente de la distribución real de las muestras bootstrap.


### H7. [BAJO — Especulación informada] Expectativa de detección puede ser incomparable

La literatura que detecta el 221 (Isi et al. 2019, Cotesta et al. 2022, etc.) típicamente usa:

- Modelos bayesianos completos con priors Kerr-informados
- Templates de dos modos ajustados simultáneamente (no sustracción secuencial)
- Ventanas de análisis optimizadas con t₀ variable
- Criterios de detección basados en Bayes factor, no en estabilidad bootstrap

Comparar directamente la expectativa de esos resultados con un pipeline agnóstico basado en Hilbert + bootstrap no es apropiado. El "fracaso" del 221 en BASURIN puede reflejar correctamente que el método actual no tiene potencia suficiente para ese modo.

**Clasificación:** Especulación que requeriría comparación directa con pipeline alternativo.

---

## 3. Evidencia que las apoya

### Para H1 (banda incorrecta):

| Evidencia | Ubicación |
|-----------|-----------|
| Construcción de `band_221` en `kerr_centered_overlap` | `s3b_multimode_estimates.py` líneas 458–463 |
| Valores de padding: `S3B_221_OVERLAP_LOW_PAD_FRAC = 0.1`, `S3B_221_OVERLAP_HIGH_PAD_FRAC = 0.2` | Líneas 55–56 |
| `band_strategy` en el artefacto de salida | `multimode_estimates.json → source.band_strategy` |
| Fallback correcto en `_split_mode_bands` da banda superior al 221 | Líneas 412–423 |

**Verificación directa:** Abrir el artefacto `multimode_estimates.json` del run problemático y comprobar `source.band_strategy.method`. Si es `"kerr_centered_overlap"`, confirma que se usó la estrategia rota. Comparar `mode_221_band_hz` con f₂₂₁ esperada del evento.

### Para H2 (bootstrap no-condicional):

| Evidencia | Ubicación |
|-----------|-----------|
| Bootstrap llama al estimador completo (incluye re-estimación del 220) | Líneas 376–388 |
| Estimador 221 re-estima 220 internamente | `_estimate_221_from_signal_in_bands()` línea 538 |
| No hay mecanismo para fijar parámetros del 220 entre iteraciones | Ausencia en `_bootstrap_mode_log_samples()` |

### Para H3 (cv_Q mal condicionada):

| Evidencia | Ubicación |
|-----------|-----------|
| cv calculado sobre `exp(samples[:, 1])` | Líneas 708–711 |
| `cv_Q = 2.47`, `lnQ_span = 2.82` — consistente con cv log-normal | Datos del run auditado |
| cv_Q flag no gatea directamente, pero lnQ_span sí | Líneas 714–717 vs 698–703 |

### Para H4 (threshold restrictivo):

| Evidencia | Ubicación |
|-----------|-----------|
| `--max-lnq-span-221 = 1.0` vs `--max-lnq-span-220 = 3.0` | Líneas 1247–1248 |
| `lnQ_span = 2.82` para 221 → dispara `lnQ_span_explosive` | Datos del run |
| Gate `ok = False` al exceder threshold | Líneas 698–703 |

---

## 4. Experimentos de diagnóstico prioritarios

### Experimento 1: Verificar banda del 221 en artefacto real

**Prioridad:** MÁXIMA — discrimina H1 sin tocar código.

- **Qué hacer:** Abrir `multimode_estimates.json` del run `mvp_GW250114_082203_221_20260319T130500Z`. Leer `source.band_strategy.method`, `source.band_strategy.mode_221_band_hz`, y `source.band_strategy.f220_kerr_hz`. Comparar `mode_221_band_hz` con f₂₂₁_Kerr esperada.
- **Qué NO tocar:** Nada. Es lectura pura.
- **Artefacto esperado:** Los valores ya existentes en el JSON.
- **Resultado esperado:** Si `method = "kerr_centered_overlap"` y `mode_221_band_hz` está centrada alrededor de f₂₂₀ (no de f₂₂₁), H1 queda confirmada definitivamente.


### Experimento 2: Re-ejecutar con fallback forzado (sin Kerr lookup)

**Prioridad:** ALTA — testa H1 experimentalmente.

- **Qué hacer:** Ejecutar `s3b_multimode_estimates` para el mismo evento pero pasando un `event_id` que no tenga Kerr lookup (forzando `_resolve_f220_kerr_hz` a devolver `None`), para que use `default_split_60_40`.
- **Qué NO tocar:** Nada más del pipeline. Mismo signal, mismo n_bootstrap, mismo seed.
- **Artefacto esperado:** Nuevo `multimode_estimates.json` con `band_strategy.method = "default_split_60_40"`.
- **Resultado que apoyaría H1:** Si con el split 60/40, `lnQ_span` del 221 baja sustancialmente y/o `cv_Q` mejora, el problema era la banda. Si no mejora, hay otros factores dominantes.


### Experimento 3: Relajar thresholds del 221

**Prioridad:** ALTA — testa H4 y distingue "modo inestable" de "gate demasiado estricto".

- **Qué hacer:** Re-ejecutar con `--max-lnq-span-221 3.0` (igualando al 220) y `--cv-threshold-221 3.0`. Observar si el 221 pasa a `usable = true` y qué valores produce.
- **Qué NO tocar:** Banda, bootstrap, nada más.
- **Artefacto esperado:** Nuevo `multimode_estimates.json`.
- **Resultado que apoyaría H4:** Si el 221 se marca usable con thresholds relajados Y produce f₂₂₁/Q₂₂₁ en el rango esperado por Kerr, el threshold era demasiado restrictivo. Si se marca usable pero los valores son físicamente absurdos, el threshold estaba protegiendo correctamente.


### Experimento 4: Bootstrap con 220 fijo

**Prioridad:** MEDIA — testa H2, requiere modificación de código.

- **Qué hacer:** Modificar `_bootstrap_mode_log_samples` (o crear variante) para que el estimador del 221 use una estimación puntual fija del 220 (la del signal original, no la del resample) al construir la plantilla de sustracción.
- **Qué NO tocar:** Thresholds, bandas (corregir H1 primero).
- **Artefacto esperado:** Nuevo artefacto con `fit.method = "hilbert_peakband_fixed220"` o similar.
- **Resultado que apoyaría H2:** Reducción significativa de `lnQ_span` y `cv_Q` del 221 respecto al baseline.


### Experimento 5: Distribución bootstrap cruda del 221

**Prioridad:** MEDIA — testa H6, diagnóstico puro.

- **Qué hacer:** Instrumentar `evaluate_mode` para volcar las muestras bootstrap raw `(lnf, lnQ)` del 221 a un archivo auxiliar (p.ej., `bootstrap_samples_221.npy`). Visualizar: scatter plot, histograma marginal de lnQ, test de multimodalidad (dip test o visual).
- **Qué NO tocar:** Nada del pipeline principal.
- **Artefacto esperado:** Archivo auxiliar con muestras crudas.
- **Resultado que apoyaría H6:** Distribución claramente bimodal o con colas extremas en lnQ. Si es unimodal y concentrada, p50 es razonable y el problema está en otra capa.

---

## 5. Cambios mínimos sugeridos

### Cambio 1: [URGENTE] Corregir banda del 221 en `kerr_centered_overlap`

**Archivo:** `mvp/s3b_multimode_estimates.py`, función `_resolve_mode_bands()`

**Cambio:** Construir `band_221` centrada en f₂₂₁_Kerr (no en f₂₂₀), usando el ratio Kerr f₂₂₁/f₂₂₀ del remnant. Si no se dispone del ratio exacto, usar el rango teórico [1.5, 1.7] × f₂₂₀.

```python
# Propuesta esquemática (no copiar literalmente sin revisar):
f221_kerr_low = 1.45 * f220_kerr_hz   # margen inferior conservador
f221_kerr_high = 1.75 * f220_kerr_hz  # margen superior conservador
band_221 = (
    max(band_low, f221_kerr_low - pad_low_hz),
    min(band_high, f221_kerr_high + pad_high_hz),
)
```

**Impacto en gobernanza:** Cambio en la lógica de `_resolve_mode_bands` solamente. No afecta contratos downstream. El campo `band_strategy` ya documenta la estrategia usada. Añadir `f221_kerr_hz_estimated` al strategy dict para auditabilidad.


### Cambio 2: [IMPORTANTE] Invertir asimetría de thresholds 220/221

**Archivo:** `mvp/s3b_multimode_estimates.py`, defaults de argparse.

**Cambio:** `--max-lnq-span-221` debería ser ≥ `--max-lnq-span-220`, no menor. Propuesta mínima: igualar a 3.0 (el valor del 220) o subir a 4.0. Hacer lo mismo con `--max-lnf-span-221`.

**Impacto en gobernanza:** Solo cambia defaults. Completamente backward-compatible.


### Cambio 3: [RECOMENDADO] Calcular `cv_Q` en espacio log

**Archivo:** `mvp/s3b_multimode_estimates.py`, función `evaluate_mode()`

**Cambio:** Reemplazar:
```python
cv_q = std(exp(lnQ)) / mean(exp(lnQ))
```
por:
```python
cv_lnq = std(lnQ_samples) / abs(mean(lnQ_samples))  # o usar MAD/median
```

Alternativamente, usar `lnQ_span` (que ya se calcula) como métrica primaria de dispersión y eliminar `cv_Q` como gate separado.

**Impacto en gobernanza:** Añade campo nuevo (`cv_lnQ` o similar). No rompe esquema existente si se mantiene `cv_Q` como campo informativo.


### Cambio 4: [RECOMENDADO] Añadir volcado de bootstrap samples como artefacto auxiliar

**Archivo:** `mvp/s3b_multimode_estimates.py`, función `evaluate_mode()`

**Cambio:** Opcionalmente guardar `bootstrap_samples_{label}.npy` en el directorio de outputs del stage. Controlado por flag `--dump-bootstrap`.

**Impacto en gobernanza:** Artefacto auxiliar, no gateador. Registrar en manifest como artefacto opcional.


### Cambio 5: [EXPLORATORIO] Separar "estimación disponible" de "estimación usable"

**Concepto:** Actualmente, `ok = False` impide que downstream use CUALQUIER información del 221. Podría haber un estado intermedio:

- `mode_221_estimated = True` (se produjo un punto estimado)
- `mode_221_usable = False` (no pasa gates de calidad)
- `mode_221_informative = True/False` (nueva: ¿aporta información parcial?)

Esto permitiría que downstream use la estimación con caveats (p.ej., para diagnóstico o como prior débil) sin tratarla como firme.

---

## 6. Riesgos de interpretación

### Lo que NO debemos concluir todavía:

1. **"No detectar 221" ≠ "221 no existe en este evento."** Con H1 confirmada (banda incorrecta), la no-detección actual no tiene valor informativo sobre la presencia física del overtone. No citar estos resultados como evidencia contra el 221.

2. **"221 inestable" ≠ "método fundamentalmente inútil."** La inestabilidad observada es mayoritariamente un artefacto de la banda incorrecta (H1). Tras corregir H1, es posible que `hilbert_peakband` funcione razonablemente para eventos de alta SNR.

3. **"220 controlado" ≠ "221 debería salir bien con el mismo método."** El 220 tiene SNR ~10–100× mayor, Q más alto, y se mide directamente (sin sustracción previa). No hay razón para esperar que los mismos thresholds o bootstrap funcionen igual para ambos modos.

4. **"Relajar thresholds mejora usabilidad" ≠ "La estimación es buena."** Si el Exp. 3 marca el 221 como usable con thresholds relajados, hay que verificar que los valores sean físicamente consistentes (f₂₂₁/f₂₂₀ ≈ 1.5–1.7, Q₂₂₁ ≈ 2–6 para chi_eff típico).

5. **"El pipeline reproduce la literatura" ≠ "El pipeline es correcto."** Incluso si tras las correcciones el 221 sale en el rango esperado, BASURIN usa un método fundamentalmente distinto (Hilbert agnóstico vs Bayesian Kerr-informado). La concordancia podría ser fortuita.

6. **No confundir el gate correcto con el gate actual.** Que `lnQ_span_explosive` se dispare no significa que el gate esté mal diseñado en general — el gate está detectando un problema real (dispersión enorme), solo que ese problema es causado upstream por la banda incorrecta, no por inestabilidad intrínseca del modo.

---

## 7. Recomendación final

### Acción inmediata (hoy):

1. **Ejecutar Experimento 1** (verificar `band_strategy` en el artefacto existente) para confirmar H1 sin tocar código.

### Acción a corto plazo (esta semana):

2. **Corregir la banda del 221** (Cambio 1) — este es el blocker principal.
3. **Igualar thresholds** (Cambio 2) — elimina la asimetría injustificada.
4. **Re-ejecutar** el evento de referencia y 2–3 eventos más con alta SNR para evaluar el efecto combinado.

### Acción a medio plazo (próximas 2 semanas):

5. **Ejecutar Experimentos 4 y 5** para caracterizar la contribución del bootstrap y la distribución real de las muestras.
6. **Implementar Cambio 3** (cv en espacio log) si los experimentos confirman que cv_Q en espacio lineal sigue siendo problemática tras corregir la banda.

### Decisión estratégica pendiente:

La pregunta de fondo — ¿debe BASURIN medir el 221 de forma agnóstica o Kerr-condicionada? — no tiene respuesta todavía. Recomiendo:

- Corregir primero los fallos de diseño (H1, H2, H4).
- Evaluar el rendimiento del método agnóstico corregido en 5–10 eventos con 221 detectado en la literatura.
- Solo entonces decidir si se necesita un estimador Kerr-condicionado como alternativa o contraste.

No implementar un estimador Bayesiano/conjunto hasta que el agnóstico esté funcionando correctamente. De lo contrario, no sabremos qué parte del "éxito" del nuevo método es corrección real y qué parte es compensación de bugs del antiguo.

---

## Apéndice: Pistas obligatorias — respuestas directas

| Pregunta | Respuesta |
|----------|-----------|
| ¿Estamos pidiendo a un método diseñado para 220 que haga algo para lo que no fue concebido? | **Sí, parcialmente.** El método es agnóstico y en principio puede medir cualquier modo, pero la configuración (bandas, thresholds, bootstrap) está de facto optimizada para el 220. |
| ¿El uso de p50 en distribuciones patológicas fabrica pseudo-medidas? | **Probable.** Si la distribución es bimodal, p50 no tiene significado físico. Necesitamos Exp. 5 para confirmar. |
| ¿cv_Q es adecuada para 221? | **No en espacio lineal.** cv sobre exp(lnQ) diverge trivialmente. Debería calcularse en espacio log o reemplazarse por lnQ_span como métrica primaria. |
| ¿lnQ_span detecta problema real o parametrización mala? | **Ambos.** Detecta spread real (amplificado por H1 y H2), pero el threshold está mal calibrado (H4) y la causa raíz es la banda incorrecta (H1). |
| ¿Habría que medir 221 condicionado al 220/Kerr? | **No todavía.** Primero corregir H1. Si tras la corrección el método agnóstico sigue fallando, entonces sí, pero como contraste, no como reemplazo. |
| ¿El problema es científico o metodológico? | **Primariamente metodológico (H1 es un bug).** La componente científica (SNR real del 221) es actualmente no-distinguible del artefacto. |
| ¿Qué parte del fracaso es información útil y qué parte es artefacto? | **Actualmente ~100% artefacto** dado H1. Tras corregir H1: desconocido, necesita evaluación empírica. |
