# Metodología de informatividad predictiva para ringdown

## Predicción analítica de configuraciones informativas antes de ejecutar el pipeline

**Versión:** 1.0  
**Fecha:** 2026-03-06  
**Contexto:** BASURIN — respuesta al fallo de informatividad en GW150914 multimodo  
**Prerrequisito teórico:** Método Brunete v1.0

---

## 0. El problema en una frase

El gate de viabilidad multimodo rechaza el run porque `rel_iqr_f220 = 0.833 > 0.5`. Buscar configuraciones informativas por ensayo y error (barridos ciegos de `window_duration_s`, `band`, `dt_start_s`) es costoso y no escalable a un catálogo de N eventos. **Se necesita un criterio analítico que, dados los parámetros QNM esperados y la PSD del detector, prediga qué configuraciones serán informativas antes de gastar un run completo.**

---

## 1. Por qué falla: el argumento de escala temporal

### 1.1 La aritmética que explica todo

Para GW150914, los parámetros del modo fundamental (2,2,0) son bien conocidos:

- f₂₂₀ ≈ 251 Hz
- Q₂₂₀ ≈ 4.3
- τ₂₂₀ = Q/(πf) = 4.3/(π × 251) ≈ **5.45 ms**

El run actual usa `dt_start_s = 0.023` (23 ms después del tiempo de referencia del merger). Esto significa que el análisis comienza **4.2 tiempos de decaimiento** después del inicio del ringdown:

$$\frac{t_0}{\tau_{220}} = \frac{23}{5.45} \approx 4.2$$

La amplitud residual de la señal a ese punto es:

$$\frac{A(t_0)}{A(0)} = e^{-t_0/\tau} = e^{-4.2} \approx 0.015$$

Es decir, **queda el 1.5% de la amplitud original**. El modo fundamental está esencialmente extinto cuando empieza el análisis. La SNR efectiva en la ventana es una fracción mínima de la SNR total del ringdown.

### 1.2 La SNR efectiva en la ventana de análisis

La SNR² acumulada en una ventana [t₀, t₀ + T] para una sinusoide amortiguada es (Método Brunete, ec. 2.9 generalizada):

$$\rho^2(t_0, T) = \frac{A^2 \tau}{2\,S_n(f_{QNM})} \left[ e^{-2t_0/\tau} - e^{-2(t_0 + T)/\tau} \right]$$

Para PSD aproximadamente constante sobre el ancho del Lorentziano (válido lejos de líneas). Definiendo la SNR total del ringdown (desde t=0 hasta infinito) como:

$$\rho_{\text{total}}^2 = \frac{A^2 \tau}{2\,S_n(f_{QNM})}$$

la fracción de SNR capturada en la ventana es:

$$\eta(t_0, T) \equiv \frac{\rho^2(t_0, T)}{\rho_{\text{total}}^2} = e^{-2t_0/\tau} - e^{-2(t_0 + T)/\tau} = e^{-2t_0/\tau}\left(1 - e^{-2T/\tau}\right)$$

**Evaluación numérica para el run actual** (t₀ = 23 ms, T = 60 ms, τ = 5.45 ms):

$$\eta = e^{-2 \times 23/5.45} \times (1 - e^{-2 \times 60/5.45}) = e^{-8.44} \times (1 - e^{-22.0}) \approx 2.2 \times 10^{-4}$$

Solo el **0.022%** de la SNR² total del ringdown está disponible en esta ventana. Aunque GW150914 tenga una SNR de ringdown total ~8, la SNR efectiva en la ventana actual es ρ_eff ≈ 8 × √(0.00022) ≈ **0.12**. Esto no es un run no informativo por accidente; es **algebraicamente imposible** que sea informativo con esta configuración para el modo (2,2,0).

### 1.3 El conflicto fundamental entre modos

El t₀_sweep seleccionó t₀ = 20 ms como "óptimo". ¿Óptimo para qué? Para el overtone (2,2,1), que necesita empezar tarde para evitar contaminación de la fase de coalescencia. Pero el (2,2,1) tiene τ₂₂₁ ≈ 1.7 ms (factor ~3 menor que τ₂₂₀), así que a 20 ms ya lleva ~12 tiempos de decaimiento y es aún más invisible.

**Conclusión de escala:** con un t₀ de 20-23 ms, ninguno de los dos modos tiene SNR suficiente en la ventana. El t₀ sweep puede haber convergido a un mínimo local de alguna métrica de estabilidad que no correlaciona con la informatividad Fisher del modo fundamental.

Esto revela un problema de diseño: **el criterio de optimización del t₀ sweep no está alineado con el criterio de informatividad del gate multimodo**. El sweep optimiza una cosa; el gate evalúa otra.

---

## 2. El marco teórico: predicción de informatividad vía Fisher

### 2.1 La conexión Cramér-Rao ↔ rel_iqr

El gate de viabilidad multimodo rechaza si `rel_iqr_f220 > 0.5`. Este estadístico mide la dispersión relativa del posterior bootstrap de f₂₂₀. Para una distribución aproximadamente gaussiana (válida a SNR moderada), el IQR se relaciona con la desviación estándar como:

$$\text{IQR} \approx 1.35\,\sigma$$

Por tanto:

$$\text{rel\_iqr}_{f220} \approx 1.35 \times \frac{\sigma_f}{f_{220}}$$

Y del Método Brunete (ec. 2.12), la cota de Cramér-Rao para la precisión relativa en frecuencia es:

$$\frac{\sigma_f}{f} = \frac{1}{\pi\,Q\,\rho\,\sqrt{2}}$$

Combinando:

$$\text{rel\_iqr}_{f220} \approx \frac{1.35}{\pi\,Q_{220}\,\rho_{\text{eff}}\,\sqrt{2}} = \frac{0.304}{Q_{220}\,\rho_{\text{eff}}}$$

### 2.2 La condición de informatividad analítica

Para que el gate pase (`rel_iqr < 0.5`):

$$\frac{0.304}{Q_{220}\,\rho_{\text{eff}}} < 0.5$$

$$\boxed{Q_{220} \times \rho_{\text{eff}}(t_0, T) > 0.61}$$

Esta es la **condición de informatividad predictiva**. Es evaluable analíticamente antes de ejecutar el pipeline, porque:

- Q₂₂₀ viene de las fórmulas de Kerr (tabla de Berti) dado el spin estimado del remanente.
- ρ_eff viene de la fórmula de la sección 1.2 dados t₀, T, y una estimación de ρ_total.

**Nota de calibración.** La cota de Cramér-Rao es un *lower bound* en σ_f: el estimador bootstrap real tendrá σ ≥ σ_CR. Por tanto, la condición Q×ρ_eff > 0.61 es necesaria pero no suficiente. En la práctica, la degradación típica del bootstrap respecto a CR es un factor ~1.5-3× (depende de la no-gaussianidad del ruido y del método espectral). Una condición operativa segura sería:

$$Q_{220} \times \rho_{\text{eff}}(t_0, T) > \alpha_{\text{safety}} \times 0.61$$

donde α_safety ∈ [2, 3] se calibra empíricamente con los runs existentes.

### 2.3 Verificación con el run fallido

Para el run actual: Q₂₂₀ = 4.3, ρ_eff ≈ 0.12.

$$Q \times \rho_{\text{eff}} = 4.3 \times 0.12 = 0.52$$

Esto está justo en el límite de la condición CR (0.61), pero recordando que el estimador real es peor que CR por un factor ~2-3×:

$$\text{rel\_iqr predicho} \approx \frac{0.304}{0.52} \times \alpha \approx 0.58 \times 2.5 \approx 1.5$$

Comparable con el observado 0.833. La discrepancia (1.5 vs 0.833) indica que el estimador no es tan malo como 2.5× CR para este caso, pero el orden de magnitud y la conclusión (no informativo) son correctos.

---

## 3. Metodología: el preflight de informatividad

### 3.1 Concepto: Stage 0.5 — Preflight Fisher

La propuesta es un **diagnóstico pre-ejecución** que, antes de lanzar s3/s3b, evalúa analíticamente si la configuración elegida tiene posibilidad de producir un run informativo. No es un stage del pipeline propiamente dicho; es un filtro de viabilidad que evita gastar cómputo en configuraciones algebraicamente condenadas.

**Inputs del preflight:**

1. Parámetros QNM esperados del remanente: (f_QNM, Q_QNM) para cada modo, derivados del spin estimado vía fórmulas de Kerr (ya disponibles en `kerr_qnm_fits.py`).
2. PSD del detector en la banda del ringdown: S_n(f_QNM). Disponible tras s1 (o estimable de la PSD analítica).
3. Amplitud efectiva A o, equivalentemente, una estimación de ρ_total del ringdown (disponible de catálogos o del matched filter de la inspiral).
4. Configuración candidata: (t₀, T, f_low, f_high).

**Outputs del preflight (por modo y por configuración):**

- η(t₀, T): fracción de SNR capturada.
- ρ_eff: SNR efectiva en la ventana.
- Q × ρ_eff: producto de informatividad.
- rel_iqr_predicho: predicción de Cramér-Rao escalada.
- **veredicto_preflight**: VIABLE / MARGINAL / INVIABLE (según umbrales calibrados).

### 3.2 El mapa de viabilidad en el plano (t₀, T)

Para un evento y modo dados, la función η(t₀, T) = e^{−2t₀/τ}(1 − e^{−2T/τ}) define una **superficie de viabilidad** en el plano (t₀, T). Las curvas de nivel Q × ρ_total × √η = constante delimitan las regiones viables.

La curva de nivel correspondiente al umbral de informatividad es:

$$e^{-2t_0/\tau}\left(1 - e^{-2T/\tau}\right) = \left(\frac{0.61 \times \alpha_{\text{safety}}}{Q \times \rho_{\text{total}}}\right)^2$$

Esto define una frontera en el plano (t₀, T) por debajo de la cual toda configuración es inviable. Para GW150914 con ρ_total ≈ 8, Q = 4.3, α = 2:

$$\eta_{\min} = \left(\frac{1.22}{4.3 \times 8}\right)^2 = \left(\frac{1.22}{34.4}\right)^2 \approx 0.00126$$

Esto requiere t₀ < 3.35τ ≈ 18.3 ms (para T largo), o equivalentemente, **empezar el análisis antes de ~18 ms post-merger** para tener posibilidad de informatividad en el (2,2,0).

### 3.3 Restricciones adicionales: resolución espectral y contaminación

El preflight no debe evaluar solo la SNR. Hay tres restricciones que definen el **dominio admisible** en (t₀, T):

**A) Resolución espectral.** El estimador necesita distinguir el pico Lorentziano del ruido de fondo. La resolución en frecuencia del estimador espectral es Δf ≈ 1/T. Para que el pico sea resoluble:

$$\frac{1}{T} < \Gamma = \frac{1}{2\pi\tau} \quad \Longrightarrow \quad T > 2\pi\tau$$

Para el (2,2,0) de GW150914: T > 2π × 5.45 ≈ 34 ms. La ventana actual (60 ms) cumple esto holgadamente.

Para el (2,2,1): τ₂₂₁ ≈ 1.7 ms → T > 11 ms. También se cumple.

**B) Contaminación pre-ringdown.** Si t₀ es demasiado pequeño, la ventana incluye la fase de coalescencia (merger), cuya estructura espectral contamina la estimación del QNM. El t₀ mínimo seguro depende de la masa del remanente y no tiene expresión analítica limpia; es el dominio del t₀_sweep. En la literatura, el rango típico para el (2,2,0) es t₀ ∈ [0, 10] ms post-pico.

**C) Banda de frecuencia.** El ancho de banda del análisis debe contener la energía del modo. De la Fase 6, pregunta 4: la banda mínima es [f − 5Γ, f + 5Γ]. Para el (2,2,0): Γ ≈ 29 Hz, así que la banda mínima es [251 − 145, 251 + 145] = [106, 396] Hz.

El **dominio admisible** es la intersección de estas tres restricciones:

$$\mathcal{D} = \{(t_0, T) : T > 2\pi\tau \text{ (resolución)}\} \cap \{t_0 > t_{0,\min} \text{ (contaminación)}\} \cap \{\eta(t_0, T) > \eta_{\min} \text{ (informatividad)}\}$$

Si este dominio es vacío, el modo es **intrínsecamente no constrainable** para ese evento con esa SNR.

### 3.4 La tensión multi-modo y su resolución

El problema descubierto en el run actual es una instancia de un conflicto general: **el t₀ óptimo para un modo no es el t₀ óptimo para otro**.

- El (2,2,0) necesita t₀ pequeño (para capturar SNR antes de que se extinga).
- El (2,2,1) necesita t₀ algo mayor que el (2,2,0) (para evitar contaminación del merger), pero su τ es tan corto que se extingue rápido de todas formas.
- Un t₀ sweep que optimice un criterio mixto (o uno dominado por estabilidad numérica) puede converger a un t₀ que no es bueno para ninguno de los dos.

**Resolución propuesta:** el preflight debe evaluarse **por modo**, no globalmente. El pipeline necesita uno de estos dos enfoques:

**Enfoque A — Ventana compartida con t₀ por modo.** Se elige un t₀ óptimo por modo y se ejecutan estimaciones separadas. El t₀ sweep produce un t₀_best(2,2,0) y un t₀_best(2,2,1), y cada estimación se hace en su ventana óptima. Esto es lo que hace la literatura (Isi et al. 2019 usa t₀ = 0 para el análisis con overtone).

**Enfoque B — Criterio de viabilidad condicional.** Se ejecuta el análisis con un t₀ optimizado para el modo fundamental (que es el que el gate evalúa), y el overtone se trata como resultado condicional: se busca evidencia del (2,2,1) solo si el (2,2,0) ya es informativo. Esto alinea el sweep con el gate.

El enfoque B es más compatible con la arquitectura actual de BASURIN, porque el gate de viabilidad ya evalúa f₂₂₀ primero.

### 3.5 Conexión con el factor conforme Ω

El Método Brunete muestra que toda la información del detector entra por el factor conforme Ω (ec. 3.9):

$$\Omega(f, \tau) = \frac{A^2\tau}{2\pi}\,J(f, \tau)$$

donde J es el funcional central (ec. 2.6). Para PSD constante, Ω = ρ₀²/2.

En una ventana finita [t₀, t₀ + T], el factor conforme efectivo se modifica a:

$$\Omega_{\text{eff}}(t_0, T) = \frac{\rho_{\text{eff}}^2(t_0, T)}{2} = \frac{\rho_{\text{total}}^2}{2}\,\eta(t_0, T)$$

La condición de informatividad Q × ρ_eff > umbral se traduce directamente en una condición sobre Ω_eff:

$$\Omega_{\text{eff}} > \frac{\text{umbral}^2}{2\,Q^2}$$

Esto tiene una interpretación geométrica limpia: **la informatividad requiere que la densidad de información Fisher (medida por Ω) sea suficientemente grande en el punto del espacio de parámetros correspondiente al QNM**. Si Ω_eff es demasiado pequeño, la "resolución" del espacio de parámetros es demasiado baja para que el QNM sea distinguible del ruido.

Además, el criterio de contaminación χ_PSD del Teorema 1 (Método Brunete, ec. 7.14) proporciona una segunda condición: aunque Ω_eff sea suficiente, si χ_PSD ≥ 1, la curvatura del espacio de parámetros está dominada por la PSD y los resultados geométricos (s4/s6) no son fiables.

**El preflight completo evalúa entonces dos condiciones:**

1. **Informatividad:** Ω_eff > Ω_min (¿hay suficiente SNR para constreñir el modo?)
2. **Limpieza:** χ_PSD < χ_max (¿la PSD contamina la geometría?)

---

## 4. La estrategia de búsqueda automática

### 4.1 Algoritmo conceptual del preflight

Dado un evento con parámetros estimados (M_f, a_f) del remanente:

**Paso 1 — Tabular QNMs.** Usando kerr_qnm_fits.py, obtener (f, τ, Q) para los modos (2,2,0) y (2,2,1).

**Paso 2 — Estimar ρ_total.** Dos opciones:
- Desde catálogo (GWTC): usar la SNR de ringdown reportada o inferirla de la SNR total y la masa.
- Desde s1/s2: si ya se tiene el strain, estimar ρ con un matched filter rápido usando la forma de onda de ringdown en toda la ventana disponible.

**Paso 3 — Evaluar η(t₀, T) en una malla.** Para una malla gruesa en t₀ ∈ [0, 30] ms y T ∈ [20, 200] ms (los rangos físicamente razonables), calcular:
- ρ_eff(t₀, T) = ρ_total × √η
- Q × ρ_eff para el modo (2,2,0)
- rel_iqr_predicho = 0.304 × α_safety / (Q × ρ_eff)

Esto es aritmética pura, sin ningún estimador espectral. Se ejecuta en milisegundos.

**Paso 4 — Identificar la región viable.** Marcar las celdas (t₀, T) donde:
- rel_iqr_predicho < 0.5 (informatividad)
- T > 2πτ (resolución)
- t₀ > t₀_min (contaminación; usar un prior conservador, p.ej. t₀_min = 3 ms)

**Paso 5 — Seleccionar la configuración óptima.** Dentro de la región viable, elegir el (t₀, T) que maximiza la SNR efectiva (o minimiza rel_iqr_predicho), respetando las restricciones de contaminación.

**Paso 6 — Emitir diagnóstico.**
- Si la región viable es no vacía: reportar la configuración recomendada y los márgenes.
- Si es vacía: el evento es **intrínsecamente no informativo** para el modo (2,2,0) con la SNR disponible. Reportar la ρ_total mínima necesaria y/o el t₀_max para viabilidad.

### 4.2 Configuración óptima: solución analítica

Para T ≫ τ (ventana mucho más larga que el decaimiento), la fracción de SNR se simplifica a:

$$\eta(t_0, T \gg \tau) \approx e^{-2t_0/\tau}$$

y la condición de informatividad se reduce a:

$$t_0 < \frac{\tau}{2}\ln\left(\frac{Q^2 \rho_{\text{total}}^2}{(\text{umbral} \times \alpha)^2}\right)$$

$$\boxed{t_{0,\max} = \frac{\tau}{2}\ln\left(\frac{Q^2 \rho_{\text{total}}^2}{(\text{umbral} \times \alpha)^2}\right)}$$

**Ejemplo numérico para GW150914 (2,2,0):** con τ = 5.45 ms, Q = 4.3, ρ_total = 8, umbral = 0.61, α = 2:

$$t_{0,\max} = \frac{5.45}{2}\ln\left(\frac{4.3^2 \times 64}{1.49}\right) = 2.73 \times \ln(793) = 2.73 \times 6.68 \approx 18.2 \text{ ms}$$

**El análisis del (2,2,0) de GW150914 es informativo solo si t₀ < ~18 ms.** El t₀ sweep eligió 20 ms; el run usó 23 ms. Ambos están fuera del dominio de viabilidad.

Para T finito, hay una corrección logarítmica menor. La fórmula completa es:

$$t_{0,\max}(T) = \frac{\tau}{2}\ln\left(\frac{Q^2 \rho_{\text{total}}^2 \times (1 - e^{-2T/\tau})}{(\text{umbral} \times \alpha)^2}\right)$$

Para T = 60 ms y τ = 5.45 ms: (1 − e^{−22}) ≈ 1, así que la corrección es despreciable.

### 4.3 La tabla de viabilidad por evento (artefacto nuevo)

El preflight debe producir un artefacto `preflight_viability.json` que contenga, para cada evento y cada modo:

```
{
  "event_id": "GW150914",
  "mode": "(2,2,0)",
  "qnm_params": {"f_hz": 251, "tau_ms": 5.45, "Q": 4.3},
  "rho_total_estimated": 8.0,
  "t0_max_ms": 18.2,
  "T_min_ms": 34.2,
  "band_min_hz": [106, 396],
  "eta_at_current_config": 0.00022,
  "rho_eff_at_current_config": 0.12,
  "rel_iqr_predicted": 1.45,
  "viability": "INVIABLE",
  "recommended_config": {"t0_ms": 5.0, "T_ms": 60, "band_hz": [120, 380]},
  "recommended_rho_eff": 3.8,
  "recommended_rel_iqr_predicted": 0.19,
  "alpha_safety": 2.0
}
```

### 4.4 Calibración del factor α_safety

El factor α_safety absorbe la diferencia entre la cota de Cramér-Rao (idealizada) y el rendimiento real del estimador bootstrap. Para calibrarlo:

1. Tomar los runs existentes donde `rel_iqr_f220` es conocido.
2. Calcular ρ_eff para cada uno usando la fórmula analítica.
3. Calcular el ratio observado: α_obs = (rel_iqr_observado × Q × ρ_eff) / 0.304.
4. Tomar α_safety = percentil 90 de los α_obs observados.

Esto solo necesita unos pocos runs (incluso los fallidos sirven para calibración). Una vez calibrado, α_safety es una constante del pipeline (hasta que cambie el método de estimación).

Si no hay suficientes runs para calibrar empíricamente, α_safety = 2.5 es un valor conservador razonable basado en la experiencia general con estimadores espectrales bootstrap de señales de SNR moderada.

---

## 5. Implicaciones para el diseño del pipeline

### 5.1 El t₀ sweep debe optimizar con restricción de informatividad

El diseño actual del t₀ sweep tiene un problema conceptual: optimiza algún criterio (estabilidad, plateau, etc.) sin verificar que el t₀ resultante esté dentro del dominio de viabilidad Fisher del modo que el gate va a evaluar.

**Propuesta:** el t₀ sweep incorpora el preflight como restricción dura. El sweep solo explora t₀ ∈ [t₀_min_contaminación, t₀_max_informatividad]. Si este intervalo es vacío, el sweep se declara inviable antes de ejecutar subruns.

Para GW150914: t₀ ∈ [3, 18] ms en vez de [0, 30] ms. Esto reduce el espacio de búsqueda y garantiza que cualquier t₀ seleccionado producirá un run con posibilidad de ser informativo.

### 5.2 El gate de viabilidad debe reportar la predicción Fisher

Cuando `s4d` rechaza por `RINGDOWN_NONINFORMATIVE`, el artefacto debería incluir la predicción Fisher como contexto diagnóstico:

- ρ_eff estimado
- Q × ρ_eff
- t₀_max para viabilidad
- η en la configuración actual

Esto transforma el rechazo de "no sabemos por qué no funciona" a "la SNR efectiva en la ventana es 0.12, necesitaríamos ≥ 0.14 según Cramér-Rao, y para eso t₀ debería ser < 18 ms".

### 5.3 Modo fundamental primero, overtone condicional

La estructura lógica óptima para multimode es:

1. **Preflight Fisher** para el (2,2,0): ¿es viable?
   - NO → el evento no soporta análisis de ringdown informativo con esta SNR. Marcar como `RINGDOWN_INSUFFICIENT_SNR`. No gastar cómputo.
   - SÍ → seleccionar t₀ y T óptimos para el (2,2,0).

2. **Ejecutar s3/s3b** con la configuración óptima para el (2,2,0).

3. **Evaluar informatividad real** del (2,2,0) (el gate actual: rel_iqr < 0.5).

4. **Solo si el (2,2,0) es informativo:** evaluar si el (2,2,1) tiene suficiente evidencia (delta_BIC, etc.).

Esto es compatible con la filosofía actual de BASURIN (determinismo, censura explícita, fail-fast) pero alinea el flujo de decisión con la física.

### 5.4 La tabla de viabilidad como artefacto de catálogo

Para el pipeline multi-evento (ex4, exclusion map, etc.), el preflight produce una **tabla de viabilidad del catálogo**:

| Evento | ρ_total | t₀_max (2,2,0) | viabilidad (2,2,0) | t₀_max (2,2,1) | viabilidad (2,2,1) |
|--------|---------|-----------------|---------------------|-----------------|---------------------|
| GW150914 | ~8 | 18 ms | VIABLE (t₀<18) | ~3 ms | MARGINAL |
| GW151226 | ~4 | 8 ms | MARGINAL | — | INVIABLE |
| ... | | | | | |

Esta tabla se calcula **una vez** para todo el catálogo (aritmética pura, sin runs), y dicta:
- Qué eventos vale la pena analizar.
- Qué configuración usar para cada uno.
- Qué modos son constraiñables para cada evento.

---

## 6. La corrección por PSD: de ρ_total a ρ_eff(S_n)

### 6.1 Cuando la PSD no es constante

La fórmula de la sección 1.2 asume S_n constante sobre el ancho del Lorentziano. Si la PSD tiene estructura (líneas, pendiente), la SNR efectiva se modifica. Del Método Brunete (ec. 6.10):

$$J(f, \tau) = \frac{\pi}{2S_n(f)}\left[1 + \frac{\beta_{[2]}}{4Q^2} + O(1/Q^4)\right]$$

donde β_{[2]} = s₁²/2 − s₂, con s₁ y s₂ las derivadas logarítmicas de la PSD.

Para LIGO en la banda del ringdown (~200-300 Hz, shot noise, α ~ −4 a −6): β_{[2]} es de orden unidad y la corrección es ~1/(4Q²) ≈ 1% para Q = 4.3. **Despreciable para el preflight.** La PSD constante es una excelente aproximación lejos de líneas instrumentales.

### 6.2 Cuando el QNM cae cerca de una línea

Si f_QNM está dentro de ~5Γ de un violin mode u otra línea de la PSD, la SNR efectiva puede caer drásticamente. El parámetro σ del Método Brunete (§6.5) entra en régimen "colapsado" (σ ≫ 1) y el funcional J tiende a cero.

El preflight debe incluir un check de proximidad a líneas conocidas:

- Si min|f_QNM − f_línea| < 5Γ: emitir warning `PSD_LINE_PROXIMITY`.
- Si min|f_QNM − f_línea| < 2Γ: la SNR predicha no es fiable; marcar como `PSD_CONTAMINATED`.

Las frecuencias de las líneas principales de LIGO son conocidas y estables (60 Hz armónicos, violin modes ~500 Hz, calibration lines). Para el (2,2,0) de GW150914 a 251 Hz, no hay líneas problemáticas.

---

## 7. Límites de la predicción analítica

### 7.1 Lo que el preflight NO predice

El preflight Fisher es una herramienta de **triage**: descarta configuraciones imposibles y sugiere regiones viables. No sustituye al pipeline completo porque:

1. **No modela la contaminación del merger.** La transición inspiral → ringdown no es limpia; el t₀_min seguro depende de la masa, del ratio de masas, y de efectos numéricos que solo se ven en los datos. El preflight pone una restricción de SNR, pero el t₀ sweep sigue siendo necesario dentro de la región viable para optimizar la separación merger/ringdown.

2. **No modela no-gaussianidad del ruido.** Glitches, non-stationarities, y colas pesadas del ruido degradan al estimador más allá de lo que predice Cramér-Rao. El factor α_safety absorbe parte de esto estadísticamente, pero eventos individuales pueden ser peores.

3. **No modela correlaciones entre modos.** En el análisis multimodo, los modos (2,2,0) y (2,2,1) no son perfectamente ortogonales (Fase 6, pregunta 2). La covarianza entre ellos depende de la PSD medida y afecta al IQR del posterior conjunto de formas no triviales.

4. **La SNR de ringdown ρ_total es en sí misma una estimación.** Si viene de catálogo, tiene su propia incertidumbre. Si viene de matched filter, depende del template usado.

### 7.2 Lo que el preflight SÍ garantiza

Si el preflight dice **INVIABLE**, es una certeza matemática (módulo la estimación de ρ_total): no existe ninguna realización del ruido que haga el run informativo, porque la señal simplemente no está ahí en la ventana.

Si dice **VIABLE**, es una condición necesaria satisfecha, no una garantía de éxito. Pero restringe la búsqueda a una región donde el éxito es posible.

---

## 8. Resumen ejecutivo de la metodología

**Diagnóstico raíz:** el run de GW150914 falla porque comienza el análisis 4.2 tiempos de decaimiento después del pico del ringdown. Solo queda 0.02% de la SNR² del modo fundamental.

**Marco teórico:** la cota de Cramér-Rao del Método Brunete predice analíticamente la precisión alcanzable en frecuencia como función de (Q, ρ_eff, t₀, T). La condición de informatividad Q × ρ_eff > 0.61 × α_safety es evaluable sin ejecutar el pipeline.

**Metodología propuesta:**

1. **Preflight Fisher** como Stage 0.5: evalúa viabilidad analítica por modo y por configuración.
2. **t₀_max analítico**: fórmula cerrada para el t₀ máximo que permite informatividad, dado el modo y la SNR.
3. **Restricción del t₀ sweep**: el sweep solo explora la región viable del preflight.
4. **Tabla de viabilidad del catálogo**: artefacto que clasifica todos los eventos antes de gastar runs.
5. **Calibración empírica de α_safety**: un solo número que conecta la teoría (Cramér-Rao) con la práctica (bootstrap).

**Para GW150914 específicamente:** el análisis del (2,2,0) requiere t₀ < ~18 ms. Cualquier configuración con t₀ > 18 ms es algebraicamente incapaz de producir un run informativo para el modo fundamental.

---

## Apéndice A. Fórmulas de referencia rápida

| Cantidad | Fórmula | Referencia |
|----------|---------|------------|
| τ del QNM | τ = Q/(πf) | Método Brunete (1.5) |
| Ancho Lorentziano | Γ = 1/(2πτ) = f/(2Q) | Método Brunete (1.1) |
| η(t₀, T) | e^{−2t₀/τ}(1 − e^{−2T/τ}) | §1.2 |
| ρ_eff | ρ_total × √η | §1.2 |
| σ_f/f (Cramér-Rao) | 1/(πQρ√2) | Método Brunete (2.12) |
| rel_iqr predicho | 0.304 × α/(Q × ρ_eff) | §2.1–2.2 |
| t₀_max | (τ/2)ln(Q²ρ²_total/(umbral×α)²) | §4.2 |
| T_min (resolución) | 2πτ | §3.3A |
| Banda mínima | [f − 5Γ, f + 5Γ] | Fase 6, preg. 4 |
| Ω_eff | ρ²_eff/2 | §3.5 |

## Apéndice B. Valores numéricos para GW150914

| Modo | f (Hz) | Q | τ (ms) | Γ (Hz) | t₀_max (ms) | T_min (ms) | Banda (Hz) |
|------|--------|---|--------|--------|-------------|-----------|------------|
| (2,2,0) | 251 | 4.3 | 5.45 | 29.2 | ~18 | 34 | [105, 397] |
| (2,2,1) | ~280 | ~2 | ~2.3 | ~70 | ~5 | 14 | [130, 430] |

**Nota:** los valores del (2,2,1) son aproximados y dependen del spin. El t₀_max del (2,2,1) es ~5 ms, lo que significa que para SNR típicas de O3, el overtone es constraiñable solo en las mejores condiciones (t₀ < 5 ms, SNR alta).

## Apéndice C. Relación con la Fase 6

Esta metodología es complementaria a la Fase 6 (PSD medida). El preflight usa la PSD (analítica o medida) como input, pero su contribución principal es la predicción de informatividad vía Fisher, que es anterior y más fundamental que la elección de PSD. La PSD medida mejora la precisión de la predicción (corrección del factor conforme Ω), pero el efecto dominante es la escala temporal t₀/τ, que es independiente de la PSD.

---

**Historial de revisiones**

| Versión | Fecha | Cambios |
|---------|-------|---------|
| 1.0 | 2026-03-06 | Documento inicial. Respuesta al fallo de informatividad en multimodo GW150914. |
