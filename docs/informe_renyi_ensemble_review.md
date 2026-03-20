# Revisión teórica: diversidad de Rényi sobre el ensemble discreto de geometrías post-ringdown en BASURIN

**Fecha**: 2026-03-13  
**Objeto**: Evaluación crítica de la base teórica para el uso de métricas de diversidad de Rényi sobre un ensemble de 106 geometrías compatibles producido por el pipeline BASURIN, con tres políticas de pesos materializadas.  
**Alcance**: Fundamentos matemáticos, estatus de las políticas, interpretación de resultados, y recomendaciones operativas. No se cubre holografía, entropía termodinámica de agujeros negros, ni analogías con Rényi en teoría cuántica de campos.

---

## 1. Veredicto ejecutivo

### ¿La línea actual tiene base teórica suficiente para seguir?

**Sí, con matices importantes.**

La línea es teóricamente sólida como herramienta **descriptiva y diagnóstica** del ensemble. Los números de Hill / diversidad de Rényi son la herramienta canónica para cuantificar concentración de distribuciones discretas, y su aplicación a un ensemble de modelos/hipótesis compatibles es matemáticamente impecable. No se necesita ninguna justificación adicional para usarlos como lo que son: medidas de concentración de una distribución discreta.

### ¿Qué parte está realmente bien fundada?

- Tratar las 106 geometrías como un espacio discreto de hipótesis compatibles y asignarles pesos normalizados.
- Calcular D_alpha (números de Hill) sobre esas distribuciones para cuantificar concentración efectiva.
- Comparar los perfiles D_0, D_1, D_2, D_inf entre políticas como diagnóstico de sensibilidad.
- Afirmar que la contracción D_0 >> D_1 >> D_2 >> D_inf indica concentración epistémica real **dada la política**.

### ¿Qué parte debe formularse con más cautela?

- **No puede afirmarse que la contracción refleja una propiedad física del remanente** sin controles de estabilidad (bootstrap, jackknife, sensibilidad a la temperatura del softmax).
- **La política delta_lnL-softmax no es un posterior bayesiano** y no debe reclamarse como tal. Es un *score de compatibilidad normalizado* que hereda todas las limitaciones del pipeline upstream.
- **La ausencia de Kerr en el soporte (kerr=0) es un resultado del pipeline**, no una exclusión teórica — y su interpretación requiere verificar que no sea un artefacto de la discretización del atlas o de los umbrales de filtrado.
- **Dinf ~ 8.5 bajo la política delta_lnL no significa "solo 8 geometrías son reales"** — significa que, bajo esa política específica de pesos, la masa epistémica está concentrada hasta el punto de que el peso máximo individual es ~11.7%.

---

## 2. Naturaleza matemática del ensemble

### 2.1 Qué objeto estamos modelando

El objeto es un **espacio discreto finito de hipótesis** (o modelos). Sea G = {g_1, ..., g_N} con N = 106 el conjunto de geometrías que sobrevivieron al filtrado del pipeline (s4 + filtros geométricos). Cada política de pesos define una función de masa de probabilidad p: G → [0,1] con Σ p(g_i) = 1.

Esto **no** es:

- La entropía termodinámica del agujero negro remanente (Bekenstein-Hawking o generalizaciones).
- Una distribución sobre microestados cuánticos del horizonte.
- Una medida en un espacio de métricas continuo.
- Un posterior bayesiano completo sobre el espacio de teorías de gravedad.

Es, estrictamente:

- Una distribución discreta sobre un catálogo finito de soluciones geométricas parametrizadas que han pasado filtros de compatibilidad observacional.
- La variable aleatoria es "qué entrada del atlas es compatible con los datos, ponderada por la política elegida".

### 2.2 Marcos teóricos que lo respaldan

**2.2.1 Bayesian model uncertainty sobre espacios discretos.**

El marco más directo es la selección bayesiana de modelos sobre un espacio finito de hipótesis. Dado un conjunto de modelos {M_1, ..., M_K}, la posterior sobre modelos es p(M_k | datos) ∝ p(datos | M_k) · p(M_k). En ese marco, medir la concentración de la posterior sobre modelos es una operación estándar.

La diferencia crucial con lo que hace BASURIN es que las políticas de pesos **no** son posteriors bayesianas formales. No hay un likelihood p(datos | geometría) bien definido en el sentido de modelo generativo completo, ni un prior explícito sobre el atlas. Lo que hay es un **score de compatibilidad** (delta_lnL) que se normaliza vía softmax. Esto es más cercano a un *approximate posterior* o a un *scoring function* que a una inferencia bayesiana completa.

Referencia directa: Hoeting, Madigan, Raftery & Volinsky (1999), "Bayesian Model Averaging: A Tutorial", Statistical Science 14(4), 382-417. Define el marco de promediar sobre modelos discretos con pesos proporcionales a evidencia bayesiana. BASURIN se sitúa en una versión más ligera de este marco.

**2.2.2 Números de Hill y diversidad verdadera.**

Los números de Hill (Hill, 1973, "Diversity and Evenness: A Unifying Notation and Its Consequences", Ecology 54(2), 427-432) definen, para una distribución discreta con pesos {p_i}:

    D_alpha = (Σ p_i^alpha)^(1/(1-alpha))    para alpha ≥ 0, alpha ≠ 1
    D_1 = exp(-Σ p_i ln p_i)                  (límite alpha → 1, exponencial de Shannon)

Estos números tienen la propiedad de ser el **"número efectivo de especies/tipos"** y satisfacen el principio de replicación: si se unen q comunidades igualmente diversas sin solapamiento, la diversidad se multiplica por q. Esto los convierte en medidas de diversidad *verdadera* (no solo índices).

Jost (2006), "Entropy and diversity", Oikos 113(2), 363-375, clarificó que los índices de Shannon y Simpson son monótonamente transformables a números de Hill pero no son ellos mismos medidas de diversidad verdadera (no satisfacen replicación). Los D_alpha sí.

La aplicación a espacios de hipótesis/modelos (en lugar de especies biológicas) es directa y no requiere justificación adicional: es una propiedad de distribuciones discretas, no del dominio de aplicación.

**2.2.3 Effective Sample Size (ESS) y perplejidad.**

D_2 = 1/Σp_i^2 es exactamente el inverso del índice de Simpson, y también es la fórmula de Kish para el tamaño de muestra efectivo en importance sampling (Kish, 1965). D_1 = exp(H_Shannon) es la perplejidad, estándar en teoría de la información y modelado de lenguaje. Ambas tienen interpretaciones operacionales bien establecidas fuera de ecología.

**2.2.4 Information geometry — tangencial pero no central.**

La geometría de información (Fisher, Amari) opera sobre familias parametrizadas continuas. El atlas de BASURIN es discreto. Puede definirse una geometría discreta (simplex de probabilidades con métrica de Fisher), pero no aporta mucho más que los D_alpha para el problema actual. Mencionable como contexto, no como fundamento.

### 2.3 Terminología recomendada

| Término | ¿Usar? | Razón |
|---------|--------|-------|
| "Rényi entropy of the ensemble" | Con cautela | Correcto matemáticamente, pero puede confundirse con entropía termodinámica del BH. Añadir siempre "del ensemble de geometrías compatibles". |
| "Rényi diversity" / "Hill numbers" | **Sí, preferido** | Es la interpretación más directa y evita confusión con entropía termodinámica. |
| "Effective number of compatible geometries" | **Sí, preferido** | Operacionalmente preciso. |
| "Epistemic diversity of the compatible ensemble" | Aceptable | Si se quiere marcar que es diversidad epistémica, no ontológica. |
| "Spectral entropy" | **No** | Confusión con el espectro de QNMs y con entropía espectral en otros contextos. |

**Recomendación terminológica para el paper/documentación**: usar "effective number of compatible geometries" como medida primaria, citando explícitamente a Hill (1973) y Jost (2006). Reservar "Rényi diversity profile" para el perfil completo {D_alpha : alpha ∈ grid}.

---

## 3. Evaluación de las tres políticas de pesos

### 3.1 uniform_support_v1

**Definición**: p(g_i) = 1/N = 1/106 para toda g_i en el soporte.

**Estatus teórico**: Es el prior de máxima ignorancia (máxima entropía) sobre el soporte. Formalmente, es la distribución que maximiza la entropía de Shannon sujeta a la restricción de soporte = G. Es el **null model** natural.

**Interpretación legítima**:
- Baseline de referencia: cuánta diversidad hay si tratamos todas las geometrías compatibles como igualmente plausibles.
- D_alpha = N = 106 para todo alpha — esto es correcto y esperable.

**Interpretación ilegítima**:
- "Todas las geometrías son igualmente probables dado los datos". No — la uniformidad es una elección de referencia, no un resultado.

**Riesgos**: Ninguno serio. Es lo que dice ser.

### 3.2 event_frequency_support_v1

**Definición**: p(g_i) ∝ |{eventos donde g_i aparece en el soporte final}|.

**Estatus teórico**: Es una medida de **recurrencia empírica**. Es análoga a una "frecuencia de aparición" o a un conteo de votos. En el marco bayesiano discreto, si cada evento proporciona un voto binario (compatible/no compatible), esto equivale a un peso proporcional al número de "éxitos" bajo un modelo binomial implícito con prior uniforme sobre la probabilidad de compatibilidad.

Es más robusto que la política uniforme porque incorpora información observacional (qué geometrías son recurrentemente compatibles), pero es una medida **cruda**: trata todos los eventos por igual y no distingue entre "apenas compatible" y "altamente compatible".

**Interpretación legítima**:
- Medida de robustez/recurrencia: las geometrías con peso alto aparecen en el soporte de muchos eventos.
- D_1 ~ 82.3 indica que, en términos de recurrencia, el ensemble efectivo es de ~82 geometrías — hay asimetría pero moderada.

**Interpretación ilegítima**:
- "Posterior marginal sobre geometrías". No — para ser un posterior marginal necesitaría un modelo generativo con likelihood por evento y prior sobre geometrías. Esto es un conteo.
- "Las geometrías con frecuencia alta son más probablemente correctas". No con este peso — solo son más frecuentemente compatibles bajo los umbrales del pipeline.

**Riesgos**:
- Si el pipeline tiene un sesgo sistemático (p.ej., ciertas familias pasan más fácilmente el filtro por tener más parámetros libres), la frecuencia amplifica ese sesgo.
- La granularidad del atlas importa: si edgb tiene 39 entradas y kerr tiene pocas, la familia con más entradas tiene más "oportunidades" de recurrencia.

### 3.3 event_support_delta_lnL_softmax_mean_v1

**Definición**: Para cada evento e con soporte S_e:
1. Se toma delta_lnL(g_i, e) para cada g_i ∈ S_e.
2. Se calcula w_i^(e) = softmax(delta_lnL(g_i, e)) = exp(delta_lnL(g_i, e)) / Σ_{j∈S_e} exp(delta_lnL(g_j, e)).
3. Se promedia sobre eventos: w_i = (1/|E|) Σ_e w_i^(e) · 1_{g_i ∈ S_e}.
4. Se normaliza: p(g_i) = w_i / Σ_j w_j.

**Estatus teórico**: Esto es un **score de compatibilidad normalizado con promediado inter-evento**. Es la política más informativa de las tres porque incorpora la magnitud de la compatibilidad, no solo su presencia/ausencia.

Sin embargo, hay varios supuestos implícitos que deben explicitarse:

**(a) La temperatura del softmax es 1.** Esto es una elección — no hay razón teórica a priori para que la "temperatura correcta" sea 1. Con temperatura T → 0, el softmax colapsa a argmax; con T → ∞, se acerca a uniforme. La temperatura T=1 implica que las unidades de delta_lnL se toman al pie de la letra como log-likelihood differences. Si delta_lnL no es una log-likelihood calibrada (que no lo es — es un score del pipeline), la temperatura efectiva es arbitraria.

**(b) El promedio sobre eventos pondera todos los eventos por igual.** Esto ignora diferencias en SNR, calidad de datos, o informatitividad del ringdown entre eventos. Eventos con ringdown no informativo contribuyen igual que GW250114.

**(c) El soporte S_e puede variar entre eventos.** Geometrías que aparecen en pocos eventos pero con delta_lnL alto pueden quedar diluidas. Geometrías que aparecen en muchos eventos con delta_lnL moderado pueden acumular peso.

**Interpretación legítima**:
- Score de compatibilidad observacional normalizado: "bajo esta política, las geometrías con delta_lnL consistentemente alto a través de múltiples eventos concentran la masa del ensemble".
- Diagnóstico de concentración: la contracción de D_1 a ~28 y D_inf a ~8.5 indica que, bajo el scoring delta_lnL, el ensemble se comporta como si tuviera ~28 miembros típicos y un "núcleo duro" de ~8-9 geometrías dominantes.

**Interpretación ilegítima**:
- **"Posterior bayesiana global sobre geometrías"**: NO. Para ser un posterior necesitaría (i) un likelihood p(datos | geometría) bien definido, (ii) un prior p(geometría), y (iii) marginalización sobre parámetros nuisance. delta_lnL no es nada de esto.
- **"Probabilidad física de que el remanente sea la geometría g_i"**: NO. Ni siquiera conceptualmente — el remanente es una única geometría, y el ensemble es una medida de compatibilidad del pipeline, no una distribución de probabilidad sobre el estado del remanente.
- **"Evidencia bayesiana relativa entre geometrías"**: NO. La evidencia bayesiana (model evidence) requiere marginalizar el likelihood sobre todos los parámetros del modelo. delta_lnL es un score puntual/filtrado.
- **"Likelihood ratio entre geometrías"**: Con cautela extrema. delta_lnL se parece a un log-likelihood ratio, pero depende del pipeline upstream y de las decisiones de filtrado.

**Formulación segura recomendada**: "normalized pipeline compatibility score" o "score-weighted epistemic distribution". Si se quiere una etiqueta más compacta: **"pseudo-posterior de compatibilidad"**, dejando claro que "pseudo" significa exactamente que no es un posterior formal.

**Riesgos metodológicos**:
- **Sensibilidad a la temperatura**: sin explorar T ≠ 1, no se sabe si la concentración observada es robusta o es un artefacto del escalado.
- **Doble conteo**: si delta_lnL ya incorpora el filtrado de soporte, el softmax puede estar amplificando una señal que ya fue seleccionada.
- **Granularidad del atlas**: familias con más entradas tienen más oportunidades de capturar peso.

---

## 4. Interpretación de los resultados Rényi

### 4.1 Significado de cada D_alpha en este problema

**D_0 = |soporte| = 106.** El número de geometrías con peso no nulo. Mide la amplitud bruta del soporte. Es invariante a la política (todas comparten soporte). Interpretación: "hay 106 geometrías que pasan los filtros del pipeline".

**D_1 = exp(H_Shannon) = perplejidad.** El número efectivo de geometrías "típicas". Operacionalmente: si se muestreara una geometría al azar según la política, D_1 es el tamaño del subconjunto al que equivaldría una distribución uniforme con la misma entropía. Interpretación: "bajo esta política, el ensemble se comporta como si tuviera D_1 miembros igualmente probables".

**D_2 = 1/Σp_i^2 = inverso de Simpson.** El número efectivo ponderado por la concentración cuadrática. Es el ESS de Kish. Es más sensible a los pesos altos que D_1. Interpretación: "si se escogieran dos geometrías al azar, D_2 es el inverso de la probabilidad de que sean la misma".

**D_inf = 1/p_max.** El inverso del peso máximo. Mide cuánto domina la geometría más pesada. Interpretación: "si una sola geometría tiene el 11.7% del peso (p_max = 0.117), entonces D_inf = 8.5 — el ensemble se comporta como si tuviera solo 8-9 opciones plausibles desde la perspectiva de la más dominante".

### 4.2 Lectura de la contracción observada

**Hecho central**: D_0 = 106 es constante entre políticas, pero D_1, D_2, D_inf decrecen monotónicamente al pasar de uniforme → frecuencia → delta_lnL. Esto es la firma clásica de **concentración epistémica progresiva**: a medida que la política incorpora más información observacional, la masa se redistribuye de uniforme hacia un subconjunto.

**Política B (frecuencia)**: D_1 ~ 82.3, D_2 ~ 73.5, D_inf ~ 52.0. Contracción moderada. La recurrencia por eventos ya distingue geometrías, pero no dramáticamente. La ratio D_inf/D_0 ~ 0.49 indica que la geometría más recurrente concentra ~2% del peso — concentración suave.

**Política C (delta_lnL)**: D_1 ~ 28.4, D_2 ~ 19.0, D_inf ~ 8.5. Contracción fuerte. La cascada D_0/D_1 ~ 3.7, D_1/D_2 ~ 1.5, D_2/D_inf ~ 2.2 revela una distribución con cola pesada: hay un núcleo de ~8-9 geometrías con peso alto, un segundo tier de ~10-20 con peso moderado, y una cola larga de ~70-80 con peso residual.

### 4.3 Qué permiten afirmar ya los resultados

1. **La concentración es real dada la política.** No hay error matemático ni artefacto de cálculo. Si la política delta_lnL es una representación razonable de la compatibilidad observacional, entonces efectivamente ~28 geometrías capturan la mayor parte de la masa epistémica, y ~8-9 concentran el núcleo.

2. **El perfil de diversidad es un diagnóstico informativo.** La diferencia entre políticas es sustancial y no trivial. Esto en sí mismo es un resultado: muestra que la información observacional (delta_lnL) discrimina fuertemente dentro del soporte.

3. **La composición ontológica del soporte (edgb:39, kerr_newman:32, dcs:35, kerr:0) es un resultado del pipeline factible de reportar**, independientemente de la diversidad de Rényi.

### 4.4 Qué NO permiten afirmar todavía

1. **No puede afirmarse que "el remanente es probablemente una de las 8-9 geometrías dominantes"** — eso requeriría que la política sea un posterior calibrado, y no lo es.

2. **No puede afirmarse que la contracción es "robusta" sin controles de estabilidad.** La concentración puede ser sensible a: (a) qué eventos se incluyen (un evento atípico con delta_lnL extremo puede dominar), (b) la temperatura del softmax, (c) la resolución del atlas.

3. **No puede afirmarse exclusión de Kerr como resultado de la diversidad.** kerr=0 en el soporte es un resultado upstream del pipeline (filtrado geométrico), no del análisis de diversidad. La diversidad mide concentración dentro del soporte, no la composición del soporte.

4. **No puede usarse D_inf ~ 8.5 como argumento para truncar el atlas a las "top 8"** — D_inf es un diagnóstico, no una prescripción de corte.

---

## 5. Recomendación para BASURIN

### 5.1 Próximos pasos teóricamente seguros

**(a) Análisis de estabilidad por jackknife de eventos.** Para cada evento e_k, recalcular las tres políticas y los D_alpha excluyendo e_k. Esto produce un perfil de influencia: ¿hay eventos cuya exclusión cambia D_1 o D_inf en más de (por ejemplo) un 20%? Si sí, el resultado depende críticamente de esos eventos y debe reportarse. Este es el control más urgente.

**(b) Sensibilidad a la temperatura del softmax.** Recalcular la política C con temperaturas T ∈ {0.5, 1.0, 2.0, 5.0, ∞}. Si D_1 varía poco en ese rango, la concentración es robusta al escalado. Si varía mucho, T=1 es una elección arbitraria que debe reconocerse explícitamente.

**(c) Descomposición por familia/teoría.** Calcular la masa total por familia bajo cada política: M_family = Σ_{g_i ∈ family} p(g_i). Esto responde a "¿qué fracción de la masa epistémica va a edgb, kerr_newman, dcs?" — pregunta científicamente más interpretable que los D_alpha solos.

**(d) Comparación top-k.** Para k ∈ {5, 10, 20, 50}, calcular la masa acumulada de las top-k geometrías bajo cada política. Esto da una curva de concentración tipo Lorenz que es visualmente informativa y fácil de comunicar.

**(e) Null model por permutación.** Permutar aleatoriamente las asignaciones delta_lnL entre geometrías dentro de cada evento (manteniendo la estructura del soporte) y recalcular D_alpha. Esto da una distribución nula de D_alpha bajo la hipótesis de que delta_lnL no discrimina. Si los D_alpha observados caen fuera del rango nulo, la concentración es estadísticamente significativa más allá de la estructura del soporte.

### 5.2 Controles obligatorios antes de cualquier claim fuerte

En orden de prioridad:

1. **Jackknife por eventos** — sin esto, no se puede saber si la concentración depende de un par de eventos de alto SNR.
2. **Sensibilidad a temperatura** — sin esto, T=1 es una elección no justificada.
3. **Null model por permutación** — sin esto, la concentración podría ser un artefacto de la estructura del soporte (si muchas geometrías comparten casi el mismo delta_lnL, el softmax puede crear concentración espuria).

### 5.3 Pasos que considero prematuros

- **Interpretación fenomenológica de "las top-8 geometrías"**: identificar las geometrías dominantes está bien como diagnóstico, pero atribuirles significado físico ("el remanente es probablemente edgb con estos parámetros") requiere un pipeline con likelihood calibrado y controles de cobertura, que hoy no existe.

- **Publicar claims sobre exclusión de familias teóricas basándose en la distribución de masa por familia bajo la política delta_lnL**: la política no es un posterior, y la composición del atlas (cuántas entradas por familia) confunde la señal.

- **Construir un "ranking de teorías de gravedad"** a partir de D_alpha: los D_alpha miden concentración del ensemble, no calidad relativa de las teorías. Para ranking de teorías se necesita model evidence (Bayes factors), que es un problema cualitativamente distinto.

- **Combinar los D_alpha de BASURIN con entropía termodinámica del agujero negro**: son objetos completamente distintos. D_alpha mide concentración de una distribución epistémica sobre un catálogo finito de hipótesis. S_BH mide entropía termodinámica de un sistema gravitacional. Compartir la etiqueta "Rényi" no implica conexión física.

### 5.4 Formulación rigurosa recomendada

Para documentación interna y eventual publicación, sugiero la siguiente formulación canónica:

> "Dado el ensemble de N geometrías compatibles producido por el pipeline de filtrado, definimos un perfil de diversidad efectiva {D_alpha} sobre tres políticas de pesos: uniforme (referencia), recurrencia por eventos, y score de compatibilidad normalizado (softmax sobre delta_lnL). Los números de Hill D_alpha (Hill 1973; Jost 2006) cuantifican el número efectivo de geometrías compatibles bajo cada política. La contracción progresiva de D_alpha al incorporar información observacional indica que el ensemble, aunque formalmente de cardinalidad N, se comporta efectivamente como un conjunto de D_1 ~ K geometrías típicas bajo la política de scoring, con un núcleo dominante de D_inf ~ k geometrías."

Esta formulación:
- No reclama posterior bayesiano.
- No atribuye significado físico a la concentración.
- Usa terminología estándar (Hill, Jost).
- Es verificable y auditable.
- Deja espacio para que los controles posteriores fortalezcan o debiliten el claim.

### 5.5 ¿En qué punto sería legítimo pasar a interpretación física?

La transición de diagnóstico descriptivo a interpretación física requiere, como mínimo:

1. **Calibración del score**: demostrar que delta_lnL se comporta como un log-likelihood ratio en algún sentido definido (p.ej., bajo inyecciones sintéticas con geometría conocida, ¿la política delta_lnL concentra peso sobre la geometría correcta?).

2. **Cobertura**: demostrar que la región de soporte (las N geometrías) tiene cobertura frecuentista aceptable (p.ej., en el X% de las inyecciones sintéticas, la geometría verdadera está en el soporte con peso no despreciable).

3. **Control de la granularidad del atlas**: demostrar que los resultados no cambian cualitativamente si se varía la resolución del atlas (más/menos entradas por familia).

Sin estos tres controles, la interpretación debe permanecer en el plano epistémico-descriptivo: "dado nuestro pipeline y este atlas, así se distribuye la compatibilidad".

---

## Apéndice A: Por qué esto NO es entropía de Rényi del agujero negro

La entropía de Rényi en el contexto de agujeros negros (Dong 2016; Belin et al. 2013; y la línea holográfica asociada) se refiere a la entropía de entrelazamiento de Rényi de regiones en teorías con dual gravitacional, o a generalizaciones de la fórmula de Bekenstein-Hawking. El objeto sobre el que se calcula es el estado cuántico del sistema gravitacional (o su dual CFT), y los grados de libertad son los microestados del horizonte.

En BASURIN, el objeto es un **catálogo finito de soluciones clásicas** que han pasado filtros de compatibilidad. No hay microestados, no hay entrelazamiento, no hay horizonte como sistema termodinámico. La coincidencia léxica ("Rényi" + "agujero negro") es exactamente eso: una coincidencia. Invocar la línea holográfica para justificar el uso de diversidad de Rényi en BASURIN sería una falsa pista.

Lo que justifica el uso de diversidad de Rényi en BASURIN es la **teoría de la información y la ecología cuantitativa** (Hill, Jost, Rényi 1961), no la física de agujeros negros.

---

## Apéndice B: Tabla resumen de diagnósticos y su estatus

| Diagnóstico | Base teórica | Estatus en BASURIN | Interpretación segura | Interpretación peligrosa |
|---|---|---|---|---|
| D_0 = N | Directa (cardinalidad) | Materializado | "106 geometrías pasan el filtro" | "Hay 106 geometrías posibles para el remanente" |
| D_1 bajo política uniforme | Directa (Shannon) | Materializado | "Baseline: D_1 = N, como debe ser" | — |
| D_1 bajo frecuencia | Directa (Hill/Jost) | Materializado | "~82 geometrías típicas por recurrencia" | "~82 teorías son viables" |
| D_1 bajo delta_lnL | Directa, pero dependiente de T=1 | Materializado, sin control T | "~28 geometrías concentran la masa del score" | "Solo ~28 geometrías son realmente compatibles" |
| D_inf bajo delta_lnL | Directa | Materializado | "La geometría dominante tiene ~12% del peso" | "El remanente es probablemente una de las top-8" |
| kerr = 0 en soporte | Pipeline upstream | Materializado | "Ninguna entrada Kerr pura pasa los filtros" | "Kerr está excluido" (sin verificar atlas+umbrales) |
| Masa por familia | Derivable, no materializada | Pendiente | "X% del peso va a edgb bajo política Y" | "edgb es la teoría correcta" |
| Estabilidad por jackknife | Necesaria, no materializada | Pendiente | — | Cualquier claim de robustez |

---

## Apéndice C: Referencias primarias citadas

- Hill, M.O. (1973). "Diversity and Evenness: A Unifying Notation and Its Consequences". Ecology 54(2), 427-432.
- Jost, L. (2006). "Entropy and diversity". Oikos 113(2), 363-375.
- Rényi, A. (1961). "On measures of entropy and information". Proceedings of the 4th Berkeley Symposium on Mathematical Statistics and Probability, 1, 547-561.
- Hoeting, J.A., Madigan, D., Raftery, A.E. & Volinsky, C.T. (1999). "Bayesian Model Averaging: A Tutorial". Statistical Science 14(4), 382-417.
- Kish, L. (1965). Survey Sampling. John Wiley & Sons. (Fórmula ESS = 1/Σw_i^2.)
- Chao, A., Chiu, C.-H. & Jost, L. (2014). "Unifying Species Diversity, Phylogenetic Diversity, Functional Diversity, and Related Similarity and Differentiation Measures Through Hill Numbers". Annual Review of Ecology, Evolution, and Systematics 45, 297-324. (Extensión de Hill numbers a diversidades funcionales y filogenéticas — relevante como ejemplo de aplicación de Hill numbers fuera de la ecología de especies.)
