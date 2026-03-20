
# Informe técnico: De observables Kerr a geometrías alternativas
### Estado del arte, límites fundamentales y rutas científicamente defendibles
**Versión 2.0 — Marzo 2026**

---

> **Conclusión anticipada:** La comunidad ha desarrollado tests nulos de consistencia con Kerr que son metodológicamente sólidos y observacionalmente maduros. No ha desarrollado inferencia geométrica observacional hacia estructuras hiperbólicas ni hacia familias de métricas alternativas a partir de datos de ondas gravitacionales. Esa brecha no es accidental ni un déficit temporal: es una consecuencia del carácter fundamentalmente sub-determinado del problema inverso con los observables disponibles.

---

## I. Resumen ejecutivo

**Lo que está sólidamente establecido:**

1. El modo cuasinormal fundamental (ℓ,m,n) = (2,2,0) se detecta robustamente en fusiones de BBH. Su frecuencia y amortiguamiento son consistentes con las predicciones de Kerr en todos los eventos bien caracterizados del catálogo LVK.

2. GW150914 es el evento de referencia. El debate sobre el overtone (2,2,1) — Isi et al. 2019 a 3.6σ vs. Cotesta et al. 2022 cuestionándolo por sensibilidad a t₀ — ilustra el cuello de botella estructural del tiempo de inicio del ringdown.

3. GW190521 muestra evidencia de dos modos angulares simultáneos — (2,2,0) y (3,3,0) — con factor de Bayes ∼56 (Capano et al. 2023). El caso más sólido de espectroscopía multi-modo antes de 2025.

4. GW250114 (enero 2025, SNR red ∼76) es el punto de inflexión: tres modos identificados — (2,2,0), (2,2,1) y (4,4,0) — con restricciones 2–3× más estrictas que el catálogo GWTC-4 combinado. El overtone aparece a >3σ.

5. Restricciones combinadas O1+O2+O3: ∼10% en δf₂₂₀ y ∼20% en δτ₂₂₀, todas consistentes con cero desviación respecto a Kerr.

6. **El enfoque dominante es el test nulo parametrizado.** Los frameworks pSEOBNR/ppE/ParSpec miden δf, δτ y verifican consistencia con cero. No reconstruyen geometrías.

7. No existe paper publicado y revisado por pares que haya inferido una métrica concreta beyond-Kerr — Johannsen, bumpy BH, EdGB, dCS — a partir de análisis de ringdown de datos GW.

**Lo que está en desarrollo:**

8. QNMs no lineales (cuadráticos) descubiertos en simulaciones NR (Cheung et al. 2023, Mitman et al. 2023). Nueva clase de observables. Detección en datos requiere SNR de ringdown ∼100+.

9. Inferencia del exponente de escalado de curvatura p, donde desviaciones ∝ M^p (Payne et al. 2024). El resultado más cercano a "inferencia de clase de corrección geométrica" con datos reales. Un solo resultado publicado.

10. Modelos IMR en teorías alternativas (EdGB, dCS) emergiendo pero perturbativos en espín lento. No aplicados rutinariamente a datos de ringdown.

**Lo que no existe:**

11. **No existe** ruta observacional aceptada desde QNMs hacia geometrías hiperbólicas en ninguno de los cinco sentidos precisos del término.

12. **No existe** inferencia poblacional de familias geométricas o métricas alternativas.

13. **No existe** aplicación operacional de geometría de la información (Fisher manifolds, sloppy models) para clasificar familias de agujeros negros con datos GW.

14. La conclusión honesta y precisa: **la comunidad no ha llegado ahí**, y los datos actuales no son suficientes para hacerlo con ninguna metodología disponible.

---

## II. Taxonomía del campo

### 1. Tests observacionales de Kerr con ringdown
**Madurez: alta. Único subtaxón con resultados observacionales robustos.**

Mide frecuencias QNM y tiempos de amortiguamiento de eventos reales y verifica consistencia con predicciones de Kerr. Opera como test nulo: la hipótesis nula es Kerr. Observables primarios: δf₂₂₀ = (f_obs − f_Kerr)/f_Kerr y δτ₂₂₀. Herramientas: análisis temporal (Carullo et al. 2019), frecuencial (Finch & Moore 2021), IMR completo (pSEOBNRv4HM, v5PHM). Tests de consistencia IMR comparan M_f, χ_f entre inspiral y ringdown. Restricciones actuales: ±10% en frecuencia, ±20% en amortiguamiento.

**Distinción crítica:** consistencia con Kerr ≠ identificación de la geometría. Un resultado nulo en δf es compatible tanto con Kerr como con cualquier métrica beyond-Kerr cuyas predicciones QNM sean indistinguibles dentro de los errores observacionales.

### 2. Parametrizaciones beyond-Kerr
**Madurez: media-alta en teoría; baja para aplicación a datos reales.**

Frameworks que introducen desviaciones libres respecto a Kerr sin comprometerse con una teoría específica:
- **ppE** (Yunes & Pretorius 2009): modifica fase/amplitud del inspiral. Extensión a ringdown es parcial.
- **Johannsen-Psaltis** (2011): deformación ε₃. No es solución exacta de ecuaciones de campo.
- **Johannsen** (2013): preserva separabilidad Hamilton-Jacobi.
- **Rezzolla-Zhidenko / KRZ** (2014, 2016): fracciones continuadas; convergencia superior.
- **ParSpec** (Maselli et al. 2020): expande frecuencias QNM como función del espín remanente y parámetros de deformación. El más directamente aplicable a espectroscopía de ringdown. No aplicado a datos reales aún.
- **pSEOBNR** (Buonanno et al., Ghosh et al. 2021): modelo IMR completo con desviaciones parametrizadas. Estándar en análisis LVK de GWTC-3.

Ninguna ha sido usada para inferir una métrica concreta a partir de datos de ringdown.

### 3. Métricas concretas de teorías alternativas
**Madurez: teórica alta; conexión con datos: baja.**

- **EdGB:** QNMs por Blázquez-Salcedo et al. (2016–2018); diferencias respecto a Kerr de ∼0.1–2% para acoplamientos actualizados. Detectables con ET/CE, no con LIGO actual.
- **dCS:** Wagle et al. (2021–2023); solo perturbativo en espín lento.
- **Kerr con pelo escalar/Proca:** Herdeiro & Radu (2014, 2016); espectro QNM calculado, modelos IMR ausentes.
- **Escalarización espontánea:** Doneva & Yazadjiev (2018); QNMs en EdGB escalarizado calculados.
- **Agujeros negros regulares (Bardeen, Hayward):** QNMs calculados por Flachi & Lemos (2013), Lin et al. (2013). Candidatos naturales para atlas pero sin modelos IMR.

Ninguna de estas métricas ha sido aplicada de forma rutinaria a datos de ringdown de LIGO/Virgo.

### 4. Direcciones AdS / holográficas / NHEK / "hiperbólicas"
**Madurez para GW astrofísicas: nula.**

Véase Sección VIII para taxonomía rigurosa en cinco categorías. La conexión con observaciones de ondas gravitacionales astrofísicas es inexistente en todos los casos relevantes para un pipeline como BASURIN.

### 5. Inferencia jerárquica multi-evento
**Madurez: metodológica alta; aplicación a ringdown-only: emergente.**

Formalizada por Isi, Chatziioannou & Farr (2019) y Zimmerman et al. (2019). Estándar LVK desde GWTC-2. Aplicado a ∼50 eventos para coeficientes PN; ∼12 para ringdown. No aplicado para inferir familias geométricas.

### 6. Geometría de la información / model manifolds
**Madurez para GW: inexistente como programa operacional.**

La FIM es herramienta estándar de forecasting (Vallisneri 2008). La geometría de Ruppeiner se aplica a termodinámica de BH (campo distinto). El framework de sloppy models (Transtrum, Machta & Sethna 2011–2015) no ha sido aplicado a formas de onda GW. **La curvatura del espacio de parámetros estadísticos no es la curvatura del espaciotiempo del agujero negro.** Confundirlas es un error categorial.

---

## III. Línea temporal de hitos

| Año | Paper / Contribución | Por qué importa | Tipo |
|-----|---------------------|-----------------|------|
| 1970 | Vishveshwara: descubrimiento de QNMs | Fundamento teórico | Teórico |
| 1973 | Teukolsky: perturbaciones de Kerr | Base matemática para QNMs en Kerr | Teórico |
| 1985 | Leaver: fracciones continuadas | Estándar computacional vigente | Metodológico |
| 1997 | Vanzo: BH topología hiperbólica en AdS | Existen BH con horizonte hiperbólico, en AdS | Teórico |
| 1999 | Bardeen & Horowitz: geometría NHEK | Factor AdS₂ near-horizon en Kerr extremal | Teórico |
| 2004 | Dreyer et al.: espectroscopía BH | Programa formal: ≥2 QNMs → test no-hair | Metodológico |
| 2004 | Collins & Hughes: bumpy BHs | Primera parametrización de deformaciones de Kerr | Teórico |
| 2006 | Berti, Cardoso & Will: LISA spectroscopy | Framework (δf, δτ); estimaciones SNR | Metodológico |
| 2008 | Vallisneri: FIM en GW | Referencia definitiva sobre límites del FIM | Metodológico |
| 2009 | Berti, Cardoso & Starinets: Living Review | Revisión canónica; tablas de QNMs | Revisión |
| 2009 | Yunes & Pretorius: ppE | Framework theory-agnostic para tests | Teórico/Metod. |
| 2009 | Guica et al.: Kerr/CFT | AdS₂ en NHEK → entropía BH extremal | Teórico |
| 2011 | Johannsen & Psaltis: métrica JP | Parametrización beyond-Kerr | Teórico |
| 2013 | Johannsen: métrica regular | Preserva separabilidad H-J | Teórico |
| 2014 | Herdeiro & Radu: Kerr con pelo | Primera solución con pelo bosónico | Teórico |
| 2014 | Rezzolla & Zhidenko: parametrización RZ | Fracciones continuadas, convergencia superior | Teórico |
| 2015 | Berti et al. (56 autores): review | Enciclopedia del campo | Revisión |
| **2015** | **Detección GW150914** | **Primera fusión BBH observada** | **Datos reales** |
| 2016 | LVK O1 TGR: primer test de ringdown | Modo (2,2,0) consistente con Kerr | Datos reales |
| 2016 | Cardoso, Franzin & Pani: ecos | Fenomenología ECOs | Teórico |
| **2019** | **Giesler et al.: overtones** | Modelado desde el pico; reencuadra el campo | Metod./NR |
| **2019** | **Isi et al.: test no-hair GW150914** | Reclamo (2,2,0)+(2,2,1) a 3.6σ | Datos reales |
| 2019 | Carullo et al.: análisis temporal | Sin evidencia >1 modo; debate abierto | Datos reales |
| **2019** | **Isi, Chatziioannou & Farr: inferencia jerárquica** | Estándar para combinar eventos | Metodológico |
| 2019 | Cardoso & Pani: Living Review ECOs | Revisión comprensiva | Revisión |
| 2020 | Maselli et al.: ParSpec | Parametrización espectro QNM beyond-Kerr | Teórico/Metod. |
| 2021 | Jaramillo et al.: pseudoespectro | Overtones altos inestables | Teórico |
| 2021 | Ghosh, Brito & Buonanno: pSEOB O3a | Restricciones parametrizadas con IMR completo | Datos reales |
| 2021 | Moore et al.: sistemáticas waveform | 10–30 eventos generan evidencia espuria | Metodológico |
| **2022** | **Cotesta et al.: cuestionamiento del overtone** | Debate sobre sensibilidad a t₀ | Datos reales |
| **2023** | **Cheung et al. / Mitman et al.: QNMs no lineales** | Modos cuadráticos en NR; nuevo observable | Teórico/NR |
| **2023** | **Capano et al.: dos modos en GW190521** | (2,2,0)+(3,3,0) con BF∼56 | Datos reales |
| **2024** | **Payne et al.: dependencia en curvatura** | Exponente de escalado; más cercano a inferencia de clase | Datos reales |
| **2025** | **GW250114: señal más fuerte** | SNR∼76; 3 modos; restricciones 2–3× GWTC-4 | Datos reales |
| 2025 | Berti et al. (68 autores): review BH spectroscopy | Estado del arte comunitario | Revisión |

---

## IV. Tabla de papers clave

| Referencia | Año | Tipo | Observables | Objeto inferido | Datos reales | Relación con geom. hiperbólica | Limitaciones |
|-----------|------|------|-------------|-----------------|:---:|-------------------------------|--------------|
| Dreyer et al. | 2004 | Metod. | f, τ QNM | M_f, χ_f (consistencia) | No | Ninguna | Requiere ≥2 modos; SNR futuro |
| Collins & Hughes | 2004 | Teórico | Multipolos | Deformaciones de Kerr | No | Ninguna | Perturbativo |
| Berti, Cardoso & Will | 2006 | Metod. | δf, δτ | Desviaciones no-hair | No | Ninguna | Solo proyecciones LISA |
| Vallisneri | 2008 | Metod. | Fisher matrix | Covarianzas paramétricas | No | Ninguna (espacio estadístico) | Aproximación gaussiana |
| Berti, Cardoso & Starinets | 2009 | Revisión | Espectro QNM | Tablas Kerr | No | Ninguna | Catálogo, no observacional |
| Yunes & Pretorius | 2009 | Teórico | Fase inspiral | Parámetros ppE | No | Ninguna | Solo inspiral; extensión a merger parcial |
| Guica et al. | 2009 | Teórico | Simetría conformal NHEK | Entropía BH extremal | No | **Conceptual** (AdS₂ hiperbólico) | Solo extremal; sin observable GW |
| Johannsen & Psaltis | 2011 | Teórico | Sombras, QNMs | Parámetros ε_i | No | Ninguna | No solución exacta de EF |
| Johannsen | 2013 | Teórico | Constantes de movimiento | Separabilidad H-J | No | Ninguna | Patologías a alto espín |
| Rezzolla & Zhidenko | 2014 | Teórico | QNMs, sombras | Coeficientes f.c. | No | Ninguna | Rotación aproximada |
| Herdeiro & Radu | 2014 | Teórico | Cuadrupolo, ISCO | BH Kerr con pelo | No | Ninguna | Modelos GW ausentes |
| Giesler et al. | 2019 | NR/Metod. | Overtones (2,2,n) | M_f, χ_f desde pico | No (NR) | Ninguna | Debate validez perturbativa |
| **Isi et al.** | **2019** | **Datos reales** | (2,2,0)+(2,2,1) GW150914 | Consistencia no-hair ∼20% | **Sí** | Ninguna | Debatido: sensibilidad a t₀ |
| Carullo et al. | 2019 | Datos reales | (2,2,0) GW150914 | Sin evidencia >1 modo | Sí | Ninguna | Contradicción con Isi et al. |
| **Isi, Chatziioannou & Farr** | **2019** | **Metod.** | Posteriors δφ̂_i multi-evento | Hiperparámetros μ, σ | **Sí** | Ninguna | Coeficientes PN, no ringdown-only |
| Maselli et al. | 2020 | Teórico | ParSpec coeficientes | Desviaciones QNM vs espín | No | Ninguna | FIM; no aplicado a datos |
| **Ghosh, Brito & Buonanno** | **2021** | **Datos reales** | δf̂₂₂₀, δf̂₂₂₁ | Consistencia Kerr O3a | **Sí** | Ninguna | No ringdown-only |
| Jaramillo et al. | 2021 | Teórico | Pseudoespectro | Inestabilidad overtones | No | Ninguna | Limita info extraíble de overtones |
| Moore et al. | 2021 | Metod. | Sistemáticas waveform | Falsos positivos beyond-GR | No | Ninguna | Advertencia crítica para catálogos |
| **Cotesta et al.** | **2022** | **Datos reales** | BF overtone GW150914 | Sin evidencia robusta | **Sí** | Ninguna | BF sensible a t₀ y prior |
| **Cheung et al.** | **2023** | **NR** | Modo cuadrático | QNMs no lineales | No (NR) | Ninguna | No detectado en datos reales |
| **Capano et al.** | **2023** | **Datos reales** | (2,2,0)+(3,3,0) GW190521 | BF∼56 para dos modos | **Sí** | Ninguna | Evento individual |
| **Payne et al.** | **2024** | **Datos reales** | Exponente p (M^p) | Clase de corrección EFT | **Sí** | Ninguna | Un resultado; alcance limitado |
| Vanzo/Birmingham/Emparan | 1997–99 | Teórico | Termodinámica AdS | BH horizonte hiperbólico | No | **Directa** (horizonte ES hiperbólico) | Puramente AdS; sin conexión GW |
| **LVK GW250114** | **2025** | **Datos reales** | (2,2,0), (2,2,1), (4,4,0) | 3 modos; Kerr a ∼% | **Sí** | Ninguna | Test más estricto a la fecha |

---

## V. Estado del arte con datos reales

### GW150914

**Detección robusta:** Modo (2,2,0) en todas las metodologías. M_f ∼ 68 M☉, χ_f ∼ 0.67. Restricciones δf₂₂₀ ≈ −0.05⁺⁰·²⁰₋₀.₂₀, δτ₂₂₀ consistente con cero.

**Evidencia debatida:** Overtone (2,2,1): Isi et al. (2019) reclaman 3.6σ; Cotesta et al. (2022) muestran que el BF es sensible al tiempo de inicio t₀. El debate metodológico sobre cuándo empieza el régimen perturbativo del ringdown es la raíz del desacuerdo. GW250114 puede estar resolviendo este debate.

**Resultados nulos:** Sin evidencia de ecos. Sin evidencia de modos angulares adicionales.

### GW190521

**Evidencia con caveats:** Capano et al. (2023) reportan BF∼56 para dos modos simultáneos — (2,2,0) y (3,3,0). La asimetría de masa (q ≈ 0.79) excita el modo (3,3,0). El caso más sólido de espectroscopía multi-modo antes de GW250114. Sin embargo, es un evento individual de alta masa (M_final ∼ 250 M☉) con señal parcialmente por debajo de la banda sensible. La comunidad no lo considera definitivo.

### GW250114

**Detección robusta de tres modos:** LVK identifican (2,2,0), (2,2,1) y (4,4,0). Restricciones individuales 2–3× más estrictas que GWTC-4 combinado. Overtone a >3σ. Las restricciones sobre δf del modo fundamental son las más precisas obtenidas de un solo evento.

**Caveats:** Paper en circulación desde septiembre 2025; pendiente de adopción comunitaria amplia. QNMs no lineales no buscados sistemáticamente con metodología publicada.

### Catálogos combinados O1+O2+O3

Tests de GR con GWTC-3: ∼70 eventos BBH para coeficientes PN, ∼12 para ringdown. Consistencia con GR en todos. Restricciones combinadas: ±10% (δf), ±20% (δτ). Test de consistencia IMR en formulaciones 2D y 4D: consistente con GR. Búsquedas de ecos: resultados nulos en todos los catálogos.

---

## VI. Estado del arte multi-evento y jerárquico

### Estándar actual

Inferencia jerárquica de Isi, Chatziioannou & Farr (2019): distribución poblacional de desviaciones como gaussiana con hiperparámetros (μ, σ). Aplicada a ∼50 eventos BBH para coeficientes PN en GWTC-3. Consistencia con GR a ∼1% en los coeficientes mejor restringidos.

El insight de Zimmerman et al. (2019): multiplicación de factores de Bayes y multiplicación de verosimilitudes son casos límite del framework jerárquico. La multiplicación de BF puede producir resultados erróneos cuando los priors son mucho más anchos que la señal; la inferencia jerárquica converge correctamente.

### Lo que se ha hecho una vez o pocas veces

Payne et al. (2024): inferencia del exponente de escalado p en desviaciones ∝ M^p con datos LVK. Resultado observacional sobre la clase de correcciones gravitacionales. Un único paper publicado.

Payne, Isi, Chatziioannou & Farr (2023): inferencia conjunta de población astrofísica + desviaciones de GR, demostrando que los supuestos sobre distribuciones de masa y espín pueden sesgar los tests.

Magee et al. (2024): cuantificación de efectos de selección en tests de GR multi-evento. Resultado: sesgos de selección solo afectan desviaciones mucho mayores que las restricciones actuales.

### Lo que no se ha hecho

- Inferencia jerárquica dedicada de desviaciones de ringdown-only (δf, δτ) combinando múltiples eventos.
- Inferencia de familias geométricas o clases de métricas a nivel poblacional.
- Aplicación de ParSpec a datos reales de catálogo.
- Combinación del ratio f₂₂₁/f₂₂₀ entre múltiples eventos.
- "Espectroscopía poblacional" en sentido de clasificación geométrica.

---

## VII. Evaluación crítica

### Lo que está maduro

Test de consistencia IMR. Test nulo (2,2,0) con δf, δτ. Inferencia jerárquica para combinar tests de GR. Tablas de QNMs de Kerr como referencia (Berti et al. 2009). Comprensión de que las sistemáticas de forma de onda son el cuello de botella dominante a largo plazo.

### Lo que está en desarrollo activo

Espectroscopía multi-modo con eventos reales (GW190521, GW250114). Cálculos de QNMs en teorías alternativas (EdGB, dCS), con errores de ∼10–50% para espines típicos. Teoría de QNMs no lineales y su eventual conexión con datos. Inferencia del exponente de escalado de curvatura. Detección sistemática del overtone (2,2,1) en eventos con SNR suficiente.

### Lo que no está formulado de forma identificable

No existe framework operacional para inferir "la familia geométrica" de un agujero negro a partir de sus QNMs observados. El problema inverso (QNMs → métrica) es sub-determinado con finitos modos observados a precisión de 10–20%. La inestabilidad pseudoespectral (Jaramillo et al. 2021) limita la información extraíble de overtones altos. La geometría de la información del espacio de modelos no ha sido formulada como herramienta de clasificación de teorías gravitacionales con datos de GW. La conexión entre "geometría hiperbólica" (en cualquier sentido) y observables de GW astrofísicas no ha sido formulada como programa de investigación.

### Focos de riesgo de overclaim

1. Consistencia con Kerr → geometría Kerr identificada. Error de afirmación.
2. ParSpec restringido → familia geométrica identificada. Confusión entre espacio de parámetros y espacio de geometrías.
3. Curvatura del espacio Fisher → curvatura del espaciotiempo. Error categorial severo.
4. "Geometría hiperbólica" en cualquier sentido → conclusión observacional sobre BH astrofísico.
5. Ecos no excluidos → evidencia positiva de ECOs.
6. "El conjunto de geometrías compatibles excluye Kerr" sin selección de modelos bayesiana formal.

---

## VIII. Taxonomía rigurosa del término "hiperbólico"

**Regla operativa:** Antes de sacar cualquier conclusión, clasificar el uso en una de estas cinco categorías. No permitir mezcla entre categorías.

### Categoría 1: Topología del horizonte en AdS

**Descripción:** Agujeros negros en espacio-tiempo Anti-de Sitter (AdS) con horizonte de topología hiperbólica (curvatura seccional −1). Satisfacen las ecuaciones de Einstein con Λ < 0 (Vanzo 1997, Birmingham 1999, Emparan 1999). Construcciones holográficas relevantes para materia fuertemente acoplada.

**Conexión con observaciones GW astrofísicas:** **Inexistente.** Los agujeros negros astrofísicos viven en espacio asintóticamente plano (Λ ≈ 0). No hay propuesta observacional que conecte la topología del horizonte AdS con señales GW medibles.

### Categoría 2: Estructura near-horizon extremal — geometría NHEK

**Descripción:** En el límite Kerr extremal (a → M, χ_f → 1), la geometría near-horizon se denomina NHEK (Bardeen & Horowitz 1999) y contiene un factor AdS₂ × S², con estructura hiperbólica bidimensional. La correspondencia Kerr/CFT (Guica et al. 2009) usa esta simetría para calcular la entropía del agujero negro extremal.

**Conexión con observaciones GW astrofísicas:** **Conceptual únicamente.** Los agujeros negros remanentes observados tienen χ_f ∼ 0.6–0.7, muy lejos del límite extremal. Las propuestas observacionales son electromagnéticas (NHEKline en rayos X, anillo de fotones en EHT), no gravitacionales. No existe propuesta observacional GW concreta que involucre la estructura NHEK.

### Categoría 3: Hiperbolicidad dinámica — exponente de Lyapunov

**Descripción:** Las geodésicas nulas circulares inestables tienen un exponente de Lyapunov λ que mide la tasa de divergencia. En el límite eikonal, las frecuencias QNM satisfacen ω_QNM ≈ Ω_c ℓ − i(n + 1/2)|λ|, donde Ω_c es la frecuencia angular orbital (Cardoso et al. 2009).

**Conexión con observaciones GW astrofísicas:** **Indirecta.** La relación eikonal es exacta solo en el límite ℓ → ∞. Para los modos dominantes (ℓ = 2, 3) es una aproximación con errores del 10–20%. No es directamente observable como "geometría hiperbólica" separada. Su conexión con frecuencias QNM observadas es parcialmente verificable pero solo como consistencia, no como inferencia independiente.

### Categoría 4: Geometría estadística — espacio de parámetros / information geometry

**Descripción:** La Fisher Information Matrix define una métrica en el espacio de parámetros del modelo. Si esa métrica tiene curvatura seccional negativa constante, el espacio es hiperbólico en sentido Riemanniano. La geometría de Ruppeiner aplica geometría diferencial a termodinámica de agujeros negros.

**Conexión con observaciones GW astrofísicas:** **Ninguna como geometría del espaciotiempo.** La curvatura del espacio de parámetros estadísticos es una propiedad de la distribución de probabilidad del modelo, no del espaciotiempo del agujero negro. Decir que "el espacio de parámetros tiene curvatura negativa" no implica ninguna afirmación sobre el horizonte del agujero negro. Esta distinción es fundamental.

**Nota para BASURIN:** El "Método Brunete" explota que la métrica base de Fisher en log(f)/log(Q) tiene curvatura negativa constante. Esto es matemáticamente correcto y puede ser operativamente útil para diseñar métricas de compatibilidad. Pero no tiene implicaciones físicas directas sobre la geometría del agujero negro observado.

### Categoría 5: Encuentros hiperbólicos

**Descripción:** Órbitas de dos cuerpos compactos con energía positiva (no ligadas), descritas por hipérbolas. Producen señales de burst GW. No generan un remanente que oscile en modos QNM.

**Conexión con observaciones GW de ringdown:** **Ninguna.** Los encuentros hiperbólicos no son relevantes para espectroscopía de agujeros negros.

### Tabla resumen

| Categoría | Noción de "hiperbólico" | Conexión con GW astrofísicas | Riesgo de confusión |
|-----------|------------------------|------------------------------|---------------------|
| 1 | Topología horizonte en AdS | **Inexistente** | Alto |
| 2 | Estructura NHEK / AdS₂ near-horizon | **Conceptual** (requiere extremalidad) | Medio |
| 3 | Lyapunov geodésicas nulas | **Indirecta** (límite eikonal) | Bajo |
| 4 | Curvatura espacio Fisher | **Ninguna sobre geometría del BH** | **Muy alto** |
| 5 | Encuentros hiperbólicos | Ninguna con ringdown | Bajo |

---

## IX. Gaps genuinos de investigación

**Gap 1: Aplicación de ParSpec a datos reales de catálogo.**
Maselli et al. (2020) propusieron el framework pero no lo aplicaron a datos reales. Con O4/O5, la aplicación sistemática de ParSpec a eventos con SNR post-merger suficiente produciría las primeras restricciones poblacionales sobre parámetros de desviación del espectro QNM expandidos en espín. Requiere: calibración del prior sobre los coeficientes de expansión, y un atlas de predicciones teóricas para teorías específicas.

**Gap 2: Extensión de la inferencia del exponente de escalado (Payne et al. 2024).**
El resultado más cercano a "inferencia de clase de corrección geométrica" con datos reales. Extenderlo para distinguir entre diferentes exponentes predichos por distintas clases de gravedad modificada (cuadrática, cúbica, etc.) es accionable con datos O4/O5 si se acumulan suficientes eventos con masa bien medida. No requiere nuevos detectores.

**Gap 3: Combinación jerárquica del ratio f₂₂₁/f₂₂₀ entre eventos.**
Una vez que el overtone se detecta con significancia suficiente en múltiples eventos (GW250114 abre esta puerta), el ratio f₂₂₁/f₂₂₀ varía solo ∼0.93–0.99 en Kerr y es independiente de la masa. Su distribución poblacional es un test de consistencia Kerr independiente del test de frecuencias individuales. Gap implementable a corto plazo.

**Gap 4: Geometría de la información aplicada a clasificación de familias de teorías.**
El framework de sloppy models (Transtrum et al.) analiza la estructura del manifold de modelos y identifica direcciones sloppy (mal restringidas) vs. stiff (bien restringidas). Aplicado a familias de formas de onda de GW — e.g., Kerr, EdGB, dCS, Kerr-Newman — podría identificar qué combinaciones de parámetros son realmente distinguibles con datos de ringdown y cuáles no. Este gap es accionable y tiene valor metodológico real para el diseño de pipelines.

**Gap 5: Primer análisis de datos reales con modelo IMR en teoría alternativa específica.**
Con modelos IMR emergentes en EdGB y dCS, un análisis Bayesiano formal de GW250114 (el evento con mejor SNR) con selección de modelos entre GR y una teoría alternativa específica sería el primer resultado de este tipo. Requiere controlar los errores del modelo perturbativo y declarar explícitamente sus límites de validez en espín.

**Gap 6: Control sistemático del t₀ en análisis de ringdown multi-modo.**
El tiempo de inicio del ringdown sigue sin un criterio independiente del modelo. Un estudio sistemático que combine criterios físicos (e.g., fracción de energía radiada post-pico), criterios de estabilidad del posterior, y criterios de consistencia entre overtones podría producir el primer estándar operativo para t₀ en análisis de ringdown. Esto es un prerequisito para cualquier espectroscopía multi-modo auditable.

---

## X. Recomendación final

### Ruta científicamente defendible

**A corto plazo (datos O4/O5):**
- Análisis de ringdown multi-modo de eventos con SNR post-merger >20, reportando restricciones sobre modos individuales con significancias explícitas.
- Combinación jerárquica de δf₂₂₀ y δτ₂₂₀ usando el framework de Isi et al. (2019), produciendo posteriors (μ, σ).
- Implementación del ratio f₂₂₁/f₂₂₀ para eventos con overtone detectado.
- Extensión de la inferencia del tipo Payne et al. (2024) hacia clases de correcciones.

**A medio plazo (ET/CE):**
- Espectroscopía de precisión con ≥3 modos por evento.
- Aplicación de ParSpec a muestras poblacionales para restricciones sobre coeficientes de teorías específicas.
- Detección potencial de QNMs no lineales y verificación de sus predicciones en Kerr.
- Primer análisis Bayesiano formal de selección de modelos entre GR y teorías alternativas específicas con SNR suficiente.

### Ruta prematura o no soportada por la literatura

Las siguientes afirmaciones serían científicamente indefendibles con datos actuales o a corto plazo:

- Que datos de ringdown revelan "la geometría" del agujero negro más allá de consistencia con Kerr.
- Que existe evidencia observacional de estructura hiperbólica/AdS near-horizon en agujeros negros astrofísicos detectados por LIGO/Virgo.
- Que la curvatura del espacio Fisher representa información sobre la geometría del espaciotiempo del agujero negro.
- Que la espectroscopía de ringdown "ha discriminado entre familias geométricas" — salvo selección de modelos bayesiana formal con modelos IMR completos.
- Que la población de geometrías compatibles con los datos excluye Kerr — contrario a todos los resultados publicados.
- Que los ecos no descartados constituyen evidencia positiva de ECOs.

---

## XI. Traducción operativa a un pipeline tipo BASURIN

### Quantities canónicas para agregación multi-evento

**Listas para producción:**
- **δf₂₂₀ y δτ₂₂₀ por evento → posterior conjunta (μ, σ).** Output auditable: posterior sobre la media poblacional y la dispersión. Si μ → 0 y σ → 0 con eventos acumulados, evidencia de consistencia con Kerr. Requiere: declarar prior, modelo de forma de onda, y t₀.
- **ΔM_f/M̄_f y Δχ_f/χ̄_f del test de consistencia IMR.** Estándar LVK, reproducible.
- **Ratio f₂₂₁/f₂₂₀ para eventos con overtone detectado.** Restricción adicional independiente de la masa. Auditable si se declara el criterio de detección del overtone.

**Experimentales pero justificables:**
- **Exponente de escalado p en desviaciones ∝ M^p** (Payne et al. 2024). Un resultado de referencia publicado. Debe marcarse como experimental.
- **Distancia de Mahalanobis en el espacio (f₂₂₀, τ₂₂₀) para comparación con atlas de métricas.** Si existe un atlas de predicciones QNM de métricas específicas con cálculos publicados, la distancia Mahalanobis entre el punto observado y cada modelo es una métrica de compatibilidad bien definida. **Advertencia crítica:** mide compatibilidad espectral, no identifica la métrica. Debe reportarse como "consistente con / inconsistente con la predicción QNM de [modelo X]".
- **Coeficientes ParSpec combinados entre eventos.** Extensión natural cuando se aplique ParSpec a datos reales.

### Outputs auditables sin sobre-interpretación

| Output | Auditable | Condición necesaria |
|--------|:---------:|---------------------|
| Posterior (μ, σ) de δf, δτ multi-evento | ✓ | Declarar prior, t₀, modelo de waveform, umbral SNR |
| Factor de Bayes overtone vs. no-overtone | ✓ | Declarar t₀ y prior sobre amplitudes relativas |
| Posterior sobre exponente de escalado p | ✓ | Declarar modelo de población y prior en p |
| Distancia Mahalanobis al atlas de métricas | ✓ (con caveats) | Declarar que es compatibilidad espectral, no identificación |
| "Fracción de la población compatible con Kerr" | Solo si | Requiere modelo forward completo con efectos de selección |

### Quantities que deben marcarse como experimentales

- Cualquier restricción sobre parámetros de métricas concretas (ε_i de JP, α de EdGB) derivada de análisis de ringdown: las predicciones QNM son perturbativas en espín, con errores del 10–50% para espines típicos.
- Coeficientes ParSpec en ausencia de validación con datos reales.
- Cualquier cantidad derivada de inestabilidad pseudoespectral o perturbaciones del operador.

### Afirmaciones que deben prohibirse por falta de identificabilidad

1. **"El agujero negro tiene geometría [X]"** para cualquier X beyond-Kerr. El problema inverso QNMs → métrica es sub-determinado con los observables disponibles.
2. **"Los datos favorecen [teoría A] sobre [teoría B]"** sin selección de modelos bayesiana formal con modelos IMR completos y calibrados en ambas teorías.
3. **"La curvatura del espacio de parámetros Fisher indica estructura hiperbólica del agujero negro"**: error categorial entre geometría estadística y geometría del espaciotiempo.
4. **"La espectroscopía poblacional ha mapeado la familia de métricas"**: ningún pipeline existente ha hecho esto.
5. **"El conjunto de geometrías compatibles excluye Kerr"**: contrario a todos los resultados publicados con datos reales.
6. **"Los ecos están acotados pero no excluidos"** como argumento para sugerir evidencia de ECOs: ausencia de refutación ≠ evidencia positiva.

### Diseño mínimo metodológicamente serio

Un pipeline de ringdown multi-evento metodológicamente serio debe:

1. **Análisis por evento:** Estimar posteriors sobre δf₂₂₀ y δτ₂₂₀ con un modelo de forma de onda calibrado. Para eventos con SNR post-merger >15, intentar análisis multi-modo y reportar el factor de Bayes overtone/multi-modo vs. modo fundamental. Declarar t₀ y justificarlo.

2. **Criterio de selección de eventos:** Umbral de SNR post-merger explícito (mínimo recomendado: 8 para modo fundamental, 15 para multi-modo). Justificación documentada en el contrato de la etapa.

3. **Agregación multi-evento:** Inferencia jerárquica con hiperparámetros (μ, σ) sobre la distribución poblacional de desviaciones. No multiplicación de factores de Bayes.

4. **Control de robustez:** Tests de sensibilidad a t₀, al prior sobre amplitudes relativas de modos, y al modelo de forma de onda. Estos tests deben ser parte del contrato de la etapa, no análisis opcionales.

5. **Propagación de incertidumbres sistemáticas:** No solo incertidumbres estadísticas. Las incertidumbres del modelo de forma de onda deben propagarse en los resultados finales.

6. **Declaración epistémica explícita en cada output:** La distinción entre "consistente con Kerr" (inferencia alcanzable con datos actuales) e "identificación de la geometría" (no identificable con los observables disponibles) debe estar documentada en el contrato de cada etapa del pipeline. El campo `interpretation_scope` de cada output debe declarar explícitamente qué se puede y qué no se puede concluir del resultado.
