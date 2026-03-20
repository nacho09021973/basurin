# Informe técnico: De observables Kerr a geometrías alternativas
### Estado del arte, límites fundamentales y rutas científicamente defendibles
**Versión 2.0 — Marzo 2026**

---

> **Conclusión anticipada:** La comunidad ha desarrollado tests nulos de consistencia con Kerr que son metodológicamente sólidos y observacionalmente maduros. No ha desarrollado inferencia geométrica observacional hacia estructuras hiperbólicas ni hacia familias de métricas alternativas a partir de datos de ondas gravitacionales. Esa brecha no es accidental ni un déficit temporal: es una consecuencia del carácter fundamentalmente sub-determinado del problema inverso con los observables disponibles.

---

## I. Resumen ejecutivo

**Lo que está sólidamente establecido:**

1. El modo cuasinormal fundamental (ℓ,m,n) = (2,2,0) se detecta de forma robusta y reproducible en fusiones de agujeros negros binarios. Su frecuencia y tiempo de amortiguamiento son consistentes con las predicciones del agujero negro de Kerr de relatividad general en todos los eventos bien caracterizados del catálogo LVK.

2. GW150914 proporciona la restricción de ringdown-only más extensamente analizada. El debate sobre si el overtone (2,2,1) es genuinamente detectable (Isi et al. 2019 vs. Cotesta et al. 2022) ilustra la sensibilidad al tiempo de inicio t₀ como cuello de botella estructural.

3. GW190521 muestra evidencia de dos modos angulares simultáneos — (2,2,0) y (3,3,0) — con factor de Bayes ∼56 (Capano et al. 2023). Este es el caso más sólido de espectroscopía multi-modo antes de 2025.

4. GW250114 (enero 2025, SNR red ∼76, SNR post-merger ∼40) es el punto de inflexión observacional: tres modos identificados — (2,2,0), (2,2,1) y (4,4,0) — con restricciones individuales 2–3 veces más estrictas que el catálogo combinado GWTC-4 entero. El overtone (2,2,1) aparece a >3σ.

5. Las restricciones combinadas del catálogo O1+O2+O3 para desviaciones parametrizadas de ringdown son del orden de ∼10% en frecuencia y ∼20% en tiempo de amortiguamiento, todas consistentes con cero desviación respecto a Kerr.

6. **El enfoque dominante en la comunidad es el test nulo parametrizado**, no la inferencia de métricas concretas. Los frameworks pSEOBNR/ppE/ParSpec operan midiendo δf, δτ respecto a Kerr y verificando consistencia con cero.

7. No existe ningún paper publicado y revisado por pares que haya inferido una métrica concreta beyond-Kerr — Johannsen, bumpy BH, EdGB, dCS — a partir de datos de ondas gravitacionales mediante análisis de ringdown.

**Lo que está en desarrollo:**

8. Los QNMs no lineales (cuadráticos) han sido descubiertos en simulaciones de relatividad numérica (Cheung et al. 2023, Mitman et al. 2023). Representan una nueva clase de observables, pero su detección en datos reales requiere SNR de ringdown ∼100+, accesible solo con detectores de tercera generación.

9. La inferencia del exponente de escalado de curvatura p, donde las desviaciones escalan como M^p (Payne et al. 2024), es el resultado más cercano a inferencia de "clase de corrección geométrica" a partir de datos reales, pero con un único resultado publicado y alcance limitado.

10. Los modelos IMR completos en teorías alternativas (primer modelo EdGB, Pierini & Gualtieri 2021–2022; primer modelo ESGB de inspiral completo, Loutrel et al. 2022) están emergiendo. Aún no aplicados rutinariamente a datos de ringdown.

**Lo que no existe:**

11. **No existe** ruta observacional aceptada desde observables de ringdown hacia geometrías hiperbólicas, en ninguno de los cinco sentidos precisos del término (AdS, NHEK extremal, Lyapunov, información, scattering).

12. **No existe** inferencia poblacional de familias geométricas o métricas alternativas. Nadie ha realizado análisis jerárquico dedicado a discriminar entre Kerr, Kerr-Newman, bumpy BH o teorías modificadas a nivel de catálogo de ringdown.

13. **No existe** aplicación operacional de geometría de la información (Fisher manifolds, sloppy models, curvatura del espacio de parámetros) para clasificar familias de agujeros negros o teorías con datos de GW.

14. La conclusión honesta y precisa es: **la comunidad no ha llegado ahí**, y los datos actuales no son suficientes para hacerlo, independientemente de la metodología.

---

## II. Taxonomía del campo

### 1. Tests observacionales de Kerr con ringdown
**Madurez: alta. Resultados observacionales robustos.**

Mide frecuencias QNM y tiempos de amortiguamiento de eventos reales y verifica consistencia con predicciones de Kerr. Opera como test nulo: la hipótesis nula es Kerr. Los observables son δf₂₂₀ = (f_obs − f_Kerr)/f_Kerr y δτ₂₂₀ análogamente. Herramientas: análisis temporal (Carullo et al. 2019), frecuencial (Finch & Moore 2021), IMR completo (pSEOBNRv4HM, Buonanno et al.; pSEOBNRv5PHM). Los tests de consistencia IMR comparan M_f, χ_f inferidos del inspiral vs. del ringdown; las restricciones en δf/δτ tienen precisión de 10–20% a nivel de catálogo O3.

**Distinción crítica:** consistencia con Kerr ≠ identificación de la geometría. Un resultado nulo en δf es compatible tanto con Kerr como con cualquier métrica beyond-Kerr cuyas predicciones QNM sean indistinguibles de Kerr dentro de los errores observacionales. Esta confusión semántica genera overclaims frecuentes.

### 2. Parametrizaciones beyond-Kerr
**Madurez: media-alta para teoría; baja para datos reales.**

Introducen desviaciones libres respecto a Kerr sin comprometerse con una teoría gravitacional específica. Principales frameworks:
- **ppE (Yunes & Pretorius 2009):** modifica la fase/amplitud del inspiral con parámetros (β, b). Aplicado extensamente a datos de inspiral; extensión a ringdown es parcial.
- **Johannsen-Psaltis (2011):** métrica with deformación ε₃; no es solución exacta de ecuaciones de campo. Aplicada a sombras (EHT) y QNMs analíticamente.
- **Johannsen (2013):** preserva separabilidad Hamilton-Jacobi. Más regular; también aplicada a sombras.
- **Rezzolla-Zhidenko (2014) / KRZ (Konoplya, Rezzolla & Zhidenko 2016):** fracciones continuadas; convergencia superior a JP.
- **ParSpec (Maselli et al. 2020):** expande las frecuencias QNM en parámetros de deformación del espectro como función del espín remanente. Es el framework más directamente aplicable a espectroscopía de ringdown.
- **pSEOBNR (Buonanno et al., Ghosh et al. 2021):** modelo IMR completo con desviaciones parametrizadas en los coeficientes del Hamiltoniano y la radiación. Estándar en los análisis LVK de GWTC-3.

**Ninguna de estas parametrizaciones ha sido usada para inferir una métrica concreta a partir de datos de ringdown.** Son herramientas para medir desviaciones, no para reconstruir geometría.

### 3. Métricas concretas de teorías alternativas
**Madurez: teórica alta; conexión con datos: baja.**

Soluciones de agujeros negros en teorías gravitacionales específicas con QNMs calculados:
- **Einstein-dilaton-Gauss-Bonnet (EdGB):** Kanti et al. (1996); QNMs por Blázquez-Salcedo et al. (2016–2018); primer modelo inspiral-merger (Loutrel et al. 2022). Las frecuencias QNM difieren de Kerr en ∼0.1–2% para acoplamientos actualizados ← por encima del umbral de detección de ET/CE, por debajo de LIGO actual.
- **Chern-Simons dinámico (dCS):** Yunes & Pretorius (2009), Wagle et al. (2021–2023); solo perturbativo en espín lento.
- **Agujeros negros con pelo escalar (Kerr-boson star, Kerr-Proca):** Herdeiro & Radu (2014, 2016); espectro QNM calculado pero modelos de señal GW completos ausentes.
- **Escalarización espontánea:** Doneva & Yazadjiev (2018), Silva et al. (2018); QNMs en EdGB escalarizado calculados.
- **Agujeros negros regulares (Bardeen, Hayward):** QNMs calculados numéricamente por Flachi & Lemos (2013), Lin et al. (2013). Candidatos naturales para un atlas observacional pero sin modelos IMR.

**Ninguna de estas métricas ha sido aplicada de forma rutinaria a análisis de datos de ringdown de LIGO/Virgo.** La brecha es instrumental (SNR insuficiente para distinguir predicciones) y técnica (ausencia de modelos IMR completos listos para producción).

### 4. Direcciones AdS / holográficas / NHEK / "hiperbólicas"
**Madurez para GW astrofísicas: nula.**

Véase Sección C para la taxonomía explícita en cinco categorías. La conclusión de esta subsección es que **ninguna de las cinco nociones de "hiperbólico" tiene conexión directa con observaciones de ondas gravitacionales astrofísicas actuales o proyectadas a corto plazo.**

### 5. Inferencia jerárquica multi-evento
**Madurez: metodológica alta; aplicación a ringdown: emergente.**

Framework formalizado por Isi, Chatziioannou & Farr (2019) y Zimmerman et al. (2019). Estándar para combinar tests de GR en LVK desde GWTC-2. Aplicado a ∼50 eventos para coeficientes PN; a ∼12 eventos para ringdown. No se ha aplicado para inferir familias geométricas o métricas alternativas.

### 6. Geometría de la información / model manifolds
**Madurez para GW: inexistente como programa operacional.**

La Fisher Information Matrix (FIM) es una herramienta estándar de forecasting (Cutler & Flanagan 1994, Vallisneri 2008). La geometría de Ruppeiner se aplica a termodinámica de agujeros negros (Sahay et al. 2010) — campo completamente distinto. El framework de sloppy models (Transtrum, Machta & Sethna 2011–2015) no ha sido aplicado a formas de onda de GW. **La curvatura del espacio de parámetros estadísticos (espacio de modelos) no es la curvatura del espaciotiempo del agujero negro.** Confundirlas es un error categorial.

---

## III. Línea temporal de hitos

| Año | Paper / Contribución | Por qué importa | Tipo |
|-----|---------------------|-----------------|------|
| 1970 | Vishveshwara: discovery of QNMs | Fundamento teórico de la espectroscopía BH | Teórico |
| 1973 | Teukolsky: perturbaciones de Kerr | Base matemática para QNMs en Kerr | Teórico |
| 1985 | Leaver: fracciones continuadas para QNMs | Estándar computacional vigente para 40 años | Metodológico |
| 1997 | Vanzo: BH topología hiperbólica en AdS | Existen BH con horizonte hiperbólico, pero en AdS | Teórico |
| 1999 | Bardeen & Horowitz: geometría NHEK | Factor AdS₂ near-horizon en Kerr extremal | Teórico |
| 2004 | Dreyer et al.: propuesta espectroscopía BH | Programa formal: ≥2 QNMs → test no-hair | Metodológico |
| 2004 | Collins & Hughes: bumpy black holes | Primera parametrización deformaciones de Kerr | Teórico |
| 2006 | Berti, Cardoso & Will: espectroscopía con LISA | Framework (δf, δτ); estimaciones SNR necesario | Metodológico |
| 2008 | Vallisneri: uso y abuso de la FIM | Referencia definitiva sobre límites del FIM | Metodológico |
| 2009 | Berti, Cardoso & Starinets: Living Review | Revisión canónica; tablas de QNMs | Revisión |
| 2009 | Yunes & Pretorius: ppE | Framework theory-agnostic para tests de inspiral | Teórico/metodológico |
| 2009 | Guica et al.: Kerr/CFT | AdS₂ en NHEK → entropía BH extremal vía Cardy | Teórico |
| 2011 | Vigeland, Yunes & Stein: bumpy BHs | Extensión a teorías alternativas | Teórico |
| 2011 | Johannsen & Psaltis: métrica JP | Parametrización beyond-Kerr campo fuerte | Teórico |
| 2013 | Johannsen: métrica regular beyond-Kerr | Preserva separabilidad H-J | Teórico |
| 2014 | Herdeiro & Radu: Kerr con pelo escalar | Primera solución asintóticamente plana con pelo | Teórico |
| 2014 | Rezzolla & Zhidenko: parametrización RZ | Fracciones continuadas, convergencia superior | Teórico |
| 2015 | Berti et al. (56 autores): review tests GR | Enciclopedia del campo | Revisión |
| **2015** | **Detección GW150914** | **Primera fusión BBH observada** | **Datos reales** |
| 2016 | LVK O1 TGR: primer test de ringdown | Modo (2,2,0) consistente con Kerr; SNR ∼14 | Datos reales |
| 2016 | Cardoso, Franzin & Pani: ecos | Fenomenología de ECOs: búsqueda de ecos | Teórico |
| **2019** | **Giesler et al.: overtones** | Modelado desde el pico; paper que reencuadra el campo | Metodológico/NR |
| **2019** | **Isi et al.: test no-hair GW150914** | Reclamo de detección (2,2,0)+(2,2,1) a 3.6σ | Datos reales |
| 2019 | Carullo et al.: análisis temporal GW150914 | Sin evidencia de >1 modo; debate abierto | Datos reales |
| **2019** | **Isi, Chatziioannou & Farr: inferencia jerárquica** | Estándar metodológico para combinar eventos | Metodológico |
| 2019 | Cardoso & Pani: Living Review ECOs | Revisión comprensiva de objetos compactos exóticos | Revisión |
| 2019 | Zimmerman et al.: framework jerárquico | Unifica métodos anteriores como casos límite | Metodológico |
| 2020 | Maselli et al.: ParSpec | Parametrización del espectro QNM beyond-Kerr | Teórico/metodológico |
| 2021 | Jaramillo et al.: pseudoespectro e inestabilidad | Overtones altos inestables bajo perturbaciones | Teórico |
| 2021 | Ghosh, Brito & Buonanno: pSEOB con O3a | Restricciones parametrizadas con modelo IMR completo | Datos reales |
| 2021 | Moore et al.: sistemáticas de forma de onda | 10–30 eventos pueden producir evidencia espuria beyond-GR | Metodológico |
| **2022** | **Cotesta et al.: cuestionamiento del overtone** | Debate metodológico sobre sensibilidad a t₀ | Datos reales |
| **2023** | **Cheung et al. / Mitman et al.: QNMs no lineales** | Modos cuadráticos en NR; nuevo observable potencial | Teórico/NR |
| **2023** | **Capano et al.: dos modos en GW190521** | (2,2,0)+(3,3,0) con BF∼56; caso más sólido pre-2025 | Datos reales |
| **2024** | **Payne et al.: dependencia en curvatura** | Infiere exponente de escalado con masa; lo más cercano a inferencia de familia | Datos reales |
| **2025** | **GW250114: señal más fuerte jamás detectada** | SNR∼76; 3 modos; restricciones 2–3× GWTC-4 | Datos reales |
| 2025 | Berti et al. (68 autores): review espectroscopía BH | Estado del arte comunitario | Revisión |

---

## IV. Tabla de papers clave

| Referencia | Año | Tipo | Observables | Objeto inferido | Datos reales | Relación con geom. hiperbólica | Limitaciones |
|-----------|------|------|-------------|-----------------|:---:|-------------------------------|--------------|
| Dreyer et al. | 2004 | Metod. | f, τ QNM | M_f, χ_f consistencia | No | Ninguna | Requiere ≥2 modos; SNR futuro |
| Collins & Hughes | 2004 | Teórico | Multipolos | Deformaciones de Kerr | No | Ninguna | Perturbativo; solo deformaciones pequeñas |
| Berti, Cardoso & Will | 2006 | Metod. | δf, δτ | Desviaciones no-hair | No | Ninguna | Proyecciones para LISA |
| Vallisneri | 2008 | Metod. | Fisher matrix | Covarianzas paramétricas | No | Ninguna (espacio estadístico) | Aproximación gaussiana; falla a bajo SNR |
| Berti, Cardoso & Starinets | 2009 | Revisión | Espectro QNM completo | Tablas Kerr | No | Ninguna | Catálogo, no observacional |
| Yunes & Pretorius | 2009 | Teórico | Fase inspiral | Parámetros ppE | No | Ninguna | Solo inspiral; extensión a merger parcial |
| Guica et al. | 2009 | Teórico | Simetría conformal NHEK | Entropía BH extremal | No | **Conceptual** (AdS₂ es hiperbólico) | Solo BHs extremales; sin observable GW |
| Johannsen & Psaltis | 2011 | Teórico | Sombras, QNMs | Parámetros ε_i | No | Ninguna | No solución exacta de ecuaciones de campo |
| Johannsen | 2013 | Teórico | Constantes de movimiento | Separabilidad H-J | No | Ninguna | Patologías a alto espín |
| Rezzolla & Zhidenko | 2014 | Teórico | QNMs, sombras, ISCOs | Coeficientes f.c. | No | Ninguna | Rotación aproximada |
| Herdeiro & Radu | 2014 | Teórico | Momento cuadrupolar | BH Kerr con pelo | No | Ninguna | Modelos GW ausentes |
| Berti et al. 56 aut. | 2015 | Revisión | Multi-observable | Catálogo de tests GR | No | Ninguna | Enciclopédico |
| Cardoso, Franzin & Pani | 2016 | Teórico | Ecos post-ringdown | ECOs sin horizonte | No | Ninguna | Amplitud dependiente del modelo |
| Giesler et al. | 2019 | NR/Metod. | Overtones (2,2,n) | M_f, χ_f desde pico | No (NR) | Ninguna | Debate sobre validez perturbativa desde el pico |
| **Isi et al.** | **2019** | **Datos reales** | (2,2,0)+(2,2,1) GW150914 | Consistencia no-hair ∼20% | **Sí** | Ninguna | Debatido: sensibilidad a t₀; 3.6σ controvertido |
| Carullo et al. | 2019 | Datos reales | (2,2,0) GW150914 | Sin evidencia de >1 modo | Sí | Ninguna | Contradicción activa con Isi et al. |
| **Isi, Chatziioannou & Farr** | **2019** | **Metod.** | Posteriors δφ̂_i multi-evento | Hiperparámetros μ, σ | **Sí** | Ninguna | Coeficientes PN, no ringdown-only |
| Maselli et al. | 2020 | Teórico | ParSpec coeficientes | Desviaciones QNM vs espín | No | Ninguna | FIM; no aplicado a datos |
| **Ghosh, Brito & Buonanno** | **2021** | **Datos reales** | δf̂₂₂₀, δf̂₂₂₁ pSEOB | Consistencia Kerr O3a | **Sí** | Ninguna | No ringdown-only; débil en overtone |
| Jaramillo et al. | 2021 | Teórico | Pseudoespectro | Inestabilidad overtones | No | Ninguna | Limita información extraíble |
| Moore et al. | 2021 | Metod. | Sistemáticas forma de onda | Falsos positivos beyond-GR | No | Ninguna | Advertencia metodológica crítica |
| **Cotesta et al.** | **2022** | **Datos reales** | BF overtone GW150914 | Sin evidencia robusta | **Sí** | Ninguna | BF sensible a t₀ y prior |
| **Cheung et al.** | **2023** | **NR** | Modo cuadrático (220×220) | QNMs no lineales | No (NR) | Ninguna | No detectado en datos reales aún |
| **Capano et al.** | **2023** | **Datos reales** | (2,2,0)+(3,3,0) GW190521 | BF∼56 para dos modos | **Sí** | Ninguna | Evento individual de alta masa |
| **Payne et al.** | **2024** | **Datos reales** | Exponente p escalado M^p | Clase de corrección EFT | **Sí** | Ninguna | Un solo resultado; alcance limitado |
| Vanzo/Birmingham/Emparan | 1997–99 | Teórico | Termodinámica AdS | BH horizonte hiperbólico | No | **Directa** (horizonte es hiperbólico) | Puramente AdS; sin conexión con GW |
| **LVK GW250114** | **2025** | **Datos reales** | (2,2,0), (2,2,1), (4,4,0) | 3 modos; Kerr a ∼% | **Sí** | Ninguna | Test más estricto a la fecha |

---

## V. Estado del arte con datos reales

### GW150914 — El evento de referencia con historia

**Detección robusta:** El modo (2,2,0) se detecta consistentemente en todas las metodologías. Las medidas de M_f (∼68 M☉) y χ_f (∼0.67) son reproducibles. Las restricciones son δf₂₂₀ = −0.05⁺⁰·²⁰₋₀.₂₀ y δτ₂₂₀ = 0.1⁺⁰·⁵₋₀.₄ aproximadamente (dependientes de la metodología).

**Evidencia debatida:** El reclamo de Isi et al. (2019) de detección del overtone (2,2,1) a 3.6σ está activamente contestado. Cotesta et al. (2022) muestran que el factor de Bayes para el overtone es sensible al tiempo de inicio t₀: para t₀ = t_peak el BF es significativo, pero para t₀ = t_peak + δt el BF cae sustancialmente. El debate metodológico sobre cuándo empieza el régimen perturbativo del ringdown (¿desde el pico de amplitud? ¿varios milisegundos después?) es la raíz del desacuerdo.

**Simple cota:** No hay evidencia de ecos post-ringdown. No hay evidencia de modos más allá del (2,2,0) con significancia no debatida.

### GW190521 — Espectroscopía multi-modal emergente

**Detección con caveats:** Capano et al. (2023) reportan evidencia de dos modos — (2,2,0) y (3,3,0) — con BF∼56 usando análisis de ringdown en el pico. La asimetría de masa (q ≈ 0.79) hace el modo (3,3,0) significativo. Este es el caso más sólido de espectroscopía multi-modo antes de GW250114, pero corresponde a un evento individual y de alta masa (M_final ∼ 250 M☉), con una señal que cae parcialmente por debajo de la banda sensible de LIGO/Virgo. La comunidad no considera este resultado definitivo.

### GW250114 — El punto de inflexión observacional (2025)

**Detección robusta de tres modos:** LVK et al. identificaron los modos (2,2,0), (2,2,1) y (4,4,0) en GW250114, con restricciones individuales sobre desviaciones parametrizadas que superan a las del catálogo GWTC-4 combinado. El modo (2,2,1) aparece a >3σ, lo que potencialmente resuelve el debate de GW150914. Las restricciones sobre desviaciones en frecuencia del modo fundamental son las más estrictas obtenidas hasta la fecha con un evento individual.

**Caveats:** Aún pendiente de revisión comunitaria amplia (paper circulando desde septiembre 2025). Los QNMs no lineales predichos por Cheung et al. (2023) no han sido buscados sistemáticamente en GW250114 con metodología publicada.

### Catálogos combinados O1+O2+O3

Los tests de GR con GWTC-3 combinan ∼70 eventos BBH para coeficientes PN y ∼12 eventos para restricciones de ringdown, encontrando consistencia con GR en todas las combinaciones. Las restricciones combinadas de ringdown son del orden de ±10% (δf) y ±20% (δτ). El test de consistencia IMR ha sido aplicado en formulaciones bidimensional y tetradimensional, encontrando en todos los casos consistencia con la predicción de GR.

**Búsquedas de ecos:** Resultados nulos en todos los catálogos LVK (GWTC-1, GWTC-2, GWTC-3). Los reclamos iniciales de Abedi et al. (2017) con GW150914 a 2.5σ no han sido replicados por análisis LVK independientes.

---

## VI. Estado del arte multi-evento y jerárquico

### Estándar actual

La inferencia jerárquica bayesiana de Isi, Chatziioannou & Farr (2019) modela la distribución poblacional de los parámetros de desviación como gaussiana con hiperparámetros μ (media) y σ (varianza), permitiendo agregar información de múltiples eventos con distinta señal-a-ruido de forma coherente.

Estudios comparativos entre la multiplicación de factores de Bayes y la inferencia jerárquica han mostrado que el primero puede producir resultados erróneos cuando los priors son mucho más anchos que la señal, mientras que la inferencia jerárquica converge genéricamente a la conclusión correcta.

GWTC-3 aplica este framework a ∼50 eventos para coeficientes PN, alcanzando consistencia con GR a nivel de ∼1% en los coeficientes mejor restringidos.

### Lo que se ha hecho una vez o pocas veces

Payne et al. (2024) demuestran que la dependencia en curvatura de los tests de GR — donde se espera que correcciones de gravedad modificada escalen con la masa del sistema elevada a algún exponente p — puede inferirse directamente de los datos LVK, obteniendo el primer resultado observacional sobre la clase de correcciones gravitacionales.

Estudios de sesgos de selección en tests de GR multi-evento han mostrado que los efectos de selección son secundarios respecto a las restricciones individuales para los niveles de desviación actuales.

Moore et al. (2021) identificaron que las sistemáticas de forma de onda pueden generar evidencia espuria de violaciones de GR al combinar 10–30 eventos, estableciendo que el techo del campo es la exactitud del modelado, no la estadística.

### Lo que no se ha hecho

- Inferencia jerárquica dedicada de desviaciones de ringdown-only (δf, δτ) combinando múltiples eventos. Todos los análisis de catálogo combinan coeficientes de fase del inspiral, donde la SNR es mucho más alta.
- Inferencia de familias geométricas o clases de métricas a nivel poblacional.
- Aplicación de ParSpec (Maselli et al. 2020) a datos reales de catálogo.
- Combinación de restricciones multi-modo entre eventos (e.g., consistencia del ratio f₂₂₁/f₂₂₀ entre GW250114 y otros).

---

## VII. Evaluación crítica

### Lo que está maduro

El test de consistencia IMR (M_f, χ_f entre inspiral y ringdown) es metodológicamente sólido, reproducible y adoptado como estándar en los análisis LVK. El test nulo del modo (2,2,0) con desviaciones parametrizadas (δf, δτ) tiene base teórica clara, implementación estable en múltiples pipelines, y resultados observacionales reproducibles. La inferencia jerárquica bayesiana para combinar múltiples eventos está formalizada y su comportamiento asintótico es bien comprendido. Las tablas de frecuencias QNM de Kerr (Berti, Cardoso & Starinets 2009) como referencia de predicción son fiables al nivel de precisión requerido por los detectores actuales.

### Lo que está en desarrollo activo

La espectroscopía multi-modo con eventos reales (GW190521, GW250114) está emergiendo de forma creíble. La teoría de QNMs no lineales (Cheung et al. 2023) y su eventual conexión con observables en datos es un frente activo. Los modelos IMR en teorías alternativas específicas (EdGB, dCS) están progresando pero son aún perturbativos en espín. La inferencia del exponente de escalado de curvatura (Payne et al. 2024) abre una dirección nueva pero con un solo resultado publicado.

### Lo que no está formulado de forma identificable

No existe un framework operacional para inferir "la familia geométrica" de un agujero negro a partir de sus QNMs observados. El problema es sub-determinado: finitos modos observados con precisión del 10–20% no determinan una métrica de forma única. El mapa QNM → potencial efectivo → métrica es bien definido matemáticamente (Völkel & Kokkotas 2017–2019) pero requiere: (a) el espectro completo, (b) la diferencia entre isospectral y no-isospectral, y (c) precisión mucho mayor que la actual. El paso "geometría hiperbólica" no tiene ninguna formulación identificable conectada con observables de GW astrofísicas.

La geometría de la información del espacio de modelos (Fisher manifolds) tampoco está formulada como herramienta para clasificar familias de teorías gravitacionales con datos de GW. Existe la maquinaria matemática (FIM para GW, Vallisneri 2008; sloppy models, Transtrum et al.), pero no existe el programa ni el paper que los conecte de forma operacional.

### Focos de riesgo de overclaim

1. **Consistencia con Kerr → geometría Kerr:** error de afirmación. Un resultado nulo en δf no identifica la geometría; solo fija una cota.
2. **ParSpec restringido → familia geométrica identificada:** confusión entre espacio de parámetros de desviación y espacio de geometrías.
3. **Curvatura del espacio Fisher → curvatura del espaciotiempo:** error categorial severo.
4. **"Geometría hiperbólica" en cualquier sentido → conclusión observacional:** ver sección C completa.
5. **Ecos no descartados → evidencia de ECOs:** ausencia de refutación ≠ evidencia positiva.

---

## VIII. Taxonomía rigurosa del término "hiperbólico"

Este término aparece en cinco contextos completamente distintos. Su confusión genera afirmaciones científicamente incorrectas y es una fuente activa de overclaim. La regla operativa es: **clasificar cualquier uso en una de estas cinco categorías antes de sacar conclusión alguna.**

### Categoría 1: Topología/geometría física del horizonte en AdS

**Qué es:** Agujeros negros en espacio-tiempo Anti-de Sitter (AdS) cuyo horizonte tiene topología hiperbólica (curvatura seccional −1), en contraposición con la esférica (Vanzo 1997, Birmingham 1999, Emparan 1999). Estas soluciones satisfacen las ecuaciones de Einstein con constante cosmológica Λ < 0.

**Conexión con observaciones GW astrofísicas:** **Inexistente.** Los agujeros negros astrofísicos observados por LIGO/Virgo viven en espaciotiempo asintóticamente plano (Λ ≈ 0). Los agujeros negros en AdS son construcciones holográficas relevantes para materia fuertemente acoplada en física de alta energía. No hay propuesta observacional que conecte la topología del horizonte AdS con señales GW medibles en la Tierra.

### Categoría 2: Estructura near-horizon extremal — geometría NHEK

**Qué es:** En el límite de Kerr extremal (a → M, χ_f → 1), la geometría near-horizon se denomina NHEK (Near-Horizon Extremal Kerr, Bardeen & Horowitz 1999) y contiene un factor AdS₂ × S², lo que le confiere estructura hiperbólica bidimensional. La correspondencia Kerr/CFT (Guica et al. 2009) explota esta simetría para calcular la entropía del agujero negro extremal a través de la fórmula de Cardy.

**Conexión con observaciones GW astrofísicas:** **Conceptual únicamente.** Los agujeros negros remanentes observados tienen espines χ_f ∼ 0.6–0.7, muy lejos de la extremalidad. El límite NHEK requiere χ_f → 1. Las propuestas de observación de geometría NHEK son electromagnéticas (NHEKline en espectroscopía de rayos X, anillo de fotones en EHT para BH en rotación rápida; Gralla, Lupsasca & Strominger 2017), no gravitacionales. **No existe propuesta observacional GW concreta que involucre la estructura NHEK.**

### Categoría 3: Hiperbolicidad dinámica — exponente de Lyapunov de geodésicas nulas

**Qué es:** Las geodésicas nulas circulares inestables alrededor de agujeros negros poseen un exponente de Lyapunov λ que mide la tasa de divergencia. La hiperbolicidad aquí se refiere al carácter hiperbólico de las ecuaciones de movimiento (valores propios reales del lineal), no a la curvatura espacial. En el régimen de frecuencias QNM grandes (eikonal limit), existe una conexión: ω_QNM ≈ Ω_c ℓ − i(n + 1/2)|λ|, donde Ω_c es la frecuencia angular de la geodésica circular (Cardoso et al. 2009).

**Conexión con observaciones GW astrofísicas:** **Indirecta.** La relación eikonal QNM-geodésica es exacta solo en el límite ℓ → ∞. Para los modos dominantes (ℓ = 2, 3) es una aproximación con errores del orden de 10–20%. No es directamente observable en datos actuales como una "geometría hiperbólica" separada. Su conexión con las frecuencias QNM observadas es parcialmente verificable pero solo como consistencia, no como inferencia independiente.

### Categoría 4: Geometría estadística — espacio de parámetros / information geometry

**Qué es:** La Fisher Information Matrix (FIM) define una métrica en el espacio de parámetros del modelo (espacio estadístico). Si esa métrica tiene curvatura seccional negativa constante, el espacio es hiperbólico en sentido Riemanniano — el "Método Brunete" de BASURIN explota precisamente este resultado para el espacio log(f)/log(Q). La geometría de Ruppeiner aplica geometría diferencial a termodinámica de agujeros negros.

**Conexión con observaciones GW astrofísicas:** **Ninguna como geometría del espaciotiempo.** La curvatura del espacio de parámetros estadísticos es una propiedad de la distribución de probabilidad del modelo, no de la geometría física del agujero negro. Decir que "el espacio de parámetros tiene curvatura negativa" no implica ninguna afirmación sobre la curvatura del horizonte del agujero negro. Esta distinción es fundamental y su confusión es un error categorial que invalida cualquier conclusión sobre la naturaleza del agujero negro.

**Nota:** La geometría hiperbólica del espacio de parámetros (Categoría 4) puede ser matemáticamente interesante y operativamente útil para el diseño de algoritmos de inferencia (e.g., distancias de Mahalanobis, métricas de compatibilidad). Pero no tiene implicaciones físicas directas sobre la geometría del agujero negro observado.

### Categoría 5: Encuentros hiperbólicos

**Qué es:** Órbitas de dos cuerpos compactos con energía positiva (no ligadas), descritas por hipérbolas en el límite kepleriano. Producen señales de burst de GW. Las colaboraciones LIGO/Virgo han buscado estas señales sin detecciones robustas hasta la fecha.

**Conexión con observaciones GW de ringdown:** **Ninguna relevante.** Los encuentros hiperbólicos no producen un remanente que oscile en modos QNM. La dinámica post-encuentro no genera ringdown de agujero negro.

### Tabla resumen de categorías

| Categoría | Noción de "hiperbólico" | Conexión con observaciones GW | Riesgo de confusión |
|-----------|------------------------|-------------------------------|---------------------|
| 1 | Topología horizonte en AdS | **Inexistente** | Alto: el término "agujero negro hiperbólico" suena a astrofísico |
| 2 | Estructura NHEK / AdS₂ near-horizon | **Conceptual** (requiere extremalidad) | Medio: sí involucra BH en rotación |
| 3 | Lyapunov geodésicas nulas | **Indirecta** (límite eikonal de QNMs) | Bajo: bien separado en la literatura |
| 4 | Curvatura espacio de parámetros (Fisher) | **Ninguna sobre geometría del BH** | **Muy alto: confusión frecuente en BASURIN** |
| 5 | Encuentros hiperbólicos | Ninguna con ringdown | Bajo |

---

## IX. Respuestas a las preguntas concretas del prompt

**Pregunta 1: ¿Qué ha medido realmente la comunidad con datos de LIGO/Virgo/KAGRA en ringdown?**
Ha medido las frecuencias y tiempos de amortiguamiento del modo fundamental (2,2,0) en docenas de eventos. Ha medido (con debate) el overtone (2,2,1) en GW150914 y de forma más sólida en GW250114. Ha medido el modo angular (3,3,0) en GW190521. Ha combinado estas medidas para producir restricciones sobre δf y δτ a nivel de catálogo. Todo esto es consistente con Kerr.

**Pregunta 2: ¿Qué modos han sido realmente detectados o acotados?**
- **(2,2,0):** Detectado robustamente en todos los eventos bien caracterizados. Estándar de campo.
- **(2,2,1):** Evidencia debatida en GW150914; evidencia significativa (>3σ) en GW250114 (2025).
- **(3,3,0):** Evidencia con BF∼56 en GW190521 (Capano et al. 2023).
- **(4,4,0):** Reportado en GW250114 (2025).
- **Modos angulares más altos, overtones n≥2:** Solo cotas superiores o análisis proyectivos.
- **QNMs no lineales:** Solo en simulaciones NR; no detectados en datos.
- **Ecos:** Búsquedas con resultado nulo en todos los catálogos.

**Pregunta 3: ¿Qué eventos son hitos reales?**
GW150914 por ser el primero y el más analizado. GW190521 por la primera evidencia creíble de dos modos angulares. GW250114 por el SNR más alto registrado y la confirmación del overtone.

**Preguntas 4–5: Consistencia y precisión.**
Total consistencia con Kerr. Precisión: ∼10% en δf (combinado de catálogo), ∼20% en δτ. GW250114 individual: ∼5% en δf del modo fundamental.

**Pregunta 6: ¿Con qué enfoque trabaja la comunidad?**
Dominante (>80% de los análisis con datos reales): tests nulos de consistencia con desviaciones parametrizadas. Secundario: métricas concretas de teorías específicas (solo theoretical/methodological papers). Minoritario: clases efectivas de modelos. La expresión "inferencia geométrica directa" no corresponde a ningún análisis publicado con datos reales.

**Preguntas 7–9: Parametrizaciones dominantes y aplicación a datos.**
pSEOBNR/pIMRPhenom dominan el análisis de datos LVK. ParSpec (Maselli et al.) es el framework más adecuado para ringdown beyond-Kerr pero no ha sido aplicado a datos reales. Las métricas concretas (JP, Johannsen, EdGB, dCS) no se usan rutinariamente en datos reales.

**Preguntas 10–12: Ruta hacia geometrías hiperbólicas.** Véase Sección VIII completa. La respuesta honesta y precisa es: no existe.

**Preguntas 13–17: Inferencia multi-evento.** Véase Sección VI. La comunidad combina jerárquicamente coeficientes PN de ∼50 eventos y desviaciones de ringdown de ∼12. No ha intentado inferencia de familias geométricas a nivel poblacional. El resultado más cercano es Payne et al. (2024).

**Preguntas 18–20: Cuellos de botella y rutas.**

Los cuellos de botella físicos y estadísticos son los siguientes. El SNR post-merger es la restricción instrumental primaria: típicamente 5–15 para fusiones en O3, insuficiente para espectroscopía multi-modo robusta. Las degeneracias entre modos — especialmente entre el modo fundamental y el overtone — hacen difícil separar contribuciones cuando el overtone decae 3–5× más rápido. El tiempo de inicio del ringdown t₀ es el cuello de botella metodológico principal en la controversia GW150914: no existe criterio independiente para determinar cuándo el régimen perturbativo es válido. Las sistemáticas de forma de onda (Moore et al. 2021) constituyen el techo a largo plazo para análisis de catálogo. La no-unicidad inversa del espectro QNM (isospectralidad) es un límite fundamental: distintas geometrías pueden producir espectros QNM idénticos o indistinguibles. La inestabilidad pseudoespectral (Jaramillo et al. 2021) muestra que los overtones altos son intrínsecamente inestables bajo perturbaciones del operador. Y la falta de modelos IMR completos en teorías beyond-GR impide la comparación directa en análisis de datos.

---

## X. Recomendación final

### Ruta científicamente defendible

**A corto plazo (datos actuales O4/O5):**
- Análisis de ringdown multi-modo de eventos con SNR post-merger >20 (GW250114 y futuros equivalentes), reportando restricciones sobre los modos individuales detectados con sus niveles de significancia.
- Combinación jerárquica de δf₂₂₀ y δτ₂₂₀ de múltiples eventos usando el framework estándar (Isi et al. 2019), produciendo posteriors conjuntas sobre (μ, σ).
- Extensión de la inferencia del exponente de escalado (Payne et al. 2024) para discriminar entre clases de correcciones de curvatura a nivel poblacional.
- Tests del ratio f₂₂₁/f₂₂₀ (que en Kerr varía solo ∼0.93–0.99 con el espín) como constrainte adicional, siempre que el overtone esté detectado con significancia suficiente.

**A medio plazo (ET/CE):**
- Espectroscopía de precisión con >3 modos por evento rutinariamente.
- Aplicación de ParSpec a muestras de eventos para restricciones poblacionales sobre coeficientes de teorías específicas.
- Detección potencial de QNMs no lineales (cuadráticos), que tienen predicciones específicas en Kerr verificables independientemente.

### Ruta prematura o no soportada por la literatura

Cualquiera de las siguientes sería científicamente indefendible con datos actuales o a corto plazo:
- Afirmar que datos de ringdown revelan la "geometría" del agujero negro más allá de "consistente con Kerr".
- Afirmar que existe evidencia observacional de estructura hiperbólica/AdS near-horizon en agujeros negros astrofísicos.
- Traducir la curvatura del espacio Fisher en curvatura del espaciotiempo del agujero negro.
- Declarar "discriminación entre familias geométricas" a partir de un análisis de catálogo con el modo fundamental solamente.
- Afirmar que ParSpec aplicado a datos reales (cuando se haga) "identifica una métrica alternativa" en lugar de "restringe parámetros de desviación".
- Afirmar que la población de geometrías compatibles con los datos LVK excluye Kerr en favor de métricas específicas, salvo que se haya realizado selección de modelos bayesiana formal con modelos IMR completos en esas teorías.

---

## XI. Traducción operativa a un pipeline tipo BASURIN

### Quantities que sí tendría sentido agregar multi-evento

**Canónicas (listas para producción):**
- **δf₂₂₀ y δτ₂₂₀ por evento → posterior conjunta (μ, σ):** Usando el framework jerárquico de Isi et al. (2019). Output: posterior sobre la media poblacional y la dispersión de las desviaciones. Interpretación: si μ ≈ 0 y σ → 0 a medida que se acumulan eventos, esto es evidencia de que el conjunto de agujeros negros es consistente con Kerr. Completamente auditable.
- **Consistencia M_f/χ_f entre inspiral y ringdown, agregada multi-evento:** Posterior conjunta sobre (ΔM_f/M̄_f, Δχ_f/χ̄_f) del test de consistencia IMR. Estándar LVK, reproducible.
- **Ratio f₂₂₁/f₂₂₀ para eventos con overtone detectado:** Varía solo ∼0.93–0.99 en Kerr. Si se detecta el overtone con significancia suficiente, el ratio es una restricción adicional sobre desviaciones. Auditable si se declara el umbral de detección del overtone.

**Experimental (justificable pero no estándar):**
- **Exponente de escalado p en correcciones M^p:** Siguiendo Payne et al. (2024). Requiere modelar la dependencia en masa de las desviaciones. Implementable pero con un único paper de referencia; debe marcarse como experimental.
- **Distancia de Mahalanobis en el espacio (f₂₂₀, τ₂₂₀) para comparación con atlas de métricas:** Si existe un atlas de predicciones QNM de métricas específicas (EdGB, dCS, Bardeen, Hayward) con cálculos publicados de QNMs, la distancia Mahalanobis entre el punto observado y cada modelo del atlas es una métrica de compatibilidad bien definida. **Advertencia crítica:** esto mide compatibilidad espectral, no identifica la métrica. Debe reportarse como "consistente con / inconsistente con la predicción QNM de [modelo]", nunca como "el agujero negro tiene geometría [modelo]".
- **Coeficientes ParSpec combinados entre eventos:** Si se implementa ParSpec con datos reales, la combinación jerárquica de coeficientes es la extensión natural. Aún sin referencia de datos reales.

### Outputs auditables sin sobre-interpretación

| Output | Auditable | Condición |
|--------|:---------:|-----------|
| Posterior (μ, σ) de δf, δτ multi-evento | ✓ | Declarar prior, t₀, modelo de waveform |
| Factor de Bayes overtone detectado vs. no detectado | ✓ | Declarar t₀ y prior sobre amplitudes relativas |
| Posterior sobre exponente de escalado p | ✓ | Declarar modelo de población y prior en p |
| Distancia Mahalanobis al atlas | ✓ (con caveats) | Declarar que es compatibilidad espectral, no identificación geométrica |
| "Fracción de la población compatible con Kerr" | Solo si | Requiere modelo forward completo con selección effects |

### Quantities que deben quedar marcadas como experimentales

- Cualquier restricción sobre parámetros de métricas concretas (ε_i de JP, α de EdGB) derivada de análisis de ringdown individual o combinado: las predicciones QNM en estas teorías son perturbativas en espín, con errores del 10–50% para espines típicos.
- Coeficientes ParSpec en ausencia de paper de aplicación a datos reales.
- Cualquier cantidad derivada del pseudoespectro o de inestabilidad espectral.

### Afirmaciones que deben prohibirse por falta de identificabilidad

1. **"El agujero negro tiene geometría [X]"** para cualquier X beyond-Kerr. Afirmación no identificable con datos de ringdown actuales ni proyectados a corto plazo.
2. **"Los datos favorecen [teoría A] sobre [teoría B]"** sin selección de modelos bayesiana formal con modelos IMR completos en ambas teorías.
3. **"La curvatura del espacio de parámetros indica estructura hiperbólica del agujero negro"**: error categorial entre geometría estadística y geometría del espaciotiempo.
4. **"La espectroscopía poblacional ha mapeado la familia de métricas"**: ningún pipeline existente ha hecho esto.
5. **"Los ecos están acotados pero no excluidos, lo que sugiere estructura ECO"**: ausencia de refutación no es evidencia positiva.
6. **"El conjunto de geometrías compatibles con los datos excluye Kerr"**: contrario a todos los resultados publicados. Solo sería aceptable si se realizara un test bayesiano formal con modelo alternativo específico.

### Diseño mínimo metodológicamente serio para un pipeline de ringdown multi-evento

Un pipeline de ringdown multi-evento methodológicamente serio debe incorporar los siguientes elementos. Como análisis por evento, debe estimar posteriors sobre δf₂₂₀ y δτ₂₂₀ usando un modelo de forma de onda calibrado (pSEOBNR o equivalente), y reportar el factor de Bayes para modelos con y sin overtone cuando el SNR post-merger lo justifique. Como criterio de selección de eventos, debe definir un umbral de SNR post-merger mínimo (recomendado: SNR >8 para análisis de modo fundamental, >15 para análisis multi-modo) con justificación explícita. Como agregación multi-evento, debe implementar la inferencia jerárquica con hiperparámetros (μ, σ) de Isi et al. (2019), no la multiplicación de factores de Bayes. Como control de robustez, debe incluir tests de sensibilidad al tiempo de inicio t₀, al prior sobre amplitudes relativas de modos, y al modelo de forma de onda. Como tratamiento de incertidumbres sistemáticas, debe propagar incertidumbres de la forma de onda (no solo estadísticas) en los resultados finales. Finalmente, como declaración epistémica explícita, todos los outputs deben incluir la distinción entre "consistente con Kerr" (inferencia alcanzable) e "identificación de la geometría" (no identificable con los observables disponibles), con esta distinción documentada en el contrato de cada etapa del pipeline.
