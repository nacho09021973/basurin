# Base teórica para sectorización geométrica y observable de Rényi en BASURIN

**Informe técnico — Marzo 2026**

---

## 1. Veredicto corto

**Sectorización HYPERBOLIC / ELLIPTIC / EUCLIDEAN:** Base teórica **parcial pero indirecta**. Existen resultados rigurosos en geometría de horizontes y en carácter de las ecuaciones de campo que utilizan exactamente esos tres términos, pero no del modo que el programa supone. El mapeo family → sector no es estándar, no es unívoco y requiere una redefinición sustancial para ser defendible.

**Rényi sobre el ensemble discreto de soluciones compatibles:** Base teórica **directa como observable informacional/epistémico**, siempre que se formule explícitamente como índice de diversidad/concentración sobre una distribución discreta de modelos compatibles (framework de números de Hill / diversidad efectiva) y **no** se presente como entropía termodinámica, holográfica ni gravitacional.

---

## 2. Sectorización geométrica

### 2.1. Tres sentidos distintos de "hiperbólico/elíptico/euclídeo" en física de agujeros negros

Los términos "hiperbólico", "elíptico" y "euclídeo" aparecen en la literatura de agujeros negros con al menos tres significados completamente distintos. Confundirlos constituye un error categorial. Los tres son:

**A. Geometría de la sección del horizonte y su embedding**

El resultado fundacional es de Smarr (1973, Phys. Rev. D 7, 289): la sección espacial del horizonte de Kerr (una 2-superficie) tiene curvatura gaussiana que depende del spin. Para spin adimensional j < √3/2 ≈ 0.866, la curvatura gaussiana es positiva en todo el horizonte y la superficie admite un embedding isométrico global en el espacio euclídeo tridimensional E³. Para j > √3/2, la curvatura gaussiana se vuelve negativa en los polos y el embedding en E³ falla globalmente.

Gibbons, Herdeiro y Rebelo (2009, PRD 80, 044014; arXiv:0906.2768) resolvieron este problema demostrando que el horizonte de Kerr-Newman admite un embedding isométrico global en el espacio hiperbólico tridimensional H³ para *cualquier* valor del momento angular. En H³ la curvatura negativa del espacio ambiente "absorbe" la curvatura negativa local de la superficie.

Esto significa que, **dentro de la familia Kerr**, existe una transición geométrica genuina parametrizada por el spin:

- j < √3/2: horizonte embebible en E³ → "euclídeo"
- j > √3/2: horizonte requiere H³ → "hiperbólico"
- j = 0 (Schwarzschild): horizonte esférico, embebible idénticamente en E³ y en H³

Herdeiro et al. (2018, arXiv:1804.04910) extendieron este análisis a agujeros negros Kerr con pelo escalar sincronizado, mostrando que la línea de Smarr (el umbral de embeddability) se desplaza según el "hairiness" de la solución.

**B. Carácter de las ecuaciones de campo (hiperbolicidad del problema de valor inicial)**

Ripley y Pretorius (2019, PRD 99, 084014; arXiv:1902.01468) demostraron numéricamente que en gravedad Einstein-dilatón-Gauss-Bonnet (EdGB), las ecuaciones de campo pueden cambiar dinámicamente de carácter hiperbólico a elíptico. Concretamente: para acoplamientos suficientemente fuertes, se forman regiones del espaciotiempo donde el sistema de EDPs deja de ser hiperbólico y se vuelve elíptico, haciendo que el problema de Cauchy esté mal puesto (ill-posed).

En su trabajo posterior (2020, PRD 101, 044015; arXiv:1911.11027) mostraron que para agujeros negros escalarizados en EdGB con acoplamientos pequeños, la región elíptica se forma *dentro* del horizonte y queda censurada; pero para acoplamientos "superextremales", la región elíptica aparece *fuera* del horizonte, invalidando la evolución exterior.

Este resultado es crucial: la clasificación hiperbólico/elíptico aquí se refiere al carácter del sistema de EDPs de la teoría, no a la geometría intrínseca del horizonte. Es una propiedad de la **teoría de gravedad** (y del régimen de acoplamiento), no de la "solución" individual como geometría estática.

Para dCS (Chern-Simons dinámico), la situación es diferente: las soluciones de agujeros negros se conocen perturbativamente sobre Kerr y no se ha documentado una pérdida análoga de hiperbolicidad en el régimen perturbativo estándar. Para Kerr-Newman en GR, el sistema es siempre hiperbólico.

**C. Topología del horizonte**

El teorema de Hawking sobre topología de horizontes (Hawking 1972, Commun. Math. Phys. 25, 152; generalizado por Galloway y Schoen 2006, Commun. Math. Phys. 266, 571) establece que en 4D, asintóticamente plano y bajo la condición de energía dominante, las secciones del horizonte de sucesos son topológicamente esferas S². Los horizontes con topología hiperbólica (superficies de género ≥ 2 con curvatura negativa constante) **solo existen** en presencia de constante cosmológica negativa (espaciotiempos asintóticamente AdS) o en dimensiones superiores.

Para todas las familias del atlas de BASURIN (Kerr, EdGB, Kerr-Newman, dCS) en 4D asintóticamente plano: los horizontes son topológicamente S². La topología no sirve como discriminador.

### 2.2. Evaluación por familia

**Kerr:** La única clasificación geométrica intrínseca bien fundamentada es el umbral de Smarr: j < √3/2 vs j > √3/2, basado en el signo de la curvatura gaussiana del horizonte en los polos. Esto es una propiedad **por solución** (depende del spin), no de la familia completa.

**EdGB:** Las soluciones astrofísicamente relevantes son perturbaciones de Kerr. Su horizonte sigue siendo topológicamente S² y geométricamente cercano al de Kerr. La distinción hiperbólico/elíptico relevante es la del carácter de las EDPs (sentido B), que depende del acoplamiento α y de la masa. No existe un invariante geométrico del horizonte que clasifique soluciones EdGB en "sectores" distintos de los de Kerr de forma estándar en la literatura.

**Kerr-Newman:** Es una solución exacta de Einstein-Maxwell. El horizonte es S², la curvatura gaussiana depende tanto del spin como de la carga. El análisis de embedding de Smarr se aplica directamente (Frolov 2006, PRD 73, 064021; Gibbons et al. 2009). La presencia de carga modifica los umbrales cuantitativamente pero no introduce un "sector" cualitativamente nuevo.

**dCS:** Las soluciones se conocen perturbativamente. En el régimen perturbativo, la geometría del horizonte difiere de Kerr a orden del parámetro de acoplamiento. No existe en la literatura una clasificación de tipo "sector geométrico" para dCS. La principal diferencia es la violación de paridad (la corrección es impar bajo reflexión), no un cambio de signo de curvatura.

### 2.3. Invariantes concretos que podrían sustentar una sectorización

Los candidatos más sólidos, todos por solución individual y no por familia, son:

1. **Curvatura gaussiana del horizonte en los polos** (signo): es el invariante que sustenta el umbral de Smarr. Computable para cualquier métrica estacionaria axisimétrica. Para Kerr es K_polo = (1 − 3a²/r_+²) / r_+², donde r_+ es el radio del horizonte.

2. **Invariantes de Zakhary-McIntosh / Abdelqader-Lake** (2015, PRD 91, 084017): proporcionan una caracterización invariante del espaciotiempo de Kerr mediante invariantes construidos a partir del tensor de Weyl. Permiten medir la "Kerrness" localmente. Son invariantes escalares del espaciotiempo completo, no solo del horizonte.

3. **Esferificidad** (spheroidicity): definida por Herdeiro et al. (2018) para cuantificar la deformación del horizonte, continua y no clasificatoria.

Ninguno de estos produce una clasificación discreta en tres sectores del tipo propuesto. Producen funciones continuas de los parámetros de la solución.

### 2.4. El mapeo es por solución, no por familia

**Conclusión firme:** Cualquier clasificación geométrica basada en invariantes del horizonte o del espaciotiempo es necesariamente *por solución individual* (depende de M, χ, Q, α_EdGB, etc.), no por familia teórica. Dentro de la familia Kerr hay soluciones "euclídeas" (j < 0.866) e "hiperbólicas" (j > 0.866). Asignar un único sector a "family = kerr" carece de base.

### 2.5. ¿Es un error categorial?

La sectorización propuesta mezcla al menos dos de los tres sentidos descritos arriba:

- El sentido A (embedding del horizonte) produce una clasificación real pero **intra-familia**, dependiente de los parámetros de la solución.
- El sentido B (carácter de las EDPs) es una propiedad de la **teoría**, no de soluciones estáticas individuales, y su relevancia observacional para el análisis de ringdown no está establecida.
- El sentido C (topología) es uniforme para todo el atlas (todo es S²) y no discrimina.

Usar los términos HYPERBOLIC/ELLIPTIC/EUCLIDEAN como si fueran etiquetas fijas por familia teórica es, en rigor, un error categorial — a menos que se redefina explícitamente qué se entiende por esos términos en el contexto del atlas.

### 2.6. Conexión con el Método Brunete

Existe un cuarto sentido que sí es relevante para BASURIN: la geometría de la información (Fisher) del espacio de observables de ringdown. Si el Método Brunete establece que la métrica de Fisher en el espacio log(f)/log(Q) tiene curvatura negativa constante (geometría hiperbólica), entonces la "hiperbolicidad" se refiere a la variedad de parámetros observacionales, no al espaciotiempo de las soluciones.

Este sentido informacional es legítimo y está bien fundamentado (la geometría de la información de Fisher es un campo maduro: Amari 1985, Ay et al. 2017). Pero es completamente distinto de clasificar *soluciones* como hiperbólicas o elípticas. Si lo que se quiere decir es "estas soluciones viven en una región del espacio observacional con curvatura negativa de Fisher", eso tiene sentido. Pero no justifica etiquetar a las soluciones mismas con un "sector geométrico" del espaciotiempo.

---

## 3. Rényi sobre el espacio de soluciones

### 3.1. Qué teoría sí apoya algo parecido

**Base directa: números de Hill y diversidad efectiva.** La entropía de Rényi de orden q sobre una distribución discreta {p₁, ..., pₙ} tiene una interpretación bien establecida y no abusiva como medida de *diversidad efectiva* del ensemble:

H_q = (1/(1−q)) · log(Σᵢ pᵢ^q)

El exponencial de la entropía de Rényi, D_q = exp(H_q), se conoce como el "número efectivo de tipos" o "número de Hill" (Hill 1973, Ecology 54, 427; Jost 2006, Oikos 113, 363). D_q es el número de categorías igualmente abundantes que producirían el mismo valor de Rényi. Para q = 0, D₀ = n (riqueza). Para q → 1, D₁ = exp(H_Shannon). Para q = 2, D₂ = 1/Σpᵢ² (inverso del índice de Simpson).

Aplicado al atlas de BASURIN: si se define una distribución de pesos {pᵢ} sobre las geometrías compatibles (por ejemplo, proporcional a alguna métrica de compatibilidad normalizada), entonces:

- H_q mide la **concentración o dispersión** de la compatibilidad sobre el atlas.
- D_q da el **número efectivo de geometrías** realmente informativas.
- El perfil q → D_q diagnostica si la compatibilidad está concentrada en pocas soluciones (D_q bajo para q grande) o distribuida (D_q alto).

Esto es matemáticamente riguroso, bien interpretable y **no requiere ninguna apelación a física de agujeros negros**. Es pura teoría de la información sobre distribuciones discretas.

**Base parcial: entropía de Rényi en hypothesis testing bayesiano.** Sason y Verdú (2018, IEEE Trans. Inf. Theory 64, 4) establecieron cotas superiores e inferiores sobre la probabilidad de error mínima en M-ary hypothesis testing bayesiano en función de la entropía condicional de Arimoto-Rényi. Esto proporciona una conexión formal entre Rényi y la capacidad de discriminación entre modelos. Aplicado a BASURIN: la Rényi del ensemble podría conectarse con la dificultad intrínseca de distinguir entre las geometrías compatibles usando los datos de ringdown.

### 3.2. Qué teoría NO aplica aquí

**Rényi holográfica / Rényi de entanglement gravitacional:** Existe una extensa literatura sobre entropía de Rényi en contextos de holografía (AdS/CFT), geometría de entrelazamiento y superficies cósmicas. Dong (2016, Nature Commun. 7, 12472) relacionó la entropía de Rényi del boundary con áreas de superficies cósmicas en el bulk. Esto **no es importable** al contexto de BASURIN porque:

1. Requiere una dualidad holográfica (espacio asintóticamente AdS, boundary CFT). Los agujeros negros de BASURIN son asintóticamente planos.
2. La Rényi holográfica es una entropía de *un estado cuántico* del boundary, no una medida de diversidad sobre un atlas clásico de soluciones.
3. La coincidencia léxica (ambos usan "Rényi") no implica conexión conceptual.

**Rényi como corrección a Bekenstein-Hawking:** Hay trabajos que modifican la entropía de Bekenstein-Hawking usando fórmulas tipo Rényi o Tsallis (por ejemplo, modelos de holographic dark energy con Rényi). Esto es termodinámica del horizonte, no informática del espacio de soluciones. No aplica.

**Entropía de Rényi en ensembles gravitacionales (GPI):** La integral gravitacional (gravitational path integral) define ensembles de geometrías en gravedad cuántica. Hay trabajos recientes sobre entropías de Rényi de ensembles de baby universes y wormholes. Esto es gravedad cuántica euclidiana con constante cosmológica, irrelevante para un atlas fenomenológico discreto de soluciones clásicas.

### 3.3. Formulación matemática menos abusiva

La formulación más limpia y menos engañosa es:

1. **Definir explícitamente la distribución** {pᵢ} sobre el atlas compatible. La elección natural es normalizar alguna métrica de compatibilidad geométrica (por ejemplo, exp(−d²/2) donde d² es la distancia de Mahalanobis, o la likelihood relativa). Documentar que esta distribución es *epistémica* (mide nuestro estado de conocimiento dado los datos), no *ontológica*.

2. **Calcular los números de Hill** D_q para q ∈ {0, 1, 2, ∞}:
   - D₀ = |atlas compatible| (cardinalidad del conjunto compatible)
   - D₁ = exp(H_Shannon) (número efectivo con pesos Shannon)
   - D₂ = 1/Σpᵢ² (número efectivo de participación)
   - D_∞ = 1/max(pᵢ) (inverso de la concentración máxima)

3. **Reportar el perfil** q → D_q como diagnóstico de la estructura del ensemble.

4. **Nomenclatura:** Llamar a esto "diversidad efectiva del ensemble de geometrías compatibles" o "perfil de concentración de Rényi", NO "entropía de Rényi del agujero negro" ni "entropía del espacio de soluciones" (que sugiere contenido ontológico que no tiene).

---

## 4. Riesgos de abuso conceptual

### 4.1. Terminología forzada

**"Sector hiperbólico" como etiqueta de familia:** Sin una redefinición explícita y operacional de qué significa HYPERBOLIC en el contexto del atlas, la etiqueta es engañosa. Un referee con background en relatividad general pensará en horizontes topológicos (sentido C) o en embedding de Smarr (sentido A) o en well-posedness de EDPs (sentido B), y ninguno de esos tres justifica un mapeo family → sector.

**"Entropía" del espacio de soluciones:** Usar la palabra "entropía" sin calificarla invita a confusión con la entropía de Bekenstein-Hawking, la entropía de entanglement, o la entropía termodinámica. El observable que propone BASURIN es un índice de diversidad/concentración, no una entropía física.

### 4.2. Afirmaciones que serían ilegítimas

1. "Las soluciones Kerr son euclídeas y las EdGB son hiperbólicas" — falso: dentro de Kerr hay soluciones con curvatura negativa en el horizonte (j > 0.866), y las soluciones EdGB perturbativas son geométricamente cercanas a Kerr.

2. "La entropía de Rényi del atlas mide la entropía del remanente" — falso: mide la concentración de la distribución epistémica sobre el atlas, no una propiedad física del objeto.

3. "El sector geométrico de una solución está determinado por la teoría de gravedad" — incorrecto: la geometría depende de los parámetros de la solución, no solo de la teoría.

4. "Existe una transición de fase hiperbólico/elíptico en el espacio de teorías" — esta afirmación mezclaría el sentido B (carácter de las EDPs) con una interpretación termodinámica que no está justificada.

### 4.3. Formulaciones seguras

1. "Para cada solución del atlas, calculamos la curvatura gaussiana del horizonte en los polos y la usamos como clasificador continuo" — legítimo, verificable, no ambiguo.

2. "Medimos la diversidad efectiva (números de Hill) del ensemble de geometrías compatibles por evento" — legítimo, bien definido, con precedente formal extenso.

3. "Estudiamos cómo la diversidad efectiva del ensemble varía poblacionalmente" — legítimo, es una estadística descriptiva del pipeline.

4. "Para soluciones EdGB con acoplamiento fuerte, la pérdida de hiperbolicidad de las EDPs constituye un criterio de exclusión del atlas" — legítimo, con base directa en Ripley & Pretorius (2019, 2020).

---

## 5. Recomendación para BASURIN

### 5.1. Sectorización: abandonar el mapeo family → sector y redefinir

**Recomendación:** Abandonar la taxonomía HYPERBOLIC/ELLIPTIC/EUCLIDEAN como etiquetas fijas por familia. Sustituir por una de estas dos alternativas:

**Opción A — Clasificador continuo por solución:** Para cada candidato del atlas, calcular un invariante geométrico intrínseco (por ejemplo, el signo y magnitud de la curvatura gaussiana del horizonte en los polos, o el "Kerrness" escalar de Abdelqader-Lake). Esto produce un **campo escalar sobre el atlas**, no una clasificación discreta. La ventaja es que es riguroso, auditable e independiente de la familia teórica. La desventaja es que pierde la simplicidad de etiquetas discretas.

**Opción B — Clasificador por carácter de la teoría + régimen:** Usar el sentido B (well-posedness del problema de Cauchy) como criterio de **exclusión**, no de clasificación. Las soluciones de teorías que pierden hiperbolicidad en el exterior del horizonte (EdGB con acoplamiento fuerte) se excluyen del atlas como físicamente inaceptables. Las demás se tratan uniformemente. Esto es conservador y defendible, pero no produce una taxonomía interna rica.

En ninguno de los dos casos se recomienda usar los términos HYPERBOLIC/ELLIPTIC/EUCLIDEAN como nombres de sectores del atlas, porque cualquier uso generará confusión con los sentidos estándar de esos términos en la literatura.

### 5.2. Observable de Rényi: mantener, con formulación precisa

**Recomendación:** Mantener la entropía de Rényi como observable informacional, con las siguientes condiciones:

1. **Denominación:** Llamarla "diversidad efectiva de Rényi del ensemble compatible" o "perfil de concentración del atlas". Nunca "entropía del agujero negro" ni "entropía del espacio de soluciones".

2. **Definición operacional:** Especificar explícitamente cómo se construye la distribución {pᵢ} (normalización de la métrica de compatibilidad, tratamiento de soluciones fuera de dominio, etc.).

3. **Interpretación:** Declarar desde el abstract que es un **diagnóstico epistémico** sobre el poder discriminante de los datos, no una cantidad termodinámica. Un D₂ bajo significa "los datos señalan pocas geometrías compatibles con mucho peso relativo" (alta discriminación). Un D₂ alto significa "muchas geometrías son comparablemente compatibles" (baja discriminación).

4. **Perfil completo:** Reportar q → D_q para varios valores de q, no solo un H_q fijo. El perfil diagnóstica la estructura de la distribución (cola pesada vs. uniforme) de forma más informativa que un solo número.

5. **Precedente formal:** Citar los números de Hill (Hill 1973), Jost (2006) y la conexión formal con Rényi como base matemática. Citar Sason & Verdú (2018) para la conexión con hypothesis testing. **No** citar Rényi holográfica ni Bekenstein-Hawking.

### 5.3. Síntesis

El programa experimental de BASURIN tiene dos componentes teóricos de muy distinta solidez:

- La sectorización geométrica family → {HYPERBOLIC, ELLIPTIC, EUCLIDEAN} **no tiene base directa** en la forma propuesta. Los ingredientes teóricos existen (embedding de Smarr, pérdida de hiperbolicidad en EdGB) pero no se combinan en una taxonomía como la que se necesita. Se recomienda redefinir o abandonar.

- La Rényi como observable informacional sobre el ensemble discreto **tiene base directa** como índice de diversidad, con precedente formal amplio y una interpretación limpia como medida de concentración epistémica. Se recomienda mantener con formulación precisa y nomenclatura cuidadosa.

---

## 6. Referencias primarias citadas

### Geometría de horizontes y embedding
- Smarr L., "Surface Geometry of Charged Rotating Black Holes", Phys. Rev. D 7, 289 (1973)
- Frolov V.P., "Embedding of the Kerr-Newman Black Hole Surface in Euclidean Space", Phys. Rev. D 73, 064021 (2006), arXiv:gr-qc/0601104
- Gibbons G.W., Herdeiro C.A.R., Rebelo C., "Global embedding of the Kerr black hole event horizon into hyperbolic 3-space", Phys. Rev. D 80, 044014 (2009), arXiv:0906.2768
- Herdeiro C.A.R. et al., "Horizon geometry for Kerr black holes with synchronised hair", arXiv:1804.04910 (2018)
- Abdelqader M., Lake K., "Invariant characterization of the Kerr spacetime", Phys. Rev. D 91, 084017 (2015), arXiv:1412.8757

### Topología de horizontes
- Hawking S.W., "Black holes in general relativity", Commun. Math. Phys. 25, 152 (1972)
- Galloway G.J., Schoen R., "A generalization of Hawking's black hole topology theorem to higher dimensions", Commun. Math. Phys. 266, 571 (2006), arXiv:gr-qc/0509107

### Hiperbolicidad y well-posedness en gravedad modificada
- Ripley J.L., Pretorius F., "Hyperbolicity in Spherical Gravitational Collapse in a Horndeski Theory", Phys. Rev. D 99, 084014 (2019), arXiv:1902.01468
- Ripley J.L., Pretorius F., "Gravitational Collapse in Einstein dilaton Gauss-Bonnet Gravity", Phys. Rev. D 100, 104008 (2019), arXiv:1903.07543
- Ripley J.L., Pretorius F., "Scalarized Black Hole dynamics in Einstein dilaton Gauss-Bonnet Gravity", Phys. Rev. D 101, 044015 (2020), arXiv:1911.11027
- Thaalba F. et al., "Hyperbolicity in scalar-Gauss-Bonnet gravity: A gauge invariant study for spherical evolution", Phys. Rev. D (2024)

### Rényi como diversidad efectiva
- Hill M.O., "Diversity and evenness: a unifying notation and its consequences", Ecology 54, 427 (1973)
- Jost L., "Entropy and diversity", Oikos 113, 363 (2006)
- Chao A., Jost L., "Estimating diversity and entropy profiles via discovery rates of new species", Methods Ecol. Evol. 6, 873 (2015)
- Gresss A.D., Rosenberg N.A., "Mathematical constraints on a family of biodiversity measures via connections with Rényi entropy", BioSystems 237, 105153 (2024)

### Rényi en hypothesis testing
- Sason I., Verdú S., "Arimoto-Rényi conditional entropy and Bayesian M-ary hypothesis testing", IEEE Trans. Inf. Theory 64, 4 (2018)

### Fisher information en GW
- Vallisneri M., "Use and Abuse of the Fisher Information Matrix in the Assessment of Gravitational-Wave Parameter-Estimation Prospects", Phys. Rev. D 77, 042001 (2008), arXiv:gr-qc/0703086
- Antonelli A. et al., "A Fisher matrix for gravitational-wave population inference", MNRAS 519, 2736 (2023)

### Geometría de la información
- Amari S., "Differential-Geometrical Methods in Statistics", Springer Lecture Notes in Statistics 28 (1985)
- Ay N. et al., "Information Geometry", Springer (2017)

---

## 7. Declaración de transparencia

Este informe se basa en literatura consultada mediante búsqueda web durante su redacción. Las afirmaciones sobre la inexistencia de resultados ("no existe en la literatura una clasificación X") son inferencias basadas en la ausencia de resultados en búsquedas dirigidas, no afirmaciones absolutas. Si existiera un trabajo que establezca una taxonomía hiperbólico/elíptico/euclídeo para soluciones de agujeros negros 4D asintóticamente planas en el sentido que BASURIN requiere, no lo he encontrado.
