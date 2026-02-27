# FASE 6 — Preguntas técnicas abiertas

Estas respuestas abordan las preguntas técnicas planteadas para la **FASE 6** del desarrollo de **BASURIN**, centradas en la transición de una PSD analítica a una PSD medida (`measured_psd.json`) y su impacto en el análisis de ringdown y la geometría de la información.

## 1. Fundamentos GR / QNM

1. ¿Bajo qué condiciones la descomposición en quasi-normal modes (QNMs) sigue siendo una base completa cuando el producto interno ⟨h|h⟩ se calcula con una S_n(f) medida que puede contener líneas espectrales no gaussianas? (`measured_psd.json`, inner product ponderado por PSD)  
   **Respuesta:** La descomposición en QNMs es una base completa en el sentido de "modos propios" de la perturbación. Sin embargo, en un espacio de Hilbert pesado por una PSD no gaussiana/no suave, la completitud "efectiva" depende de que $S_n(f)$ no tenga ceros (nodos) en el soporte de la señal. Las líneas espectrales actúan como filtros de muesca (notches); si una línea coincide exactamente con un QNM, ese modo se vuelve "invisible" (no proyectable), rompiendo la completitud práctica para la reconstrucción.

2. ¿Cómo cambia la ortogonalidad efectiva entre el modo fundamental (ℓ=m=2, n=0) y los overtones cuando se sustituye la PSD analítica por la PSD de Welch, dado que las regiones de overlap en frecuencia pueden estar dominadas por artefactos instrumentales? (`measured_psd.json` vs PSD analítica, matched filter ⟨h_n|h_m⟩)  
   **Respuesta:** La ortogonalidad $\langle h_n | h_m \rangle = 0$ se pierde. La PSD de Welch introduce "color" local y artefactos de ventaneo (spectral leakage). Si los overtones están cerca de líneas instrumentales, el producto interno se verá dominado por el ruido en esa banda, aumentando la covarianza entre modos y dificultando la distinción entre el modo fundamental y el primer overtone.

3. Si la PSD medida introduce una estructura fina en frecuencia que la PSD analítica no resuelve, ¿puede esto generar un sesgo sistemático en la estimación de (f_QNM, τ_QNM) que mime una desviación de GR? (`stage_summary.json`, δf/f₀ y δτ/τ₀)  
   **Respuesta:** Sí. Una estructura fina no resuelta (o mal estimada) en $S_n(f)$ actúa como un filtro de fase/amplitud efectivo. Si la PSD medida sobreestima el ruido en $f_{QNM}$, el fit tenderá a desplazar la frecuencia hacia regiones de menor ruido aparente, generando un $\delta f/f_0$ espurio que podría malinterpretarse como una violación del teorema de no-pelo.

4. ¿Cuál es la banda de frecuencia mínima [f_low, f_high] alrededor de cada QNM para la cual la PSD medida debe ser fiable antes de que el producto interno pierda más de un ε en SNR² respecto del caso ideal? (`Ω(f)` output, SNR²(f) acumulado)  
   **Respuesta:** Típicamente $[f_n - 5\Gamma_n, f_n + 5\Gamma_n]$, donde $\Gamma = 1/\tau$. Para que el error en $SNR^2$ sea $<\epsilon$, la PSD debe ser estable en el ancho de banda donde se concentra el 95-99% de la energía del modo.

5. ¿Qué criterio cuantitativo permite distinguir una desviación genuina de GR en el espacio (f, Q) de un artefacto inducido por una mala estimación de S_n(f) en la vecindad de la frecuencia del modo? (`dual_method_comparison.json`, Bayes factor δGR vs δPSD)  
   **Respuesta:** El factor de Bayes $\ln \mathcal{B}$. Si un modelo que incluye una "perturbación de PSD" (p.ej. marginalizando sobre splines de la PSD) explica los datos tan bien como un modelo $\delta GR$, la desviación no es genuina. Se busca que $\ln \mathcal{B}_{GR+\delta PSD} < \ln \mathcal{B}_{\delta GR}$.

## 2. Ruido / PSD

6. ¿Cuántos segmentos de datos y qué longitud de FFT (NFFT) se necesitan en el estimador de Welch para que el error relativo de Ŝ_n(f) sea ≤ ε en la banda del ringdown, dado un nivel de confianza p? (`measured_psd.json`, varianza χ² del periodograma)  
   **Respuesta:** Para un error relativo $\epsilon \approx \sqrt{1/K}$ (donde $K$ es el número de segmentos), se requieren $K \approx 1/\epsilon^2$ segmentos. Para $\epsilon=0.1$ al 90% de confianza, se necesitan $\sim 100$ segmentos. La $NFFT$ debe ser tal que $\Delta f = f_s/NFFT \ll 1/(2\pi\tau)$ para no suavizar la estructura del QNM.

7. ¿Qué esquema de overlap (50%, Hann) y ventana (Hann, Tukey, Kaiser) minimiza el sesgo espectral en las frecuencias de los QNMs sin degradar la resolución en frecuencia por debajo de Δf_QNM ~ 1/(2πτ)? (`measured_psd.json`, bias-varianza del estimador de Welch)  
   **Respuesta:** 50% de overlap con ventana **Hann** es el estándar (balance bias-varianza). **Kaiser** es superior si se requiere suprimir lóbulos laterales de líneas muy intensas que podrían contaminar la frecuencia del ringdown.

8. ¿Cómo se propaga la incertidumbre de Ŝ_n(f) al inner product ⟨a|b⟩ = 4 Re ∫ ã*(f) b̃(f) / S_n(f) df, y cuál es la expresión analítica de σ²[⟨a|b⟩] en función de σ²[Ŝ_n(f)]? (`measured_psd.json`, producto interno, Fisher metric)  
   **Respuesta:** $\sigma^2[\langle a|b \rangle] \approx \int \frac{|\tilde{a}(f)|^2 |\tilde{b}(f)|^2}{S_n^4(f)} \sigma^2[S_n(f)] df$. La incertidumbre escala inversamente con el número de segmentos $K$ utilizados en Welch.

9. En la interpolación log-log de la PSD medida, ¿qué orden de spline o esquema (lineal, cúbico, PCHIP) preserva la positividad de S_n(f) > 0 y no introduce oscilaciones espurias entre puntos de soporte? (`measured_psd.json`, interpolación)  
   **Respuesta:** El esquema **PCHIP** (Piecewise Cubic Hermite Interpolating Polynomial) es preferible porque preserva la monotonicidad y la positividad, evitando las oscilaciones (overshooting) de los splines cúbicos naturales en las cercanías de líneas espectrales.

10. ¿Cómo se deben tratar los bordes del espectro (f → f_low, f → f_high) en la interpolación: zero-padding, extensión constante, roll-off exponencial? ¿Qué impacto tiene cada elección en Ω(f) cerca de los extremos? (`measured_psd.json`, `Ω(f)`, condiciones de contorno espectrales)  
    **Respuesta:** Extensión constante o roll-off suave (planck-taper) hacia un nivel de ruido alto. El zero-padding es peligroso porque $1/S_n \to \infty$. El impacto en $\Omega(f)$ es crítico si el QNM está cerca de $f_{low}$ (típicamente 20 Hz en LIGO).

11. ¿Qué criterio se usa para detectar y excluir (o interpolar sobre) líneas instrumentales (violin modes, calibration lines, 60 Hz) presentes en la PSD medida antes de inyectarla al inner product? (`measured_psd.json`, notch list, SNR de líneas)  
    **Respuesta:** Se usa un umbral de SNR (p.ej. líneas que sobresalen $5\sigma$ del promedio local). Es mejor interpolar sobre ellas (in-painting) que usar un notch rígido si el QNM cae cerca, para evitar discontinuidades en el integrando del matched filter.

12. ¿Es necesario estimar la PSD con datos inmediatamente adyacentes al segmento de ringdown (on-source), o basta con una PSD off-source promediada sobre un intervalo mayor? ¿Cuál es la escala temporal de no-estacionariedad relevante? (`measured_psd.json`, time-frequency drift ΔS_n/Δt)  
    **Respuesta:** Se prefiere **off-source** (datos adyacentes pero no incluyendo el evento) para evitar que la propia señal contamine la estimación del ruido. La escala de no-estacionariedad relevante es de minutos; la PSD debe estimarse en una ventana de $\sim 128-512$ segundos.

13. ¿Qué test estadístico (Anderson-Darling, KS sobre residuos whitened) valida que los datos blanqueados con la PSD medida son efectivamente gaussianos y estacionarios en la ventana de análisis? (`measured_psd.json`, whitened residuals, p-value de gaussianidad)  
    **Respuesta:** El test de **Anderson-Darling** sobre los residuos blanqueados es más sensible a las colas de la distribución que KS. Un p-value $> 0.05$ sugiere que el blanqueado fue exitoso.

14. Cuando se usa --psd-path con una PSD externa, ¿debe BASURIN verificar automáticamente la consistencia entre la resolución en frecuencia de la PSD suministrada y el Δf implícito del segmento de datos? (`measured_psd.json`, Δf_PSD vs Δf_data)  
    **Respuesta:** **Sí**, BASURIN debe verificar que $\Delta f_{data} \geq \Delta f_{PSD}$. Si la PSD es más gruesa que los datos, se requiere interpolación; si es más fina, se debe diezmar o promediar para evitar aliasing espectral en el producto interno.

## 3. Estimación y estadística

15. ¿Cómo se modifica la likelihood p(d|θ, S_n) cuando S_n(f) tiene incertidumbre propia: se marginaliza sobre Ŝ_n, se usa una likelihood marginalizada tipo Whittle-Student, o se trata como error sistemático? (`stage_summary.json`, likelihood, posterior de θ_QNM)  
    **Respuesta:** Se debería usar una **Whittle-Student Likelihood** (marginalizada sobre la incertidumbre de la varianza del ruido), la cual penaliza las regiones donde la PSD es menos fiable, transformando la gaussiana en una distribución con colas más pesadas.

16. ¿Cuál es la expresión correcta del SNR óptimo ρ_opt cuando la PSD no es conocida exactamente sino estimada, y cómo escala la degradación de ρ con el número de segmentos K del estimador de Welch? (`measured_psd.json`, ρ² = 4∫|h̃|²/S_n df, varianza de ρ̂)  
    **Respuesta:** $\rho_{opt}^2$ sigue una distribución no central si $S_n$ es estimada. La degradación escala como $(1 - 1/K)$, donde $K$ es el número de promedios de Welch.

17. ¿Cambian los priors efectivos sobre (f, Q) al pasar de PSD analítica a PSD medida, dado que regiones del espacio de parámetros con alta S_n(f) quedan suprimidas de forma diferente? (`stage_summary.json`, prior × likelihood volume)  
    **Respuesta:** Sí cambian. La PSD medida actúa como un filtro en el espacio de parámetros. Si $S_n(f)$ aumenta en una banda, el volumen del prior "efectivo" (donde la likelihood es significativa) se contrae en esa frecuencia.

18. ¿Cómo se redefine el Bayes factor entre el modelo GR y el modelo con desviaciones δf, δτ cuando la PSD tiene incertidumbre: se calcula un Bayes factor condicional a cada realización de PSD, o se marginaliza conjuntamente? (`dual_method_comparison.json`, Bayes factor, evidence integral)  
    **Respuesta:** Se debe marginalizar: $P(d|M) = \int P(d|M, S_n) P(S_n) dS_n$. En la práctica, esto se aproxima mediante "PSD variations" o usando la likelihood marginalizada mencionada en el punto 15.

19. ¿Es el matched filter score ⟨d|h(θ)⟩/√⟨h|h⟩ un estadístico suficiente cuando S_n(f) es estimada y no conocida, o pierde suficiencia y hay que recurrir a un estadístico más general? (`stage_summary.json`, suficiencia del filtro adaptado)  
    **Respuesta:** El matched filter pierde suficiencia si el ruido es no gaussiano o mal estimado. Se requieren estadísticos adicionales (como el $\chi^2$ time-frequency discriminator) para asegurar que el trigger es una señal y no un artefacto de la PSD.

20. En la combinación multi-evento, ¿los pesos relativos de cada evento cambian cuando se usa PSD medida por evento vs una PSD analítica común? ¿Cómo afecta esto al combined posterior de δGR? (`multi_event_combination`, pesos ∝ 1/σ²_i, posterior combinado)  
    **Respuesta:** Los pesos cambian radicalmente. Un evento con SNR moderada en una región de PSD "limpia" (medida) puede tener más peso que un evento con SNR alta pero contaminado por una PSD incierta o con líneas.

## 4. Geometría de la información

21. Si la Fisher information matrix Γ_ij = ⟨∂_i h | ∂_j h⟩ se calcula con Ŝ_n(f) en lugar de S_n(f) exacta, ¿cuál es la corrección a primer orden en δS = Ŝ − S a los elementos Γ_ij, y bajo qué condiciones es despreciable? (`Ω(f)`, Fisher metric, δΓ/Γ)  
    **Respuesta:** $\delta \Gamma_{ij} \approx -4 \text{Re} \int \frac{\partial_i \tilde{h}^* \partial_j \tilde{h}}{S_n^2} \delta S_n df$. Es despreciable si $|\delta S_n / S_n| \ll 1/\rho$, es decir, si el error en la PSD es mucho menor que el inverso de la SNR.

22. ¿La métrica de Fisher g_ij(θ) calculada con PSD medida sigue siendo definida positiva en todo el espacio de parámetros explorado, o pueden aparecer regiones degeneradas por artefactos en Ŝ_n(f)? (`Ω(f)`, positividad de det(g), condición de Sylvester)  
    **Respuesta:** En regiones de baja SNR o PSD mal estimada, el determinante de la métrica de Fisher puede volverse numéricamente inestable o no definido positivo (especialmente si hay fuertes correlaciones inducidas por líneas de ruido). Se requiere el criterio de Sylvester.

23. ¿Cómo se propaga la incertidumbre de Ŝ_n al escalar de curvatura R del espacio de parámetros, y cuál es σ[R]/R en función de K (segmentos de Welch) y del SNR del evento? (`Ω(f)`, scalar curvature R, propagación de errores)  
    **Respuesta:** La incertidumbre $\sigma[R]/R$ escala inversamente con $\sqrt{K}$ y con una potencia alta de la SNR ($\sim \rho^{-2}$ o $\rho^{-4}$ dependiendo de la dimensión). La PSD medida añade "ruido geométrico" a la superficie de la likelihood.

24. ¿Es consistente interpretar anomalías en R(f, Q) como evidencia de objetos exóticos si la propia PSD medida introduce variaciones locales de curvatura comparables en magnitud? (`Ω(f)`, R_exotic vs R_PSD-artifact, criterio de separabilidad)  
    **Respuesta:** Es peligroso. Una línea instrumental no modelada en la PSD creará un "pozo" o "pico" de curvatura local en $(f, Q)$ que puede imitar la firma de un objeto exótico (como un eco de un gravastar). Se requiere que $\Delta R_{PSD} \ll R_{model}$.

25. ¿El pullback de la métrica de Fisher al subespacio (f, Q) conmuta con el cambio de PSD analítica → PSD medida, o el orden de las operaciones (primero marginalizar, luego cambiar PSD) importa? (`Ω(f)`, pullback metric, marginalización)  
    **Respuesta:** No conmuta. La marginalización sobre parámetros de ruido y el cambio de métrica son operaciones no lineales. El orden correcto es calcular la métrica sobre la likelihood ya condicionada/marginalizada por la mejor estimación de la PSD.

26. ¿Cómo varía el volumen del espacio de parámetros Vol(g) = ∫√det(g) dθ al sustituir la PSD, y puede usarse ΔVol/Vol como diagnóstico de consistencia entre PSD analítica y medida? (`Ω(f)`, volumen métrico, template bank density)  
    **Respuesta:** $\Delta Vol/Vol$ es un excelente diagnóstico. Un salto brusco al usar la PSD medida indica que la resolución de parámetros está siendo dominada por características del ruido y no por la morfología de la señal.

27. ¿Debe la métrica Ω(f) en la FASE 6 incluir un término de corrección por la varianza de Ŝ_n(f), análogo a la "expected Fisher information" vs "observed Fisher information"? (`Ω(f)`, expected vs observed Fisher, corrección de Bartlett)  
    **Respuesta:** **Sí**, para ser rigurosos, $\Omega(f)$ debería incluir el término de "información observada" que de cuenta de que la PSD es una realización estocástica.

28. ¿Las geodésicas en el espacio (f, Q) calculadas con PSD medida difieren significativamente de las calculadas con PSD analítica, y cómo se cuantifica esa diferencia (distancia de Hausdorff, divergencia geodésica)? (`Ω(f)`, ecuación geodésica, distancia entre curvas)  
    **Respuesta:** La distancia de Hausdorff entre geodésicas (analítica vs medida) cuantifica la "deriva de interpretación". Si la distancia es mayor que el radio de la elipse de incertidumbre, las conclusiones sobre los parámetros físicos no son robustas.

## 5. Multi-detector

29. ¿Cuál es la definición operacional de "coherencia multi-detector" en el contexto de BASURIN: se refiere a la coherencia espectral γ²(f) = |S_xy|²/(S_xx S_yy) entre pares de detectores, o a la consistencia de posteriors de θ_QNM entre detectores? (`multi_detector_coherence`, γ²(f) o Kullback-Leibler entre posteriors)  
    **Respuesta:** En BASURIN, se refiere a la consistencia de los posteriors (KL divergence pequeña) y a que el residuo conjunto tras el blanqueado no muestre correlaciones cruzadas.

30. ¿Cómo se combinan las PSDs medidas de N detectores para construir un inner product multi-detector: se usa la suma ponderada Σ_I ⟨a|b⟩_I con pesos por detector, o se construye un producto interno conjunto con la matriz de correlación cruzada completa (incluyendo términos off-diagonal si hay ruido correlacionado)? (`measured_psd.json` por detector, inner product multi-detector)  
    **Respuesta:** Si el ruido es no correlacionado, $\langle a|b \rangle_{net} = \sum_I \langle a|b \rangle_I$. Si hay ruido correlacionado (p.ej. resonancias magnéticas globales), se requiere la matriz de covarianza cruzada $S_{IJ}(f)$.

31. ¿Bajo qué condiciones puede asumirse que el ruido entre detectores (H1, L1, V1) es no correlacionado, y cómo se testea esta hipótesis con datos reales antes de asumir una likelihood factorizada? (`cross-PSD C_IJ(f)`, test de coherencia residual)  
    **Respuesta:** Se asume no correlacionado por encima de 20-30 Hz. Se testea mediante el estimador de coherencia $\gamma_{IJ}^2(f)$. Si $\gamma^2$ es significativamente mayor que $1/N_{segments}$, la asunción de verosimilitud factorizada falla.

32. Si las PSDs medidas de dos detectores difieren significativamente en la banda del ringdown, ¿cómo se asignan pesos relativos para la métrica combinada: inversamente proporcionales a S_n,I(f), a la varianza de la estimación, o al SNR individual? (`measured_psd.json` por detector, pesos w_I, combined Fisher)  
    **Respuesta:** Los pesos naturales son $1/S_{n,I}(f)$, que surgen automáticamente en la likelihood óptima.

33. ¿Cómo se trata el caso en que un detector tiene una línea espectral en la frecuencia del QNM y otro no: se excluye ese detector en esa banda, se penaliza su peso, o se interpola? (`measured_psd.json`, notch handling por detector, impacto en SNR combinado)  
    **Respuesta:** No se debe excluir el detector completo. Se aplica un notch o se reduce el peso *solo* en los bins afectados. La información de los otros detectores "rellena" el hueco si la señal es coherente.

34. ¿Qué test cuantitativo de consistencia inter-detector se aplica antes de combinar: un χ² sobre la diferencia de best-fit (f, Q) entre detectores, o un Bayes factor de coherencia vs incoherencia? (`multi_detector_coherence`, test de consistencia, Bayes factor coherente/incoherente)  
    **Respuesta:** El factor de Bayes coherente vs. incoherente ($\ln \mathcal{B}_{coh/inc}$) es el estándar de oro. Un valor $>5$ indica que los datos en todos los detectores prefieren la misma señal.

35. ¿Debe la combined metric g_ij^{(comb)} = Σ_I w_I g_ij^{(I)} heredar la positividad de cada g_ij^{(I)}, o pueden los pesos negativos (en esquemas de substracción de ruido) romper esta propiedad? (`Ω(f)` combinada, positividad, pesos)  
    **Respuesta:** Sí, hereda la positividad si los pesos $w_I$ son positivos (lo cual es cierto para $1/S_n$).

36. ¿Cómo se incorpora el antenna pattern F_{+,×}(α, δ, ψ) de cada detector en la métrica combinada, y se marginaliza sobre la posición del cielo o se fija al valor del PE completo del evento? (`multi_detector_coherence`, antenna pattern, marginalización sky location)  
    **Respuesta:** Se debe marginalizar sobre la posición del cielo $(\alpha, \delta)$ usando el prior del PE (Parameter Estimation) de la fase de inspección de la señal, integrando el efecto de $F_+, F_\times$ en la amplitud efectiva de cada detector.

## 6. Validación / Robustez

37. ¿Qué inyecciones de señal (injection campaign) se requieren para validar que el pipeline con PSD medida recupera los parámetros inyectados sin sesgo: barrido en SNR, en (f, Q), en posición relativa a líneas instrumentales? (`injection_study`, recovered − injected, p-p plot)  
    **Respuesta:** Se requiere un barrido "grid" en SNR (10 a 100) y posiciones de frecuencia que crucen líneas conocidas de la PSD medida.

38. ¿El pipeline con PSD medida pasa un test de cobertura frecuentista (p-p plot) al mismo nivel que el pipeline con PSD analítica, para un ensemble de ≥ 500 inyecciones? (`injection_study`, p-p plot, cobertura de intervalos credibles)  
    **Respuesta:** Es el test definitivo. Si el 90% de las inyecciones caen dentro del intervalo de credibilidad del 90% usando la PSD medida, el pipeline es estadísticamente consistente.

39. ¿Cómo se cuantifica la backward compatibility: qué métrica (KL divergence entre posteriors, |Δρ|/ρ, |ΔlogB|) certifica que sin --psd-path el resultado es bitwise-idéntico al de la versión anterior? (`regression_test`, posteriors, evidencia, SNR)  
    **Respuesta:** La divergencia KL entre el posterior con PSD analítica (usando los mismos parámetros que el modelo analítico) y el resultado anterior debe ser $\approx 0$.

40. ¿Qué ocurre con la estabilidad numérica del inner product cuando Ŝ_n(f) → 0 en algún bin (p.ej. por un artefacto de interpolación): se impone un floor S_min, se usa regularización, o se trunca la integral? (`measured_psd.json`, regularización, estabilidad numérica de ⟨a|b⟩)  
    **Respuesta:** Se debe imponer un **floor de ruido** ($S_{min}$) basado en el límite de cuantización o el ruido térmico teórico para evitar divisiones por cero en el integrando.

41. ¿Es la estimación de la PSD sensible al segmento temporal elegido (glitches, non-stationarities), y qué pipeline de data quality (DQ flags, gating) se aplica antes de estimar Ŝ_n? (`measured_psd.json`, DQ vetoes, gating, impacto en posterior)  
    **Respuesta:** Es crítico. Glitches en el segmento de estimación de PSD pueden inflar artificialmente el ruido y "suprimir" la señal. Se requiere un veto basado en el trigger de Omicron o un gating de los datos.

42. ¿Se implementa un test de sensibilidad tipo "PSD perturbation": inyectar δS_n(f) controladas y verificar que |Δθ_MAP| < σ_θ para cada parámetro? (`sensitivity_test`, δS_n → δθ, robustez)  
    **Respuesta:** Este test es vital para publicar. Si una variación del 10% en la PSD cambia el MAP de la frecuencia más de $1\sigma$, el resultado es "ruido-dependiente".

43. ¿Cómo se diagnostica que la interpolación log-log no ha introducido features espurias: comparación visual, residuos de interpolación, o test χ² entre PSD original y PSD interpolada evaluada en los bins originales? (`measured_psd.json`, residuos de interpolación)  
    **Respuesta:** Test $\chi^2$ entre la PSD cruda de Welch y la versión interpolada. Los residuos deben ser ruido blanco con varianza proporcional a $1/K$.

44. ¿Se necesita un test de convergencia del sampler (nested sampling, MCMC) específico para la PSD medida, dado que la likelihood landscape puede tener estructura más fina que con PSD analítica? (`stage_summary.json`, n_eff, evidence error, autocorrelación del sampler)  
    **Respuesta:** Una PSD medida con mucha estructura puede crear una superficie de log-likelihood con muchos mínimos locales. Se requiere monitorear $n_{eff}$ (número de muestras efectivas) y asegurar que el esquema de "tempering" sea suficiente.

45. ¿Qué sucede con los resultados cuando se varía la longitud del segmento de PSD (e.g., 4s vs 16s vs 64s): son estables los posteriors dentro de las barras de error, o hay una dependencia sistemática? (`measured_psd.json`, estudio de convergencia en T_seg)  
    **Respuesta:** Si el posterior de $f, Q$ cambia significativamente entre usar 4s y 16s para la PSD, indica no-estacionariedad o que la resolución espectral de la PSD es insuficiente.

46. ¿Cómo se valida la combinación multi-detector end-to-end: inyección coherente en N detectores con PSDs medidas independientes, recuperación de parámetros, y verificación de que el SNR combinado escala como √(Σ ρ_I²)? (`injection_study`, SNR network, parámetros recuperados)  
    **Respuesta:** Se valida verificando que $\rho_{network} = \sqrt{\sum \rho_I^2} \pm \epsilon$.

## 7. Claims publicables

47. ¿Qué nivel de mejora en la estimación de (f, Q) respecto de la PSD analítica constituye un resultado publicable: reducción de σ_f, σ_Q, mejora en log-evidence, o reducción de sesgo sistemático medible en inyecciones? (criterio de significancia, σ improvement)  
    **Respuesta:** Una reducción del error sistemático (bias) es más valiosa que una reducción de $\sigma$. Si se demuestra que la PSD analítica desviaba el resultado en $>0.5\sigma$, el uso de la medida es indispensable.

48. ¿Se puede afirmar que la PSD medida "reduce el systematic error budget" del test de no-hair theorem, y qué evidencia cuantitativa soporta esa afirmación? (`dual_method_comparison.json`, Δ(δf/f) analítica vs medida)  
    **Respuesta:** Se puede afirmar si se demuestra que $\Delta(\delta f/f)$ (analítica - medida) es comparable a la incertidumbre estadística de los tests de no-pelo previos.

49. ¿Es publicable la comparación de curvatura escalar R(f,Q) calculada con PSD analítica vs medida como diagnóstico de robustez, o se necesita además un modelo teórico de cómo R debería cambiar? (`Ω(f)`, ΔR/R, interpretación física)  
    **Respuesta:** Es publicable como "diagnóstico de robustez". Si $R$ es suave en la región del posterior, el resultado es geométricamente estable.

50. ¿Qué afirmaciones sobre coherencia multi-detector son falsificables con los datos actuales (O3/O4), y cuáles requieren sensibilidad de detectores futuros (O5, ET, CE)? (`multi_detector_coherence`, SNR umbral, horizonte de detección)  
    **Respuesta:** Con O3/O4 se pueden falsificar claims de "detecciones de un solo detector" que no sean consistentes con la PSD medida en la red. Para O5/ET, esto será la base de la astronomía de precisión.

51. ¿Puede BASURIN, con PSD medida, hacer un claim sobre la detección de overtones que sea robusto frente a la elección de PSD, o el "overtone significance" depende críticamente de la estimación de ruido? (`stage_summary.json`, Bayes factor overtone, dependencia en PSD)  
    **Respuesta:** **Sí.** Este es el punto más sensible. La significancia de un overtone (frecuentemente de baja SNR) depende totalmente de que el ruido en su frecuencia no esté subestimado. BASURIN debe demostrar que el overtone no es un artefacto de la PSD.

52. ¿Qué comparación con pipelines existentes (pyRing, ringdown, RIFT) se requiere para validar externamente los resultados de BASURIN con PSD medida, y cuáles son las diferencias metodológicas que deben documentarse? (cross-pipeline comparison, systematics budget)  
    **Respuesta:** Se requiere comparar con `pyRing` (que usa métodos similares de PSD). La diferencia principal de BASURIN debe ser la integración de la **información geométrica** ($\Omega(f)$) como validador.

53. ¿Se puede cuantificar la ganancia de información (en bits) de usar PSD medida vs analítica para la comunidad GW, y es esa ganancia estadísticamente significativa para el catálogo actual de eventos de ringdown? (`stage_summary.json`, KL divergence posterior_measured vs posterior_analytic, significance across catalog)  
    **Respuesta:** Se mide con la divergencia KL. Una ganancia de $>1$ bit de información al usar la PSD medida indicaría un refinamiento significativo de la física del evento.

54. ¿Qué systematic error budget global se puede construir para la FASE 6 que incluya: error de estimación de PSD, error de interpolación, error de combinación multi-detector, y cómo se compara con el statistical error en (f, Q) para eventos típicos de O3/O4? (error budget table, systematic vs statistical)  
    **Respuesta:** Debe incluir: $\sigma_{stat}$ (SNR limitada), $\sigma_{PSD-cal}$ (estimación de Welch), $\sigma_{interp}$ (error de spline) y $\sigma_{wave}$ (error de modelo).

55. ¿El framework de información geométrica con PSD medida aporta insight físico más allá de lo que ya proporciona el análisis bayesiano estándar, y cómo se articula esa value proposition en un paper? (`Ω(f)`, curvatura, interpretación, novelty claim)  
    **Respuesta:** La propuesta de valor es que BASURIN no solo dice "estos son los parámetros", sino "esta es la fiabilidad geométrica de la región del espacio de parámetros donde residen esos parámetros", permitiendo descartar falsos positivos de física exótica de manera objetiva.