**MEMORÁNDUM TÉCNICO: RELACIÓN QNM BULK / ESPECTRO CFT**

**PARA:** Investigador Principal (Pipeline de Selección Geométrica)
**DE:** Físico Teórico (Especialista en AdS/CFT & QNM)
**FECHA:** 6 de Febrero de 2026
**ASUNTO:** Estado del arte sobre la correspondencia QNM/CFT para selección de métricas.

Este documento resume la relación física rigurosa entre las perturbaciones gravitacionales en el *bulk* (AdS) y la teoría de campos dual (CFT) en el borde, con el fin de evaluar la viabilidad de identificar geometrías mediante observables de *ringdown*.

---

### 1. QNM del Bulk y Polos del Correlador Retardado

**Mecanismo Físico:**
La correspondencia AdS/CFT establece una identidad entre la función de partición de la gravedad en el *bulk* y el funcional generador de la CFT. En el límite de respuesta lineal, una perturbación del campo gravitacional $\delta g_{\mu\nu}$ en el borde actúa como una fuente $J$ para el operador de energía-momento $T_{\mu\nu}$ en la CFT.

Los modos cuasi-normales (QNM) son las frecuencias naturales de oscilación del agujero negro bajo condiciones de contorno puramente entrantes en el horizonte (*ingoing*) y Dirichlet (u otras condiciones reflexivas) en el borde asintótico AdS.
En la CFT, estas condiciones de contorno en el horizonte corresponden físicamente a la disipación: la energía fluye desde los grados de libertad colectivos hacia el baño térmico (el horizonte), lo que impide el retorno de la información. Matemáticamente, esto implica que **los QNM coinciden exactamente con los polos de la función de Green retardada** $\langle \mathcal{O}\mathcal{O} \rangle_R$ en el plano complejo de frecuencias.

**Dominio de Validez:**
Esta relación es:
*   **Exacta** en el límite de 't Hooft grande y acoplamiento fuerte.
*   **Válida** para AdS-Schwarzschild (temperatura finita), AdS planar (hidrodinámica) y AdS Global.
*   **Válida** para cualquier espín de perturbación (escalar, vectorial, tensorial) siempre que se identifique el operador dual correcto.

---

### 2. Relación Explícita $\omega_n$ vs. Autovalores ($\Delta$, $M^2$)

**No existe una fórmula cerrada universal** que relacione $\omega_n$ con $\Delta$ para un agujero negro AdS genérico. La situación se divide en dos regímenes:

*   **AdS Puro (Sin Agujero Negro):** Existe una relación analítica exacta y discreta (espectro real):
    $$ \omega_{n,l} = \Delta + l + 2n $$
    Aquí, $\Delta$ es la dimensión conforme del operador dual, fijada por la masa del campo en el bulk ($m^2 L^2 = \Delta(\Delta - d)$).

*   **AdS con Agujero Negro (Temperatura Finita $T > 0$):**
    La presencia del horizonte rompe la simetría y hace $\omega$ compleja.
    *   **No hay solución analítica general:** La relación entre $\omega_n$, $T$ y $\Delta$ se obtiene resolviendo numéricamente una ecuación de ondas tipo Schrödinger con un potencial efectivo $V(r)$.
    *   **Rol de $\Delta$ / $M^2$:** El parámetro $\Delta$ no fija la frecuencia directamente, sino que modifica la altura y forma asintótica de la barrera de potencial $V(r)$ en la ecuación maestra.
    *   **Aproximaciones:**
        *   *Límite Hidrodinámico ($\omega \to 0$):* Para el modo más bajo en ciertas simetrías, $\omega = -i D k^2$, donde $D$ depende de la temperatura.
        *   *Agujeros Negros BTZ (3D):* Es la única excepción donde existe una fórmula analítica cerrada exacta para los QNM en términos de $r_+$ y $r_-$.

---

### 3. Predicción Práctica de QNM desde la Geometría

En la práctica, para "predecir" o ajustar QNM sin resolver la ecuación diferencial completa en cada paso, se utilizan los siguientes enfoques jerárquicos:

1.  **Aproximación Eikonal (Óptica Geométrica):**
    Para números de onda angulares grandes ($l \gg 1$), los QNM están determinados por las propiedades de la esfera de fotones (geodésicas nulas inestables):
    $$ \omega_{n} \approx \Omega_c l - i (n + 1/2) \lambda_L $$
    *   $\Omega_c$: Frecuencia angular orbital en la esfera de fotones.
    *   $\lambda_L$: Exponente de Lyapunov (tasa de divergencia de geodésicas).
    *   *Uso:* Permite estimar la parte real e imaginaria basándose puramente en la geometría de la geodésica nula, sin resolver la ecuación de ondas.

2.  **Escalas Universales:**
    Se asume que $\omega_R \sim T_{Hawking}$ y $\omega_I \sim T_{Hawking}$. Se utilizan relaciones de escala adimensionales $\frac{\omega}{T}$. Cualquier desviación de la linealidad con $T$ indica correcciones de curvatura superior o materia exótica.

3.  **Dependencia de Modos:**
    Se utiliza casi exclusivamente el modo fundamental ($n=0$). Los sobretonos ($n > 0$) tienen partes imaginarias mucho mayores (decaimiento rápido) y son exponencialmente difíciles de extraer de una señal ruidosa o numérica, aunque contienen información sobre la estructura cerca del horizonte ("pelo" del agujero negro).

---

### 4. Clasificación Epistemológica de las Relaciones

*   **(A) Derivadas (Sólidas):**
    *   Equivalencia QNM $\leftrightarrow$ Polos del Propagador Retardado.
    *   Espectro de AdS Puro y BTZ.
    *   Límite Eikonal (relación con geodésicas nulas para $l \to \infty$).
    *   Límite Hidrodinámico (relación con viscosidad de cizalla $\eta/s$ en el régimen de longitud de onda larga).

*   **(B) Ampliamente Asumidas (Estándar):**
    *   Estabilidad lineal de la métrica (si $\text{Im}(\omega) < 0$ para todos los modos).
    *   Unicidad de la solución de la ecuación de ondas para condiciones de contorno fijas (salvo puntos algebraicamente especiales).

*   **(C) Conjeturales pero Razonables:**
    *   **Conjetura de Hod:** Relación entre la parte real asintótica de los QNM y la cuantización del área del horizonte (físicamente sugerente, pero matemáticamente no siempre robusta en todos los contextos AdS).
    *   Isospectralidad entre diferentes potenciales de materia (ciertas teorías pueden dar los mismos QNM, haciendo indistinguible la geometría).

*   **(D) Problemáticas / Falsas:**
    *   "Existe una relación lineal simple entre $\omega$ y la masa del campo escalar para $n$ bajo": **Falso**. La dependencia es no lineal y pasa a través del potencial efectivo.
    *   "Los QNM determinan unívocamente la métrica": **Problemático**. Es un problema inverso. Existen métricas "mimickers" que producen espectros QNM casi idénticos dentro de la precisión numérica.

---

### 5. Análisis Epistemológico de la Selección de Geometría

**¿Es razonable seleccionar geometría usando (f, τ, Q)?**
Sí, es operacionalmente razonable, pero es un **problema inverso mal condicionado**.

1.  **Degeneración:** Un conjunto dado de $(f, \tau)$ puede ser generado por:
    *   Un agujero negro de Schwarzschild-AdS con temperatura $T_1$.
    *   Un agujero negro con carga (Reissner-Nordström-AdS) con temperatura $T_2$.
    *   Una teoría de gravedad modificada (ej. Gauss-Bonnet).
    Sin información externa (como la ecuación de estado de la CFT o simetrías conservadas), los QNM por sí solos no fijan unívocamente la teoría subyacente, solo la *clase de equivalencia* de potenciales efectivos cerca del máximo (esfera de fotones).

2.  **Pérdida de Información:**
    Los QNM están dominados por el comportamiento del potencial en el pico (cerca de la esfera de fotones) y las condiciones de contorno. Se pierde casi toda la información sobre la estructura profunda cerca del horizonte (para modos $n$ bajos) y sobre la estructura asintótica detallada lejos del agujero negro (si el decaimiento es muy rápido).

**Conclusión:** El pipeline puede descartar familias de soluciones incorrectas, pero difícilmente podrá identificar una métrica única sin *priors* teóricos fuertes sobre la acción de gravedad permitida.

---

### 6. Referencias Clave (Lectura Obligatoria)

1.  **Son & Starinets (2002):** *Minkowski-space correlators in AdS/CFT...*
    *   Establece la receta prescriptiva rigurosa para calcular funciones de Green retardadas y su relación con condiciones de contorno en el horizonte.
2.  **Berti, Cardoso & Starinets (2009):** *Quasinormal modes of black holes and black branes.*
    *   La revisión enciclopédica estándar. El capítulo sobre AdS/CFT es fundamental para entender el diccionario.
3.  **Horowitz & Hubeny (2000):** *Quasinormal modes of AdS black holes and the approach to thermal equilibrium.*
    *   Trabajo seminal que demuestra numéricamente cómo los QNM describen la termalización en la CFT.
4.  **Festuccia & Liu (2005):** *Excursions beyond the horizon: Black hole singularities in Yang-Mills theories.*
    *   Proporciona intuición sobre cómo las singularidades y la geometría interior se codifican (o no) en los correladores de frontera.
5.  **Konoplya & Zhidenko (2011):** *Quasinormal modes of black holes: From astrophysics to string theory.*
    *   Excelente para métodos de aproximación (WKB/Eikonal) que necesitarás para tu pipeline si quieres evitar resolver ecuaciones diferenciales completas.
6.  **Cardoso et al. (2016):** *Is the gravitational-wave ringdown a probe of the event horizon?*
    *   Critica epistemológica sobre qué parte de la geometría (esfera de fotones vs horizonte) estamos viendo realmente con los QNM; crucial para tu pregunta 5.