Este es un problema fascinante que conecta la física matemática (problemas espectrales inversos) con el aprendizaje profundo (consistencia cíclica y *manifold learning*).

El fallo en la consistencia cíclica ($r \to \hat{\Delta} \to \hat{r} \neq r$) en un modelo AdS/Hard-wall, a pesar de parecer un sistema "simple" (ecuaciones tipo Bessel), sugiere que estás lidiendo con un **problema mal condicionado** numéricamente o una **pérdida de inyectividad efectiva** en el espacio de características.

Aquí tienes un desglose detallado de las causas, referencias teóricas y soluciones prácticas.

---

### 1. Identificabilidad y Biyectividad ($\Delta \leftrightarrow \{r_n\}$)

En el modelo Hard-wall estándar, los modos $M_n$ vienen dados por los ceros de funciones de Bessel $J_\nu(M_n z_{IR}) = 0$ (o combinaciones lineales dependiendo de las condiciones de contorno). El orden $\nu$ depende directamente de $\Delta$ (e.g., $\nu = |\Delta - 2|$ para escalares).

**El análisis matemático:**
*   **Monotonicidad:** Los ceros de Bessel $j_{\nu, n}$ son funciones monótonas crecientes del orden $\nu$. Por lo tanto, teóricamente, el mapeo $\Delta \to \{r_n\}$ es **inyectivo**. No debería haber verdadera isospectralidad (dos $\Delta$ distintos produciendo el mismo espectro exacto) si el potencial está fijo como $1/z^2$ + Hard-wall.
*   **Teoremas de Unicidad:** El **Teorema de Borg-Levinson** es la referencia clásica. Establece que, bajo ciertas condiciones, el espectro (o dos espectros con diferentes CC) determina unívocamente el potencial. En tu caso, el "potencial" está parametrizado solo por $\Delta$. Si el mapa directo es inyectivo, el inverso existe.

**¿Por qué falla el inverso entonces?**
El problema no es la biyectividad teórica, sino la **identificabilidad práctica**. La dependencia de los ratios $r_n$ respecto a $\Delta$ se vuelve extremadamente débil para ciertos regímenes.

*   **Referencias clave:**
    *   *Borg-Levinson Theorem* (Inverse Sturm-Liouville problems).
    *   *Ambarzumian's Theorem* (el primer resultado de unicidad espectral).
    *   Busca papers sobre "Inverse Spectral Problems for Bessel Operators".

### 2. Degeneraciones, Isospectralidad y "Asymptotic Insensitivity"

Aquí es donde probablemente reside tu error de ciclo.

**El problema de la asíntota de Weyl:**
Para cualquier problema de Sturm-Liouville regular, los autovalores asintóticos ($n \to \infty$) siguen la ley de Weyl:
$$ M_n^2 \sim \frac{\pi^2 n^2}{z_{IR}^2} + \mathcal{O}(1) $$
Tus *features* son ratios $r_n = M_n^2 / M_0^2$. Para $n$ grande:
$$ r_n \approx \frac{n^2}{1^2} = n^2 $$
Observa que $\Delta$ desaparece en el término dominante. La información sobre $\Delta$ está contenida en los términos de orden inferior (correcciones sub-dominantes).

**Consecuencia para la Red Neuronal:**
Si tu vector de entrada $\{r_n\}$ contiene muchos modos altos, la red aprende que $r_n \approx n^2$. La dependencia en $\Delta$ es una "señal pequeña" montada sobre una "tendencia grande".
*   Si usas MSE estándar en $r$, el error está dominado por la magnitud de los $r_n$ grandes. La red minimiza el error prediciendo la tendencia $n^2$ e ignorando las variaciones sutiles debidas a $\Delta$.
*   Esto crea una **isospectralidad efectiva**: muchos valores de $\Delta$ producen $r_n$ numéricamente indistinguibles para la red dado el ruido o la precisión flotante.

### 3. Enfoques para Consistencia Cíclica ($f \circ g \approx I$)

Si $r \to \hat{\Delta}$ es tu modelo inverso (Encoder) y $\Delta \to \hat{r}$ es el directo (Decoder/Physics), el fallo en el cierre indica que el Encoder está mapeando distintos $r$ (que difieren solo sutilmente por $\Delta$) al mismo $\hat{\Delta}$ promedio, o viceversa.

**Estrategias de corrección:**

A. **Entrenamiento Híbrido (Physics-Informed):**
No entrenes el inverso solo con pares supervisados. Entrena el ciclo completo end-to-end.
$$ \mathcal{L} = \lambda_1 || \Delta_{GT} - \text{Inv}(r) ||^2 + \lambda_2 || r - \text{Fwd}(\text{Inv}(r)) ||^2 $$
Donde $\text{Fwd}$ puede ser la función de Bessel exacta (diferenciable mediante librerías como JAX o PyTorch) o un modelo surrogado pre-entrenado. Esto fuerza al modelo inverso a encontrar un $\Delta$ que recupere el espectro.

B. **Ponderación de Características (Feature Reweighting):**
Dado que la información de $\Delta$ está concentrada en los primeros modos ($n=0, 1, 2$), debes forzar al modelo a prestarles atención.
No uses el vector $r$ "crudo". Usa un mecanismo de atención o pesos manuales en la *loss*:
$$ \text{Loss} = \sum_n w_n (r_n - \hat{r}_n)^2, \quad \text{donde } w_n \sim \frac{1}{n^p} $$

C. **Arquitecturas Invertibles (INNs):**
En lugar de una MLP estándar, usa una **Invertible Neural Network (INN)** o *Normalizing Flows* (RealNVP). Estas garantizan por construcción que el mapeo es biyectivo y preservan la información, evitando el colapso de modos.

*   **Referencias:**
    *   *Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (CycleGAN)* - Concepto base.
    *   *Ardizzone et al., "Analyzing Inverse Problems with Invertible Neural Networks"* - **Lectura obligatoria para tu caso**.

### 4. Sensibilidad, Condicionamiento y Métricas

El problema de condicionamiento se puede visualizar mediante el Jacobiano $J = \partial r / \partial \Delta$.
Si $||J|| \to 0$ (como ocurre para $n$ grandes), el inverso es inestable ($1/0$).

**Recomendaciones de Métricas:**

1.  **Evita RMSE en crudo para el espectro:** Dado que $r_n$ crece cuadráticamente, un error del 1% en el modo $n=10$ contribuye 100 veces más al MSE que un error del 1% en el modo fundamental $n=1$. Pero el modo fundamental tiene *más* información física sobre $\Delta$.
2.  **Usa Métricas Relativas o Logarítmicas:**
    *   Entrena sobre $\log(r_n)$ en lugar de $r_n$. Esto aplana la escala y da igual importancia a las correcciones relativas.
    *   Métrica MAPE (Mean Absolute Percentage Error).
3.  **Métrica de Consistencia:**
    Para evaluar el ciclo, no mires solo el promedio. Mira el error en el parámetro físico $\Delta$.
    $$ \text{Error}_{\text{cycle}} = || \Delta - \text{Inv}(\text{Fwd}(\Delta)) || $$

### Resumen del Plan de Acción

1.  **Diagnóstico:** Grafica $r_n$ vs $\Delta$ para $n=1$ y $n=10$. Verás que para $n=10$ la curva es casi plana. Eso confirma el mal condicionamiento.
2.  **Pre-procesamiento:** Transforma tus inputs a $\rho_n = \log(r_n)$ o simplemente usa solo los primeros $k$ modos (e.g., $n=1...5$) donde la relación señal/ruido respecto a $\Delta$ es alta. Descartar modos altos *mejora* la inversibilidad en este contexto.
3.  **Modelo:** Implementa una pérdida cíclica explícita usando un solucionador de Bessel diferenciable en el bucle de entrenamiento.

### Referencias Específicas para búsqueda

*   **Matemáticas:** "Inverse Sturm-Liouville problem uniqueness", "Borg-Levinson theorem application".
*   **AdS/QCD:** "Holographic QCD spectrum scaling dimension dependence", "Light-front holography spectral inverse".
*   **ML aplicado:** "Solving inverse physics problems with Invertible Neural Networks", "Physics-informed deep learning for inverse spectral problems".