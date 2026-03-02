# Método Brunete

## Geometría de la información de Fisher para ringdown gravitacional: estructura conforme, curvatura y diagnóstico de contaminación por PSD

**Versión:** 1.0  
**Fecha:** 2026-02-27  
**Contexto:** BASURIN — Fase 6 (Teoría)

---

## 0. Resumen ejecutivo

Este documento desarrolla la geometría de la información de Fisher para una señal de ringdown (sinusoide amortiguada) en el espacio de parámetros observable $(f, \tau)$ y su transformación a $(\ln f, \ln Q)$, que es el sistema operativo de BASURIN.

**Resultados principales:**

1. La Fisher information matrix es **diagonal en $(f, \tau)$ para cualquier PSD real** (resultado exacto, Proposición 1).
2. La métrica en $(\ln f, \ln Q)$ se factoriza como $g = \Omega \cdot \bar{g}$, donde $\bar{g}$ es una **métrica hiperbólica de curvatura constante $\bar{K} = -1$** (Proposición 2).
3. Toda la información sobre la PSD del detector entra exclusivamente por el factor conforme escalar $\Omega$ (Proposición 3).
4. La curvatura gaussiana admite una separación limpia señal/instrumento con un criterio cuantitativo de contaminación $\chi_{\text{PSD}}$ (Teorema 1).
5. La expansión perturbativa en $1/Q$ se resuma mediante una familia de integrales gaussianamente regularizadas $\mathcal{J}_0(\sigma)$ con forma cerrada en $\text{erfc}$ (Proposición 4).

---

## 1. Modelo de señal y transformada de Fourier

### 1.1 Señal single-pole

Componente analítica (positivo-frecuencia) del ringdown para $t \geq 0$:

$$h_+(t) = \frac{A}{2}\,e^{-t/\tau}\,e^{i(2\pi f t + \phi_0)}\,\Theta(t) \tag{1.1}$$

Transformada de Fourier (polo simple cerca de la resonancia $\nu \approx f$):

$$\tilde{h}(\nu) = \frac{A\tau}{2}\,\frac{e^{i\phi_0}}{1 + 2\pi i(\nu - f)\tau} \tag{1.2}$$

Espectro de potencia (perfil Lorentziano):

$$|\tilde{h}(\nu)|^2 = \frac{A^2\tau^2}{4}\,\frac{1}{1 + [2\pi(\nu-f)\tau]^2} \tag{1.3}$$

Ancho a media altura en frecuencia: $\Gamma = 1/(2\pi\tau)$.

### 1.2 Variable espectral adimensional

Definimos la variable adimensional centrada en la resonancia:

$$z \equiv 2\pi\tau(\nu - f), \qquad d\nu = \frac{dz}{2\pi\tau} \tag{1.4}$$

y el parámetro de resolución espectral:

$$\epsilon \equiv \frac{1}{2\pi f\tau} = \frac{1}{2Q}, \qquad Q \equiv \pi f\tau \tag{1.5}$$

que mide el ancho de línea del Lorentziano relativo a la frecuencia central.

---

## 2. Fisher information matrix en $(f, \tau)$

### 2.1 Definición

$$\Gamma_{ij} = 4\,\text{Re}\int_0^\infty \frac{\partial_i\tilde{h}^*(\nu)\;\partial_j\tilde{h}(\nu)}{S_n(\nu)}\,d\nu \tag{2.1}$$

### 2.2 Derivadas de la señal

$$\partial_f\tilde{h} = \frac{A}{2}\,e^{i\phi_0}\,\frac{2\pi i\,\tau^2}{(1+iz)^2} \tag{2.2a}$$

$$\partial_\tau\tilde{h} = \frac{A}{2}\,e^{i\phi_0}\,\frac{1}{(1+iz)^2} \tag{2.2b}$$

**Identidad algebraica clave:**

$$\partial_f\tilde{h} = 2\pi i\tau^2\,\partial_\tau\tilde{h} \tag{2.3}$$

Las dos derivadas difieren solo por un factor puramente imaginario.

### 2.3 Proposición 1 (diagonalidad exacta)

**Enunciado.** Para cualquier PSD real $S_n(\nu) > 0$, la Fisher information matrix en coordenadas $(f, \tau)$ es diagonal:

$$\Gamma_{f\tau} = 0 \quad\text{exactamente} \tag{2.4}$$

**Demostración.** De la identidad (2.3):

$$\partial_f\tilde{h}^*\,\partial_\tau\tilde{h} = (-2\pi i\tau^2)\,|\partial_\tau\tilde{h}|^2$$

El factor $(-2\pi i\tau^2)$ es puramente imaginario y $|\partial_\tau\tilde{h}|^2 / S_n(\nu)$ es real. Por tanto $\text{Re}(\partial_f\tilde{h}^*\,\partial_\tau\tilde{h}/S_n) = 0$ para cada $\nu$, y la integral (2.1) se anula. $\square$

**Corolario.** Las coordenadas $(f, \tau)$ son ejes principales de la Fisher para cualquier PSD real. No se necesitan coordenadas rotadas ni elipsoidales en este plano.

### 2.4 Elementos diagonales: forma exacta

Módulos cuadrados:

$$|\partial_f\tilde{h}|^2 = \frac{A^2\pi^2\tau^4}{(1+z^2)^2}, \qquad |\partial_\tau\tilde{h}|^2 = \frac{A^2}{4(1+z^2)^2} \tag{2.5}$$

Definimos el funcional central:

$$J(f,\tau) \equiv \int_{-\infty}^{\infty}\frac{dz}{(1+z^2)^2}\,\frac{1}{S_n\!\left(f + \frac{z}{2\pi\tau}\right)} \tag{2.6}$$

Entonces:

$$\Gamma_{ff} = 2A^2\pi\tau^3\,J(f,\tau), \qquad \Gamma_{\tau\tau} = \frac{A^2}{2\pi\tau}\,J(f,\tau) \tag{2.7}$$

**Ratio exacto (cualquier PSD):**

$$\frac{\Gamma_{ff}}{\Gamma_{\tau\tau}} = 4\pi^2\tau^4 = \frac{4Q^2}{f^2} \tag{2.8}$$

### 2.5 Caso PSD constante

Con $S_n(\nu) = S = \text{const}$, $J = \pi/(2S)$. Definiendo la SNR óptima (en PSD plana coincide con la SNR local $\rho_0$ del caso general):

$$\rho_0^2 \equiv 4\int_0^\infty\frac{|\tilde{h}|^2}{S_n}\,d\nu = \frac{A^2\tau}{2S} \tag{2.9}$$

Los elementos Fisher son:

$$\Gamma_{ff} = 2\pi^2\rho_0^2\tau^2, \qquad \Gamma_{\tau\tau} = \frac{\rho_0^2}{2\tau^2}, \qquad \Gamma_{f\tau} = 0 \tag{2.10}$$

Incertidumbres Cramér-Rao:

$$\sigma_f = \frac{1}{\pi\tau\rho_0\sqrt{2}}, \qquad \sigma_\tau = \frac{\tau\sqrt{2}}{\rho_0} \tag{2.11}$$

En forma relativa:

$$\frac{\sigma_f}{f} = \frac{1}{\pi Q\rho_0\sqrt{2}}, \qquad \frac{\sigma_\tau}{\tau} = \frac{\sqrt{2}}{\rho_0} \tag{2.12}$$

La precisión en frecuencia mejora con $Q$ (modos espectralmente afilados); la precisión en $\tau$ depende solo de la SNR.

---

## 3. Métrica en coordenadas logarítmicas

### 3.1 Coordenadas $(\ln f, \ln\tau)$

Con $(u, v) = (\ln f, \ln\tau)$ y la transformación tensorial $g_{ab} = (\partial\theta^i/\partial\theta'^a)(\partial\theta^j/\partial\theta'^b)\Gamma_{ij}$:

$$g_{uu} = f^2\Gamma_{ff} = 2\rho_0^2 Q^2 \tag{3.1a}$$

$$g_{vv} = \tau^2\Gamma_{\tau\tau} = \frac{\rho_0^2}{2} \tag{3.1b}$$

$$g_{uv} = f\tau\Gamma_{f\tau} = 0 \tag{3.1c}$$

Forma matricial (PSD plana):

$$g^{(\ln f,\ln\tau)} = \rho_0^2\begin{pmatrix} 2Q^2 & 0 \\ 0 & 1/2 \end{pmatrix} \tag{3.2}$$

Incertidumbres en coordenadas logarítmicas:

$$\sigma_{\ln f} = \frac{1}{\rho_0\sqrt{2}\,Q}, \qquad \sigma_{\ln\tau} = \frac{\sqrt{2}}{\rho_0}, \qquad \frac{\sigma_{\ln\tau}}{\sigma_{\ln f}} = 2Q \tag{3.3}$$

La elipse de error es estrecha en $\ln f$ y ancha en $\ln\tau$ para modos de alto $Q$.

### 3.2 Coordenadas $(\ln f, \ln Q)$

Con $(u, w) = (\ln f, \ln Q)$ y $Q = \pi f\tau$, $\tau = Qe^{-u}/\pi$, $v = w - u - \ln\pi$:

$$dv = dw - du \tag{3.4}$$

Sustituyendo en $ds^2 = \Omega(4Q^2\,du^2 + dv^2)$:

$$ds^2 = \Omega\left[(4Q^2 + 1)\,du^2 - 2\,du\,dw + dw^2\right] \tag{3.5}$$

Forma matricial:

$$g^{(\ln f,\ln Q)} = \Omega(u,w)\begin{pmatrix} 4Q^2+1 & -1 \\ -1 & 1 \end{pmatrix}, \qquad Q = e^w \tag{3.6}$$

El término cruzado $g_{uw} = -\Omega$ es **puramente geométrico** (del cambio $\tau \to Q$), no proviene de la PSD.

Correlación adimensional:

$$r = \frac{g_{uw}}{\sqrt{g_{uu}\,g_{ww}}} = \frac{-1}{\sqrt{4Q^2+1}} \tag{3.7}$$

Ángulo de rotación de la elipse de error:

$$\tan 2\alpha = \frac{2g_{uw}}{g_{uu}-g_{ww}} = -\frac{1}{2Q^2} \tag{3.8}$$

| $Q$ | $r$ | $\alpha$ |
|-----|-----|----------|
| $4.3$ (modo fundamental GW150914) | $-0.115$ | $-0.8°$ |
| $2$ (overtone típico) | $-0.24$ | $-3.6°$ |

### 3.3 Factor conforme y factorización

Definimos:

$$\Omega(u,v) \equiv \frac{A^2\tau}{2\pi}\,J(f,\tau) = \frac{A^2 e^v}{2\pi}\,J(e^u, e^v) \tag{3.9}$$

La métrica en $(\ln f, \ln\tau)$ toma la forma:

$$ds^2 = \Omega(u,v)\left[4Q^2\,du^2 + dv^2\right] \tag{3.10}$$

y en $(\ln f, \ln Q)$:

$$g = \Omega(u,w)\,\bar{g}(w) \tag{3.11}$$

con la **métrica base**:

$$\bar{g}(w) = \begin{pmatrix} 4e^{2w}+1 & -1 \\ -1 & 1 \end{pmatrix}, \qquad \det\bar{g} = 4e^{2w} = 4Q^2 \tag{3.12}$$

---

## 4. Curvatura de la métrica base

### 4.1 Proposición 2 (hiperbolicidad)

**Enunciado.** La métrica base $\bar{g}$ en $(\ln f, \ln Q)$ tiene curvatura gaussiana constante:

$$\bar{K} = -1, \qquad \bar{R} = -2 \tag{4.1}$$

**Demostración.** Usamos la fórmula de Brioschi para $ds^2 = E\,du^2 + 2F\,du\,dw + G\,dw^2$ con $E = 4e^{2w}+1$, $F = -1$, $G = 1$.

Derivadas no nulas: $E_w = 8e^{2w}$, $E_{ww} = 16e^{2w}$. Todas las demás derivadas de $E$, $F$, $G$ se anulan ($F$ y $G$ son constantes, $E$ no depende de $u$).

Primer determinante de Brioschi:

$$D_1 = \begin{vmatrix} -E_{ww}/2 & E_u/2 & E_w/2 \\ F_u - E_w/2 & E & F \\ G_w/2 & F & G \end{vmatrix} = \begin{vmatrix} -8e^{2w} & 0 & 4e^{2w} \\ -4e^{2w} & 4e^{2w}+1 & -1 \\ 0 & -1 & 1 \end{vmatrix}$$

Expandiendo por la primera columna:

$$D_1 = -8e^{2w}\left[(4e^{2w}+1)\cdot 1 - (-1)^2\right] = -8e^{2w}\cdot 4e^{2w} = -32e^{4w}$$

Segundo determinante:

$$D_2 = \begin{vmatrix} 0 & E_w/2 & F_w - G_u/2 \\ E_w/2 & E & F \\ F_w - G_u/2 & F & G \end{vmatrix} = \begin{vmatrix} 0 & 4e^{2w} & 0 \\ 4e^{2w} & 4e^{2w}+1 & -1 \\ 0 & -1 & 1 \end{vmatrix}$$

$$D_2 = -4e^{2w}\cdot 4e^{2w} = -16e^{4w}$$

Curvatura:

$$\bar{K} = \frac{D_1 - D_2}{(EG-F^2)^2} = \frac{-32e^{4w} + 16e^{4w}}{(4e^{2w})^2} = \frac{-16e^{4w}}{16e^{4w}} = -1 \quad\square \tag{4.2}$$

**Verificación por Christoffel.** Metric inversa: $\bar{g}^{uu} = 1/(4Q^2)$, $\bar{g}^{uw} = 1/(4Q^2)$, $\bar{g}^{ww} = (4Q^2+1)/(4Q^2)$.

Símbolos de Christoffel no nulos ($(\bar{g}_{uu})_w = 8e^{2w}$ es la única derivada no nula):

$$\bar{\Gamma}^u_{uu} = -1, \quad \bar{\Gamma}^w_{uu} = -(4e^{2w}+1), \quad \bar{\Gamma}^u_{uw} = 1, \quad \bar{\Gamma}^w_{uw} = 1 \tag{4.3}$$

Componente de Riemann: $R_{uwuw} = -4e^{2w}$, luego $\bar{K} = R_{uwuw}/\det\bar{g} = -4e^{2w}/(4e^{2w}) = -1$. $\checkmark$

### 4.2 Nota

La métrica $\bar{g}$ es isométrica al plano hiperbólico $\mathbb{H}^2$ (de curvatura $-1$) por cambio de coordenadas. En las coordenadas $(u,w)$ no adopta la forma canónica de Poincaré, pero la propiedad $\bar{K} = -1$ constante garantiza la isometría.

---

## 5. Curvatura con PSD plana

### 5.1 Transformación conforme en 2D

Con $g = e^{2\varphi}\bar{g}$, donde $\Omega = e^{2\varphi}$ y $\varphi = \omega/2$ con $\omega = \ln\Omega$:

$$K = \frac{1}{\Omega}\left(\bar{K} - \bar{\Delta}\varphi\right) = \frac{1}{\Omega}\left(-1 - \bar{\Delta}\varphi\right) \tag{5.1}$$

donde $\bar{\Delta}$ es el Laplace-Beltrami respecto de $\bar{g}$.

### 5.2 Laplaciano $\bar{\Delta}$ explícito

$$\bar{\Delta}\omega = \frac{1}{\sqrt{\bar{g}}}\,\partial_a\!\left(\sqrt{\bar{g}}\,\bar{g}^{ab}\partial_b\omega\right) \tag{5.2}$$

Con $\sqrt{\bar{g}} = 2Q = 2e^w$ y la métrica inversa:

$$\bar{\Delta}\omega = \frac{\omega_{uu} + 2\omega_{uw} + \omega_{ww} - \omega_u - \omega_w}{4Q^2} + \omega_{ww} + \omega_w \tag{5.3}$$

### 5.3 Evaluación para PSD plana

Con $S_n = \text{const}$ y $\Omega_0 = A^2\tau/(4S_n) = \rho_0^2/2$:

$$\omega^{(0)} = \text{const} + v = \text{const} + w - u - \ln\pi \tag{5.4}$$

Derivadas:

$$\omega_u = -1, \quad \omega_w = 1, \quad \omega_{uu} = \omega_{uw} = \omega_{ww} = 0 \tag{5.5}$$

Laplaciano:

$$\bar{\Delta}\omega = \frac{0+0+0-(-1)-1}{4Q^2} + 0 + 1 = 1 \tag{5.6}$$

$$\bar{\Delta}\varphi = \frac{1}{2} \tag{5.7}$$

Curvatura:

$$K = \frac{1}{\Omega_0}\left(-1 - \frac{1}{2}\right) = -\frac{3}{2\Omega_0} = -\frac{3}{\rho_0^2} \tag{5.8}$$

$$R = 2K = -\frac{6}{\rho_0^2} \tag{5.9}$$

### 5.4 Interpretación en tres capas

La curvatura total con PSD plana se descompone como:

| Capa | Contribución a $K$ | Origen |
|------|---------------------|--------|
| Base hiperbólica | $\bar{K}/\Omega_0 = -1/\Omega_0$ | Geometría de $(\ln f, \ln Q)$ con $Q = \pi f\tau$ |
| Factor conforme (señal) | $-\bar{\Delta}\varphi/\Omega_0 = -1/(2\Omega_0)$ | $\Omega \propto \tau$: la SNR crece con $\tau$ |
| **Total** | $-3/(2\Omega_0) = -3/\rho_0^2$ | |

La geometría es hiperbólica ($K < 0$). Alta SNR $\Rightarrow$ curvatura pequeña $\Rightarrow$ aproximación plana (Mahalanobis) válida. Baja SNR $\Rightarrow$ curvatura significativa $\Rightarrow$ Mahalanobis subestima separaciones.

---

## 6. PSD variable: funcional $J$ y resummación

### 6.1 Forma exacta de $J$

$$S_n(f)\,J(f,\tau) = \int_{-\infty}^{\infty}\frac{dz}{(1+z^2)^2}\,\exp\!\left(-\phi(z)\right) \tag{6.1}$$

donde:

$$\phi(z) \equiv \ln S_n\!\left(f + \frac{z}{2\pi\tau}\right) - \ln S_n(f) = \sum_{k \geq 1}\ell_k\,(\epsilon z)^k \tag{6.2}$$

con los coeficientes logarítmicos:

$$\ell_k \equiv \frac{f^k}{k!}\,\frac{d^k\ln S_n}{df^k}\bigg|_f \tag{6.3}$$

En particular: $\ell_1 = s_1$, $\ell_2 = s_2 = \kappa/2$.

### 6.2 Separación por paridad

Con el kernel par $K(z) = (1+z^2)^{-2}$ y la descomposición $\phi = \phi_+ + \phi_-$ (par/impar):

$$S_n(f)\,J = \int K(z)\,e^{-\phi_+(z)}\cosh(\phi_-(z))\,dz \tag{6.4}$$

El término $\sinh(\phi_-)$ es impar y se anula al integrar con el kernel par.

### 6.3 Proposición 4 (resummación cuadrática)

A $O(\epsilon^2)$, reteniendo solo el término cuadrático $\phi_+ \approx s_2\epsilon^2 z^2$ y $\cosh(\phi_-) \approx 1 + s_1^2\epsilon^2 z^2/2$:

$$S_n(f)\,J = \mathcal{J}_0(\sigma) + \frac{s_1^2\epsilon^2}{2}\,\mathcal{J}_1(\sigma) + O(\epsilon^4) \quad\text{(a $\sigma$ fijo)} \tag{6.5}$$

donde $\sigma \equiv s_2\epsilon^2 = \kappa/(8Q^2)$ y la familia regularizada:

$$\mathcal{J}_n(\sigma) \equiv \int_{-\infty}^{\infty}\frac{z^{2n}}{(1+z^2)^2}\,e^{-\sigma z^2}\,dz \tag{6.6}$$

**Forma cerrada de $\mathcal{J}_0$:**

$$\mathcal{J}_0(\sigma) = \pi\left[\left(\sigma + \frac{1}{2}\right)e^{\sigma}\,\text{erfc}(\sqrt{\sigma}) - \sqrt{\frac{\sigma}{\pi}}\right] \tag{6.7}$$

Verificación: $\mathcal{J}_0(0) = \pi/2$. $\checkmark$

**Expansión para $\sigma$ pequeño:**

$$\mathcal{J}_0(\sigma) = \frac{\pi}{2}\left(1 - \sigma + O(\sigma^{3/2})\right) \tag{6.8}$$

$$\mathcal{J}_1(\sigma) = \frac{\pi}{2}\left(1 + O(\sigma^{1/2})\right) \tag{6.9}$$

Al reexpandir en $\sigma \ll 1$, $\mathcal{J}_0(\sigma)$ contiene términos $O(\sigma^{3/2}) \sim O(\epsilon^3)$; por eso el residuo estructural de (6.5) es $O(\epsilon^4)$ pero al expandir $\mathcal{J}_0$ aparecen contribuciones $O(\epsilon^3)$.

Sustituyendo en (6.5):

$$S_n(f)\,J = \frac{\pi}{2}\left[1 + \beta_{[2]}\epsilon^2 + O(\epsilon^3)\right] \quad\text{(porque $\sigma^{3/2} \sim \epsilon^3$)} \tag{6.10}$$

con $\beta_{[2]} \equiv s_1^2/2 - s_2$, $\epsilon = 1/(2Q)$.

### 6.4 Factor conforme corregido

$$\Omega = \Omega_0(1 + \delta), \qquad \delta = \frac{\beta_{[2]}}{4Q^2} \tag{6.11}$$

$$\omega = \omega^{(0)} + \delta\omega, \qquad \delta\omega = \frac{\beta_{[2]}(u)}{4Q^2} \tag{6.12}$$

### 6.5 Régimen del parámetro $\sigma$

El parámetro $\sigma = \kappa/(8Q^2)$ gobierna la validez de la expansión perturbativa:

| Régimen | Condición | Comportamiento | Diagnóstico |
|---------|-----------|----------------|-------------|
| Perturbativo | $\sigma \ll 1$ | Usar fórmula cerrada de $K$ (Teorema 1) | PSD suave o $Q$ alto |
| Intermedio | $\sigma \sim 1$ | Usar $\mathcal{J}_0(\sigma)$ completo | Overtones cerca de líneas |
| Colapsado | $\sigma \gg 1$ | $\mathcal{J}_0 \sim 3\sqrt{\pi}/(8\sigma^{3/2})$: $J$ suprimido $\Rightarrow$ $\Omega$ muy pequeña $\Rightarrow$ métrica degenerada | Inferencia no fiable |

### 6.6 Notación consolidada para las derivadas logarítmicas de la PSD

$$s_1 \equiv f\,\frac{d\ln S_n}{df}\bigg|_f \tag{6.13}$$

$$\kappa \equiv f^2\,\frac{d^2\ln S_n}{df^2}\bigg|_f = 2s_2 \tag{6.14}$$

$$s_1 + \kappa = f\,\frac{ds_1}{df} \tag{6.15}$$

**Nota sobre ley de potencias.** Para $S_n \propto f^\alpha$: $s_1 = \alpha$ (constante), $\kappa = -\alpha$, $f\,ds_1/df = 0$, pero $s_1^2 + \kappa = \alpha^2 - \alpha \neq 0$ salvo $\alpha \in \{0, 1\}$.

---

## 7. Curvatura con PSD variable (Teorema principal)

### 7.1 Derivadas de $\omega$ a $O(\epsilon^2)$

*Nota de coordenadas:* Las ecuaciones (7.1)–(7.5) se calculan en $(u,v) = (\ln f, \ln\tau)$, donde la métrica es diagonal y las derivadas de $\delta\omega$ respecto de $v$ son directas. Las ecuaciones (7.6)–(7.8) traducen estos resultados a $(u,w) = (\ln f, \ln Q)$ vía el Laplaciano $\bar{\Delta}$ de la métrica base $\bar{g}(w)$.

**Parte $\omega^{(0)}$ (macroscópica), en $(u,w)$:**

$$\omega_u^{(0)} = -1 - s_1, \qquad \omega_w^{(0)} = 1 \tag{7.1}$$

$$\omega_{uu}^{(0)} = -(s_1 + \kappa), \qquad \omega_{ww}^{(0)} = 0, \qquad \omega_{uw}^{(0)} = 0 \tag{7.2}$$

**Parte $\delta\omega = \beta_{[2]}(u)/(4Q^2)$ (corrección de ancho de línea), en $(u,v)$:**

Con $\partial_v(1/Q^2) = -2/Q^2$ y $\partial_{vv}(1/Q^2) = 4/Q^2$:

$$(\delta\omega)_v = -\frac{\beta_{[2]}}{2Q^2}, \qquad (\delta\omega)_{vv} = \frac{\beta_{[2]}}{Q^2} \tag{7.3}$$

**Combinaciones relevantes (en variables $(u,v)$, para uso en la fórmula exacta de $K$ de §5b):**

$$\omega_{vv} + \omega_v + 2 = 3 + \frac{\beta_{[2]}}{2Q^2} \tag{7.4}$$

$$\omega_{uu} - \omega_u = -\kappa + O(1/Q^2) \tag{7.5}$$

Las correcciones $O(1/Q^2)$ en (7.5) entran en $K$ multiplicadas por $1/(8Q^2)$, dando $O(1/Q^4)$, y se descartan consistentemente.

### 7.2 Laplaciano $\bar{\Delta}\varphi$ a $O(\epsilon^2)$ (en variables $(u,w)$)

**Parte macroscópica:**

$$\bar{\Delta}\omega^{(0)} = \frac{-(s_1+\kappa) + 0 + 0 - (-1-s_1) - 1}{4Q^2} + 0 + 1 = \frac{-\kappa}{4Q^2} + 1 \tag{7.6}$$

**Corrección $\delta\omega$:** las derivadas dominantes son las del grupo no suprimido por $1/(4Q^2)$:

$$\bar{\Delta}(\delta\omega)\big|_{\text{dom}} = (\delta\omega)_{ww} + (\delta\omega)_w = \frac{\beta_{[2]}}{Q^2} - \frac{\beta_{[2]}}{2Q^2} = \frac{\beta_{[2]}}{2Q^2} \tag{7.7}$$

**Total:**

$$\bar{\Delta}\varphi = \frac{1}{2} + \frac{s_1^2 - 2\kappa}{8Q^2} + O(1/Q^4) \tag{7.8}$$

### 7.3 Teorema 1 (curvatura con separación señal/PSD)

**Enunciado.** En coordenadas $(\ln f, \ln Q)$, la curvatura gaussiana del espacio de parámetros Fisher para una sinusoide amortiguada con PSD $S_n(f)$ es, a $O(1/Q^2)$:

$$K = -\frac{3}{\rho_0^2}\left(1 - \frac{s_1^2 + \kappa}{24Q^2}\right) + O(1/Q^4) \tag{7.9}$$

$$R = 2K \tag{7.10}$$

donde $\rho_0^2 = A^2\tau/(2S_n(f))$ es la SNR local, $s_1 = f\,d\ln S_n/df$, $\kappa = f^2\,d^2\ln S_n/df^2$, y $Q = \pi f\tau$.

**Demostración.** Sustituyendo (7.8) en (5.1) y expandiendo $1/\Omega = (1-\delta)/\Omega_0$:

$$K = \frac{1}{\Omega_0}(1-\delta)\left(-1 - \frac{1}{2} - \frac{s_1^2-2\kappa}{8Q^2}\right)$$

con $\delta = \beta_{[2]}/(4Q^2) = (s_1^2 - \kappa)/(8Q^2)$.

Primer bloque (numerador × denominador):

$$\left(3 + \frac{s_1^2-2\kappa}{4Q^2}\right)\left(1 - \frac{s_1^2-\kappa}{8Q^2}\right) = 3 + \frac{s_1^2-2\kappa}{4Q^2} - \frac{3(s_1^2-\kappa)}{8Q^2} + O(1/Q^4)$$

$$= 3 + \frac{2(s_1^2-2\kappa) - 3(s_1^2-\kappa)}{8Q^2} = 3 - \frac{s_1^2+\kappa}{8Q^2}$$

Dividiendo por $2\Omega_0 = \rho_0^2$:

$$K = -\frac{3}{\rho_0^2} + \frac{s_1^2+\kappa}{8Q^2\rho_0^2} = -\frac{3}{\rho_0^2}\left(1 - \frac{s_1^2+\kappa}{24Q^2}\right) \quad\square \tag{7.11}$$

### 7.4 Descomposición y criterio de contaminación

$$K = K_{\text{signal}} + K_{\text{PSD}} \tag{7.12}$$

$$K_{\text{signal}} = -\frac{3}{\rho_0^2}, \qquad K_{\text{PSD}} = +\frac{s_1^2+\kappa}{8Q^2\rho_0^2} \tag{7.13}$$

**Criterio de contaminación:**

$$\chi_{\text{PSD}} \equiv \left|\frac{K_{\text{PSD}}}{K_{\text{signal}}}\right| = \frac{|s_1^2+\kappa|}{24Q^2} \tag{7.14}$$

Si $\chi_{\text{PSD}} \gtrsim 1$, la curvatura del espacio de parámetros está dominada por estructura instrumental de la PSD. Los resultados de clasificación geométrica (s4/s6 de BASURIN) no son fiables en ese régimen.

---

## 8. Verificaciones

### 8.1 PSD plana ($S_n = \text{const}$)

$s_1 = 0$, $\kappa = 0$ $\Rightarrow$ $K = -3/\rho_0^2$. $\checkmark$

### 8.2 Ley de potencias ($S_n \propto f^\alpha$)

$s_1 = \alpha$, $\kappa = -\alpha$ $\Rightarrow$ $s_1^2 + \kappa = \alpha^2 - \alpha = \alpha(\alpha-1)$.

Para $\alpha = 0$: $\chi_{\text{PSD}} = 0$. (PSD plana, trivial.) $\checkmark$

Para $\alpha = 1$ ($S_n \propto f$): $\chi_{\text{PSD}} = 0$. (No trivial: una PSD lineal en $f$ no contamina.)

Para $\alpha = -4$ (shot noise típico): $\chi_{\text{PSD}} = 20/(24Q^2)$. Para $Q = 4.3$: $\chi \approx 0.045$. Pequeño.

### 8.3 Invariancia bajo cambio de coordenadas

La curvatura escalar $R = 2K$ es un invariante geométrico. Las fórmulas (7.9) se han derivado tanto en $(\ln f, \ln\tau)$ (Pasos 3-4, donde la Fisher es diagonal) como en $(\ln f, \ln Q)$ (Paso 5b, con la base hiperbólica), obteniendo el mismo resultado. $\checkmark$

### 8.4 Límite $Q \to \infty$

$K \to -3/\rho_0^2$ para cualquier PSD finita. La corrección por PSD se suprime como $1/Q^2$. Modos de alto $Q$ son robustos frente a estructura instrumental. $\checkmark$

---

## 9. Estimaciones numéricas

Los valores de $s_1$ y $\kappa$ citados en esta sección son orientativos (órdenes de magnitud). En BASURIN se evaluarán numéricamente a partir de la PSD estimada por detector en la frecuencia del modo, usando la PSD real disponible en el artefacto de s3/s4.

### 9.1 PSD analítica (aLIGO design), banda 200-300 Hz

$\ln S_n$ varía suavemente (shot noise, $\alpha \sim -4$ a $-6$). $s_1 \sim -5$, $\kappa \sim -1$.

$s_1^2 + \kappa \approx 24$. Para $Q = 4.3$: $\chi_{\text{PSD}} \approx 24/(24 \times 18.5) \approx 0.054$.

Corrección $\sim 5\%$. Pequeña pero no completamente despreciable.

### 9.2 PSD medida, lejos de líneas espectrales

Estructura moderada: $s_1 \sim 3$-$8$, $\kappa \sim 5$-$20$.

$s_1^2 + \kappa \sim 30$-$84$. Para $Q = 4.3$: $\chi \sim 0.07$-$0.19$.

Comparable a PSD analítica. Lejos de líneas, la PSD medida no contamina significativamente más.

### 9.3 PSD medida con línea espectral (violin mode)

Un pico en $\ln S_n$ con anchura $\delta f$ y altura logarítmica $h$ produce $\kappa \sim h(f/\delta f)^2$.

Para $h \sim 3$, $f = 250$ Hz, $\delta f = 0.5$ Hz: $\kappa \sim 7.5 \times 10^5$, $\chi_{\text{PSD}} \sim 10^3$.

**Dominación total por el artefacto instrumental.** La curvatura geométrica no es interpretable como física.

### 9.4 Parámetro de control $\sigma$ para eventos concretos

| Evento | Modo | $Q$ | $f$ (Hz) | $\rho$ (ringdown) | $\sigma$ estimado |
|--------|------|-----|----------|-------|-----------|
| GW150914 | $(2,2,0)$ | $4.3$ | $251$ | $\sim 8$ | $\ll 1$ (perturbativo) |
| GW150914 | $(2,2,1)$ | $\sim 2$ | $\sim 280$ | $\sim 3$ | $\sim 0.1$-$1$ (cuidado) |
| Evento baja SNR | $(2,2,0)$ | $\sim 3$ | $\sim 200$ | $\sim 4$ | Depende de PSD local |

---

## 10. Implicaciones para BASURIN

### 10.1 Coordenadas operativas

El resultado principal recomienda **trabajar en $(\ln f, \ln\tau)$** cuando se busca diagonalidad de la Fisher (para interpretación de ejes de error) y en **$(\ln f, \ln Q)$** cuando se conecta con el atlas y la comunidad GW (aceptando la correlación $r \sim -1/(2Q)$, que es pequeña para modos fundamentales).

### 10.2 Diagnóstico por evento

Tres cantidades escalares por modo y por evento:

1. **$\sigma = \kappa/(8Q^2)$**: régimen de la expansión perturbativa.
2. **$\chi_{\text{PSD}} = |s_1^2+\kappa|/(24Q^2)$**: contaminación relativa de la curvatura.
3. **$K \cdot \ell^2/6$**: corrección a la distancia geodésica respecto de Mahalanobis para separación $\ell$.

### 10.3 Validez de Mahalanobis (s4)

La distancia geodésica Fisher difiere de la Mahalanobis para separaciones $\ell$ como:

$$d_{\text{geodésica}} = \ell\left(1 - \frac{K\ell^2}{6} + O(\ell^4)\right)$$

Con $K < 0$ (curvatura negativa), a orden $\ell^3$ y localmente, la geodésica es mayor que Mahalanobis. La corrección tiende a aumentar distancias respecto a la Mahalanobis local, lo que sugiere que la clasificación actual de s4 es moderadamente conservadora para separaciones pequeñas. La dirección del efecto para umbrales finitos depende de cómo se fije el threshold (en distancia o en log-likelihood).

### 10.4 Estructura en tres capas (arquitectura conceptual para s6)

| Capa | Cantidad | Origen | ¿Modificable? |
|------|----------|--------|---------------|
| Base | $\bar{K} = -1$ | Álgebra de $Q = \pi f\tau$ | No (universal) |
| Señal | $\bar{\Delta}\varphi = 1/2$ | Forma Lorentziana | No (modelo) |
| PSD | $\delta(\bar{\Delta}\varphi)$ | Derivadas de $\ln S_n$ | Sí (por evento/detector) |

---

## Apéndice A. Integrales de referencia

$$I_{0,2} = \int_{-\infty}^{\infty}\frac{dz}{(1+z^2)^2} = \frac{\pi}{2} \tag{A.1}$$

$$I_{2,2} = \int_{-\infty}^{\infty}\frac{z^2\,dz}{(1+z^2)^2} = \frac{\pi}{2} \tag{A.2}$$

$$I_{4,2} = \int_{-\infty}^{\infty}\frac{z^4\,dz}{(1+z^2)^2} = \text{diverge} \tag{A.3}$$

$$\mathcal{J}_0(\sigma) = \pi\left[\left(\sigma+\frac{1}{2}\right)e^{\sigma}\,\text{erfc}(\sqrt{\sigma}) - \sqrt{\frac{\sigma}{\pi}}\right] \tag{A.4}$$

$$\mathcal{J}_0(0) = \frac{\pi}{2}, \qquad \lim_{\sigma\to\infty}\mathcal{J}_0(\sigma) = \frac{3\sqrt{\pi}}{8\,\sigma^{3/2}} + O(\sigma^{-5/2}) \tag{A.5}$$

> **Nota editorial:** el asintótico (A.5) se deriva de (A.4) usando $\mathrm{erfcx}(x)\sim x^{-1}/\sqrt{\pi}$ para $x\to\infty$, lo que da $\mathcal{J}_0(\sigma)\sim\pi(\sigma+\tfrac{1}{2})\cdot\sigma^{-1/2}/(\sqrt{\pi}\,\sigma^{1/2})\cdot[\text{corrección}]$. El cálculo exacto a orden dominante arroja $3\sqrt{\pi}/(8\sigma^{3/2})$, que es $O(\sigma^{-3/2})$, **no** $O(\sigma^{-1})$. La forma cerrada (A.4)/(6.7) es la definición normativa para implementación; esta fórmula asintótica es solo orientativa para el régimen $\sigma\gg 1$.

---

## Apéndice B. Tabla de símbolos

| Símbolo | Definición | Ecuación |
|---------|-----------|----------|
| $f, \tau$ | Frecuencia y tiempo de decaimiento del QNM | (1.1) |
| $Q$ | Factor de calidad: $Q = \pi f\tau$ | (1.5) |
| $\epsilon$ | Parámetro de resolución: $1/(2Q)$ | (1.5) |
| $z$ | Variable espectral adimensional: $2\pi\tau(\nu-f)$ | (1.4) |
| $\Gamma_{ij}$ | Fisher information matrix | (2.1) |
| $\rho_0$ | SNR local: $\rho_0^2 = A^2\tau/(2S_n(f))$ | (2.9) |
| $J(f,\tau)$ | Funcional central | (2.6) |
| $\Omega$ | Factor conforme de la métrica: $A^2\tau J/(2\pi)$ | (3.9) |
| $\bar{g}$ | Métrica base en $(\ln f, \ln Q)$ | (3.12) |
| $s_1$ | Pendiente logarítmica de PSD: $f\,d\ln S_n/df$ | (6.13) |
| $\kappa$ | Curvatura logarítmica de PSD: $f^2\,d^2\ln S_n/df^2$ | (6.14) |
| $\sigma$ | Parámetro de control: $\kappa/(8Q^2)$ | §6.5 |
| $\chi_{\text{PSD}}$ | Ratio de contaminación: $|s_1^2+\kappa|/(24Q^2)$ | (7.14) |
| $\mathcal{J}_n(\sigma)$ | Familia de integrales regularizadas | (6.6) |

---

## Historial de revisiones

| Versión | Fecha | Cambios |
|---------|-------|---------|
| 1.0 | 2026-02-27 | Documento inicial. Pasos 1-5b consolidados. |
