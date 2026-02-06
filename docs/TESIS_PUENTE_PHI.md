# Tesis del Puente Phi: Ringdown <-> Holografia

**Fecha:** 2026-02-06
**Estado:** Conjetura falsable (v0)

---

## 1. Tesis (una frase)

> Dado un espectro de ringdown (f, tau, Q) de un evento gravitacional,
> el pipeline puede seleccionar la geometria AdS correcta entre N=128
> candidatos del atlas holografico con accuracy top-1 >= 70% y top-3 >= 95%.

## 2. Criterios de falsacion

| Criterio | Umbral | Significado si falla |
|----------|--------|----------------------|
| accuracy_top1 | >= 70% (X) | Phi no discrimina geometrias |
| accuracy_topk (k=3) | >= 95% (Y) | Ni siquiera un ranking grueso funciona |
| N atlas | >= 64 | Resultado trivial si N es demasiado pequeno |
| alpha bootstrap | convergencia | Si alpha no converge, el modelo fenomenologico falla |

Si accuracy_top1 < 70% con N > 64, se rechaza Phi (o se demuestra
que alpha no es calibrable por bootstrap).

## 3. El mapa Phi (conjetura fenomenologica)

### 3.1 Hipotesis fisica

El fundamental (n=0) y primer overtone (n=1) del espectro holografico
codifican la geometria efectiva. La relacion es:

```
f   = sqrt(lambda_0) / (2 * pi * L)
tau = L / (alpha * sqrt(lambda_0) * (r_1 - 1))
Q   = pi * f * tau = 1 / (2 * alpha * (r_1 - 1))
```

Donde:
- lambda_0 = M_0^2 (autovalor fundamental del espectro holografico)
- r_1 = M_1^2 / M_0^2 (primer ratio del atlas, primer overtone/fundamental)
- L = radio AdS (de geometry.h5)
- alpha = parametro de calibracion (alpha=1 para AdS puro)

### 3.2 Modelo directo (atlas -> ringdown predicho)

Para cada teoria i del atlas con (delta_i, M2_0_i, r_1_i):

```python
omega_0 = sqrt(abs(M2_0)) / L        # frecuencia angular fundamental
f_pred  = omega_0 / (2 * pi)         # frecuencia Hz
gamma   = alpha * omega_0 * (r_1 - 1)  # tasa de damping
tau_pred = 1.0 / gamma               # tiempo de decay
Q_pred  = pi * f_pred * tau_pred      # factor de calidad
```

### 3.3 Modelo inverso (ringdown observado -> espacio de ratios)

Dadas observaciones (f_obs, tau_obs):

```python
omega_0 = 2 * pi * f_obs
gamma   = 1.0 / tau_obs
r1_pred = 1.0 + gamma / (alpha * omega_0)
        = 1.0 + 1 / (alpha * 2 * pi * f_obs * tau_obs)
        = 1.0 + 1 / (alpha * 2 * Q_obs)
```

### 3.4 Calibracion de alpha

En AdS puro con dimension Delta, d=3:
- lambda_n ~ (n + Delta)^2 / L^2 (simplificado)
- r_1 = (1 + Delta)^2 / Delta^2

Alpha se calibra minimizando el error de reconstruccion en el atlas
sintetico (puro AdS). Valor inicial: alpha = 1.0.

## 4. Metrica de ranking

Distancia euclidea en espacio de log-ratios:

```
d(r_obs, r_atlas) = || log(r1_obs) - log(r1_atlas) ||
```

Usando log-ratios porque los ratios son positivos y la escala
relativa importa mas que la absoluta.

Para k_features > 1, se usa la distancia euclidea en R^k:

```
d(r_obs, r_atlas) = sqrt( sum_j (log(r_j_obs) - log(r_j_atlas))^2 )
```

## 5. Protocolo de validacion

1. Generar atlas con N=128 geometrias (sweep de Delta)
2. Para cada geometria i del atlas:
   a. Usar modelo directo para generar (f_i, tau_i, Q_i) predichos
   b. Anadir ruido gaussiano (sigma relativo ~ 5%)
   c. Usar modelo inverso para mapear a r1_pred
   d. Buscar top-k mas cercanos en atlas
   e. Registrar si la geometria correcta esta en top-k
3. Calcular accuracy_top1 y accuracy_topk
4. Comparar con umbrales X=70%, Y=95%

## 6. Extension

- Si alpha=1 falla, calibrar alpha por bootstrap en el atlas
- Si alpha calibrado falla, probar inclusion de r_2 (segundo overtone)
- Si dim4 funciona, probar dim6 (generalizacion mas alla de AdS)
- Si sintetico funciona, probar con datos reales (EXP_08)
