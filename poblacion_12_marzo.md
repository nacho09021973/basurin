# Poblacion 12 de marzo

## Contexto

Este documento resume la campaña operativa que cerró el 12 de marzo de 2026 sobre la cohorte BBH de BASURIN. El objetivo fue responder tres preguntas:

1. Por qué la ruta monomodo explícita estaba devolviendo regiones vacías en casi toda la cohorte.
2. Qué cambios de software eran legítimos y cuáles no debían hacerse.
3. Qué estimador debe considerarse baseline real del pipeline a partir de ahora.

## Diagnóstico inicial

La primera cohorte BBH parecía colapsar a `NO_SUPPORT_REGION` de forma casi sistemática. El patrón observado en `s3_ringdown_estimates` era:

- `f_hz` pegado al borde de banda;
- `tau_s` pegado al límite inferior del ajuste Lorentziano;
- `Q` absurdamente bajo;
- `s4g_mode220_geometry_filter` aceptando `0` geometrías.

Eso no era una conclusión física robusta; era un fallo del estimador espectral.

## Fix 1: degeneración Lorentziana en s3

Se confirmó que el ajuste Lorentziano podía converger a una solución degenerada en la que el ancho era tan grande que el perfil era casi constante sobre la banda. El arreglo fue quirúrgico:

- elevar `tau_min`;
- rechazar perfiles con `gamma` demasiado ancho respecto de la banda;
- rechazar soluciones con `Q < 1`;
- rechazar picos `f0` pegados al borde;
- emitir `fit_degenerate=true` y `fit_failure_reason` explícita.

Impacto observado: una cohorte BBH de referencia pasó de región casi siempre vacía a una población en la que `42/49` eventos ya dejaban `SUPPORT_REGION_AVAILABLE`.

## Cohorte BBH con spectral tras arreglar s3

Agregado de referencia: `runs/mvp_aggregate_20260312T105608Z/s5_aggregate/outputs/aggregate.json`

Resumen:

- `49` eventos BBH agregados;
- `42/49` con `GEOMETRY_PRESENT_BUT_NONINFORMATIVE`;
- `7/49` con `NO_SUPPORT_REGION`;
- `0/49` con `MULTIMODE_USABLE`.

Conclusión: BASURIN ya no estaba vacío, pero seguía siendo una población dominada por soporte monomodo condicionado y no por multimodo fuerte.

## Mejora A: gwtc_events.py respaldado por CSV

Se eliminó la dependencia práctica de una tabla manual de `10` eventos en `mvp/gwtc_events.py`. A partir de esta campaña, `gwtc_events.py` se resuelve desde `gwtc_quality_events.csv` para exponer al menos:

- `m_final_msun`;
- `snr_network`.

`chi_final` se mantiene solo donde existe curación explícita fiable. No se aceptó la idea de sustituirlo silenciosamente por `chi_eff` como si fueran la misma cantidad física.

## Mejora B: banda adaptativa antes del bandpass

Se implementó una expansión de la banda de entrada en `s3_ringdown_estimates` cuando existe hint Kerr y la banda fija original cortaría la frecuencia esperada del modo `220`.

Resultado:

- la mejora funciona técnicamente;
- `GW190521` sí activó banda `kerr_centered`;
- pero `A+B` no cambió por sí solo la estadística poblacional global, porque muchos eventos seguían sin `chi_final` suficientemente informativo para activar la misma lógica.

Decisión explícita:

- no se aceptó relajar el umbral `chi²` para “rescatar” eventos;
- esa vía se descartó por oportunista y por empeorar el rigor físico del soporte geométrico.

## Campaña sobre los 7 eventos fallidos

Los `7` eventos que aún quedaban en `NO_SUPPORT_REGION` tras el fix de `s3` fueron:

- `GW190513_205428`;
- `GW190521`;
- `GW190620_030421`;
- `GW190630_185205`;
- `GW190706_222641`;
- `GW190708_232457`;
- `GW190803_022701`.

Se probó `--estimator dual` sobre esos `7` y el resultado fue inequívoco:

- `7/7` pasaron a `SUPPORT_REGION_AVAILABLE`;
- `7/7` quedaron como `GEOMETRY_PRESENT_BUT_NONINFORMATIVE`;
- los tamaños de región final pasaron a valores entre `23` y `79`, no a rescates marginales.

Ejemplos:

- `GW190521`: `n_final = 78`;
- `GW190620_030421`: `n_final = 79`;
- `GW190803_022701`: `n_final = 47`;
- `GW190706_222641`: `n_final = 29`.

Conclusión: `dual` no es una rareza para outliers. Es una mejora operativa del baseline del pipeline.

## Decisión de política

A partir del 12 de marzo de 2026, el estimador por defecto de `python -m mvp.pipeline` pasa a ser `dual` en todos los modos relevantes:

- `single`;
- `multi`;
- `multimode`;
- `batch`.

Interpretación:

- `spectral` queda como baseline histórico y comparativo;
- `hilbert` queda como ruta legacy;
- `dual` pasa a ser la opción normal y recomendada para campañas reales y poblacionales.

## Lectura científica honesta

Incluso con `dual`, la población actual no debe venderse como “multimodo fuerte”. Lo que BASURIN está dando hoy es otra cosa, y es valiosa:

- una cohorte amplia de eventos BBH con región geométrica final no vacía;
- una ruta dominante `MODE220_NO_AREA_CONSTRAINT` en la cohorte canónica actual;
- semántica conservadora vía `s4k_event_support_region`;
- distinción clara entre `GEOMETRY_PRESENT_BUT_NONINFORMATIVE` y `NO_SUPPORT_REGION`.

El objeto correcto para población no es la intersección multimodo estricta, sino `golden_geometry_support_region`.

Matiz importante:

- `s4j_hawking_area_filter` solo aplica una restricción física efectiva cuando recibe `s4j_hawking_area_filter/inputs/area_obs.json`;
- si ese input falta, `s4j` actúa como pass-through y ahora lo declara explícitamente con `area_constraint_applied=false`;
- por tanto, la cohorte dual agregada del 12 de marzo de 2026 no debe describirse como “post-Hawking” sino como población `MODE220_NO_AREA_CONSTRAINT`.

La evidencia operativa de esa semántica quedó en:

- `runs/mvp_aggregate_dual_semantic_20260312T123500Z/s5_aggregate/outputs/aggregate.json`, donde `49/49` eventos caen en `MODE220_NO_AREA_CONSTRAINT`;
- un único run legacy bajo `runs/` con `area_obs.json` real y recorte efectivo (`mvp_GW150914_multimode_t0sweep_fixed_20260306T114521Z`), que sirve como demostración de que `s4j` sí filtra cuando recibe datos.

## Siguiente paso operativo

Tras documentar este cambio de política, la acción correcta es relanzar la cohorte BBH completa usando `dual` como default o explícitamente vía `--estimator dual`, y reagregar población con `s5_aggregate`.
