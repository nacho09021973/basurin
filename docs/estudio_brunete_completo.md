# BRUNETE en BASURIN: geometría Fisher del ringdown, efectos de PSD y etapa canónica lista para integrar

## Resumen ejecutivo

El documento adjunto `metodo_brunete.md` (ruta local: `/mnt/data/metodo_brunete.md`, SHA256 `a74e1bc0087c0a42c573c945794e281e72e821012a3f12baea38cdd66032f964`) fija un marco teórico muy “implementable” para BRUNETE: (i) el ringdown analítico produce una Fisher **exactamente diagonal en \((f,\tau)\) para cualquier PSD real** (Proposición 1), (ii) en coordenadas operativas \((\ln f,\ln Q)\) la métrica se factoriza como \(g=\Omega\bar g\) con \(\bar K=-1\) (Proposición 2), y (iii) la corrección de curvatura instrumental entra en \(O(1/Q^2)\) mediante \(s_1\) y \(\kappa\), con un criterio escalar de contaminación \(\chi_{\mathrm{PSD}}\) (Teorema 1). fileciteturn0file0

Desde ingeniería de pipeline, el “mínimo viable canonizable” es un stage nuevo `s6c_brunete_psd_curvature` que **no inventa PSD**: consume estimaciones de ringdown (de `s3b_multimode_estimates.py`) y una PSD local canónica (propuesta como `external_input` `psd_model.json` si aún no existe un stage PSD upstream). Este stage computa por evento/modo: derivadas logarítmicas \((s_1,\kappa)\), regímenes \((\sigma,\chi_{\mathrm{PSD}})\), y \(K,R\) a \(O(1/Q^2)\), emitiendo banderas para régimen “intermedio/colapsado” y para dominación por PSD. fileciteturn0file0

Diagnóstico contract-first: en `metodo_brunete.md` hay una inconsistencia interna entre la forma cerrada de \(\mathcal J_0(\sigma)\) (Ec. 6.7) y su asintótico declarado (Apéndice A.5), que de no corregirse hará fallar un test de resummación si se implementa estrictamente. Se propone el **cambio mínimo**: corregir el asintótico de \(\mathcal J_0(\sigma)\) para \(\sigma\to\infty\) y actualizar el texto asociado; y fijar tests que refuercen el contrato matemático real derivable de (6.7). fileciteturn0file0

En estado del arte (solo fuentes primarias/official): se ancla ringdown/QNM en clásicos y reviews (p. ej., QNMs y espectroscopia), Fisher/metricas en GW (formulación estándar y limitaciones), y PSD systematics (estimación Bayesiana moderna vs métodos clásicos). citeturn0search2turn0search5turn1search1turn2search0turn2search4turn5search2turn1search0turn4search37

## Inventario y restricciones de gobernanza BASURIN

### Estado de acceso al repositorio

En este entorno **solo** está disponible el fichero adjunto `metodo_brunete.md` en `/mnt/data/`. No existe acceso al árbol del repo (no se pueden leer `contracts.py`, `s3b_multimode_estimates.py`, `s6b_information_geometry_3d.py`, `README.md`, etc.). Por tanto, no es posible afirmar rutas/API reales del pipeline, ni realizar el inventario “leyendo el repo”.

En consecuencia, este informe entrega:

- Un diseño y staging **compatibles** con las restricciones de BASURIN (IO determinista, gating, abort semantics, artefactos/hashes), con **puntos de adaptación explícitos** a la API real de `contracts.py`.
- Un PR virtual en formato patch con código y tests **deterministas** y sin red.
- Un conjunto de comandos exactos para que el inventario se ejecute en el entorno real del repo y se “cierre la trazabilidad” sin conjeturas.

### Comandos exactos para inventario reproducible en vuestro entorno

Ejecutar en el root del repositorio (solo lectura):

```bash
# Trazabilidad del estado del repo
git rev-parse HEAD
git status --porcelain=v1

# Verificar existencia de inputs citados en el request
ls -la metodo_brunete.md contracts.py README.md readme_rutas.md request_flow.md || true
ls -la s6b_information_geometry_3d.py analysis_s6_3d_manifold.md s3b_multimode_estimates.py || true

# Enumerar stages y posibles artefactos PSD / espectros / ruido
git ls-files | grep -E '(^|/)(s[0-9]+[a-z]?_.*\.py$|contracts\.py$|.*psd.*|.*spectr.*|.*noise.*|.*welch.*)' | sort

# Localizar helpers de IO/gobernanza/hashing (nombres reales)
git grep -n "resolve_out_root\\(\"runs\"\\)|BASURIN_RUNS_ROOT|require_run_valid|RUN_VALID|manifest\\.json|stage_summary\\.json|sha256" -- .

# Localizar el schema real de outputs de s3b (campos y rutas)
git grep -n "ringdown|multimode|estimate|f_hz|tau|Q\\b|snr|rho" s3b_multimode_estimates.py || true
```

Estos comandos determinan sin ambigüedad: (i) si ya existe un artefacto PSD canónico upstream, (ii) el nombre real del gating (`RUN_VALID` u otro), y (iii) la forma exacta de declarar contracts/artefactos en `contracts.py`.

## Consolidación teórica implementable

Todo lo que sigue se deriva del adjunto `metodo_brunete.md`, con referencias a ecuaciones/proposiciones/teoremas para que la implementación sea trazable línea a línea. fileciteturn0file0

### Fisher en \((f,\tau)\) y diagonalidad exacta

Modelo analítico positive-frequency del ringdown (Ec. 1.1–1.3) y variable espectral \(z\) (Ec. 1.4) definen las derivadas \(\partial_f\tilde h\) y \(\partial_\tau\tilde h\) con la identidad algebraica clave \(\partial_f\tilde h = 2\pi i\tau^2\,\partial_\tau\tilde h\) (Ec. 2.3). Esa identidad implica que el término cruzado \(\Gamma_{f\tau}\) es exactamente cero para cualquier PSD real positiva (Proposición 1, Ec. 2.4). Este resultado es un contrato útil: **no se necesitan rotaciones elipsoidales en \((f,\tau)\)** y el pipeline puede usar coordenadas logarítmicas sin perder ejes principales “físicos” de la Fisher. fileciteturn0file0

Elementos diagonales se expresan vía el funcional central \(J(f,\tau)\) (Ec. 2.6) y se obtiene un ratio exacto \(\Gamma_{ff}/\Gamma_{\tau\tau} = 4\pi^2\tau^4\) (Ec. 2.8), que sirve como test de consistencia si en futuro se implementa un cálculo explícito de Fisher desde un \(\tilde h\) numérico. fileciteturn0file0

### Coordenadas logarítmicas y métrica en \((\ln f,\ln\tau)\) y \((\ln f,\ln Q)\)

Se definen \(u=\ln f\), \(v=\ln\tau\), \(w=\ln Q\) con \(Q=\pi f\tau\), por lo que \(v=w-u-\ln\pi\) y \(dv=dw-du\) (Ec. 3.4). En PSD plana se obtiene la forma explícita en \((u,v)\) (Ec. 3.2). En el caso general, el contrato operativo no es “PSD plana” sino **factor conforme**:

\[
ds^2 = \Omega(u,v)\left[4Q^2\,du^2 + dv^2\right]
\quad\text{(Ec. 3.10)}
\]

y en \((u,w)\):

\[
ds^2=\Omega(u,w)\left[(4Q^2+1)\,du^2 -2\,du\,dw+dw^2\right]
\quad\text{(Ec. 3.5)}
\]

con \(Q=e^w\) y una métrica base

\[
\bar g(w)=
\begin{pmatrix}
4e^{2w}+1 & -1\\
-1 & 1
\end{pmatrix},
\quad \det\bar g=4e^{2w}=4Q^2
\quad\text{(Ec. 3.12)}.
\]

Así, \(g=\Omega\,\bar g\) (Ec. 3.11). Este es el punto de integración perfecto: `s6c` solo necesita \(\Omega\) (vía PSD local y SNR proxy) y \(\bar g\) es universal. fileciteturn0file0

### Hiperbolicidad de la base y curvatura total a \(O(1/Q^2)\)

La base \(\bar g\) tiene curvatura gaussiana constante \(\bar K=-1\) (Proposición 2, Ec. 4.1–4.2), demostrable por Brioschi con derivadas simples (Ec. 4.2). Este hecho se puede reforzar con un test numérico independiente (ver sección de tests). fileciteturn0file0

La curvatura gaussiana bajo transformación conforme en 2D sigue:

\[
K=\frac{1}{\Omega}\left(\bar K-\bar\Delta\varphi\right)
,\quad \Omega=e^{2\varphi}
\quad\text{(Ec. 5.1)}.
\]

Para PSD plana, \(\Omega_0=\rho_0^2/2\) (Ec. 5.3–5.4), \(\bar\Delta\varphi=1/2\) (Ec. 5.7) y resulta:

\[
K=-\frac{3}{\rho_0^2},\qquad R=2K=-\frac{6}{\rho_0^2}
\quad\text{(Ec. 5.8–5.9)}.
\]

Para PSD variable, el Teorema 1 fija el contrato requerido por BASURIN:

\[
K= -\frac{3}{\rho_0^2}\left(1-\frac{s_1^2+\kappa}{24Q^2}\right)+O\!\left(\frac{1}{Q^4}\right),
\qquad R=2K,
\]

con

\[
s_1=f\frac{d\ln S_n}{df},\quad
\kappa=f^2\frac{d^2\ln S_n}{df^2},\quad
\sigma=\frac{\kappa}{8Q^2},\quad
\chi_{\mathrm{PSD}}=\frac{|s_1^2+\kappa|}{24Q^2}.
\]

(Ecs. 6.13–6.14 y 7.9–7.14). En pipeline, \(\chi_{\mathrm{PSD}}\) es el “semáforo” escalar para justificar gating downstream (p. ej., no interpretar manifold/clustering si \(\chi\gtrsim 1\)). fileciteturn0file0

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["gravitational wave ringdown damped sinusoid waveform","hyperbolic plane curvature -1 illustrated metric"] ,"num_per_query":1}

### Resummación por PSD: \(\mathcal J_0(\sigma)\), \(\mathcal J_1(\sigma)\) y régimen \(\sigma\)

El documento define la expansión logarítmica de \(J\) en torno a \(f\) vía \(\phi(z)=\ln S_n(f+z/(2\pi\tau))-\ln S_n(f)\) con coeficientes \(\ell_k\) (Ecs. 6.2–6.3) y separa por paridad (Ec. 6.4). Reteniendo el término cuadrático par y el primer término por \(\cosh(\phi_-)\) se llega a (Proposición 4):

\[
S_n(f)\,J=\mathcal J_0(\sigma)+\frac{s_1^2\epsilon^2}{2}\mathcal J_1(\sigma)+O(\epsilon^4)
\quad\text{(Ec. 6.5)},
\]

donde \(\epsilon=1/(2Q)\), \(\sigma=\kappa/(8Q^2)\), y

\[
\mathcal J_n(\sigma)\equiv \int_{-\infty}^{\infty}\frac{z^{2n}}{(1+z^2)^2}\,e^{-\sigma z^2}\,dz
\quad\text{(Ec. 6.6)}.
\]

Forma cerrada para \(\mathcal J_0\):

\[
\mathcal J_0(\sigma)=\pi\left[\left(\sigma+\frac{1}{2}\right)e^{\sigma}\,\mathrm{erfc}(\sqrt\sigma)-\sqrt{\frac{\sigma}{\pi}}\right]
\quad\text{(Ec. 6.7)}.
\]

Además, por derivación bajo el integral:

\[
\mathcal J_1(\sigma)=-\frac{d}{d\sigma}\mathcal J_0(\sigma),
\]

lo que habilita un cálculo determinista de \(\mathcal J_1\) sin integrar numéricamente. fileciteturn0file0

**Diagnóstico contract-first (inconsistencia asintótica):** la Ec. 6.7 implica, usando la expansión asintótica estándar de \(\mathrm{erfc}\) para \(z\to\infty\) (DLMF 7.12.1), que \(e^\sigma\mathrm{erfc}(\sqrt\sigma)\sim 1/(\sqrt\pi\,\sqrt\sigma)\) y, por tanto, \(\mathcal J_0(\sigma)\sim \sqrt\pi/(2\sqrt\sigma)\), no \(\pi/(2\sigma)\). Esto contradice el Apéndice A.5 y el texto de “colapsado” en §6.5 tal como está escrito. fileciteturn0file0 citeturn7search6

**Cambio mínimo propuesto:** corregir el asintótico en el método y alinear el test de resummación a esa consecuencia algebraica de (6.7). Esto reduce riesgo de que BASURIN “congele” un comportamiento incorrecto en tests. fileciteturn0file0 citeturn7search6

## Estado del arte y referencias primarias

La tabla siguiente contiene referencias primarias con DOI/arXiv y su relevancia directa para BRUNETE (ringdown/QNMs, Fisher/métrica, límites del FIM, PSD systematics, AdS/CFT y polos QNM). El objetivo es tener un “backbone” citacional que justifique cada bloque del pipeline y sus warnings/banderas.

| Tema | Referencia primaria (ID) | Relevancia para BRUNETE |
|---|---|---|
| Ringdown clásico (estimación de parámetros) | “Gravitational-wave measurements of the mass and angular momentum of a black hole” (Phys. Rev. D 40, 3194; DOI: 10.1103/PhysRevD.40.3194). citeturn0search2 | Ancla histórica del uso de ringdown para inferir masa/spin y del escalado de incertidumbres con SNR y amortiguamiento; útil como punto de contraste para la formulación moderna de BRUNETE en coordenadas logarítmicas y con PSD coloreada. |
| Fisher en GW (formulación estándar) | “Detection, measurement, and gravitational radiation” (Phys. Rev. D 46, 5236; DOI: 10.1103/PhysRevD.46.5236). citeturn0search5 | Formaliza producto interno ruido-ponderado y estimación/detección en ruido gaussiano coloreado; justifica que la FIM induce una métrica y que la PSD entra en el denominador del inner product, base conceptual del diagnóstico \((s_1,\kappa)\). |
| Fisher/errores en inspiral (metodología) | “Gravitational waves from merging compact binaries: …” (Phys. Rev. D 49, 2658; DOI: 10.1103/PhysRevD.49.2658). citeturn1search3 | Referencia primaria para práctica de FIM y cómo se traduce a covarianzas; aunque el waveform sea inspiral, la misma lógica aplica al ringdown: BRUNETE debe registrar cuándo el régimen (SNR, degeneracias, PSD) vuelve frágil la aproximación. |
| Métrica en espacio de plantillas | “Search templates for gravitational waves… Choice of template spacing” (Phys. Rev. D 53, 6749; DOI: 10.1103/PhysRevD.53.6749). citeturn0search0 | Fundamenta el uso geométrico de una métrica (inducida por pérdida de SNR) sobre el espacio de parámetros para construcción de bancos/clustering; es el “puente” natural entre FIM y manifold BRUNETE en BASURIN. |
| Limitaciones del formalismo Fisher | “Use and abuse of the Fisher information matrix…” (Phys. Rev. D 77, 042001; DOI: 10.1103/PhysRevD.77.042001). citeturn1search1 | Justifica warnings de BRUNETE: FIM puede fallar para SNR moderada, degeneracias o priors significativas; apoya que BASURIN registre régimen \(\chi_{\mathrm{PSD}}\) y que downstream no asuma geometría “física” cuando el régimen es instrumental/degenerado. |
| QNM review (astro + branes + holografía) | “Quasinormal modes of black holes and black branes” (Class. Quantum Grav. 26, 163001; DOI: 10.1088/0264-9381/26/16/163001; arXiv:0905.2975). citeturn5search2turn5search0 | Review de referencia que conecta QNMs en relatividad y en gauge/gravity duality; sirve para el eje AdS/CFT del request (polos QNM como relajación) y para contextualizar por qué Q, f, τ se interpretan como parámetros “universales”. |
| QNM review clásico (Living Reviews) | “Quasi-Normal Modes of Stars and Black Holes” (Living Rev Relativ 2, 2; DOI: 10.12942/lrr-1999-2). citeturn2search4 | Revisión detallada (matemática y astrofísica) de QNMs; útil para documentar elección de parametrización, escalas típicas de Q y cuándo esperar estructura espectral (líneas) que dispare \(\kappa\) y \(\chi_{\mathrm{PSD}}\). |
| AdS/CFT y QNMs como relajación | “Quasinormal modes of AdS black holes and the approach to thermal equilibrium” (Phys. Rev. D 62, 024027; DOI: 10.1103/PhysRevD.62.024027). citeturn1search0 | Conecta explícitamente QNMs con tiempos de relajación hacia equilibrio térmico en la teoría dual; relevante si BRUNETE quiere mapear geometría Fisher/curvatura a diagnósticos de régimen “instrumental vs físico” en contextos holográficos. |
| Systematics PSD (modelo BayesLine) | “Bayesian inference for spectral estimation of gravitational wave detector noise” (Phys. Rev. D 91, 084034; DOI: 10.1103/PhysRevD.91.084034). citeturn4search37 | Fuente primaria para sostener que la PSD es una fuente crítica de sesgo y que conviene modelarla/inferirla, especialmente cuando no hay off-source largo; BRUNETE usa esto como motivación para estandarizar un artefacto PSD y derivadas locales deterministas. |
| PSD estimation moderna vs Welch | “Bayesian power spectral estimation of gravitational wave detector noise revisited” (Phys. Rev. D 109, 064040; DOI: 10.1103/PhysRevD.109.064040). citeturn2search0 | Comparación explícita de técnicas y evaluación con tests de normalidad de datos blanqueados; respalda la necesidad de registrar el método de derivación de PSD (ventana, smoothing, etc.) y su impacto en diagnósticos locales como \(s_1,\kappa\). |
| Info geometry (fundamentos generales) | entity["book","Methods of Information Geometry","amari & nagaoka 2000"] (AMS, DOI: 10.1090/mmono/191). citeturn6search0 | Base formal para interpretar la Fisher como métrica en una variedad estadística y para justificar construcciones geométricas (curvatura, invariantes) sin depender de detalles del modelo GW; útil para documentar por qué \(R=2K\) (2D) y por qué cambios coordenados no deben alterar invariantes. |

## Diseño canónico de implementación en BASURIN

### Decisión de diseño mínimo viable

Se implementa un stage nuevo `s6c_brunete_psd_curvature` (en vez de integrarlo en `s6b_information_geometry_3d.py`) por separación de responsabilidades:

- `s6b_*` (si existe): geometría/manifold/atlas a nivel “variedad” (posible 3D, embeddings, etc.).
- `s6c_*`: **diagnóstico local PSD + curvatura ringdown** por evento/modo, que es un prerequisito de confianza para manifold/clustering.

Esta separación es coherente con el documento: el teorema de curvatura y \(\chi_{\mathrm{PSD}}\) son diagnósticos por modo y por evento, y deben existir **antes** de cualquier downstream interpretativo. fileciteturn0file0

### Inputs mínimos y contrato de PSD

Inputs mínimos contractuales:

1) Estimaciones ringdown por evento/modo desde `s3b_multimode_estimates.py`:
   - `event_id`, `mode`
   - `f_hz`
   - `Q` **o** `tau_s` (derivar el otro por \(Q=\pi f\tau\))
   - `rho0` / `snr_proxy` (si existe; si no, `K/R/Omega` quedan `null` + warning)

2) PSD local por detector/evento cerca de `f_hz`. Si **no existe** aún un stage upstream canónico de PSD en BASURIN, se formaliza como `external_input`:
   - `runs/<run_id>/external_inputs/psd_model.json`

3) Metadatos mínimos (sin duplicar upstream): `detector` y (en el artefacto PSD) el método de estimación/condiciones. BRUNETE registra solo lo necesario para reproducir derivadas \((s_1,\kappa)\): ventana, puntos usados, y método.

**Comparativa de opciones para PSD (decisión contract-first):**

| Opción | Pros | Contras | Decisión MVP |
|---|---|---|---|
| `external_input` `psd_model.json` | No fuerza rediseño de `s1..s5`; permite empezar scripts hoy; auditable (artefacto explícito). | Requiere disciplina de provisión; hay que definir schema. | **Elegida** para desbloquear implementación. |
| Stage nuevo `s2x_psd_estimate` | Canoniza end-to-end; reduce riesgo de PSD inconsistente entre runs. | Requiere acceso a strain/off-source/ventaneado; depende fuerte de repo/stages actuales. | Plan futuro tras inventario real. |

La elección MVP es consistente con “no inventar PSD”: si no existe, se declara external y se fijan contratos/artefactos. fileciteturn0file0 citeturn2search0turn4search37

### Outputs y artefactos por stage

Directorio de stage:

- `runs/<run_id>/s6c_brunete_psd_curvature/`
  - `outputs/brunete_metrics.json`
  - `outputs/psd_derivatives.json`
  - `stage_summary.json`
  - `manifest.json` (SHA256 por fichero)

**Schema mínimo de `outputs/brunete_metrics.json`** (por evento/mode):

- `event_id`, `mode`, `detector`
- `f_hz`, `Q`, `tau_s`, `rho0`
- `s1`, `kappa`, `sigma`, `chi_psd`
- `J0`, `J1` (resummación diagnóstica; `null` si no aplica)
- `Omega` (estimada por Ec. 6.5+3.9 usando `rho0`, o `null`)
- `K`, `R` (Teorema 1, o `null`)
- `regime_sigma`, `regime_chi_psd`
- `warnings` (lista determinista)

**Schema mínimo de `outputs/psd_derivatives.json`**:

- `event_id`, `mode`, `detector`, `f0_hz`
- `method` = `"polyfit_lnS_vs_lnf_deg2"`
- `half_window_hz` (con `c_window` configurable)
- `n_points`, `poly_coeffs` (trazabilidad)
- `s1`, `kappa`, `sigma`

**`stage_summary.json`**:

- `counts.n_records`
- `counts.sigma_regimes` en \(\{|σ|<0.1,\ 0.1\le|σ|\le 1,\ |σ|>1\}\)
- `counts.chi_psd_regimes` en \(\{\chi<0.1,\ 0.1\le\chi\le 1,\ \chi>1\}\)
- `counts.n_with_warnings`

Todo lo anterior implementa directamente el diagnóstico por régimen de §6.5 y §7.4. fileciteturn0file0

### Implementación numérica determinista y precisa

- Derivadas de PSD: ajuste local de \(\ln S(f)\) con polinomio grado 2 sobre \(\Delta u=\ln f - \ln f_0\), dentro de una ventana determinista \(\Delta f_{\mathrm{half}} = c/\tau\) (o equivalente \(c\,\pi f/Q\)), con `c_window` registrado en `stage_summary.json`. Esta elección es estable y evita el ruido de diferencias finitas, mientras respeta IO determinista. fileciteturn0file0

- Resummación: implementar \(\mathcal J_0(\sigma)\) por la forma cerrada (Ec. 6.7) usando `math.erfc` y **evaluación estable** mediante `erfcx(x)=e^{x^2}\mathrm{erfc}(x)` para \(x=\sqrt\sigma\). Para estabilidad a \(x\) grande se usa la expansión asintótica de \(\mathrm{erfc}\) (DLMF 7.12.1) que induce una expansión para `erfcx`. fileciteturn0file0 citeturn7search6

- \(\mathcal J_1(\sigma)\): computar como \(-d\mathcal J_0/d\sigma\) (identidad exacta por derivación bajo el integral de 6.6), usando derivada cerrada para \(\sigma>0\) y ramas perturbativas para \(|\sigma|<0.1\). fileciteturn0file0

- Umbral perturbativo vs resummado: por defecto `sigma_switch=0.1` como sugiere el propio método (régimen “perturbativo” vs “intermedio”). fileciteturn0file0

## PR virtual: diffs, tests y comandos

### Diffs en formato patch

A continuación se entrega un conjunto de patches autocontenidos. Donde la API real de `contracts.py` no puede conocerse, se deja un stub contract-first con TODO explícito (para que el inventario del repo lo aterrice sin ambigüedad).

#### Stage nuevo `s6c_brunete_psd_curvature.py`

```diff
diff --git a/s6c_brunete_psd_curvature.py b/s6c_brunete_psd_curvature.py
new file mode 100644
index 0000000..b2df2f7
--- /dev/null
+++ b/s6c_brunete_psd_curvature.py
@@ -0,0 +1,720 @@
+#!/usr/bin/env python3
+"""
+s6c_brunete_psd_curvature.py
+
+BASURIN / BRUNETE:
+  - Deriva s1 y kappa (derivadas logarítmicas locales de la PSD) alrededor de f del modo.
+  - Calcula sigma = kappa/(8 Q^2) y chi_PSD = |s1^2 + kappa|/(24 Q^2).
+  - Calcula K y R (R=2K) a O(1/Q^2) según Teorema 1 de metodo_brunete.md.
+  - Implementa resummación en sigma mediante J0/J1 (calligrafía J_n) cuando aplica.
+
+Gobernanza (MVP):
+  - IO determinista: solo escribe bajo runs/<run_id>/... (o BASURIN_RUNS_ROOT).
+  - Prefiere basurin_io.resolve_out_root("runs") si existe.
+  - require_run_valid gating (por defecto vía RUN_VALID file; adaptar a contracts.py real).
+  - Artefactos: outputs/* + stage_summary.json + manifest.json con hashes SHA256.
+
+NOTA IMPORTANTE (SSOT de contratos):
+  Si vuestro framework de contracts.py genera manifest/hashes en finalize(), podéis
+  desactivar el manifest local exportando:
+    BASURIN_CONTRACTS_MANAGED_MANIFEST=1
+"""
+
+from __future__ import annotations
+
+import argparse
+import dataclasses
+import hashlib
+import json
+import math
+import os
+from dataclasses import dataclass
+from pathlib import Path
+from typing import Any, Dict, List, Optional, Tuple
+
+import numpy as np
+
+
+# -------------------------
+# IO determinista: raíz runs
+# -------------------------
+
+def resolve_runs_root() -> Path:
+    """
+    Resuelve la raíz de runs siguiendo la regla:
+      1) usar basurin_io.resolve_out_root("runs") si existe;
+      2) si no, BASURIN_RUNS_ROOT;
+      3) si no, "runs".
+    """
+    try:
+        # type: ignore
+        import basurin_io  # noqa: F401
+        # type: ignore
+        return Path(basurin_io.resolve_out_root("runs"))
+    except Exception:
+        return Path(os.environ.get("BASURIN_RUNS_ROOT", "runs"))
+
+
+def require_run_valid(run_dir: Path) -> None:
+    """
+    Gating MVP: espera runs/<run_id>/RUN_VALID con contenido 'PASS'.
+    Adaptar a la semántica real de BASURIN si difiere.
+    """
+    flag = run_dir / "RUN_VALID"
+    if not flag.exists():
+        raise RuntimeError(f"[RUN_VALID] missing: {flag}")
+    val = flag.read_text(encoding="utf-8").strip()
+    if val != "PASS":
+        raise RuntimeError(f"[RUN_VALID] not PASS: {flag} -> {val!r}")
+
+
+# -------------------------
+# JSON + hashing
+# -------------------------
+
+def sha256_file(path: Path) -> str:
+    h = hashlib.sha256()
+    with path.open("rb") as f:
+        for chunk in iter(lambda: f.read(1024 * 1024), b""):
+            h.update(chunk)
+    return h.hexdigest()
+
+
+def write_json_atomic(path: Path, obj: Any) -> None:
+    path.parent.mkdir(parents=True, exist_ok=True)
+    tmp = path.with_suffix(path.suffix + ".tmp")
+    data = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
+    tmp.write_text(data + "\n", encoding="utf-8")
+    tmp.replace(path)
+
+
+def read_json(path: Path) -> Any:
+    return json.loads(path.read_text(encoding="utf-8"))
+
+
+# -------------------------
+# Parámetros ringdown
+# -------------------------
+
+def Q_from_f_tau(f_hz: float, tau_s: float) -> float:
+    if f_hz <= 0:
+        raise ValueError("f_hz must be > 0")
+    if tau_s <= 0:
+        raise ValueError("tau_s must be > 0")
+    return math.pi * f_hz * tau_s
+
+
+def tau_from_f_Q(f_hz: float, Q: float) -> float:
+    if f_hz <= 0:
+        raise ValueError("f_hz must be > 0")
+    if Q <= 0:
+        raise ValueError("Q must be > 0")
+    return Q / (math.pi * f_hz)
+
+
+# -------------------------
+# Derivadas logarítmicas PSD (s1, kappa)
+# -------------------------
+
+@dataclass(frozen=True)
+class PSDDerivatives:
+    f0_hz: float
+    s1: float
+    kappa: float
+    half_window_hz: float
+    n_points: int
+    poly_coeffs: List[float]  # [a0,a1,a2] en lnS ≈ a0 + a1 Δu + a2 Δu^2
+
+
+def estimate_s1_kappa_polyfit(
+    *,
+    f0_hz: float,
+    freqs_hz: np.ndarray,
+    psd: np.ndarray,
+    half_window_hz: float,
+    min_points: int = 11,
+) -> PSDDerivatives:
+    """
+    Método determinista (Opción 1 BRUNETE):
+      Ajuste local de ln S(f) con polinomio grado 2 en Δu = ln f - ln f0.
+
+    Con u = ln f y L(u)=ln S:
+      s1 = dL/du en u0
+      kappa = f^2 d^2 ln S / df^2 = d^2L/du^2 - s1
+
+    Si L ≈ a0 + a1 Δu + a2 Δu^2:
+      s1 = a1
+      d^2L/du^2 = 2 a2
+      kappa = 2a2 - a1
+    """
+    if f0_hz <= 0:
+        raise ValueError("f0_hz must be > 0")
+    if half_window_hz <= 0:
+        raise ValueError("half_window_hz must be > 0")
+
+    freqs_hz = np.asarray(freqs_hz, dtype=float)
+    psd = np.asarray(psd, dtype=float)
+    if freqs_hz.shape != psd.shape:
+        raise ValueError("freqs_hz and psd must have same shape")
+
+    mask = (
+        np.isfinite(freqs_hz)
+        & np.isfinite(psd)
+        & (freqs_hz > 0)
+        & (psd > 0)
+        & (np.abs(freqs_hz - f0_hz) <= half_window_hz)
+    )
+    idx = np.where(mask)[0]
+    if idx.size < min_points:
+        raise ValueError(
+            f"Insufficient PSD samples in window: n={idx.size} < {min_points} "
+            f"(f0={f0_hz}, half_window={half_window_hz})"
+        )
+
+    f = freqs_hz[idx]
+    S = psd[idx]
+
+    u0 = math.log(f0_hz)
+    u = np.log(f)
+    x = u - u0
+    y = np.log(S)
+
+    # Least squares determinista
+    A = np.column_stack([np.ones_like(x), x, x**2])
+    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
+    a0, a1, a2 = coeffs.tolist()
+
+    s1 = float(a1)
+    kappa = float(2.0 * a2 - a1)
+
+    return PSDDerivatives(
+        f0_hz=float(f0_hz),
+        s1=s1,
+        kappa=kappa,
+        half_window_hz=float(half_window_hz),
+        n_points=int(idx.size),
+        poly_coeffs=[float(a0), float(a1), float(a2)],
+    )
+
+
+# -------------------------
+# Resummación: J0/J1 (math.erfc)
+# -------------------------
+
+@dataclass(frozen=True)
+class JResum:
+    sigma: float
+    J0: Optional[float]
+    J1: Optional[float]
+    method: str  # "perturbative" | "erfc" | "asymptotic" | "not_applicable"
+
+
+def _erfcx_pos(x: float) -> Tuple[float, str]:
+    """
+    erfcx(x)=exp(x^2)*erfc(x) para x>=0, estable:
+      - rama directa para x moderado;
+      - expansión asintótica para x grande (derivada de DLMF 7.12.1).
+    """
+    if x < 0:
+        raise ValueError("x must be >= 0 for _erfcx_pos")
+
+    # Umbral conservador para evitar underflow de erfc y overflow/cancelación
+    if x <= 6.0:
+        return math.exp(x * x) * math.erfc(x), "erfc"
+
+    # Asintótico: erfcx(x) ~ 1/(sqrt(pi) x) * (1 + 1/(2x^2) + 3/(4x^4) + 15/(8x^6) + 105/(16x^8))
+    inv = 1.0 / x
+    inv2 = inv * inv
+    series = 1.0 + 0.5 * inv2 + 0.75 * inv2**2 + 1.875 * inv2**3 + 6.5625 * inv2**4
+    return (inv / math.sqrt(math.pi)) * series, "asymptotic"
+
+
+def J0_J1_metodo_brunete(sigma: float, sigma_switch: float = 0.1) -> JResum:
+    """
+    Implementa la Proposición 4 (metodo_brunete.md):
+      J0(sigma) := 𝓙0(sigma) = π * [ (sigma+1/2) e^sigma erfc(sqrt(sigma)) - sqrt(sigma/pi) ]
+      J1(sigma) := 𝓙1(sigma) = - d/dsigma 𝓙0(sigma)
+
+    Política:
+      - |sigma| < sigma_switch: perturbativo (Ecs. 6.8–6.9): J0≈π/2(1-sigma), J1≈π/2
+      - sigma >= sigma_switch: usa erfc/erfcx estable (solo sigma>=0)
+      - sigma <= -sigma_switch: no aplicable (la regularización gaussiana e^{-sigma z^2} no converge)
+    """
+    if not math.isfinite(sigma):
+        raise ValueError("sigma must be finite")
+
+    if abs(sigma) < sigma_switch:
+        J0 = (math.pi / 2.0) * (1.0 - sigma)
+        J1 = (math.pi / 2.0)
+        return JResum(sigma=float(sigma), J0=float(J0), J1=float(J1), method="perturbative")
+
+    if sigma < 0:
+        return JResum(sigma=float(sigma), J0=None, J1=None, method="not_applicable")
+
+    # sigma > 0: forma cerrada (6.7) usando erfcx(sqrt(sigma))
+    x = math.sqrt(sigma)
+    erfcx, m = _erfcx_pos(x)
+
+    # 𝓙0 = π [ (σ+1/2) * erfcx(√σ) - √(σ/π) ]
+    J0 = math.pi * ((sigma + 0.5) * erfcx - math.sqrt(sigma / math.pi))
+
+    # Derivada cerrada:
+    # Sea E(σ)=e^σ erfc(√σ)=erfcx(√σ). Entonces:
+    # dE/dσ = E - 1/(√π √σ)
+    # d/dσ[(σ+1/2)E] = E + (σ+1/2)(E - 1/(√π √σ)) = (σ+3/2)E - (σ+1/2)/(√π √σ)
+    # d/dσ[ -π √(σ/π) ] = -(√π/2) * 1/√σ
+    # dJ0/dσ = π[(σ+3/2)E - (σ+1/2)/(√π √σ)] - (√π/2)(1/√σ)
+    # J1 = -dJ0/dσ
+    if x == 0.0:
+        # no debería ocurrir aquí (sigma >= sigma_switch), pero por seguridad:
+        return JResum(sigma=float(sigma), J0=float(J0), J1=float(math.pi / 2.0), method=m)
+
+    term = (sigma + 1.5) * erfcx - (sigma + 0.5) / (math.sqrt(math.pi) * x)
+    dJ0 = math.pi * term - (math.sqrt(math.pi) / 2.0) * (1.0 / x)
+    J1 = -dJ0
+    return JResum(sigma=float(sigma), J0=float(J0), J1=float(J1), method=m)
+
+
+# -------------------------
+# Curvatura (Teorema 1)
+# -------------------------
+
+@dataclass(frozen=True)
+class CurvatureKR:
+    K: Optional[float]
+    R: Optional[float]
+
+
+def K_R_theorem1(*, rho0: Optional[float], Q: float, s1: float, kappa: float) -> CurvatureKR:
+    """
+    Teorema 1 (metodo_brunete.md):
+      K = -(3/rho0^2) * (1 - (s1^2 + kappa)/(24 Q^2)) + O(1/Q^4)
+      R = 2K
+    Si rho0 falta, devuelve None.
+    """
+    if Q <= 0:
+        raise ValueError("Q must be > 0")
+    if rho0 is None:
+        return CurvatureKR(K=None, R=None)
+    if rho0 <= 0:
+        raise ValueError("rho0 must be > 0")
+
+    corr = (s1 * s1 + kappa) / (24.0 * Q * Q)
+    K = -(3.0 / (rho0 * rho0)) * (1.0 - corr)
+    return CurvatureKR(K=float(K), R=float(2.0 * K))
+
+
+def chi_psd(*, Q: float, s1: float, kappa: float) -> float:
+    return abs(s1 * s1 + kappa) / (24.0 * Q * Q)
+
+
+# -------------------------
+# PSD model IO
+# -------------------------
+
+def read_ringdown_estimates(path: Path) -> List[Dict[str, Any]]:
+    """
+    Acepta:
+      - lista JSON de registros, o
+      - dict con key "records"
+    Cada registro debe contener:
+      event_id, mode, f_hz y (Q o tau_s). Opcional: rho0/snr_proxy, detector.
+    """
+    obj = read_json(path)
+    rows = obj["records"] if isinstance(obj, dict) and "records" in obj else obj
+    if not isinstance(rows, list):
+        raise ValueError("ringdown estimates must be a list or {records:[...]}")
+
+    out: List[Dict[str, Any]] = []
+    for r in rows:
+        if not isinstance(r, dict):
+            raise ValueError("each ringdown record must be an object")
+        event_id = str(r["event_id"])
+        mode = str(r["mode"])
+        detector = r.get("detector", None)
+        f_hz = float(r["f_hz"])
+        Q = r.get("Q", None)
+        tau_s = r.get("tau_s", None)
+        if Q is None and tau_s is None:
+            raise ValueError("record requires Q or tau_s")
+        if Q is None:
+            Q = Q_from_f_tau(f_hz, float(tau_s))
+        if tau_s is None:
+            tau_s = tau_from_f_Q(f_hz, float(Q))
+        rho0 = r.get("rho0", r.get("snr_proxy", None))
+        rho0 = None if rho0 is None else float(rho0)
+        out.append(
+            {
+                "event_id": event_id,
+                "mode": mode,
+                "detector": None if detector is None else str(detector),
+                "f_hz": float(f_hz),
+                "Q": float(Q),
+                "tau_s": float(tau_s),
+                "rho0": rho0,
+            }
+        )
+
+    out.sort(key=lambda x: (x["event_id"], x["mode"], x["detector"] or ""))
+    return out
+
+
+def read_psd_model(path: Path) -> Dict[str, Any]:
+    """
+    Formato canónico propuesto:
+      {
+        "version": "psd_model.v1",
+        "psds": [
+          {"detector":"H1", "event_id":"...", "freqs_hz":[...], "psd":[...], "meta": {...}},
+          {"detector":"L1", "freqs_hz":[...], "psd":[...], "meta": {...}}
+        ]
+      }
+    Indexado por (event_id, detector) con fallback a (None, detector) o único registro.
+    """
+    obj = read_json(path)
+    if not isinstance(obj, dict) or "psds" not in obj or not isinstance(obj["psds"], list):
+        raise ValueError("psd_model must be {psds:[...]}")
+
+    by_key: Dict[Tuple[Optional[str], str], Dict[str, Any]] = {}
+    for rec in obj["psds"]:
+        if not isinstance(rec, dict):
+            raise ValueError("each psd entry must be object")
+        detector = str(rec["detector"])
+        event_id = rec.get("event_id", None)
+        event_id = None if event_id is None else str(event_id)
+        freqs = np.asarray(rec["freqs_hz"], dtype=float)
+        psd = np.asarray(rec["psd"], dtype=float)
+        meta = rec.get("meta", {})
+        by_key[(event_id, detector)] = {"freqs_hz": freqs, "psd": psd, "meta": meta}
+
+    return {"raw": obj, "by_key": by_key}
+
+
+def get_psd_curve(psd_db: Dict[str, Any], *, event_id: str, detector: Optional[str]) -> Dict[str, Any]:
+    by_key = psd_db["by_key"]
+    det = detector or "UNKNOWN"
+    if (event_id, det) in by_key:
+        return by_key[(event_id, det)]
+    if (None, det) in by_key:
+        return by_key[(None, det)]
+    if len(by_key) == 1:
+        return next(iter(by_key.values()))
+    raise KeyError(f"No PSD for event_id={event_id!r} detector={det!r}")
+
+
+# -------------------------
+# Stage runner
+# -------------------------
+
+@dataclass(frozen=True)
+class StageConfig:
+    c_window: float = 4.0
+    min_points: int = 11
+    sigma_switch: float = 0.1
+
+
+def _regime_sigma(sigma: float) -> str:
+    a = abs(sigma)
+    if a < 0.1:
+        return "lt_0p1"
+    if a <= 1.0:
+        return "0p1_to_1"
+    return "gt_1"
+
+
+def _regime_chi(chi: float) -> str:
+    if chi < 0.1:
+        return "lt_0p1"
+    if chi <= 1.0:
+        return "0p1_to_1"
+    return "gt_1"
+
+
+def run_stage(
+    *,
+    run_id: str,
+    ringdown_estimates_path: Path,
+    psd_model_path: Path,
+    config: StageConfig,
+) -> None:
+    runs_root = resolve_runs_root()
+    run_dir = runs_root / run_id
+    require_run_valid(run_dir)
+
+    stage_dir = run_dir / "s6c_brunete_psd_curvature"
+    out_dir = stage_dir / "outputs"
+    out_dir.mkdir(parents=True, exist_ok=True)
+
+    ring = read_ringdown_estimates(ringdown_estimates_path)
+    psd_db = read_psd_model(psd_model_path)
+
+    metrics: List[Dict[str, Any]] = []
+    derivs: List[Dict[str, Any]] = []
+
+    sigma_counts = {"lt_0p1": 0, "0p1_to_1": 0, "gt_1": 0}
+    chi_counts = {"lt_0p1": 0, "0p1_to_1": 0, "gt_1": 0}
+    n_warn = 0
+
+    for rec in ring:
+        event_id = rec["event_id"]
+        mode = rec["mode"]
+        detector = rec.get("detector")
+        f0 = float(rec["f_hz"])
+        Q = float(rec["Q"])
+        tau_s = float(rec["tau_s"])
+        rho0 = rec.get("rho0")
+
+        warnings: List[str] = []
+
+        psd_curve = get_psd_curve(psd_db, event_id=event_id, detector=detector)
+        freqs = psd_curve["freqs_hz"]
+        psd = psd_curve["psd"]
+
+        half_window_hz = config.c_window / tau_s
+        d = estimate_s1_kappa_polyfit(
+            f0_hz=f0,
+            freqs_hz=freqs,
+            psd=psd,
+            half_window_hz=half_window_hz,
+            min_points=config.min_points,
+        )
+
+        sigma = d.kappa / (8.0 * Q * Q)
+        chi = chi_psd(Q=Q, s1=d.s1, kappa=d.kappa)
+
+        js = J0_J1_metodo_brunete(sigma, sigma_switch=config.sigma_switch)
+        if js.method == "not_applicable":
+            warnings.append("sigma_negative_large:no_resummation")
+        if rho0 is None:
+            warnings.append("rho0_missing:K_R_null")
+
+        KR = K_R_theorem1(rho0=rho0, Q=Q, s1=d.s1, kappa=d.kappa)
+
+        rs = _regime_sigma(sigma)
+        rc = _regime_chi(chi)
+        sigma_counts[rs] += 1
+        chi_counts[rc] += 1
+        if warnings:
+            n_warn += 1
+
+        metrics.append(
+            {
+                "event_id": event_id,
+                "mode": mode,
+                "detector": detector,
+                "f_hz": f0,
+                "Q": Q,
+                "tau_s": tau_s,
+                "rho0": rho0,
+                "s1": d.s1,
+                "kappa": d.kappa,
+                "sigma": sigma,
+                "chi_psd": chi,
+                "J0": js.J0,
+                "J1": js.J1,
+                "K": KR.K,
+                "R": KR.R,
+                "regime_sigma": rs,
+                "regime_chi_psd": rc,
+                "resummation_method": js.method,
+                "warnings": warnings,
+            }
+        )
+
+        derivs.append(
+            {
+                "event_id": event_id,
+                "mode": mode,
+                "detector": detector,
+                "f0_hz": f0,
+                "half_window_hz": d.half_window_hz,
+                "n_points": d.n_points,
+                "method": "polyfit_lnS_vs_lnf_deg2",
+                "poly_coeffs": d.poly_coeffs,
+                "s1": d.s1,
+                "kappa": d.kappa,
+                "sigma": sigma,
+            }
+        )
+
+    # Escritura determinista
+    metrics_path = out_dir / "brunete_metrics.json"
+    derivs_path = out_dir / "psd_derivatives.json"
+    summary_path = stage_dir / "stage_summary.json"
+    manifest_path = stage_dir / "manifest.json"
+
+    write_json_atomic(metrics_path, metrics)
+    write_json_atomic(derivs_path, derivs)
+
+    summary = {
+        "stage": "s6c_brunete_psd_curvature",
+        "run_id": run_id,
+        "config": dataclasses.asdict(config),
+        "counts": {
+            "n_records": len(metrics),
+            "n_with_warnings": n_warn,
+            "sigma_regimes": sigma_counts,
+            "chi_psd_regimes": chi_counts,
+        },
+    }
+    write_json_atomic(summary_path, summary)
+
+    # Manifest local (opcional si contracts.py ya lo gestiona)
+    if os.environ.get("BASURIN_CONTRACTS_MANAGED_MANIFEST", "0") != "1":
+        manifest = {
+            "stage": "s6c_brunete_psd_curvature",
+            "run_id": run_id,
+            "files": {
+                str(metrics_path.relative_to(stage_dir)): sha256_file(metrics_path),
+                str(derivs_path.relative_to(stage_dir)): sha256_file(derivs_path),
+                str(summary_path.relative_to(stage_dir)): sha256_file(summary_path),
+            },
+        }
+        write_json_atomic(manifest_path, manifest)
+
+
+def build_argparser() -> argparse.ArgumentParser:
+    p = argparse.ArgumentParser(description="BASURIN BRUNETE: s6c_brunete_psd_curvature")
+    p.add_argument("--run-id", required=True)
+    p.add_argument("--ringdown-estimates", required=True, type=Path)
+    p.add_argument("--psd-model", required=True, type=Path)
+    p.add_argument("--c-window", type=float, default=4.0)
+    p.add_argument("--min-points", type=int, default=11)
+    p.add_argument("--sigma-switch", type=float, default=0.1)
+    return p
+
+
+def main(argv: Optional[List[str]] = None) -> int:
+    args = build_argparser().parse_args(argv)
+    cfg = StageConfig(c_window=args.c_window, min_points=args.min_points, sigma_switch=args.sigma_switch)
+    run_stage(
+        run_id=str(args.run_id),
+        ringdown_estimates_path=args.ringdown_estimates,
+        psd_model_path=args.psd_model,
+        config=cfg,
+    )
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
```

#### Tests deterministas `tests/test_brunete_psd_curvature.py`

Incluye los tests requeridos: PSD plana, ley de potencia, invariancia por rutas (aquí implementada como invariancia de derivadas en \(\ln f\) frente a la identidad algebraica \(s_1=d\ln S/d\ln f\), y consistencia de \(\kappa\) por cambio de variable), y resummación con \(J_0(0)=\pi/2\) y asintótico corregido (ver patch a `metodo_brunete.md`). Los tests no generan runs y no usan red.

```diff
diff --git a/tests/test_brunete_psd_curvature.py b/tests/test_brunete_psd_curvature.py
new file mode 100644
index 0000000..d1f1a2a
--- /dev/null
+++ b/tests/test_brunete_psd_curvature.py
@@ -0,0 +1,260 @@
+import math
+import numpy as np
+
+import s6c_brunete_psd_curvature as s6c
+
+
+def _grid(fmin=10.0, fmax=1000.0, n=5001):
+    return np.linspace(fmin, fmax, n)
+
+
+def test_psd_flat_s1_kappa():
+    freqs = _grid()
+    psd = np.ones_like(freqs) * 3.0
+    res = s6c.estimate_s1_kappa_polyfit(
+        f0_hz=250.0,
+        freqs_hz=freqs,
+        psd=psd,
+        half_window_hz=40.0,
+        min_points=101,
+    )
+    assert abs(res.s1) < 1e-10
+    assert abs(res.kappa) < 1e-10
+    assert abs(s6c.chi_psd(Q=10.0, s1=res.s1, kappa=res.kappa)) < 1e-12
+
+
+def test_psd_powerlaw_s1_kappa_and_chi():
+    # S(f)=f^alpha -> lnS=alpha ln f
+    alpha = 4.0
+    freqs = _grid()
+    psd = freqs**alpha
+    Q = 20.0
+    res = s6c.estimate_s1_kappa_polyfit(
+        f0_hz=200.0,
+        freqs_hz=freqs,
+        psd=psd,
+        half_window_hz=60.0,
+        min_points=401,
+    )
+    assert abs(res.s1 - alpha) < 5e-4
+    assert abs(res.kappa - (-alpha)) < 5e-4
+
+    chi = s6c.chi_psd(Q=Q, s1=res.s1, kappa=res.kappa)
+    target = (alpha * alpha - alpha) / (24.0 * Q * Q)
+    assert abs(chi - target) < 5e-6
+
+
+def test_curvature_flat_psd():
+    # Teorema 1: s1=0,kappa=0 => K=-3/rho0^2
+    rho0 = 8.0
+    out = s6c.K_R_theorem1(rho0=rho0, Q=15.0, s1=0.0, kappa=0.0)
+    assert out.K is not None
+    assert abs(out.K - (-(3.0 / (rho0 * rho0)))) < 1e-12
+    assert abs(out.R - 2.0 * out.K) < 1e-15
+
+
+def test_invariance_s1_definition_via_log_derivative():
+    # s1 = d ln S / d ln f. Para S=f^alpha debe ser constante alpha.
+    alpha = -4.0
+    freqs = _grid()
+    psd = freqs**alpha
+    res = s6c.estimate_s1_kappa_polyfit(
+        f0_hz=300.0,
+        freqs_hz=freqs,
+        psd=psd,
+        half_window_hz=80.0,
+        min_points=501,
+    )
+    assert abs(res.s1 - alpha) < 5e-4
+
+
+def test_resummation_J0_zero():
+    r = s6c.J0_J1_metodo_brunete(0.0, sigma_switch=0.0)
+    assert r.J0 is not None
+    assert abs(r.J0 - (math.pi / 2.0)) < 1e-12
+
+
+def test_resummation_J0_asymptotic_corrected():
+    # De (6.7) + asintótico de erfc: J0(σ) ~ sqrt(pi)/(2*sqrt(σ))
+    sigma = 1e6
+    r = s6c.J0_J1_metodo_brunete(sigma, sigma_switch=0.0)
+    assert r.J0 is not None
+    target = math.sqrt(math.pi) / (2.0 * math.sqrt(sigma))
+    # tolerancia relativa laxa por truncación asintótica en erfcx
+    assert abs(r.J0 / target - 1.0) < 5e-3
+
+
+def test_resummation_negative_sigma_not_applicable():
+    r = s6c.J0_J1_metodo_brunete(-1.0, sigma_switch=0.1)
+    assert r.method == "not_applicable"
+    assert r.J0 is None and r.J1 is None
```

#### Documento de investigación `docs/BRUNETE_RESEARCH_SUMMARY.md`

```diff
diff --git a/docs/BRUNETE_RESEARCH_SUMMARY.md b/docs/BRUNETE_RESEARCH_SUMMARY.md
new file mode 100644
index 0000000..c4f7c2c
--- /dev/null
+++ b/docs/BRUNETE_RESEARCH_SUMMARY.md
@@ -0,0 +1,260 @@
+# BRUNETE — Research Summary
+
+## Executive summary
+
+Se consolida la geometría Fisher del ringdown como estructura conforme g=Ω·ḡ en (ln f, ln Q),
+con curvatura base constante K̄=-1 y curvatura total
+  K = -(3/ρ0^2)(1 - (s1^2+κ)/(24 Q^2)) + O(1/Q^4),
+y diagnóstico instrumental χ_PSD = |s1^2+κ|/(24 Q^2).
+
+Se define un stage canónico s6c_brunete_psd_curvature que computa por evento/modo
+(s1, κ, σ, χ_PSD, K, R) y banderas de régimen, listos para gating downstream.
+
+## Teoría implementable (metodo_brunete.md)
+
+### Fisher en (f, τ)
+
+Diagonalidad exacta para cualquier PSD real: Γ_fτ=0 (Proposición 1).
+Elementos diagonales expresados vía el funcional J(f,τ) (Ecs. 2.6–2.7).
+
+### Coordenadas logarítmicas
+
+u=ln f, v=ln τ, w=ln Q con Q=π f τ y dv=dw-du.
+
+En (u,v):
+  ds^2 = Ω(u,v) [ 4Q^2 du^2 + dv^2 ].
+
+En (u,w):
+  ds^2 = Ω(u,w) [ (4Q^2+1) du^2 -2 du dw + dw^2 ].
+
+Factorización g = Ω·ḡ con
+  ḡ(w) = [[4e^{2w}+1, -1],[-1, 1]],
+y K̄=-1 (Proposición 2).
+
+### Curvatura y criterio PSD
+
+Teorema 1:
+  K = -(3/ρ0^2) (1 - (s1^2+κ)/(24 Q^2)) + O(1/Q^4),
+  R = 2K.
+
+s1 = f d ln S/df, κ = f^2 d^2 ln S/df^2,
+σ = κ/(8Q^2), χ_PSD = |s1^2+κ|/(24Q^2).
+
+### Resummación
+
+Proposición 4:
+  S_n(f) J = 𝓙0(σ) + (s1^2 ε^2/2) 𝓙1(σ) + O(ε^4),  ε=1/(2Q).
+
+  𝓙0(σ) = π[(σ+1/2) e^σ erfc(√σ) - √(σ/π)].
+  𝓙1(σ) = - d/dσ 𝓙0(σ).
+
+Nota: la forma cerrada implica asintótico 𝓙0(σ) ~ √π/(2√σ) para σ→∞.
+
+## Referencias primarias (ver informe principal)
+
+Se citan trabajos clásicos y reviews de ringdown/QNMs, Fisher en GW, límites del FIM,
+PSD systematics y AdS/CFT QNM poles.
```

#### Plan de implementación `docs/BRUNETE_IMPLEMENTATION_PLAN.md` con diagramas Mermaid

```diff
diff --git a/docs/BRUNETE_IMPLEMENTATION_PLAN.md b/docs/BRUNETE_IMPLEMENTATION_PLAN.md
new file mode 100644
index 0000000..9a1a0d8
--- /dev/null
+++ b/docs/BRUNETE_IMPLEMENTATION_PLAN.md
@@ -0,0 +1,240 @@
+# BRUNETE — Implementation Plan (BASURIN)
+
+## Stage propuesto
+
+Nombre: s6c_brunete_psd_curvature
+
+Responsabilidad:
+  - derivadas PSD locales (s1, κ) alrededor de f del modo ringdown
+  - σ y χ_PSD (régimen y contaminación)
+  - curvatura K y escalar R=2K (Teorema 1, O(1/Q^2))
+  - outputs por evento/mode con warnings para gating downstream
+
+## Flujo IO (Mermaid)
+
+```mermaid
+flowchart LR
+  s3b[s3b_multimode_estimates] -->|ringdown_estimates.json| s6c[s6c_brunete_psd_curvature]
+  psd[external_inputs/psd_model.json] --> s6c
+  s6c -->|outputs/brunete_metrics.json| down[(downstream: s6b/s7...)]
+  s6c -->|outputs/psd_derivatives.json| down
+  s6c -->|stage_summary.json + manifest.json| audit[(auditoría)]
+```
+
+## Decisión PSD (MVP)
+
+En ausencia de un stage PSD canónico upstream, se usa external_input:
+  runs/<run_id>/external_inputs/psd_model.json
+
+Luego, tras inventario, se decide si se canoniza como s2x_psd_estimate (stage) o se mantiene external.
+
+## Parámetros configurables (registrados en stage_summary.json)
+
+  c_window: half_window_hz = c_window / tau
+  min_points: puntos mínimos para ajuste
+  sigma_switch: umbral perturbativo vs resummado
+
+## Timeline (Mermaid)
+
+```mermaid
+timeline
+  title BRUNETE: integración por fases
+  2026-02-27 : Consolidación teórica (metodo_brunete.md)
+  2026-03 : Implementación s6c + tests unitarios deterministas
+  2026-03 : Integración contracts.py + wiring a s3b y artefacto PSD
+  2026-03 : Validación en runs reales (plan de validación, sin claims)
+```
```

#### Patch mínimo a `metodo_brunete.md` (corrección asintótica de \(\mathcal J_0\))

Este es el único cambio teórico propuesto: alinear el Apéndice A.5 y el texto de §6.5 con lo que implica la Ec. 6.7.

```diff
diff --git a/metodo_brunete.md b/metodo_brunete.md
index 1234567..89abcde 100644
--- a/metodo_brunete.md
+++ b/metodo_brunete.md
@@ -1,0 +1,0 @@
@@ -215,7 +215,9 @@
-| Colapsado | $\sigma \gg 1$ | $\mathcal{J}_0 \sim \pi/(2\sigma)$: $J$ suprimido $\Rightarrow$ $\Omega$ muy pequeña $\Rightarrow$ métrica degenerada | Inferencia no fiable |
+| Colapsado | $\sigma \gg 1$ | $\mathcal{J}_0(\sigma) \sim \sqrt{\pi}/(2\sqrt{\sigma})$ (derivado de (6.7) y el asintótico estándar de $\mathrm{erfc}$) $\Rightarrow$ $J$ suprimido $\Rightarrow$ $\Omega$ pequeña $\Rightarrow$ métrica degenerada | Inferencia no fiable |
@@ -420,7 +422,9 @@
-$$\mathcal{J}_0(0) = \frac{\pi}{2}, \qquad \lim_{\sigma\to\infty}\mathcal{J}_0(\sigma) = \frac{\pi}{2\sigma} + O(\sigma^{-2}) \tag{A.5}$$
+$$\mathcal{J}_0(0) = \frac{\pi}{2}, \qquad \lim_{\sigma\to\infty}\mathcal{J}_0(\sigma) = \frac{\sqrt{\pi}}{2\sqrt{\sigma}} + O(\sigma^{-3/2}) \tag{A.5}$$
+
+*(Nota: este asintótico se obtiene sustituyendo el asintótico de $\mathrm{erfc}$ en (6.7).)*
```

#### Stub de contrato en `contracts.py` (a aterrizar tras inventario)

```diff
diff --git a/contracts.py b/contracts.py
index 7654321..fedcba9 100644
--- a/contracts.py
+++ b/contracts.py
@@ -1,0 +1,60 @@
+# ---------------------------------------------------------------------
+# BRUNETE: s6c_brunete_psd_curvature (STUB contract-first)
+# ---------------------------------------------------------------------
+#
+# IMPORTANTE: este bloque es una plantilla. Debe adaptarse a:
+#   - API real de registro de stages en contracts.py (SSOT)
+#   - ruta real del output de s3b_multimode_estimates
+#   - convención real para external_inputs
+#   - mecanismo real de require_run_valid / abort semantics / finalize(manifest)
+#
+# Concepto mínimo:
+#   - stage name: "s6c_brunete_psd_curvature"
+#   - require_run_valid=True
+#   - inputs: ringdown_estimates (from s3b) + psd_model (external_input)
+#   - outputs: outputs/brunete_metrics.json, outputs/psd_derivatives.json, stage_summary.json, manifest.json
+#
+# TODO: sustituir por la estructura real, p.ej. STAGES[...] = StageContract(...)
```

### Comandos exactos solicitados

#### Ejecutar inventario (sin escribir)

```bash
git rev-parse HEAD
git status --porcelain=v1
git ls-files | grep -E '(^|/)(s[0-9]+[a-z]?_.*\.py$|contracts\.py$|.*psd.*|.*noise.*)' | sort
git grep -n "resolve_out_root\\(\"runs\"\\)|require_run_valid|RUN_VALID|manifest\\.json|stage_summary\\.json" -- .
```

#### Ejecutar tests (sin prometer resultados)

```bash
pytest -q
```

#### Ejemplo de ejecución del stage (sin ejecutar)

Asumiendo que el pipeline pasa rutas reales, un CLI de referencia sería:

```bash
python s6c_brunete_psd_curvature.py \
  --run-id <run_id> \
  --ringdown-estimates runs/<run_id>/s3b_multimode_estimates/outputs/ringdown_estimates.json \
  --psd-model runs/<run_id>/external_inputs/psd_model.json \
  --c-window 4.0 \
  --min-points 11 \
  --sigma-switch 0.1
```

Si vuestro `contracts.py` ya genera `manifest.json` en `finalize()`, exportar:

```bash
export BASURIN_CONTRACTS_MANAGED_MANIFEST=1
```

para evitar doble manifest. (Este comportamiento está diseñado para respetar SSOT cuando exista.)

## Definition of Done, riesgos y supuestos

### Checklist Definition of Done

- `contracts.py` registra `s6c_brunete_psd_curvature` como SSOT con:
  - `require_run_valid=True`
  - upstream correcto hacia el artefacto de ringdown de `s3b_multimode_estimates.py`
  - PSD declarada como upstream canónica **o** `external_input` `psd_model.json` (sin inventar)
  - outputs declarados exactamente: `outputs/brunete_metrics.json`, `outputs/psd_derivatives.json`, `stage_summary.json`, `manifest.json`
- El stage escribe únicamente bajo `runs/<run_id>/s6c_brunete_psd_curvature/` y nunca fuera de `runs/<run_id>` (o `BASURIN_RUNS_ROOT`).
- Gating efectivo: si `RUN_VALID != PASS` (o equivalente real), el stage aborta y no deja artefactos downstream.
- Artefactos presentes:
  - `outputs/brunete_metrics.json`, `outputs/psd_derivatives.json`, `stage_summary.json`
  - `manifest.json` con hashes SHA256 (ya sea por el stage o por `contracts.py finalize()`).
- Tests deterministas pasan:
  - PSD plana: \(s_1=0,\kappa=0\Rightarrow\chi=0\) y \(K=-3/\rho_0^2\). fileciteturn0file0
  - PSD potencia \(S\propto f^\alpha\): \(s_1=\alpha,\ \kappa=-\alpha,\ \chi=(\alpha^2-\alpha)/(24Q^2)\). fileciteturn0file0
  - Resummación: \(J_0(0)=\pi/2\) y asintótico consistente con Ec. 6.7 + DLMF 7.12.1 tras aplicar el patch del método. fileciteturn0file0 citeturn7search6
  - Negativo grande de \(\sigma\): comportamiento determinista (no aplicable) y warning reproducible.
- Documentación añadida:
  - `docs/BRUNETE_RESEARCH_SUMMARY.md`
  - `docs/BRUNETE_IMPLEMENTATION_PLAN.md`
  - patch mínimo en `metodo_brunete.md` si se acepta la corrección asintótica

### Riesgos y supuestos

- **Repo no accesible aquí:** hasta ejecutar el inventario, no se puede fijar (i) ruta exacta de outputs de `s3b`, (ii) API real de `contracts.py`, (iii) si ya existe un stage PSD upstream. Este informe deja stubs y comandos para resolverlo sin ambigüedad.
- **Semántica exacta de `RUN_VALID`:** el gating MVP usa un fichero `RUN_VALID` con `PASS`. Si BASURIN usa otra convención, debe reemplazarse por la función real de gobernanza desde `contracts.py` (prioritario para no romper abort semantics).
- **Resummación y signo de \(\sigma\):** la regularización \(\exp(-\sigma z^2)\) solo converge si \(\sigma\ge 0\). En PSD suaves tipo potencia, \(\kappa\) puede ser negativa y \(|\sigma|\) pequeño; el pipeline debe tratarlo como perturbativo (lo hace el código) y marcar no aplicable si \(\sigma\) es negativa grande. fileciteturn0file0
- **Limitaciones del FIM:** incluso con curvatura calculada, el FIM puede no ser predictor fiable a SNR moderada o con degeneracias; por eso es correcto que BRUNETE emita warnings/regímenes y que downstream no “sobreinterprete” manifolds cuando los diagnósticos lo contraindican. citeturn1search1