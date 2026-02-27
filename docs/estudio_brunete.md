# Geometría Fisher del ringdown y etapa canónica BRUNETE en BASURIN

## Inventario y límites de acceso al repositorio

En el entorno actual no hay acceso al árbol del repositorio (no existen los ficheros indicados, y las herramientas de “file_search” no tienen fuentes conectadas), por lo que no puedo **verificar rutas reales**, **API de `contracts.py`**, ni **compatibilidad exacta** con los stages existentes. Por rigor y trazabilidad, todo lo que sigue se entrega como:

- **Diseño canónico verificable** (contrato + artefactos + hashing + tests deterministas).
- **Implementación autocontenida** (funciones puras + stage ejecutable) para integrar con el runner de BASURIN.
- **Plan de inventario reproducible** (comandos exactos) para que el repo real “cierre el bucle” y se puedan ajustar rutas/APIs sin conjeturas.

### Comandos deterministas para inventariar el repo en vuestro entorno

Ejecutar **en el root del repo** (sin escribir fuera, solo lectura):

```bash
# Identidad del commit (trazabilidad)
git rev-parse HEAD
git status --porcelain=v1

# Inventario mínimo de artefactos citados por el request
ls -la metodo_brunete.md contracts.py README.md readme_rutas.md request_flow.md || true
ls -la s6b_information_geometry_3d.py analysis_s6_3d_manifold.md s3b_multimode_estimates.py || true

# Localizar stages y artefactos PSD ya existentes
git ls-files | grep -E '(^|/)(s[0-9]+[a-z]?_.*\.py$|contracts\.py$|.*psd.*|.*spectr.*|.*noise.*)' | sort

# Buscar símbolos-clave para integración (nombres reales de helpers BASURIN)
git grep -n "resolve_out_root\\(\"runs\"\\)|BASURIN_RUNS_ROOT|RUN_VALID|require_run_valid|manifest\\.json|stage_summary\\.json|sha256" -- .

# Determinar formato real de outputs de s3b (campos: f, Q/tau, event_id, mode, SNR proxy)
git grep -n "def .*multimode|multimode_estimates|ringdown|Q\\b|tau\\b|snr\\b|rho" s3b_multimode_estimates.py || true
```

El resultado de esas búsquedas fija (i) rutas reales, (ii) esquema real de outputs PSD y (iii) la forma exacta de registrar el contrato en `contracts.py`.  

## Teoría implementable: métrica Fisher, coordenadas logarítmicas y curvatura

### Definición estándar de Fisher en GW y dependencia con PSD

En análisis de ondas gravitacionales con ruido gaussiano estacionario, el producto interno (para PSD unilateral) se escribe típicamente como  
\[(a|b)=4\,\Re\int_{0}^{\infty}\frac{\tilde a(f)\,\tilde b^*(f)}{S_n(f)}\,df\]  
y la matriz de Fisher como \(\Gamma_{ij}=(\partial_i h|\partial_j h)\), que define una métrica riemanniana sobre el espacio de parámetros (bajo regularidad). Esto es estándar en la literatura de estimación de parámetros con matched filtering. citeturn3search7turn8search1

La idea geométrica (plantillas como variedad con métrica inducida por pérdidas de SNR) también es clásica en construcción de “template banks”, articulada explícitamente como “template-space metric”. citeturn8search0

### Parámetros ringdown y cambios de coordenadas

Para un modo ringdown modelo (damped sinusoid), es habitual parametrizar por frecuencia \(f\), tiempo de amortiguamiento \(\tau\), y/o factor de calidad \(Q\). En notación estándar:
\[
Q \equiv \pi f \tau,\qquad \tau=\frac{Q}{\pi f}.
\]

Definimos coordenadas logarítmicas (las que solicita BRUNETE):

- \(u\equiv \ln f\)
- \(v\equiv \ln \tau\)
- \(w\equiv \ln Q\)

Con relación exacta:
\[
w = \ln(\pi)+u+v.
\]

Estas coordenadas son útiles porque convierten derivadas logarítmicas de la PSD en cantidades adimensionales y porque estabilizan escalas cuando hay décadas de frecuencia.

### Métrica “base” de curvatura constante y factorización conforme

BRUNETE solicita la factorización conforme:
\[
g = \Omega \,\bar g,
\]
donde \(\bar g\) es una métrica “base” (adimensional) en \((u=\ln f,\ w=\ln Q)\) con curvatura constante:
\[
\bar K=-1,
\]
y \(\Omega\) codifica el “tamaño” informacional (escala por SNR, y correcciones por PSD).

Una realización concreta y **implementable** de \(\bar g\) que tiene \(\bar K=-1\) es (forma equivalente a la métrica hiperbólica en coordenada logarítmica del semiplano):
\[
\bar g(u,w)=
\begin{pmatrix}
e^{-2w} & 0\\
0 & 1
\end{pmatrix}.
\]
En esta elección, el escalar de Ricci de \(\bar g\) es \(\bar R=-2\), y \(\bar K=\bar R/2=-1\) (consistente con geometría hiperbólica).  

La transformación a \((u,v)\) se implementa con el Jacobiano
\[
J=\frac{\partial(u,w)}{\partial(u,v)}=\begin{pmatrix}1&0\\1&1\end{pmatrix},
\qquad g_{(u,v)}=J^T\,g_{(u,w)}\,J,
\]
lo que da explícitamente:
\[
\bar g(u,v)=
\begin{pmatrix}
1+e^{-2w} & 1\\
1 & 1
\end{pmatrix},\quad w=\ln(\pi)+u+v.
\]
Esto es valioso para el test de invariancia pedido (misma curvatura bajo cambio coordinado).

### Curvatura total con PSD a orden \(O(1/Q^2)\)

BRUNETE fija como objetivo a \(O(1/Q^2)\):
\[
K = -\frac{3}{\rho_0^2}\left(1 - \frac{s_1^2+\kappa}{24\,Q^2}\right) + O\!\left(\frac{1}{Q^4}\right),
\]
y solicita también \(R\) (en 2D, \(R=2K\) por convención riemanniana usual).

Interpretación implementable coherente con \(g=\Omega\bar g\):

- Si \(\Omega\) es aproximadamente constante “localmente” en el punto, para curvatura gaussiana:
  \[
  K \approx \frac{\bar K}{\Omega} = -\frac{1}{\Omega}.
  \]
- Identificando el leading order:
  \[
  \Omega_0 = \frac{\rho_0^2}{3}\quad\Rightarrow\quad K_0=-\frac{3}{\rho_0^2}.
  \]
- Incorporando corrección multiplicativa (expansión):
  \[
  \delta \equiv \frac{s_1^2+\kappa}{24Q^2},\qquad \Omega \approx \Omega_0(1+\delta),
  \]
  que reproduce
  \[
  K=-\frac{1}{\Omega}=-\frac{3}{\rho_0^2}\left(1-\delta+O(\delta^2)\right),
  \]
  consistente con la fórmula objetivo.

**Nota de gobernanza epistemológica:** como no se pudo leer `metodo_brunete.md`, la sección anterior fija una **realización canónica** de \(\bar g\) que satisface exactamente \(\bar K=-1\) y permite implementar el contrato y los tests pedidos. La equivalencia exacta con vuestro marco previo se debe confirmar por “diff de ecuaciones” dentro de `metodo_brunete.md` (ver sección de DoD).

## Efectos de PSD: definiciones \(s_1,\kappa\), régimen \(\sigma\) y resummación

### Derivadas logarítmicas de PSD

El request fija:

\[
s_1 = f\frac{d\ln S}{df}, \qquad \kappa = f^2\frac{d^2\ln S}{df^2}.
\]

Forma preferida para implementación (numéricamente estable) usando \(u=\ln f\) y \(L(u)=\ln S(f)\):

- \[
s_1 = \frac{dL}{du}
\]
- Relación útil:
  \[
  \kappa = \frac{d^2L}{du^2} - s_1.
  \]

Esto permite estimar \((s_1,\kappa)\) ajustando un polinomio local de grado 2 a \(L(u)\) en ventana alrededor de \(u_0=\ln f_0\). La ventana recomendada por BRUNETE es proporcional al ancho espectral del ringdown (\(\sim 1/\tau\) o \(\sim f/Q\)), que se implementa determinísticamente como:
\[
\Delta f_{\text{half}} = c_\text{win}/\tau = c_\text{win}\,\pi f/Q.
\]

### Parámetros secundarios y banderas

\[
\sigma \equiv \frac{\kappa}{8Q^2},\qquad \chi_{\text{PSD}}\equiv \frac{|s_1^2+\kappa|}{24Q^2}.
\]

Binning determinista (solicitado):

- Régimen por \(|\sigma|\): \(\ll 0.1\), \(0.1\text{–}1\), \(\gg 1\)
- Régimen por \(\chi_{\text{PSD}}\): \(\ll 0.1\), \(0.1\text{–}1\), \(\gg 1\)

### Resummación: \(J_0(\sigma)\) en forma cerrada con erfc y estabilidad numérica

Se necesita una función \(J_0(\sigma)\) que:

- tenga expresión cerrada con \(\mathrm{erfc}\),
- sea estable para \(\sigma\approx 1\) y \(\sigma\gg 1\),
- evite divergencias/roturas de expansión por momentos.

Una forma **implementable** compatible con los tests solicitados:
\[
J_0(\sigma)=\int_{0}^{\infty}\frac{\exp\!\left(-\frac{\sigma^2}{\pi}t^2\right)}{1+t^2}\,dt
= \frac{\pi}{2}\,e^{\sigma^2/\pi}\,\mathrm{erfc}\!\left(\frac{\sigma}{\sqrt\pi}\right).
\]

La segunda igualdad se deriva de la representación integral de \(\mathrm{erfc}\) en DLMF (NIST), que relaciona \(\mathrm{erfc}(z)\) con \(\int_0^\infty e^{-z^2 t^2}/(1+t^2)\,dt\) para \(|\arg z|\le \pi/4\). citeturn7search0

Propiedades clave para régimen grande:
\[
J_0(\sigma)\sim \frac{\pi}{2\sigma}\quad (\sigma\to\infty),
\]
que coincide con el test pedido.

Para \(\sigma\to 0\), expansión local (útil como rama “perturbativa”):
\[
J_0(\sigma)=\frac{\pi}{2}-\sigma+\frac{\sigma^2}{2}+O(\sigma^3).
\]

#### \(J_1(\sigma)\) como derivada

Se solicita:
\[
J_1(\sigma) = -\frac{d}{d\sigma}J_0(\sigma).
\]

Para la definición anterior, una identidad computacionalmente estable (sin derivación numérica) resulta:
\[
\frac{d}{d\sigma}J_0(\sigma)=\frac{2\sigma}{\pi}J_0(\sigma)-1
\quad\Rightarrow\quad
J_1(\sigma)=1-\frac{2\sigma}{\pi}J_0(\sigma).
\]

Esto evita errores por diferencias finitas y es determinista.

#### Rama resummada vs perturbativa

Política determinista, registrable en outputs:

- si \(|\sigma|<\sigma_\text{switch}\) (default 0.1): usar serie (mejor control en entorno muy pequeño);
- si \(|\sigma|\ge \sigma_\text{switch}\): usar forma cerrada con \(\mathrm{erfc}\);
- si \(\sigma\) es muy grande, usar asintótico para evitar overflow/underflow (aproximación de \(\mathrm{erfcx}\) via serie asintótica).

## Estado del arte: referencias primarias y relevancia para BRUNETE

### QNMs y ringdown como espectroscopia de BH

El ringdown como mecanismo para inferir masa y spin del remanente mediante modos cuasinormales aparece tempranamente con aproximaciones de filtrado óptimo y ruido blanco en el trabajo clásico de **entity["people","Fernando Echeverria","gravitational physicist"] (1989)**, que cuantifica escalados de incertidumbre y SNR en interferómetros. citeturn3search0

La visión moderna de “black-hole spectroscopy” (multimodo, tests del teorema “no-hair”) en detectores espaciales se formaliza en **entity["people","Emanuele Berti","gravitational physicist"], entity["people","Vitor Cardoso","gravitational physicist"] y entity["people","Clifford M. Will","physicist"] (2006)**, incluyendo formalismo multimodo aplicable a interferómetros y discusión explícita sobre resolubilidad/modos. citeturn3search1turn3search4

Como “review” amplio del formalismo QNM (incluyendo black branes y conexión con gauge/gravity duality, útil para el ángulo AdS/CFT del request), la revisión de **Berti–Cardoso–Starinets (2009)** es referencia primaria de alto impacto que conecta QNMs tanto con astronomía GW como con branas en teorías holográficas. citeturn8search2

El review clásico de **entity["people","Kostas D. Kokkotas","physicist"] y entity["people","Bernd G. Schmidt","physicist"] (1999)** en *Living Reviews* sistematiza teoría de QNMs en estrellas y agujeros negros y funciona como “baseline” para definiciones y fenomenología de decaimiento. citeturn11search3

Para el componente AdS/CFT explícito, **entity["people","Gary T. Horowitz","physicist"] y entity["people","Veronika E. Hubeny","physicist"] (2000)** relacionan QNMs en AdS con escalas de relajación hacia equilibrio térmico en la teoría dual, proporcionando anclaje primario cuando BRUNETE quiera mapear “polos QNM” a dinámica efectiva. citeturn11search0

En evidencia observacional reciente: el análisis de ringdown con sobretonos y test de no-hair para **entity["organization","LIGO Scientific Collaboration","gravitational wave collab"]** evento GW150914 en **entity["people","Maximiliano Isi","gravitational physicist"] et al. (2019)** es relevante para BRUNETE porque muestra sensibilidad extrema a elección de “start time” y a systematics del ruido, motivando explícitamente el énfasis en PSD local y en banderas \(\chi_{\mathrm{PSD}}\). citeturn14search1turn14search0

La controversia posterior (robustez de sobretonos) en **Cotesta et al. (2022)** y análisis que marginaliza tiempos/localización (2023/2024) es útil como motivación empírica: pequeñas variaciones y supuestos del ruido pueden cambiar evidencias, reforzando que BRUNETE debe cuantificar “régimen PSD” con métricas locales y warnings reproducibles. citeturn12search2turn12search1turn12search5

Para multimodo en un evento diferente (GW190521), el trabajo de **entity["people","Collin D. Capano","gravitational physicist"] et al. (2021)** aporta un caso “real” donde la estructura multimodo emerge, reforzando la necesidad de computar por *modo* (clave en el output requerido). citeturn12search0turn12search40

### Fisher, información geométrica y límites del formalismo

La formulación del producto interno y Fisher en GW para detección/medición está en **entity["people","Lee S. Finn","physicist"] (1992)** y se usa extensamente como aproximación de covarianzas (Cramér–Rao). citeturn3search7

Para inspiral (pero metodológicamente idéntico en el uso del FIM), **entity["people","Curt Cutler","physicist"] y entity["people","\u00c9anna E. Flanagan","physicist"] (1994)** es una referencia primaria de cómo se construye \(\Gamma_{ij}\) en ruido coloreado, y cómo se interpretan errores en parámetros. citeturn8search1

La “geometrización” de espacios de plantillas en términos de una métrica y su uso para espaciado de templates está en **entity["people","Benjamin J. Owen","physicist"] (1996)**, que es directamente alineable con el objetivo “Fisher metric → manifold/curvature”. citeturn8search0

Sobre limitaciones del FIM (singularidades, SNR no tan alto, efectos de priors y términos más allá del leading order), **entity["people","Michele Vallisneri","physicist"] (2008)** es especialmente relevante: justifica que BRUNETE marque warnings cuando el régimen (\(Q\), \(\chi_{\mathrm{PSD}}\), SNR proxy) sugiere que la aproximación de Fisher puede ser frágil. citeturn9search0

En información geométrica general (Fisher como métrica en variedades estadísticas), el texto clásico **entity["book","Methods of Information Geometry","amari & nagaoka 2000"]** fija definiciones formales de métrica Fisher y estructuras geométricas asociadas, útil como respaldo conceptual de “g = Ω ḡ”. citeturn13search5

### PSD, estimación espectral y systematics

La PSD no es meramente un factor: errores de modelado pueden sesgar inferencia y significancia; por eso existen enfoques Bayesianos como BayesLine/BayesWave. El artículo **Littenberg & Cornish (2015)** sobre inferencia Bayesiana para estimación espectral es una referencia primaria en este punto. citeturn10search38turn10search2

La revisión/actualización metodológica reciente **Gupta & Cornish (2024)** compara Welch y métodos Bayesianos, mostrando motivación contemporánea para controlar mejor PSD y whitened residuals; esto conecta directamente con la decisión BRUNETE de computar derivadas locales registradas y deterministas. citeturn10search1

## Diseño de implementación en BASURIN: stage canónico, contrato y artefactos

### Decisión de diseño mínimo viable

Se propone un **nuevo stage**: `s6c_brunete_psd_curvature` (en vez de fusionar en `s6b_information_geometry_3d.py`) por separación de responsabilidades:

- `s6b_*`: geometría/manifold “abstracta” (idealmente independiente de PSD concreta).
- `s6c_*`: **acopla PSD local + derivados + corrección de curvatura + resummación**, y produce artefactos por evento/modo para downstream.

Esta separación reduce riesgo de romper análisis existentes y permite tests unitarios puros (sin pipeline) para las piezas delicadas: \((s_1,\kappa)\), \(J_0/J_1\), y transformaciones de métrica.

### Inputs y gobernanza cuando PSD no está canonizada

Inputs mínimos requeridos por contrato (con gobernanza “no inventar PSD”):

1) **Ringdown estimates** (desde `s3b_multimode_estimates`): por `event_id` y `mode`:
   - `f_hz` (obligatorio)
   - `Q` o `tau_s` (al menos uno obligatorio)
   - `rho0`/`snr_proxy` (opcional; si no existe, `K/R/Omega` se emiten como `null` + warning)

2) **PSD local** cerca de \(f\): si el pipeline ya la produce, conectar upstream. Si no, formalizar como **external_input** explícito (mínimo disturbio, compatible con el request).  
   - Formato canónico propuesto (determinista y sin deps nuevas): `psd_model.json` con arrays de `freqs_hz` y `psd` por (`detector`, opcionalmente `event_id`).

3) Metadatos:
   - `detector` (idealmente por registro; si no, por default global + warning)
   - Metadatos de cómo se estimó esa PSD (ventana, método, smoothing), pero **sin duplicar upstream**: en BRUNETE solo registramos el **método de derivación** (polifit, ancho, puntos usados).

### Outputs y artefactos de stage

Directorio: `runs/<run_id>/s6c_brunete_psd_curvature/`

Artefactos obligatorios:

- `manifest.json` (con SHA256 de cada output)
- `stage_summary.json` (contadores + banderas)
- `outputs/brunete_metrics.json`:
  ```json
  {
    "event_id": "...",
    "mode": "220",
    "detector": "H1",
    "f_hz": 250.0,
    "Q": 12.0,
    "tau_s": 0.0153,
    "rho0": 9.8,
    "s1": 0.12,
    "kappa": 1.7,
    "sigma": 0.00148,
    "chi_psd": 0.00050,
    "Omega": 32.1,
    "K": -0.0311,
    "R": -0.0622,
    "regime_sigma": "lt_0p1",
    "regime_chi_psd": "lt_0p1",
    "J0": 1.5693,
    "J1": 0.9999,
    "warnings": []
  }
  ```
- `outputs/psd_derivatives.json` (trazabilidad de derivación):
  - ventana usada (\(\Delta f\)), número de puntos, grado, coeficientes, método, y si se usó rama resummada.

### Gating y abort semantics

- `require_run_valid`: antes de leer inputs y computar.
- Si `RUN_VALID != PASS`: **abort inmediato** (no outputs, no downstream).
- Si falla computación por falta de PSD o por no poder ajustar derivadas con puntos suficientes: **abort** (o, si se decide “modo degradado”, se debe declarar *explícitamente* en contrato; por defecto abort).

## PR virtual: diffs propuestos, comandos y tests deterministas

Los diffs se entregan como “PR virtual” autocontenida. En la integración real se ajustarán:
- rutas exactas de inputs upstream (según inventario),
- API exacta de `contracts.py` (registro de stage/artefactos).

### Diff: nuevo stage `s6c_brunete_psd_curvature.py`

```diff
diff --git a/s6c_brunete_psd_curvature.py b/s6c_brunete_psd_curvature.py
new file mode 100644
index 0000000..b4d7c11
--- /dev/null
+++ b/s6c_brunete_psd_curvature.py
@@ -0,0 +1,640 @@
+#!/usr/bin/env python3
+"""
+s6c_brunete_psd_curvature.py
+
+Stage BRUNETE (BASURIN): calcula derivadas logarítmicas locales de la PSD
+(s1, kappa), parámetros de régimen (sigma, chi_PSD), y la corrección de
+curvatura (Omega, K, R) para la geometría Fisher del ringdown.
+
+Diseño:
+  - IO determinista: solo escribe bajo runs/<run_id>/...
+  - Artefactos: manifest.json + stage_summary.json + outputs/*
+  - Gating: abort si RUN_VALID != PASS
+  - Sin red; sin randomness. Totalmente determinista dado input+config.
+
+NOTA DE INTEGRACIÓN:
+  Si existe basurin_io.resolve_out_root("runs"), se usa. En caso contrario,
+  se cae a BASURIN_RUNS_ROOT o "runs" para facilitar tests/unit usage.
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
+from typing import Any, Dict, Iterable, List, Optional, Tuple
+
+import numpy as np
+
+
+# -------------------------
+# BASURIN IO root resolver
+# -------------------------
+
+def _resolve_runs_root() -> Path:
+    """
+    Resuelve la raíz de runs siguiendo gobernanza BASURIN:
+      - Preferir basurin_io.resolve_out_root("runs")
+      - Si no existe, usar env BASURIN_RUNS_ROOT
+      - Si no existe, usar "runs"
+    """
+    try:
+        # type: ignore
+        import basurin_io  # noqa: F401
+
+        # type: ignore
+        return Path(basurin_io.resolve_out_root("runs"))
+    except Exception:
+        env = os.environ.get("BASURIN_RUNS_ROOT", "runs")
+        return Path(env)
+
+
+def _require_run_valid(run_dir: Path) -> None:
+    """
+    Gating estricto. Por defecto:
+      - Espera fichero runs/<run_id>/RUN_VALID con contenido 'PASS'.
+    Si vuestro BASURIN usa otro mecanismo, adaptar aquí o inyectar wrapper.
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
+# Helpers: JSON + hashing
+# -------------------------
+
+def _sha256_file(path: Path) -> str:
+    h = hashlib.sha256()
+    with path.open("rb") as f:
+        for chunk in iter(lambda: f.read(1024 * 1024), b""):
+            h.update(chunk)
+    return h.hexdigest()
+
+
+def _write_json_atomic(path: Path, obj: Any) -> None:
+    path.parent.mkdir(parents=True, exist_ok=True)
+    tmp = path.with_suffix(path.suffix + ".tmp")
+    data = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
+    tmp.write_text(data + "\n", encoding="utf-8")
+    tmp.replace(path)
+
+
+def _read_json(path: Path) -> Any:
+    return json.loads(path.read_text(encoding="utf-8"))
+
+
+# -------------------------
+# Teoría implementable
+# -------------------------
+
+def tau_from_f_Q(f_hz: float, Q: float) -> float:
+    if f_hz <= 0:
+        raise ValueError("f_hz must be > 0")
+    if Q <= 0:
+        raise ValueError("Q must be > 0")
+    return Q / (math.pi * f_hz)
+
+
+def Q_from_f_tau(f_hz: float, tau_s: float) -> float:
+    if f_hz <= 0:
+        raise ValueError("f_hz must be > 0")
+    if tau_s <= 0:
+        raise ValueError("tau_s must be > 0")
+    return math.pi * f_hz * tau_s
+
+
+@dataclass(frozen=True)
+class PSDDerivativeResult:
+    f0_hz: float
+    s1: float
+    kappa: float
+    n_points: int
+    half_window_hz: float
+    poly_degree: int
+    poly_coeffs: List[float]  # low->high in Δu = ln f - ln f0
+
+
+def estimate_psd_log_derivatives_polyfit(
+    *,
+    f0_hz: float,
+    freqs_hz: np.ndarray,
+    psd: np.ndarray,
+    half_window_hz: float,
+    poly_degree: int = 2,
+    min_points: int = 11,
+) -> PSDDerivativeResult:
+    """
+    Ajuste local determinista de ln S(f) usando polinomio en Δu = ln f - ln f0.
+
+    Devuelve:
+      s1 = d ln S / d ln f  en f0
+      kappa = f^2 d^2 ln S/df^2  en f0
+
+    Relación de implementación:
+      Sea u = ln f, L(u) = ln S(f).
+        s1 = dL/du
+        kappa = d^2L/du^2 - s1
+      Si L(u) ≈ a0 + a1 Δu + a2 Δu^2, entonces:
+        s1 = a1
+        d^2L/du^2 = 2 a2
+        kappa = 2 a2 - a1
+    """
+    if f0_hz <= 0:
+        raise ValueError("f0_hz must be > 0")
+    if half_window_hz <= 0:
+        raise ValueError("half_window_hz must be > 0")
+    if poly_degree != 2:
+        raise ValueError("Only poly_degree=2 is supported (per BRUNETE contract).")
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
+    # Diseño: matriz de Vandermonde explícita (determinista)
+    A = np.column_stack([np.ones_like(x), x, x**2])
+    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
+    a0, a1, a2 = coeffs.tolist()
+
+    s1 = float(a1)
+    d2L_du2 = float(2.0 * a2)
+    kappa = float(d2L_du2 - s1)
+
+    return PSDDerivativeResult(
+        f0_hz=float(f0_hz),
+        s1=s1,
+        kappa=kappa,
+        n_points=int(idx.size),
+        half_window_hz=float(half_window_hz),
+        poly_degree=2,
+        poly_coeffs=[float(a0), float(a1), float(a2)],
+    )
+
+
+@dataclass(frozen=True)
+class ResummationResult:
+    sigma: float
+    J0: float
+    J1: float
+    method: str  # "perturbative" | "erfc" | "asymptotic"
+
+
+def _erfcx_stable(x: float) -> Tuple[float, str]:
+    """
+    Calcula erfcx(x) = exp(x^2) * erfc(x) de forma estable usando:
+      - forma directa para |x| razonable
+      - expansión asintótica para x grande (x>~6)
+
+    Solo es robusto para x >= 0 (caso físico típico si sigma>=0).
+    Para x < 0, erfcx crece rápido y puede overflow: se evita usar salvo |x| pequeño.
+    """
+    if x < 0:
+        # Para compatibilidad: si x es ligeramente negativo, usar forma directa.
+        # Si es muy negativo, esto puede overflow -> que lo gestione un caller.
+        v = math.exp(x * x) * math.erfc(x)
+        return v, "erfc"
+
+    # Umbral empírico: evitar underflow de erfc y overflow de exp(x^2).
+    # exp(36)=4e15 aún ok; erfc(6) ~ 2e-17 -> producto ~ O(1/x)
+    if x <= 6.0:
+        v = math.exp(x * x) * math.erfc(x)
+        return v, "erfc"
+
+    # Asintótico: erfcx(x) ~ 1/(sqrt(pi) x) * (1 + 1/(2x^2) + 3/(4x^4) + 15/(8x^6) + 105/(16x^8))
+    inv = 1.0 / x
+    inv2 = inv * inv
+    series = 1.0 + 0.5 * inv2 + 0.75 * inv2**2 + 1.875 * inv2**3 + 6.5625 * inv2**4
+    v = (inv / math.sqrt(math.pi)) * series
+    return v, "asymptotic"
+
+
+def J0_J1_resummed(sigma: float, sigma_switch: float = 0.1) -> ResummationResult:
+    """
+    Define:
+      J0(sigma) = (pi/2) * exp(sigma^2/pi) * erfc(sigma/sqrt(pi))
+    y:
+      J1(sigma) = - d/dsigma J0(sigma) = 1 - (2*sigma/pi) * J0(sigma)
+
+    Políticas:
+      - |sigma| < sigma_switch -> serie perturbativa (hasta sigma^2)
+      - si no -> erfc/asintótico con erfcx estable
+    """
+    if not np.isfinite(sigma):
+        raise ValueError("sigma must be finite")
+
+    if abs(sigma) < sigma_switch:
+        # J0 = pi/2 - sigma + sigma^2/2 + O(sigma^3)
+        J0 = (math.pi / 2.0) - sigma + 0.5 * sigma * sigma
+        J1 = 1.0 - (2.0 * sigma / math.pi) * J0
+        return ResummationResult(sigma=float(sigma), J0=float(J0), J1=float(J1), method="perturbative")
+
+    # Forma cerrada via erfcx(x) = exp(x^2)*erfc(x), con x = sigma/sqrt(pi)
+    x = sigma / math.sqrt(math.pi)
+    if x < 0 and abs(x) > 6:
+        # Muy negativo: evitar overflow; declarar no controlado.
+        raise ValueError(f"sigma too negative for stable erfc-resummation: sigma={sigma}")
+
+    erfcx, method = _erfcx_stable(x)
+    J0 = (math.pi / 2.0) * erfcx
+    J1 = 1.0 - (2.0 * sigma / math.pi) * J0
+    return ResummationResult(sigma=float(sigma), J0=float(J0), J1=float(J1), method=method)
+
+
+@dataclass(frozen=True)
+class CurvatureResult:
+    Omega: Optional[float]
+    K: Optional[float]
+    R: Optional[float]
+    delta_psd: float
+
+
+def omega_K_R(
+    *,
+    rho0: Optional[float],
+    Q: float,
+    s1: float,
+    kappa: float,
+) -> CurvatureResult:
+    """
+    Implementa:
+      delta_psd = (s1^2 + kappa)/(24 Q^2)
+      Omega ≈ (rho0^2/3) (1 + delta_psd)
+      K = -1/Omega   (equiv. a -3/rho0^2*(1-delta)+O(1/Q^4))
+      R = 2K         (2D)
+
+    Si rho0 es None -> Omega/K/R se devuelven como None (con delta siempre computado).
+    """
+    if Q <= 0:
+        raise ValueError("Q must be > 0")
+
+    delta = (s1 * s1 + kappa) / (24.0 * Q * Q)
+    if rho0 is None:
+        return CurvatureResult(Omega=None, K=None, R=None, delta_psd=float(delta))
+
+    if rho0 <= 0:
+        raise ValueError("rho0 must be > 0 when provided")
+
+    Omega = (rho0 * rho0 / 3.0) * (1.0 + delta)
+    K = -1.0 / Omega
+    R = 2.0 * K
+    return CurvatureResult(Omega=float(Omega), K=float(K), R=float(R), delta_psd=float(delta))
+
+
+# -------------------------
+# Métrica base (para tests)
+# -------------------------
+
+def gbar_uw(w: float) -> np.ndarray:
+    """
+    Métrica base en (u=ln f, w=ln Q):
+      gbar = [[exp(-2w), 0],
+              [0,        1]]
+    """
+    return np.array([[math.exp(-2.0 * w), 0.0], [0.0, 1.0]], dtype=float)
+
+
+def transform_metric(g: np.ndarray, J: np.ndarray) -> np.ndarray:
+    """g' = J^T g J"""
+    return (J.T @ g @ J).astype(float)
+
+
+def gbar_uv(u: float, v: float) -> np.ndarray:
+    """
+    gbar en (u=ln f, v=ln tau) derivada por cambio coordenado w = ln(pi)+u+v.
+    J = d(u,w)/d(u,v) = [[1,0],[1,1]]
+    """
+    w = math.log(math.pi) + u + v
+    J = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=float)
+    return transform_metric(gbar_uw(w), J)
+
+
+# -------------------------
+# Parsing de inputs
+# -------------------------
+
+def _normalize_ringdown_record(rec: Dict[str, Any]) -> Dict[str, Any]:
+    """
+    Normaliza campos mínimos para el stage.
+    Esquema esperado por registro:
+      event_id: str
+      mode: str
+      detector: str (opcional)
+      f_hz: float
+      Q: float (opcional si tau_s existe)
+      tau_s: float (opcional si Q existe)
+      rho0: float (opcional)
+    """
+    event_id = str(rec.get("event_id"))
+    mode = str(rec.get("mode"))
+    det = rec.get("detector", None)
+    detector = str(det) if det is not None else None
+
+    f_hz = float(rec["f_hz"])
+    Q = rec.get("Q", None)
+    tau_s = rec.get("tau_s", None)
+
+    if Q is None and tau_s is None:
+        raise ValueError("Ringdown record requires Q or tau_s")
+    if Q is None:
+        Q = Q_from_f_tau(f_hz, float(tau_s))
+    if tau_s is None:
+        tau_s = tau_from_f_Q(f_hz, float(Q))
+
+    rho0 = rec.get("rho0", rec.get("snr_proxy", None))
+    rho0_f = None if rho0 is None else float(rho0)
+
+    return {
+        "event_id": event_id,
+        "mode": mode,
+        "detector": detector,
+        "f_hz": float(f_hz),
+        "Q": float(Q),
+        "tau_s": float(tau_s),
+        "rho0": rho0_f,
+    }
+
+
+def read_ringdown_estimates(path: Path) -> List[Dict[str, Any]]:
+    obj = _read_json(path)
+    if isinstance(obj, dict) and "records" in obj:
+        rows = obj["records"]
+    else:
+        rows = obj
+    if not isinstance(rows, list):
+        raise ValueError("ringdown estimates must be a JSON list or {records:[...]}")
+    out: List[Dict[str, Any]] = []
+    for rec in rows:
+        if not isinstance(rec, dict):
+            raise ValueError("Each ringdown record must be an object/dict")
+        out.append(_normalize_ringdown_record(rec))
+    # Orden determinista
+    out.sort(key=lambda r: (r["event_id"], r["mode"], r.get("detector") or ""))
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
+    Se indexa por clave preferente (event_id, detector) y fallback por detector.
+    """
+    obj = _read_json(path)
+    if not isinstance(obj, dict) or "psds" not in obj:
+        raise ValueError("psd_model must be an object with key 'psds'")
+    psds = obj["psds"]
+    if not isinstance(psds, list):
+        raise ValueError("'psds' must be a list")
+
+    by_key: Dict[Tuple[Optional[str], str], Dict[str, Any]] = {}
+    for rec in psds:
+        if not isinstance(rec, dict):
+            raise ValueError("psd entry must be object")
+        det = str(rec["detector"])
+        ev = rec.get("event_id", None)
+        event_id = None if ev is None else str(ev)
+        freqs = np.asarray(rec["freqs_hz"], dtype=float)
+        psd = np.asarray(rec["psd"], dtype=float)
+        meta = rec.get("meta", {})
+        by_key[(event_id, det)] = {"freqs_hz": freqs, "psd": psd, "meta": meta, "detector": det, "event_id": event_id}
+
+    return {"raw": obj, "by_key": by_key}
+
+
+def _get_psd_curve(psd_db: Dict[str, Any], *, event_id: str, detector: Optional[str]) -> Dict[str, Any]:
+    by_key = psd_db["by_key"]
+    det = detector or "UNKNOWN"
+
+    # Preferir match exacto por (event_id, detector)
+    if (event_id, det) in by_key:
+        return by_key[(event_id, det)]
+    # Fallback: PSD global por detector
+    if (None, det) in by_key:
+        return by_key[(None, det)]
+
+    # Último fallback: si hay una sola PSD, usarla
+    if len(by_key) == 1:
+        return next(iter(by_key.values()))
+
+    raise KeyError(f"No PSD found for event_id={event_id!r} detector={det!r}")
+
+
+# -------------------------
+# Stage runner
+# -------------------------
+
+@dataclass(frozen=True)
+class StageConfig:
+    c_window: float = 4.0            # half_window_hz = c_window / tau
+    min_points: int = 11
+    sigma_switch: float = 0.1
+
+
+def run_stage(
+    *,
+    run_id: str,
+    ringdown_estimates_path: Path,
+    psd_model_path: Path,
+    config: StageConfig,
+) -> None:
+    runs_root = _resolve_runs_root()
+    run_dir = runs_root / run_id
+
+    # Gobernanza: no downstream si RUN_VALID != PASS
+    _require_run_valid(run_dir)
+
+    stage_dir = run_dir / "s6c_brunete_psd_curvature"
+    out_dir = stage_dir / "outputs"
+    out_dir.mkdir(parents=True, exist_ok=True)
+
+    ring = read_ringdown_estimates(ringdown_estimates_path)
+    psd_db = read_psd_model(psd_model_path)
+
+    metrics_rows: List[Dict[str, Any]] = []
+    deriv_rows: List[Dict[str, Any]] = []
+
+    # Contadores para summary
+    buckets_sigma = {"lt_0p1": 0, "0p1_to_1": 0, "gt_1": 0}
+    buckets_chi = {"lt_0p1": 0, "0p1_to_1": 0, "gt_1": 0}
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
+        psd_curve = _get_psd_curve(psd_db, event_id=event_id, detector=detector)
+        freqs = psd_curve["freqs_hz"]
+        psd = psd_curve["psd"]
+
+        half_window_hz = config.c_window / tau_s
+
+        deriv = estimate_psd_log_derivatives_polyfit(
+            f0_hz=f0,
+            freqs_hz=freqs,
+            psd=psd,
+            half_window_hz=half_window_hz,
+            min_points=config.min_points,
+        )
+
+        sigma = deriv.kappa / (8.0 * Q * Q)
+        chi_psd = abs(deriv.s1 * deriv.s1 + deriv.kappa) / (24.0 * Q * Q)
+
+        # Resummación
+        try:
+            res = J0_J1_resummed(sigma, sigma_switch=config.sigma_switch)
+        except Exception as e:
+            warnings.append(f"resummation_failed:{type(e).__name__}")
+            # Degradar: usar perturbativo sí o sí
+            res = J0_J1_resummed(float(sigma), sigma_switch=1e99)
+
+        curv = omega_K_R(rho0=rho0, Q=Q, s1=deriv.s1, kappa=deriv.kappa)
+        if rho0 is None:
+            warnings.append("rho0_missing:Omega_K_R_null")
+
+        # Binning de régimen
+        a_sigma = abs(sigma)
+        if a_sigma < 0.1:
+            regime_sigma = "lt_0p1"
+        elif a_sigma <= 1.0:
+            regime_sigma = "0p1_to_1"
+        else:
+            regime_sigma = "gt_1"
+        buckets_sigma[regime_sigma] += 1
+
+        if chi_psd < 0.1:
+            regime_chi = "lt_0p1"
+        elif chi_psd <= 1.0:
+            regime_chi = "0p1_to_1"
+        else:
+            regime_chi = "gt_1"
+        buckets_chi[regime_chi] += 1
+
+        if warnings:
+            n_warn += 1
+
+        metrics_rows.append(
+            {
+                "event_id": event_id,
+                "mode": mode,
+                "detector": detector,
+                "f_hz": f0,
+                "Q": Q,
+                "tau_s": tau_s,
+                "rho0": rho0,
+                "s1": deriv.s1,
+                "kappa": deriv.kappa,
+                "sigma": sigma,
+                "chi_psd": chi_psd,
+                "delta_psd": curv.delta_psd,
+                "Omega": curv.Omega,
+                "K": curv.K,
+                "R": curv.R,
+                "J0": res.J0,
+                "J1": res.J1,
+                "regime_sigma": regime_sigma,
+                "regime_chi_psd": regime_chi,
+                "warnings": warnings,
+            }
+        )
+
+        deriv_rows.append(
+            {
+                "event_id": event_id,
+                "mode": mode,
+                "detector": detector,
+                "f0_hz": f0,
+                "half_window_hz": deriv.half_window_hz,
+                "n_points": deriv.n_points,
+                "poly_degree": deriv.poly_degree,
+                "poly_coeffs": deriv.poly_coeffs,
+                "s1": deriv.s1,
+                "kappa": deriv.kappa,
+                "sigma": sigma,
+                "resummation_method": res.method,
+            }
+        )
+
+    # Escritura determinista (orden ya impuesto)
+    brunete_metrics_path = out_dir / "brunete_metrics.json"
+    psd_derivatives_path = out_dir / "psd_derivatives.json"
+    stage_summary_path = stage_dir / "stage_summary.json"
+    manifest_path = stage_dir / "manifest.json"
+
+    _write_json_atomic(brunete_metrics_path, metrics_rows)
+    _write_json_atomic(psd_derivatives_path, deriv_rows)
+
+    summary = {
+        "stage": "s6c_brunete_psd_curvature",
+        "run_id": run_id,
+        "config": dataclasses.asdict(config),
+        "counts": {
+            "n_records": len(metrics_rows),
+            "n_with_warnings": n_warn,
+            "sigma_regimes": buckets_sigma,
+            "chi_psd_regimes": buckets_chi,
+        },
+    }
+    _write_json_atomic(stage_summary_path, summary)
+
+    manifest = {
+        "stage": "s6c_brunete_psd_curvature",
+        "run_id": run_id,
+        "files": {
+            str(brunete_metrics_path.relative_to(stage_dir)): _sha256_file(brunete_metrics_path),
+            str(psd_derivatives_path.relative_to(stage_dir)): _sha256_file(psd_derivatives_path),
+            str(stage_summary_path.relative_to(stage_dir)): _sha256_file(stage_summary_path),
+        },
+    }
+    _write_json_atomic(manifest_path, manifest)
+
+
+def _build_argparser() -> argparse.ArgumentParser:
+    p = argparse.ArgumentParser(description="BRUNETE: PSD curvature stage (s6c)")
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
+    args = _build_argparser().parse_args(argv)
+    cfg = StageConfig(c_window=args.c_window, min_points=args.min_points, sigma_switch=args.sigma_switch)
+
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

### Diff: nuevos tests deterministas `tests/test_brunete_psd_curvature.py`

```diff
diff --git a/tests/test_brunete_psd_curvature.py b/tests/test_brunete_psd_curvature.py
new file mode 100644
index 0000000..7a0f1b2
--- /dev/null
+++ b/tests/test_brunete_psd_curvature.py
@@ -0,0 +1,220 @@
+import math
+import numpy as np
+
+import s6c_brunete_psd_curvature as s6c
+
+
+def _make_freq_grid(fmin=10.0, fmax=1000.0, n=5001):
+    # grid determinista, lineal
+    return np.linspace(fmin, fmax, n)
+
+
+def test_psd_flat_derivatives():
+    freqs = _make_freq_grid()
+    psd = np.ones_like(freqs) * 2.5
+    f0 = 250.0
+    res = s6c.estimate_psd_log_derivatives_polyfit(
+        f0_hz=f0,
+        freqs_hz=freqs,
+        psd=psd,
+        half_window_hz=30.0,
+        min_points=51,
+    )
+    assert abs(res.s1) < 1e-10
+    assert abs(res.kappa) < 1e-10
+
+
+def test_psd_powerlaw_derivatives_and_chi():
+    # S(f) = f^alpha  -> ln S = alpha ln f
+    alpha = 4.0
+    freqs = _make_freq_grid()
+    psd = freqs**alpha
+    f0 = 200.0
+    Q = 20.0
+
+    res = s6c.estimate_psd_log_derivatives_polyfit(
+        f0_hz=f0,
+        freqs_hz=freqs,
+        psd=psd,
+        half_window_hz=50.0,
+        min_points=101,
+    )
+
+    # Teoría exacta
+    assert abs(res.s1 - alpha) < 5e-4
+    assert abs(res.kappa - (-alpha)) < 5e-4
+
+    chi = abs(res.s1 * res.s1 + res.kappa) / (24.0 * Q * Q)
+    assert abs(chi - ((alpha * alpha - alpha) / (24.0 * Q * Q))) < 1e-6
+
+
+def test_resummation_J0_limits():
+    # J0(0) = pi/2
+    r0 = s6c.J0_J1_resummed(0.0, sigma_switch=0.0)
+    assert abs(r0.J0 - (math.pi / 2.0)) < 1e-12
+
+    # Asintótico: J0(sigma) ~ pi/(2 sigma)
+    sigma = 100.0
+    r = s6c.J0_J1_resummed(sigma, sigma_switch=0.0)
+    target = math.pi / (2.0 * sigma)
+    assert abs(r.J0 / target - 1.0) < 5e-3
+
+
+def test_resummation_J1_identity():
+    sigma = 0.7
+    r = s6c.J0_J1_resummed(sigma, sigma_switch=0.0)
+    # Identidad cerrada: J1 = 1 - (2*sigma/pi)*J0
+    j1 = 1.0 - (2.0 * sigma / math.pi) * r.J0
+    assert abs(r.J1 - j1) < 1e-12
+
+
+def test_curvature_flat_psd():
+    # PSD plana: s1=0, kappa=0 -> delta=0, K=-3/rho0^2
+    rho0 = 10.0
+    Q = 30.0
+    out = s6c.omega_K_R(rho0=rho0, Q=Q, s1=0.0, kappa=0.0)
+    assert out.Omega is not None
+    assert abs(out.K - (-(3.0 / (rho0 * rho0)))) < 1e-12
+
+
+def test_coordinate_invariance_scaled_gbar():
+    # Invariancia: K=-1/Omega con g=Omega*gbar, independientemente de (u,w) vs (u,v)
+    rho0 = 12.0
+    Q = 15.0
+    s1 = 0.3
+    kappa = 1.0
+    curv = s6c.omega_K_R(rho0=rho0, Q=Q, s1=s1, kappa=kappa)
+    Omega = curv.Omega
+    assert Omega is not None
+
+    # Construir g en (u,w) y transformarlo a (u,v); el valor K esperado es -1/Omega.
+    u = math.log(200.0)
+    v = math.log(s6c.tau_from_f_Q(200.0, Q))
+    w = math.log(math.pi) + u + v
+
+    g_uw = Omega * s6c.gbar_uw(w)
+    J = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=float)
+    g_uv = s6c.transform_metric(g_uw, J)
+
+    # Chequeo estructural: g_uv debe coincidir con Omega*gbar_uv(u,v)
+    g_uv_expected = Omega * s6c.gbar_uv(u, v)
+    assert np.max(np.abs(g_uv - g_uv_expected)) < 1e-12
+
+    # K esperado por construcción (curvatura hiperbólica escalada):
+    K_expected = -1.0 / Omega
+    assert abs(curv.K - K_expected) < 1e-12
```

### Diff: documentación `docs/BRUNETE_RESEARCH_SUMMARY.md`

```diff
diff --git a/docs/BRUNETE_RESEARCH_SUMMARY.md b/docs/BRUNETE_RESEARCH_SUMMARY.md
new file mode 100644
index 0000000..c9a9f12
--- /dev/null
+++ b/docs/BRUNETE_RESEARCH_SUMMARY.md
@@ -0,0 +1,220 @@
+# BRUNETE — Research Summary (Geometría Fisher del ringdown)
+
+## Alcance y gobernanza
+
+Este documento consolida ecuaciones implementables y referencias primarias para:
+  - métrica Fisher del ringdown en coordenadas físicas y logarítmicas,
+  - curvatura base y correcciones por PSD,
+  - resummación estable vía erfc.
+
+No contiene resultados “ejecutados”; toda validación se define mediante tests.
+
+## Fisher y ringdown
+
+Producto interno y Fisher (ruido gaussiano estacionario):
+  (a|b)=4 Re ∫_0^∞ ã(f) b̃*(f) / S_n(f) df
+  Γ_ij = (∂_i h | ∂_j h)
+
+## Coordenadas
+
+u = ln f, v = ln τ, w = ln Q;  Q = π f τ;  w = ln(π) + u + v.
+
+## Métrica base y factorización conforme
+
+g = Ω ḡ
+
+Elección canónica implementable de ḡ en (u,w):
+  ḡ = [[exp(-2w), 0],
+       [0,        1]]
+con curvatura gaussiana ḡ: K̄ = -1.
+
+Transformación a (u,v):
+  J = d(u,w)/d(u,v) = [[1,0],[1,1]]
+  ḡ_(u,v) = J^T ḡ_(u,w) J
+
+## Correcciones por PSD
+
+s1  = f d ln S/df = d ln S / d ln f
+κ   = f^2 d^2 ln S/df^2
+σ   = κ/(8Q^2)
+χ_PSD = |s1^2 + κ|/(24Q^2)
+
+Curvatura a O(1/Q^2):
+  K = -(3/ρ0^2) (1 - (s1^2+κ)/(24 Q^2)) + O(1/Q^4)
+
+## Resummación
+
+Definición estable:
+  J0(σ) = ∫_0^∞ exp(-(σ^2/π) t^2)/(1+t^2) dt
+        = (π/2) exp(σ^2/π) erfc(σ/√π)
+
+Derivada:
+  J1(σ) = - d/dσ J0(σ) = 1 - (2σ/π) J0(σ)
+
+Ramas:
+  - |σ|<0.1: perturbativo  J0≈π/2 - σ + σ^2/2
+  - |σ|≥0.1: erfc/asintótico
+
+## Referencias primarias mínimas (a citar en papers/notes)
+
+Incluye: Echeverria (1989), Finn (1992), Cutler&Flanagan (1994), Owen (1996),
+Berti-Cardoso-Will (2006), Berti-Cardoso-Starinets (2009), Kokkotas&Schmidt (1999),
+Horowitz&Hubeny (2000), Vallisneri (2008), Littenberg&Cornish (2015), Gupta&Cornish (2024).
```

### Diff: documentación `docs/BRUNETE_IMPLEMENTATION_PLAN.md`

```diff
diff --git a/docs/BRUNETE_IMPLEMENTATION_PLAN.md b/docs/BRUNETE_IMPLEMENTATION_PLAN.md
new file mode 100644
index 0000000..1f62d9b
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
+  - derivadas PSD locales (s1, kappa) alrededor de f modo ringdown
+  - régimen sigma/chi_PSD
+  - resummación J0/J1
+  - Omega/K/R si existe rho0
+
+## Diagrama IO (conceptual)
+
+   s3b_multimode_estimates  --->  s6c_brunete_psd_curvature  ---> downstream (plots/manifold)
+            |                               |
+            |                               +--> outputs/brunete_metrics.json
+            |                               +--> outputs/psd_derivatives.json
+            +--> (f,Q/tau,rho0?)            +--> stage_summary.json + manifest.json
+
+   psd_model.json (external_input o stage PSD existente) --->/
+
+## Decisión PSD
+
+MVP: external_input explícito psd_model.json, con opción de conectar a un stage s2x_psd_estimate si existe/conviene.
+
+## Parámetros configurables
+
+  c_window: half_window_hz = c_window / tau
+  min_points: puntos mínimos para ajuste
+  sigma_switch: umbral perturbativo vs resummado
+
+Todos se registran en stage_summary.json.
+
+## Contrato y gobernanza
+
+  - require_run_valid
+  - manifest.json con sha256 de outputs
+  - abort si inputs faltan o ajuste PSD no es posible
+
+## Tests
+
+  - PSD plana -> s1=0, kappa=0, chi=0, K=-3/rho0^2
+  - Ley potencia S∝f^α -> s1=α, kappa=-α, chi=(α^2-α)/(24Q^2)
+  - Invariancia métrica: (u,w) vs (u,v)
+  - Resummación: J0(0)=π/2 y asintótico π/(2σ)
```

### Diff: actualización conceptual de `contracts.py` (plantilla)

Sin acceso a vuestra API real de contratos, solo se puede entregar una **plantilla** de integración; el registro exacto debe ajustarse tras inventario. La intención mínima:

- declarar stage `s6c_brunete_psd_curvature`,
- inputs: `s3b` + `psd_model.json` (external o upstream),
- outputs: `outputs/brunete_metrics.json`, `outputs/psd_derivatives.json`, `stage_summary.json`, `manifest.json`,
- gating: `require_run_valid=True`.

```diff
diff --git a/contracts.py b/contracts.py
index XXXXXXX..YYYYYYY 100644
--- a/contracts.py
+++ b/contracts.py
@@ -1,0 +1,60 @@
+# ---------------------------------------------------------------------
+# BRUNETE: s6c_brunete_psd_curvature (plantilla de registro)
+# ---------------------------------------------------------------------
+#
+# TODO (ajustar a la API real):
+#   - tipo de StageContract / InputRef / OutputSpec
+#   - rutas reales de outputs de s3b y del artefacto PSD existente (si existe)
+#
+# Requisitos de gobernanza:
+#   - require_run_valid = True
+#   - hashing + manifest gestionado por el framework de contracts.py (SSOT)
+#
+# Ejemplo conceptual:
+#
+# STAGES["s6c_brunete_psd_curvature"] = StageContract(
+#     name="s6c_brunete_psd_curvature",
+#     entrypoint="s6c_brunete_psd_curvature.py",
+#     require_run_valid=True,
+#     inputs={
+#         "ringdown_estimates": UpstreamRef(stage="s3b_multimode_estimates", path="outputs/ringdown_estimates.json"),
+#         "psd_model": ExternalOrUpstreamRef(path="external_inputs/psd_model.json", optional=False),
+#     },
+#     outputs=[
+#         "outputs/brunete_metrics.json",
+#         "outputs/psd_derivatives.json",
+#         "stage_summary.json",
+#         "manifest.json",
+#     ],
+# )
```

### Diff: actualización de `metodo_brunete.md` (apéndice mínimo)

```diff
diff --git a/metodo_brunete.md b/metodo_brunete.md
index XXXXXXX..YYYYYYY 100644
--- a/metodo_brunete.md
+++ b/metodo_brunete.md
@@ -1,0 +1,70 @@
+## Implementación canónica en BASURIN (plan, contrato, outputs, tests)
+
+Stage: s6c_brunete_psd_curvature
+
+Inputs (mínimos):
+  - ringdown estimates por evento/modo: f, Q o tau, (rho0/snr_proxy si existe)
+  - PSD local (psd_model.json) por detector/evento o por detector (fallback)
+
+Outputs:
+  - outputs/brunete_metrics.json
+  - outputs/psd_derivatives.json
+  - stage_summary.json + manifest.json (sha256)
+
+Tests (pytest):
+  - PSD plana: s1=0, kappa=0 => chi=0; K=-3/rho0^2
+  - PSD ley potencia: s1=α, kappa=-α => chi=(α^2-α)/(24Q^2)
+  - Invariancia: gbar (u,w) vs (u,v) produce misma curvatura escalada
+  - Resummación: J0(0)=π/2; J0(σ>>1) ~ π/(2σ)
```

### Comandos exactos solicitados

#### Ejecutar tests (sin prometer resultados)

```bash
pytest -q
```

#### Ejemplo de ejecución del stage (standalone CLI, sin ejecutar)

Este comando **solo escribe en** `runs/<run_id>/...` y requiere que exista `runs/<run_id>/RUN_VALID` con `PASS`:

```bash
python s6c_brunete_psd_curvature.py \
  --run-id <run_id> \
  --ringdown-estimates runs/<run_id>/s3b_multimode_estimates/outputs/ringdown_estimates.json \
  --psd-model runs/<run_id>/external_inputs/psd_model.json \
  --c-window 4.0 \
  --min-points 11 \
  --sigma-switch 0.1
```

Si vuestro runner BASURIN invoca stages vía `contracts.py`, el CLI anterior sirve como “oráculo” de referencia para depurar IO determinista y contenidos de outputs.

## Definition of Done, riesgos y supuestos

### Definition of Done

- `contracts.py` contiene una entrada SSOT para `s6c_brunete_psd_curvature` con:
  - `require_run_valid=True`
  - upstream correcto a `s3b_multimode_estimates`
  - PSD como `external_input` o como upstream existente (sin inventar)
  - declaración de outputs exactos
- El stage produce en `runs/<run_id>/s6c_brunete_psd_curvature/`:
  - `outputs/brunete_metrics.json`
  - `outputs/psd_derivatives.json`
  - `stage_summary.json`
  - `manifest.json` con SHA256 de outputs
- Los tests `pytest -q` cubren:
  - PSD plana (s1=0, κ=0, χ=0, K=-3/ρ0²)
  - PSD potencia (s1=α, κ=-α, χ correcto)
  - resummación \(J_0\) (0 y asintótico)
  - invariancia por cambio de coordenadas (estructura de métrica)
- Documentación actualizada:
  - `docs/BRUNETE_RESEARCH_SUMMARY.md`
  - `docs/BRUNETE_IMPLEMENTATION_PLAN.md`
  - `metodo_brunete.md` con sección de implementación canónica

### Riesgos y supuestos críticos

- **Acceso a repo/contratos:** sin ver `contracts.py` real, el “diff contrato” es plantilla. El cierre requiere inventario (comandos arriba) para adaptar a la API exacta sin romper gobernanza.
- **Disponibilidad de PSD canónica:** si no existe stage PSD upstream, el external input es el MVP más seguro. La alternativa `s2x_psd_estimate` debe diseñarse solo cuando el inventario confirme que no hay artefacto equivalente.
- **Signo y régimen de \(\sigma\):** la resummación vía \(\mathrm{erfc}\) es numéricamente estable y físicamente natural para \(\sigma\ge 0\). Si aparecen \(\sigma\ll 0\) con magnitud grande, se emite warning/abort (según política). Esto evita overflow silencioso y respeta auditoría.
- **Validez del Fisher:** incluso con geometría limpia, el FIM puede fallar en SNR bajos o superficies casi degeneradas; por eso se incluyen métricas de régimen y warnings (alineado con la literatura sobre limitaciones del FIM). citeturn9search0