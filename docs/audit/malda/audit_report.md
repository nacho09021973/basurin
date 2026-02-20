# Auditoría de Scripts — carpeta `malda/`

**Fecha:** 2026-02-20
**Rama:** `claude/audit-malda-scripts-AJX0G`
**Auditor:** Claude (Anthropic)
**Total de scripts auditados:** 19

---

## Resumen ejecutivo

La carpeta `malda/` contiene el pipeline completo del proyecto **CUERDAS-Maldacena**, cuyo objetivo es descubrir geometría holográfica emergente a partir de datos de frontera (CFT / LIGO). El pipeline se organiza en tres bloques:

- **Bloque 0 (00–00b):** Carga y validación de datos.
- **Bloque A (01–05):** Generación de sandbox, extracción de polos de ringdown, reconstrucción de geometría emergente y descubrimiento de ecuaciones de bulk.
- **Bloque C (06–09):** Espectro escalar, diccionario holográfico emergente y contratos finales.

El pipeline exhibe una preocupación sostenida por la **honestidad epistémica**: separa rigurosamente los datos de frontera visibles al learner de la verdad del bulk (usada solo para validación), etiqueta explícitamente los análisis post-hoc, e implementa controles negativos formales.

---

## Scripts auditados

### `00_load_ligo_data.py` (v3)

**Propósito:** Adaptador que convierte archivos NPZ de GWOSC/GWpy (ventanas de ringdown) en artefactos HDF5 de frontera CUERDAS, con manifest determinista y hashes SHA-256.

**Funcionalidades clave:**
- Detrending (mean / linear), ventana Hann, FFT opcional.
- Symlinks atómicos (`os.replace`) para evitar escrituras parciales.
- Protección contra path traversal (`..` rechazado explícitamente).
- Fingerprinting de entradas (SHA-256) en manifest.

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| BAJA | `project_root = Path(__file__).resolve().parent` resuelve como `malda/`, no como la raíz del repositorio. Rutas relativas pasadas por CLI se interpretan bajo `malda/`, lo que puede sorprender al usuario. |
| BAJA | El wrapper `_sha256_file` lee en chunks de 1 MB, correcto para archivos grandes. Sin embargo, no verifica integridad tras escritura (solo antes). |
| INFO | No hay validación del rango de `fs` (frecuencia de muestreo). Un NPZ con `fs=0` causará `ZeroDivisionError` al calcular `dt`. |

**Estado general:** ✅ Robusto y auditable.

---

### `00_validate_io_contracts.py` (v1.0)

**Propósito:** Validador de contratos IO v1 para archivos sandbox HDF5, emergent HDF5, CSV de modos y JSON de diccionario.

**Funcionalidades clave:**
- Validación de atributos raíz (`name`, `family`, `d`, `category`, `provenance`).
- Verificación de monotonicidad de `z_grid` y longitudes consistentes entre datasets.
- Detección de aliases legacy (`A_emergent` → `A_of_z`).
- Reporte JSON con niveles ERROR / WARN / INFO y exit code configurable (`--strict`).

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| MEDIA | La serialización de `issues` en el reporte JSON usa `asdict(i) if hasattr(i, '__dataclass_fields__') else i` (línea ~886). Si `issues` contiene objetos no-dataclass por alguna ruta de código no habitual, el JSON resultante no será uniforme. |
| BAJA | El conjunto `VALID_PROVENANCES` se define pero no se valida en los archivos sandbox (solo en emergent). Inconsistencia menor de contrato. |
| BAJA | Encoding UTF-8 incorrecto en el docstring (caracteres mojibake visibles), probablemente artefacto de edición. No afecta ejecución. |
| INFO | No valida que `lambda_sl` sea numéricamente razonable (solo comprueba NaN/Inf). |

**Estado general:** ✅ Funcionalmente correcto con mejoras menores pendientes.

---

### `00b_physics_sanity_checks.py` (v1.0)

**Propósito:** Verificación física **post-hoc** (NO filtra datos). Contrasta autovalores λ_SL y dimensiones Δ con relaciones teóricas de AdS/CFT.

**Checks implementados:**
1. **Cota Breitenlohner-Freedman:** `λ_SL ≥ -(d/2)²` (AGMOO Sec. 2.2.2, Eq. 2.42).
2. **Relación masa-dimensión:** `Δ = d/2 ± √(d²/4 + λ_SL)` (AGMOO Sec. 3.1.2, Eq. 3.14).
3. **Cota de unitariedad CFT:** `Δ ≥ (d_CFT - 2)/2` (AGMOO Sec. 3.1.3).

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| MEDIA | La convención de dimensiones puede ser ambigua: el script usa `d` como dimensión del bulk, y calcula `d_cft = d - 1` para la cota de unitariedad. Si los scripts upstream usan `d` como dimensión del boundary (CFT), los checks de unitariedad estarán desplazados en 1. Requiere verificación de convención global. |
| BAJA | Exit code siempre 0 (nunca rechaza datos). Esto es intencional y está documentado, pero puede confundir en pipelines CI que esperan exit ≠ 0 ante anomalías. |
| INFO | Las citas teóricas son precisas y rastreables (sección + ecuación del AGMOO Review). |

**Estado general:** ✅ Bien diseñado. Honestidad epistémica explícita y documentada.

---

### `01_extract_ringdown_poles.py` (v1.1)

**Propósito:** Extracción operacional de polos de exponenciales amortiguadas ("polos de ringdown") a partir de señales de strain LIGO mediante el método ESPRIT / matrix-pencil.

**Funcionalidades clave:**
- Sin dependencias de SciPy ni GWpy (solo NumPy).
- Sin inyección de teoría GR/Kerr.
- Hankel matrix → SVD → eigenvalues → polos continuos `q = log(z)/dt`.
- Ordenación por menor amortiguamiento; filtro opcional de modos decayentes.
- Salida: JSON + CSV de polos, `run_manifest.json` actualizado.

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| MEDIA | `q = np.log(z) / dt` usa la rama principal del logaritmo complejo. Para polos cerca del eje real (z ≈ real positivo), la fase puede ser discontinua entre iteraciones, produciendo `omega_qnm` inestables. No hay manejo de esta discontinuidad. |
| BAJA | La fusión de polos joint (H1+L1) redondea a 3 decimales para deduplicar; esto puede eliminar polos físicamente distintos muy cercanos. |
| BAJA | `L = min(max(2, Nw // 2), 4096)` es un heurístico razonable pero no documentado en la ayuda CLI. |
| INFO | El método ESPRIT asume señal como suma de exponenciales complejas. Para señales LIGO con ruido no-gaussiano, la precisión de los polos depende fuertemente de la elección de ventana. |

**Estado general:** ✅ Correcto para uso operacional. Limitaciones propias del método ESPRIT documentadas.

---

### `01_generate_sandbox_geometries.py` (v3)

**Propósito:** Generador de universos sandbox con geometrías holográficas controladas (AdS, Lifshitz, hyperscaling, Dp-branas, deformed, unknown). Produce `boundary/` (visible al learner) y `bulk_truth/` (solo para validación).

**Funcionalidades clave:**
- Familias: `ads`, `lifshitz`, `hyperscaling`, `dpbrane`, `deformed`, `unknown`.
- Jitter de parámetros físicos para diversidad de datos.
- Correlador geodésico holográfico (`correlator_2pt_geodesic`) que sí depende de A(z).
- Auto-fix IO: si el nombre codifica `_d<k>_`, corrige `geo.d` antes de generar datos.
- Backend EMD opcional (`EMDLifshitzSolver`).

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| ALTA | `global np, gamma_func` dentro del bloque `try` en `main()` (línea ~1071) es un patrón poco idiomático. Si `import numpy` falla, `np` queda como el `numpy` importado globalmente al inicio del módulo, no como `None`, por lo que el `global` es redundante e indica un refactor incompleto. `gamma_func` se importa pero nunca se usa en el código visible. |
| MEDIA | `correlator_2pt_geodesic` hace fallback silencioso al correlador térmico si obtiene <3 puntos. Este fallback puede enmascarar fallos del integrador sin aviso. |
| BAJA | `add_standard_arguments(parser)` y `parse_stage_args(parser)` se llaman incondicionalmente, pero si `stage_utils` no está disponible estas funciones son `None`. Esto lanzará `TypeError` en runtime. El `HAS_STAGE_UTILS = False` path necesita implementar stubs. |
| BAJA | El parámetro `--output-dir` se describe como DEPRECATED pero sigue presente en el CLI sin warning al usuario. |
| INFO | La separación `boundary/` vs `bulk_truth/` está bien implementada y documentada. |

**Estado general:** ⚠️ Funcional pero con deuda técnica en manejo de imports opcionales y uso de `global`.

---

### `02R_build_ringdown_boundary_dataset.py` (v1.0)

**Propósito:** Puente entre Stage 01 (polos de ringdown) y Stage 02 (geometría emergente). Convierte polos en embeddings de frontera surrogate (G_R, G2).

**Funcionalidades clave:**
- `poles_to_gr`: respuesta tipo Green function `GR(ω) = Σ a_n/(ω - ω_n)`.
- `poles_to_g2`: observable positivo `|Σ a_n exp((-γ+iω)x)|²`.
- P-values incondicional y condicional desde null test scores.
- Protección contra escape de `PROJECT_ROOT` en resolución de rutas.
- Provenance completa en HDF5 (snapshots JSON crudos en `/ringdown_raw/`).

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| MEDIA | `PROJECT_ROOT = Path(__file__).resolve().parent` apunta a `malda/`. La verificación `resolved.relative_to(PROJECT_ROOT)` rechazará cualquier ruta fuera de `malda/`, incluyendo rutas legítimas del resto del repositorio. |
| BAJA | `np.string_(json.dumps(...))` crea bytes en formato NumPy; la decodificación downstream debe ser explícita. |
| INFO | Los embeddings surrogate están explícitamente etiquetados como "not claimed to be physical CFT correlators". Honestidad epistémica correcta. |

**Estado general:** ✅ Bien diseñado. El issue de PROJECT_ROOT es arquitectural (compartido con otros scripts).

---

### `02_emergent_geometry_engine.py` (v2.3)

**Propósito:** Motor de reconstrucción de geometría emergente. Red neuronal (PyTorch) que aprende A(z), f(z) desde datos de frontera CFT. R(z) se calcula deterministamente desde A y f.

**Funcionalidades clave:**
- Modes: `train` (con sandbox) e `inference` (sin acceso a bulk_truth).
- R(z) calculado desde geometría diferencial, no como decoder independiente (fix v2.3).
- Soporte de `checkpoint` para continuar entrenamiento.
- `CuerdasDataLoader` bloquea acceso a bulk_truth en modo inference.

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| MEDIA | El archivo tiene codificación de caracteres incorrecta en comentarios (mojibake extenso), lo que sugiere edición en un editor con configuración de encoding diferente. No afecta ejecución pero dificulta la lectura. |
| MEDIA | Script muy largo (>25K tokens). Considerablemente difícil de mantener y auditar en su totalidad sin paginación. |
| BAJA | Mismo patrón de `HAS_STAGE_UTILS` + llamadas incondicionales que en script 01. |
| INFO | La separación train/inference con bloqueo explícito de bulk_truth es una garantía de honestidad importante y bien implementada. |

**Estado general:** ⚠️ Arquitectura correcta. Mantenibilidad comprometida por tamaño y codificación.

---

### `03_discover_bulk_equations.py`

**Propósito:** Regresión simbólica (PySR) sobre geometría emergente para descubrir ecuaciones de campo en el bulk.

**Funcionalidades clave:**
- Soporta HDF5 y NPZ como entrada.
- Comparación con ecuaciones de Einstein etiquetada explícitamente como "post-hoc".
- Fallback a resolución de rutas legacy si `io_contract_resolver` no disponible.

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| MEDIA | `HAS_PYSR = False` hace que el script se ejecute en modo degradado sin aviso de error claro. PySR es una dependencia funcional core, no opcional. |
| BAJA | `io_contract_resolver` con fallback legacy: si el módulo no existe, la resolución de rutas puede ser silenciosamente incorrecta. |
| INFO | La documentación de honestidad epistémica es clara y explícita. |

**Estado general:** ✅ Correcto en diseño. Mejoras en manejo de dependencias faltantes.

---

### `04_geometry_physics_contracts.py` (v2.2)

**Propósito:** Evaluador de contratos físicos: geometría emergente vs bulk_truth (sandbox) y criterios físicos genéricos (inference).

**Funcionalidades clave:**
- Modo A (sandbox): bulk_truth disponible, métricas R².
- Modo B (inference): bulk_truth ausente, contratos genéricos.
- Gauge conformal documentado en comentarios.
- Detección de mezcla de dimensiones `d` entre sistemas incompatibles.

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| MEDIA | `cuerdas_io` importado como opcional, pero varias funciones del pipeline dependen de él para resolver rutas. El fallback no está implementado completamente. |
| BAJA | Mojibake en comentarios (igual que 02). |
| INFO | La distinción sandbox/inference es arquitecturalmente importante y está bien implementada. |

**Estado general:** ✅ Correcto. Dependencias opcionales parcialmente implementadas.

---

### `04b_negative_control_contracts.py`

**⚠️ MODO PLANTILLA (PLACEHOLDER)**

**Propósito declarado:** Contratos negativos para detectar alucinación geométrica (aceptar ruido como geometría válida).

**Estado actual:**
- `PLACEHOLDER_MODE = True`, `SCRIPT_MODE = "placeholder"` declarados explícitamente.
- Las métricas (`A_r2`, `f_r2`, etc.) son **stubs**, no conectadas a datos reales.
- Los veredictos de este script **NO deben usarse** para claims científicos.

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| ALTA | Si este script se ejecuta en producción sin que el usuario lea el warning de placeholder, puede generar falsos "PASS" o "FAIL" que no reflejan la realidad del pipeline. Debería emitir un warning explícito en stdout al inicio. |
| MEDIA | No hay mecanismo para prevenir que pipelines automatizados usen sus outputs. Considerar añadir exit code especial o marker en JSON. |

**Estado general:** 🔴 **No apto para producción.** Requiere conexión a métricas reales.

---

### `04c_negative_controls.py`

**Propósito:** Control negativo con **ruido blanco gaussiano** (más fuerte que un campo de Klein-Gordon).

**Funcionalidades clave:**
- Genera ruido blanco deliberadamente (no un campo masivo físico).
- Criterio: pass_rate < 20% → SUCCESS; > 50% → ALERT.
- Documentación de honestidad: nota que explica por qué el ruido blanco es un control más fuerte.

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| BAJA | La función `generate_massive_scalar_flat_space` genera ruido, pero su nombre sugiere campo masivo. Puede confundir. |
| BAJA | El parámetro `--mass` en CLI no tiene efecto funcional (el ruido es independiente de la masa). Debería ser eliminado o documentado como legacy. |
| INFO | La implementación de control negativo es científicamente sólida: ruido blanco como caso worst-case. |

**Estado general:** ✅ Correcto en intención. Limpieza de interfaz CLI pendiente.

---

### `04d_negative_hawking.py`

**Propósito:** Control negativo Hawking-Page: simula fase confining (T < Tc) con area law, donde no debería emerger holografía válida.

**Funcionalidades clave:**
- Correladores: `exp(-mass * r) + noise` (area law, no conforme).
- Entanglement entropy proporcional al área de boundary.
- Inspirado post-hoc en Bao, Cao & Zhu (2022), citado explícitamente.

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| BAJA | `np.random.seed(seed)` usa la API legacy de NumPy. Debería usar `np.random.default_rng(seed)` para reproducibilidad moderna. |
| INFO | La separación teoría vs implementación ("inspirado post-hoc") está bien documentada. |

**Estado general:** ✅ Correcto. Actualización de API de NumPy recomendada.

---

### `05_analyze_bulk_equations.py`

**Propósito:** Análisis post-hoc de ecuaciones de bulk descubiertas por PySR. Clasifica por régimen físico (Lifshitz z, hyperscaling θ).

**Funcionalidades clave:**
- Extrae z, θ de nombres de archivo (no de teoría inyectada).
- Nomenclatura moderna (z, θ) documentada como posterior al AGMOO 1999.
- Referencia explícita a Kachru et al. (2008) y Gouteraux & Kiritsis (2011).

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| BAJA | Extracción de parámetros desde nombres de archivo es frágil ante renombrados. |
| INFO | La distinción entre "terminología del texto de referencia" y "nomenclatura moderna" es epistemológicamente honesta. |

**Estado general:** ✅ Correcto como herramienta de análisis.

---

### `05_exp03_c3_metric_sensitivity_v3.py` (v0.3.0)

**Propósito:** Experimento 03 — demuestra que el contrato C3 es sensible a la elección de métrica y ponderación, con un control negativo (config naive → FAIL) y un control positivo (config robusta → PASS).

**Funcionalidades clave:**
- Ejecuta dos subprocesos (corridas A y B) y compara.
- Captura stdout/stderr de cada corrida en logs.
- Manifest con hashes SHA-256 de todos los artefactos.

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| MEDIA | Usa `subprocess` para lanzar `04_diccionario.py` — este script no está en `malda/`. Dependencia externa no verificada en este audit. |
| BAJA | `PROJECT_ROOT` resuelve como `malda/`; el script asume que `04_diccionario.py` es accesible desde allí. |
| INFO | El diseño experimental (negativo + positivo) es metodológicamente sólido. |

**Estado general:** ✅ Correcto en diseño. Dependencia en script externo sin validar.

---

### `06_build_bulk_eigenmodes_dataset.py`

**Propósito:** Construye dataset de modos escalares bulk (autovalores Sturm-Liouville λ_SL y dimensiones Δ extraídas de correladores).

**Funcionalidades clave:**
- Sin fórmula masa-dimensión inyectada; λ_SL son autovalores SL, no masas holográficas.
- Delta extraído de G2(x) ~ x^(-2Δ) es una medición, no teoría.
- Usa `bulk_scalar_solver_v2` con fallback a `bulk_scalar_solver`.
- Salida: CSV + JSON de metadatos.

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| MEDIA | Dependencia en `bulk_scalar_solver_v2`/`bulk_scalar_solver` no incluidos en `malda/`; si no están en el path Python, el script falla silenciosamente en el fallback. |
| BAJA | Mismo patrón de `HAS_STAGE_UTILS` con llamadas incondicionales. |
| INFO | La honestidad sobre la naturaleza de λ_SL (autovalores, no masas) está explícitamente documentada. |

**Estado general:** ✅ Correcto. Dependencias externas deben estar documentadas en requirements.

---

### `07_emergent_lambda_sl_dictionary.py`

**Propósito:** Aprende la relación emergente λ_SL ↔ Δ usando KAN y/o PySR. Compara con teoría solo si `--compare-theory` está activo.

**Funcionalidades clave:**
- Evaluación por régimen (split por familia, d, modo) para detectar mezcla engañosa de escalas.
- `pearsonr` y `r2_score` como métricas de ajuste.
- Pareto front de ecuaciones PySR exportado como CSV.

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| BAJA | PySR es opcional pero es la funcionalidad central del script. Sin él, el output es limitado. |
| BAJA | `warnings.filterwarnings` puede suprimir advertencias importantes de sklearn/scipy. |
| INFO | La evaluación por régimen es una buena práctica para detectar sobreajuste o mezcla de distribuciones. |

**Estado general:** ✅ Correcto. Diseño epistémicamente honesto.

---

### `07b_discover_lambda_delta_relation.py`

**Propósito:** Descubrimiento puro de λ_SL ↔ Δ con PySR, usando datos externos (bootstrap, lattice, CFT exacta). La fórmula teórica se usa **solo post-hoc**.

**Funcionalidades clave:**
- `load_ground_truth`: carga pares (d, λ_SL, Δ) desde JSON externo.
- `theoretical_delta`: implementa `Δ = d/2 + √(d²/4 + λ_SL)` solo para comparación.
- PySR busca la relación sin restricciones de forma funcional.

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| BAJA | `lambda_sl is None` filter es correcto, pero no hay filtro para `lambda_sl = NaN` en el JSON. Podría causar errores en PySR. |
| INFO | El script es el núcleo del claim científico del proyecto: si PySR descubre Δ = d/2 + √(d²/4 + λ_SL) sin haberla inyectado, constituye evidencia de AdS/CFT emergente. |

**Estado general:** ✅ Correcto y metodológicamente central.

---

### `08_build_holographic_dictionary.py`

**Propósito:** Construye el atlas holográfico interno (operadores por sistema/familia/dimensión) y calcula m²L² = Δ(Δ-d) como diagnóstico post-hoc.

**Funcionalidades clave:**
- Convención documentada: `d` = dimensión del boundary (CFT_d), `D = d+1` = dimensión del bulk.
- Cota BF: `m²R² ≥ -d²/4` para detectar modos taquiónicos.
- Los cálculos de m²L² son diagnósticos, no entran en entrenamiento.

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| BAJA | Mojibake en comentarios (igual que otros scripts del repo). |
| BAJA | `HAS_PYSR` declarado pero PySR no parece usarse en las primeras líneas visibles. Posible legado. |
| INFO | La documentación de notación dimensional (Sec. 3.14 del texto de referencia) es precisa y rastreable. |

**Estado general:** ✅ Correcto.

---

### `09_real_data_and_dictionary_contracts.py` (v3)

**Propósito:** Contratos finales con datos reales y diccionario emergente. Mide False Positive Rate (FPR) sobre señales holográficas en datos que no deben ser holográficos.

**Funcionalidades clave:**
- FPR = señales holográficas disparadas / señales evaluables.
- Dos tipos de validaciones explícitamente distinguidas:
  - **Tipo A** (AGMOO 1999): unitarity bound, relación masa-dimensión.
  - **Tipo B** (literatura moderna): Ising 3D (Δ_σ=0.518), KSS bound (η/s ≥ 1/4π), strange metal scaling.
- Referencias bibliográficas explícitas para cada contrato.

**Observaciones / Riesgos:**
| Severidad | Descripción |
|-----------|-------------|
| MEDIA | Los contratos tipo B dependen de valores numéricos específicos de la literatura (e.g., Δ_σ=0.518 para Ising 3D). Si estos valores cambian por nuevas mediciones, el código debe actualizarse manualmente. Hardcodear constantes físicas externas crea deuda de mantenimiento. |
| BAJA | Mismo patrón HAS_STAGE_UTILS. |
| INFO | La distinción explícita entre contratos del texto de referencia vs literatura moderna es ejemplar en términos de honestidad epistémica. |

**Estado general:** ✅ Correcto y bien documentado. Mantenimiento de constantes físicas a revisar periódicamente.

---

## Problemas transversales

### 1. `PROJECT_ROOT` inconsistente

**Afecta:** `00_load_ligo_data.py`, `01_extract_ringdown_poles.py`, `02R_build_ringdown_boundary_dataset.py`, y potencialmente otros.

Todos calculan `PROJECT_ROOT = Path(__file__).resolve().parent`, lo que resuelve a `malda/` en lugar de la raíz del repositorio. Esto limita la resolución de rutas relativas al subdirectorio, y el check de escape en `02R` rechazará rutas legítimas del resto del repo.

**Recomendación:** Unificar PROJECT_ROOT como `Path(__file__).resolve().parent.parent` (sube a la raíz del repo) o usar una variable de entorno (`CUERDAS_PROJECT_ROOT`).

### 2. Patrón `HAS_STAGE_UTILS` con llamadas incondicionales

**Afecta:** `01_generate_sandbox_geometries.py`, `02_emergent_geometry_engine.py`, `03_discover_bulk_equations.py`, `05_analyze_bulk_equations.py`, `06_build_bulk_eigenmodes_dataset.py`, `07_emergent_lambda_sl_dictionary.py`, `08_build_holographic_dictionary.py`, `09_real_data_and_dictionary_contracts.py`.

El patrón es:
```python
HAS_STAGE_UTILS = False
StageContext = None
add_standard_arguments = None
...
try:
    from stage_utils import StageContext, add_standard_arguments, ...
    HAS_STAGE_UTILS = True
except ImportError:
    pass
...
add_standard_arguments(parser)  # TypeError si HAS_STAGE_UTILS=False
```

Cuando `stage_utils` no está disponible, las llamadas a `None(...)` lanzan `TypeError`. El flag `HAS_STAGE_UTILS` no previene las llamadas.

**Recomendación:** Implementar stubs no-op cuando el módulo no está disponible, o hacer `stage_utils` una dependencia obligatoria instalada vía requirements.

### 3. Encoding de caracteres en comentarios

**Afecta:** Múltiples scripts (`01_generate_sandbox_geometries.py`, `02_emergent_geometry_engine.py`, `04_geometry_physics_contracts.py`, `08_build_holographic_dictionary.py`).

Los comentarios contienen caracteres mojibake extensos (texto latino codificado como Latin-1 leído como UTF-8 u otro esquema). Esto no afecta la ejecución pero dificulta la lectura y el mantenimiento.

**Recomendación:** Establecer `# -*- coding: utf-8 -*-` en todos los scripts y re-guardar con encoding correcto.

### 4. `04b_negative_control_contracts.py` en modo placeholder

Este script está marcado explícitamente como placeholder. Cualquier pipeline automatizado que lo incluya como paso de validación producirá resultados sin significado.

**Recomendación:** Añadir `sys.exit(99)` con mensaje claro si se llama en modo no-interactivo, hasta que esté en modo producción.

---

## Tabla resumen

| Script | Estado | Severidad máxima | Notas |
|--------|--------|-----------------|-------|
| `00_load_ligo_data.py` | ✅ OK | BAJA | PROJECT_ROOT local |
| `00_validate_io_contracts.py` | ✅ OK | MEDIA | Serialización frágil |
| `00b_physics_sanity_checks.py` | ✅ OK | MEDIA | Convención `d` a verificar |
| `01_extract_ringdown_poles.py` | ✅ OK | MEDIA | Rama principal log |
| `01_generate_sandbox_geometries.py` | ⚠️ Deuda | ALTA | `global np`, stubs ausentes |
| `02R_build_ringdown_boundary_dataset.py` | ✅ OK | MEDIA | PROJECT_ROOT |
| `02_emergent_geometry_engine.py` | ⚠️ Deuda | MEDIA | Tamaño, encoding |
| `03_discover_bulk_equations.py` | ✅ OK | MEDIA | Deps opcionales |
| `04_geometry_physics_contracts.py` | ✅ OK | MEDIA | `cuerdas_io` parcial |
| `04b_negative_control_contracts.py` | 🔴 Placeholder | ALTA | No usar en producción |
| `04c_negative_controls.py` | ✅ OK | BAJA | CLI legacy |
| `04d_negative_hawking.py` | ✅ OK | BAJA | API NumPy legacy |
| `05_analyze_bulk_equations.py` | ✅ OK | BAJA | Extracción por nombre |
| `05_exp03_c3_metric_sensitivity_v3.py` | ✅ OK | MEDIA | Dep. script externo |
| `06_build_bulk_eigenmodes_dataset.py` | ✅ OK | MEDIA | Deps externas |
| `07_emergent_lambda_sl_dictionary.py` | ✅ OK | BAJA | PySR opcional |
| `07b_discover_lambda_delta_relation.py` | ✅ OK | BAJA | NaN en JSON |
| `08_build_holographic_dictionary.py` | ✅ OK | BAJA | Encoding |
| `09_real_data_and_dictionary_contracts.py` | ✅ OK | MEDIA | Constantes físicas hardcoded |

---

## Conclusiones

El pipeline CUERDAS-Maldacena en `malda/` demuestra un nivel elevado de **honestidad epistémica**:
- Separación explícita de datos de frontera vs verdad del bulk.
- Análisis post-hoc etiquetados como tales.
- Controles negativos formales (ruido blanco, fase confining).
- Referencias bibliográficas precisas para cada relación física verificada.

Los problemas encontrados son principalmente de **calidad de código** (imports opcionales sin stubs, PROJECT_ROOT inconsistente, encoding de caracteres) y **deuda técnica** (04b en modo placeholder). Ningún script presenta vulnerabilidades de seguridad ni inyección de física en el pipeline de aprendizaje.

**Prioridades de corrección:**
1. 🔴 Conectar `04b_negative_control_contracts.py` a métricas reales o bloquearlo en pipelines automatizados.
2. ⚠️ Implementar stubs para `stage_utils` ausente en los ~8 scripts afectados.
3. ⚠️ Unificar `PROJECT_ROOT` en todos los scripts.
4. ℹ️ Corregir encoding de caracteres en comentarios.
5. ℹ️ Limpiar `global np, gamma_func` en `01_generate_sandbox_geometries.py`.
