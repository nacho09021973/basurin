# Revisión de Arquitectura de Gobernanza — BASURIN/BRUNETE

**Fecha:** 2026-02-27
**Rama:** `claude/review-governance-architecture-e1jtK`
**Fuentes revisadas:** `AGENTS.md`, `README.md`, `mvp/brunete/core.py`, `mvp/brunete/__init__.py`, `basurin_io.py`, `mvp/contracts.py`, `docs/BRUNETE_INVENTORY.md`, `docs/BRUNETE_INCONSISTENCIES.md`, `docs/BRUNETE_TEXT_FIXES.md`, `docs/metodo_brunete.md`

---

## 1. Resumen ejecutivo

La arquitectura de gobernanza de BASURIN implementa un modelo **contract-first, fail-fast, IO determinista** sólido. Las reglas de gobernanza están bien definidas en `AGENTS.md`, el pipeline respeta semánticas de abort, y el módulo BRUNETE expone una API pura y estable. Se identifican tres áreas de mejora: (a) una inconsistencia textual documentada entre A.4/6.7 y A.5 en `metodo_brunete.md`; (b) ausencia de un stage canónico de PSD en `CONTRACTS`; (c) riesgo de proliferación de ficheros de test sin regla de consolidación activa.

---

## 2. Reglas de gobernanza (AGENTS.md) — estado de cumplimiento

### 2.1 IO determinista
**Regla:** toda escritura debe ocurrir bajo `runs/<run_id>/...` (o `BASURIN_RUNS_ROOT`).

**Estado: ✅ Implementado**
- `basurin_io.resolve_out_root("runs")` respeta `BASURIN_RUNS_ROOT` vía variable de entorno.
- `assert_no_symlink_ancestors` protege contra rutas simbólicas que puedan salir del árbol.
- `write_json_atomic` y `write_manifest` escriben siempre bajo `stage_dir`, que es subpath de `run_dir`.
- `ensure_stage_dirs` crea `<out_root>/<run_id>/<stage>/outputs/` de forma canónica.

**Observación:** la función `resolve_out_root` toma `root_name="runs"` como default pero el caller en `contracts.py` lo pasa explícitamente (`resolve_out_root("runs")`), lo que es consistente.

### 2.2 Gating RUN_VALID
**Regla:** nada downstream si `RUN_VALID != PASS`; usar `require_run_valid` cuando aplique.

**Estado: ✅ Implementado**
- `require_run_valid(out_root, run_id)` verifica la existencia de `RUN_VALID/verdict.json` y el valor `"PASS"`.
- `StageContract` tiene el campo `check_run_valid: bool = True` (default) para habilitar/deshabilitar el check.
- Sólo `s0_oracle_mvp` tiene `check_run_valid=False` (justificado: es el stage de inicialización).
- `pipeline.py` crea `RUN_VALID` al inicializar y lo actualiza al finalizar.

### 2.3 Abort semantics
**Regla:** si un stage falla, el run "no existe" a efectos downstream.

**Estado: ✅ Implementado**
- `contracts.abort(ctx, reason)` escribe un `stage_summary.json` con `verdict=FAIL` y llama `sys.exit(EXIT_CONTRACT_FAIL)` (código 2).
- `pipeline.py` corta en rc != 0 sin continuar.
- Semántica fail-fast documentada en README y AGENTS.md.

### 2.4 Artefactos obligatorios por stage
**Regla:** cada stage debe producir `manifest.json`, `stage_summary.json` y `outputs/*`, con hashes SHA-256.

**Estado: ✅ Implementado**
- `contracts.finalize(ctx, artifacts, results)` escribe ambos ficheros y calcula hashes vía `write_manifest` y `write_stage_summary`.
- `write_manifest` recorre todos los artefactos declarados y registra `sha256_file(path)` para cada uno.
- Los contratos en `CONTRACTS` declaran explícitamente `produced_outputs` por stage.

### 2.5 Rutas canónicas (no hardcodeadas)
**Regla:** usar `<RUNS_ROOT>/<run_id>/<stage>/...`; nunca rutas absolutas hardcodeadas.

**Estado: ✅ Implementado**
- `ensure_stage_dirs(run_id, stage_name, base_dir)` construye las rutas como subpath de `base_dir`.
- Ningún módulo de `mvp/` referencia rutas absolutas como `/home/...`.

### 2.6 Logging anti-pérdida-de-tiempo
**Regla:** imprimir `OUT_ROOT`, `STAGE_DIR`, `OUTPUTS_DIR`, `STAGE_SUMMARY`, `MANIFEST` al final de entrypoints.

**Estado: ⚠️ Parcialmente implementado**
- Los stages individuales (`s1`, `s2`, ..., `s6c`) imprimen parte de esta información.
- No existe una convención estandarizada ni un helper centralizado que garantice que todos los stages emiten exactamente estas cinco variables.
- **Recomendación:** añadir un helper `log_stage_paths(ctx)` en `contracts.py` que imprima el bloque canónico y llamarlo al final de cada entrypoint. Esto previene omisiones silenciosas en stages nuevos.

### 2.7 CLI anti-ambigüedad
**Regla:** `--seed` (int) y `--seed-dir` (path) son argumentos distintos; mensajes de error con ruta exacta + comando para regenerar.

**Estado: ✅ Implementado**
- Los argumentos están diferenciados en la CLI de `experiment_t0_sweep_full.py`.
- El oracle y los stages con inputs faltantes emiten la ruta exacta esperada + comando de regeneración.

### 2.8 Tests obligatorios
**Regla:** unit tests de determinismo/bordes, integration-lite con `BASURIN_RUNS_ROOT` temporal, golden tests.

**Estado: ✅ Implementado**
- `tests/` contiene >80 ficheros de test abarcando todas las categorías.
- Los tests de integración usan `BASURIN_RUNS_ROOT` temporal para verificar aislamiento de IO.
- Golden tests para oracle (`test_oracle_t0_v1_2_golden.py`, `test_experiment_t0_sweep_full_finalize_golden.py`).
- Tests de contratos en `test_mvp_contracts.py`.

**Riesgo documentado (README §"Diagnóstico de crecimiento en tests"):** sin una regla activa de consolidación, el número de ficheros de test puede crecer sin control. El README identifica este riesgo y propone presupuestos por tipo de cambio. Se recomienda formalizar esta regla en `AGENTS.md`.

---

## 3. Módulo BRUNETE — revisión de superficie técnica

### 3.1 API pública (`mvp/brunete/__init__.py`)

El módulo exporta 9 símbolos en `__all__`:

| Símbolo | Tipo | Contrato |
|---------|------|----------|
| `psd_log_derivatives_polyfit` | función pura | inputs validados, devuelve `(s1, kappa, meta)` |
| `estimate_s1_kappa_polyfit` | alias backward-compat | delega a `psd_log_derivatives_polyfit` con `min_points=3` |
| `sigma` | función pura | `kappa/(8Q²)` |
| `chi_psd` | función pura | `|s1²+kappa|/(24Q²)` |
| `curvature_KR` | función pura | Teorema 1, devuelve `(K, R)` |
| `K_R` | alias backward-compat | delega a `curvature_KR` |
| `J0_J1` | función pura | ramas perturbativa/cerrada con contrato sigma<0 |
| `J0` | función pura | forma cerrada A.4/6.7 únicamente |
| `J1` | función pura | forma cerrada para J1 |

**Estado: ✅ Superficie minimal y estable**
- Todas las funciones son puras (sin side effects, sin IO).
- Los aliases de backward-compatibility están claramente marcados como tales en comentarios.
- `J0` y `J1` tienen contratos explícitos (ValueError para sigma<0).
- `J0_J1` devuelve `(None, None, {"status": "not_applicable"})` para sigma<0 fuera del rango perturbativo, evitando excepciones en pipelines que necesitan continuar.

### 3.2 Implementación numérica

**`_erfcx_stable`:** implementa `erfcx(x) = e^{x²} erfc(x)` con rama asintótica para `x >= 25.0`, previniendo overflow en `math.exp(x*x)` para valores grandes de sigma.

**`_j0_closed_form_nonnegative` / `_j1_closed_form_nonnegative`:** funciones internas no exportadas, coherente con el patrón de API mínima.

**`psd_log_derivatives_polyfit`:** realiza todas las validaciones de precondición antes del cálculo (dimensiones, positividad estricta, puntos mínimos). La fórmula `kappa = 2*a2 - s1` implementa correctamente la identidad BRUNETE: dado que el ajuste es `L(Δu) = a2·Δu² + a1·Δu + a0`, `s1 = a1` y `kappa = f²·d²L/du² = 2·a2`...

**Nota técnica:** la fórmula en el docstring dice `kappa = d²L/du² - s1` pero el código implementa `kappa = 2*a2 - s1`. Con `d²L/du² = 2*a2`, ambas son equivalentes sólo si la identidad es `kappa = 2a2 - s1`. Revisando `docs/metodo_brunete.md` ec. (6.14): `kappa = f²·d²lnSn/df² = 2s2`. Con `s2 = a2` (coeficiente de Δu²), `kappa = 2a2`. El término `-s1` en el docstring parece un error tipográfico en el docstring que no afecta al código correcto. **Recomendación:** corregir el docstring para que diga `kappa = 2*a2` en lugar de `d²L/du² - s1`.

### 3.3 Consistencia con la teoría (metodo_brunete.md)

| Fórmula teórica | Implementada | Consistente |
|-----------------|-------------|-------------|
| `sigma = kappa/(8Q²)` (§6.5) | `sigma(Q, kappa)` | ✅ |
| `chi_PSD = |s1²+kappa|/(24Q²)` (7.14) | `chi_psd(Q, s1, kappa)` | ✅ |
| `K = -3/rho0²·(1-(s1²+kappa)/(24Q²))`, `R=2K` (7.9-7.10) | `curvature_KR` | ✅ |
| `J0(sigma)` forma cerrada A.4/6.7 | `_j0_closed_form_nonnegative` | ✅ |
| `J0(sigma) ≈ π/(2sigma)` asintótico A.5 | **NO implementada** | N/A — código usa A.4 como SSOT |

**Inconsistencia textual documentada (A.5):** el asintótico correcto derivado de A.4 es `~3√π/(8σ^{3/2})`, no `π/(2σ)`. Esta discrepancia está documentada en `docs/BRUNETE_INCONSISTENCIES.md` y `docs/BRUNETE_TEXT_FIXES.md`. El código es correcto; el texto de `metodo_brunete.md` necesita corrección en A.5. Esta es la única inconsistencia teoría↔implementación identificada.

---

## 4. Sistema de contratos (`mvp/contracts.py`) — revisión

### 4.1 `StageContract` (frozen dataclass)

```
name: str
required_inputs: list[str]   # relativo a run_dir
produced_outputs: list[str]  # relativo a stage_dir
upstream_stages: list[str]   # stages con PASS previo requerido
check_run_valid: bool = True
```

**Estado: ✅ Diseño sólido**
- Inmutable (`frozen=True`): los contratos no pueden mutarse en runtime.
- Separación clara inputs (relativos a `run_dir`) vs outputs (relativos a `stage_dir`).
- `upstream_stages` permite declarar dependencias transitivas sin hardcodearlas en cada stage.

### 4.2 Registro `CONTRACTS`

17 stages registrados (ver `BRUNETE_INVENTORY.md`). Incluye canónicos (`s1`–`s6c`), experimentales (`s3b`, `s4b`, `s4c`, `s4d`, `s3_spectral_estimates`, `experiment_geometry_evidence_vs_gr`) y stages base (`s0`).

**Observación:** el README (§"PRs auditables") menciona la posibilidad de cambio futuro de `CONTRACTS` → `STAGE_CONTRACTS`. Si esto ocurre, el test `tests/test_mvp_contracts.py` y todos los snippets de documentación deben actualizarse en el mismo PR (punto único de mantenimiento).

### 4.3 Stage `s6c_brunete_psd_curvature`

El contrato declara:
- `required_inputs`: `s3_ringdown_estimates/outputs/estimates.json` + dos variantes de PSD.
- `produced_outputs`: `outputs/brunete_metrics.json`, `outputs/psd_derivatives.json`.
- `upstream_stages`: `["s3_ringdown_estimates"]`.

Esto es consistente con el documento operativo `docs/BRUNETE_S6C.md`.

### 4.4 Ausencia de stage canónico de PSD

**Hallazgo:** `mvp/extract_psd.py` existe como helper no-contract. No hay ningún stage en `CONTRACTS` con `produced_outputs` que contenga artefactos PSD. El `s6c` consume PSD como input preferido desde `runs/<run_id>/psd/measured_psd.json`, pero ningún stage canónico produce ese artefacto con `manifest.json` + `stage_summary.json`.

**Impacto:** la cadena de trazabilidad está incompleta para la PSD. Si `extract_psd.py` falla silenciosamente, `s6c` cae al fallback `external_inputs/psd_model.json` sin auditoría del origen.

**Recomendación:** formalizar `extract_psd.py` como stage `s2b_extract_psd` (o equivalente) con contrato completo, o documentar explícitamente en `AGENTS.md` que la PSD es un "input externo" que no requiere stage canónico (con los riesgos asociados).

---

## 5. IO layer (`basurin_io.py`) — revisión

### 5.1 Atomicidad de escrituras JSON

`write_json_atomic` usa `tempfile.mkstemp` + `os.replace` en el mismo directorio. Esto garantiza que los readers nunca ven un fichero parcial (operación atómica en POSIX). El cleanup en caso de excepción está correctamente manejado con `try/except BaseException`.

### 5.2 Protección anti-symlink

`assert_no_symlink_ancestors` recorre el árbol desde el path hasta el boundary (o la raíz del FS). Esto previene ataques de symlink que redirijan escrituras fuera del árbol de `runs/`. La función está invocada en `resolve_out_root` y en `write_json_atomic`.

**Observación menor:** `assert_no_symlink_ancestors` usa `os.path.abspath` (no `Path.resolve()`) lo que significa que symlinks en medio del path podrían no detectarse antes de llegar al componente que es symlink. Sin embargo, dado que itera componente a componente con `is_symlink()`, esto es correcto.

### 5.3 Validación de `run_id`

Regex `^[A-Za-z0-9._-]+$` con límite de 128 caracteres. Esto previene path traversal (`../`) y caracteres de control. Adecuado para el uso como segmento de path.

### 5.4 Hashing

`sha256_file` lee en chunks de 64 KiB, correcto para archivos grandes (HDF5). El hash se registra en `manifest.json` bajo la clave `"hashes"`.

---

## 6. Hallazgos consolidados

### Conformidades (✅)

1. IO estrictamente contenido bajo `BASURIN_RUNS_ROOT` / `runs/<run_id>/`.
2. `require_run_valid` integrado en el patrón de contrato con opción de bypass explícito.
3. Abort semantics fail-fast implementadas y documentadas.
4. Artefactos obligatorios (`manifest.json`, `stage_summary.json`, `outputs/*`) con SHA-256.
5. API BRUNETE minimal, pura, backward-compatible y consistente con la teoría (A.4/6.7).
6. Suite de tests abarcando unit, integration, contract y golden.
7. Separación canónico (`mvp/`) vs experimental (`runs/.../experiment/`) respetada.

### No conformidades / áreas de mejora (⚠️)

| # | Hallazgo | Severidad | Ubicación | Acción recomendada |
|---|----------|-----------|-----------|-------------------|
| 1 | Asintótico A.5 inconsistente con A.4 | Baja (solo texto) | `docs/metodo_brunete.md` A.5 | Corregir A.5 para reflejar `~3√π/(8σ^{3/2})` o añadir nota de reconciliación |
| 2 | Docstring erróneo en `psd_log_derivatives_polyfit` | Baja | `mvp/brunete/core.py:23` | Cambiar `kappa=d²L/du² - s1` → `kappa=2*a2` en el docstring |
| 3 | No existe stage canónico de PSD | Media | `mvp/contracts.py`, `mvp/extract_psd.py` | Crear `s2b_extract_psd` con contrato, o documentar explícitamente la excepción en `AGENTS.md` |
| 4 | Logging de paths no estandarizado | Baja | múltiples entrypoints | Añadir helper `log_stage_paths(ctx)` en `contracts.py` |
| 5 | Regla de consolidación de tests no en `AGENTS.md` | Baja | `AGENTS.md` | Mover la guía del README §"Diagnóstico de crecimiento en tests" a `AGENTS.md` |

---

## 7. Conclusión

La arquitectura de gobernanza de BASURIN es **sólida y bien implementada**. Los principios de IO determinista, gating por `RUN_VALID`, y contratos explícitos por stage están correctamente aplicados en el código. El módulo BRUNETE cumple el contrato matemático (Teorema 1, J0/J1, sigma/chi_psd) con fidelidad a la teoría en `metodo_brunete.md`.

Los hallazgos son menores: una inconsistencia textual (no en código) en el asintótico A.5, un docstring incorrecto que no afecta al comportamiento, y la ausencia de un stage formal de PSD que deja un hueco en la cadena de trazabilidad de `s6c`. Ninguno de estos hallazgos compromete la reproducibilidad o auditoría del pipeline actual.
