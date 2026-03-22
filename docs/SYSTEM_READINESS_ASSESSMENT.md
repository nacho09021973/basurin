# BRUNETE — System Readiness Assessment

**Fecha:** 2026-03-21
**Evaluador:** Auditoría técnica automatizada
**Alcance:** Sistema BRUNETE (fachada pública de BASURIN)
**Versión:** v1.0

---

## 1. VEREDICTO

```
VERDICT:  BRUNETE no está actualmente listo para operación canónica
          sin investigación ad hoc.

CAUSE:    Falta de autoridad única para arranque, cohorte raíz
          y manual operativo ejecutable verificado end-to-end.

IMPACT:   Alto coste cognitivo por run; riesgo de uso no auditable.

STATUS:   NOT OPERATIONALLY READY — technically sound, operationally ungoverned.
```

---

## 2. INVENTARIO DE ACTIVOS VERIFICADOS

Los siguientes componentes **existen y no son humo**:

### 2.1 Runner batch real
- **`brunete/brunete_run_batch.py`** (346 líneas)
- Ejecuta cohortes preparadas en modo 220 (single-event) o 221 (multimode)
- Consume salida normalizada de `brunete_prepare_events`
- Cada evento aislado en sandbox bajo `event_runs/brunete_<EVENT_ID>_m{220,221}/`

### 2.2 Stages separados
- **`brunete/brunete_prepare_events.py`** (167 líneas) — normalización de cohorte
- **`brunete/brunete_classify_geometries.py`** (390 líneas) — clasificación geométrica conjunta
- **`brunete/runtime.py`** (230 líneas) — contexto compartido con `BruneteContext`

### 2.3 Cohortes materializadas
- **`brunete/cohorts/events_support_multi.txt`** — 40 eventos robustos primarios
- **`brunete/cohorts/events_support_singleton.txt`** — 8 eventos frontera
- **`brunete/cohorts/events_no_common_region.txt`** — 10 eventos investigativos

### 2.4 Tests verdes recientes
- `tests/test_multimode_wiring_unittest.py` — wiring entre stages
- Tests de integración sobre `minimal_run=True`
- 130+ ficheros de test en `tests/`

### 2.5 Pipeline MVP subyacente
- **107 módulos Python** en `mvp/`
- Orquestador principal: `mvp/pipeline.py` (2165 líneas)
- Contratos formales: `mvp/contracts.py` (40KB+)
- I/O determinista: `basurin_io.py` con escritura atómica y SHA256

---

## 3. DIAGNÓSTICO DE OPERABILIDAD

### 3.1 Lo que funciona

| Aspecto | Estado | Evidencia |
|---------|--------|-----------|
| Ejecución por stages | OK | Tres scripts CLI con argparse documentado |
| Cohortes versionadas | OK | Tres ficheros estables en `brunete/cohorts/` |
| I/O determinista | OK | `basurin_io.py` con escritura atómica y manifiestos |
| Gating entre stages | OK | `require_run_valid()` exige `verdict.json == PASS` |
| Tests unitarios | OK | Suite de 130+ tests |
| Manual operativo | EXISTE | `brunete/MANUAL_OPERATIVO.md` (372 líneas) |

### 3.2 Lo que falta para operación canónica

| Defecto | Severidad | Descripción |
|---------|-----------|-------------|
| **Sin launcher único** | ALTA | No hay un solo comando que ejecute el pipeline completo de principio a fin. El operador debe encadenar 4 comandos manualmente con run-ids correctos. |
| **Cohorte raíz no declarada** | ALTA | No hay una declaración formal de cuál es la cohorte por defecto para un run canónico. Las tres cohortes existen pero ninguna tiene autoridad declarada. |
| **Manual no verificado E2E** | ALTA | `MANUAL_OPERATIVO.md` existe pero no hay evidencia de que los comandos exactos allí listados produzcan un run completo sin intervención. |
| **Dependencia de datos LOSC** | MEDIA | Requiere `--losc-root` con datos pre-descargados. No hay verificación de completitud del cache local. |
| **Sin smoke test operativo** | MEDIA | No existe un test que ejecute el ciclo completo prepare → batch220 → batch221 → classify en CI con datos sintéticos o mínimos. |
| **Parámetros implícitos** | MEDIA | `--epsilon 2500 --estimator spectral` aparecen en el manual pero no hay defaults documentados ni justificación de por qué esos valores. |

---

## 4. EVALUACIÓN DE RESULTADOS CIENTÍFICOS

### 4.1 Conocimiento producido (valor real)

1. **221 probablemente no es constreñible en GWTC-3** bajo los criterios de este pipeline — resultado negativo con valor informativo.
2. **Pipeline que evita falsos positivos** — el diseño por contratos y gating previene afirmaciones no sustentadas.
3. **Aislamiento del colapso** — identificado como problema de contratos y gobernanza, no de física ni de código.
4. **Demostración empírica** — sin cohorte raíz y sin launcher documentado no hay operación sostenible.

### 4.2 Lo que no se ha conseguido

- Detección operativa de modo 221 como señal discriminante.
- Un sistema que un operador nuevo pueda ejecutar sin conocimiento tácito.
- Cierre de producto interno.

---

## 5. ANÁLISIS DE RIESGO

### 5.1 Riesgo principal: colapso de juicios

Existe riesgo de colapsar tres juicios distintos en uno solo:

| Juicio | Verdad | Consecuencia de confundirlos |
|--------|--------|------------------------------|
| "No hemos detectado 221" | Probablemente cierto | Resultado científico válido |
| "BRUNETE no está operable hoy sin esfuerzo" | Cierto | Defecto de ingeniería corregible |
| "Todo el trabajo fue inútil" | **No se sigue** | Falacia de composición |

### 5.2 Riesgo operativo

- **Conocimiento tácito como cuello de botella:** solo el desarrollador original puede ejecutar el sistema con confianza.
- **Degradación temporal:** sin ejecución regular, los datos LOSC cacheados y las dependencias pueden desincronizarse.
- **Uso no auditable:** sin launcher canónico, cada ejecución es ad hoc y potencialmente no reproducible.

---

## 6. FORMULACIÓN RIGUROSA

> El proyecto produjo conocimiento y software útil, pero no alcanzó
> todavía un estado operable canónico.

Esto es distinto de:
- "Era imposible" — no hay evidencia de imposibilidad.
- "Un año tirado a la basura" — existe valor técnico y científico demostrable.
- "Está listo" — no lo está para uso normal.

---

## 7. DEUDA DE CRISTALIZACIÓN

La deuda principal no es de código sino de **cristalización operativa**:

```
CÓDIGO        ████████████████████░░  ~85% (stages, contracts, I/O, tests)
GOBERNANZA    ████████░░░░░░░░░░░░░░  ~35% (cohortes existen pero sin autoridad)
OPERABILIDAD  ████░░░░░░░░░░░░░░░░░░  ~20% (manual existe pero no verificado E2E)
DOCUMENTACIÓN ██████████████░░░░░░░░  ~65% (extensa pero dispersa)
```

---

## 8. ACCIÓN MÍNIMA RECOMENDADA (si se decide continuar)

No se recomienda intentar "salvar el año" ni correr el pipeline hoy.
Se recomienda una única acción: **registrar este veredicto** como artefacto
auditable del proyecto.

Si en el futuro se decide reactivar, el camino mínimo sería:

1. **Declarar cohorte raíz canónica** (probablemente `events_support_multi.txt`)
2. **Crear launcher único** que encadene los 4 stages con defaults sanos
3. **Verificar manual E2E** ejecutando los comandos exactos de `MANUAL_OPERATIVO.md`
4. **Añadir smoke test operativo** en CI con datos sintéticos mínimos

Cada uno de estos pasos es individualmente pequeño. El problema nunca fue
la magnitud del trabajo pendiente, sino la ausencia de una decisión de cierre.

---

## 9. RESUMEN EJECUTIVO

| Dimensión | Estado |
|-----------|--------|
| **Software** | Existe, tiene tests, es técnicamente sólido |
| **Ciencia** | Produjo resultados negativos informativos y metodología válida |
| **Operabilidad** | No alcanzada — requiere conocimiento tácito para cada ejecución |
| **Veredicto** | NOT READY para operación canónica; valor técnico preservado |

---

*Este documento es un artefacto auditable del estado del sistema a fecha 2026-03-21.*
