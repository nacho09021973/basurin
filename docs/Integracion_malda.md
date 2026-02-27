# Informe de integración cruzada: `malda/` ↔ BASURIN (`mvp/`)

**Fecha:** 2026-02-20  
**Alcance:** análisis documental/técnico sin cambios de código funcional.  
**Objetivo:** evaluar cómo los scripts de `malda/` pueden ayudar a BASURIN y cómo BASURIN puede ayudar a `malda/`.

---

## 1) Resumen ejecutivo

La carpeta `malda/` y BASURIN atacan problemas vecinos (ringdown/holografía), pero con filosofías operativas complementarias:

- **BASURIN** prioriza **orquestación reproducible**, contratos de etapa, fail-fast y trazabilidad de artefactos por `run_id`.
- **`malda/`** aporta una capa más rica de **validación física post-hoc**, **controles negativos**, y exploración de **diccionario emergente** (incluyendo descubrimiento simbólico).

**Conclusión principal:**
la mayor ganancia no está en “fusionar” pipelines, sino en **interoperarlos por contrato de artefactos**:
1) BASURIN conserva su core mínimo/canónico para producción reproducible.
2) `malda/` se usa como “carril de investigación auditable” enchufado a salidas de BASURIN.

---

## 2) Qué aporta `malda/` a BASURIN

## 2.1 Capa de validación física post-hoc más profunda

`00b_physics_sanity_checks.py` deja explícito que los checks son informativos y no bloquean el pipeline (post-hoc), evitando contaminación del aprendizaje con teoría impuesta. Esta separación es valiosa para BASURIN cuando quiera ampliar sus validaciones sin romper neutralidad de stage core.  

**Valor para BASURIN:**
- Añadir un “subpipeline de auditoría física” opcional tras `s4`/`s6`.
- Mantener semántica canónica (PASS/FAIL operativo) separada de “alertas científicas”.

## 2.2 Controles negativos formalizados

`malda/04c_negative_controls.py` y `malda/04d_negative_hawking.py` representan una práctica fuerte de anti-autoengaño experimental (ruido blanco, escenarios que deben fallar o degradarse).  

**Valor para BASURIN:**
- Convertir controles negativos en pruebas de regresión científicas para etapas avanzadas (`s3b`, `s4b`, `s4c`, `s6`).
- Medir explícitamente tasas de falsos positivos en geometría compatible.

## 2.3 Validación de contratos IO externa al pipeline

`00_validate_io_contracts.py` ofrece un “auditor de artefactos en lote”, complementario a los contratos en runtime de BASURIN.  

**Valor para BASURIN:**
- Auditorías masivas post-run (histórico de `runs/`) sin re-ejecutar stages.
- Capa de compliance para migraciones de esquema de artefactos.

## 2.4 Ruta de descubrimiento de diccionario (λ_SL ↔ Δ)

El bloque `07*`–`09` de `malda/` modela una fase de descubrimiento y verificación del diccionario holográfico (incluyendo SR/PySR opcional).  

**Valor para BASURIN:**
- Explorar hipótesis físicas en entorno “experimental” sin ampliar superficie canónica.
- Alimentar nuevas métricas para `s0_oracle_mvp` o futuros oráculos especializados.

---

## 3) Qué aporta BASURIN a `malda/`

## 3.1 Gobernanza fuerte por contratos y contexto de etapa

BASURIN centraliza requisitos/salidas por stage en `mvp/contracts.py`, con inicialización/finalización/abort homogéneos y escritura de `manifest.json` + `stage_summary.json`.  

**Valor para `malda/`:**
- Reducir deriva entre scripts heterogéneos.
- Definir claramente entradas obligatorias, dependencias upstream y outputs esperados.

## 3.2 IO determinista y seguro en disco

`basurin_io.py` aporta primitives reutilizables: resolución consistente de raíz de runs, chequeo anti-symlink-ancestor, escritura JSON atómica y hashing SHA-256.  

**Valor para `malda/`:**
- Homogeneizar robustez de escritura/lectura.
- Mejorar auditabilidad y reproducibilidad de artefactos largos.

## 3.3 Orquestación fail-fast con timeline

`mvp/pipeline.py` implementa parada temprana por etapa fallida, timeout por stage y timeline persistido (`pipeline_timeline.json`).  

**Valor para `malda/`:**
- Mejor observabilidad operacional.
- Menor costo de depuración en pipelines largos con dependencias opcionales.

---

## 4) Mapa de compatibilidad práctica (sin escribir código)

## 4.1 Integración mínima de bajo riesgo

**Idea:** ejecutar BASURIN (`s1→s4`) como ruta canónica y usar `malda` como post-proceso auditable.

- Punto de acople principal: salidas de ringdown/estimación (`s2`, `s3`) transformadas al formato de entrada de `malda` (`boundary.h5`/CSV según etapa).
- Mantener directorios separados:
  - BASURIN: `runs/<run_id>/...`
  - MALDA auditoría: `runs/<run_id>/experiment/malda_bridge/...`

## 4.2 Integración bidireccional (fase 2)

- Reimportar a BASURIN solo resultados “estables” de `malda`:
  - métricas de contrato físico,
  - score de consistencia,
  - banderas de control negativo.
- Estos resultados entrarían como artefacto adicional de stage experimental (no canónico) con manifest y hash.

---

## 5) Riesgos de integración y mitigaciones

1. **Diferencias de convención física y dimensionalidad.**  
   *Mitigación:* documento de mapeo de variables (`d`, `D`, `Δ`, `λ_SL`) y tests de consistencia semántica.

2. **Dependencias opcionales en `malda` (p.ej. PySR, stage_utils).**  
   *Mitigación:* matriz de capacidades por entorno (core vs full-research) con degradación explícita.

3. **Deriva de formatos de artefactos.**  
   *Mitigación:* contrato puente versionado y validador de compatibilidad antes de cada corrida experimental.

4. **Crecimiento de superficie canónica en BASURIN.**  
   *Mitigación:* regla estricta: lo incierto/físico-exploratorio vive en `experiment/`; el core no se infla.

---

## 6) Recomendación estratégica

### Recomendación A (inmediata): “Acople por contrato”

- No fusionar repos conceptualmente.
- Definir un **contrato puente mínimo** (campos, hashes, unidades, convención de dimensión).
- Correr `malda` como carril de auditoría/investigación sobre salidas BASURIN.

### Recomendación B (mediano plazo): “Oráculos científicos incrementales”

- Promover a BASURIN solo aquellas validaciones de `malda` que demuestren:
  - estabilidad inter-run,
  - bajo costo computacional,
  - valor real para PASS/FAIL o ranking reproducible.

### Recomendación C (larga): “Matriz de madurez por script”

Clasificar scripts `malda` en tres niveles:
1. **Production-ready bridge** (adoptable ya),
2. **Research-stable** (usable con banderas/entorno),
3. **Exploratorio** (solo sandbox).

---

## 7) Plan sugerido de ejecución (sin código, orientado a gestión)

1. **Inventario de contratos y variables** (1 sesión técnica).
2. **Especificación de contrato puente v1** (entrada/salida, hashes, unidades).
3. **Prueba piloto sobre 1 evento** (GW150914) con reporte de discrepancias.
4. **Definición de KPIs de integración**:
   - reproducibilidad,
   - tiempo de ejecución,
   - tasa de falsos positivos en controles negativos,
   - estabilidad del veredicto.
5. **Go/No-Go** para adopción parcial en operación regular.

---

## 8) Dictamen final

Sí hay una sinergia real `malda` ↔ BASURIN, pero la vía correcta es **interoperabilidad disciplinada** y no mezcla directa de pipelines.

- BASURIN puede ganar profundidad científica (sanity checks, controles negativos, diccionario emergente).
- `malda` puede ganar robustez operativa (contratos unificados, IO determinista, fail-fast y trazabilidad).

En términos de riesgo/beneficio, la estrategia óptima es: **core pequeño y estable + carril experimental fuerte, conectado por contratos versionados**.

---

## Fuentes revisadas

- `README.md` (visión y reglas operativas BASURIN).
- `mvp/pipeline.py` (orquestación fail-fast, timeline, stages).
- `mvp/contracts.py` (contratos por stage y lifecycle).
- `basurin_io.py` (helpers de IO determinista/atómico/hash).
- Scripts en `malda/` (00–09 y variantes) y auditoría existente en `docs/audit/malda/audit_report.md`.
