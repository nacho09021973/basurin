# Informe técnico — C8_WORK_RELEVANCE  
**BASURIN / Gate experimental: bookkeeping vs extractable**

**Fecha:** 2026-01-29  
**Run de referencia:** `2026-01-22__e2e`  
**Stage evaluado:** `bridge_f4_1_alignment`  
**Gate:** `C8_WORK_RELEVANCE` (tools/c8_work_relevance_gate.py)

---

## 1. Motivación

En BASURIN, múltiples experimentos mostraban métricas internas muy altas
(correlaciones canónicas, scores de alineamiento, estabilidad bootstrap),
pero con **controles negativos elevados y degeneración estructural**.

Esto plantea una ambigüedad crítica:

> ¿El score observado es **operacionalmente extraíble** o es solo
> **bookkeeping interno** inflado por leakage / degeneración?

El gate **C8_WORK_RELEVANCE** se introdujo para formalizar esta distinción de
forma ejecutiva y auditable, sin romper la gobernanza del pipeline.

---

## 2. Problema inicial

Al ejecutar C8 sobre runs reales, el veredicto era frecuentemente:

```json
"verdict": "UNDERDETERMINED",
"failure_mode": "MISSING_BOOK_SCORE"
```

aunque:

- `metrics.json` existía
- los valores numéricos estaban presentes
- los controles negativos y de degeneración eran altos

La causa raíz era **técnica**, no conceptual.

---

## 3. Diagnóstico técnico

### 3.1 Métricas anidadas (dotpaths)

El schema real de `metrics.json` usa **estructuras anidadas**, por ejemplo:

- `results.canonical_corr_mean`
- `degeneracy.degeneracy_index_median`
- `structure_preservation.negative.overlap_mean`

Sin embargo, C8 v0.1.0 solo leía claves **top-level** mediante:

```python
payload.get(book_key)
```

Esto hacía imposible resolver correctamente `book_score` y controles.

---

### 3.2 Controles presentes pero mal interpretados

Incluso tras inferir correctamente valores vía fallback desde `metrics.json`,
el veredicto seguía siendo `UNDERDETERMINED` debido a una condición errónea:

- “controles ausentes” se infería por **ausencia de ficheros auxiliares**,
  no por ausencia de **valores numéricos efectivos**

Esto anulaba FAIL incluso cuando:

- `negative_control.value >= neg_max`
- `degeneracy_index >= threshold`
- `gap >> gap_max`

---

## 4. Cambios implementados

Todos los cambios fueron **mínimos**, localizados y contract-safe.

### 4.1 Soporte de dotpaths

Se añadió un helper genérico:

```python
_get_by_path(obj, "a.b[0].c")
```

Y se aplicó a:

- `book_score` (`--book-key results.canonical_corr_mean`)
- `degeneracy_index`
- fallbacks de controles (`structure_preservation.*`, `control_positive.*`)

---

### 4.2 Fallback correcto de controles

Cuando los JSON auxiliares no devolvían valores, C8 ahora usa:

- `structure_preservation.negative.overlap_mean`
- `structure_preservation.real.overlap_mean`
- `control_positive.overlap_mean`
- `degeneracy.degeneracy_index_median`

Siempre trazado desde `metrics.json`.

---

### 4.3 Corrección de la lógica de veredicto

Se redefinió **presencia de controles** como:

> “existen valores numéricos efectivos”

no como “existen ficheros auxiliares”.

Regla ejecutiva final:

- Si `neg_mean >= neg_max` ⇒ **FAIL** (leakage-like)
- La etiqueta `WEAK_EVIDENCE` **no puede anular** un FAIL cuando hay violación
  de thresholds.

---

## 5. Resultado final (run real)

```json
{
  "book_score": 0.9936,
  "negative_control.value": 0.4691,
  "degeneracy.value": 1.0,
  "total_penalty": 0.7691,
  "extractable_score": 0.2245,
  "gap": 0.7691,
  "failure_mode": "LEAKAGE_NEGATIVE",
  "verdict": "FAIL"
}
```

---

## 6. Interpretación

El caso ilustra el objetivo de C8:

- **Bookkeeping:** excelente
- **Extractable:** colapsa
- La señal no es operacionalmente extraíble
- El score alto es compatible con leakage + degeneración

---

## 7. Estado del sistema

- ✔ C8_WORK_RELEVANCE operativo
- ✔ Tests verdes
- ✔ Integrado en `main`
- ✔ Compatible con gobernanza BASURIN
