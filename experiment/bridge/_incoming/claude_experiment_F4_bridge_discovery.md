# Experimento F4-1: Bridge Discovery + Validation

**Versión**: 1.0.0-draft  
**Fecha**: 2026-01-20  
**Fase BASURIN**: 4 — Principios universales de identificabilidad  
**Status**: Propuesta para revisión

---

## 1. Problema Científico

Dados dos espacios de features:
- **Espacio A (atlas holográfico)**: ratios `r_n = M²_n/M²_0` derivados del solver Sturm-Liouville en geometrías AdS/bulk
- **Espacio B (ringdown externo)**: features de modos quasi-normales (QNM) de agujeros negros: `(f_220, Q_220, f_221/f_220, Q_221/Q_220)`

**Pregunta**: ¿Existe un "puente" estructural entre A y B que:
1. No sea trivial (mejor que aleatorio)
2. Sea estable bajo perturbaciones
3. Preserve estructura local (vecindarios)
4. No sea completamente degenerado (múltiples puntos de A no colapsan al mismo punto de B)

**Restricción epistémica crítica**: No se impone ningún feature-map físico QNM↔ratios como hipótesis oculta. El puente se descubre de forma agnóstica mediante correlación estructural, no teoría holográfica.

---

## 2. Hipótesis Formales

### H0 (Nula)
No existe puente estructural no-trivial entre atlas y ringdown. Cualquier correlación aparente es:
- Espuria (correlación con ruido)
- Completamente degenerada (colapso a subespacio trivial)
- Equivalente a control negativo (permutación aleatoria)

### H1 (Alternativa)
Existe un puente que:
- Preserva vecindarios locales significativamente mejor que control negativo
- No colapsa más del 50% de puntos a regiones degeneradas
- Es estable bajo bootstrap (CV < 0.3)

---

## 3. Datasets

### 3.1 Atlas Interno (Espacio A)
**Fuente**: `runs/<run_atlas>/dictionary/outputs/atlas.json`  
**Schema**:
```json
{
  "theories": [
    {"id": 0, "delta": 1.55, "ratios": [r_1, r_2, ..., r_k], ...},
    ...
  ]
}
```
**Features utilizados**: `ratios` (vector de dimensión k)  
**Etiqueta auxiliar**: `delta` (para diagnósticos, NO se usa en el puente)

### 3.2 Ringdown Externo (Espacio B)
**Fuente**: `runs/<run_ringdown>/ringdown/outputs/features.json`  
**Schema propuesto**:
```json
{
  "n_events": N,
  "feature_definition": {
    "f0": "f_220 [Hz]",
    "Q0": "Q_220",
    "f_ratio": "f_221/f_220",
    "Q_ratio": "Q_221/Q_220"
  },
  "events": [
    {"id": "GW150914", "features": [f0, Q0, f_ratio, Q_ratio], "meta": {...}},
    ...
  ]
}
```

**Nota**: Si el ringdown actual solo produce summary (no features por evento), se requiere:
1. Opción A: Usar features sintéticos de un sweep paramétrico (masa, spin)
2. Opción B: Modificar `stage_ringdown_features.py` para producir este formato

Para F4-1, **recomiendo Opción A** (sintético controlado) como primer paso, permitiendo validar la maquinaria antes de usar datos reales.

### 3.3 Controles

**Control Positivo (C+)**: Split del atlas en dos mitades disjuntas A₁ y A₂.
- Si el puente funciona, A₁↔A₂ debe dar PASS (estructura idéntica).

**Control Negativo (C-)**: Permutación aleatoria de features en B.
- Si el puente detecta estructura real, B_permuted debe dar FAIL.

**Control Negativo Fuerte (C--)**: Ruido gaussiano con misma estadística marginal.
- Elimina cualquier correlación estructural residual.

---

## 4. Metodología

### 4.1 Preprocesamiento
```
normalizar(X) := (X - mean(X)) / std(X)   # Por columna
```
Ambos datasets se normalizan antes de cualquier cálculo.

### 4.2 Métrica de Estructura: Preservación de Vecindarios (kNN)

**Definición**: Para un punto `i` en espacio A con k vecinos más cercanos `N_k(i)`, definimos:
```
overlap_i := |N_k(i) ∩ N_k(π(i))| / k
```
donde `π(i)` es el punto correspondiente en B tras aplicar el puente.

**Agregación**:
```
kNN_preservation := mean_i(overlap_i)
```

**Rango**: [0, 1]. 
- 1 = vecindarios perfectamente preservados
- 1/N ≈ 0 = equivalente a aleatorio

### 4.3 Métrica de Degeneración: Índice de Colapso

**Definición**: Fracción de puntos cuya imagen bajo el puente cae a distancia < ε de otro punto.
```
degeneracy_index := |{i : ∃j≠i, ||π(i) - π(j)|| < ε}| / N
```

**Umbral**: ε = 0.1 × diámetro del espacio imagen.

**Rango**: [0, 1].
- 0 = no degenerado (inyectivo localmente)
- 1 = completamente colapsado

### 4.4 Estabilidad: Bootstrap CV

**Procedimiento**:
1. Generar B=100 muestras bootstrap de (A, B) emparejados
2. Para cada muestra, calcular el puente y kNN_preservation
3. Reportar CV = std(kNN_preservation) / mean(kNN_preservation)

**Umbral**: CV < 0.3 indica estabilidad aceptable.

### 4.5 Método de Puente: Canonical Correlation Analysis (CCA) Agnóstica

**Elección**: CCA proyecta ambos espacios a un subespacio común maximizando correlación canónica, SIN imponer estructura física.

**Implementación**:
```python
from sklearn.cross_decomposition import CCA
cca = CCA(n_components=min(dim_A, dim_B, max_components))
A_proj, B_proj = cca.fit_transform(A_normalized, B_normalized)
```

**Alternativas para ablación**:
- PCA compartido (proyección no supervisada)
- Kernel CCA (no lineal)
- Procrustes (alineamiento rígido)

---

## 5. Contratos de Fase 4

### C7a — Estructura (kNN Preservation)
```
PASS: kNN_preservation > 0.3 AND kNN_preservation > 2 × kNN_control_negativo
FAIL: otherwise
```

### C7b — No-Degeneración
```
PASS: degeneracy_index < 0.5
FAIL: degeneracy_index >= 0.5
DIAGNÓSTICO: Si FAIL, reportar "DEGENERATE_BRIDGE"
```

### C7c — Estabilidad
```
PASS: bootstrap_CV < 0.3
FAIL: bootstrap_CV >= 0.3
DIAGNÓSTICO: Si FAIL, reportar "UNSTABLE_BRIDGE"
```

### C7d — No-Falso-Positivo
```
PASS: control_negativo C7a = FAIL
FAIL: control_negativo C7a = PASS
DIAGNÓSTICO: Si FAIL, reportar "FALSE_POSITIVE_LEAKAGE"
```

### C7e — Control Positivo
```
PASS: control_positivo (A₁↔A₂) C7a = PASS
WARN: control_positivo C7a = FAIL
DIAGNÓSTICO: Si WARN, metodología puede tener bugs
```

### Resultado Global F4-1
```
PASS: C7a ∧ C7b ∧ C7c ∧ C7d ∧ C7e
FAIL_DEGENERACY: ¬C7b (hay puente pero es degenerado)
FAIL_STRUCTURE: ¬C7a ∧ C7d (no hay puente detectado, control negativo correcto)
FAIL_UNSTABLE: ¬C7c (puente inestable)
FAIL_LEAKAGE: ¬C7d (falso positivo, revisar metodología)
```

---

## 6. Run Plan

### Paso 0: Setup
```bash
# Crear estructura de run
mkdir -p runs/f4_bridge_pilot/bridge/outputs
mkdir -p runs/f4_bridge_pilot/bridge/controls
```

### Paso 1: Generar Atlas (si no existe)
```bash
# Opción: usar run existente con atlas
# cp runs/<existing_run>/dictionary/outputs/atlas.json runs/f4_bridge_pilot/atlas_input.json

# O generar nuevo:
python 01_genera_ads_puro.py --run f4_bridge_pilot
python 03_sturm_liouville.py --run f4_bridge_pilot --n-delta 50 --n-modes 10
python 04_diccionario.py --run f4_bridge_pilot --k-features 4 --enable-c3
```

### Paso 2: Generar Features Ringdown Sintético
```bash
python experiment/bridge/stage_ringdown_synthetic.py \
  --run f4_bridge_pilot \
  --n-points 50 \
  --mass-range 30 100 \
  --spin-range 0.1 0.9 \
  --seed 42
```

### Paso 3: Ejecutar Bridge Discovery
```bash
python experiment/bridge/stage_bridge_discovery.py \
  --run f4_bridge_pilot \
  --atlas runs/f4_bridge_pilot/dictionary/outputs/atlas.json \
  --external runs/f4_bridge_pilot/ringdown_synthetic/outputs/features.json \
  --method cca \
  --k-neighbors 5 \
  --n-bootstrap 100 \
  --seed 42
```

### Paso 4: Verificar Outputs
```bash
cat runs/f4_bridge_pilot/bridge/outputs/bridge_results.json
cat runs/f4_bridge_pilot/bridge/stage_summary.json
```

---

## 7. Estructura de Outputs

```
runs/f4_bridge_pilot/bridge/
├── manifest.json
├── stage_summary.json
└── outputs/
    ├── bridge_results.json      # Métricas principales y veredicto
    ├── degeneracy_analysis.json # Detalle de degeneración por punto
    ├── stability_analysis.json  # Bootstrap samples y estadísticas
    ├── projections.npz          # Proyecciones CCA (A_proj, B_proj)
    └── controls/
        ├── positive_control.json  # A₁↔A₂
        ├── negative_control.json  # B permutado
        └── negative_strong.json   # B ruido
```

### bridge_results.json (schema)
```json
{
  "version": "1.0.0",
  "method": "cca",
  "contracts": {
    "C7a_structure": {"status": "PASS|FAIL", "kNN_preservation": 0.45, "threshold": 0.3},
    "C7b_degeneracy": {"status": "PASS|FAIL", "degeneracy_index": 0.12, "threshold": 0.5},
    "C7c_stability": {"status": "PASS|FAIL", "bootstrap_cv": 0.18, "threshold": 0.3},
    "C7d_no_false_positive": {"status": "PASS|FAIL", "control_neg_knn": 0.08},
    "C7e_positive_control": {"status": "PASS|WARN|SKIP", "control_pos_knn": 0.72}
  },
  "global_status": "PASS|FAIL_DEGENERACY|FAIL_STRUCTURE|FAIL_UNSTABLE|FAIL_LEAKAGE",
  "failure_mode": null | "DEGENERATE_BRIDGE" | "NO_BRIDGE" | "UNSTABLE_BRIDGE" | "FALSE_POSITIVE",
  "diagnostics": {
    "canonical_correlations": [0.85, 0.42, 0.11],
    "explained_variance_A": 0.78,
    "explained_variance_B": 0.65,
    "effective_dimensionality": 2
  }
}
```

---

## 8. Cambios de Código Necesarios

### 8.1 Scripts Nuevos

| Script | Función | Prioridad |
|--------|---------|-----------|
| `experiment/bridge/stage_bridge_discovery.py` | Motor principal F4 | CRÍTICO |
| `experiment/bridge/bridge_metrics.py` | Funciones kNN, degeneracy, CCA | CRÍTICO |
| `experiment/bridge/stage_ringdown_synthetic.py` | Generador de features QNM sintéticos | CRÍTICO |
| `experiment/bridge/controls.py` | Generadores de controles C+/C-/C-- | MEDIO |

### 8.2 Modificaciones a Código Existente

| Archivo | Cambio | Prioridad |
|---------|--------|-----------|
| `04_diccionario.py` | Ninguno (ya produce atlas.json) | — |
| `experiment/ringdown/export_atlas_points.py` | Verificar compatibilidad de schema | BAJO |

### 8.3 Dependencias
```
numpy
scipy
scikit-learn  # Para CCA, kNN
h5py
```

---

## 9. Justificación Epistémica

### Si F4-1 da PASS:
- **Evidencia**: Existe correlación estructural no trivial entre atlas holográfico y features de ringdown
- **NO significa**: Que haya relación física/holográfica (correlación ≠ causalidad)
- **Siguiente paso**: F4-2 — Investigar si la correlación tiene interpretación física o es artefacto de escala/normalización

### Si F4-1 da FAIL_DEGENERACY:
- **Evidencia**: Hay un puente, pero es degenerado (múltiples puntos de A mapean al mismo de B)
- **Interpretación**: El espacio de ringdown tiene menos información que el atlas (posible no-identificabilidad)
- **Siguiente paso**: Analizar qué direcciones del atlas colapsan → identificar "modos ciegos"

### Si F4-1 da FAIL_STRUCTURE:
- **Evidencia**: No se detecta puente estructural (mejor que aleatorio)
- **Interpretación**: Los espacios son genuinamente incompatibles (resultado válido y esperado si no hay relación física)
- **Valor científico**: Confirma que ringdown QNM y espectros holográficos AdS son fenómenos independientes (hasta el límite de esta metodología)

### Si F4-1 da FAIL_LEAKAGE:
- **Evidencia**: El control negativo también "pasa" → metodología defectuosa
- **Acción**: Debuggear pipeline, NO interpretar resultados previos

---

## 10. Kill-Switches y Salvaguardas

### Kill-Switch 1: Leakage Detection
Si `kNN_control_negativo > 0.25` → ABORT, revisar normalización/preprocesamiento.

### Kill-Switch 2: Dimensionality Collapse
Si todas las correlaciones canónicas < 0.1 → ABORT con "UNINFORMATIVE_PROJECTION".

### Kill-Switch 3: Sample Size Warning
Si N < 30 en cualquier dataset → WARN con "LOW_SAMPLE_SIZE".

### Invarianza a Reparametrizaciones
- El método CCA es invariante a transformaciones lineales invertibles de cada espacio por separado
- Para verificar: aplicar rotación aleatoria a A antes del puente, verificar que métricas no cambian (ablation test)

---

## 11. Cronograma Sugerido

| Día | Tarea |
|-----|-------|
| D+0 | Implementar `bridge_metrics.py` (kNN, degeneracy) |
| D+1 | Implementar `stage_ringdown_synthetic.py` |
| D+2 | Implementar `stage_bridge_discovery.py` |
| D+3 | Ejecutar piloto con datos sintéticos |
| D+4 | Análisis de resultados, ajustes |
| D+5 | Documentación y commit |

---

## Apéndice A: Relación con Fase 3 (C6/F3-closure)

F4-1 extiende el diagnóstico `INCOMPATIBLE_FEATURE_SPACE` de Fase 3:
- C6 detectaba incompatibilidad **binaria** (compatible/incompatible)
- C7 provee diagnóstico **gradual** con modos de fallo específicos

La transición es:
```
C6 = INCOMPATIBLE  →  F4-1 determina si es:
                       ├── FAIL_DEGENERACY (puente existe pero degenerado)
                       ├── FAIL_STRUCTURE (no hay puente)
                       └── FAIL_UNSTABLE (puente inestable)
```

---

## Apéndice B: Opciones de Extensión (Post-F4-1)

1. **Métodos no lineales**: Kernel CCA, autoencoders compartidos
2. **Validación con datos reales**: Usar features de ringdown de GW150914, GW170817, etc.
3. **Interpretación física**: Si PASS, analizar qué combinaciones de ratios correlacionan con qué features de QNM
4. **Transfer learning**: Usar el puente como prior para reconstrucción holográfica informada
