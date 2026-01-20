# F4-1: Bridge Discovery — Resumen Ejecutivo

**Versión**: 1.0.0  
**Autor**: Claude + Nacho  
**Fecha**: 2026-01-20

---

## 1. Objetivo de Fase 4

**Pregunta central**: ¿Existe un puente estructural entre el atlas holográfico (ratios del diccionario) y features de datos externos (ringdown QNM), sin imponer teoría física a priori?

**Distinción clave (lo que pidió el usuario)**:
- **(i) Hay puente pero es degenerado/no inyectivo** → FAIL_DEGENERACY
- **(ii) No hay puente** → FAIL_STRUCTURE

---

## 2. Arquitectura F4-1

```
┌─────────────────────────────────────────────────────────────────┐
│                       FASE 4 - F4-1                             │
│                    Bridge Discovery                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌──────────────────┐                 │
│  │  atlas.json  │         │  features.json   │                 │
│  │  (ratios)    │         │  (QNM sintético) │                 │
│  └──────┬───────┘         └────────┬─────────┘                 │
│         │                          │                            │
│         ▼                          ▼                            │
│  ┌─────────────────────────────────────────────┐               │
│  │         stage_bridge_discovery.py           │               │
│  │  ┌─────────────────────────────────────┐   │               │
│  │  │          CCA (agnóstico)            │   │               │
│  │  └─────────────────────────────────────┘   │               │
│  │                    │                        │               │
│  │         ┌──────────┴──────────┐            │               │
│  │         ▼                     ▼            │               │
│  │   ┌──────────┐          ┌──────────┐       │               │
│  │   │ A_proj   │          │ B_proj   │       │               │
│  │   └────┬─────┘          └────┬─────┘       │               │
│  │        │                     │             │               │
│  │        └─────────┬───────────┘             │               │
│  │                  ▼                         │               │
│  │    ┌────────────────────────────┐          │               │
│  │    │   Evaluación C7a-e         │          │               │
│  │    │   - kNN preservation       │          │               │
│  │    │   - Degeneracy index       │          │               │
│  │    │   - Bootstrap stability    │          │               │
│  │    │   - Controls +/-           │          │               │
│  │    └────────────────────────────┘          │               │
│  └─────────────────────────────────────────────┘               │
│                         │                                       │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────┐               │
│  │              bridge_results.json            │               │
│  │  status: PASS | FAIL_DEGENERACY |           │               │
│  │          FAIL_STRUCTURE | FAIL_UNSTABLE |   │               │
│  │          FAIL_LEAKAGE                       │               │
│  └─────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Contratos C7 (nuevos para F4)

| Contrato | Métrica | Umbral | Descripción |
|----------|---------|--------|-------------|
| **C7a** | kNN preservation | > 0.3 AND > 2×control_neg | Preservación de vecindarios |
| **C7b** | Degeneracy index | < 0.5 | Fracción de puntos colapsados |
| **C7c** | Bootstrap CV | < 0.3 | Estabilidad del puente |
| **C7d** | Control negativo | kNN < 0.3 | No falsos positivos |
| **C7e** | Control positivo | kNN > 0.3 | Split del atlas funciona |

---

## 4. Run Plan Ejecutable

### Paso 0: Copiar scripts al repositorio
```bash
# Desde la sesión actual, los archivos están en:
# /home/claude/experiment/bridge/
#   - bridge_metrics.py
#   - stage_ringdown_synthetic.py  
#   - stage_bridge_discovery.py

# Copiar a tu repo:
mkdir -p experiment/bridge/
cp /home/claude/experiment/bridge/*.py experiment/bridge/
```

### Paso 1: Generar un run de atlas (si no existe)
```bash
python 01_genera_ads_puro.py --run f4_pilot
python 03_sturm_liouville.py --run f4_pilot --n-delta 50 --n-modes 8
python 04_diccionario.py --run f4_pilot --k-features 4 --enable-c3
```

### Paso 2: Generar features QNM sintéticos
```bash
python experiment/bridge/stage_ringdown_synthetic.py \
    --run f4_pilot \
    --n-points 50 \
    --mass-range 30 100 \
    --spin-range 0.1 0.9 \
    --seed 42
```

### Paso 3: Ejecutar Bridge Discovery
```bash
python experiment/bridge/stage_bridge_discovery.py \
    --run f4_pilot \
    --atlas runs/f4_pilot/dictionary/outputs/atlas.json \
    --external runs/f4_pilot/ringdown_synthetic/outputs/features.json \
    --k-neighbors 5 \
    --n-bootstrap 100 \
    --seed 42
```

### Paso 4: Verificar resultados
```bash
cat runs/f4_pilot/bridge/outputs/bridge_results.json | jq '.status, .failure_mode, .contracts'
```

---

## 5. Estructura de Outputs

```
runs/f4_pilot/
├── geometry/ads_puro.h5
├── spectrum/outputs/spectrum.h5
├── dictionary/outputs/
│   ├── atlas.json           ← INPUT a F4
│   └── validation.json
├── ringdown_synthetic/outputs/
│   └── features.json        ← INPUT a F4
└── bridge/                  ← OUTPUTS de F4
    ├── manifest.json
    ├── stage_summary.json
    └── outputs/
        ├── bridge_results.json
        ├── degeneracy_analysis.json
        ├── stability_analysis.json
        ├── projections.npz
        └── controls/
            ├── positive_control.json
            └── negative_control.json
```

---

## 6. Lista de Cambios de Código

### Archivos Nuevos (CRÍTICO)
| Archivo | Líneas | Función |
|---------|--------|---------|
| `experiment/bridge/bridge_metrics.py` | ~470 | Métricas kNN, degeneracy, CCA, bootstrap |
| `experiment/bridge/stage_ringdown_synthetic.py` | ~200 | Generador de QNM sintéticos |
| `experiment/bridge/stage_bridge_discovery.py` | ~280 | Motor principal F4-1 |

### Archivos Existentes (NO modificar)
- `04_diccionario.py`: Ya produce `atlas.json` correctamente
- `03_sturm_liouville.py`: No requiere cambios

### Dependencias
```
numpy
scipy
scikit-learn  # Para CCA
h5py
```

---

## 7. Interpretación de Resultados

### PASS
- **Evidencia**: Existe correlación estructural no trivial
- **NO significa**: Relación física holográfica
- **Siguiente**: F4-2 — Investigar interpretación física

### FAIL_DEGENERACY (nuevo en F4)
- **Evidencia**: Hay puente pero es degenerado (colapso >50%)
- **Interpretación**: Espacio externo tiene menos información
- **Siguiente**: Identificar "modos ciegos" del atlas

### FAIL_STRUCTURE
- **Evidencia**: No hay puente estructural
- **Interpretación**: Espacios genuinamente incompatibles
- **Valor**: Confirma independencia (resultado válido)

### FAIL_UNSTABLE
- **Evidencia**: Puente inestable bajo bootstrap
- **Acción**: Aumentar datos o simplificar método

### FAIL_LEAKAGE
- **Evidencia**: Control negativo pasa (bug metodológico)
- **Acción**: Debuggear pipeline, NO interpretar

---

## 8. Salvaguardas y Kill-Switches

| Condición | Acción |
|-----------|--------|
| `kNN_control_neg > 0.25` | ABORT + revisar normalización |
| Todas correlaciones canónicas < 0.1 | ABORT con UNINFORMATIVE_PROJECTION |
| N < 30 en cualquier dataset | WARN con LOW_SAMPLE_SIZE |

---

## 9. ¿Por qué esto es Fase 4?

Conexión con roadmap BASURIN:

| Fase | Objetivo | F4-1 contribuye |
|------|----------|-----------------|
| F3 | Contrastación externa | C6 detecta INCOMPATIBLE (binario) |
| **F4** | **Principios universales** | **C7 detecta DEGENERACY vs NO_BRIDGE (gradual)** |
| F5 | Diccionario mínimo | Identificar modos ciegos |
| F6 | Emergencia de geometría | Si PASS, investigar interpretación |

F4-1 operacionaliza la distinción crítica:
- "No encontré puente" ≠ "No existe puente"
- "Puente degenerado" → información parcial (no nula)

---

## 10. Validación del Diseño (Self-Test)

Ejecutado exitosamente:
```
Test 1 (estructura compartida): FAIL_DEGENERACY
  kNN preservation: 0.383
  Degeneracy: 0.833

Test 2 (independiente): FAIL_STRUCTURE  
  kNN preservation: 0.087

Test 3 (control negativo funciona):
  Control neg kNN (test1): 0.083
  Control neg kNN (test2): 0.113
```

**Conclusión**: El sistema distingue correctamente entre los modos de fallo.

---

## Aprobación

- [ ] Revisar documento `experiment_F4_bridge_discovery.md`
- [ ] Copiar scripts a repositorio
- [ ] Ejecutar piloto con datos sintéticos
- [ ] Commit a main

¿Preguntas o ajustes antes de ejecutar?
