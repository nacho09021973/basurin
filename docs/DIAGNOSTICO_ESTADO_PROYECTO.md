# BASURIN -- Diagnostico de estado del proyecto

**Fecha:** 2026-02-06
**Proposito:** Evaluar honestamente donde esta el proyecto, que funciona, que falta,
y por que 187 commits, 173 ficheros Python y 81 tests no han producido
un pipeline ejecutable de principio a fin.

---

## 1. Que es BASURIN (en una frase)

**BASURIN es un pipeline de investigacion que deberia inferir propiedades geometricas
del bulk (holografia) a partir de observables espectrales, usando ondas gravitacionales
(ringdown) como banco de pruebas fisico.**

La tesis central (del roadmap, Fase 6):

> "La geometria emerge como resultado inferencial, no como postulado."

---

## 2. El estado real: dos pipelines desconectados

El proyecto contiene **dos pipelines que apenas se hablan**:

### Pipeline A: Holografico (el corazon teorico)

```
01_genera_ads_puro.py --> 03_sturm_liouville.py --> 04_diccionario.py
    geometry.h5              spectrum.h5              dictionary.h5 + atlas.json
```

- **Tiene orquestador:** `tools/run_v1.py` (ejecuta 01->03->04->RUN_VALID)
- **Proposito:** Dado un bulk (geometria), calcular espectro, construir diccionario
  inverso, y verificar si la geometria es reconstruible
- **Estado:** Funcional pero inerte. Nadie lo consume downstream de forma sistematica.

### Pipeline B: Ringdown (donde se ha invertido el 80% del esfuerzo)

```
RUN_VALID --> ringdown_synth --> EXP_00..09 (sinteticos) --> EXP_08/09 (datos reales)
```

- **No tiene orquestador completo.** Solo `run_qnm_validation.sh` (parcial: spectrum + QNM_00 + QNM_01)
- **Proposito:** Validar que se pueden extraer parametros (f, tau, Q) de senales ringdown
- **Estado:** 10 experimentos, ~9,750 lineas, todos con contract_verdict.json

### El puente roto

Los dos pipelines deberian conectarse asi:

```
Geometria (bulk) --> Espectro --> Diccionario --> Atlas
                                                    |
Senal GW (ringdown) --> Features --> Comparacion con Atlas --> Inferencia del bulk
```

Pero en la practica:

- `stage_C6_ringdown_meta.py` **explicitamente dice** que no puede comparar
  porque el atlas esta en espacio "ratios" (dim=9) y ringdown en espacio "qnm" (dim=4)
- `stage_F3_closure.py` confirma: INCOMPATIBLE_FEATURE_SPACE
- `export_atlas_points.py` exporta puntos del atlas, pero **nadie los consume**
  en el pipeline ringdown de forma automatica
- Solo los experimentos QNM (00/01) leen `spectrum.h5` -- los demas (00-09) no tocan
  ningun artefacto holografico

**Conclusion:** El pipeline holografico y el pipeline ringdown operan como dos proyectos
separados que comparten infraestructura (basurin_io, contratos, runs/).

---

## 3. Inventario completo y clasificacion

### 3.1 Numeros globales

| Metrica | Valor |
|---------|-------|
| Ficheros Python | 173 |
| Lineas de codigo Python | ~43,500 |
| Tests | 81 ficheros, ~11,000 lineas |
| Commits | 187 (95 en main tras merges) |
| Documentos de spec | 18+ |
| Contratos formales | 4 |
| Stages canonicos | 11 |
| Experimentos | ~20 (en 9 familias) |

### 3.2 Clasificacion de experimentos por criticidad

#### ESENCIALES (columna vertebral del pipeline ringdown)

| Experimento | Que hace | Depende de | Lo usa |
|------------|----------|------------|--------|
| **EXP_01** injection-recovery | Estima f, tau, Q de sinteticos | ringdown_synth | EXP_03, 05, 06, 07 |
| **EXP_08** real smoke | Proof-of-concept GW150914 | pipeline real completo | EXP_09 |

Estos dos son el nucleo. Si EXP_01 falla, EXP_03/05/06/07 no pueden correr.
Si EXP_08 falla, no hay resultado cientifico con datos reales.

#### IMPORTANTES (robustez y validacion)

| Experimento | Que hace | Por que importa |
|------------|----------|-----------------|
| **EXP_00** stability sweep | Variaciones de preprocesado | Demuestra que EXP_01 no es fragil |
| **EXP_02** recovery robustness | Ruido y glitches | Robustez ante contaminacion |
| **EXP_QNM_00** open BC | Horizonte absorbente | Valida consistencia espectral |
| **EXP_QNM_01** closed-open limit | Limite cerrado/abierto | Consistencia interna |

#### SECUNDARIOS (nice-to-have, no bloquean)

| Experimento | Que hace | Riesgo de eliminarlo |
|------------|----------|---------------------|
| **EXP_03** observable minimality | Que observables son necesarios | Bajo (resultado academico) |
| **EXP_04** PSD validity | Validar espectro de potencia | Bajo (diagnostico) |
| **EXP_05** prior hyperparam | Sweep de hiperparametros | Bajo (sensibilidad) |
| **EXP_06** PSD robustness | Robustez de PSD | Bajo (duplica EXP_04 parcialmente) |
| **EXP_07** nonstationary stress | Ruido no estacionario | Medio (relevante para datos reales) |
| **EXP_09** real atlas sweep | Sweep multi-configuracion | Bajo (meta-experimento sobre EXP_08) |

#### PERIFERICOS (fuera del nucleo)

| Componente | Que hace | Estado |
|-----------|----------|--------|
| bridge/F4_1_alignment | Alineacion bridge | Desconectado |
| lpt_proxy (6 variantes) | Comparaciones LPT | Aislado, sin tests formales |
| ope_coefficients | Coeficientes OPE | Periferico |
| uldm_laser | Coherencia laser ULDM | Nuevo canal, desconectado |
| hsc_detector | Detector HSC | Input stage sin consumidor claro |
| exp_03/04 (dual spectrum, tangentes) | Experimentos legacy | Inactivos |

---

## 4. Los 5 problemas reales

### Problema 1: No existe un orquestador end-to-end

- `run_v1.py` hace 01->03->04->RUN_VALID (holografico)
- `run_qnm_validation.sh` hace spectrum->QNM_00->QNM_01 (parcial)
- **Nadie ejecuta**: RUN_VALID -> ringdown_synth -> EXP_01 -> EXP_00..07 -> resultado

**No hay un comando que diga "ejecuta el pipeline completo y dame el resultado."**

### Problema 2: El puente holografico-ringdown no existe

La razon de ser del proyecto es:
> "Inferir geometria del bulk a partir de observables"

Pero el mapa de features es incompatible:
- Atlas holografico: espacio "ratios", dim=9
- Ringdown features: espacio "qnm", dim=4

**No hay feature map definido.** `stage_C6_ringdown_meta.py` lo dice textualmente:
> "El atlas esta en espacio 'ratios' (dim=9) y ringdown_features esta en espacio QNM (dim=4).
> No se calculan distancias sin un feature-map explicito."

Esto es el gap critico. Sin este puente, los dos pipelines son dos proyectos separados.

### Problema 3: Los tests validan contratos, no ciencia

Los 81 tests verifican:
- Que los ficheros JSON tienen las claves correctas
- Que los paths existen
- Que los SHA256 se calculan
- Que los contratos pasan

**Pero no verifican:**
- Que los parametros (f, tau, Q) extraidos sean fisicamente correctos
- Que la inferencia holografica recupere la geometria original
- Que el pipeline end-to-end produzca un resultado cientifico correcto

"Todos los tests pasan" = "la fontaneria funciona". No = "el agua es potable".

### Problema 4: Cada experimento es un mundo cerrado

Patron repetido en los ultimos 187 commits:
1. Se crea EXP_N
2. Se escriben tests de contrato
3. Todo PASS
4. Se documenta
5. Se empieza EXP_N+1

Pero **ningun experimento produce un output que alimente una conclusion global**.
No hay un `final_report.json` que diga:
> "Para la geometria X, con espectro Y, el pipeline concluye que la geometria
> es/no es reconstruible con confianza Z"

### Problema 5: El roadmap esta abandonado

El roadmap define 7 fases:

| Fase | Nombre | Estado real |
|------|--------|-------------|
| 0 | Fundamentacion y blindaje | Parcialmente cubierta |
| 1 | Identificabilidad controlada | EXP_01/03 tocan esto |
| 2 | Generalizacion estructural | **No empezada** |
| 3 | Contrastacion externa | EXP_08 es un primer paso |
| 4 | Principios universales | **No empezada** |
| 5 | Diccionario minimo | **No empezada** |
| 6 | Emergencia de geometria | **No empezada** |

El proyecto se ha quedado girando entre Fase 0 y Fase 1, aniadiendo
mas experimentos de robustez (EXP_04, 05, 06, 07) en lugar de avanzar
hacia la pregunta cientifica central.

---

## 5. Diagrama de dependencias real

```
                   PIPELINE HOLOGRAFICO              PIPELINE RINGDOWN
                   ==================               =================

              01_genera_ads_puro.py
                      |
              03_sturm_liouville.py            stages/ringdown_synth_stage.py
                      |                                    |
              04_diccionario.py               experiment/run_valid/stage_run_valid.py
                      |                                    |
                 atlas.json                    synthetic_event(s).json
                      |                           /    |    \      \
                [atlas_points]          EXP_00  EXP_01  EXP_02  EXP_04
                      |                          |   /  |   \
              INCOMPATIBLE            EXP_03  EXP_05 EXP_06 EXP_07
              FEATURE_SPACE
                      |                    real data pipeline
                  (sin puente) -------->  EXP_08 --> EXP_09
                                              |
                                       (GW150914 smoke)

                 ???                          ???
                  \                          /
                   \                        /
                    +-- CONCLUSION FINAL --+
                         (NO EXISTE)
```

---

## 6. Que hacer (recomendacion priorizada)

### Prioridad 1: Definir el output final del proyecto

Antes de escribir una sola linea mas de codigo, responder:

> "Cuando BASURIN este terminado, que comando ejecuto y que resultado obtengo?"

Ejemplo concreto:
```bash
python basurin_run.py --geometry ads_puro --event GW150914
# --> runs/<id>/conclusion/inference_result.json
# {
#   "geometry_reconstructible": true,
#   "confidence": 0.94,
#   "best_model": "AdS_d4_delta_2.1",
#   "parameters": {"f_hz": 251.3, "tau_s": 0.004, "Q": 3.15}
# }
```

Si no puedes definir esto, el proyecto no tiene meta.

### Prioridad 2: Construir el feature map (el puente)

El gap critico: traducir features ringdown (f, tau, Q) a features holograficos
(ratios espectrales). Opciones:

1. **Analitico:** Derivar la relacion teorica QNM <-> espectro holografico
2. **Aprendido:** Entrenar un mapa con datos sinteticos donde se conoce la correspondencia
3. **Redefinir el atlas:** Construir el atlas en espacio QNM directamente

Sin esto, los dos pipelines nunca convergen.

### Prioridad 3: Un orquestador unico

Un solo script que ejecute el pipeline completo:

```
RUN_VALID
  --> geometry (01)
  --> spectrum (03)
  --> dictionary (04)
  --> ringdown_synth
  --> EXP_01 (injection-recovery)
  --> feature_map (EL PUENTE)
  --> comparacion atlas
  --> conclusion
```

### Prioridad 4: Congelar experimentos de robustez

EXP_00, 02, 03, 04, 05, 06, 07 aportan valor marginal mientras no exista el pipeline
completo. Congelarlos (no borrar, no ampliar) hasta que las prioridades 1-3 esten resueltas.

### Prioridad 5: Un test end-to-end real

Un unico test que:
1. Genere geometria sintetica con parametros conocidos
2. Calcule espectro
3. Genere senal ringdown sintetica correspondiente
4. Ejecute inferencia
5. Verifique que los parametros recuperados coinciden con los inyectados

Este test vale mas que los 81 tests de contrato actuales combinados.

---

## 7. Resumen ejecutivo

| Aspecto | Estado |
|---------|--------|
| Infraestructura (IO, contratos, SHA256) | Solida |
| Pipeline holografico (01->03->04) | Funcional pero sin consumidor |
| Pipeline ringdown (synth + EXP_00-09) | Extenso, todos PASS |
| Conexion holografico <-> ringdown | **ROTA** (feature space incompatible) |
| Orquestador end-to-end | **NO EXISTE** |
| Test de ciencia end-to-end | **NO EXISTE** |
| Output final definido | **NO** |
| Roadmap Fases 2-6 | **NO EMPEZADAS** |

**El proyecto tiene excelente fontaneria pero no tiene agua corriendo por las tuberias.**

Los 81 tests pasan porque verifican que cada pieza individual cumple su contrato formal.
Pero nadie ha verificado que todas las piezas juntas producen un resultado cientifico.

La proxima linea de codigo no deberia ser EXP_10. Deberia ser el puente.
