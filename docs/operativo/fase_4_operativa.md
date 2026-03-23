# Fase 4 operativa

## 1. Definición operativa

En el estado actual del repositorio, Fase 4 es la capa downstream que consume el resultado de clasificación geométrica y materializa artefactos relacionados con soporte común y consistencia tipo Hawking/área.

Para usuarios BRUNETE, Fase 4 no arranca como flujo independiente.
Su prerrequisito operativo es disponer de un classify run válido:

```text
runs/<classify_run_id>/classify_geometries/
```

La interfaz pública BRUNETE llega hasta `brunete_classify_geometries`.
La implementación downstream de Fase 4 existe en BASURIN como backend y evidencia operativa, no como CLI pública principal para BRUNETE.

## 2. Inputs requeridos

Input mínimo de entrada:

```text
runs/<classify_run_id>/RUN_VALID/verdict.json
runs/<classify_run_id>/classify_geometries/outputs/geometry_summary.json
```

Dependencias previas implícitas:

- batch `220` válido
- batch `221` válido
- clasificación conjunta ya emitida por BRUNETE

El input externo primario sigue siendo `data/losc/<EVENT_ID>/`, pero Fase 4 no lo consume como interfaz pública directa; llega a ella a través del pipeline ya materializado en `runs/`.

## 3. Relación con `classify`

`brunete.brunete_classify_geometries` es la superficie BRUNETE inmediatamente anterior a Fase 4.

Comando BRUNETE relevante:

```bash
python -m brunete.brunete_classify_geometries \
  --batch-220 brunete_batch_220_local \
  --batch-221 brunete_batch_221_local \
  --run-id brunete_classify_local
```

Artefactos de classify relevantes para Fase 4:

```text
runs/<classify_run_id>/classify_geometries/outputs/geometry_summary.json
runs/<classify_run_id>/classify_geometries/outputs/geometry_summary.csv
```

## 4. Artefactos de salida relevantes

La evidencia actual de Fase 4 en el repo se apoya en artefactos experimentales bajo:

```text
runs/<host_run_id>/experiment/phase4_hawking_area_common_support/
```

Outputs esperados según la batería de regresión:

- `manifest.json`
- `stage_summary.json`
- `outputs/*`

Este documento no eleva esos artefactos a interfaz pública BRUNETE.
Los presenta como superficie downstream operativa existente y testeada.

## 5. Evidencia de test

La evidencia principal está en:

- [tests/test_experiment_phase4_hawking_area_common_support.py](/home/ignac/work/basurin/tests/test_experiment_phase4_hawking_area_common_support.py)

Lo que ese test prueba hoy:

- gating del host run
- gating de batches de entrada
- dependencia real en artefactos previos ya materializados
- escritura exclusiva bajo `runs/<host_run_id>/experiment/phase4_hawking_area_common_support/`
- emisión de `manifest.json` y `stage_summary.json`

## 6. Estado operativo actual

Fase 4 puede presentarse hoy como:

- una superficie downstream real
- respaldada por tests
- dependiente de classify como punto de entrada operativo

Fase 4 no debe presentarse hoy como:

- una CLI pública BRUNETE independiente
- un flujo de onboarding separado del carril BRUNETE principal

## 7. Límites actuales

- No existe aún un wrapper BRUNETE específico para Fase 4.
- La documentación pública para Fase 4 todavía convive con material histórico en `docs/fases/`.
- La implementación subyacente vive en BASURIN/MVP y debe describirse como backend, no como interfaz recomendada para usuarios BRUNETE.

## 8. Referencias

- Flujo BRUNETE actual: [brunete_flujo_actual.md](brunete_flujo_actual.md)
- Rutas e IO: [../readme_rutas.md](../readme_rutas.md)
- Documento transitorio heredado de Fase 4: [../fases/fase_4_readme.md](../fases/fase_4_readme.md)

