# Fase 5 operativa

## 1. Alcance actual

En el estado actual del repositorio, Fase 5 es la capa experimental downstream que consume classify runs válidos y escribe artefactos bajo:

```text
runs/<classify_run_id>/experiment/<name>/
```

Para BRUNETE, la superficie operativa actual de Fase 5 sí tiene wrappers propios bajo `brunete/experiment/`.

## 2. Contrato de entrada

El contrato de entrada parte de un classify run válido:

```text
runs/<classify_run_id>/RUN_VALID/verdict.json
runs/<classify_run_id>/classify_geometries/outputs/geometry_summary.json
```

La resolución de event runs y vistas auxiliares se gobierna desde:

```text
brunete/experiment/base_contract.py
```

Ese contrato restaura hoy, entre otras, estas APIs:

- `validate_classify_run(...)`
- `enumerate_event_runs(...)`
- `resolve_event_run_dirs(...)`
- `materialize_event_run_view(...)`
- `ensure_experiment_dir(...)`

## 3. Wrappers B5 disponibles

Wrappers actualmente visibles y ejecutables:

- `python -m brunete.experiment.b5a --classify-run-id <id> --mode 220|221`
- `python -m brunete.experiment.b5b --classify-run-id <id> --mode 220|221`
- `python -m brunete.experiment.b5c --classify-run-id <id> --mode 220|221`
- `python -m brunete.experiment.b5e --classify-run-id <id> --query <expr>`
- `python -m brunete.experiment.b5f --classify-run-id <id>`
- `python -m brunete.experiment.b5h --classify-run-id <id> --mode 220|221`
- `python -m brunete.experiment.b5z --classify-run-id <id> --mode 220|221`

Flags reales verificadas:

- `--classify-run-id`
- `--mode {220,221}` cuando aplica
- `--query` en `b5e`
- `--strategy {intersection,majority,frequency_weighted}` en `b5h`
- `--runs-root`
- `--dry-run`

## 4. Artefactos de salida

Artefactos BRUNETE actualmente emitidos por wrapper:

- `runs/<classify_run_id>/experiment/b5a_multi_event_aggregation/aggregation_result.json`
- `runs/<classify_run_id>/experiment/b5a_multi_event_aggregation/jaccard_matrix.json`
- `runs/<classify_run_id>/experiment/b5a_multi_event_aggregation/persistence_histogram.json`
- `runs/<classify_run_id>/experiment/b5b_jackknife/stability_per_geometry.json`
- `runs/<classify_run_id>/experiment/b5b_jackknife/influence_ranking.json`
- `runs/<classify_run_id>/experiment/b5b_jackknife/summary.json`
- `runs/<classify_run_id>/experiment/b5c_geometry_ranking/ranking_by_event.json`
- `runs/<classify_run_id>/experiment/b5e_query/query_<query_id>.json`
- `runs/<classify_run_id>/experiment/b5f_verdict_aggregation/population_verdict.json`
- `runs/<classify_run_id>/experiment/b5h_blind_prediction/prediction_results.json`
- `runs/<classify_run_id>/experiment/b5z_continuous_emulator/predicted_minima_by_event.json`

Además, cada wrapper materializa `manifest.json` en su directorio experimental.

Los nombres de directorio y ficheros anteriores están respaldados por los wrappers actuales de `brunete/experiment/`.

## 5. Ejemplos de CLI

Ejemplos directos:

```bash
python -m brunete.experiment.b5a \
  --classify-run-id brunete_classify_local \
  --mode 220
```

```bash
python -m brunete.experiment.b5f \
  --classify-run-id brunete_classify_local
```

```bash
python -m brunete.experiment.b5h \
  --classify-run-id brunete_classify_local \
  --mode 221 \
  --strategy intersection
```

## 6. Evidencia de test

La evidencia principal está en:

- [tests/test_brunete_experiment.py](/home/ignac/work/basurin/tests/test_brunete_experiment.py)

Lo que esa batería prueba hoy:

- que los wrappers B5 importan
- que `base_contract` resuelve classify runs y event runs
- que `materialize_event_run_view(...)` y `resolve_event_run_dirs(...)` existen y funcionan
- que `b5a` y `b5h` respetan la resolución mode-specific
- que `b5f` expone compatibilidad con `n_events` y resumen derivado de `has_joint_support`

Evidencia complementaria:

- [tests/test_e5_governance.py](/home/ignac/work/basurin/tests/test_e5_governance.py)
- [tests/test_e5z_gpr_emulator.py](/home/ignac/work/basurin/tests/test_e5z_gpr_emulator.py)

## 7. Estado operativo actual

Fase 5 puede presentarse hoy como:

- una capa experimental real y ejecutable
- ya integrada con classify runs BRUNETE
- respaldada por wrappers propios en `brunete/experiment/`

Fase 5 no debe presentarse hoy como:

- un catálogo abierto de cualquier módulo `mvp/experiment/*`
- una interfaz pública basada directamente en CLI internas de `mvp/*`

## 8. Límites actuales

- Los wrappers B5 cubren una selección operativa concreta, no todo el namespace experimental interno.
- El input gobernante sigue siendo el classify run; no hay atajo público recomendado que salte ese contrato.
- Parte de la documentación histórica de Fase 5 sigue existiendo en `docs/fases/` y debe entenderse como material transitorio hasta completar la migración documental.

## 9. Referencias

- Flujo BRUNETE actual: [brunete_flujo_actual.md](brunete_flujo_actual.md)
- Rutas e IO: [../readme_rutas.md](../readme_rutas.md)
- Documento transitorio heredado de Fase 5: [../fases/fase_5_readme.md](../fases/fase_5_readme.md)
