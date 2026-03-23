# BASURIN

BASURIN es un repositorio contract-first para análisis reproducible de ringdown.
En su estado actual, la fachada operativa pública recomendada para usuarios es BRUNETE.

## Estado actual

La superficie operativa vigente documentada hoy es:

- preparación y auditoría de cohortes BRUNETE
- ejecución batch local en modo `220` y `221`
- clasificación geométrica conjunta
- consumo downstream de classify en Fase 4 y Fase 5

El onboarding operativo no debe partir ya de Fase 1 ni Fase 2 históricas.
Tampoco debe usar `mvp/*` como interfaz pública cuando BRUNETE ya ofrece wrapper.

## Fachada operativa vigente

El flujo público BRUNETE actual es:

1. `python -m brunete.brunete_list_events`
2. `python -m brunete.brunete_audit_cohort_authority`
3. `python -m brunete.brunete_prepare_events`
4. `python -m brunete.brunete_run_batch --mode 220`
5. `python -m brunete.brunete_run_batch --mode 221`
6. `python -m brunete.brunete_classify_geometries`

Inputs externos:

```text
data/losc/<EVENT_ID>/
```

Outputs auditables:

```text
runs/<run_id>/...
runs/<classify_run_id>/experiment/<name>/...
```

## Documentación vigente

- Índice operativo: [docs/operativo/README.md](docs/operativo/README.md)
- Manual técnico BRUNETE: [brunete/MANUAL_OPERATIVO.md](brunete/MANUAL_OPERATIVO.md)
- SSOT de rutas e IO: [docs/readme_rutas.md](docs/readme_rutas.md)
- Documentación histórica: [docs/historico/README.md](docs/historico/README.md)

## Qué no usar como onboarding

- [docs/fases/fase_1_readme.md](docs/fases/fase_1_readme.md)
- [docs/fases/fase_2_readme.md](docs/fases/fase_2_readme.md)
- [docs/SYSTEM_READINESS_ASSESSMENT.md](docs/SYSTEM_READINESS_ASSESSMENT.md)

Esos documentos se conservan por trazabilidad, no como descripción del flujo operativo actual.

