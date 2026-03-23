# Fase 1 — Documento histórico

## Estado

Este documento se conserva por trazabilidad histórica.
Ya no describe el onboarding operativo actual del repositorio.

## Qué describía

La antigua Fase 1 documentaba un flujo de preparación y auditoría LOSC/t0 apoyado en tooling previo a la fachada BRUNETE actual.
Ese flujo incluía artefactos y pasos de preparación que ya no son la entrada pública recomendada para operar el sistema hoy.

## Por qué ya no es flujo principal

- la superficie operativa pública vigente para usuarios es BRUNETE
- el flujo actual parte de wrappers bajo `brunete/`
- la caché externa canónica vigente se documenta como `data/losc/<EVENT_ID>/`
- el onboarding actual está centralizado en la documentación operativa, no en esta fase histórica

## Reemplazo operativo actual

Usar en su lugar:

- [docs/operativo/brunete_flujo_actual.md](../operativo/brunete_flujo_actual.md)
- [brunete/MANUAL_OPERATIVO.md](../../brunete/MANUAL_OPERATIVO.md)
- [docs/readme_rutas.md](../readme_rutas.md)

## Regla

No usar este documento como guía de arranque ni como descripción del flujo vigente.
Solo debe consultarse para reconstrucción histórica de decisiones o artefactos anteriores.

