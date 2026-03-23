# Fase 2 — Documento histórico

## Estado

Este documento se conserva por trazabilidad histórica.
Ya no describe el onboarding operativo actual del repositorio.

## Qué describía

La antigua Fase 2 documentaba batches offline basados en tooling y artefactos previos al carril BRUNETE actual.
Ese material sigue siendo útil para auditoría y contexto, pero no debe presentarse como interfaz pública vigente.

## Por qué ya no es flujo principal

- la ejecución batch pública vigente se hace con `brunete.brunete_run_batch`
- la preparación de cohortes pública vigente se hace con `brunete.brunete_prepare_events`
- el downstream operativo actual arranca desde runs BRUNETE y classify, no desde este documento

## Reemplazo operativo actual

Usar en su lugar:

- [docs/operativo/brunete_flujo_actual.md](../operativo/brunete_flujo_actual.md)
- [brunete/MANUAL_OPERATIVO.md](../../brunete/MANUAL_OPERATIVO.md)
- [docs/readme_rutas.md](../readme_rutas.md)

## Regla

No usar este documento como guía de arranque ni como referencia principal para ejecutar batches hoy.
Solo debe consultarse para trazabilidad histórica de runs, decisiones y nomenclatura anterior.

