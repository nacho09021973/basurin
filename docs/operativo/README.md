# Documentación operativa

Este índice reúne la documentación que describe la superficie operativa vigente del repositorio.
Su ámbito es ejecución, rutas, contratos y outputs auditables.

## Documentos principales

- [Flujo BRUNETE actual](brunete_flujo_actual.md)
  Entry points públicos, prerequisitos, cohortes y secuencia `list -> audit -> prepare -> batch 220 -> batch 221 -> classify`.

- [Fase 4 operativa](../fases/fase_4_readme.md)
  Enlace transitorio al documento actual de Fase 4 mientras no exista `docs/operativo/fase_4_operativa.md`.

- [Fase 5 operativa](../fases/fase_5_readme.md)
  Enlace transitorio al documento actual de Fase 5 mientras no exista `docs/operativo/fase_5_operativa.md`.

- [Rutas e IO](../readme_rutas.md)
  SSOT de inputs externos, outputs internos y contratos de escritura.

- [Manual operativo de BRUNETE](../../brunete/MANUAL_OPERATIVO.md)
  Guía técnica detallada del carril BRUNETE.

## Alcance

Esta carpeta documenta lo que un operador debe ejecutar hoy.
No sustituye auditorías históricas ni documentación de transición.

## Límites

- `mvp/*` puede existir como backend o implementación subyacente, pero no se documenta aquí como interfaz pública cuando BRUNETE ya ofrece wrapper.
- La documentación histórica de Fase 1, Fase 2 y readiness antiguo queda fuera de este índice.
- Los ejemplos de esta carpeta asumen inputs externos en `data/losc/<EVENT_ID>/` y escritura auditable en `runs/<run_id>/...`.
