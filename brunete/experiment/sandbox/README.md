# Sandbox Científico B5-G

Espacio libre para hipótesis, visualizaciones y métricas nuevas sin compromiso de contrato.

## Reglas

1. **Nunca importar** `brunete.experiment` ni ningún módulo del core. Usa copias locales de artefactos.
2. **Siempre presente**: el archivo `ISOLATION_MARKER` en este directorio.
3. Cada experimento sandbox vive en su propio subdirectorio con nombre + fecha:
   ```
   sandbox/<nombre_YYYYMMDD>/
     input_snapshot/    ← copias de artefactos (nunca symlinks)
     notebooks/         ← Jupyter permitido solo aquí
     scratch/           ← sin esquema requerido
     README.md          ← hipótesis + resultado
     ISOLATION_MARKER   ← archivo vacío
   ```

## Cómo usar

```bash
# Copiar artefacto (nunca referenciar directamente)
cp runs/<classify_run_id>/classify_geometries/outputs/geometry_summary.json \
   brunete/experiment/sandbox/mi_experimento_20260321/input_snapshot/
```

## Criterio de promoción

Ningún output sale directamente al core. Camino:
`sandbox → formalizar como B5-X → evaluar promoción`
