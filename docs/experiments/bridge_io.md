# Bridge F4-1 IO (atlas + tangentes locales)

Este stage conecta un atlas (X) con features externas (Y) usando CCA, con IO determinista bajo `runs/<run>/bridge_f4_1_alignment/`.

## Entradas requeridas

### Atlas (X)
Formatos soportados:

- **Atlas bridge** (recomendado):

```json
{
  "ids": ["id_0", "id_1"],
  "X": [[...], [...]],
  "meta": {
    "feature_key": "ratios",
    "columns": ["ratios_0", "ratios_1"],
    "schema_version": "1"
  }
}
```

- **Atlas points** (diccionario):

```json
{
  "feature_key": "ratios",
  "n_points": 2,
  "points": [
    {"id": "id_0", "features": [0.1, 0.2]},
    {"id": "id_1", "features": [0.3, 0.4]}
  ]
}
```

### Tangentes locales (Y)
Formato exportado por `05_tangentes_locales.py`:

```json
{
  "ids": ["idx_0", "idx_1"],
  "Y": [[...], [...]],
  "X_path": "X.npy",
  "Y_path": "Y.npy",
  "shapes": {"n": 2, "dx": 2, "dy": 6},
  "meta": {
    "feature_key": "tangentes_locales_v1",
    "columns": ["d_eff", "m", "parallel", "perp", "rho_clipped", "log10_rho"],
    "k_neighbors": 7,
    "schema_version": "1"
  }
}
```

## Cómo ejecutar

1) Exportar tangentes locales:

```bash
python 05_tangentes_locales.py --run <RUN> --k-neighbors 7
```

2) Ejecutar Bridge F4-1:

```bash
python -m experiment.bridge.stage_F4_1_alignment \
  --run <RUN> \
  --atlas runs/<RUN>/dictionary/outputs/atlas_points.json \
  --features runs/<RUN>/tangentes_locales/outputs/features_points_k7.json
```

## Compatibilidad y kill-switch

- Si `feature_key` o `columns` coinciden entre X e Y, el stage aborta con `LEAKAGE_SUSPECTED`.
- Si los espacios son distintos, el kill-switch solo aborta por identidad numérica casi exacta.
