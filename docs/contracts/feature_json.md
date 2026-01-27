# Feature JSON Contract (canonical)

Este documento define el contrato canónico para `feature JSON` usado por
`basurin_io.load_feature_json()` y los stages downstream (por ejemplo
`experiment/bridge/stage_F4_1_alignment.py`). **Es el único punto de entrada**.

## Formato canónico (recomendado)

### Atlas (X)

```json
{
  "schema_version": "1",
  "feature_key": "ratios",
  "ids": ["id_0", "id_1"],
  "X": [[0.1, 0.2], [0.3, 0.4]],
  "X_path": "X.npy",
  "shapes": {"n": 2, "dx": 2, "dy": 6},
  "meta": {
    "feature_key": "ratios",
    "columns": ["ratios_0", "ratios_1"],
    "created": "ISO-8601",
    "source_atlas": "runs/<run_id>/dictionary/outputs/atlas.json",
    "shapes": {"n": 2, "dx": 2, "dy": 6}
  }
}
```

### Features / Ringdown (Y)

```json
{
  "schema_version": "1",
  "feature_key": "tangentes_locales_v1",
  "ids": ["id_0", "id_1"],
  "Y": [[0.1, 0.2], [0.3, 0.4]],
  "X_path": "X.npy",
  "Y_path": "Y.npy",
  "shapes": {"n": 2, "dx": 2, "dy": 2},
  "meta": {
    "feature_key": "tangentes_locales_v1",
    "columns": ["d_eff", "m"],
    "k_neighbors": 7,
    "ids_source": "atlas_points.json",
    "created": "ISO-8601",
    "source_atlas": "runs/<run_id>/dictionary/outputs/atlas_points.json",
    "shapes": {"n": 2, "dx": 2, "dy": 2}
  }
}
```

### Reglas mínimas

- `schema_version` y `feature_key` **son obligatorios**.
- `ids` y `X`/`Y` **deben tener la misma longitud**.
- `X_path`/`Y_path` apuntan a binarios en `outputs/` (relativos a `features.json`).
- `shapes` debe incluir `n`, `dx`, `dy` para diagnosticar tamaños sin cargar matrices.
- Si `meta.columns` falta, el loader deriva columnas como
  `f"{feature_key}_{i}"`.
- `meta` puede incluir campos auxiliares (`source_atlas`, `ids_source`,
  `created`, etc.).

## Formatos legacy soportados (solo compatibilidad)

El loader acepta únicamente los siguientes formatos legacy y marca
`meta.resolved_from_legacy = true`:

1. **Atlas con lista de teorías o puntos**:
   - `{"theories": [{"id": ..., "ratios": [...]}, ...]}`
   - `{"points": [{"id": ..., "features": [...]}, ...]}`

2. **Lista de dicts**:
   - `[{ "id": ..., "x": [...] }, ...]` para atlas.
   - `[{ "id": ..., "y": [...] }, ...]` para features/ringdown.

3. **Features legacy con `X` en lugar de `Y`**:
   - `{"ids": [...], "X": [[...], ...]}` (solo `kind="features"`).

Cualquier formato distinto **es inválido** y el loader aborta con
`SystemExit(2)`.
