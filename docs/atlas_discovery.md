# Descubrimiento de Atlas (geometrías)

# Atlas (geometrías) — localización inmediata (<60s)

## Rutas canónicas dentro del repo (no inventar “atlas.json”)
Estos ficheros viven versionados en el repo (preferidos por gobernanza). Lista explícita:

- `docs/ringdown/atlas/atlas_real_v2_s4.json`  *(recomendado para `s4_geometry_filter`)*
- `docs/ringdown/atlas/atlas_berti_v2_s4.json`
- `docs/ringdown/atlas/atlas_real_v1_s4.json`
- `docs/ringdown/atlas/atlas_berti_v2.json`
- `docs/ringdown/atlas/atlas_real_v1.json`
- `mvp/test_atlas_fixture.json` *(solo tests; no usar en runs reales)*

**Regla:** no existe (ni debe sugerirse) `atlas.json` en raíz. Si ves docs/ejemplos con `--atlas-path atlas.json`, trátalo como anti-ejemplo heredado.

## Qué atlas usar dónde (regla operativa)
- `s4_geometry_filter`: preferir `docs/ringdown/atlas/*_s4.json`
- tests: `mvp/test_atlas_fixture.json`
- batch/offline (`experiment_offline_batch`): documentar el atlas efectivo usado por esa CLI:
  - hoy está documentado como `docs/ringdown/atlas/atlas_berti_v2.json` en `docs/readme_experiment_4.md`
  - migración a `*_s4.json` solo cuando se haga explícita (no asumir)

## Descubrimiento (copy/paste)
Si dudas de rutas o estás en un checkout distinto:

```bash
find . -maxdepth 6 -type f \( -name "atlas*.json" -o -name "*atlas*.json" \) | sort
```

Comprobación rápida (estructura JSON):

```bash
python -c 'import json; p="docs/ringdown/atlas/atlas_real_v2_s4.json"; print("OK", p, "top_keys=", list(json.load(open(p)).keys())[:10])'
```

## Ejemplo de uso en pipeline (single-event)

```bash
python mvp/pipeline.py single \
  --event-id GW191113_071753 \
  --atlas-path docs/ringdown/atlas/atlas_real_v2_s4.json \
  --run-id <RUN_ID>
```

Si vas con defaults:

- usa `--atlas-default` (si está soportado por tu CLI) en lugar de inventar rutas.
