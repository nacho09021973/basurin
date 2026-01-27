# Convención de rutas e IO (BASURIN)

Este documento define **rutas canónicas** y reglas de compatibilidad para evitar perder tiempo buscando artefactos.

## Estructura obligatoria por ejecución

Cada ejecución vive en:

```
runs/<run_id>/<stage>/
```

Archivos mínimos por stage:

- `manifest.json`
- `stage_summary.json`
- `outputs/`

> Regla: **no se escribe fuera de `runs/<run_id>/`**.

## Stages canónicos

- `geometry`
- `spectrum`
- `dictionary`
- `exp*` (por ejemplo `exp01`, `exp_local`, etc.)

## Layout NUEVO vs LEGACY

**NUEVO (preferido):**

```
runs/<run_id>/<stage>/outputs/<artifact>
```

**LEGACY (compatibilidad):**

```
runs/<run_id>/<stage>/<artifact>
```

**Regla de compatibilidad:**
1) Buscar siempre en `outputs/` primero.
2) Si no existe, caer a legacy.
3) No mezclar escrituras: nuevos artefactos **siempre** en `outputs/`.

**Casos reales observados:**
- `runs/<run_id>/dictionary/outputs/validation.json`
- `runs/<run_id>/spectrum/outputs/spectrum.h5`

## Paths canónicos por stage

- `geometry`: `runs/<run_id>/geometry/outputs/`
- `spectrum`: `runs/<run_id>/spectrum/outputs/spectrum.h5`
- `dictionary`: `runs/<run_id>/dictionary/outputs/validation.json`
- `exp*`: `runs/<run_id>/exp*/outputs/`

## Comandos `find` útiles

Buscar `validation.json`:

```bash
find runs -path "*/dictionary/*" -name "validation.json"
```

Buscar `spectrum.h5`:

```bash
find runs -path "*/spectrum/*" -name "spectrum.h5"
```

Buscar `stage_summary.json`:

```bash
find runs -name "stage_summary.json"
```

Buscar `manifest.json`:

```bash
find runs -name "manifest.json"
```

## Checklist diario “anti-perdida de horas”

- [ ] Estoy dentro de `runs/<run_id>/` y el `run_id` correcto.
- [ ] Cada stage tiene `manifest.json` y `stage_summary.json`.
- [ ] Busco artefactos en `outputs/` **antes** de revisar legacy.
- [ ] Si no encuentro un archivo, corro los `find` de arriba.
- [ ] No escribo ni muevo archivos fuera de `runs/<run_id>/`.

## Apéndice: resolvers recomendados (pseudocódigo)

**Resolver de `spectrum.h5`:**

```
function resolve_spectrum_h5(run_id):
    new_path = runs/<run_id>/spectrum/outputs/spectrum.h5
    legacy_path = runs/<run_id>/spectrum/spectrum.h5

    if exists(new_path):
        return new_path
    if exists(legacy_path):
        return legacy_path

    raise FileNotFoundError("spectrum.h5 no encontrado en outputs/ ni legacy")
```

**Resolver de `validation.json`:**

```
function resolve_validation_json(run_id):
    new_path = runs/<run_id>/dictionary/outputs/validation.json
    legacy_path = runs/<run_id>/dictionary/validation.json

    if exists(new_path):
        return new_path
    if exists(legacy_path):
        return legacy_path

    return None  # o raise, según necesidad
```
