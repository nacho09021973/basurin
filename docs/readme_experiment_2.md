# readme_experiment_2.md — Experimento 2: Run single-event + geometría (s4/s6/s6b) con auditoría

## Objetivo
Ejecutar el pipeline **single-event** en un evento que exista en el cache LOSC local, y producir:
- `s4_geometry_filter/outputs/compatible_set.json`
- `s6b_information_geometry_ranked/outputs/ranked_geometries.json`

Incluye checks contract-first (veredictos + hashes) y un “tuning” controlado de `epsilon` si `n_compatible==0`.

---

## Reglas BASURIN (no negociables)
- IO determinista **solo** bajo `runs/<run_id>/...` (prohibido escribir fuera).
- Gating: downstream solo si `RUN_VALID/verdict.json` tiene `"verdict": "PASS"`.
- Artefactos por stage: `manifest.json`, `stage_summary.json`, `outputs/` con SHA256 auditables.
- Si un stage falla: el run “no existe” downstream (fail-fast).

---

## Pre-requisitos
### 1) Entorno
```bash
source .venv/bin/activate
python -V
```

### 2) Cache LOSC local
Se asume que el cache está en:
- `data/losc/<EVENT_ID>/...*.h5|*.hdf5`

Comprueba qué eventos tienes:
```bash
find data/losc -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
```

> Recomendación: usar un `GW*` canónico que exista localmente (por ejemplo `GW190924_021846`), para minimizar problemas de metadatos.

---

## Selección rápida de evento “candidato”
### Opción A (recomendada): elegir un evento ya probado
Si existe en tu carpeta: `GW190924_021846`.

### Opción B: verificar que un evento tiene H1/L1
```bash
EVENT_ID="GW190924_021846"
find "data/losc/${EVENT_ID}" -type f \( -iname '*H1*.h5' -o -iname '*H1*.hdf5' -o -iname '*L1*.h5' -o -iname '*L1*.hdf5' \)
```

---

## Offline-first (recomendado)

Para reducir fallos por resolución dinámica de metadatos en GWOSC, usa un flujo **batch + catálogo t0** antes del orquestador.

Artefactos esperados de auditoría t0 (en run de auditoría):
- `runs/audit_gwosc_t0_*/experiment/losc_quality/t0_catalog_gwosc_v2.json`
- `runs/audit_gwosc_t0_*/experiment/losc_quality/gwosc_ready_events.txt`

### Receta para generar artefactos batch (GWOSC API v2)

```bash
AUDIT_RUN="audit_gwosc_t0_$(date -u +%Y%m%dT%H%M%SZ)"

python -m mvp.experiment_losc_quality \
  --run "$AUDIT_RUN" \
  --gwosc-api-version v2 \
  --batch-gwosc \
  --write-t0-catalog
```

Con ese preprocesado, el orquestador consume un catálogo ya “curado” y evita errores típicos tipo **"missing gps keys"** al resolver eventos en caliente.

---

## Run ID (determinista)
Crea un run_id nuevo por ejecución:
```bash
RUN_ID="mvp_${EVENT_ID}_real_$(date -u +%Y%m%dT%H%M%SZ)"
echo "$RUN_ID"
```

---

## Ejecución: pipeline single-event
> Nota: el orquestador requiere atlas; usa `--atlas-default` si está soportado por `mvp.pipeline`.

```bash
python -m mvp.pipeline single \
  --event-id "${EVENT_ID}" \
  --run-id "${RUN_ID}" \
  --atlas-default \
  --offline-s2 \
  --window-catalog "runs/audit_gwosc_t0_*/experiment/losc_quality/t0_catalog_gwosc_v2.json"
```

> Si no existe catálogo t0 aún, puedes lanzar sin `--window-catalog`. Cuando exista, úsalo para mantener el flujo offline-first. (Alias soportado: `--t0-catalog`)

---

## Gating: comprobar RUN_VALID
`RUN_VALID` es un directorio. El veredicto está en JSON:

```bash
python -c "import json,sys; p=f'runs/${RUN_ID}/RUN_VALID/verdict.json'; d=json.load(open(p)); print(d.get('verdict')); sys.exit(0 if d.get('verdict')=='PASS' else 1)"
```

Si no es PASS, parar aquí.

---

## Auditoría rápida del stage final s6b
### 1) Artefactos existen
```bash
ls -la "runs/${RUN_ID}/s6b_information_geometry_ranked"
```

Debe contener: `manifest.json`, `stage_summary.json`, `outputs/`.

### 2) Verificar hash del output principal
```bash
python -c "import json,hashlib; base=f'runs/${RUN_ID}'; s=json.load(open(base+'/s6b_information_geometry_ranked/stage_summary.json')); rel=s['outputs'][0]['path']; exp=s['outputs'][0]['sha256']; got=hashlib.sha256(open(base+'/'+rel,'rb').read()).hexdigest(); print('verdict=',s.get('verdict')); print('path=',rel); print('MATCH=',got==exp)"
```

### 3) Comprobar tamaños del ranking
```bash
python -c "import json; p=f'runs/${RUN_ID}/s6b_information_geometry_ranked/outputs/ranked_geometries.json'; d=json.load(open(p)); print('event_id=',d.get('event_id')); print('len_ranked=',len(d.get('ranked') or [])); print('len_compatible=',len(d.get('compatible') or [])); print('source=',(d.get('compatibility_criterion') or {}).get('params',{}).get('source'))"
```

---

## Si `n_compatible == 0`: tuning mínimo (s4) con epsilon
Síntoma típico: `compatible_geometries=[]` porque el umbral (`threshold_d2`) es demasiado estricto.

### 1) Mirar escala de d2 en s4
```bash
python -c "import json; p=f'runs/${RUN_ID}/s4_geometry_filter/outputs/compatible_set.json'; d=json.load(open(p)); ra=d.get('ranked_all') or []; d2=[x['d2'] for x in ra if 'd2' in x]; print('epsilon=',d.get('epsilon'),'threshold_d2=',d.get('threshold_d2')); print('n_compatible=',d.get('n_compatible')); print('d2_min=',min(d2) if d2 else None,'d2_median=',sorted(d2)[len(d2)//2] if d2 else None,'d2_max=',max(d2) if d2 else None)"
```

### 2) Rerun de s4 con epsilon mayor (sin tocar el resto)
Obtén el atlas_path real trazado por el run:
```bash
python -c "import json; p=f'runs/${RUN_ID}/s4_geometry_filter/stage_summary.json'; d=json.load(open(p)); print((d.get('parameters') or {}).get('atlas_path'))"
```

Rerun (ejemplo: `epsilon=1200`, ajusta según el rango de d2):
```bash
python -m mvp.s4_geometry_filter --run "${RUN_ID}" --atlas-path "/home/ignac/work/basurin/docs/ringdown/atlas/atlas_berti_v2.json" --epsilon 1200 --metric mahalanobis_log
```

> Importante: este stage **no** acepta `--event-id` y **requiere** `--atlas-path`.

### 3) Verificar que ahora hay compatibles
```bash
python -c "import json; p=f'runs/${RUN_ID}/s4_geometry_filter/outputs/compatible_set.json'; d=json.load(open(p)); print('n_compatible=',d.get('n_compatible')); print('compatible_len=',len(d.get('compatible_geometries') or []))"
```

### 4) Regenerar s6b para reflejar el nuevo s4
```bash
python -m mvp.s6b_information_geometry_ranked --run "${RUN_ID}"
```

Y re-check:
```bash
python -c "import json; p=f'runs/${RUN_ID}/s6b_information_geometry_ranked/outputs/ranked_geometries.json'; d=json.load(open(p)); print('len_compatible=',len(d.get('compatible') or []))"
```

---

## Diagnóstico rápido de fallos frecuentes
### A) “GWOSC lookup failed … 404 / missing gps keys”
- Solución: usar un evento `GW*` canónico presente en `data/losc/` (p.ej. `GW190924_021846`).
- Evitar IDs “cortos” tipo `151008` si GWOSC no los resuelve.

### B) “No se pudieron auto-resolver HDF5 … h5_count=0”
- El `event_id` no existe en `data/losc/<event_id>/`.
- Solución: elegir un `event_id` que exista, o crear symlink canónico:
  `data/losc/GWXXXX -> data/losc/XXXX` si aplica.

### C) `--atlas-path is required`
- Algunos stages exigen atlas explícito aunque el pipeline tenga `--atlas-default`.
- Solución: leer `atlas_path` desde `stage_summary.json` del stage que ya lo usó y pasarlo explícito en reruns.

---

## Qué entregar al final (para reproducibilidad)
- `RUN_ID`
- `EVENT_ID`
- Veredicto global: `runs/<RUN_ID>/RUN_VALID/verdict.json`
- Hashes auditados:
  - `runs/<RUN_ID>/s4_geometry_filter/stage_summary.json`
  - `runs/<RUN_ID>/s6b_information_geometry_ranked/stage_summary.json`
- Si hubo tuning: valor final de `epsilon` en `s4_geometry_filter/parameters`.

Fin.
