# MAPA OPERATIVO DE BASURIN (para humanos)

> **Propósito**: este documento existe *solo* para orientarte a ti cuando vuelvas al proyecto después de días/semanas.
> No es normativo, no gobierna contratos y no sustituye a `BASURIN_README_SUPER.md`.
> Su función es **ahorrar horas** y evitar búsquedas arqueológicas.

---

## 0. Qué documento manda y cuáles no

### Documento soberano (gobierna la realidad)
- **`BASURIN_README_SUPER.md`**  
  Define reglas duras: RUN_VALID, IO determinista, estructura de stages, aborts, DoD.
  Si algo contradice esto, **esto gana**.

### Documentos auxiliares (contexto, no contratos)
- `README.md` → puerta de entrada, descripción general.
- `TESIS_PUENTE_PHI.md` → hipótesis científica que se intenta falsar.
- `RELACION_QNM_BULK.md` → marco físico / intuición (no ejecutable).
- Este documento → **mapa mental y operativo**.

---

## 1. Dos raíces importantes (fuente del 80% de los problemas)

```
~/basurin/        ← raíz del repo (documentos, scripts históricos)
~/basurin/work/   ← raíz operativa real (runs/, ejecución)
```

⚠️ **Regla práctica**:  
Todo lo que escriba en `runs/<run_id>/` debe ejecutarse **desde `work/`**.

---

## 2. Dónde vive un run de verdad

```
work/runs/<RUN_ID>/
├── RUN_VALID/
├── geometry/
├── geometry_contracts/
├── spectrum/            (o sturm_liouville)
├── dictionary/
├── ringdown_featuremap_v0/
├── geometry_select_v0/
└── experiment/
```

Si un stage escribe en `~/basurin/runs/...` en vez de `work/runs/...`, el run queda **partido** y hay que regenerar.

---

## 3. Pipeline mínimo real (orden mental correcto)

Este es el orden **real**, no el imaginado:

1. **RUN_VALID**  
   Script: `experiment/run_valid/stage_run_valid.py`

2. **Geometría (generación)**  
   Script: `work/01_genera_ads_puro.py`
   - Produce: `geometry/outputs/ads_puro.h5`

3. **Geometría (contratos post-hoc)**  
   Script: `work/02b_geometry_contracts_stage.py`
   - Consume: `ads_puro.h5`
   - Produce: `geometry_contracts/outputs/contracts.json`

4. **Espectro (Sturm–Liouville)**  
   Script: `work/03_sturm_liouville.py`
   - Consume: geometría (H5 o JSON)
   - Produce: `*spectrum*.h5`

5. **Diccionario**  
   Script: `work/04_diccionario.py`
   - Consume: spectrum H5
   - Produce:
     - `dictionary/outputs/dictionary.h5`
     - `dictionary/outputs/atlas_points.json` (con `--export-atlas-points`)

6. **Ringdown → featuremap**  
   Script: `stages/ringdown_featuremap_v0_stage.py`
   - Consume: parámetros ringdown (JSONL)
   - Produce: `mapped_features.json`

7. **Selección de geometría**  
   Script: `stages/geometry_select_v0_stage.py`
   - Consume: `mapped_features.json` + atlas
   - Produce: ranking

---

## 4. Cosas que NO generan datos (pero confunden)

- `02b_geometry_contracts_stage.py` **no genera geometría**.
- `geometry_select_v0_stage.py` **no genera atlas**.
- El diccionario **no se puede correr sin spectrum**, aunque parezca autónomo.

---

## 5. Inputs típicos y su formato real

- Ringdown params → **JSONL**, no JSON bonito.
- Atlas → ahora mismo es `atlas_points.json` (no `atlas.json`).
- Geometría base → HDF5 (`ads_puro.h5`).

---

## 6. Regla de oro para no perder horas

> Si un stage pide un input y no sabes de dónde sale:
> **no busques el código** → busca **qué stage lo produce**.

Comando mental estándar:
```
ls work | grep -E '01_|02_|03_|04_'
```

---

## 7. Estado mental correcto del proyecto

- BASURIN **no es lineal**, es un grafo.
- Muchos scripts son históricos, no canónicos.
- El README no es un tutorial: es una constitución.
- Este mapa existe para que no tengas que recordar nada.

---

## 8. Qué NO debería volver a pasarte

- Buscar `atlas.json` cuando el diccionario exporta `atlas_points.json`.
- Ejecutar generadores desde la raíz equivocada.
- Asumir que un script "de contratos" genera datos.
- Pensar que el README te va a guiar paso a paso.

---

## 9. Nota honesta (para ti)

Este documento **no es para IA, ni para futuros colaboradores**.
Es para que, cuando vuelvas cansado o dentro de 3 meses, puedas leer 10 minutos y **recordar dónde estás**.

Si mañana quieres, este mapa se puede destilar en:
- un diagrama,
- o una checklist imprimible,
- o un `make smoke-run`.

Pero esto ya evita el 90% del desgaste que acabamos de vivir.
