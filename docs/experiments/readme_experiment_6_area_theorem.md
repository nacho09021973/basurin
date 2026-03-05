# Experimento 6 (T6): Teorema del Área (Hawking) como test de coherencia ringdown vs IMR

## 1) Propósito y alcance

T6 evalúa **coherencia interna** entre dos canales de estimación para

\[
\Delta A = A_f - (A_1 + A_2)
\]

- **Canal control IMR-consistency**: usa todo desde el mismo posterior IMR (`final_mass_source`, `final_spin`, `mass_1_source`, `mass_2_source`, `a_1`, `a_2`). Este control debe dar `p_violate=0`.
- **Canal ringdown-vs-IMR**: usa `A_f` desde ringdown (`Af_RD`) derivado de intersección física Kerr `220∩221`, y `A_1+A_2` desde IMR.

Interpretación correcta:

- Si el canal ringdown reporta `p_violate>0`, eso se interpreta como **tensión ringdown vs IMR** (o limitación del atlas/intersección),
- **NO** como “violación del teorema del área de Hawking”, porque el control IMR pasa.

---

## 2) Definiciones y fórmula usadas (consistentes con E5)

En unidades geométricas (`G=c=1`):

\[
A(M,\chi)=8\pi M^2\left(1+\sqrt{1-\chi^2}\right)
\]

\[
\Delta A = A_f - (A_1 + A_2),\qquad p_{violate}=P(\Delta A<0)
\]

Regla crítica para componentes iniciales:

- usar **magnitud de spin** `|chi|`, implementada como `a_1` y `a_2`;
- **no** usar `spin_z`.

Usar `spin_z` en `A_1/A_2` induce falsos positivos de “violación aparente” por subestimar la magnitud de spin efectiva en la fórmula del área.

---

## 3) Inputs y preparación

Run raíz del experimento:

- `runs/analysis_area_theorem_20260304T173747Z/`

Inputs IMR descargados (HDF5, pilot) en:

- `runs/analysis_area_theorem_20260304T173747Z/external_inputs/gwtc_posteriors/raw/`

Archivos pilot:

- `IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_mixed_nocosmo.h5`
- `IGWN-GWTC2p1-v2-GW190814_211039_PEDataRelease_mixed_nocosmo.h5`
- `IGWN-GWTC3p0-v2-GW191105_143521_PEDataRelease_mixed_nocosmo.h5`
- `IGWN-GWTC3p0-v2-GW191113_071753_PEDataRelease_mixed_nocosmo.h5`
- `IGWN-GWTC3p0-v2-GW191204_110529_PEDataRelease_mixed_nocosmo.h5`
- `IGWN-GWTC3p0-v2-GW200322_091133_PEDataRelease_mixed_nocosmo.h5`

Conversión a JSON canónico en:

- `runs/analysis_area_theorem_20260304T173747Z/external_inputs/gwtc_posteriors/<EVENT>.json`
- manifiesto SHA256:
  - `runs/analysis_area_theorem_20260304T173747Z/external_inputs/gwtc_posteriors/manifest_inputs_sha256.json`

Lista pilot usada:

- `runs/analysis_area_theorem_20260304T173747Z/external_inputs/gwtc_posteriors/pilot_events.txt`

Batches ringdown usados para mapear `event -> run`:

- `B220 = batch_with_t0_220_eps2500_fixlen_20260304T160054Z`
- `B221 = batch_with_t0_221_eps2500_fixlen_20260304T160617Z`

---

## 4) Cálculo de `Af_RD` (ringdown)

`Af_RD` se deriva de la intersección física Kerr `220∩221` por:

- `phys_key = (family, source, M_solar, chi)`

No se intersecta por `geometry_id` (porque el modo está embebido y 220/221 no coinciden literal por ID).

Si para un evento no hay compatibles en 220 que intersecten con 221 en `phys_key`, la intersección es vacía y:

- `Af_RD` queda indefinido,
- se reporta `AF_EMPTY` (comportamiento conservador por contrato).

---

## 5) Cálculo de `A1 + A2` (IMR)

`A_1+A_2` se computa desde el posterior IMR de cada evento con:

- `mass_1_source`, `mass_2_source`, `a_1`, `a_2`.

Punto clave:

- `a_1` y `a_2` son magnitudes de spin (`|chi|`) y son las variables correctas para la fórmula de área Kerr.
- usar `spin_z` en su lugar distorsiona `A_1+A_2` y puede crear falsos positivos en `p_violate`.

---

## 6) Outputs T6 (rutas exactas)

Directorio base de outputs:

- `runs/analysis_area_theorem_20260304T173747Z/experiment/area_theorem/outputs/`

Artefactos:

1. Histórico (`Af_RD` + `A1+A2` con spins iniciales mal interpretados como `chi1/chi2`):
   - `per_event.csv`
2. Cálculo correcto (`Af_RD` + `A1+A2` con magnitud `a_1/a_2` desde H5):
   - `per_event_spinmag.csv`
   - `summary_spinmag.json`
3. Control IMR-consistency (todo IMR con `|spin|`):
   - `per_event_imrconsistency.csv`
   - `summary_imrconsistency.json`
4. Diagnóstico MAP-only (GW200322):
   - `per_event_map_only.csv`

---

## 7) Resultados y lectura

### 7.1 Resultado principal con spin magnitud correcta (`per_event_spinmag.csv`)

- `GW200322_091133`: `p_violate=0.379205`, `dA_p50=84720.1`, `n_mc=1962` → **tensión ringdown vs IMR**.
- `GW191204_110529`: `p_violate=0.0141024` → cola negativa pequeña.
- `GW190814`: `p_violate=0`.
- `GW191105_143521`: `p_violate=0`.
- `GW190521`, `GW191113_071753`: `AF_EMPTY` (sin intersección Kerr `220∩221` para definir `Af_RD`).

### 7.2 Control IMR-consistency (`per_event_imrconsistency.csv`)

- Para todos los eventos pilot: `p_violate=0` (control pasa), con `n_mc` según el CSV.

Conclusión de T6:

- la “violación aparente” del canal ringdown **no** se interpreta como violación de Hawking;
- se interpreta como **tensión entre Af_RD (ringdown/intersección de atlas) y A1+A2 del IMR**.

---

## 8) Auditoría (rutas y trazabilidad)

### 8.1 Inputs IMR y hashes

- `runs/analysis_area_theorem_20260304T173747Z/external_inputs/gwtc_posteriors/manifest_inputs_sha256.json`

### 8.2 Evidencia por evento en batches ringdown

Para cada evento del pilot, obtener el `run_id` en cada batch desde `results.csv` y auditar:

- `runs/<RUN220>/s4_geometry_filter/outputs/compatible_set.json`
- `runs/<RUN220>/s4_geometry_filter/stage_summary.json`
- `runs/<RUN221>/s4_geometry_filter/outputs/compatible_set.json`
- `runs/<RUN221>/s4_geometry_filter/stage_summary.json`

donde `<RUN220>` y `<RUN221>` salen de:

- `runs/batch_with_t0_220_eps2500_fixlen_20260304T160054Z/experiment/offline_batch/outputs/results.csv`
- `runs/batch_with_t0_221_eps2500_fixlen_20260304T160617Z/experiment/offline_batch/outputs/results.csv`

---

## 9) Comandos exactos de verificación rápida

### 9.1 Mostrar CSVs de salida de T6

```bash
BASE="runs/analysis_area_theorem_20260304T173747Z/experiment/area_theorem/outputs"
for f in per_event.csv per_event_spinmag.csv per_event_imrconsistency.csv per_event_map_only.csv; do
  echo "=== $f ==="
  sed -n '1,40p' "$BASE/$f"
done
```

### 9.2 Verificar pilot events con `missing=0`

```bash
python - <<'PY'
from pathlib import Path
root = Path('runs/analysis_area_theorem_20260304T173747Z/external_inputs/gwtc_posteriors')
pilot = [x.strip() for x in (root/'pilot_events.txt').read_text().splitlines() if x.strip()]
missing = [ev for ev in pilot if not (root/f'{ev}.json').exists()]
print('pilot_n=', len(pilot))
print('missing_n=', len(missing))
print('missing=', missing)
PY
```

Esperado: `missing_n=0`.

### 9.3 Listar H5 descargados en `raw/`

```bash
ls -1 runs/analysis_area_theorem_20260304T173747Z/external_inputs/gwtc_posteriors/raw/*.h5
```

---

## 10) Próximos pasos (enunciado)

Para “atacar GW200322” (tensión principal):

1. ponderar `Af_RD` por `delta_lnL`,
2. revisar ventana temporal de ringdown,
3. desaturar/ajustar tratamiento del modo 221,
4. repetir sensibilidad manteniendo control IMR-consistency como guardrail.

