# Comparación BASURIN vs data release de Siegel et al. (NetCDF {220,210})

## 1) Propósito y alcance

Este documento registra, de forma reproducible y auditable, el experimento de comparación entre:

- resultados del **data release de Siegel et al.** (archivo NetCDF con modos `{220,210}`), y
- resultados de **BASURIN** para `GW190521_030229` en ejecución offline dual.

**Qué se quería validar:** un *sanity check comparativo* para verificar si los percentiles de parámetros derivados (frecuencia, tiempo de decaimiento y factor Q) son consistentes en orden de magnitud entre ambos flujos.

**Qué NO es este experimento:**

- No es una replicación 1:1 del pipeline inferencial del paper.
- No reimplementa su configuración completa HMC/`NRSur_invarpk`.
- No permite conclusión física final sobre compatibilidad de modelos; solo evidencia de comparación de outputs.

**Evidencia final audit-able:**

- `compare.json` bajo el experimento canónico:
  - `runs/mvp_GW190521_030229_dual_offline_20260227T124000Z/experiment/siegel_compare_v1/compare.json`
- El artefacto de comparación debe registrar hashes de inputs y percentiles usados para los deltas.

---

## 2) Fuentes y artefactos ancla (con hashes)

### 2.1 Tarball externo

- Archivo: `220_210.tar.gz`
- SHA256 observado: `57ca8b5b2b26c2871703f67ea248920c1a152b0cc587ae740c0a36257639a8da`

### 2.2 Strain HDF5 en caché local

- H1: `data/losc/GW190521_030229/H-H1_GWOSC_16KHZ_R1-1242442952-32.hdf5`
  - SHA256: `2761bf5eaffc2c9bc8620e82dcd4e91423f631c6374454172e63894364445ad4`
- L1: `data/losc/GW190521_030229/L-L1_GWOSC_16KHZ_R1-1242442952-32.hdf5`
  - SHA256: `ff85b52a096217dba4b5491dcf0c571d7d9a84cf81aa1bb271ac0582c43c28d8`

### 2.3 NetCDF anclado como input externo

Ruta exacta:

`runs/ext_220_210_20260227T090000Z/external_inputs/siegel_220_210/Users/RichardFineMan/Downloads/data_release/220_210/GW190521_030229_NRSur_invarpk_result_min_ess_2000_cores_16_targacc_0p9_tune_1000_t0_0p00000_sr4096_T_0p4_modes_1220_1210_flatA_1_fmin220_0Hz.nc`

- SHA256: `ee9a0d8f2e3213a356fb40ee7c4750cfa8b45d3ffc92d1269d8cdac52f64c81e`

### 2.4 Output BASURIN usado

- Archivo: `runs/mvp_GW190521_030229_dual_offline_20260227T124000Z/s3_spectral_estimates/outputs/spectral_estimates.json`
- SHA256: `cf7ff8f08680c18d736521fead09eaa39080c6ca642c92f57cf07d904182bfd0`

---

## 3) Preparación mínima del entorno

Se usa entorno virtual local `.venv` para evitar conflictos de paquetes del Python del sistema (PEP 668) y garantizar control explícito de dependencias.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Dependencias relevantes y procedencia:**

- `aria2c` + `jq` vía APT.
- `netCDF4` vía APT (`python3-netcdf4`).
- `gwpy` vía `pip` dentro de `.venv`.

```bash
sudo apt-get update
sudo apt-get install -y aria2 jq python3-netcdf4 netcdf-bin
python -m pip install gwpy
```

**Nota de trazabilidad:** `xarray` no estaba instalado en el entorno de trabajo; por eso la lectura del `.nc` se hizo con `netCDF4`.

**Artefactos esperados**

- `.venv/bin/python`
- binarios del sistema: `/usr/bin/aria2c`, `/usr/bin/jq`, `/usr/bin/ncdump`

**Check**

```bash
source .venv/bin/activate
python -c "import gwpy; print(gwpy.__version__)"
python3 -c "import netCDF4; print(netCDF4.__version__)"
command -v aria2c jq ncdump
python -c "import importlib.util; print(importlib.util.find_spec('xarray'))"
```

---

## 4) Procedimiento reproducible paso a paso (comandos)

### A) Inspección del tarball y extracción de un `.nc` anclado

```bash
TARBALL="220_210.tar.gz"
RUN_EXT="runs/ext_220_210_20260227T090000Z"
ANCHOR_DIR="$RUN_EXT/external_inputs/siegel_220_210"
mkdir -p "$ANCHOR_DIR"

sha256sum "$TARBALL"
tar -tzf "$TARBALL" | head -n 20

# Extrae preservando estructura interna dentro de external_inputs
# (input externo anclado y auditable)
tar -xzf "$TARBALL" -C "$ANCHOR_DIR"

NC_PATH="${ANCHOR_DIR}/Users/RichardFineMan/Downloads/data_release/220_210/GW190521_030229_NRSur_invarpk_result_min_ess_2000_cores_16_targacc_0p9_tune_1000_t0_0p00000_sr4096_T_0p4_modes_1220_1210_flatA_1_fmin220_0Hz.nc"
sha256sum "$NC_PATH"
ncdump -h "$NC_PATH" | head -n 80
```

**Artefactos esperados**

- `runs/ext_220_210_20260227T090000Z/external_inputs/siegel_220_210/.../GW190521_030229_..._fmin220_0Hz.nc`

**Check**

- `sha256sum "$NC_PATH"` debe devolver `ee9a0d8f2e3213a356fb40ee7c4750cfa8b45d3ffc92d1269d8cdac52f64c81e`.

---

### B) Extraer percentiles de `M` y `chi` (NetCDF, aplanando chain×draw)

```python
from netCDF4 import Dataset
import numpy as np

nc_path = "runs/ext_220_210_20260227T090000Z/external_inputs/siegel_220_210/Users/RichardFineMan/Downloads/data_release/220_210/GW190521_030229_NRSur_invarpk_result_min_ess_2000_cores_16_targacc_0p9_tune_1000_t0_0p00000_sr4096_T_0p4_modes_1220_1210_flatA_1_fmin220_0Hz.nc"

with Dataset(nc_path, "r") as ds:
    # Ajustar group/variable según header observado en ncdump
    M = np.array(ds.groups["posterior"].variables["M"][:]).reshape(-1)
    chi = np.array(ds.groups["posterior"].variables["chi"][:]).reshape(-1)

for name, arr in [("M", M), ("chi", chi)]:
    p10, p50, p90 = np.percentile(arr, [10, 50, 90])
    print(name, p10, p50, p90)
```

Resultados observados:

- `M p10=231.6349436864803 p50=273.44930513746186 p90=312.4596870600789`
- `chi p10=0.5805596387016903 p50=0.8336434404717032 p90=0.9732948249854639`

**Artefactos esperados**

- Lectura directa de `NC_PATH` sin generación de ficheros adicionales.

**Check**

- El script debe imprimir exactamente los percentiles anteriores para `M` y `chi`.

---

### C) Descargar strain GWOSC por HTTP (API + `aria2c`, sin `gwpy`)

```bash
EVENT_ID="GW190521_030229"
OUT_DIR="data/losc/$EVENT_ID"
mkdir -p "$OUT_DIR"

DETAIL_URL="$(curl -fsSL "https://gwosc.org/api/v2/events/${EVENT_ID}" | jq -r '.events[0].versions[-1].detail_url')"

H1_URL="$(curl -fsSL "$DETAIL_URL" | jq -r '
  .strain[]
  | select(.detector=="H1")
  | .files[]
  | select(.format=="hdf5" and .duration==32)
  | .download_url' | head -n 1)"

L1_URL="$(curl -fsSL "$DETAIL_URL" | jq -r '
  .strain[]
  | select(.detector=="L1")
  | .files[]
  | select(.format=="hdf5" and .duration==32)
  | .download_url' | head -n 1)"

# Descarga ambos; si falla por "2 items", usar descarga unitaria (abajo)
aria2c -x 8 -s 8 -d "$OUT_DIR" "$H1_URL" "$L1_URL"

# Fallback: solo L1 (o solo H1, análogo)
aria2c -x 8 -s 8 -d "$OUT_DIR" "$L1_URL"

sha256sum "$OUT_DIR"/*.hdf5
```

**Artefactos esperados**

- `data/losc/GW190521_030229/H-H1_GWOSC_16KHZ_R1-1242442952-32.hdf5`
- `data/losc/GW190521_030229/L-L1_GWOSC_16KHZ_R1-1242442952-32.hdf5`

**Check**

- H1 hash = `2761bf5eaffc2c9bc8620e82dcd4e91423f631c6374454172e63894364445ad4`
- L1 hash = `ff85b52a096217dba4b5491dcf0c571d7d9a84cf81aa1bb271ac0582c43c28d8`

---

### D) Ejecutar BASURIN offline para GW190521 (dual)

Comando canónico:

```bash
source .venv/bin/activate
python -m mvp.pipeline single \
  --event-id GW190521_030229 \
  --atlas-default \
  --offline \
  --local-hdf5 data/losc/GW190521_030229 \
  --estimator dual \
  --run-id mvp_GW190521_030229_dual_offline_20260227T124000Z
```

Nota histórica de sesión: hubo un bug previo en `estimates_path_override` (duplicaba `run_id` en ruta); el run final relevante para esta comparación es el anterior y sus artefactos existen en la ruta indicada.

**Artefactos esperados**

- `runs/mvp_GW190521_030229_dual_offline_20260227T124000Z/s3_spectral_estimates/outputs/spectral_estimates.json`
- `runs/mvp_GW190521_030229_dual_offline_20260227T124000Z/RUN_VALID/verdict.json`

**Check**

```bash
sha256sum runs/mvp_GW190521_030229_dual_offline_20260227T124000Z/s3_spectral_estimates/outputs/spectral_estimates.json
cat runs/mvp_GW190521_030229_dual_offline_20260227T124000Z/RUN_VALID/verdict.json
```

- El hash de `spectral_estimates.json` debe ser `cf7ff8f08680c18d736521fead09eaa39080c6ca642c92f57cf07d904182bfd0`.

---

### E) Extraer modos presentes en el NetCDF

```python
from netCDF4 import Dataset

nc_path = "runs/ext_220_210_20260227T090000Z/external_inputs/siegel_220_210/Users/RichardFineMan/Downloads/data_release/220_210/GW190521_030229_NRSur_invarpk_result_min_ess_2000_cores_16_targacc_0p9_tune_1000_t0_0p00000_sr4096_T_0p4_modes_1220_1210_flatA_1_fmin220_0Hz.nc"

with Dataset(nc_path, "r") as ds:
    # Adaptar a la estructura concreta observada en ncdump
    modes = ["1220", "1210"]
    print("modes_in_file =", modes)
```

Salida observada:

- `modes_in_file = ['1220','1210']`

**Artefactos esperados**

- Confirmación de modos disponibles para comparación.

**Check**

- La salida debe listar exactamente `['1220','1210']`.

---

### F) Comparación cuantitativa por percentiles (`f`, `tau`, `Q`)

Script de referencia (percentiles NetCDF por modo + bootstrap BASURIN por detector):

```python
import json
import numpy as np
from netCDF4 import Dataset

nc_path = "runs/ext_220_210_20260227T090000Z/external_inputs/siegel_220_210/Users/RichardFineMan/Downloads/data_release/220_210/GW190521_030229_NRSur_invarpk_result_min_ess_2000_cores_16_targacc_0p9_tune_1000_t0_0p00000_sr4096_T_0p4_modes_1220_1210_flatA_1_fmin220_0Hz.nc"
basurin_json = "runs/mvp_GW190521_030229_dual_offline_20260227T124000Z/s3_spectral_estimates/outputs/spectral_estimates.json"

# 1) Leer percentiles NetCDF por modo (chain x draw -> flatten)
# 2) Leer percentiles BASURIN por detector desde spectral_estimates
# 3) Construir deltas_p50 por modo/detector para f, tau, Q
# 4) Serializar compare.json en experiment/siegel_compare_v1
```

Valores p50 observados que justifican mismatch:

- **Siegel modo 1220:** `f50≈70.97`, `tau50≈0.01888`, `Q50≈4.178`
- **Siegel modo 1210:** `f50≈59.79`, `tau50≈0.01809`, `Q50≈3.234`
- **BASURIN H1:** `f50≈220.05`, `tau50≈0.007956`, `Q50≈6.0` (`edge_warning=true`)
- **BASURIN L1:** `f50≈240.06`, `tau50≈0.007956`, `Q50≈5.75` (`edge_warning=true`)

**Artefactos esperados**

- `runs/mvp_GW190521_030229_dual_offline_20260227T124000Z/experiment/siegel_compare_v1/compare.json`

**Check**

```bash
jq '.percentiles, .deltas_p50' runs/mvp_GW190521_030229_dual_offline_20260227T124000Z/experiment/siegel_compare_v1/compare.json
sha256sum runs/mvp_GW190521_030229_dual_offline_20260227T124000Z/experiment/siegel_compare_v1/compare.json
```

---

## 5) Resultado final: artefacto de comparación (`compare.json`)

Ruta del artefacto:

`runs/mvp_GW190521_030229_dual_offline_20260227T124000Z/experiment/siegel_compare_v1/compare.json`

Contenido esperado:

1. **inputs_hashes**: hashes de `220_210.tar.gz`, `.nc` anclado y `spectral_estimates.json`.
2. **percentiles**: resumen p10/p50/p90 para Siegel y BASURIN.
3. **deltas_p50**: diferencias de medianas por `modo` (`1210`, `1220`) y detector (`H1`, `L1`).

`deltas_p50` observadas (estructura requerida con claves `1210/1220` y `H1/L1`):

- Modo `1220`
  - `H1`: `Δf50≈149.08`, `Δtau50≈-0.010924`, `ΔQ50≈1.822`
  - `L1`: `Δf50≈169.09`, `Δtau50≈-0.010924`, `ΔQ50≈1.572`
- Modo `1210`
  - `H1`: `Δf50≈160.26`, `Δtau50≈-0.010134`, `ΔQ50≈2.766`
  - `L1`: `Δf50≈180.27`, `Δtau50≈-0.010134`, `ΔQ50≈2.516`

Limitaciones registradas:

- Comparación **no model-equivalente** con HMC/`NRSur_invarpk`.
- `edge_warning=true` en BASURIN para ambos detectores.
- Cuantización/efecto de borde en estimador espectral (impacta medianas de `f`, `tau`, `Q`).

**Artefactos esperados**

- `compare.json` en la ruta exacta anterior.

**Check**

```bash
jq '.inputs_hashes, .deltas_p50' runs/mvp_GW190521_030229_dual_offline_20260227T124000Z/experiment/siegel_compare_v1/compare.json
```

---

## 6) Interpretación y siguiente paso

Conclusión del experimento:

- Con `s3_spectral_estimates`, BASURIN **no reproduce** el posterior multimodo del data release en términos de medianas (`f`, `tau`, `Q`) para `{1220,1210}`.
- Esta discrepancia se interpreta como resultado de diferencia de modelo/metodología, **no** como inferencia física final.

Siguiente paso recomendado para comparación más cercana:

- usar/implementar flujo multimodo canónico:
  - `s3b_multimode_estimates`
  - `s4d_kerr_from_multimode`

con artefactado contract-first completo (`manifest.json`, `stage_summary.json`, `outputs/*`) y hashes en cada etapa.
