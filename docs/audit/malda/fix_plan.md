# Plan de correcciones — `malda/` (post-auditoría)

**Basado en:** `audit_report.md` (2026-02-20)
**Rama de trabajo:** `claude/audit-malda-scripts-AJX0G`

---

## Prioridades

| # | Problema | Archivos afectados | Severidad | Fase |
|---|----------|-------------------|-----------|------|
| 1 | `stage_utils` no existe → `TypeError` en 8+ scripts | 8 scripts | ALTA | 1 |
| 2 | `04b` placeholder aceptable en pipelines automatizados | `04b_negative_control_contracts.py` | ALTA | 2 |
| 3 | `PROJECT_ROOT` apunta a `malda/` en vez de raíz del repo | 3 scripts IO | MEDIA | 3 |
| 4 | `global np, gamma_func` en `01_generate_sandbox_geometries.py` | 1 script | MEDIA | 4 |
| 5 | `np.random.seed()` API legacy en `04d_negative_hawking.py` | 1 script | BAJA | 4 |
| 6 | `fs=0` sin validación en `00_load_ligo_data.py` | 1 script | BAJA | 4 |
| 7 | `--mass` sin efecto en `04c_negative_controls.py` | 1 script | BAJA | 4 |
| 8 | Encoding mojibake en comentarios | ~4 scripts | INFO | 5 |

---

## Fase 1 — Crear `malda/stage_utils.py` (fundación)

**Por qué primero:** Sin este módulo, los 8 scripts que lo importan fallan con `TypeError`
al llamar `add_standard_arguments(None)` / `parse_stage_args(None)` si `stage_utils`
no está en el path. Es la deuda técnica de mayor impacto.

### Interfaz requerida (inferida del uso en los scripts)

```python
# Constantes
EXIT_OK = 0
EXIT_ERROR = 3
STATUS_OK = "OK"
STATUS_ERROR = "ERROR"

# Funciones
def add_standard_arguments(parser: argparse.ArgumentParser) -> None:
    """Añade --experiment y --runs-dir al parser."""

def parse_stage_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Llama parser.parse_args() con validaciones comunes."""

def infer_experiment() -> str:
    """Infiere nombre de experimento desde env CUERDAS_EXPERIMENT o devuelve 'default'."""

# Clase
class StageContext:
    experiment: str        # nombre del experimento/run
    run_root: Path         # runs/<experiment>/
    stage_dir: Path        # runs/<experiment>/<stage_number>_<stage_slug>/

    @classmethod
    def from_args(cls, args, stage_number: str, stage_slug: str) -> "StageContext": ...

    def record_artifact(self, key_or_path, path=None) -> None:
        """Registra un artefacto (por clave+ruta o solo ruta)."""

    def write_manifest(self, outputs=None, metadata=None) -> None:
        """Escribe <stage_dir>/manifest.json."""

    def write_summary(self, status, exit_code=0, error_message=None, counts=None) -> None:
        """Escribe <stage_dir>/stage_summary.json."""
```

### Estructura del archivo `malda/stage_utils.py`

```
malda/
  stage_utils.py   ← NUEVO (≈ 120-150 líneas)
```

### Detalles de implementación

**`add_standard_arguments(parser)`:**
- `--experiment` (str, default: valor de `CUERDAS_EXPERIMENT` o `"default"`)
- `--runs-dir` (str, default: `"runs"`) — directorio raíz donde viven los runs

**`parse_stage_args(parser)`:**
- Llama `parser.parse_args()` y devuelve el resultado.
- En el futuro: podría añadir validaciones comunes.

**`StageContext.from_args(args, stage_number, stage_slug)`:**
- `run_root = resolve_runs_dir(args.runs_dir) / args.experiment`
- `stage_dir = run_root / f"{stage_number}_{stage_slug}"`
- Crea `stage_dir` con `mkdir(parents=True, exist_ok=True)`.

**`record_artifact(key_or_path, path=None)`:**
- Acumula en `self._artifacts: dict`.
- Si se pasa solo `path` (Path), usar `path.name` como clave.
- Si se pasa `(key: str, path: Path)`, usar esa clave.

**`write_manifest(outputs, metadata)`:**
- Escribe `stage_dir/manifest.json` con `created_at`, `experiment`, `stage`, `outputs`, `metadata`, `artifacts`.
- Escritura atómica (tmp → `os.replace`).

**`write_summary(status, exit_code, error_message, counts)`:**
- Escribe `stage_dir/stage_summary.json`.
- Escritura atómica.

**`PROJECT_ROOT`** dentro de `stage_utils`:
- `REPO_ROOT = Path(__file__).resolve().parent.parent` — apunta a la raíz del repo (`basurin/`).
- Esta constante puede ser exportada para que los scripts la usen.

---

## Fase 2 — Guardar `04b` en modo placeholder

**Archivo:** `malda/04b_negative_control_contracts.py`

**Cambio:** Añadir al inicio del `if __name__ == "__main__":` (o en `main()` si existe):

```python
if PLACEHOLDER_MODE:
    import sys
    print(
        "[04b] ERROR: Este script está en MODO PLANTILLA (placeholder).\n"
        "  Sus métricas son stubs y no deben usarse para claims científicos.\n"
        "  Cambia PLACEHOLDER_MODE = False solo cuando las métricas reales estén conectadas.",
        file=sys.stderr,
    )
    sys.exit(2)
```

- Exit code `2` (distinto de `EXIT_ERROR=3`) para distinguirlo de errores normales.
- Si hay llamadas programáticas (import), el flag `PLACEHOLDER_MODE` en el módulo
  ya es suficiente advertencia; el `sys.exit` solo aplica a ejecución directa.

---

## Fase 3 — Unificar `PROJECT_ROOT`

**Archivos afectados:** `00_load_ligo_data.py`, `01_extract_ringdown_poles.py`,
`02R_build_ringdown_boundary_dataset.py`.

**Cambio:** Reemplazar el cálculo local de `project_root` / `PROJECT_ROOT` por una
importación desde `stage_utils`:

```python
# Antes (en cada script):
project_root = Path(__file__).resolve().parent   # → malda/

# Después:
from stage_utils import REPO_ROOT
project_root = REPO_ROOT   # → basurin/
```

**Nota de compatibilidad:** La verificación de escape en `02R` (`resolved.relative_to(PROJECT_ROOT)`)
pasará a validar rutas relativas a la raíz del repo, no a `malda/`. Esto es más permisivo
y correcto: permite referir archivos en cualquier parte del repositorio.

---

## Fase 4 — Correcciones menores

### 4a — `global np, gamma_func` en `01_generate_sandbox_geometries.py`

**Líneas:** ~1071-1073

**Problema:** `global np, gamma_func` dentro de un `try` en `main()` es redundante
e inútil; `gamma_func` nunca se usa.

**Cambio:**
- Eliminar la línea `global np, gamma_func`.
- Eliminar `from scipy.special import gamma as gamma_func` si no se usa en ningún otro lugar del fichero.
- `import h5py` puede permanecer dentro del try/except para dar error claro si falta.

### 4b — `np.random.seed()` legacy en `04d_negative_hawking.py`

**Línea:** `np.random.seed(seed)` en `generate_hawking_negative_data`.

**Cambio:** Reemplazar con generador moderno:
```python
rng = np.random.default_rng(seed)
# Luego usar rng.normal(...) en lugar de np.random.normal(...)
```

Esto garantiza reproducibilidad aislada (el RNG no afecta el estado global de NumPy).

### 4c — Validación `fs=0` en `00_load_ligo_data.py`

**Función:** `_load_npz`

**Cambio:** Añadir validación tras cargar `fs`:
```python
if float(_scalar("fs")) <= 0:
    raise ValueError(f"fs must be > 0 in {npz_path}; got fs={float(_scalar('fs'))}")
```

### 4d — Deprecar `--mass` sin efecto en `04c_negative_controls.py`

La función genera ruido blanco independientemente del valor de `--mass`.

**Cambio:** Añadir en el argparser:
```python
parser.add_argument(
    "--mass",
    type=float,
    default=1.0,
    help="[IGNORADO] Parámetro legacy. Este script genera ruido blanco; mass no tiene efecto.",
)
```

Y emitir un warning al inicio de `generate_massive_scalar_flat_space`:
```python
import warnings
warnings.warn("--mass no tiene efecto: este script genera ruido blanco gaussiano puro.", stacklevel=2)
```

---

## Fase 5 — Encoding de comentarios (baja prioridad)

**Archivos:** `01_generate_sandbox_geometries.py`, `02_emergent_geometry_engine.py`,
`04_geometry_physics_contracts.py`, `08_build_holographic_dictionary.py`.

**Cambio:** Re-guardar cada archivo con UTF-8 correcto. Muchos comentarios contienen
caracteres como `ÃƒÆ'Ã†â€™` que son UTF-8 mal decodificado desde Latin-1.

**Proceso sugerido:**
```bash
# Detectar encoding actual:
file -i malda/01_generate_sandbox_geometries.py

# Re-codificar si es necesario:
iconv -f latin1 -t utf-8 <script> -o <script>.fixed && mv <script>.fixed <script>
```

Esto es cosmético y puede hacerse en un commit separado con mensaje claro.

---

## Orden de commits propuesto

```
feat(malda): add stage_utils.py with StageContext, add_standard_arguments, parse_stage_args
fix(04b): block execution in placeholder mode with clear error and exit code 2
fix(routing): use REPO_ROOT from stage_utils in 00, 01R, 02R scripts
fix(01-sandbox): remove global np/gamma_func, drop unused gamma import
fix(04d): replace np.random.seed() with default_rng for reproducibility
fix(00-load): validate fs > 0 before use
fix(04c): deprecate --mass param with warning
chore(encoding): fix mojibake in Spanish comments across malda/ scripts
```

---

## Tests recomendados a añadir

Una vez aplicadas las correcciones, es recomendable añadir tests mínimos:

| Test | Ubicación | Qué verifica |
|------|-----------|-------------|
| `test_stage_utils_context.py` | `tests/` | `StageContext.from_args` crea dirs; `write_summary` es atómico |
| `test_stage_utils_root.py` | `tests/` | `REPO_ROOT` == raíz del repo (tiene `README.md`) |
| `test_04b_blocks_in_placeholder.py` | `tests/` | `subprocess.run(["python", "04b..."])` devuelve exit code 2 |
| `test_00_load_fs_validation.py` | `tests/` | `_load_npz` con `fs=0` lanza `ValueError` |

---

## Estimación de cambios

| Fase | Archivos nuevos | Archivos modificados | Líneas aprox. |
|------|----------------|---------------------|---------------|
| 1 | 1 (`stage_utils.py`) | 0 | +140 |
| 2 | 0 | 1 | +8 |
| 3 | 0 | 3 | +3 / -3 (net 0) |
| 4 | 0 | 4 | +15 total |
| 5 | 0 | 4 | cosmético |
| **Total** | **1** | **12** | **~165 líneas netas** |
