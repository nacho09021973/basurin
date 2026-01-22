# BASURIN

Pipeline mínimo, completo, reproducible y auditable para explorar emergencia geométrica y diccionarios holográficos guiados por datos (enfoque “honesto y falsable”). En verdad es una basura por ahora, no pierdas el tiempo hasta que esto no avance.

## Estructura de IO (determinista)

Cada ejecución vive en:

runs/<run_id>/

Cada etapa escribe como mínimo:

- runs/<run_id>/<stage>/manifest.json
- runs/<run_id>/<stage>/stage_summary.json
- runs/<run_id>/<stage>/outputs/

El pipeline no debe escribir fuera de runs/<run_id>/.

## IO / Runs / Paths

Guía operativa y normativa de rutas, layouts y resolvers:

- [readme_rutas.md](readme_rutas.md)

## Quickstart (WSL/Linux)

Requisitos: Python 3.11+.

```bash
cd ~/basurin/work
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Si existe requirements.txt:
# pip install -r requirements.txt
# Si existe pyproject.toml:
# pip install -e .
```

## Runner v1 (pipeline mínimo 01→02b→03→04→RUN_VALID)

Ejecuta el runner canónico desde `tools/run_v1.py`:

```bash
python tools/run_v1.py --run <run_id>
```
