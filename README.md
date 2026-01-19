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
