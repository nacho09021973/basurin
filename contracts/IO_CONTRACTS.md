# IO Contracts (Basurin)

## Run root
All outputs MUST live under:

runs/<experiment>/<stage>/

## Mandatory files per stage
Each stage MUST write:

- runs/<experiment>/<stage>/stage_summary.json
- runs/<experiment>/<stage>/manifest.json

## Manifest schema (minimal)
`manifest.json` MUST include:

- "stage": <stage_name>
- "experiment": <experiment>
- "created_utc": ISO8601
- "artifacts": { <artifact_name>: <relative_path_under_run_root> }

All artifact paths MUST be relative to `runs/<experiment>/`.

## Data outputs
All data files MUST be placed under:

runs/<experiment>/<stage>/outputs/

No stage may write outside `runs/<experiment>/`.
