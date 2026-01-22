# IO Contracts (Basurin)

## Run root
All outputs MUST live under:

runs/<run_id>/

## Mandatory files per stage
Each stage MUST write:

- runs/<run_id>/<stage>/stage_summary.json
- runs/<run_id>/<stage>/manifest.json

## Manifest schema (minimal)
`manifest.json` MUST include:

- "stage": <stage_name>
- "run": <run_id>
- "created": ISO8601 (UTC)
- "version": <string>
- "files": { <label>: <relative_path_under_stage_dir> }

All artifact paths MUST be relative to the stage directory
(`runs/<run_id>/<stage>/`).

## Data outputs
All data files MUST be placed under:

runs/<run_id>/<stage>/outputs/

No stage may write outside `runs/<run_id>/`.

## bridge_f4_1_alignment outputs
`degeneracy_per_point.json` entries include audit fields for pairing:

- pairing_policy
- paired_by
- atlas_id
- event_id
- row_i (index for order pairing)
