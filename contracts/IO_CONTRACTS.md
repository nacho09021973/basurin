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

## C7 gate (bridge/alignment)
Inputs:
- runs/<run_id>/<bridge_stage>/outputs/metrics.json
- Optional: knn_preservation_real.json, knn_preservation_negative.json,
  knn_preservation_control_positive.json

Outputs:
- runs/<run_id>/C7/outputs/c7_report.json
- runs/<run_id>/C7/stage_summary.json
- runs/<run_id>/C7/manifest.json

Verdicts:
- PASS: corr/stability pass, degeneracy below max_deg_pass.
- DEGENERATE: corr/stability pass, degeneracy above degenerate_deg_median.
- FAIL: leakage detected via negative ratio or permutation thresholds.
- UNDERDETERMINED: missing or ambiguous inputs, or evidence does not meet any rule.
