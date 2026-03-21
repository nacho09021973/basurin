"""BRUNETE experimental alternatives — B5 catalog.

These modules are certified readers of BRUNETE classify runs.
They consume artifacts produced by brunete_classify_geometries.py
(and transitively, the per-event BASURIN subruns stored under
runs/<batch_run_id>/run_batch/event_runs/).

Modules:
    base_contract     — shared utilities and BRUNETE-aware run resolver
    b5f_verdict_aggregation      — population-level classification rates
    b5a_multi_event_aggregation  — intersection/union/Jaccard across events
    b5b_jackknife                — leave-one-out stability audit
    b5h_blind_prediction         — cross-event predictive power (LOO)
    b5c_ranking                  — geometry ranking per event
    b5e_query                    — reproducible query engine
    b5z_gpr_emulator             — Gaussian Process surface reconstruction
"""
