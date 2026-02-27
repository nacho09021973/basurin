"""Centralized contract definitions for the MVP pipeline.

Single source of truth for:
  - What each stage requires (inputs) and produces (outputs).
  - Common init/finalize/abort logic (no boilerplate per stage).
  - SHA256 hashing and validation of all artifacts.
  - Deterministic IO rules (writes only under runs/<run_id>/).

Usage in a stage:
    from mvp.contracts import CONTRACTS, init_stage, check_inputs, finalize, abort

    ctx = init_stage("my_run", "s1_fetch_strain", params={...})
    inputs = check_inputs(ctx, {"strain": path_to_strain})
    # ... do work ...
    finalize(ctx, artifacts={"result": output_path}, results={...})

    # On error:
    abort(ctx, "reason for failure")
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- BASURIN import bootstrap ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import (
    assert_no_symlink_ancestors,
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)

EXIT_CONTRACT_FAIL = 2


# ── Contract Definitions ──────────────────────────────────────────────────

@dataclass(frozen=True)
class StageContract:
    """Declares what a stage needs and produces."""
    name: str
    required_inputs: list[str]     # Paths relative to run_dir (upstream outputs)
    produced_outputs: list[str]    # Paths relative to stage_dir
    upstream_stages: list[str]     # Stages that must have PASS verdict
    check_run_valid: bool = True   # Whether to check RUN_VALID before running

    def input_paths(self, run_dir: Path) -> dict[str, Path]:
        """Resolve required_inputs to absolute paths under run_dir."""
        return {p: run_dir / p for p in self.required_inputs}

    def output_paths(self, stage_dir: Path) -> dict[str, Path]:
        """Resolve produced_outputs to absolute paths under stage_dir."""
        return {p: stage_dir / p for p in self.produced_outputs}


CONTRACTS: dict[str, StageContract] = {
    "s0_oracle_mvp": StageContract(
        name="s0_oracle_mvp",
        required_inputs=[],
        produced_outputs=[
            "outputs/oracle_metrics.json",
        ],
        upstream_stages=[],
        check_run_valid=False,
    ),
    "s1_fetch_strain": StageContract(
        name="s1_fetch_strain",
        required_inputs=[],
        produced_outputs=[
            "outputs/strain.npz",
            "outputs/provenance.json",
        ],
        upstream_stages=[],
        check_run_valid=False,  # s1 is the first stage after RUN_VALID
    ),
    "s2_ringdown_window": StageContract(
        name="s2_ringdown_window",
        required_inputs=[
            "s1_fetch_strain/outputs/strain.npz",
        ],
        produced_outputs=[
            "outputs/H1_rd.npz",
            "outputs/L1_rd.npz",
            "outputs/window_meta.json",
        ],
        upstream_stages=["s1_fetch_strain"],
    ),
    "s3_ringdown_estimates": StageContract(
        name="s3_ringdown_estimates",
        required_inputs=[
            # Dynamic: discovers {H1,L1,V1}_rd.npz at runtime
        ],
        produced_outputs=[
            "outputs/estimates.json",
        ],
        upstream_stages=["s2_ringdown_window"],
    ),
    "s4_geometry_filter": StageContract(
        name="s4_geometry_filter",
        required_inputs=[
            # Dynamic: estimates can be overridden via --estimates-path.
            # Runtime validates and hashes the actual consumed file.
        ],
        produced_outputs=[
            "outputs/compatible_set.json",
        ],
        upstream_stages=["s3_ringdown_estimates"],
    ),
    "s4_spectral_geometry_filter": StageContract(
        name="s4_spectral_geometry_filter",
        required_inputs=[
            # Dynamic: estimates can be overridden via --estimates-path.
            # Runtime validates and hashes the actual consumed file.
        ],
        produced_outputs=[
            "outputs/compatible_set.json",
        ],
        upstream_stages=["s3_ringdown_estimates"],
    ),
    "s5_aggregate": StageContract(
        name="s5_aggregate",
        required_inputs=[],  # Dynamic: depends on source_runs list
        produced_outputs=[
            "outputs/aggregate.json",
        ],
        upstream_stages=[],  # Checks source runs, not same-run upstream
        check_run_valid=False,  # Creates its own RUN_VALID
    ),
    "s6_information_geometry": StageContract(
        name="s6_information_geometry",
        required_inputs=[
            "s4_geometry_filter/outputs/compatible_set.json",
            # Dynamic: estimates can be overridden via --estimates-path.
            # Runtime validates and hashes the actual consumed file.
        ],
        produced_outputs=[
            "outputs/curvature.json",
            "outputs/metric_diagnostics.json",
        ],
        upstream_stages=["s3_ringdown_estimates", "s4_geometry_filter"],
    ),
    "s6b_information_geometry_3d": StageContract(
        name="s6b_information_geometry_3d",
        required_inputs=[
            "s3_ringdown_estimates/outputs/estimates.json",
            "s3b_multimode_estimates/outputs/multimode_estimates.json",
            "s4_geometry_filter/outputs/compatible_set.json",
        ],
        produced_outputs=[
            "outputs/curvature_3d.json",
            "outputs/metric_diagnostics_3d.json",
            "outputs/coords_3d.json",
        ],
        upstream_stages=["s3_ringdown_estimates", "s3b_multimode_estimates", "s4_geometry_filter"],
    ),
    "s6b_information_geometry_ranked": StageContract(
        name="s6b_information_geometry_ranked",
        required_inputs=[
            "s4_geometry_filter/outputs/compatible_set.json",
            "s6_information_geometry/outputs/curvature.json",
        ],
        produced_outputs=[
            "outputs/ranked_geometries.json",
        ],
        upstream_stages=["s4_geometry_filter", "s6_information_geometry"],
    ),
    "s6c_brunete_psd_curvature": StageContract(
        name="s6c_brunete_psd_curvature",
        required_inputs=[
            "s3_ringdown_estimates/outputs/estimates.json",
            # Prefer canonical upstream PSD produced by helper extract_psd.py.
            # Runtime fallback is external_inputs/psd_model.json.
            # (contract-first: explicit path documented in BRUNETE_S6C.md)
            "psd/measured_psd.json",
        ],
        produced_outputs=[
            "outputs/brunete_metrics.json",
            "outputs/psd_derivatives.json",
            "stage_summary.json",
            "manifest.json",
        ],
        upstream_stages=["s3_ringdown_estimates"],
        check_run_valid=True,
    ),
    "s6c_population_geometry": StageContract(
        name="s6c_population_geometry",
        required_inputs=[
            "s5_aggregate/outputs/aggregate.json",
        ],
        produced_outputs=[
            "outputs/population_scalar_claim.json",
            "outputs/population_diagnostics.json",
        ],
        upstream_stages=["s5_aggregate"],
        check_run_valid=True,
    ),
    "s4b_spectral_curvature": StageContract(
        name="s4b_spectral_curvature",
        required_inputs=[
            "s3_ringdown_estimates/outputs/estimates.json",
            # atlas_path is external, checked separately
        ],
        produced_outputs=[
            "outputs/spectral_diagnostics.json",
        ],
        upstream_stages=["s3_ringdown_estimates"],
    ),
    "s3b_multimode_estimates": StageContract(
        name="s3b_multimode_estimates",
        required_inputs=[
            "s3_ringdown_estimates/outputs/estimates.json",
            # Dynamic: discovers {H1,L1,V1}_rd.npz at runtime (like s3)
        ],
        produced_outputs=[
            "outputs/multimode_estimates.json",
            "outputs/model_comparison.json",
        ],
        upstream_stages=["s2_ringdown_window", "s3_ringdown_estimates"],
    ),
    "s4c_kerr_consistency": StageContract(
        name="s4c_kerr_consistency",
        required_inputs=[
            "s3_ringdown_estimates/outputs/estimates.json",
            "s3b_multimode_estimates/outputs/multimode_estimates.json",
        ],
        produced_outputs=[
            "outputs/kerr_consistency.json",
        ],
        upstream_stages=["s3_ringdown_estimates", "s3b_multimode_estimates"],
    ),
    "s4d_kerr_from_multimode": StageContract(
        name="s4d_kerr_from_multimode",
        required_inputs=[
            "s3b_multimode_estimates/outputs/multimode_estimates.json",
        ],
        produced_outputs=[
            "outputs/kerr_from_multimode.json",
            "outputs/kerr_from_multimode_diagnostics.json",
        ],
        upstream_stages=["s3b_multimode_estimates"],
    ),
    "s3_spectral_estimates": StageContract(
        name="s3_spectral_estimates",
        required_inputs=[
            # Dynamic: discovers {H1,L1,V1}_rd.npz at runtime (same as s3)
        ],
        produced_outputs=[
            "outputs/spectral_estimates.json",
        ],
        upstream_stages=["s2_ringdown_window"],
    ),
    "experiment_geometry_evidence_vs_gr": StageContract(
        name="experiment_geometry_evidence_vs_gr",
        required_inputs=[
            "s4_geometry_filter/outputs/compatible_set.json",
            "s6_information_geometry/outputs/curvature.json",
        ],
        produced_outputs=[
            "outputs/evidence_vs_gr.json",
        ],
        upstream_stages=["s4_geometry_filter", "s6_information_geometry"],
        check_run_valid=True,
    ),
}


# ── Stage Context ─────────────────────────────────────────────────────────

@dataclass
class StageContext:
    """Mutable context object passed through a stage's lifecycle."""
    run_id: str
    stage_name: str
    contract: StageContract
    out_root: Path
    run_dir: Path
    stage_dir: Path
    outputs_dir: Path
    params: dict[str, Any] = field(default_factory=dict)
    inputs_record: list[dict[str, str]] = field(default_factory=list)


# ── API Functions ─────────────────────────────────────────────────────────

def init_stage(
    run_id: str,
    stage_name: str,
    *,
    params: dict[str, Any] | None = None,
) -> StageContext:
    """Initialize a stage: resolve paths, validate run_id, check RUN_VALID.

    Returns a StageContext for use throughout the stage.
    Raises SystemExit(2) on contract violation.
    """
    if stage_name not in CONTRACTS:
        _fatal(f"Unknown stage: {stage_name}. Registered: {list(CONTRACTS.keys())}")

    contract = CONTRACTS[stage_name]
    out_root = resolve_out_root("runs")
    validate_run_id(run_id, out_root)
    run_dir = out_root / run_id

    try:
        assert_no_symlink_ancestors(run_dir)
    except RuntimeError as exc:
        _fatal(f"[{stage_name}] {exc}")

    if contract.check_run_valid:
        try:
            require_run_valid(out_root, run_id)
        except Exception as exc:
            _fatal(f"[{stage_name}] RUN_VALID check failed: {exc}")

    stage_dir, outputs_dir = ensure_stage_dirs(run_id, stage_name, base_dir=out_root)

    return StageContext(
        run_id=run_id,
        stage_name=stage_name,
        contract=contract,
        out_root=out_root,
        run_dir=run_dir,
        stage_dir=stage_dir,
        outputs_dir=outputs_dir,
        params=params or {},
    )


def check_inputs(
    ctx: StageContext,
    paths: dict[str, Path],
    *,
    optional: dict[str, Path] | None = None,
) -> list[dict[str, str]]:
    """Validate that all required input files exist and compute SHA256.

    Args:
        ctx: Stage context.
        paths: Required inputs as {label: path}.
        optional: Optional inputs (recorded if present, not fatal if missing).

    Returns:
        List of input records [{path, sha256, label}] stored in ctx.inputs_record.

    Raises SystemExit(2) if any required input is missing.
    """
    records: list[dict[str, str]] = []
    missing: list[str] = []

    for label, path in paths.items():
        if not path.exists():
            missing.append(f"{label}: {path}")
            records.append({"label": label, "path": str(path), "sha256": ""})
        else:
            try:
                rel = str(path.relative_to(ctx.run_dir))
            except ValueError:
                rel = str(path)
            records.append({"label": label, "path": rel, "sha256": sha256_file(path)})

    if optional:
        for label, path in optional.items():
            if path.exists():
                try:
                    rel = str(path.relative_to(ctx.run_dir))
                except ValueError:
                    rel = str(path)
                records.append({"label": label, "path": rel, "sha256": sha256_file(path)})

    if missing:
        abort(ctx, f"Missing required inputs: {'; '.join(missing)}")

    ctx.inputs_record = records
    return records


def finalize(
    ctx: StageContext,
    artifacts: dict[str, Path],
    *,
    verdict: str = "PASS",
    results: dict[str, Any] | None = None,
    extra_summary: dict[str, Any] | None = None,
) -> None:
    """Write manifest.json + stage_summary.json and print OK.

    Args:
        ctx: Stage context.
        artifacts: {label: path} of produced files (will be hashed in manifest).
        verdict: "PASS" or other verdict string.
        results: Optional results dict to embed in stage_summary.
        extra_summary: Additional fields to merge into stage_summary.
    """
    # Build outputs list
    outputs_list: list[dict[str, str]] = []
    for label, path in artifacts.items():
        if label == "stage_summary":
            continue
        try:
            rel = str(path.relative_to(ctx.run_dir))
        except ValueError:
            rel = str(path)
        outputs_list.append({"path": rel, "sha256": sha256_file(path)})

    final_verdict = verdict
    summary_error: str | None = None
    if verdict == "PASS" and not outputs_list:
        final_verdict = "FAIL"
        summary_error = "PASS_WITHOUT_OUTPUTS"

    # Write stage_summary.json
    summary: dict[str, Any] = {
        "stage": ctx.stage_name,
        "run": ctx.run_id,
        "runs_root": str(ctx.out_root),
        "created": utc_now_iso(),
        "version": "v1",
        "parameters": ctx.params,
        "inputs": ctx.inputs_record,
        "outputs": outputs_list,
        "verdict": final_verdict,
    }
    if summary_error is not None:
        summary["error"] = summary_error
    if results:
        summary["results"] = results
    if extra_summary:
        summary.update(extra_summary)

    sp = write_stage_summary(ctx.stage_dir, summary)
    artifacts["stage_summary"] = sp

    # Write manifest.json
    manifest_extra: dict[str, Any] = {"inputs": ctx.inputs_record}
    if summary_error is not None:
        manifest_extra.update({"verdict": "FAIL", "error": summary_error})
    write_manifest(ctx.stage_dir, artifacts, extra=manifest_extra)

    print(f"OK: {ctx.stage_name} {final_verdict}")


def abort(ctx: StageContext, reason: str) -> None:
    """Write FAIL summary + manifest and exit with code 2.

    Never returns — always raises SystemExit(2).
    """
    summary: dict[str, Any] = {
        "stage": ctx.stage_name,
        "run": ctx.run_id,
        "runs_root": str(ctx.out_root),
        "created": utc_now_iso(),
        "version": "v1",
        "parameters": ctx.params,
        "inputs": ctx.inputs_record,
        "outputs": [],
        "verdict": "FAIL",
        "error": reason,
    }
    sp = write_stage_summary(ctx.stage_dir, summary)
    write_manifest(
        ctx.stage_dir,
        {"stage_summary": sp},
        extra={"verdict": "FAIL", "error": reason},
    )
    _fatal(f"[{ctx.stage_name}] {reason}")


def enforce_outputs(ctx: StageContext) -> list[str]:
    """Verify that all declared produced_outputs exist after finalize.

    Returns list of missing outputs (empty if all present).
    """
    missing: list[str] = []
    for rel in ctx.contract.produced_outputs:
        path = ctx.stage_dir / rel
        if not path.exists():
            missing.append(rel)
    return missing


# ── Internal ──────────────────────────────────────────────────────────────

def _fatal(message: str) -> None:
    """Print error and exit. Never returns."""
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)
