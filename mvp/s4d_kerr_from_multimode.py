"""
s4d_kerr_from_multimode

Canonical stage (Phase B):
- Inputs: s3b multimode outputs (multimode_estimates.json; optional model_comparison.json)
- Outputs: kerr_from_multimode.json, kerr_from_multimode_diagnostics.json

NOTE: Implementation intentionally deferred.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mvp.contracts import StageContext, abort, finalize, init_stage

STAGE = "s4d_kerr_from_multimode"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=STAGE)
    p.add_argument("--run-id", required=True)
    return p


def _execute(ctx: StageContext) -> dict[str, Path]:
    # TODO(phase-b): implement inversion + audit outputs
    raise NotImplementedError("s4d_kerr_from_multimode not implemented yet")


def _exit_code(value: object, default: int) -> int:
    return value if isinstance(value, int) else default


def _abort_with_reason(ctx: StageContext, reason: str) -> int:
    try:
        abort(ctx, reason=reason)
    except SystemExit as exc:
        return _exit_code(exc.code, 2)
    return 2


def main() -> int:
    args = build_argparser().parse_args()
    ctx = init_stage(args.run_id, STAGE)
    try:
        artifacts = _execute(ctx)
    except NotImplementedError:
        return _abort_with_reason(ctx, f"{STAGE} failed: NOT_IMPLEMENTED")
    except SystemExit as exc:
        return _exit_code(exc.code, 1)
    except Exception as exc:  # deterministic abort reason; no traceback in reason field
        reason = f"{STAGE} failed: {type(exc).__name__}: {exc}"
        return _abort_with_reason(ctx, reason)

    if not artifacts:
        return _abort_with_reason(ctx, f"{STAGE} failed: NO_OUTPUTS")

    finalize(ctx, artifacts=artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
