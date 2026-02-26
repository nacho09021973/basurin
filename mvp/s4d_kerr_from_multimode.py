"""
s4d_kerr_from_multimode

Canonical stage (Phase B):
- Inputs: s3b multimode outputs (multimode_estimates.json; optional model_comparison.json)
- Outputs: kerr_from_multimode.json, kerr_from_multimode_diagnostics.json

NOTE: Implementation intentionally deferred.
"""
from __future__ import annotations

import argparse

from mvp.contracts import StageContext, abort, finalize, init_stage

STAGE = "s4d_kerr_from_multimode"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=STAGE)
    p.add_argument("--run-id", required=True)
    return p


def _execute(ctx: StageContext) -> None:
    # TODO(phase-b): implement inversion + audit outputs
    raise NotImplementedError("s4d_kerr_from_multimode not implemented yet")


def main() -> int:
    args = build_argparser().parse_args()
    ctx = init_stage(args.run_id, STAGE)
    try:
        _execute(ctx)
        finalize(ctx, artifacts={})
        return 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code
    except Exception as exc:  # deterministic abort reason; no traceback in reason field
        reason = f"{STAGE} failed: {type(exc).__name__}: {exc}"
        try:
            abort(ctx, reason=reason)
        except SystemExit as abort_exc:
            code = abort_exc.code if isinstance(abort_exc.code, int) else 2
            return code
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
