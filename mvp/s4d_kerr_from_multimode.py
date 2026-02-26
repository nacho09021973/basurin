"""
s4d_kerr_from_multimode

Canonical stage (Phase B):
- Inputs: s3b multimode outputs (multimode_estimates.json; optional model_comparison.json)
- Outputs: kerr_from_multimode.json, kerr_from_multimode_diagnostics.json

NOTE: Implementation intentionally deferred.
"""
from __future__ import annotations

import argparse

from mvp.contracts import StageContext, run_stage


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="s4d_kerr_from_multimode")
    p.add_argument("--run-id", required=True)
    return p


def _execute(ctx: StageContext) -> None:
    # TODO(phase-b): implement inversion + audit outputs
    raise NotImplementedError("s4d_kerr_from_multimode not implemented yet")


def main() -> None:
    args = build_argparser().parse_args()
    run_stage("s4d_kerr_from_multimode", run_id=args.run_id, execute_fn=_execute)


if __name__ == "__main__":
    main()
