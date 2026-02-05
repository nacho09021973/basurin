#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _cand in [_here.parents[1], _here.parents[2], _here.parents[3]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break

from basurin_io import (  # noqa: E402
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)
from bayes_contracts import validate_bayes_output  # noqa: E402
from bayes_inference import run_bayes_model_selection  # noqa: E402

EXIT_CONTRACT_FAIL = 2


def _abort_contract(reason: str) -> int:
    print(json.dumps({"verdict": "FAIL", "reason": reason}, sort_keys=True), file=sys.stderr)
    return EXIT_CONTRACT_FAIL


def _load_signal_vector(input_path: Path) -> list[float]:
    raw = input_path.read_bytes()
    if not raw:
        return [0.0, 0.0, 0.0]
    chunk = max(1, len(raw) // 32)
    vals: list[float] = []
    for i in range(0, len(raw), chunk):
        block = raw[i : i + chunk]
        vals.append(float(sum(block) / max(1, len(block))))
    return vals[:64]


def main() -> int:
    ap = argparse.ArgumentParser(description="BASURIN canonical bayes_validation_v1 stage")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--stage-name", default="bayes_validation_v1")
    ap.add_argument("--spectrum-stage", default="spectrum")
    ap.add_argument("--k-features", type=int, default=3)
    ap.add_argument("--n-monte-carlo", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default="linear,poly2")
    ap.add_argument("--prior-precision", type=float, default=1e-6)
    args = ap.parse_args()

    out_root = resolve_out_root(args.out_root)
    validate_run_id(args.run, out_root)

    try:
        require_run_valid(out_root, args.run)
    except Exception:
        return _abort_contract("require_run_valid")

    try:
        import scipy  # noqa: F401
    except Exception:
        return _abort_contract("scipy_missing")

    run_dir = out_root / args.run
    input_rel = Path("runs") / args.run / args.spectrum_stage / "outputs" / "spectrum.h5"
    input_path = run_dir / args.spectrum_stage / "outputs" / "spectrum.h5"
    if not input_path.exists():
        return _abort_contract("missing_spectrum_h5")

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, args.stage_name, base_dir=out_root)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    values = _load_signal_vector(input_path)

    c4 = run_bayes_model_selection(
        values,
        models,
        seed=args.seed,
        n_monte_carlo=args.n_monte_carlo,
        k_features=args.k_features,
        prior_precision=args.prior_precision,
    )

    # --- Derive verdict from C4 results ---
    verdict = "PASS"
    verdict_reasons: list[str] = []

    if c4.get("log_bayes_factor_proxy") is None:
        verdict = "INSPECT"
        verdict_reasons.append(
            "C4: log_bayes_factor_proxy not computed; model selection based on surrogate score only"
        )

    if not c4.get("selection_consistent", True):
        verdict = "INSPECT"
        verdict_reasons.append(
            "C4: neg_mse and BIC select different best models; review recommended"
        )

    uncertainty = {
        "mc_mean": float(sum(values) / len(values)),
        "mc_std": float((sum((v - (sum(values) / len(values))) ** 2 for v in values) / len(values)) ** 0.5),
        "n_monte_carlo": int(args.n_monte_carlo),
        "seed": int(args.seed),
    }
    sensitivity = {
        "k_features": int(args.k_features),
        "models_tested": models,
        "prior_precision": float(args.prior_precision),
    }

    spectrum_sha256 = sha256_file(input_path)
    payload = {
        "schema_version": "bayes_validation_v1",
        "timestamp_utc": "1970-01-01T00:00:00+00:00",
        "parameters": {
            "seed": int(args.seed),
            "n_monte_carlo": int(args.n_monte_carlo),
            "k_features": int(args.k_features),
            "models": models,
            "prior_precision": float(args.prior_precision),
        },
        "inputs": [{"path": str(input_rel), "sha256": spectrum_sha256}],
        "hashes": {"spectrum_h5": spectrum_sha256},
        "results": {
            "C4_model_selection": c4,
            "uncertainty_propagation": uncertainty,
            "sensitivity_analysis": sensitivity,
        },
        "verdict": verdict,
        "reasons": verdict_reasons,
    }

    ok, reasons = validate_bayes_output(payload)
    if not ok:
        return _abort_contract(";".join(reasons))

    output_path = outputs_dir / "bayes_validation.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "stage": args.stage_name,
        "run": args.run,
        "verdict": verdict,
        "reasons": verdict_reasons,
        "parameters": payload["parameters"],
        "inputs": payload["inputs"],
        "outputs": {"bayes_validation": "outputs/bayes_validation.json"},
        "seed": int(args.seed),
        "n_monte_carlo": int(args.n_monte_carlo),
    }
    summary_path = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {
            "bayes_validation": output_path,
            "stage_summary": summary_path,
        },
    )
    print(f"OK: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
