from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {
    "schema_version",
    "timestamp_utc",
    "parameters",
    "inputs",
    "results",
    "verdict",
    "reasons",
}


def validate_bayes_output(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    missing = sorted(REQUIRED_TOP_LEVEL - set(payload.keys()))
    if missing:
        reasons.append(f"missing_keys:{','.join(missing)}")

    if payload.get("schema_version") != "bayes_validation_v1":
        reasons.append("invalid_schema_version")

    params = payload.get("parameters", {})
    for key in ["seed", "n_monte_carlo", "k_features", "models", "prior_precision"]:
        if key not in params:
            reasons.append(f"missing_parameter:{key}")

    inputs = payload.get("inputs", [])
    if not isinstance(inputs, list) or not inputs:
        reasons.append("missing_inputs")

    return len(reasons) == 0, reasons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="bayes_contracts is a library-first validator; CLI only validates and prints."
    )
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    ok, reasons = validate_bayes_output(payload)
    print(json.dumps({"ok": ok, "reasons": reasons}, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
