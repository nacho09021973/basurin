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

_KNOWN_SCORE_NAMES = frozenset({"neg_mse", "bic", "neg_bic", "log_evidence"})


def _validate_c4(c4: dict[str, Any], reasons: list[str]) -> None:
    """Validate C4_model_selection sub-block semantics."""
    required_keys = {"best_model", "model_scores", "score_name", "score_higher_is_better", "delta_score"}
    c4_missing = sorted(required_keys - set(c4.keys()))
    if c4_missing:
        reasons.append(f"C4_missing_keys:{','.join(c4_missing)}")
        return

    score_name = c4.get("score_name")
    if score_name not in _KNOWN_SCORE_NAMES:
        reasons.append(f"C4_unknown_score_name:{score_name}")

    if not isinstance(c4.get("score_higher_is_better"), bool):
        reasons.append("C4_score_higher_is_better_not_bool")

    # best_model must be consistent with model_scores and score semantics
    model_scores = c4.get("model_scores", {})
    higher = c4.get("score_higher_is_better")
    if model_scores and isinstance(higher, bool):
        expected = max(model_scores, key=model_scores.get) if higher else min(model_scores, key=model_scores.get)
        if c4.get("best_model") != expected:
            reasons.append(
                f"C4_best_model_inconsistent:expected={expected},got={c4.get('best_model')}"
            )


def _validate_verdict_bf_rule(payload: dict[str, Any], reasons: list[str]) -> None:
    """Verdict cannot be PASS if log_bayes_factor_proxy is null."""
    results = payload.get("results", {})
    c4 = results.get("C4_model_selection", {})
    if payload.get("verdict") == "PASS" and c4.get("log_bayes_factor_proxy") is None:
        reasons.append("C4_pass_without_bayes_factor:log_bayes_factor_proxy is null")


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

    # Validate C4_model_selection sub-block
    results = payload.get("results", {})
    c4 = results.get("C4_model_selection", {})
    if c4:
        _validate_c4(c4, reasons)

    # Verdict-level rule: no PASS without BF proxy
    _validate_verdict_bf_rule(payload, reasons)

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
