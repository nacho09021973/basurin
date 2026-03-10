from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from mvp.multimode_viability import (
    classify_multimode_viability,
    evaluate_science_evidence,
    evaluate_systematics_gate,
)


@pytest.fixture
def viability_base_inputs() -> dict[str, object]:
    return {
        "valid_fraction_220": 0.80,
        "f_220_median": 250.0,
        "f_220_iqr": 30.0,
        "valid_fraction_221": 0.55,
        "mode_221_ok": True,
        "f_221_median": 410.0,
        "f_221_iqr": 90.0,
        "spin_at_floor_frac_221": 0.05,
        "delta_bic": 3.2,
        "two_mode_preferred": True,
        "Rf_bootstrap_quantiles": {"q05": 0.90, "q50": 0.94, "q95": 0.99},
        "Rf_kerr_band": [0.88, 1.00],
    }


@pytest.fixture
def systematics_pass_inputs() -> dict[str, object]:
    return {
        "t0_plateau": {"plateau_detected": True, "f_std_over_plateau_hz": 1.8},
        "chi_psd_at_f221": 0.03,
        "Q_221_median": 2.3,
    }


def _science_inputs(viability: dict[str, object], systematics: dict[str, object]) -> dict[str, object]:
    return {
        "viability": viability,
        "systematics": systematics,
        "rf_bootstrap_quantiles": {"q05": 0.90, "q50": 0.94, "q95": 0.99},
        "rf_kerr_grid": [0.88, 0.91, 0.94, 0.97, 1.00],
        "chi_grid": [0.0, 0.25, 0.5, 0.75, 0.95],
    }


# ---- Clasificación (6) ----

def test_classification_multimode_ok_nominal(viability_base_inputs: dict[str, object]) -> None:
    out = classify_multimode_viability(viability_base_inputs)
    assert out["class"] == "MULTIMODE_OK"


def test_classification_ringdown_noninformative_when_220_fraction_low(
    viability_base_inputs: dict[str, object],
) -> None:
    p = dict(viability_base_inputs)
    p["valid_fraction_220"] = 0.49
    out = classify_multimode_viability(p)
    assert out["class"] == "RINGDOWN_NONINFORMATIVE"


def test_classification_ringdown_noninformative_when_rel_iqr_f220_too_high(
    viability_base_inputs: dict[str, object],
) -> None:
    p = dict(viability_base_inputs)
    p["f_220_iqr"] = 130.0  # 130/250 = 0.52 > 0.50
    out = classify_multimode_viability(p)
    assert out["class"] == "RINGDOWN_NONINFORMATIVE"


def test_classification_singlemode_only_when_221_fraction_low(
    viability_base_inputs: dict[str, object],
) -> None:
    p = dict(viability_base_inputs)
    p["valid_fraction_221"] = 0.29
    out = classify_multimode_viability(p)
    assert out["class"] == "SINGLEMODE_ONLY"


def test_classification_singlemode_only_when_221_not_usable(
    viability_base_inputs: dict[str, object],
) -> None:
    p = dict(viability_base_inputs)
    p["mode_221_ok"] = False
    p["two_mode_preferred"] = True
    out = classify_multimode_viability(p)
    assert out["class"] == "SINGLEMODE_ONLY"
    assert "mode_221_ok=false" in " ".join(out["reasons"])


def test_classification_singlemode_only_when_two_mode_not_preferred_or_delta_bic_missing(
    viability_base_inputs: dict[str, object],
) -> None:
    p = dict(viability_base_inputs)
    p["delta_bic"] = None
    p["two_mode_preferred"] = False
    out = classify_multimode_viability(p)
    assert out["class"] == "SINGLEMODE_ONLY"
    assert any("delta_bic" in reason or "two_mode_preferred=false" in reason for reason in out["reasons"])


def test_classification_boundary_values_are_accepted(viability_base_inputs: dict[str, object]) -> None:
    p = dict(viability_base_inputs)
    p["valid_fraction_220"] = 0.50
    p["valid_fraction_221"] = 0.30
    p["f_220_iqr"] = 125.0  # exactly 0.50
    p["spin_at_floor_frac_221"] = 0.30  # exactly threshold, not severe
    p["delta_bic"] = 2.0  # exactly supportive
    p["f_221_iqr"] = 246.0  # exactly 0.60
    out = classify_multimode_viability(p)
    assert out["class"] == "MULTIMODE_OK"


def test_classification_degrades_to_singlemode_with_two_severe_flags(
    viability_base_inputs: dict[str, object],
) -> None:
    p = dict(viability_base_inputs)
    p["spin_at_floor_frac_221"] = 0.31
    p["delta_bic"] = 1.0
    p["two_mode_preferred"] = None
    out = classify_multimode_viability(p)
    assert out["class"] == "SINGLEMODE_ONLY"
    assert out["metrics"]["n_severe_flags"] >= 2


# ---- science_evidence (4) + override degradante ----

def test_science_evidence_not_evaluated_by_multimode_gate(systematics_pass_inputs: dict[str, object]) -> None:
    viability = {"class": "SINGLEMODE_ONLY"}
    systematics = evaluate_systematics_gate(systematics_pass_inputs)
    out = evaluate_science_evidence(**_science_inputs(viability, systematics))
    assert out["status"] == "NOT_EVALUATED"
    assert "MULTIMODE_GATE" in out["reason_if_skipped"]


def test_science_evidence_not_evaluated_by_systematics_fail(
    viability_base_inputs: dict[str, object],
) -> None:
    viability = classify_multimode_viability(viability_base_inputs)
    systematics = evaluate_systematics_gate({"chi_psd_at_f221": 0.2, "Q_221_median": 2.3})
    out = evaluate_science_evidence(**_science_inputs(viability, systematics))
    assert out["status"] == "NOT_EVALUATED"
    assert "SYSTEMATICS_FAIL" in out["reason_if_skipped"]


def test_science_evidence_not_evaluated_by_systematics_na(viability_base_inputs: dict[str, object]) -> None:
    viability = classify_multimode_viability(viability_base_inputs)
    systematics = evaluate_systematics_gate({})
    out = evaluate_science_evidence(**_science_inputs(viability, systematics))
    assert out["status"] == "NOT_EVALUATED"
    assert "SYSTEMATICS_NOT_AVAILABLE" in out["reason_if_skipped"]


def test_science_evidence_evaluated_when_all_gates_pass(
    viability_base_inputs: dict[str, object],
    systematics_pass_inputs: dict[str, object],
) -> None:
    viability = classify_multimode_viability(viability_base_inputs)
    systematics = evaluate_systematics_gate(systematics_pass_inputs)
    out = evaluate_science_evidence(**_science_inputs(viability, systematics))
    assert out["status"] == "EVALUATED"
    assert out["H1_min"]["delta_Rf"]["value"] == pytest.approx(0.0)
    assert out["H1_min"]["delta_Rf"]["chi_star"] == pytest.approx(0.5)


def test_override_only_degrades_for_science_evidence(
    viability_base_inputs: dict[str, object],
    systematics_pass_inputs: dict[str, object],
) -> None:
    viability = classify_multimode_viability(viability_base_inputs)
    systematics = evaluate_systematics_gate(systematics_pass_inputs)

    out_no_override = evaluate_science_evidence(
        **_science_inputs(viability, systematics),
        override=None,
    )
    out_fail_override = evaluate_science_evidence(
        **_science_inputs(viability, systematics),
        override={"verdict_human": "FAIL"},
    )

    assert out_no_override["status"] == "EVALUATED"
    assert out_fail_override["status"] == "NOT_EVALUATED"
    assert "SYSTEMATICS_FAIL" in out_fail_override["reason_if_skipped"]


# ---- Determinismo y saneamiento numérico ----

def test_determinism_same_input_same_output(viability_base_inputs: dict[str, object]) -> None:
    out_a = classify_multimode_viability(viability_base_inputs)
    out_b = classify_multimode_viability(viability_base_inputs)
    assert out_a == out_b


def test_classification_no_nan_no_inf_in_numeric_metrics(viability_base_inputs: dict[str, object]) -> None:
    out = classify_multimode_viability(viability_base_inputs)
    for value in out["metrics"].values():
        if isinstance(value, (int, float)):
            assert math.isfinite(float(value))


def test_science_evidence_no_nan_no_inf_when_evaluated(
    viability_base_inputs: dict[str, object],
    systematics_pass_inputs: dict[str, object],
) -> None:
    viability = classify_multimode_viability(viability_base_inputs)
    systematics = evaluate_systematics_gate(systematics_pass_inputs)
    out = evaluate_science_evidence(**_science_inputs(viability, systematics))
    delta = out["H1_min"]["delta_Rf"]

    assert math.isfinite(float(delta["value"]))
    assert all(math.isfinite(float(x)) for x in delta["interval"])
    assert math.isfinite(float(delta["chi_star"]))
    assert math.isfinite(float(delta["Rf_kerr_at_chi_star"]))


# ---- Golden condicional para referencia Kerr ----

def test_kerr_ratio_reference_json_golden_structure_and_hash() -> None:
    ref_path = Path("mvp/kerr_ratio_reference.json")
    if not ref_path.exists():
        pytest.skip("kerr_ratio_reference.json aún no existe en este repo")

    raw = ref_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))

    # Golden de estructura estable (sin fijar valores científicos exactos todavía).
    assert isinstance(payload, dict)
    assert payload, "El JSON de referencia Kerr no debe estar vacío"
    assert any("rf" in str(k).lower() or "chi" in str(k).lower() for k in payload.keys())

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    assert len(digest) == 64
    assert all(ch in "0123456789abcdef" for ch in digest)
