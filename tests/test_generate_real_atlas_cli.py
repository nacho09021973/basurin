from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mvp.generate_atlas_from_fits import _format_mass_token
from mvp.tools.generate_multimode_atlas_v3 import (
    canonical_geometry_id,
    extract_physical_parameters,
    validate_physical_parameters,
)

MODULE_PATH = Path("mvp/tools/generate_real_atlas.py")
MULTIMODE_MODULE_PATH = Path("mvp/tools/generate_multimode_atlas_v3.py")


def test_parse_args_defines_required_out_flag():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))

    add_arg_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ]

    found_required_out = False
    for call in add_arg_calls:
        if not call.args:
            continue
        first_arg = call.args[0]
        if isinstance(first_arg, ast.Constant) and first_arg.value == "--out":
            for kw in call.keywords:
                if kw.arg == "required" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    found_required_out = True
                    break

    assert found_required_out, "--out debe ser obligatorio en parse_args"


def test_script_has_no_home_claude_hardcoded_path():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "/home/claude" not in text


def test_multimode_generator_has_no_home_claude_hardcoded_path():
    text = MULTIMODE_MODULE_PATH.read_text(encoding="utf-8")
    assert "/home/claude" not in text


def test_canonical_geometry_id_strips_mode_suffix():
    assert canonical_geometry_id("Kerr_M90_a0.8631_l2m2n1") == "Kerr_M90_a0.8631"
    assert canonical_geometry_id("EdGB_M62_a0.67_z0.3") == "EdGB_M62_a0.67_z0.3"


def test_extract_physical_parameters_populates_kerr_fields():
    entry = {
        "geometry_id": "Kerr_M90_a0.8631_l2m2n0",
        "theory": "GR_Kerr",
        "metadata": {
            "family": "kerr",
            "M_solar": 90.0,
            "chi": 0.8631,
            "mode": "(2,2,0)",
        },
    }

    params = extract_physical_parameters(entry, "Kerr_M90_a0.8631")
    assert params["M_solar"] == 90.0
    assert params["chi"] == 0.8631
    assert params["a_over_m"] == 0.8631
    assert params["J_over_M2"] == 0.8631
    assert params["kerr_r_plus_over_M"] > 1.0
    assert params["kerr_area_over_M2"] > 0.0


def test_validate_physical_parameters_rejects_superextremal_kerr():
    entry = {
        "geometry_id": "Kerr_M90_a1.1000_l2m2n0",
        "theory": "GR_Kerr",
        "metadata": {
            "family": "kerr",
            "M_solar": 90.0,
            "chi": 1.1,
            "mode": "(2,2,0)",
        },
    }

    with pytest.raises(ValueError, match=r"\|chi\| must be <= 1"):
        validate_physical_parameters(entry, "Kerr_M90_a1.1000", {"chi": 1.1})


def test_format_mass_token_preserves_non_integer_resolution():
    assert _format_mass_token(2.4) == "2.4"
    assert _format_mass_token(2.45) == "2.45"
    assert _format_mass_token(2.4567) == "2.4567"


def test_format_mass_token_keeps_integer_ids_stable():
    assert _format_mass_token(62.0) == "62"
    assert _format_mass_token(90.0) == "90"
