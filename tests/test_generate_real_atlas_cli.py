from __future__ import annotations

import ast
from pathlib import Path


MODULE_PATH = Path("mvp/tools/generate_real_atlas.py")


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
