#!/usr/bin/env python3
"""
CI guard: ensure docs/technical_note_theory.md stays contract-true vs mvp/contracts.py
and prevent reintroducing known math errors (QNM sign/dimensions, whitening sqrt).

Fail-fast semantics: any violation exits non-zero.
No external deps.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / "docs" / "technical_note_theory.md"
CONTRACTS_PATH = REPO_ROOT / "mvp" / "contracts.py"


# --- helpers ---------------------------------------------------------------

def die(msg: str, *, code: int = 1) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(code)


def read_text(path: Path) -> str:
    if not path.exists():
        die(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def extract_contract_outputs_from_ast(py_text: str) -> Set[str]:
    """
    Parse mvp/contracts.py and extract all strings under produced_outputs and required_inputs.
    This avoids importing the repo (no sys.path side effects in CI).
    """
    tree = ast.parse(py_text)

    outputs: Set[str] = set()
    inputs: Set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Dict(self, node: ast.Dict) -> None:
            # Look for CONTRACTS = { ... } but keep it generic:
            # whenever we see a keyword 'produced_outputs' or 'required_inputs' in a call,
            # collect string constants from the list.
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            # Identify StageContract(... produced_outputs=[...], required_inputs=[...])
            for kw in node.keywords:
                if kw.arg in ("produced_outputs", "required_inputs"):
                    if isinstance(kw.value, (ast.List, ast.Tuple)):
                        for elt in kw.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                if kw.arg == "produced_outputs":
                                    outputs.add(elt.value)
                                else:
                                    inputs.add(elt.value)
            self.generic_visit(node)

    Visitor().visit(tree)
    # We guard only doc mentions of outputs/..., so include both:
    # - produced outputs (stage_dir-relative)
    # - required inputs (run_dir-relative), because docs may cite upstream paths like
    #   s2_ringdown_window/outputs/window_meta.json
    return outputs.union(inputs)


def find_doc_output_mentions(doc_text: str) -> Set[str]:
    """
    Find all substrings that look like:
      - outputs/<something.ext>
      - <stage>/outputs/<something.ext>
    Capture up to extension; keep the path as written.
    """
    # capture either "outputs/..." or "something/outputs/..."
    pattern = re.compile(r"\b(?:[A-Za-z0-9_]+/)?outputs/[A-Za-z0-9_.-]+\b")
    return set(pattern.findall(doc_text))


def normalize_to_contract_key(path: str) -> str:
    """
    Contract stores:
      produced_outputs: outputs/<file>
      required_inputs: <upstream_stage>/outputs/<file>
    Doc may mention both. Keep as-is, but also provide fallback to outputs/<file>.
    """
    return path.strip()


def check_no_phantom_extensions(doc_text: str) -> Tuple[bool, str]:
    banned = [
        ".gwf",
        ".h5",
        "psd.txt",
        "whitened_data",
        "tapered_strain",
        "metric_grid",
        "curvature_scalar",
    ]
    for token in banned:
        if token in doc_text:
            return False, f"Banned token found in doc: '{token}'"
    return True, ""


def check_qnm_formula(doc_text: str) -> Tuple[bool, str]:
    """
    Guard against common wrong form: '- iτ' (dimensionally wrong).
    We do NOT enforce a single exact LaTeX string; we only ban the known wrong one.
    """
    # catch variants like "-iτ", "−iτ", "- i τ", "-i tau" etc.
    wrong = re.compile(r"[-−]\s*i\s*τ\b|[-−]\s*i\s*tau\b|[-−]\s*i\s*\\tau\b")
    m = wrong.search(doc_text)
    if m:
        return False, f"QNM formula appears to use '-iτ' (wrong). Match: '{m.group(0)}'"
    return True, ""


def check_whitening_sqrt(doc_text: str) -> Tuple[bool, str]:
    """
    Ban whitening definitions that divide by S_n without sqrt.
    We allow many conventions; the key is the sqrt.
    """
    # If the doc contains a whitening equation, ensure sqrt appears nearby.
    # Conservative: flag if we see "/S_n" or "/Sn(" etc without sqrt on the same line.
    bad_lines = []
    for i, line in enumerate(doc_text.splitlines(), start=1):
        if re.search(r"/\s*S_n|/\s*Sn|/\s*\\?S_n", line):
            if "sqrt" not in line and "\\sqrt" not in line and "√" not in line:
                bad_lines.append((i, line.strip()))
    if bad_lines:
        preview = "; ".join([f"L{i}:{l}" for i, l in bad_lines[:3]])
        return False, f"Whitening division by S_n without sqrt detected: {preview}"
    return True, ""


# --- main ------------------------------------------------------------------

def main() -> None:
    doc_text = read_text(DOC_PATH)
    contracts_text = read_text(CONTRACTS_PATH)

    allowed_paths = extract_contract_outputs_from_ast(contracts_text)
    mentioned_paths = {normalize_to_contract_key(p) for p in find_doc_output_mentions(doc_text)}

    # 1) Ban phantom artifacts/extensions
    ok, msg = check_no_phantom_extensions(doc_text)
    if not ok:
        die(msg)

    # 2) QNM formula guard
    ok, msg = check_qnm_formula(doc_text)
    if not ok:
        die(msg)

    # 3) Whitening sqrt guard
    ok, msg = check_whitening_sqrt(doc_text)
    if not ok:
        die(msg)

    # 4) Contract-true outputs/paths
    # Any mentioned path must exist in contract outputs/inputs.
    # Additionally allow doc mentions that drop upstream stage prefix (outputs/<file>)
    # if the contract only lists the upstream prefixed version.
    def is_allowed(p: str) -> bool:
        if p in allowed_paths:
            return True
        # allow outputs/<file> if contract contains */outputs/<file>
        if p.startswith("outputs/"):
            suffix = p
            for a in allowed_paths:
                if a.endswith("/" + suffix) or a == suffix:
                    return True
        return False

    violations = sorted([p for p in mentioned_paths if not is_allowed(p)])
    if violations:
        sample = "\n".join(f"  - {v}" for v in violations[:20])
        die(
            "Doc mentions output paths not present in mvp/contracts.py (contract drift):\n"
            f"{sample}\n"
            "Fix: update docs to use only contract-declared artifacts OR update the contract explicitly."
        )

    print("[PASS] docs/technical_note_theory.md is contract-true and math-guarded.")


if __name__ == "__main__":
    main()
# noop: trigger docs_contract_guard
