#!/usr/bin/env python3
"""Lightweight CI guard for theory doc presence and contract visibility.

This check is intentionally dependency-free and deterministic so it can run in
minimal CI runners.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def _extract_contract_stage_names(contracts_text: str) -> list[str]:
    # Match top-level dict keys like "s1_fetch_strain": StageContract(
    return sorted(set(re.findall(r'\n\s*"([a-zA-Z0-9_]+)"\s*:\s*StageContract\(', contracts_text)))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    doc_path = repo_root / "docs" / "technical_note_theory.md"
    contracts_path = repo_root / "mvp" / "contracts.py"

    errors: list[str] = []
    warnings: list[str] = []

    if not contracts_path.exists():
        errors.append(f"missing contracts file: {contracts_path}")
    else:
        contracts_text = contracts_path.read_text(encoding="utf-8")
        stage_names = _extract_contract_stage_names(contracts_text)
        if not stage_names:
            errors.append("could not discover any StageContract entries in mvp/contracts.py")

    if not doc_path.exists():
        errors.append(f"missing theory doc: {doc_path}")
    else:
        doc_text = doc_path.read_text(encoding="utf-8").strip()
        if not doc_text:
            errors.append("docs/technical_note_theory.md is empty")
        if not doc_text.startswith("#"):
            warnings.append("theory doc should start with a markdown heading")
        if "Content coming soon." in doc_text:
            warnings.append("theory doc still has placeholder content")

    if errors:
        for msg in errors:
            print(f"[doc-guard] ERROR: {msg}", file=sys.stderr)
        return 2

    for msg in warnings:
        print(f"[doc-guard] WARNING: {msg}")

    print("[doc-guard] OK: docs/technical_note_theory.md and mvp/contracts.py are present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
