#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

@dataclass(frozen=True)
class StageSpec:
    name: str
    entrypoint: str
    required_files: Tuple[str, ...]  # relative to runs/<run_id>/
    hints: Tuple[str, ...] = ()

RINGDOWN_MIN_SPECS = (
    StageSpec(
        name="RUN_VALID",
        entrypoint="experiment/run_valid/stage_run_valid.py",
        required_files=(
            "RUN_VALID/verdict.json",
            "RUN_VALID/stage_summary.json",
            "RUN_VALID/manifest.json",
        ),
        hints=("Soberano: el run no existe si esto no está en PASS.",),
    ),
    StageSpec(
        name="ringdown_synth",
        entrypoint="stages/ringdown_synth_stage.py",
        required_files=(
            "ringdown_synth/outputs/synthetic_event.json",
            "ringdown_synth/outputs/synthetic_events.json",
            "ringdown_synth/stage_summary.json",
            "ringdown_synth/manifest.json",
        ),
        hints=("Generador canónico; no inventar sintéticos downstream.",),
    ),
)

def _repo_root() -> Path:
    # Determinista: usa CWD
    return Path.cwd().resolve()

def _run_dir(run_id: str, out_root: Optional[str]) -> Path:
    root = Path(out_root).expanduser().resolve() if out_root else (_repo_root() / "runs")
    return (root / run_id).resolve()

def _exists_all(base: Path, rel_paths: Iterable[str]) -> Tuple[bool, list[str]]:
    missing = []
    for rp in rel_paths:
        if not (base / rp).exists():
            missing.append(rp)
    return (len(missing) == 0), missing

def _read_run_valid_verdict(run_dir: Path) -> Optional[str]:
    p = run_dir / "RUN_VALID" / "verdict.json"
    if not p.exists():
        return None
    try:
        import json
        v = json.loads(p.read_text(encoding="utf-8"))
        verdict = str(v.get("verdict", v.get("status", ""))).upper()
        return verdict or None
    except Exception:
        return "UNREADABLE"

def main() -> int:
    ap = argparse.ArgumentParser(description="BASURIN where: entrypoints + missing canonical artifacts for a RUN_ID.")
    ap.add_argument("--run", required=True, help="run_id (folder name under runs/)")
    ap.add_argument("--out-root", default=None, help="runs root override (default: ./runs)")
    ap.add_argument("--ringdown-min", action="store_true", help="report minimal Ringdown chain (RUN_VALID + ringdown_synth)")
    args = ap.parse_args()

    rr = _repo_root()
    run_dir = _run_dir(args.run, args.out_root)

    print(f"[repo_root] {rr}")
    print(f"[run_dir]   {run_dir}")

    if not run_dir.exists():
        print("STATUS: MISSING_RUN_DIR")
        print("HINT: ejecuta RUN_VALID para crear el run.")
        print(f"ENTRYPOINT: {RINGDOWN_MIN_SPECS[0].entrypoint}")
        return 2

    if args.ringdown_min:
        specs = RINGDOWN_MIN_SPECS
    else:
        specs = RINGDOWN_MIN_SPECS  # por ahora

    print("\n[checks]")
    overall_ok = True
    for spec in specs:
        ok, missing = _exists_all(run_dir, spec.required_files)
        tag = "OK" if ok else "MISSING"
        print(f"- {spec.name}: {tag}")
        print(f"  entrypoint: {spec.entrypoint}")
        if missing:
            overall_ok = False
            for m in missing:
                print(f"  missing: {m}")
        for h in spec.hints:
            print(f"  hint: {h}")

        if spec.name == "RUN_VALID":
            verdict = _read_run_valid_verdict(run_dir)
            if verdict is None:
                overall_ok = False
                print("  verdict: <missing>")
            else:
                print(f"  verdict: {verdict}")
                if verdict != "PASS":
                    overall_ok = False

    print("\n[result]")
    if overall_ok:
        print("READY: YES")
        return 0
    print("READY: NO")
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
