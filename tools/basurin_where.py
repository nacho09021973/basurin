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
            "ringdown_synth/outputs/synthetic_events_list.json",
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

def _has_strain_cases(run_dir: Path) -> bool:
    cases_dir = run_dir / "ringdown_synth" / "outputs" / "cases"
    return bool(list(cases_dir.glob("*/strain.npz")))

def main() -> int:
    ap = argparse.ArgumentParser(description="BASURIN where: entrypoints + missing canonical artifacts for a RUN_ID.")
    ap.add_argument("--run", required=True, help="run_id (folder name under runs/)")
    ap.add_argument("--out-root", default=None, help="runs root override (default: ./runs)")
    ap.add_argument("--ringdown-min", action="store_true", help="report minimal Ringdown chain (RUN_VALID + ringdown_synth)")
    ap.add_argument("--ringdown-exp03", action="store_true", help="report Ringdown EXP03 chain (EXP01 + OBSERVABLES_V1 + EXP03)")
    ap.add_argument("--ringdown-exp04", action="store_true", help="report Ringdown EXP04 chain (RUN_VALID + ringdown_synth + EXP04)")
    ap.add_argument("--ringdown-exp05", action="store_true", help="report Ringdown EXP05 chain (RUN_VALID + ringdown_synth + EXP01 + EXP05)")
    ap.add_argument("--ringdown-exp06", action="store_true", help="report Ringdown EXP06 chain (RUN_VALID + ringdown_synth + EXP01 + EXP06)")
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

    if args.ringdown_exp04:
        specs = RINGDOWN_MIN_SPECS
    elif args.ringdown_exp06:
        specs = RINGDOWN_MIN_SPECS
    elif args.ringdown_exp05:
        specs = RINGDOWN_MIN_SPECS
    elif args.ringdown_exp03:
        specs = RINGDOWN_MIN_SPECS
    elif args.ringdown_min:
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

    if args.ringdown_exp04:
        exp04_entry = "experiment/ringdown/exp_ringdown_04_psd_validity.py"
        exp04_outputs = (
            "experiment/ringdown/EXP_RINGDOWN_04__psd_validity/outputs/psd_diagnostics.json",
            "experiment/ringdown/EXP_RINGDOWN_04__psd_validity/outputs/per_case_psd.jsonl",
            "experiment/ringdown/EXP_RINGDOWN_04__psd_validity/outputs/contract_verdict.json",
        )

        if not _has_strain_cases(run_dir):
            overall_ok = False
            print("- ringdown_synth strain cases: MISSING")
            print("  missing: ringdown_synth/outputs/cases/*/strain.npz")
        else:
            print("- ringdown_synth strain cases: OK")
        print("  hint: requiere al menos un strain.npz en cases/.")

        ok, missing = _exists_all(run_dir, exp04_outputs)
        tag = "OK" if ok else "MISSING"
        print(f"- EXP_RINGDOWN_04__psd_validity: {tag}")
        print(f"  entrypoint: {exp04_entry}")
        if missing:
            overall_ok = False
            for m in missing:
                print(f"  missing: {m}")
        print("  hint: requiere RUN_VALID PASS + ringdown_synth outputs.")

    if args.ringdown_exp05:
        exp01_cases = (
            "experiment/ringdown_01_injection_recovery/outputs/recovery_cases.jsonl"
        )
        exp05_entry = "experiment/ringdown/exp_ringdown_05_prior_hyperparam_sweep.py"
        exp05_outputs = (
            "experiment/ringdown/EXP_RINGDOWN_05__prior_hyperparam_sweep/outputs/prior_sweep.json",
            "experiment/ringdown/EXP_RINGDOWN_05__prior_hyperparam_sweep/outputs/per_case.jsonl",
            "experiment/ringdown/EXP_RINGDOWN_05__prior_hyperparam_sweep/outputs/contract_verdict.json",
        )

        ok, missing = _exists_all(run_dir, (exp01_cases,))
        tag = "OK" if ok else "MISSING"
        print(f"- EXP_RINGDOWN_01_injection_recovery: {tag}")
        print("  entrypoint: experiment/ringdown/exp_ringdown_01_injection_recovery.py")
        if missing:
            overall_ok = False
            for m in missing:
                print(f"  missing: {m}")
        print("  hint: EXP05 requiere recovery_cases.jsonl de EXP01.")

        ok, missing = _exists_all(run_dir, exp05_outputs)
        tag = "OK" if ok else "MISSING"
        print(f"- EXP_RINGDOWN_05__prior_hyperparam_sweep: {tag}")
        print(f"  entrypoint: {exp05_entry}")
        if missing:
            overall_ok = False
            for m in missing:
                print(f"  missing: {m}")
        print("  hint: requiere RUN_VALID PASS + EXP01 outputs.")

    if args.ringdown_exp06:
        exp01_cases = (
            "experiment/ringdown_01_injection_recovery/outputs/recovery_cases.jsonl"
        )
        exp06_entry = (
            "PYTHONPATH=. python experiment/ringdown/exp_ringdown_06_psd_robustness.py --run \"$RUN\""
        )
        exp06_inputs = (
            "RUN_VALID/verdict.json",
            "ringdown_synth/outputs/synthetic_events_list.json",
            exp01_cases,
        )
        exp06_outputs = (
            "experiment/ringdown/EXP_RINGDOWN_06__psd_robustness/outputs/psd_sweep_metrics.json",
            "experiment/ringdown/EXP_RINGDOWN_06__psd_robustness/outputs/psd_cases.jsonl",
            "experiment/ringdown/EXP_RINGDOWN_06__psd_robustness/outputs/contract_verdict.json",
        )

        ok, missing = _exists_all(run_dir, (exp01_cases,))
        tag = "OK" if ok else "MISSING"
        print(f"- EXP_RINGDOWN_01_injection_recovery: {tag}")
        print("  entrypoint: experiment/ringdown/exp_ringdown_01_injection_recovery.py")
        if missing:
            overall_ok = False
            for m in missing:
                print(f"  missing: {m}")
        print("  hint: EXP06 requiere recovery_cases.jsonl de EXP01.")

        ok, missing = _exists_all(run_dir, exp06_outputs)
        tag = "OK" if ok else "MISSING"
        print(f"- EXP_RINGDOWN_06__psd_robustness: {tag}")
        print(f"  entrypoint: {exp06_entry}")
        print("  inputs:")
        for path in exp06_inputs:
            print(f"    - {path}")
        print("  outputs:")
        for path in exp06_outputs:
            print(f"    - {path}")
        if missing:
            overall_ok = False
            for m in missing:
                print(f"  missing: {m}")
        print("  hint: requiere RUN_VALID PASS + ringdown_synth + EXP01 outputs.")

    if args.ringdown_exp03:
        exp01_pref = (
            "experiment/ringdown/EXP_RINGDOWN_01_injection_recovery/outputs/recovery_cases.jsonl"
        )
        exp01_legacy = (
            "experiment/ringdown_01_injection_recovery/outputs/recovery_cases.jsonl"
        )
        exp01_summary_pref = (
            "experiment/ringdown/EXP_RINGDOWN_01_injection_recovery/outputs/recovery_summary.json"
        )
        exp01_summary_legacy = (
            "experiment/ringdown_01_injection_recovery/outputs/recovery_summary.json"
        )
        exp01_entry = "experiment/ringdown/exp_ringdown_01_injection_recovery.py"

        exp03_entry = "experiment/ringdown/exp_ringdown_03_observable_minimality.py"
        obs_entry = "experiment/ringdown/stage_ringdown_observables_v1.py"

        exp03_outputs = (
            "experiment/ringdown/EXP_RINGDOWN_03__observable_minimality/outputs/identifiability_report.json",
            "experiment/ringdown/EXP_RINGDOWN_03__observable_minimality/outputs/ablations.jsonl",
            "experiment/ringdown/EXP_RINGDOWN_03__observable_minimality/outputs/contract_verdict.json",
        )
        obs_outputs = (
            "experiment/ringdown/STAGE_RINGDOWN_OBSERVABLES_V1/outputs/observables.jsonl",
            "experiment/ringdown/STAGE_RINGDOWN_OBSERVABLES_V1/outputs/contract_verdict.json",
        )

        print("- EXP_RINGDOWN_01_injection_recovery: ", end="")
        exp01_cases_ok = (run_dir / exp01_pref).exists() or (run_dir / exp01_legacy).exists()
        exp01_summary_ok = (run_dir / exp01_summary_pref).exists() or (run_dir / exp01_summary_legacy).exists()
        if exp01_cases_ok and exp01_summary_ok:
            print("OK")
        else:
            print("MISSING")
            overall_ok = False
            if not exp01_cases_ok:
                print(f"  missing: {exp01_pref} (or legacy {exp01_legacy})")
            if not exp01_summary_ok:
                print(f"  missing: {exp01_summary_pref} (or legacy {exp01_summary_legacy})")
        print(f"  entrypoint: {exp01_entry}")
        print("  hint: EXP03 usa recovery_cases.jsonl como base.")

        ok, missing = _exists_all(run_dir, obs_outputs)
        tag = "OK" if ok else "MISSING"
        print(f"- STAGE_RINGDOWN_OBSERVABLES_V1: {tag}")
        print(f"  entrypoint: {obs_entry}")
        if missing:
            overall_ok = False
            for m in missing:
                print(f"  missing: {m}")
        print("  hint: ejecuta stage_ringdown_observables_v1.py si no existe.")

        ok, missing = _exists_all(run_dir, exp03_outputs)
        tag = "OK" if ok else "MISSING"
        print(f"- EXP_RINGDOWN_03__observable_minimality: {tag}")
        print(f"  entrypoint: {exp03_entry}")
        if missing:
            overall_ok = False
            for m in missing:
                print(f"  missing: {m}")
        print("  hint: requiere RUN_VALID PASS + EXP01/OBSERVABLES_V1.")

    print("\n[result]")
    if overall_ok:
        print("READY: YES")
        return 0
    print("READY: NO")
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
