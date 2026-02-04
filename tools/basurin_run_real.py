#!/usr/bin/env python3
"""
Runner canónico real de ringdown (1 comando por evento).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# --- BASURIN import bootstrap (no depende de PYTHONPATH) ---
_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1], _here.parents[2]):
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break
# -----------------------------------------------------------

from basurin_io import get_run_dir, get_runs_root, require_run_valid, sha256_file, validate_run_id


def _parse_band_hz(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("band-hz debe tener formato 'low,high'")
    try:
        return [float(parts[0]), float(parts[1])]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("band-hz debe ser numérico") from exc


def _compute_suffix(dt_start_s: float, duration_s: float) -> str:
    dt_ms = int(round(dt_start_s * 1000))
    dur_ms = int(round(duration_s * 1000))
    return f"dt{dt_ms:04d}ms__dur{dur_ms:04d}ms"


def _format_band_suffix(band_hz: list[float]) -> str:
    low = int(round(band_hz[0]))
    high = int(round(band_hz[1]))
    return f"b{low:04d}_{high:04d}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON inválido en {path}: {exc}") from exc


def _extract_verdict(payload: dict[str, Any]) -> str | None:
    for key in ("verdict", "overall_verdict", "status", "decision", "result"):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, dict):
            if isinstance(value.get("overall"), str):
                return value["overall"]
            if isinstance(value.get("verdict"), str):
                return value["verdict"]
        if isinstance(value, str):
            return value
    return None


def _stage_summary_verdict(stage_dir: Path) -> str | None:
    summary_path = stage_dir / "stage_summary.json"
    if not summary_path.exists():
        return None
    payload = _read_json(summary_path)
    return _extract_verdict(payload)


def _real_v0_stage_verdict(stage_dir: Path) -> str | None:
    summary_path = stage_dir / "stage_summary.json"
    if not summary_path.exists():
        return None
    payload = _read_json(summary_path)
    verdict = payload.get("verdict")
    if isinstance(verdict, dict):
        overall = verdict.get("overall")
        if isinstance(overall, str):
            return overall
        nested = verdict.get("verdict")
        if isinstance(nested, str):
            return nested
    if isinstance(verdict, str):
        return verdict
    return _extract_verdict(payload)


def _real_v0_ready(run_dir: Path) -> tuple[bool, str | None]:
    stage_dir = run_dir / "ringdown_real_v0"
    output_path = stage_dir / "outputs" / "real_v0_events_list.json"
    if output_path.exists():
        return True, None
    return False, _real_v0_stage_verdict(stage_dir)


def _real_v0_ready_by_artifact(run: str) -> bool:
    runs_root = get_runs_root()
    sentinel = (
        runs_root
        / run
        / "ringdown_real_v0"
        / "outputs"
        / "real_v0_events_list.json"
    )
    return sentinel.exists()


def _locate_run_valid_path(run_dir: Path) -> Path:
    preferred = run_dir / "RUN_VALID" / "outputs" / "run_valid.json"
    if preferred.exists():
        return preferred
    fallback = run_dir / "RUN_VALID" / "verdict.json"
    return fallback


def _run_command(
    command: list[str],
    repo_root: Path,
    env: dict[str, str],
    label: str,
) -> None:
    print(f"RUN_REAL: exec {label}: {' '.join(command)}")
    result = subprocess.run(command, cwd=repo_root, env=env, check=False)
    if result.returncode != 0:
        print(f"FAILED {label} exit={result.returncode}")
        raise RuntimeError(f"falló comando: {' '.join(command)} (exit={result.returncode})")
    print(f"completed {label} exit=0")


def _run_command_capture(
    command: list[str],
    repo_root: Path,
    env: dict[str, str],
    output_path: Path,
    label: str,
) -> None:
    print(f"RUN_REAL: exec {label}: {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        print(f"FAILED {label} exit={result.returncode}")
        raise RuntimeError(f"falló comando: {' '.join(command)} (exit={result.returncode})")
    print(f"completed {label} exit=0")


def _stage_names(dt_start_s: float, duration_s: float, band_hz: list[float]) -> dict[str, str]:
    suffix = _compute_suffix(dt_start_s, duration_s)
    band_suffix = _format_band_suffix(band_hz)
    full_suffix = f"{suffix}__{band_suffix}"
    return {
        "suffix": full_suffix,
        "window": f"ringdown_real_ringdown_window_v1__{full_suffix}",
        "observables": f"ringdown_real_observables_v0__{full_suffix}",
        "features": f"ringdown_real_features_v0__{full_suffix}",
        "inference": f"ringdown_real_inference_v0__{full_suffix}",
    }


def _should_run_stage(stage_dir: Path, force: bool) -> bool:
    verdict = _stage_summary_verdict(stage_dir)
    if verdict == "PASS" and not force:
        return False
    return True


def _collect_artifact(path: Path, run_dir: Path) -> dict[str, str]:
    rel = path.relative_to(run_dir)
    return {"path": str(rel), "sha256": sha256_file(path)}


def _final_verdict(
    verdicts: dict[str, str | None],
    inference_report: Path,
) -> str:
    required = (
        "RUN_VALID",
        "ringdown_real_v0",
        "window",
        "observables",
        "features",
        "inference",
    )
    for key in required:
        if verdicts.get(key) != "PASS":
            return "FAIL"
    decision_verdict = None
    if inference_report.exists():
        payload = _read_json(inference_report)
        decision = payload.get("decision")
        if isinstance(decision, dict):
            decision_verdict = decision.get("verdict")
        elif isinstance(payload.get("verdict"), str):
            decision_verdict = payload.get("verdict")
    if decision_verdict == "INSPECT":
        return "INSPECT"
    return "PASS"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run canonical real ringdown pipeline")
    ap.add_argument("--run", required=True, help="run_id")
    ap.add_argument("--dt-start-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=0.25)
    ap.add_argument("--band-hz", default="150,400", type=_parse_band_hz)
    ap.add_argument("--do-exp08", action="store_true", help="run exp_ringdown_08_real_v0_smoke")
    ap.add_argument("--dry-run", action="store_true", help="print plan only")
    ap.add_argument("--force", action="store_true", help="re-run stages if dir exists")

    args = ap.parse_args()

    runs_root = get_runs_root()
    validate_run_id(args.run, runs_root)
    run_dir = get_run_dir(args.run, base_dir=runs_root).resolve()

    try:
        run_valid_payload = require_run_valid(runs_root, args.run)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    stage_names = _stage_names(args.dt_start_s, args.duration_s, args.band_hz)

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    commands = {
        "ringdown_real_v0": [
            "python",
            "stages/ringdown_real_v0_stage.py",
            "--run",
            args.run,
        ],
        "window": [
            "python",
            "stages/ringdown_real_ringdown_window_v1_stage.py",
            "--run",
            args.run,
            "--dt-start-s",
            str(args.dt_start_s),
            "--duration-s",
            str(args.duration_s),
            "--stage-name",
            stage_names["window"],
        ],
        "observables": [
            "python",
            "stages/ringdown_real_observables_v0_stage.py",
            "--run",
            args.run,
            "--window-stage",
            stage_names["window"],
            "--stage-name",
            stage_names["observables"],
            "--band-hz",
            f"{args.band_hz[0]},{args.band_hz[1]}",
        ],
        "features": [
            "python",
            "stages/ringdown_real_features_v0_stage.py",
            "--run",
            args.run,
            "--window-stage",
            stage_names["window"],
            "--stage-name",
            stage_names["features"],
        ],
        "inference": [
            "python",
            "stages/ringdown_real_inference_v0_stage.py",
            "--run",
            args.run,
            "--window-stage",
            stage_names["window"],
            "--stage-name",
            stage_names["inference"],
            "--band-hz",
            f"{args.band_hz[0]},{args.band_hz[1]}",
        ],
    }

    exp08_events_array = (
        runs_root
        / args.run
        / "ringdown_real_v0"
        / "outputs"
        / "real_v0_events_array.json"
    )
    exp08_command = [
        "python",
        "experiment/ringdown/exp_ringdown_08_real_v0_smoke.py",
        "--run",
        args.run,
        "--real-v0-events-json",
        str(exp08_events_array),
    ]

    real_v0_ok = _real_v0_ready_by_artifact(args.run)
    real_v0_verdict = "PASS" if real_v0_ok else None
    if not real_v0_ok:
        real_v0_ok, real_v0_verdict = _real_v0_ready(run_dir)

    if args.do_exp08 and not exp08_events_array.exists():
        print(
            f"MISSING_REAL_V0_EVENTS_ARRAY_FOR_EXP08: {exp08_events_array}",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print("[DRY-RUN] Stage names:")
        print(f"- window: {stage_names['window']}")
        print(f"- observables: {stage_names['observables']}")
        print(f"- features: {stage_names['features']}")
        print(f"- inference: {stage_names['inference']}")
        print("[DRY-RUN] Commands:")
        if not real_v0_ok:
            print(f"- ringdown_real_v0: {' '.join(commands['ringdown_real_v0'])}")
        for key in ("window", "observables", "features", "inference"):
            stage_dir = run_dir / stage_names[key]
            if _should_run_stage(stage_dir, args.force):
                print(f"- {key}: {' '.join(commands[key])}")
        if args.do_exp08:
            print(f"- exp08: {' '.join(exp08_command)}")
        return 0

    print(
        "RUN_REAL: start"
        f" run_id={args.run}"
        f" dt-start-s={args.dt_start_s}"
        f" duration-s={args.duration_s}"
        f" band-hz={args.band_hz}"
        f" do-exp08={args.do_exp08}"
        f" dry-run={args.dry_run}"
    )
    print("RUN_REAL: stage_names")
    print(f"- window: {stage_names['window']}")
    print(f"- observables: {stage_names['observables']}")
    print(f"- features: {stage_names['features']}")
    print(f"- inference: {stage_names['inference']}")

    ringdown_real_v0_dir = run_dir / "ringdown_real_v0"
    if not real_v0_ok:
        try:
            _run_command(commands["ringdown_real_v0"], repo_root, env, "ringdown_real_v0")
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    for key in ("window", "observables", "features", "inference"):
        stage_dir = run_dir / stage_names[key]
        if not _should_run_stage(stage_dir, args.force):
            continue
        try:
            _run_command(commands[key], repo_root, env, key)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    exp08_output = None
    exp08_verdict = None
    if args.do_exp08:
        exp08_output = (
            run_dir / "experiment" / "exp_ringdown_08_real_v0_smoke" / "output.txt"
        )
        try:
            _run_command_capture(exp08_command, repo_root, env, exp08_output, "exp08")
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        exp08_verdict = "PASS"

    run_valid_path = _locate_run_valid_path(run_dir)
    stages: dict[str, Any] = {
        "RUN_VALID": {
            "verdict": _extract_verdict(run_valid_payload),
            "path": str(run_valid_path.relative_to(run_dir)),
            "sha256": sha256_file(run_valid_path),
        }
    }

    ringdown_real_v0_summary = ringdown_real_v0_dir / "stage_summary.json"
    if ringdown_real_v0_summary.exists():
        stages["ringdown_real_v0"] = {
            "verdict": real_v0_verdict,
            "path": str(ringdown_real_v0_summary.relative_to(run_dir)),
            "sha256": sha256_file(ringdown_real_v0_summary),
        }
    else:
        stages["ringdown_real_v0"] = {"verdict": real_v0_verdict}

    for key in ("window", "observables", "features", "inference"):
        stage_dir = run_dir / stage_names[key]
        summary_path = stage_dir / "stage_summary.json"
        entry: dict[str, Any] = {
            "name": stage_names[key],
            "verdict": _stage_summary_verdict(stage_dir),
            "artifacts": [],
        }
        if summary_path.exists():
            entry["artifacts"].append(_collect_artifact(summary_path, run_dir))
        stages[key] = entry

    if args.do_exp08:
        exp08_entry: dict[str, Any] = {"verdict": exp08_verdict, "artifacts": []}
        if exp08_output is not None:
            exp08_entry["artifacts"].append(_collect_artifact(exp08_output, run_dir))
        stages["exp08"] = exp08_entry

    inference_report = (
        run_dir / stage_names["inference"] / "outputs" / "inference_report.json"
    )

    summary = {
        "run_id": args.run,
        "params": {
            "dt_start_s": float(args.dt_start_s),
            "duration_s": float(args.duration_s),
            "band_hz": args.band_hz,
        },
        "stages": stages,
        "final_verdict": _final_verdict(
            {
                "RUN_VALID": stages["RUN_VALID"].get("verdict"),
                "ringdown_real_v0": stages["ringdown_real_v0"].get("verdict"),
                "window": stages["window"].get("verdict"),
                "observables": stages["observables"].get("verdict"),
                "features": stages["features"].get("verdict"),
                "inference": stages["inference"].get("verdict"),
            },
            inference_report,
        ),
    }

    summary_path = run_dir / "REAL_PIPELINE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
