from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_out_root(out_root: str, runs_root: Path | str = "runs") -> Path:
    runs_root_path = Path(runs_root).resolve()
    if out_root == str(runs_root):
        return runs_root_path
    candidate = Path(out_root)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    try:
        candidate.relative_to(runs_root_path)
    except ValueError as exc:
        raise ValueError(
            f"out_root must be '{runs_root_path.name}' or a subdir under {runs_root_path}"
        ) from exc
    return candidate


def validate_run_id(run_id: str, out_root: Path) -> None:
    run_id_path = Path(run_id)
    if run_id_path.is_absolute() or ".." in run_id_path.parts:
        raise ValueError("run_id must be a relative path without '..'")
    run_path = (out_root / run_id).resolve()
    try:
        run_path.relative_to(out_root.resolve())
    except ValueError as exc:
        raise ValueError("run_id escapes out_root") from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_run_dir(run_id: str, base_dir: Path | str = "runs") -> Path:
    return Path(base_dir) / run_id


def stage_dir(run_id: str, stage_name: str, base_dir: Path | str = "runs") -> Path:
    return get_run_dir(run_id, base_dir=base_dir) / stage_name


def ensure_stage_dirs(
    run_id: str, stage_name: str, base_dir: Path | str = "runs"
) -> tuple[Path, Path]:
    sdir = stage_dir(run_id, stage_name, base_dir=base_dir)
    outputs_dir = sdir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return sdir, outputs_dir


def spectrum_outputs_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "spectrum" / "outputs" / "spectrum.h5"


def spectrum_legacy_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "spectrum" / "spectrum.h5"


def resolve_spectrum_path(run_dir: Path | str) -> Path:
    """Resolve the spectrum path for a run directory.

    Canonical:
      runs/<run>/spectrum/outputs/spectrum.h5

    Legacy (read-only fallback):
      runs/<run>/spectrum/spectrum.h5
    """
    outputs = spectrum_outputs_path(run_dir)
    legacy = spectrum_legacy_path(run_dir)
    if outputs.exists():
        return outputs
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        "spectrum.h5 no encontrado; rutas esperadas: "
        f"{outputs} | {legacy}"
    )


def assert_within_runs(run_dir: Path, path: Path) -> None:
    run_dir_resolved = run_dir.resolve()
    path_resolved = path.resolve()
    try:
        path_resolved.relative_to(run_dir_resolved)
    except ValueError as exc:
        raise ValueError(f"Path {path} is not under run dir {run_dir_resolved}") from exc


def resolve_geometry_path(
    run: str,
    geometry_file: str,
    base_dir: Path | str = "runs",
) -> tuple[Path, str, str | None, str]:
    """Resolve the geometry path for a run.

    Canonical:
      runs/<run>/geometry/outputs/<file>

    Legacy (read-only fallback):
      runs/<run>/geometry/<file>
    """
    if ".." in Path(geometry_file).parts:
        raise ValueError(
            "geometry_file no puede contener '..'; usa rutas dentro de "
            f"runs/{run}/geometry/."
        )

    run_dir = get_run_dir(run, base_dir=base_dir)
    is_absolute = Path(geometry_file).is_absolute()
    starts_with_runs = geometry_file.startswith("runs/")

    if is_absolute or starts_with_runs:
        candidate = Path(geometry_file)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()

        assert_within_runs(run_dir, candidate)

        geometry_dir = (run_dir / "geometry").resolve()
        try:
            candidate.relative_to(geometry_dir)
        except ValueError as exc:
            raise ValueError(
                f"geometry_file sale de runs/{run}/geometry/: {geometry_file}"
            ) from exc

        resolved_abs = candidate
        geometry_path = geometry_file if starts_with_runs else str(resolved_abs)
        geometry_resolution = "absolute"
        input_geometry_absolute = str(resolved_abs) if is_absolute else None
        return resolved_abs, geometry_path, input_geometry_absolute, geometry_resolution

    outputs_dir = run_dir / "geometry" / "outputs"
    outputs_path = outputs_dir / geometry_file
    legacy_path = run_dir / "geometry" / geometry_file

    if outputs_path.exists():
        resolved_abs = outputs_path.resolve()
        geometry_path = f"runs/{run}/geometry/outputs/{geometry_file}"
        geometry_resolution = "canonical"
    elif legacy_path.exists():
        resolved_abs = legacy_path.resolve()
        geometry_path = f"runs/{run}/geometry/{geometry_file}"
        geometry_resolution = "legacy"
    else:
        raise FileNotFoundError(
            "geometry.h5 no encontrado; rutas esperadas: "
            f"{outputs_path} | {legacy_path}. "
            f"Verifica --geometry-file o genera geometría en runs/{run}/geometry/outputs/."
        )

    input_geometry_absolute = None
    return resolved_abs, geometry_path, input_geometry_absolute, geometry_resolution


def _relative_to_stage(stage_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(stage_dir))
    except ValueError as exc:
        raise ValueError(f"Artifact {path} is not under stage dir {stage_dir}") from exc


def write_manifest(
    stage_dir: Path,
    artifacts: Mapping[str, Path],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    stage_dir_resolved = stage_dir.resolve()
    run_dir = stage_dir_resolved.parent
    env_runs_root = None
    if "BASURIN_RUNS_ROOT" in os.environ:
        env_runs_root = Path(os.environ["BASURIN_RUNS_ROOT"]).expanduser()
        if not env_runs_root.is_absolute():
            env_runs_root = (Path.cwd() / env_runs_root).resolve()
        else:
            env_runs_root = env_runs_root.resolve()
    runs_root = env_runs_root if env_runs_root is not None else run_dir.parent
    try:
        run_dir.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError(
            f"stage_dir must be under {runs_root}/<run_id>/, got {stage_dir_resolved}"
        ) from exc
    assert_within_runs(run_dir, stage_dir_resolved)

    files: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for label, path in artifacts.items():
        assert_within_runs(run_dir, path)
        resolved_path = path.resolve()
        rel = _relative_to_stage(stage_dir_resolved, resolved_path)
        files[label] = rel
        hashes[rel] = sha256_file(resolved_path)

    manifest: dict[str, Any] = {
        "stage": stage_dir.name,
        "run": stage_dir.parent.name,
        "created": utc_now_iso(),
        "files": files,
        "hashes": hashes,
    }
    if extra:
        manifest.update(extra)

    manifest_path = stage_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def write_stage_summary(stage_dir: Path, summary_dict: dict[str, Any]) -> Path:
    if "created" not in summary_dict:
        summary_dict["created"] = utc_now_iso()
    if "stage" not in summary_dict:
        summary_dict["stage"] = stage_dir.name
    if "run" not in summary_dict:
        summary_dict["run"] = stage_dir.parent.name

    summary_path = stage_dir / "stage_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_dict, f, indent=2)
    return summary_path
