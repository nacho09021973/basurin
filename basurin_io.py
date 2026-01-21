from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    files: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for label, path in artifacts.items():
        rel = _relative_to_stage(stage_dir, path)
        files[label] = rel
        hashes[rel] = sha256_file(path)

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
