from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_out_root(out_root: str, runs_root: Path | str = "runs") -> Path:
        # Canon: if user passes "runs", it must respect BASURIN_RUNS_ROOT.
    # runs_root may be overridden by env via get_runs_root().
    runs_root_path = Path(get_runs_root()).expanduser()
    if not runs_root_path.is_absolute():
        runs_root_path = (Path.cwd() / runs_root_path).resolve()
    else:
        runs_root_path = runs_root_path.resolve()

    if out_root == "runs" or out_root == str(runs_root) or out_root == runs_root_path.name:
        return runs_root_path
    candidate = Path(out_root)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.name == runs_root_path.name:
        return candidate
    # Permitir roots alternativos tipo runs_tmp/... si están bajo el repo
    # y el primer segmento es runs* (kill-switch: no escribir fuera del repo).
    repo_root = Path.cwd().resolve()
    try:
        rel = candidate.relative_to(repo_root)
        if rel.parts and rel.parts[0].startswith("runs"):
            return candidate
    except ValueError:
        pass
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


def get_runs_root() -> Path:
    """Return the runs root directory, respecting BASURIN_RUNS_ROOT env var."""
    env_root = os.environ.get("BASURIN_RUNS_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path("runs")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256(path: str | Path) -> str:
    return sha256_file(Path(path))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def load_run_valid_verdict(run_root: Path) -> tuple[bool, Path]:
    preferred = run_root / "RUN_VALID" / "verdict.json"
    legacy = run_root / "RUN_VALID" / "outputs" / "run_valid.json"
    verdict_path = preferred if preferred.exists() else legacy
    if not verdict_path.exists():
        return False, verdict_path
    payload = read_json(verdict_path)
    verdict = payload.get("verdict", payload.get("overall_verdict", payload.get("status")))
    return verdict == "PASS", verdict_path


def get_run_dir(run_id: str, base_dir: Path | str | None = None) -> Path:
    if base_dir is None:
        base_dir = get_runs_root()
    return Path(base_dir) / run_id


def require_run_valid(out_root: Path, run_id: str) -> dict[str, Any]:
    run_dir = out_root / run_id
    preferred = run_dir / "RUN_VALID" / "verdict.json"
    legacy = run_dir / "RUN_VALID" / "outputs" / "run_valid.json"
    run_valid_path = preferred if preferred.exists() else legacy
    if not run_valid_path.exists():
        raise RuntimeError(f"[BASURIN ABORT] RUN_VALID missing at {run_valid_path}")
    try:
        payload = json.loads(run_valid_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"[BASURIN ABORT] RUN_VALID invalid JSON at {run_valid_path}: {exc}"
        ) from exc
    verdict = payload.get("verdict")
    if verdict is None:
        verdict = payload.get("overall_verdict")
    if verdict is None:
        verdict = payload.get("status")
    if verdict != "PASS":
        raise RuntimeError(f"[BASURIN ABORT] RUN_VALID={verdict} at {run_valid_path}")
    return payload


def stage_dir(run_id: str, stage_name: str, base_dir: Path | str | None = None) -> Path:
    return get_run_dir(run_id, base_dir=base_dir) / stage_name


def ensure_stage_dirs(
    run_id: str, stage_name: str, base_dir: Path | str | None = None
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


def assert_within_runs(run_dir: Path, path: Path | str) -> None:
    if isinstance(path, str):
        path = Path(path)
    run_dir_resolved = run_dir.resolve()
    path_resolved = path.resolve()
    try:
        path_resolved.relative_to(run_dir_resolved)
    except ValueError as exc:
        raise ValueError(f"Path {path} is not under run dir {run_dir_resolved}") from exc


def resolve_geometry_path(
    run: str,
    geometry_file: str,
    base_dir: Path | str | None = None,
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
    artifacts: Mapping[str, Path] | None = None,
    extra: Mapping[str, Any] | None = None,
    *,
    stage: str | None = None,
    artifact_relpaths: list[Path] | None = None,
) -> Path:
    if artifact_relpaths is not None:
        stage_name = stage or stage_dir.name
        manifest_path = stage_dir / "manifest.json"
        if not manifest_path.exists() and any(str(r) == "manifest.json" for r in artifact_relpaths):
            write_json_atomic(manifest_path, {"version": "manifest.v1", "stage": stage_name, "artifacts": {}})

        artifact_hashes: dict[str, dict[str, str]] = {}
        for rel in artifact_relpaths:
            rel_str = str(rel)
            artifact_hashes[rel_str] = {"sha256": compute_sha256(stage_dir / rel)}

        manifest = {
            "version": "manifest.v1",
            "stage": stage_name,
            "artifacts": artifact_hashes,
        }
        write_json_atomic(manifest_path, manifest)
        return manifest_path

    if artifacts is None:
        raise ValueError("artifacts must be provided when artifact_relpaths is not used")

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
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


def write_stage_summary(
    stage_dir: Path,
    summary_dict: dict[str, Any] | None = None,
    *,
    stage: str | None = None,
    impl_module: str | None = None,
    params: dict | None = None,
    inputs: dict | None = None,
    outputs: dict | None = None,
    verdict: str | None = None,
    model_family: str | None = None,
    epistemic_status: str | None = None,
) -> Path:
    if summary_dict is not None:
        if "created" not in summary_dict:
            summary_dict["created"] = utc_now_iso()
        if "stage" not in summary_dict:
            summary_dict["stage"] = stage_dir.name
        if "run" not in summary_dict:
            summary_dict["run"] = stage_dir.parent.name

        summary_path = stage_dir / "stage_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2, sort_keys=True)
            f.write("\n")
        return summary_path

    if None in (stage, impl_module, params, inputs, outputs, verdict, model_family, epistemic_status):
        raise ValueError("Missing required keyword arguments for stage_summary.v1")

    summary_v1 = {
        "version": "stage_summary.v1",
        "stage": stage,
        "impl": {"module": impl_module, "version": "v0"},
        "model": {
            "family": model_family,
            "epistemic_status": epistemic_status,
        },
        "params": params,
        "inputs": inputs,
        "outputs": outputs,
        "verdict": verdict,
    }
    summary_path = stage_dir / "stage_summary.json"
    write_json_atomic(summary_path, summary_v1)
    return summary_path


def _contract_error(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def _coerce_vector(vec: Any) -> Optional[list[float]]:
    if isinstance(vec, list):
        return [float(x) for x in vec]
    if isinstance(vec, dict):
        for key in ["values", "vector", "data", "features"]:
            if key in vec:
                return _coerce_vector(vec[key])
    return None


def _coerce_matrix(rows: Any, path: Path, key_label: str) -> np.ndarray:
    try:
        mat = np.asarray(rows, dtype=float)
    except Exception as exc:
        _contract_error(f"Formato inválido en {path}: '{key_label}' no es matriz numérica ({exc}).")
    if mat.ndim != 2:
        _contract_error(f"Formato inválido en {path}: '{key_label}' debe ser 2D.")
    return mat


def _resolve_feature_key(
    obj: Any,
    kind: str,
    feature_key_hint: Optional[str],
) -> str:
    if isinstance(obj, dict):
        meta = obj.get("meta")
        if isinstance(meta, dict) and meta.get("feature_key"):
            return str(meta.get("feature_key"))
        if obj.get("feature_key"):
            return str(obj.get("feature_key"))
    if feature_key_hint:
        return str(feature_key_hint)
    if kind == "atlas":
        return "ratios"
    if kind in {"features", "ringdown"}:
        return "tangentes_locales_v1"
    _contract_error(f"kind inválido: {kind}")
    return ""


def _extract_meta(obj: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if not isinstance(obj, dict):
        return meta
    if "meta" in obj and isinstance(obj["meta"], dict):
        meta.update(obj["meta"])
    for mk in ["feature_key", "schema_version", "created", "source_atlas", "ids_source"]:
        if mk in obj:
            meta[mk] = obj[mk]
    if "columns" in obj:
        meta["columns"] = obj["columns"]
    if "feature_names" in obj and "columns" not in meta:
        meta["columns"] = obj["feature_names"]
    return meta


def load_feature_json(
    path: Path,
    kind: str,
    feature_key_hint: Optional[str] = None,
) -> tuple[Optional[list[Any]], np.ndarray, dict[str, Any]]:
    """Carga JSON y devuelve ids (si existen), matriz y metadatos mínimos.

    Formatos soportados (documentados):
      - Canonical: dict con ids + X/Y + meta (schema_version/feature_key).
      - Legacy: dict con points/theories/events o lista de dicts con id + x/y/features.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and isinstance(obj.get("features"), dict):
        obj = obj["features"]

    key_primary = "X" if kind == "atlas" else "Y"
    key_secondary = "x" if kind == "atlas" else "y"
    feature_key = _resolve_feature_key(obj, kind, feature_key_hint)
    meta = _extract_meta(obj)
    resolved_from_legacy = False

    schema_version = meta.get("schema_version")
    if schema_version is None and isinstance(obj, dict) and obj.get("schema_version"):
        schema_version = obj.get("schema_version")
    if schema_version is None:
        schema_version = "legacy"
        resolved_from_legacy = True
    meta["schema_version"] = str(schema_version)

    if feature_key and "feature_key" not in meta:
        meta["feature_key"] = feature_key

    if isinstance(obj, dict) and "ids" in obj:
        ids = obj["ids"]
        for candidate in (key_primary, key_secondary):
            if candidate in obj:
                mat = _coerce_matrix(obj[candidate], path, candidate)
                if ids is not None and len(ids) != mat.shape[0]:
                    _contract_error(f"{path}: ids no coincide con filas de '{candidate}'.")
                if "columns" not in meta and meta.get("feature_key"):
                    meta["columns"] = [f"{meta['feature_key']}_{i}" for i in range(mat.shape[1])]
                meta["resolved_from_legacy"] = resolved_from_legacy
                return ids, mat, meta
        if kind in {"features", "ringdown"} and "X" in obj:
            resolved_from_legacy = True
            mat = _coerce_matrix(obj["X"], path, "X")
            if ids is not None and len(ids) != mat.shape[0]:
                _contract_error(f"{path}: ids no coincide con filas de 'X'.")
            if "columns" not in meta and meta.get("feature_key"):
                meta["columns"] = [f"{meta['feature_key']}_{i}" for i in range(mat.shape[1])]
            meta["resolved_from_legacy"] = resolved_from_legacy
            meta["legacy_matrix_key"] = "X"
            return ids, mat, meta

    rows = None
    if isinstance(obj, dict):
        for k in ["points", "theories", "events", "rows", "data"]:
            if isinstance(obj.get(k), list):
                rows = obj[k]
                break
    elif isinstance(obj, list):
        rows = obj

    if isinstance(rows, list):
        ids: list[Any] = []
        vectors: list[list[float]] = []
        for row in rows:
            if not isinstance(row, dict):
                _contract_error(f"{path}: filas deben ser dicts con id y '{key_secondary}'.")
            rid = row.get("id", row.get("uid", row.get("name")))
            vec = row.get(key_secondary, row.get(key_primary))
            if vec is None and kind == "atlas":
                for fallback in ["features", "ratios", "vector"]:
                    if fallback in row:
                        vec = row[fallback]
                        break
            if vec is None and isinstance(row.get("theories"), dict):
                theories = row["theories"]
                if feature_key in theories:
                    vec = theories[feature_key]
                elif len(theories) == 1:
                    only_key, only_val = next(iter(theories.items()))
                    vec = only_val
                    if "feature_key" not in meta:
                        meta["feature_key"] = only_key
                else:
                    _contract_error(f"{path}: theories no contiene '{feature_key}'.")
            if vec is None:
                _contract_error(f"{path}: falta vector '{key_secondary}' en fila.")
            vec_list = _coerce_vector(vec)
            if vec_list is None:
                _contract_error(f"{path}: vector '{key_secondary}' inválido.")
            ids.append(rid)
            vectors.append(vec_list)
        mat = _coerce_matrix(vectors, path, key_secondary)
        if "columns" not in meta and meta.get("feature_key"):
            meta["columns"] = [f"{meta['feature_key']}_{i}" for i in range(mat.shape[1])]
        meta["resolved_from_legacy"] = True
        return ids, mat, meta

    _contract_error(
        f"Formato no reconocido en {path}. "
        f"Se esperaba dict con ids+{key_primary} o lista de dicts con '{key_secondary}'."
    )
    raise SystemExit(2)
