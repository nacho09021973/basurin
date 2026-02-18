from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_experiment_module():
    try:
        return importlib.import_module("mvp.experiment_t0_sweep_full")
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.ndarray = object
        fake_numpy.load = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("numpy not available in this test"))
        fake_numpy.asarray = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("numpy not available in this test"))
        fake_numpy.all = lambda *_args, **_kwargs: False
        fake_numpy.isfinite = lambda *_args, **_kwargs: False
        fake_numpy.savez = lambda *_args, **_kwargs: None
        fake_numpy.float64 = float
        sys.modules.setdefault("numpy", fake_numpy)
        return importlib.import_module("mvp.experiment_t0_sweep_full")


def test_t0_sweep_full_paths_follow_basurin_runs_root(monkeypatch, tmp_path: Path) -> None:
    exp = _import_experiment_module()

    base_run_id = "base_run"
    runs_root = tmp_path / "runsroot"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    stage_dir, outputs_dir, subruns_root = exp.resolve_experiment_paths(base_run_id)

    expected_stage_dir = runs_root / base_run_id / "experiment" / "t0_sweep_full"
    expected_subruns_root = expected_stage_dir / "runs"

    assert stage_dir == expected_stage_dir
    assert outputs_dir == expected_stage_dir / "outputs"
    assert subruns_root == expected_subruns_root

    assert stage_dir.is_dir()
    assert outputs_dir.is_dir()
    assert subruns_root.is_dir()
