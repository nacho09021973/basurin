from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


def _import_stage03():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "03_sturm_liouville.py"
    spec = spec_from_file_location("sturm_liouville_stage03", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bf_skip_in_sweep_delta_border_point():
    stage03 = _import_stage03()
    assert hasattr(stage03, "bf_filter_sweep_deltas")

    d = 3
    bf_bound = d / 2.0
    deltas = np.linspace(bf_bound, bf_bound + 0.4, 5)

    valid_deltas, bf_per_delta, bf_skipped, _ = stage03.bf_filter_sweep_deltas(
        d, deltas
    )

    assert bf_skipped == 1
    assert bf_per_delta[0]["bf_ok"] is False
    assert bf_per_delta[0]["skipped"] is True
    assert valid_deltas[0] == float(deltas[1])
    assert valid_deltas == sorted(valid_deltas)
