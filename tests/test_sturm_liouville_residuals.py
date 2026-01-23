from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


def _load_sturm_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "03_sturm_liouville.py"
    spec = spec_from_file_location("sturm_liouville", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_residuals_backward_error_exact_pair():
    sl = _load_sturm_module()
    K = np.array([[2.0, 0.0], [0.0, 3.0]])
    W = np.eye(2)
    eigenvalues = np.array([2.0, 3.0])
    eigenvectors = np.eye(2)

    cfg = sl.Config(run="test")
    result = sl.validate_residuals(K, W, eigenvalues, eigenvectors, cfg=cfg)

    assert result["residual_metric"] == "backward_error"
    assert result["residual_max"] < 1e-12
    assert result["residual_argmax_mode"] == 0
    assert result["residual_argmax_value"] == 0.0
