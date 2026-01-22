from __future__ import annotations

from pathlib import Path


def test_experiment_scripts_do_not_write_outside_runs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    experiment_root = repo_root / "experiment"
    python_files = list(experiment_root.rglob("*.py"))

    forbidden_markers = ["/tmp", "../"]
    forbidden_absolute_outputs = ['"/outputs', "'/outputs"]

    violations: list[str] = []
    for py_file in python_files:
        text = py_file.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            if marker in text:
                violations.append(f"{py_file}: contiene '{marker}'")
        for marker in forbidden_absolute_outputs:
            if marker in text:
                violations.append(f"{py_file}: contiene path absoluto a outputs ({marker})")

    assert not violations, "Violaciones de guardarraíles:\n" + "\n".join(sorted(violations))
