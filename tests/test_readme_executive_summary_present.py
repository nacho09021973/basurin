from pathlib import Path

def test_readme_has_sovereign_disclaimer_and_exec_summary():
    p = Path("README.md")
    assert p.exists(), "README.md missing"
    s = p.read_text(encoding="utf-8")

    assert "BASURIN_README_SUPER.md" in s
    assert "resumen ejecutivo no normativo" in s.lower()
    assert "Resumen ejecutivo (contract-first)" in s
    assert "RUN_VALID" in s
    assert "ringdown_synth" in s
    assert "runs/<run_id>/" in s
