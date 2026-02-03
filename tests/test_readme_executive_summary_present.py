from pathlib import Path


def test_readme_executive_summary_present() -> None:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    content = readme.read_text(encoding="utf-8")
    lowered = content.lower()

    assert "BASURIN_README_SUPER.md" in content, "README must reference BASURIN_README_SUPER.md"
    assert (
        "resumen ejecutivo no normativo" in lowered
    ), "README must include 'resumen ejecutivo no normativo' (case-insensitive)"
    assert (
        "Resumen ejecutivo (contract-first)" in content
    ), "README must include 'Resumen ejecutivo (contract-first)'"
    assert "RUN_VALID" in content, "README must mention RUN_VALID"
    assert "ringdown_synth" in content, "README must mention ringdown_synth"
    assert "runs/<run_id>/" in content, "README must mention runs/<run_id>/"
