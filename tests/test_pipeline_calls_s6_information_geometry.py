from pathlib import Path


def test_pipeline_calls_s6_information_geometry() -> None:
    txt = Path("mvp/pipeline.py").read_text(encoding="utf-8")
    assert '_run_stage("s6_information_geometry.py"' in txt
    assert 's6_args: list[str] = ["--run", run_id]' in txt
    assert '_run_stage(\n        "s6b_information_geometry_ranked.py"' in txt
