from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# .txt permitidos en la raíz (configuración/metadata estática)
ALLOWED_ROOT_TXT = {
    "requirements.txt",
}

def test_no_top_level_experiment_package():
    assert not (ROOT / "experiment").exists(), "No debe existir ./experiment; usar mvp/experiment"

def test_no_txt_outputs_in_repo_root():
    bad = sorted(p.name for p in ROOT.glob("*.txt") if p.name not in ALLOWED_ROOT_TXT)
    assert bad == [], f"Outputs .txt no permitidos en raíz: {bad}"
