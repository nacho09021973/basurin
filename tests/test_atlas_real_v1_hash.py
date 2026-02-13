import hashlib
from pathlib import Path

def test_atlas_real_v1_s4_sha256_golden():
    p = Path("docs/ringdown/atlas/atlas_real_v1_s4.json")
    assert p.exists(), f"Missing atlas file: {p}"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == "815e29338d15f0c303c3d24ab1ed293a86911bac4e72f0ec574a9888be7a9221"
