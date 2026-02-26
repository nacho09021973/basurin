from __future__ import annotations


def test_s4d_module_import_exposes_main() -> None:
    import mvp.s4d_kerr_from_multimode as s4d

    assert callable(s4d.main)
