"""Tests for config/external_oracle_gw150914.yaml and docs/validation/gw150914_external_oracle.md.

Covers:
- YAML parsability and top-level key presence
- Schema integrity (all required fields)
- applies_to / checks consistency (no drift)
- Verdict strings in checks sync with golden_geometry_spec constants
- spec_version sync with GOLDEN_GEOMETRY_SPEC_VERSION
- Markdown file existence and required sections
- Canonical artifact path patterns in the markdown
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# pyyaml is a project dependency (requirements.txt) but may not be present in
# the minimal pytest venv.  Add the system dist-packages so the import works
# without modifying the test runner environment.
try:
    import yaml
except ModuleNotFoundError:
    import sys as _sys
    _sys.path.insert(0, "/usr/lib/python3/dist-packages")
    import yaml  # type: ignore[import]

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.golden_geometry_spec import (
    GOLDEN_GEOMETRY_SPEC_VERSION,
    VERDICT_NO_DATA,
    VERDICT_NOT_UNIQUE,
    VERDICT_ROBUST_UNIQUE,
    VERDICT_UNSTABLE_UNIQUE,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

YAML_PATH = REPO_ROOT / "config" / "external_oracle_gw150914.yaml"
MARKDOWN_PATH = REPO_ROOT / "docs" / "validation" / "gw150914_external_oracle.md"

EXPECTED_ARTIFACTS = {
    "s4g_mode220_geometry_filter",
    "s4h_mode221_geometry_filter",
    "s4i_common_geometry_intersection",
    "s4j_hawking_area_filter",
    "experiment_single_event_golden_robustness",
}

REQUIRED_TOP_LEVEL_FIELDS = {
    "event",
    "spec_version",
    "applies_to",
    "references",
    "checks",
    "interpretation_notes",
}

REQUIRED_CHECK_VERDICT_KEYS = {"pass", "warn", "fail"}

# Robustness verdicts that must appear quoted inside the YAML check strings.
ROBUSTNESS_VERDICT_STRINGS = {
    VERDICT_ROBUST_UNIQUE,
    VERDICT_UNSTABLE_UNIQUE,
    VERDICT_NOT_UNIQUE,
    VERDICT_NO_DATA,
}

REQUIRED_MARKDOWN_SECTIONS = [
    "Propósito",
    "Alcance",
    "PASS",
    "WARN",
    "FAIL",
    "s4g_mode220_geometry_filter",
    "s4h_mode221_geometry_filter",
    "s4i_common_geometry_intersection",
    "s4j_hawking_area_filter",
    "experiment_single_event_golden_robustness",
]


@pytest.fixture(scope="module")
def oracle_yaml() -> dict:
    """Load and return the parsed YAML document."""
    raw = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    return raw


@pytest.fixture(scope="module")
def oracle_block(oracle_yaml) -> dict:
    """Return the inner external_oracle mapping."""
    return oracle_yaml["external_oracle"]


@pytest.fixture(scope="module")
def markdown_text() -> str:
    return MARKDOWN_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. File existence
# ---------------------------------------------------------------------------


def test_yaml_file_exists():
    assert YAML_PATH.exists(), f"YAML not found at {YAML_PATH}"


def test_markdown_file_exists():
    assert MARKDOWN_PATH.exists(), f"Markdown not found at {MARKDOWN_PATH}"


# ---------------------------------------------------------------------------
# 2. YAML parsability and top-level structure
# ---------------------------------------------------------------------------


def test_yaml_loads_without_error():
    """YAML must parse without exceptions."""
    yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))


def test_yaml_has_external_oracle_top_level_key(oracle_yaml):
    assert "external_oracle" in oracle_yaml, (
        "Top-level key 'external_oracle' missing; got: " + str(list(oracle_yaml.keys()))
    )


def test_yaml_top_level_is_a_mapping(oracle_yaml):
    assert isinstance(oracle_yaml, dict)


# ---------------------------------------------------------------------------
# 3. Schema integrity — required fields inside external_oracle
# ---------------------------------------------------------------------------


def test_oracle_block_has_all_required_fields(oracle_block):
    missing = REQUIRED_TOP_LEVEL_FIELDS - set(oracle_block.keys())
    assert not missing, f"Missing required fields in external_oracle: {missing}"


def test_event_is_gw150914(oracle_block):
    assert oracle_block["event"] == "GW150914"


def test_spec_version_matches_golden_geometry_spec(oracle_block):
    """spec_version in the YAML must match GOLDEN_GEOMETRY_SPEC_VERSION."""
    assert oracle_block["spec_version"] == GOLDEN_GEOMETRY_SPEC_VERSION, (
        f"YAML spec_version '{oracle_block['spec_version']}' != "
        f"GOLDEN_GEOMETRY_SPEC_VERSION '{GOLDEN_GEOMETRY_SPEC_VERSION}'"
    )


def test_applies_to_is_a_non_empty_list(oracle_block):
    applies = oracle_block["applies_to"]
    assert isinstance(applies, list) and len(applies) > 0


def test_references_is_a_non_empty_mapping(oracle_block):
    refs = oracle_block["references"]
    assert isinstance(refs, dict) and len(refs) > 0, "references block is empty or not a mapping"


def test_interpretation_notes_is_a_non_empty_list(oracle_block):
    notes = oracle_block["interpretation_notes"]
    assert isinstance(notes, list) and len(notes) > 0


# ---------------------------------------------------------------------------
# 4. applies_to coverage — expected artifacts
# ---------------------------------------------------------------------------


def test_applies_to_contains_all_expected_artifacts(oracle_block):
    actual = set(oracle_block["applies_to"])
    missing = EXPECTED_ARTIFACTS - actual
    assert not missing, f"Artifacts missing from applies_to: {missing}"


def test_applies_to_has_no_unexpected_artifacts(oracle_block):
    actual = set(oracle_block["applies_to"])
    extra = actual - EXPECTED_ARTIFACTS
    assert not extra, (
        f"Unexpected artifacts in applies_to (update EXPECTED_ARTIFACTS if intentional): {extra}"
    )


# ---------------------------------------------------------------------------
# 5. checks / applies_to consistency (no drift)
# ---------------------------------------------------------------------------


def test_checks_covers_all_applies_to_artifacts(oracle_block):
    """Every artifact in applies_to must have an entry in checks."""
    applies = set(oracle_block["applies_to"])
    checks = set(oracle_block["checks"].keys())
    missing = applies - checks
    assert not missing, f"Artifacts in applies_to without a checks entry: {missing}"


def test_checks_has_no_extra_artifacts_vs_applies_to(oracle_block):
    """checks must not contain artifact keys absent from applies_to."""
    applies = set(oracle_block["applies_to"])
    checks = set(oracle_block["checks"].keys())
    extra = checks - applies
    assert not extra, (
        f"checks entries not listed in applies_to (orphan checks): {extra}"
    )


# ---------------------------------------------------------------------------
# 6. Each check has pass / warn / fail keys and non-empty lists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("artifact", sorted(EXPECTED_ARTIFACTS))
def test_check_entry_has_pass_warn_fail_keys(oracle_block, artifact):
    check = oracle_block["checks"].get(artifact, {})
    missing = REQUIRED_CHECK_VERDICT_KEYS - set(check.keys())
    assert not missing, f"Artifact '{artifact}' check missing verdict keys: {missing}"


@pytest.mark.parametrize("artifact", sorted(EXPECTED_ARTIFACTS))
def test_check_entry_verdict_lists_are_non_empty(oracle_block, artifact):
    check = oracle_block["checks"][artifact]
    for verdict_key in REQUIRED_CHECK_VERDICT_KEYS:
        entries = check[verdict_key]
        assert isinstance(entries, list) and len(entries) > 0, (
            f"Artifact '{artifact}': '{verdict_key}' must be a non-empty list"
        )


# ---------------------------------------------------------------------------
# 7. Robustness verdict strings sync with golden_geometry_spec constants
# ---------------------------------------------------------------------------


def _extract_quoted_strings(text: str) -> set[str]:
    """Return all double-quoted substrings found in *text*."""
    return set(re.findall(r'"([^"]+)"', text))


def test_robustness_check_verdict_strings_match_spec_constants(oracle_block):
    """Quoted verdict strings in experiment_single_event_golden_robustness checks
    must match the canonical constants exported from golden_geometry_spec."""
    artifact = "experiment_single_event_golden_robustness"
    check = oracle_block["checks"][artifact]

    all_condition_strings = (
        check.get("pass", []) + check.get("warn", []) + check.get("fail", [])
    )
    found_verdicts: set[str] = set()
    for condition in all_condition_strings:
        found_verdicts |= _extract_quoted_strings(str(condition))

    unknown = found_verdicts - ROBUSTNESS_VERDICT_STRINGS
    assert not unknown, (
        f"Unknown verdict strings in '{artifact}' checks (not in golden_geometry_spec): {unknown}"
    )

    # Every canonical robustness verdict must appear at least once.
    missing = ROBUSTNESS_VERDICT_STRINGS - found_verdicts
    assert not missing, (
        f"Canonical robustness verdicts missing from '{artifact}' checks: {missing}"
    )


# ---------------------------------------------------------------------------
# 8. Markdown section coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("section_text", REQUIRED_MARKDOWN_SECTIONS)
def test_markdown_contains_required_section(markdown_text, section_text):
    assert section_text in markdown_text, (
        f"Expected text '{section_text}' not found in {MARKDOWN_PATH}"
    )


# ---------------------------------------------------------------------------
# 9. Canonical artifact path patterns in the markdown
# ---------------------------------------------------------------------------


def test_markdown_references_canonical_run_relative_paths(markdown_text):
    """The markdown must document run-relative paths following the runs/<run_id>/… convention."""
    # The canonical pattern: runs/<run_id>/<stage>/outputs/<file>.json
    canonical_pattern = re.compile(r"runs/<run_id>/\S+\.json")
    matches = canonical_pattern.findall(markdown_text)
    assert len(matches) >= len(EXPECTED_ARTIFACTS), (
        f"Expected at least {len(EXPECTED_ARTIFACTS)} canonical artifact paths in the markdown, "
        f"found {len(matches)}: {matches}"
    )


def test_markdown_artifact_paths_cover_all_five_stages(markdown_text):
    """Each of the five stage/experiment artifact basenames must appear in the markdown."""
    expected_filenames = [
        "mode220_filter.json",
        "mode221_filter.json",
        "common_intersection.json",
        "hawking_area_filter.json",
        "robustness_summary.json",
    ]
    for fname in expected_filenames:
        assert fname in markdown_text, (
            f"Canonical artifact filename '{fname}' not found in {MARKDOWN_PATH}"
        )
