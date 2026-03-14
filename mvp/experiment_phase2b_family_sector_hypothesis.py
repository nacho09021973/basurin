#!/usr/bin/env python3
"""Apply an explicit, auditable family/theory -> sector hypothesis table on the supported cohort."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    h5py = None  # type: ignore[assignment]

from basurin_io import write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths

STAGE = "experiment/phase2b_family_sector_hypothesis"
SCHEMA_VERSION = "family_sector_hypothesis_v1"
RULES_SCHEMA_VERSION = "family_to_sector_rules_v1"
DEFAULT_INPUT_NAME = "phase1_geometry_cohort.h5"
DEFAULT_FAMILY_MAP_NAME = "family_map_v1.json"
DEFAULT_RULES_PATH = "docs/experiments/family_to_sector_rules_v1.json"
DEFAULT_OUTPUT_NAME = "family_sector_hypothesis_v1.json"
SUPPORT_BASIS_NAME = "golden_post_hawking_union_v1"
ALLOWED_SECTORS = ("HYPERBOLIC", "ELLIPTIC", "EUCLIDEAN", "UNKNOWN")
RULE_STATUSES = ("RULE_APPLIED", "NO_RULE", "CONFLICT", "NOT_SUPPORTED")
CRITERION_VERSION = "v1"
NO_RULE_CRITERION = "no_matching_family_theory_rule_v1"
CONFLICT_CRITERION = "conflicting_family_theory_rules_v1"


def _repo_root() -> Path:
    return _here.parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create an auditable family/theory -> sector hypothesis map")
    ap.add_argument("--run-id", required=True, help="Aggregate run id containing phase1 + phase2a artifacts")
    ap.add_argument("--input-name", default=DEFAULT_INPUT_NAME)
    ap.add_argument("--family-map-name", default=DEFAULT_FAMILY_MAP_NAME)
    ap.add_argument("--rules-path", default=DEFAULT_RULES_PATH)
    ap.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    return ap.parse_args(argv)


def _resolve_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    repo_candidate = (_repo_root() / p).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (Path.cwd() / p).resolve()


def _display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(_repo_root()))
    except ValueError:
        return str(path)


def _coerce_text(value: Any) -> str:
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value is None:
        return ""
    return str(value)


def _read_supported_geometry_ids(source_h5: Path, ctx: Any) -> list[str]:
    assert h5py is not None
    try:
        h5 = h5py.File(source_h5, "r")
    except Exception as exc:
        abort(ctx, f"Unable to open H5 input {source_h5}: {exc}")
        raise AssertionError("unreachable")

    with h5:
        if "atlas" not in h5 or "membership" not in h5:
            abort(ctx, f"Missing required atlas/membership groups in {source_h5}")
            raise AssertionError("unreachable")
        atlas_group = h5["atlas"]
        membership_group = h5["membership"]
        if "geometry_id" not in atlas_group:
            abort(ctx, f"Missing required dataset atlas/geometry_id in {source_h5}")
            raise AssertionError("unreachable")
        if "golden_post_hawking" not in membership_group:
            abort(ctx, f"Missing required dataset membership/golden_post_hawking in {source_h5}")
            raise AssertionError("unreachable")

        geometry_ids = [_coerce_text(value).strip() for value in atlas_group["geometry_id"][...]]
        matrix = membership_group["golden_post_hawking"][...]
        if matrix.shape[-1] != len(geometry_ids):
            abort(
                ctx,
                f"Membership matrix width mismatch in {source_h5}: "
                f"golden_post_hawking.shape={matrix.shape} geometry_ids={len(geometry_ids)}",
            )
            raise AssertionError("unreachable")
        supported = [
            geometry_ids[index]
            for index in range(len(geometry_ids))
            if bool(matrix[:, index].any())
        ]
        return supported


def _load_family_map(path: Path, ctx: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        abort(ctx, f"Corrupt family map JSON at {path}: {exc}")
        raise AssertionError("unreachable")

    rows = payload.get("rows")
    if not isinstance(rows, list):
        abort(ctx, f"Family map at {path} lacks rows list")
        raise AssertionError("unreachable")

    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            abort(ctx, f"Family map at {path} contains non-object row")
            raise AssertionError("unreachable")
        raw_geometry_id = _coerce_text(row.get("raw_geometry_id")).strip()
        if not raw_geometry_id:
            abort(ctx, f"Family map row missing raw_geometry_id at {path}")
            raise AssertionError("unreachable")
        if raw_geometry_id in indexed:
            duplicates.append(raw_geometry_id)
        indexed[raw_geometry_id] = row
    if duplicates:
        abort(ctx, f"Duplicate raw_geometry_id values in family map {path}: {sorted(set(duplicates))[:10]}")
        raise AssertionError("unreachable")
    return indexed, payload


def _load_rules(path: Path, ctx: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        abort(ctx, f"Corrupt rules JSON at {path}: {exc}")
        raise AssertionError("unreachable")

    if not isinstance(payload, dict) or payload.get("schema_version") != RULES_SCHEMA_VERSION:
        abort(ctx, f"Rules file at {path} must have schema_version={RULES_SCHEMA_VERSION}")
        raise AssertionError("unreachable")
    rules = payload.get("rules")
    if not isinstance(rules, list):
        abort(ctx, f"Rules file at {path} lacks rules list")
        raise AssertionError("unreachable")

    seen_rule_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            abort(ctx, f"Rules file at {path} contains non-object rule")
            raise AssertionError("unreachable")
        rule_id = _coerce_text(rule.get("rule_id")).strip()
        proposed_sector = _coerce_text(rule.get("proposed_sector")).strip()
        rule_source = _coerce_text(rule.get("rule_source")).strip()
        criterion = _coerce_text(rule.get("criterion")).strip()
        criterion_version = _coerce_text(rule.get("criterion_version")).strip()
        confidence_class = _coerce_text(rule.get("confidence_class")).strip()
        notes = _coerce_text(rule.get("notes")).strip()
        match = rule.get("match")
        if not rule_id or rule_id in seen_rule_ids:
            abort(ctx, f"Rules file at {path} has missing or duplicate rule_id={rule_id!r}")
            raise AssertionError("unreachable")
        if proposed_sector not in ALLOWED_SECTORS:
            abort(ctx, f"Rule {rule_id} in {path} has invalid proposed_sector={proposed_sector!r}")
            raise AssertionError("unreachable")
        if not rule_source or not criterion or criterion_version != CRITERION_VERSION:
            abort(ctx, f"Rule {rule_id} in {path} has incomplete rule_source/criterion metadata")
            raise AssertionError("unreachable")
        if not isinstance(match, dict):
            abort(ctx, f"Rule {rule_id} in {path} must define match object")
            raise AssertionError("unreachable")
        atlas_family = _coerce_text(match.get("atlas_family")).strip()
        atlas_theory = _coerce_text(match.get("atlas_theory")).strip()
        if not atlas_family or not atlas_theory:
            abort(ctx, f"Rule {rule_id} in {path} must match both atlas_family and atlas_theory")
            raise AssertionError("unreachable")
        seen_rule_ids.add(rule_id)
        validated.append(
            {
                "rule_id": rule_id,
                "match": {
                    "atlas_family": atlas_family,
                    "atlas_theory": atlas_theory,
                },
                "proposed_sector": proposed_sector,
                "criterion": criterion,
                "criterion_version": criterion_version,
                "rule_source": rule_source,
                "confidence_class": confidence_class,
                "notes": notes,
            }
        )
    return validated, payload


def _pair_key(family: str, theory: str) -> tuple[str, str]:
    return family, theory


def _evaluate_pair_rules(
    family: str,
    theory: str,
    *,
    rules_by_pair: dict[tuple[str, str], list[dict[str, Any]]],
    rules_display_path: str,
) -> dict[str, Any]:
    matching_rules = sorted(
        rules_by_pair.get(_pair_key(family, theory), []),
        key=lambda rule: rule["rule_id"],
    )
    if not matching_rules:
        return {
            "proposed_sector": "UNKNOWN",
            "rule_status": "NO_RULE",
            "rule_source": None,
            "rule_id": None,
            "criterion": NO_RULE_CRITERION,
            "criterion_version": CRITERION_VERSION,
            "evidence_fields_used": [
                "family_map.atlas_family",
                "family_map.atlas_theory",
                "rules.schema_version",
            ],
            "evidence": {
                "family_map.atlas_family": family,
                "family_map.atlas_theory": theory,
                "rules.schema_version": RULES_SCHEMA_VERSION,
            },
            "pair_has_rule": False,
            "pair_has_conflict": False,
        }

    proposed_sectors = sorted({rule["proposed_sector"] for rule in matching_rules})
    if len(proposed_sectors) > 1:
        return {
            "proposed_sector": "UNKNOWN",
            "rule_status": "CONFLICT",
            "rule_source": "MULTIPLE_RULES",
            "rule_id": "CONFLICT",
            "criterion": CONFLICT_CRITERION,
            "criterion_version": CRITERION_VERSION,
            "evidence_fields_used": [
                "family_map.atlas_family",
                "family_map.atlas_theory",
                "matching_rule_ids",
                "matching_proposed_sectors",
            ],
            "evidence": {
                "family_map.atlas_family": family,
                "family_map.atlas_theory": theory,
                "matching_rule_ids": [rule["rule_id"] for rule in matching_rules],
                "matching_proposed_sectors": proposed_sectors,
            },
            "pair_has_rule": True,
            "pair_has_conflict": True,
        }

    selected_rule = matching_rules[0]
    proposed_sector = selected_rule["proposed_sector"]
    rule_status = "NOT_SUPPORTED" if proposed_sector == "UNKNOWN" else "RULE_APPLIED"
    evidence_fields_used = [
        "family_map.atlas_family",
        "family_map.atlas_theory",
        "rules.match.atlas_family",
        "rules.match.atlas_theory",
        "rules.proposed_sector",
        "rules.rule_id",
        "rules.rule_source",
    ]
    evidence = {
        "family_map.atlas_family": family,
        "family_map.atlas_theory": theory,
        "rules.match.atlas_family": selected_rule["match"]["atlas_family"],
        "rules.match.atlas_theory": selected_rule["match"]["atlas_theory"],
        "rules.proposed_sector": proposed_sector,
        "rules.rule_id": selected_rule["rule_id"],
        "rules.rule_source": selected_rule["rule_source"],
    }
    if len(matching_rules) > 1:
        evidence_fields_used.append("matching_rule_ids")
        evidence["matching_rule_ids"] = [rule["rule_id"] for rule in matching_rules]

    return {
        "proposed_sector": proposed_sector,
        "rule_status": rule_status,
        "rule_source": selected_rule["rule_source"],
        "rule_id": selected_rule["rule_id"],
        "criterion": selected_rule["criterion"],
        "criterion_version": selected_rule["criterion_version"],
        "evidence_fields_used": evidence_fields_used,
        "evidence": evidence,
        "pair_has_rule": True,
        "pair_has_conflict": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx = init_stage(
        args.run_id,
        STAGE,
        params={
            "input_name": args.input_name,
            "family_map_name": args.family_map_name,
            "rules_path": args.rules_path,
            "output_name": args.output_name,
            "support_basis_name": SUPPORT_BASIS_NAME,
        },
    )

    if h5py is None:
        abort(ctx, "h5py is required to read the phase-1 geometry H5")

    source_h5_rel = Path("experiment") / "phase1_geometry_h5" / "outputs" / args.input_name
    source_h5_path = ctx.run_dir / source_h5_rel
    if not source_h5_path.exists():
        abort(
            ctx,
            f"Missing required input: expected_path={source_h5_path}; "
            f"regen_cmd='python mvp/experiment_phase1_geometry_h5.py --run-id {args.run_id}'",
        )

    source_family_map_rel = Path("experiment") / "phase2a_atlas_family_map" / "outputs" / args.family_map_name
    source_family_map_path = ctx.run_dir / source_family_map_rel
    if not source_family_map_path.exists():
        abort(
            ctx,
            f"Missing required input: expected_path={source_family_map_path}; "
            f"regen_cmd='python mvp/experiment_phase2a_atlas_family_map.py --run-id {args.run_id}'",
        )

    rules_path = _resolve_path(args.rules_path)
    if not rules_path.exists():
        abort(ctx, f"Missing rules input: expected_path={rules_path}")

    rules_display_path = _display_path(rules_path)
    check_inputs(
        ctx,
        {
            "phase1_geometry_h5": source_h5_path,
            "family_map_v1": source_family_map_path,
            "family_to_sector_rules": rules_path,
        },
    )

    supported_geometry_ids = _read_supported_geometry_ids(source_h5_path, ctx)
    family_map_by_gid, family_map_payload = _load_family_map(source_family_map_path, ctx)
    rules, rules_payload = _load_rules(rules_path, ctx)
    rules_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        rules_by_pair[_pair_key(rule["match"]["atlas_family"], rule["match"]["atlas_theory"])].append(rule)

    rows: list[dict[str, Any]] = []
    rule_status_counts = Counter()
    proposed_sector_counts = Counter()
    family_counts = Counter()
    theory_counts = Counter()
    supported_pairs: set[tuple[str, str]] = set()
    pair_evaluations: dict[tuple[str, str], dict[str, Any]] = {}

    for raw_geometry_id in supported_geometry_ids:
        family_row = family_map_by_gid.get(raw_geometry_id)
        if family_row is None:
            abort(ctx, f"Supported geometry_id missing from family_map_v1: {raw_geometry_id}")
        atlas_family = _coerce_text(family_row.get("atlas_family")).strip()
        atlas_theory = _coerce_text(family_row.get("atlas_theory")).strip()
        if not atlas_family or not atlas_theory:
            abort(ctx, f"Family map row for {raw_geometry_id} lacks atlas_family/atlas_theory")
        pair = _pair_key(atlas_family, atlas_theory)
        supported_pairs.add(pair)
        if pair not in pair_evaluations:
            pair_evaluations[pair] = _evaluate_pair_rules(
                atlas_family,
                atlas_theory,
                rules_by_pair=rules_by_pair,
                rules_display_path=rules_display_path,
            )

        evaluation = pair_evaluations[pair]
        row = {
            "raw_geometry_id": raw_geometry_id,
            "normalized_geometry_id": family_row.get("normalized_geometry_id"),
            "atlas_family": atlas_family,
            "atlas_theory": atlas_theory,
            "proposed_sector": evaluation["proposed_sector"],
            "rule_status": evaluation["rule_status"],
            "rule_source": evaluation["rule_source"],
            "rule_id": evaluation["rule_id"],
            "criterion": evaluation["criterion"],
            "criterion_version": evaluation["criterion_version"],
            "evidence_fields_used": evaluation["evidence_fields_used"],
            "evidence": evaluation["evidence"],
        }
        rows.append(row)
        rule_status_counts.update([row["rule_status"]])
        proposed_sector_counts.update([row["proposed_sector"]])
        family_counts.update([atlas_family])
        theory_counts.update([atlas_theory])

    supported_pairs_with_rule = sum(1 for evaluation in pair_evaluations.values() if evaluation["pair_has_rule"])
    conflict_count = sum(1 for evaluation in pair_evaluations.values() if evaluation["pair_has_conflict"])
    supported_pair_total = len(supported_pairs)
    supported_family_coverage = (
        float(supported_pairs_with_rule) / float(supported_pair_total)
        if supported_pair_total
        else 0.0
    )
    downstream_ready = (
        supported_pair_total > 0
        and supported_family_coverage == 1.0
        and conflict_count == 0
        and rule_status_counts.get("NO_RULE", 0) == 0
        and rule_status_counts.get("CONFLICT", 0) == 0
        and rule_status_counts.get("NOT_SUPPORTED", 0) == 0
        and proposed_sector_counts.get("UNKNOWN", 0) == 0
    )
    downstream_blockers: list[str] = []
    if supported_family_coverage != 1.0:
        downstream_blockers.append(f"supported_family_coverage={supported_family_coverage:.6f}")
    if conflict_count > 0:
        downstream_blockers.append(f"conflict_count={conflict_count}")
    for status in ("NO_RULE", "CONFLICT", "NOT_SUPPORTED"):
        if rule_status_counts.get(status, 0) > 0:
            downstream_blockers.append(f"rule_status.{status}={rule_status_counts[status]}")
    if proposed_sector_counts.get("UNKNOWN", 0) > 0:
        downstream_blockers.append(f"proposed_sector.UNKNOWN={proposed_sector_counts['UNKNOWN']}")

    family_counts_payload = {family: int(family_counts[family]) for family in sorted(family_counts)}
    theory_counts_payload = {theory: int(theory_counts[theory]) for theory in sorted(theory_counts)}
    rule_status_counts_payload = {status: int(rule_status_counts[status]) for status in RULE_STATUSES if rule_status_counts.get(status, 0) > 0}
    proposed_sector_counts_payload = {
        sector: int(proposed_sector_counts[sector])
        for sector in ALLOWED_SECTORS
        if proposed_sector_counts.get(sector, 0) > 0
    }

    output_payload = {
        "schema_version": SCHEMA_VERSION,
        "rules_schema_version": RULES_SCHEMA_VERSION,
        "support_basis_name": SUPPORT_BASIS_NAME,
        "source_h5": str(source_h5_rel),
        "source_family_map": str(source_family_map_rel),
        "source_rules": rules_display_path,
        "n_rows": len(rows),
        "family_counts": family_counts_payload,
        "theory_counts": theory_counts_payload,
        "rule_status_counts": rule_status_counts_payload,
        "proposed_sector_counts": proposed_sector_counts_payload,
        "supported_pair_total": supported_pair_total,
        "supported_pairs_with_rule": supported_pairs_with_rule,
        "supported_family_coverage": supported_family_coverage,
        "conflict_count": conflict_count,
        "downstream_ready": downstream_ready,
        "downstream_blockers": downstream_blockers,
        "rows": rows,
    }
    output_path = ctx.outputs_dir / args.output_name
    write_json_atomic(output_path, output_payload)

    finalize(
        ctx,
        artifacts={"family_sector_hypothesis_v1": output_path},
        results={
            "n_rows": len(rows),
            "supported_pair_total": supported_pair_total,
            "supported_pairs_with_rule": supported_pairs_with_rule,
            "supported_family_coverage": supported_family_coverage,
            "conflict_count": conflict_count,
            "downstream_ready": downstream_ready,
        },
        extra_summary={
            "schema_version": SCHEMA_VERSION,
            "rules_schema_version": RULES_SCHEMA_VERSION,
            "support_basis_name": SUPPORT_BASIS_NAME,
            "source_h5": str(source_h5_rel),
            "source_family_map": str(source_family_map_rel),
            "source_rules": rules_display_path,
            "n_rows": len(rows),
            "family_counts": family_counts_payload,
            "theory_counts": theory_counts_payload,
            "rule_status_counts": rule_status_counts_payload,
            "proposed_sector_counts": proposed_sector_counts_payload,
            "supported_pair_total": supported_pair_total,
            "supported_pairs_with_rule": supported_pairs_with_rule,
            "supported_family_coverage": supported_family_coverage,
            "conflict_count": conflict_count,
            "downstream_ready": downstream_ready,
            "downstream_blockers": downstream_blockers,
        },
    )
    log_stage_paths(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
