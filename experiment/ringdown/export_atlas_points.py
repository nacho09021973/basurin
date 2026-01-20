#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True, help="runs/<id>/dictionary/atlas.json")
    ap.add_argument("--out", default=None, help="output atlas_points.json (default: alongside atlas)")
    ap.add_argument("--feature-key", default="ratios", choices=["ratios","delta","M2_0"], help="which numeric field to export as features")
    args = ap.parse_args()

    atlas_path = Path(args.atlas)
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    theories = atlas.get("theories", [])
    if not theories:
        raise SystemExit("atlas has no theories")

    points = []
    for th in theories:
        if args.feature_key not in th:
            continue
        feat = th[args.feature_key]
        # normalize scalars to list
        if isinstance(feat, (int, float)):
            feat = [float(feat)]
        points.append({
            "id": th.get("id"),
            "regime": th.get("regime"),
            "delta": th.get("delta"),
            "M2_0": th.get("M2_0"),
            "features": [float(x) for x in feat],
        })

    out_path = Path(args.out) if args.out else atlas_path.with_name("atlas_points.json")
    out_path.write_text(json.dumps({
        "source_atlas": str(atlas_path),
        "feature_key": args.feature_key,
        "n_points": len(points),
        "points": points,
    }, indent=2, sort_keys=True), encoding="utf-8")

    print(out_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
