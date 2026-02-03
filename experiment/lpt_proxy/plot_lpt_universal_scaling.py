#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    run_dir = Path("runs") / args.run
    in_path = run_dir / "experiment" / "lpt_multi_compare" / "outputs" / "lpt_multi_comparison.json"
    if not in_path.exists():
        raise SystemExit(f"ERROR: missing {in_path}")

    j = read_json(in_path)
    rows = j["rows"]

    # Data
    T_min = [r["period_min"] for r in rows]
    L_eff = [r["L_eff_s"] for r in rows]
    names = [r["name"] for r in rows]

    out_dir = run_dir / "experiment" / "lpt_multi_compare" / "outputs"
    out_png = out_dir / "lpt_universal_scaling.png"
    out_csv = out_dir / "lpt_universal_scaling.csv"

    # CSV for auditability
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "period_min", "L_eff_s"])
        for n, t, l in zip(names, T_min, L_eff):
            w.writerow([n, f"{t:.6f}", f"{l:.12f}"])

    # Plot (no fixed colors)
    plt.figure()
    plt.scatter(L_eff, T_min)
    for x, y, label in zip(L_eff, T_min, names):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))

    plt.xlabel("L_eff (s)")
    plt.ylabel("T_obs (min)")
    plt.title("LPT universal scaling (proxy-based)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print("PASS: wrote", out_png.as_posix())
    print("PASS: wrote", out_csv.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
