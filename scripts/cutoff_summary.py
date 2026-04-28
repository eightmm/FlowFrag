#!/usr/bin/env python
"""Aggregate outputs/cutoff_sweep/* results into a single table.

Reads outputs/cutoff_sweep/{astex,pb}_c{N}/results.json which is the
eval_benchmark.py summary, and emits a markdown table across cutoffs and
selectors (oracle / vina / confidence).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("outputs/cutoff_sweep")
DATASETS = [
    ("astex", [5, 6, 7, 8, 10]),
    ("pb", [5, 8, 10]),
]
SELECTORS = ["oracle", "vina", "confidence"]
REFINE = "none"


def load(ds: str, c: int) -> dict | None:
    f = ROOT / f"{ds}_c{c}" / "results.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def fmt_cell(stat: dict | None) -> str:
    if stat is None:
        return "    -    "
    pct = stat.get("pct_lt_2A")
    mean = stat.get("mean_rmsd")
    if pct is None or mean is None:
        return "    -    "
    return f"{pct:5.1f}% ({mean:.2f})"


def main() -> None:
    for ds, cuts in DATASETS:
        print(f"\n## {ds.upper()} (refine={REFINE})  —  <2A SR (mean RMSD)\n")
        header = "| Selector " + "".join(f"| c={c} " for c in cuts) + "|"
        sep = "|---" * (len(cuts) + 1) + "|"
        print(header)
        print(sep)
        for sel in SELECTORS:
            row = f"| {sel:10s} "
            for c in cuts:
                rj = load(ds, c)
                stat = None
                if rj is not None:
                    stat = rj.get("stats", {}).get(f"{REFINE}+{sel}")
                row += f"| {fmt_cell(stat)} "
            row += "|"
            print(row)


if __name__ == "__main__":
    main()
