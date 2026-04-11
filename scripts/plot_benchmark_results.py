#!/usr/bin/env python
"""Aggregate Astex + PoseBusters eval results and render summary plots.

Reads per-complex RMSDs from:
  outputs/eval_{astex,posebusters_v2}_{10s,40s}/
    results.json                (oracle, cluster for none+mmff)
    rescore_rank.json           (none+rank)
    rescore_mmff_rank.json      (mmff+rank)

Produces:
  outputs/plots/
    sr_2A_bars.png              grouped bar chart of <2A success rate
    rmsd_cdf.png                cumulative RMSD distribution
    n_scaling.png               10→40 samples comparison
    oracle_vs_rank.png          per-complex scatter
    summary_table.md            all numbers in markdown
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "astex":         {"label": "Astex Diverse (84)",     "color_10": "#87CEEB", "color_40": "#1f77b4"},
    "posebusters":   {"label": "PoseBusters v2 (308)",   "color_10": "#FFB6B6", "color_40": "#d62728"},
}


def load_dataset(dataset: str, n: int) -> dict[str, dict]:
    """Load combined results for (dataset, N samples).

    Returns dict with keys "none+oracle", "none+cluster", "none+rank",
    "mmff+oracle", "mmff+cluster", "mmff+rank", each with a list of per-complex dicts.
    """
    if dataset == "astex":
        base = Path(f"outputs/eval_astex_{n}s")
        main_file = base / "results.json"
    elif dataset == "posebusters":
        base = Path(f"outputs/eval_posebusters_v2_{n}s")
        main_file = base / "results.json"
        if not main_file.exists():
            main_file = base / "posebusters_v2_results.json"
    else:
        raise ValueError(dataset)

    main = json.load(open(main_file))

    # results.json has per_complex[{refine}+{select}] = list of dicts with pdb_id + rmsd
    combos: dict[str, list[dict]] = {}
    for key, entries in main["per_complex"].items():
        combos[key] = entries

    # rescore files add the "rank" selection
    for refine, fname in [("none", "rescore_rank.json"), ("mmff", "rescore_mmff_rank.json")]:
        rescore_file = base / fname
        if rescore_file.exists():
            rescore = json.load(open(rescore_file))
            combos[f"{refine}+rank"] = rescore["results_rank"]

    return combos


def sr_at(rmsds: np.ndarray, thresh: float) -> float:
    return float((rmsds < thresh).mean() * 100)


COMBOS = [
    "none+oracle", "none+cluster", "none+rank",
    "mmff+oracle", "mmff+cluster", "mmff+rank",
]
COMBO_LABELS = [
    "oracle", "cluster", "rank",
    "oracle", "cluster", "rank",
]

# ---------------------------------------------------------------------------
# 1. Grouped bar chart: <2A success rate, all combos x datasets x N
# ---------------------------------------------------------------------------

def plot_sr_bars(data: dict[tuple[str, int], dict[str, list]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, dataset in zip(axes, ["astex", "posebusters"]):
        d10 = data[(dataset, 10)]
        d40 = data[(dataset, 40)]

        sr_10 = [sr_at(np.array([r["rmsd"] for r in d10[c]]), 2.0) for c in COMBOS]
        sr_40 = [sr_at(np.array([r["rmsd"] for r in d40[c]]), 2.0) for c in COMBOS]

        x = np.arange(len(COMBOS))
        width = 0.38

        info = DATASETS[dataset]
        ax.bar(x - width/2, sr_10, width, label="N=10", color=info["color_10"], edgecolor="black", linewidth=0.5)
        ax.bar(x + width/2, sr_40, width, label="N=40", color=info["color_40"], edgecolor="black", linewidth=0.5)

        # Value labels
        for i, (a, b) in enumerate(zip(sr_10, sr_40)):
            ax.text(i - width/2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
            ax.text(i + width/2, b + 1, f"{b:.0f}", ha="center", fontsize=8)

        # Group separators
        ax.axvline(2.5, color="gray", linestyle=":", alpha=0.5)
        ax.text(1, 105, "no refinement", ha="center", fontsize=9, style="italic", color="gray")
        ax.text(4, 105, "MMFF refined",  ha="center", fontsize=9, style="italic", color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(COMBO_LABELS)
        ax.set_ylim(0, 110)
        ax.set_title(info["label"])
        ax.set_ylabel("Success rate < 2Å (%)" if dataset == "astex" else "")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", framealpha=0.9)

    fig.suptitle("FlowFrag: Docking Success Rate (RMSD < 2Å)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sr_2A_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. RMSD CDF (cumulative distribution)
# ---------------------------------------------------------------------------

def plot_rmsd_cdf(data: dict[tuple[str, int], dict[str, list]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    plot_combos = [
        ("none+oracle",  "none, oracle",   "#87CEEB", "--"),
        ("mmff+oracle",  "MMFF, oracle",   "#1f77b4", "-"),
        ("mmff+cluster", "MMFF, cluster",  "#2ca02c", "-"),
        ("mmff+rank",    "MMFF, rank",     "#d62728", "-"),
    ]

    for ax, dataset in zip(axes, ["astex", "posebusters"]):
        d40 = data[(dataset, 40)]

        for combo, label, color, ls in plot_combos:
            rmsds = np.sort(np.array([r["rmsd"] for r in d40[combo]]))
            y = np.arange(1, len(rmsds) + 1) / len(rmsds) * 100
            ax.plot(rmsds, y, label=label, color=color, linestyle=ls, linewidth=2)

        ax.axvline(2.0, color="gray", linestyle=":", alpha=0.6)
        ax.axvline(1.0, color="gray", linestyle=":", alpha=0.3)
        ax.text(2.05, 5, "2Å", color="gray", fontsize=9)
        ax.text(1.05, 5, "1Å", color="gray", fontsize=9)

        ax.set_xlabel("Ligand RMSD (Å)")
        ax.set_ylabel("Cumulative fraction (%)" if dataset == "astex" else "")
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 100)
        ax.set_title(f"{DATASETS[dataset]['label']}, N=40")
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", framealpha=0.9)

    fig.suptitle("RMSD Cumulative Distribution (N=40)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rmsd_cdf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. N scaling (10 vs 40 for each method)
# ---------------------------------------------------------------------------

def plot_n_scaling(data: dict[tuple[str, int], dict[str, list]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    markers = {"oracle": "o", "cluster": "s", "rank": "^"}
    colors = {"none": "#888888", "mmff": "#1f77b4"}

    for ax, dataset in zip(axes, ["astex", "posebusters"]):
        for refine in ["none", "mmff"]:
            for select in ["oracle", "cluster", "rank"]:
                combo = f"{refine}+{select}"
                sr10 = sr_at(np.array([r["rmsd"] for r in data[(dataset, 10)][combo]]), 2.0)
                sr40 = sr_at(np.array([r["rmsd"] for r in data[(dataset, 40)][combo]]), 2.0)
                ls = "-" if refine == "mmff" else "--"
                ax.plot([10, 40], [sr10, sr40],
                        marker=markers[select],
                        color=colors[refine],
                        linestyle=ls,
                        markersize=8,
                        label=f"{refine}+{select}")

        ax.set_xticks([10, 40])
        ax.set_xlabel("Number of samples (N)")
        ax.set_ylabel("Success rate < 2Å (%)" if dataset == "astex" else "")
        ax.set_ylim(30, 100)
        ax.set_title(DATASETS[dataset]["label"])
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)

    fig.suptitle("Effect of sample count on success rate", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Per-complex scatter: oracle vs rank (selection gap)
# ---------------------------------------------------------------------------

def plot_oracle_vs_rank(data: dict[tuple[str, int], dict[str, list]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, dataset in zip(axes, ["astex", "posebusters"]):
        d40 = data[(dataset, 40)]

        # Build per-complex dict for mmff+oracle and mmff+rank
        oracle_by_pdb = {r["pdb_id"]: r["rmsd"] for r in d40["mmff+oracle"]}
        rank_by_pdb = {r["pdb_id"]: r["rmsd"] for r in d40["mmff+rank"]}

        common = sorted(set(oracle_by_pdb) & set(rank_by_pdb))
        x = np.array([oracle_by_pdb[p] for p in common])
        y = np.array([rank_by_pdb[p] for p in common])

        # Classify by whether rank is close to oracle (within 0.5A)
        gap = y - x
        good = gap < 0.5
        bad = ~good

        ax.scatter(x[good], y[good], s=20, c="#1f77b4", alpha=0.6, label=f"Δ<0.5Å  (n={good.sum()})")
        ax.scatter(x[bad], y[bad], s=20, c="#d62728", alpha=0.6, label=f"Δ≥0.5Å  (n={bad.sum()})")

        lim = 10
        ax.plot([0, lim], [0, lim], "k--", alpha=0.4, linewidth=1)
        ax.axhline(2.0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(2.0, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Oracle RMSD (Å)")
        ax.set_ylabel("Rank-selected RMSD (Å)")
        ax.set_title(f"{DATASETS[dataset]['label']}, MMFF, N=40")
        ax.set_aspect("equal")
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", framealpha=0.9)

    fig.suptitle("Selection Gap: Oracle vs Vina-Rank (per-complex)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "oracle_vs_rank.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Markdown summary table
# ---------------------------------------------------------------------------

def write_summary_table(data: dict[tuple[str, int], dict[str, list]]) -> None:
    lines = ["# FlowFrag Benchmark Summary", ""]
    lines.append("Checkpoint: `outputs/train_unified_ne_contact_adamw_1000/checkpoints/best.pt` (step 315000)")
    lines.append("")
    lines.append("Settings: 25 ODE steps, late schedule (power=3), prior σ=1.0, 8Å pocket cutoff around reference ligand")
    lines.append("")

    for dataset in ["astex", "posebusters"]:
        label = DATASETS[dataset]["label"]
        for n in [10, 40]:
            lines.append(f"## {label} — N={n}")
            lines.append("")
            lines.append("| Refine | Select | Mean | Median | <1Å | <2Å | <5Å |")
            lines.append("|--------|--------|------|--------|-----|-----|-----|")
            for combo in COMBOS:
                entries = data[(dataset, n)][combo]
                rmsds = np.array([r["rmsd"] for r in entries])
                refine, select = combo.split("+")
                lines.append(
                    f"| {refine} | {select} | {rmsds.mean():.2f} | {np.median(rmsds):.2f} | "
                    f"{sr_at(rmsds,1):.1f}% | {sr_at(rmsds,2):.1f}% | {sr_at(rmsds,5):.1f}% |"
                )
            lines.append("")

    (OUT_DIR / "summary_table.md").write_text("\n".join(lines))


def main() -> None:
    print("Loading results...")
    data: dict[tuple[str, int], dict[str, list]] = {}
    for dataset in ["astex", "posebusters"]:
        for n in [10, 40]:
            data[(dataset, n)] = load_dataset(dataset, n)
            combos_found = list(data[(dataset, n)].keys())
            print(f"  {dataset} N={n}: {len(combos_found)} combos")

    print("\nGenerating plots...")
    plot_sr_bars(data)
    print("  ✓ sr_2A_bars.png")
    plot_rmsd_cdf(data)
    print("  ✓ rmsd_cdf.png")
    plot_n_scaling(data)
    print("  ✓ n_scaling.png")
    plot_oracle_vs_rank(data)
    print("  ✓ oracle_vs_rank.png")
    write_summary_table(data)
    print("  ✓ summary_table.md")

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
