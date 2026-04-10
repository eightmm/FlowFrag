#!/usr/bin/env python
"""Re-score saved poses with different selection strategies.

Loads raw poses from a previous eval_benchmark run and applies new
selection/refinement without re-sampling.

Usage:
    # Vina ranking on PoseBusters
    python scripts/rescore_poses.py \
        --poses_dir outputs/eval_posebusters_v2_10s/poses \
        --data_dir /mnt/data/PLI/PoseBusters/posebusters_benchmark_set

    # Vina ranking on Astex
    python scripts/rescore_poses.py \
        --poses_dir outputs/eval_astex_10s/poses \
        --data_dir /mnt/data/PLI/Astex-diverse-set
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_benchmark import (
    detect_complex_files,
    load_ligand,
    compute_rmsd,
    compute_stats,
    mmff_refine,
)
from src.scoring.pose_ranking import rank_poses


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score saved poses with Vina ranking")
    parser.add_argument("--poses_dir", type=str, required=True,
                        help="Directory with saved {pdb_id}.pt files")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Original dataset directory (for protein PDB + ligand)")
    parser.add_argument("--mmff", action="store_true",
                        help="Apply MMFF before ranking")
    args = parser.parse_args()

    poses_dir = Path(args.poses_dir)
    data_dir = Path(args.data_dir)

    pt_files = sorted(poses_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} saved pose files")
    print(f"Data dir: {data_dir}")
    print(f"MMFF: {args.mmff}\n")

    results_rank = []
    results_oracle = []
    failures = []
    t_start = time.time()

    for idx, pt_file in enumerate(pt_files):
        pdb_id = pt_file.stem
        saved = torch.load(pt_file, map_location="cpu", weights_only=False)

        raw_poses = saved["raw_poses"]
        ref_pos = saved["ref_pos"]
        pocket_center = saved["pocket_center"]

        complex_dir = data_dir / pdb_id
        detected = detect_complex_files(complex_dir, pdb_id)
        if detected is None:
            print(f"[{idx+1:3d}/{len(pt_files)}] {pdb_id}: SKIP (missing data files)")
            failures.append({"pdb_id": pdb_id, "error": "missing data files"})
            continue

        pocket_pdb, ligand_file, fmt = detected

        try:
            mol = load_ligand(ligand_file, fmt)

            # Optionally apply MMFF
            if args.mmff:
                poses = [mmff_refine(mol, p, pocket_center) for p in raw_poses]
            else:
                poses = raw_poses

            rmsds = [compute_rmsd(p, ref_pos) for p in poses]

            # Vina ranking — pocket_cutoff pre-filters protein atoms for speed
            ranked = rank_poses(
                mol, poses, pocket_pdb, pocket_center,
                device=torch.device("cpu"),
                pocket_cutoff=10.0,
            )
            rank_idx = ranked[0]["idx"]
            rank_rmsd = rmsds[rank_idx]
            vina_score = ranked[0]["vina_score"]
            oracle_rmsd = min(rmsds)

            results_rank.append({"pdb_id": pdb_id, "rmsd": rank_rmsd, "vina_score": vina_score})
            results_oracle.append({"pdb_id": pdb_id, "rmsd": oracle_rmsd})

            ok = "OK" if rank_rmsd < 2.0 else "  "
            print(f"[{idx+1:3d}/{len(pt_files)}] {pdb_id}: "
                  f"rank={rank_rmsd:5.2f}  oracle={oracle_rmsd:5.2f}  "
                  f"vina={vina_score:6.2f}  {ok}")

        except Exception as e:
            print(f"[{idx+1:3d}/{len(pt_files)}] {pdb_id}: FAIL ({e})")
            failures.append({"pdb_id": pdb_id, "error": str(e)})

    elapsed = time.time() - t_start

    # Summary
    print(f"\n{'='*60}")
    print(f"Re-scoring: {len(results_rank)} complexes, {elapsed:.0f}s")
    print(f"{'='*60}")

    if results_rank:
        rank_rmsds = np.array([r["rmsd"] for r in results_rank])
        oracle_rmsds = np.array([r["rmsd"] for r in results_oracle])

        refine_label = "mmff" if args.mmff else "none"
        print(f"\n{'Method':<16} {'Mean':>6} {'Med':>6} {'<1A':>6} {'<2A':>6} {'<5A':>6}")
        print("-" * 52)

        s = compute_stats(oracle_rmsds)
        print(f"{refine_label}+oracle   {s['mean_rmsd']:6.2f} {s['median_rmsd']:6.2f} "
              f"{s['pct_lt_1A']:5.1f}% {s['pct_lt_2A']:5.1f}% {s['pct_lt_5A']:5.1f}%")

        s = compute_stats(rank_rmsds)
        print(f"{refine_label}+rank     {s['mean_rmsd']:6.2f} {s['median_rmsd']:6.2f} "
              f"{s['pct_lt_1A']:5.1f}% {s['pct_lt_2A']:5.1f}% {s['pct_lt_5A']:5.1f}%")

    # Save
    out_file = poses_dir.parent / f"rescore_{'mmff_' if args.mmff else ''}rank.json"
    summary = {
        "poses_dir": str(poses_dir),
        "data_dir": str(data_dir),
        "mmff": args.mmff,
        "elapsed_seconds": elapsed,
        "results_rank": results_rank,
        "failures": failures,
    }
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
