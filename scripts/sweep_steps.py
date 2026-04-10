#!/usr/bin/env python
"""Sweep ODE step count and measure SR + speed.

Usage:
    python scripts/sweep_steps.py \
        --checkpoint outputs/train_unified_ne_contact_adamw_1000/checkpoints/latest.pt \
        --config configs/train_unified_ne_contact_adamw_1000.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dock import preprocess_complex, sample_unified
from scripts.eval_benchmark import load_mol2_robust, mmff_refine, compute_rmsd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--astex_dir", type=str, default="/mnt/data/PLI/Astex-diverse-set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    astex_dir = Path(args.astex_dir)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model_cfg = dict(cfg["model"])
    model_cfg.pop("model_type", None)
    from src.models.unified import UnifiedFlowFrag
    model = UnifiedFlowFrag(**model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)

    sigma = cfg["data"].get("prior_sigma", 1.0)
    pdb_ids = sorted([d.name for d in astex_dir.iterdir() if d.is_dir()])

    step_counts = [5, 10, 15, 20, 25, 50]

    # Preprocess all complexes once
    print("Preprocessing complexes...")
    complexes = []
    for pdb_id in pdb_ids:
        complex_dir = astex_dir / pdb_id
        pocket_pdb = complex_dir / f"{pdb_id}_pocket.pdb"
        ligand_mol2 = complex_dir / f"{pdb_id}_ligand.mol2"
        if not pocket_pdb.exists() or not ligand_mol2.exists():
            continue
        try:
            mol = load_mol2_robust(ligand_mol2)
            graph, lig_data, meta = preprocess_complex(pocket_pdb, mol, ligand_has_pose=True)
            pocket_center = meta["pocket_center"]
            ref_pos = lig_data["atom_coords"] - pocket_center
            complexes.append({
                "pdb_id": pdb_id, "mol": mol, "graph": graph,
                "lig_data": lig_data, "meta": meta, "ref_pos": ref_pos,
                "pocket_center": pocket_center,
            })
        except Exception:
            pass
    print(f"  {len(complexes)} complexes ready\n")

    # Use fixed seeds for fair comparison
    base_seeds = list(range(args.num_samples))

    print(f"{'Steps':>5s}  {'Mean':>6s} {'Median':>6s}  {'<1Å':>8s}  {'<2Å':>8s}  {'<3Å':>8s}  {'<5Å':>8s}  {'Time/cmplx':>10s}")
    print("-" * 75)

    for num_steps in step_counts:
        all_rmsds = []
        t_start = time.time()

        for ci, c in enumerate(complexes):
            best_rmsd = float("inf")
            for seed_i in base_seeds:
                torch.manual_seed(42 + seed_i * 1000 + ci)
                result = sample_unified(
                    model, c["graph"], c["lig_data"], c["meta"],
                    num_steps=num_steps, translation_sigma=sigma,
                    time_schedule="late", schedule_power=3.0,
                    device=device,
                )
                pred_pos = result["atom_pos_pred"]
                refined = mmff_refine(c["mol"], pred_pos, c["pocket_center"])
                rmsd = compute_rmsd(refined, c["ref_pos"])
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
            all_rmsds.append(best_rmsd)

        elapsed = time.time() - t_start
        rmsds = np.array(all_rmsds)
        n = len(rmsds)
        per_complex = elapsed / n

        sr1 = (rmsds < 1.0).sum() / n * 100
        sr2 = (rmsds < 2.0).sum() / n * 100
        sr3 = (rmsds < 3.0).sum() / n * 100
        sr5 = (rmsds < 5.0).sum() / n * 100

        print(f"{num_steps:5d}  {rmsds.mean():6.2f} {np.median(rmsds):6.2f}  "
              f"{sr1:5.1f}% ({(rmsds<1.0).sum():2d})  "
              f"{sr2:5.1f}% ({(rmsds<2.0).sum():2d})  "
              f"{sr3:5.1f}% ({(rmsds<3.0).sum():2d})  "
              f"{sr5:5.1f}% ({(rmsds<5.0).sum():2d})  "
              f"{per_complex:8.1f}s")


if __name__ == "__main__":
    main()
