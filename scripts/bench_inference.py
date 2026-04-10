#!/usr/bin/env python
"""Benchmark inference timing: ODE, MMFF, Vina scoring breakdown.

Usage:
    python scripts/bench_inference.py \
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
from src.scoring.pose_ranking import rank_poses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--astex_dir", type=str, default="/mnt/data/PLI/Astex-diverse-set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_complexes", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=5)
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
    print(f"Model loaded, device={device}")

    sigma = cfg["data"].get("prior_sigma", 1.0)
    pdb_ids = sorted([d.name for d in astex_dir.iterdir() if d.is_dir()])[:args.num_complexes]

    # Timing accumulators
    t_preprocess = []
    t_ode = []
    t_mmff = []
    t_vina = []
    n_atoms_list = []

    for idx, pdb_id in enumerate(pdb_ids):
        complex_dir = astex_dir / pdb_id
        pocket_pdb = complex_dir / f"{pdb_id}_pocket.pdb"
        ligand_mol2 = complex_dir / f"{pdb_id}_ligand.mol2"
        if not pocket_pdb.exists() or not ligand_mol2.exists():
            continue

        try:
            # --- Preprocess ---
            t0 = time.perf_counter()
            mol = load_mol2_robust(ligand_mol2)
            graph, lig_data, meta = preprocess_complex(pocket_pdb, mol, ligand_has_pose=True)
            t_preprocess.append(time.perf_counter() - t0)

            pocket_center = meta["pocket_center"]
            ref_pos = lig_data["atom_coords"] - pocket_center
            n_atoms_list.append(meta["num_atom"])

            raw_poses = []
            mmff_poses = []

            # --- ODE sampling ---
            for s in range(args.num_samples):
                torch.cuda.synchronize() if device.type == "cuda" else None
                t0 = time.perf_counter()
                result = sample_unified(
                    model, graph, lig_data, meta,
                    num_steps=25, translation_sigma=sigma,
                    time_schedule="late", schedule_power=3.0,
                    device=device,
                )
                torch.cuda.synchronize() if device.type == "cuda" else None
                t_ode.append(time.perf_counter() - t0)
                raw_poses.append(result["atom_pos_pred"])

            # --- MMFF refinement ---
            for pose in raw_poses:
                t0 = time.perf_counter()
                refined = mmff_refine(mol, pose, pocket_center)
                t_mmff.append(time.perf_counter() - t0)
                mmff_poses.append(refined)

            # --- Vina + validity scoring ---
            t0 = time.perf_counter()
            ranked = rank_poses(
                mol, mmff_poses, pocket_pdb, pocket_center,
                device=torch.device("cpu"),
            )
            t_vina.append(time.perf_counter() - t0)

            print(f"[{idx+1}/{len(pdb_ids)}] {pdb_id}: "
                  f"atoms={meta['num_atom']:3d}, frags={meta['num_frag']:2d}, "
                  f"preproc={t_preprocess[-1]:.3f}s, "
                  f"ode={np.mean(t_ode[-args.num_samples:]):.3f}s, "
                  f"mmff={np.mean(t_mmff[-args.num_samples:]):.3f}s, "
                  f"vina={t_vina[-1]:.3f}s")

        except Exception as e:
            print(f"[{idx+1}/{len(pdb_ids)}] {pdb_id}: FAIL ({e})")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Timing Benchmark ({len(t_preprocess)} complexes, {args.num_samples} seeds)")
    print(f"{'='*60}")
    print(f"Mean atoms: {np.mean(n_atoms_list):.0f}")
    print(f"\nPer-operation (seconds):")
    print(f"  Preprocess:    {np.mean(t_preprocess):.3f} ± {np.std(t_preprocess):.3f}")
    print(f"  ODE (1 seed):  {np.mean(t_ode):.3f} ± {np.std(t_ode):.3f}")
    print(f"  MMFF (1 pose): {np.mean(t_mmff):.3f} ± {np.std(t_mmff):.3f}")
    print(f"  Vina+PB ({args.num_samples} poses): {np.mean(t_vina):.3f} ± {np.std(t_vina):.3f}")

    total_per_complex = (
        np.mean(t_preprocess)
        + args.num_samples * np.mean(t_ode)
        + args.num_samples * np.mean(t_mmff)
        + np.mean(t_vina)
    )
    print(f"\nTotal per complex ({args.num_samples} seeds): {total_per_complex:.1f}s")
    print(f"  ODE:  {args.num_samples * np.mean(t_ode) / total_per_complex * 100:.0f}%")
    print(f"  MMFF: {args.num_samples * np.mean(t_mmff) / total_per_complex * 100:.0f}%")
    print(f"  Vina: {np.mean(t_vina) / total_per_complex * 100:.0f}%")
    print(f"  Prep: {np.mean(t_preprocess) / total_per_complex * 100:.0f}%")


if __name__ == "__main__":
    main()
