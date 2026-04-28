#!/usr/bin/env python
"""Dock a ligand into a protein pocket using a trained FlowFrag model.

Thin CLI wrapper around :mod:`src.inference.preprocess` and
:mod:`src.inference.sampler`. Mirrors the training data pipeline:
  1. Parse full protein → all atoms/residues (no cutoff)
  2. Crop protein around a pocket center (user-provided or derived from ligand)
  3. Build unified graph (same as training)
  4. Run ODE integration from prior to final pose

Usage:
    # Re-docking (ligand SDF/MOL2 gives pocket center from crystal)
    python scripts/dock.py \
        --protein pocket.pdb --ligand ligand.sdf \
        --checkpoint latest.pt --config configs/train_v3_b200.yaml

    # Blind docking (SMILES requires explicit pocket center)
    python scripts/dock.py \
        --protein protein.pdb --ligand "CCO" --pocket_center 12.3,-4.5,8.1 \
        --checkpoint latest.pt --config configs/train_v3_b200.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from rdkit.Chem import rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.io import (
    write_multi_sdf, write_sdf, write_traj_pdb, write_traj_sdf,
)
from src.inference.preprocess import load_ligand, preprocess_complex
from src.inference.sampler import sample_unified, sample_unified_multi_sigma, parse_sigma_list


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_center(s: str | None) -> torch.Tensor | None:
    if s is None:
        return None
    parts = s.replace(",", " ").split()
    assert len(parts) == 3, f"--pocket_center expects 3 floats, got {s}"
    return torch.tensor([float(p) for p in parts], dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dock a ligand into a protein pocket using FlowFrag")
    parser.add_argument("--protein", type=str, required=True)
    parser.add_argument("--ligand", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pocket_center", type=str, default=None,
                        help="Binding site x,y,z (required for SMILES input)")
    parser.add_argument("--pocket_cutoff", type=float, default=8.0)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--time_schedule", type=str, default="late",
                        choices=("uniform", "late", "early"))
    parser.add_argument("--schedule_power", type=float, default=3.0)
    parser.add_argument("--sigma", type=float, default=None,
                        help="Single prior σ. Ignored when --sigma_list is set.")
    parser.add_argument("--sigma_list", type=str, default=None,
                        help='Multi-σ inference. "2,3,4,5" splits --num_samples '
                             'across 4 σ values; "2:10,3:10,4:20" explicit '
                             'per-σ counts. Requires v4 σ-conditional model.')
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed torch RNG before sampling (fixes the SE(3) prior).")
    parser.add_argument("--save_traj", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/docked")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--phys_guidance", action="store_true",
                        help="Enable Vina-gradient guidance during ODE sampling.")
    parser.add_argument("--phys_lambda_max", type=float, default=0.3)
    parser.add_argument("--phys_power", type=float, default=2.0)
    parser.add_argument("--phys_start_t", type=float, default=0.3)
    parser.add_argument("--phys_max_force", type=float, default=10.0)
    parser.add_argument("--phys_weight_preset", type=str, default="vina",
                        choices=("vina", "vinardo"))
    parser.add_argument("--confidence_ckpt", type=str,
                        default="weights/confidence_v1.pt",
                        help="Path to a trained confidence head. When the file "
                             "exists and num_samples > 1, poses are reordered "
                             "by predicted pose RMSD (best first). Pass an "
                             "empty string to disable.")
    args = parser.parse_args()

    protein_pdb = Path(args.protein)
    assert protein_pdb.exists(), f"Protein PDB not found: {protein_pdb}"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Model
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "model_type"}
    from src.models.unified import UnifiedFlowFrag
    model = UnifiedFlowFrag(**model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)
    print(f"Model loaded: {args.checkpoint} (step {ckpt.get('step', '?')})")

    # Ligand
    print(f"Loading ligand: {args.ligand}")
    mol, has_pose = load_ligand(args.ligand)
    print(f"  Atoms: {mol.GetNumAtoms()}, "
          f"Formula: {rdMolDescriptors.CalcMolFormula(mol)}, has_pose={has_pose}")

    pocket_center = _parse_center(args.pocket_center)
    if pocket_center is None and not has_pose:
        raise ValueError(
            "SMILES input requires explicit --pocket_center x,y,z "
            "(cannot derive from ligand without a pose)."
        )

    print("Preprocessing...")
    graph, lig_data, meta = preprocess_complex(
        protein_pdb, mol,
        pocket_center=pocket_center,
        pocket_cutoff=args.pocket_cutoff,
    )
    print(f"  Pocket center: {meta['pocket_center'].tolist()}")
    print(f"  Ligand: {meta['num_atom']} atoms, {meta['num_frag']} fragments")
    print(f"  Graph: {graph['num_nodes'].item()} nodes, "
          f"{graph['num_prot_atom'].item()} prot atoms, "
          f"{graph['num_prot_res'].item()} residues, "
          f"{graph['edge_index'].shape[1]} edges")

    sigma = args.sigma if args.sigma is not None else cfg["data"].get("prior_sigma", 5.0)

    phys = None
    if args.phys_guidance:
        from src.scoring.physics_guidance import PhysicsGuidance
        phys = PhysicsGuidance(
            mol=mol,
            pocket_pdb=str(protein_pdb),
            pocket_center=meta["pocket_center"],
            device=device,
            weight_preset=args.phys_weight_preset,
            max_force_per_atom=args.phys_max_force,
        )
        print(f"Physics guidance ON: lambda_max={args.phys_lambda_max}, "
              f"power={args.phys_power}, start_t={args.phys_start_t}, "
              f"preset={args.phys_weight_preset}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    sigma_list, sigma_counts = parse_sigma_list(args.sigma_list, args.num_samples)
    if sigma_list:
        print(f"\nGenerating {args.num_samples} pose(s) (multi-σ), "
              f"{args.num_steps} ODE steps, "
              f"σ schedule = {list(zip(sigma_list, sigma_counts))} ...")
        results = sample_unified_multi_sigma(
            model, graph, lig_data, meta,
            sigma_list=sigma_list,
            samples_per_sigma=sigma_counts,
            num_steps=args.num_steps,
            time_schedule=args.time_schedule,
            schedule_power=args.schedule_power,
            device=device,
            save_traj=args.save_traj,
            phys_guidance=phys,
            phys_lambda_max=args.phys_lambda_max,
            phys_power=args.phys_power,
            phys_start_t=args.phys_start_t,
        )
    else:
        print(f"\nGenerating {args.num_samples} pose(s), {args.num_steps} ODE steps, sigma={sigma}...")
        results = sample_unified(
            model, graph, lig_data, meta,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            translation_sigma=sigma,
            time_schedule=args.time_schedule,
            schedule_power=args.schedule_power,
            device=device,
            save_traj=args.save_traj,
            phys_guidance=phys,
            phys_lambda_max=args.phys_lambda_max,
            phys_power=args.phys_power,
            phys_start_t=args.phys_start_t,
        )
    all_poses = [r["atom_pos_pred"] for r in results]
    all_trajs = results if args.save_traj else []
    pc = meta["pocket_center"]

    # Confidence-based reranking ------------------------------------------
    confidence_pred = None
    use_confidence = (
        args.confidence_ckpt
        and args.num_samples > 1
        and Path(args.confidence_ckpt).exists()
    )
    if use_confidence:
        from src.inference.confidence_features import score_poses_with_confidence
        print(f"Scoring {args.num_samples} poses with confidence head: "
              f"{args.confidence_ckpt}")
        confidence_pred, sorted_idx = score_poses_with_confidence(
            model, args.confidence_ckpt, graph, lig_data, meta,
            all_poses, device,
        )
        all_poses = [all_poses[i] for i in sorted_idx]
        all_trajs = [all_trajs[i] for i in sorted_idx] if all_trajs else all_trajs
        confidence_pred = [confidence_pred[i] for i in sorted_idx]
        print(f"  Top-3 predicted RMSD: "
              f"{confidence_pred[0]:.2f}, {confidence_pred[1]:.2f}, {confidence_pred[2]:.2f}")

    if args.num_samples == 1:
        out_path = out_dir / "docked.sdf"
        write_sdf(mol, all_poses[0], pc, out_path)
        print(f"\nDocked pose saved to {out_path}")
    else:
        out_path = out_dir / "docked_poses.sdf"
        write_multi_sdf(mol, all_poses, pc, out_path)
        print(f"\n{args.num_samples} poses saved to {out_path}")

    if args.save_traj:
        for i, res in enumerate(all_trajs):
            suffix = f"_{i}" if args.num_samples > 1 else ""
            write_traj_sdf(mol, res["traj"], res["traj_times"], pc,
                           out_dir / f"traj{suffix}.sdf")
            write_traj_pdb(mol, res["traj"], pc, out_dir / f"traj{suffix}.pdb")
            print(f"  Trajectory{suffix}: {len(res['traj'])} frames")

    torch.save({
        "pocket_center": pc,
        "frag_centers": lig_data["frag_centers"],
        "frag_sizes": lig_data["frag_sizes"],
        "poses": [{"atom_pos_pred": p} for p in all_poses],
        "trajectories": [
            {"traj": r["traj"], "traj_times": r["traj_times"]} for r in all_trajs
        ] if args.save_traj else None,
        "confidence_pred_rmsd": confidence_pred,  # None if --confidence_ckpt not set
    }, out_dir / "results.pt")
    print(f"Raw tensors saved to {out_dir / 'results.pt'}")


if __name__ == "__main__":
    main()
