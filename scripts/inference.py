"""FlowFrag inference entrypoint.

Usage:
    python scripts/inference.py --config configs/train.yaml --checkpoint outputs/checkpoints/latest.pt
    python scripts/inference.py --config configs/train.yaml --checkpoint outputs/checkpoints/latest.pt --num_steps 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FlowFrag inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--num_steps", type=int, default=25, help="ODE integration steps")
    parser.add_argument(
        "--time_schedule",
        type=str,
        default="late",
        choices=("uniform", "late", "early"),
        help="Time grid schedule (default: late, denser near t=1 / crystal)",
    )
    parser.add_argument("--schedule_power", type=float, default=3.0, help="Power for non-uniform time schedules")
    parser.add_argument("--sigma", type=float, default=None, help="Prior translation sigma (default: from config)")
    parser.add_argument("--out_dir", type=str, default="outputs/predictions", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples (0=all)")
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate (train/val)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.models.flowfrag import FlowFrag

    model = FlowFrag(**cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)
    print(f"Loaded checkpoint: {args.checkpoint} (step {ckpt.get('step', '?')})")

    from src.data.dataset import FlowFragDataset

    dcfg = cfg["data"]
    ds_kwargs = dict(
        root=dcfg["data_dir"],
        translation_sigma=dcfg.get("prior_sigma", 10.0),
        max_atoms=dcfg.get("max_atoms", 80),
        max_frags=dcfg.get("max_frags", 20),
        min_atoms=dcfg.get("min_atoms", 5),
        rotation_augmentation=dcfg.get("rotation_augmentation", "none"),
        deterministic=dcfg.get("deterministic", False),
        deterministic_augmentation=dcfg.get("deterministic_augmentation"),
        deterministic_prior=dcfg.get("deterministic_prior"),
        deterministic_time=dcfg.get("deterministic_time"),
        prior_bank_size=dcfg.get("prior_bank_size", 1),
        time_bank_size=dcfg.get("time_bank_size", 1),
        seed=cfg.get("training", {}).get("seed", 42),
    )
    split_file = dcfg.get("split_file")
    if split_file:
        ds = FlowFragDataset(split_file=split_file, split_key=args.split, **ds_kwargs)
        print(f"Split: {args.split} ({len(ds)} samples)")
    else:
        ds = FlowFragDataset(**ds_kwargs)

    from src.inference.sampler import FlowFragSampler
    from src.inference.metrics import ligand_rmsd, centroid_distance, frag_centroid_rmsd
    from src.geometry.se3 import quaternion_to_matrix

    sigma = args.sigma if args.sigma is not None else cfg["data"].get("prior_sigma", 10.0)
    sampler = FlowFragSampler(
        model,
        num_steps=args.num_steps,
        translation_sigma=sigma,
        time_schedule=args.time_schedule,
        schedule_power=args.schedule_power,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(ds) if args.max_samples <= 0 else min(args.max_samples, len(ds))
    all_rmsds = []
    all_cent_dist = []
    all_frag_rmsds = []

    print(
        f"Running inference on {n_samples} complexes, {args.num_steps} ODE steps "
        f"(schedule={args.time_schedule}, power={args.schedule_power})..."
    )

    for i in range(n_samples):
        data = ds[i]
        pdb_id = data.pdb_id

        result = sampler.sample(data, device=device)

        # Ground-truth atom positions (crystal, pocket-centered)
        frag_id = data["atom"].fragment_id
        local_pos = data["atom"].local_pos
        T_target = data["fragment"].T_target
        q_target = getattr(data["fragment"], "q_target", None)
        if q_target is not None:
            R_target = quaternion_to_matrix(q_target.to(device))
            true_atom_pos = torch.einsum("nij,nj->ni", R_target[frag_id], local_pos.to(device)) + T_target.to(device)[frag_id]
        else:
            true_atom_pos = local_pos + T_target[frag_id]
        true_atom_pos = true_atom_pos.to(device)

        rmsd_val = ligand_rmsd(result["atom_pos_pred"], true_atom_pos).item()
        cd_val = centroid_distance(result["atom_pos_pred"], true_atom_pos).item()
        fr_val = frag_centroid_rmsd(result["T_pred"], T_target.to(device)).item()

        all_rmsds.append(rmsd_val)
        all_cent_dist.append(cd_val)
        all_frag_rmsds.append(fr_val)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n_samples}] {pdb_id}: RMSD={rmsd_val:.2f}A, "
                  f"CentDist={cd_val:.2f}A, FragRMSD={fr_val:.2f}A")

        torch.save({
            "pdb_id": pdb_id,
            "T_pred": result["T_pred"].cpu(),
            "q_pred": result["q_pred"].cpu(),
            "atom_pos_pred": result["atom_pos_pred"].cpu(),
            "rmsd": rmsd_val,
            "centroid_distance": cd_val,
            "frag_centroid_rmsd": fr_val,
        }, out_dir / f"{pdb_id}.pt")

    rmsds_t = torch.tensor(all_rmsds)
    cent_t = torch.tensor(all_cent_dist)
    frag_t = torch.tensor(all_frag_rmsds)

    print(f"\n{'='*50}")
    print(f"Results ({n_samples} complexes, {args.num_steps} steps):")
    print(f"  Ligand RMSD:     {rmsds_t.mean():.2f} +/- {rmsds_t.std():.2f} A")
    print(f"  Centroid Dist:   {cent_t.mean():.2f} +/- {cent_t.std():.2f} A")
    print(f"  Frag RMSD:       {frag_t.mean():.2f} +/- {frag_t.std():.2f} A")
    print(f"  Success (<2A):   {(rmsds_t < 2.0).float().mean():.1%}")
    print(f"  Success (<5A):   {(rmsds_t < 5.0).float().mean():.1%}")
    print(f"  Median RMSD:     {rmsds_t.median():.2f} A")

    torch.save({
        "rmsds": rmsds_t,
        "centroid_distances": cent_t,
        "frag_rmsds": frag_t,
        "num_steps": args.num_steps,
        "time_schedule": args.time_schedule,
        "schedule_power": args.schedule_power,
        "checkpoint": args.checkpoint,
    }, out_dir / "summary.pt")
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
