"""Evaluate unified models on overfit set with multiple prior seeds."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.unified_dataset import UnifiedDataset, unified_collate
from src.models.unified import UnifiedFlowFrag
from src.geometry.flow_matching import integrate_se3_step, sample_prior_poses
from src.geometry.se3 import quaternion_to_matrix
from src.inference.sampler import build_time_grid
from src.inference.metrics import ligand_rmsd


@torch.no_grad()
def rollout_single(model, sample, sigma, device, seed, num_steps=25):
    """Run one ODE rollout, return final RMSD."""
    n_frag = sample["num_lig_frag"].item()
    frag_sizes = sample["frag_sizes"].to(device)
    frag_id = sample["frag_id_for_atoms"].to(device)
    local_pos = sample["local_pos"].to(device)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    T, q = sample_prior_poses(
        n_frag, torch.zeros(3), sigma,
        frag_sizes=frag_sizes.cpu(), dtype=torch.float32, generator=gen,
    )
    T, q = T.to(device), q.to(device)

    time_grid = build_time_grid(num_steps, schedule="late", power=3.0,
                                device=device, dtype=torch.float32)

    for step in range(num_steps):
        t_val = time_grid[step]
        dt = time_grid[step + 1] - time_grid[step]

        R = quaternion_to_matrix(q)
        atom_pos = torch.einsum("nij,nj->ni", R[frag_id], local_pos) + T[frag_id]

        batch = unified_collate([sample])
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

        node_coords = batch_gpu["node_coords"].clone()
        frag_slice = batch_gpu["lig_frag_slice"][0]
        node_coords[frag_slice[0].item():frag_slice[0].item() + n_frag] = T
        atom_slice = batch_gpu["lig_atom_slice"][0]
        node_coords[atom_slice[0].item():atom_slice[0].item() + atom_pos.shape[0]] = atom_pos

        batch_gpu["node_coords"] = node_coords
        batch_gpu["T_frag"] = T
        batch_gpu["q_frag"] = q
        batch_gpu["frag_sizes"] = frag_sizes
        batch_gpu["t"] = t_val.view(1, 1)

        out = model(batch_gpu)
        T, q = integrate_se3_step(T, q, out["v_pred"], out["omega_pred"], dt,
                                  frag_sizes=frag_sizes)

    # Final positions
    R_final = quaternion_to_matrix(q)
    atom_pos_final = torch.einsum("nij,nj->ni", R_final[frag_id], local_pos) + T[frag_id]

    # Crystal target
    R_tgt = quaternion_to_matrix(sample["q_target"].to(device))
    crystal_pos = (
        torch.einsum("nij,nj->ni", R_tgt[frag_id], local_pos)
        + sample["T_target"].to(device)[frag_id]
    )

    return ligand_rmsd(atom_pos_final, crystal_pos).item()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=25)
    args = parser.parse_args()

    device = torch.device("cuda")

    configs = [
        ("Direct+2e", "configs/overfit_unified.yaml", "outputs/overfit_unified/checkpoints/latest.pt"),
        ("N-E+2e", "configs/overfit_unified_newton_euler.yaml", "outputs/overfit_unified_newton_euler/checkpoints/latest.pt"),
        ("No 2e", "configs/overfit_unified_no2e.yaml", "outputs/overfit_unified_no2e/checkpoints/latest.pt"),
        ("2e+2o", "configs/overfit_unified_2e2o.yaml", "outputs/overfit_unified_2e2o/checkpoints/latest.pt"),
        ("2e+2o+NE", "configs/overfit_unified_2e2o_ne.yaml", "outputs/overfit_unified_2e2o_ne/checkpoints/latest.pt"),
    ]

    seeds = list(range(args.num_seeds))

    # Load dataset once (same for all)
    ds = UnifiedDataset(
        root="data/processed_v2", max_atoms=80, max_frags=20,
        translation_sigma=1.0, rotation_augmentation="ligand_uniform",
        deterministic_augmentation=True, seed=42,
    )
    # Overfit set: first 16 samples
    n_samples = min(16, len(ds))

    # Collect pdb_ids
    pdb_ids = []
    for i in range(n_samples):
        s = ds[i]
        pdb_ids.append(s["pdb_id"])

    all_results = {}

    for name, cfg_path, ckpt_path in configs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        model_kwargs = {k: v for k, v in cfg["model"].items() if k != "model_type"}
        model = UnifiedFlowFrag(**model_kwargs).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.train(False)

        sigma = cfg["data"].get("prior_sigma", 1.0)
        results = np.zeros((n_samples, args.num_seeds))

        for i in range(n_samples):
            sample = ds[i]
            for s in seeds:
                rmsd = rollout_single(model, sample, sigma, device, seed=s * 1000 + i)
                results[i, s] = rmsd
            mean = results[i].mean()
            std = results[i].std()
            print(f"  {pdb_ids[i]:6s}: {mean:.2f} +/- {std:.2f} A  "
                  f"(min={results[i].min():.2f}, max={results[i].max():.2f})")

        all_results[name] = results
        del model
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Mean RMSD (Å) over 10 seeds")
    print(f"{'='*80}")

    header = f"{'PDB':>6s}"
    for name, _, _ in configs:
        header += f"  {name:>12s}"
    print(header)
    print("-" * len(header))

    for i in range(n_samples):
        row = f"{pdb_ids[i]:>6s}"
        for name, _, _ in configs:
            m = all_results[name][i].mean()
            s = all_results[name][i].std()
            row += f"  {m:5.2f}±{s:.2f}"
        print(row)

    # Overall stats
    print("-" * len(header))
    row = f"{'MEAN':>6s}"
    for name, _, _ in configs:
        row += f"  {all_results[name].mean():5.2f}±{all_results[name].mean(axis=1).std():.2f}"
    print(row)

    row = f"{'MED':>6s}"
    for name, _, _ in configs:
        medians = np.median(all_results[name], axis=1)
        row += f"  {np.median(medians):11.2f}"
    print(row)

    row = f"{'<2A%':>6s}"
    for name, _, _ in configs:
        pct = (all_results[name] < 2.0).mean() * 100
        row += f"  {pct:10.1f}%"
    print(row)


if __name__ == "__main__":
    main()
