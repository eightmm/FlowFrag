"""Visualize ODE trajectory for UnifiedFlowFrag model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.unified_dataset import UnifiedDataset, unified_collate
from src.models.unified import UnifiedFlowFrag
from src.geometry.flow_matching import integrate_se3_step, sample_prior_poses
from src.geometry.se3 import quaternion_to_matrix
from src.inference.sampler import build_time_grid
from src.inference.metrics import ligand_rmsd

ELEM_COLORS = {
    0: "#909090", 1: "#3050F8", 2: "#FF0D0D", 3: "#FFFF30",
    4: "#FF8000", 5: "#90E050", 6: "#1FF01F", 7: "#A62929", 8: "#940094",
}


def run_ode_unified(model, sample, num_steps=25, sigma=1.0,
                    device=torch.device("cuda"), time_schedule="late",
                    schedule_power=3.0, prior_seed=None):
    """Run ODE integration on a single unified sample."""
    model.train(False)

    n_frag = sample["num_lig_frag"].item()
    frag_sizes = sample["frag_sizes"].to(device)
    frag_id = sample["frag_id_for_atoms"].to(device)
    local_pos = sample["local_pos"].to(device)

    gen = None
    if prior_seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(prior_seed)

    # Sample prior on CPU (generator is CPU), then move to device
    T, q = sample_prior_poses(
        n_frag, torch.zeros(3), sigma,
        frag_sizes=frag_sizes.cpu(), dtype=torch.float32,
        generator=gen,
    )
    T = T.to(device)
    q = q.to(device)

    time_grid = build_time_grid(num_steps, schedule=time_schedule,
                                power=schedule_power, device=device,
                                dtype=torch.float32)
    trajectory = []

    for step in range(num_steps + 1):
        t_val = time_grid[step]
        R = quaternion_to_matrix(q)
        atom_pos = torch.einsum("nij,nj->ni", R[frag_id], local_pos) + T[frag_id]

        trajectory.append({
            "T": T.cpu().clone(), "q": q.cpu().clone(),
            "atom_pos": atom_pos.cpu().clone(), "t": t_val.item(),
        })

        if step >= num_steps:
            break

        dt = time_grid[step + 1] - time_grid[step]

        # Build single-sample batch
        batch = unified_collate([sample])
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

        # Override dynamic state
        node_coords = batch_gpu["node_coords"].clone()
        frag_slice = batch_gpu["lig_frag_slice"][0]
        frag_start = frag_slice[0].item()
        node_coords[frag_start:frag_start + n_frag] = T

        atom_slice = batch_gpu["lig_atom_slice"][0]
        atom_start = atom_slice[0].item()
        node_coords[atom_start:atom_start + atom_pos.shape[0]] = atom_pos

        batch_gpu["node_coords"] = node_coords
        batch_gpu["T_frag"] = T
        batch_gpu["q_frag"] = q
        batch_gpu["frag_sizes"] = frag_sizes
        batch_gpu["t"] = t_val.view(1, 1)

        with torch.no_grad():
            out = model(batch_gpu)

        T, q = integrate_se3_step(
            T, q, out["v_pred"], out["omega_pred"], dt, frag_sizes=frag_sizes,
        )

    return trajectory


def render_frame(ax, atom_pos, crystal_pos, prot_pos, bond_index, elem,
                 step, total_steps, t_val, pdb_id, rmsd, xlim, ylim, zlim):
    ax.cla()
    if prot_pos is not None and len(prot_pos) > 0:
        ax.scatter(prot_pos[:, 0], prot_pos[:, 1], prot_pos[:, 2],
                   c="#CCCCCC", s=8, alpha=0.3, depthshade=True)
    # Crystal (green wireframe)
    ax.scatter(crystal_pos[:, 0], crystal_pos[:, 1], crystal_pos[:, 2],
               c="#00CC00", s=30, alpha=0.25, edgecolors="#009900", linewidths=0.5)
    if bond_index is not None:
        for i, j in bond_index.T:
            if i < len(crystal_pos) and j < len(crystal_pos):
                ax.plot([crystal_pos[i, 0], crystal_pos[j, 0]],
                        [crystal_pos[i, 1], crystal_pos[j, 1]],
                        [crystal_pos[i, 2], crystal_pos[j, 2]],
                        c="#00CC00", alpha=0.2, lw=1)
    # Current pose
    colors = [ELEM_COLORS.get(int(e), "#909090") for e in elem]
    ax.scatter(atom_pos[:, 0], atom_pos[:, 1], atom_pos[:, 2],
               c=colors, s=50, alpha=0.9, edgecolors="black", linewidths=0.5,
               depthshade=True)
    if bond_index is not None:
        for i, j in bond_index.T:
            if i < len(atom_pos) and j < len(atom_pos):
                ax.plot([atom_pos[i, 0], atom_pos[j, 0]],
                        [atom_pos[i, 1], atom_pos[j, 1]],
                        [atom_pos[i, 2], atom_pos[j, 2]],
                        c="#333333", alpha=0.7, lw=1.5)

    title = f"{pdb_id}  |  step {step}/{total_steps}  (t={t_val:.2f})  |  RMSD={rmsd:.2f}A"
    ax.set_title(title, fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)
    ax.tick_params(labelsize=7)


def fig_to_image(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return Image.open(buf).copy()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--time_schedule", type=str, default="late")
    parser.add_argument("--schedule_power", type=float, default=3.0)
    parser.add_argument("--complex_idx", type=int, default=0)
    parser.add_argument("--prior_seed", type=int, default=None, help="Seed for prior sampling")
    parser.add_argument("--out", type=str, default="outputs/trajectory_unified.gif")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_kwargs = {k: v for k, v in cfg["model"].items() if k != "model_type"}
    model = UnifiedFlowFrag(**model_kwargs).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)

    # Load dataset
    dcfg = cfg["data"]
    ds = UnifiedDataset(
        root=dcfg["data_dir"],
        translation_sigma=dcfg.get("prior_sigma", 1.0),
        max_atoms=dcfg.get("max_atoms", 80),
        max_frags=dcfg.get("max_frags", 20),
        min_atoms=dcfg.get("min_atoms", 5),
        rotation_augmentation=dcfg.get("rotation_augmentation", "none"),
        deterministic_augmentation=dcfg.get("deterministic_augmentation"),
        seed=cfg.get("training", {}).get("seed", 42),
    )

    sample = ds[args.complex_idx]
    pdb_id = sample["pdb_id"]
    print(f"Visualizing: {pdb_id}")

    # Crystal (target) positions
    q_tgt = sample["q_target"]
    R_tgt = quaternion_to_matrix(q_tgt)
    fid = sample["frag_id_for_atoms"]
    lp = sample["local_pos"]
    crystal_pos = (torch.einsum("nij,nj->ni", R_tgt[fid], lp) + sample["T_target"][fid]).numpy()

    # Protein CA coords
    ca_slice = sample["prot_ca_slice"]
    prot_pos = sample["node_coords"][ca_slice[0]:ca_slice[1]].numpy()

    # Ligand atom info (real atoms only, excluding dummies)
    lig_s = sample["lig_atom_slice"][0].item()
    lig_e = sample["lig_atom_slice"][1].item()
    is_dummy = sample["node_is_dummy"][lig_s:lig_e]
    real_mask = ~is_dummy
    elem_all = sample["node_element"][lig_s:lig_e]
    elem = elem_all[real_mask].numpy()

    # Map global atom indices to real-only local indices
    real_indices = real_mask.nonzero(as_tuple=True)[0]  # positions of real atoms
    global_to_real = torch.full((lig_e - lig_s,), -1, dtype=torch.long)
    global_to_real[real_indices] = torch.arange(len(real_indices))

    # Bond edges (edge_type == 0, remap to real-only indices)
    emask = sample["edge_type"] == 0
    bond_ei_global = sample["edge_index"][:, emask] - lig_s
    # Keep only bonds between real atoms
    src_real = global_to_real[bond_ei_global[0]]
    dst_real = global_to_real[bond_ei_global[1]]
    valid = (src_real >= 0) & (dst_real >= 0)
    bond_ei = torch.stack([src_real[valid], dst_real[valid]]).numpy()

    # Run ODE
    sigma = dcfg.get("prior_sigma", 1.0)
    traj = run_ode_unified(
        model, sample, num_steps=args.num_steps, sigma=sigma, device=device,
        time_schedule=args.time_schedule, schedule_power=args.schedule_power,
        prior_seed=args.prior_seed,
    )

    # Compute axis limits
    all_pos = np.concatenate([crystal_pos] + [s["atom_pos"].numpy() for s in traj])
    center = crystal_pos.mean(axis=0)
    spread = max(np.abs(all_pos - center).max() + 3, 5)
    xlim = (center[0] - spread, center[0] + spread)
    ylim = (center[1] - spread, center[1] + spread)
    zlim = (center[2] - spread, center[2] + spread)

    if len(prot_pos) > 0:
        pdist = np.linalg.norm(prot_pos - center, axis=-1)
        prot_pos = prot_pos[pdist < spread + 3]

    crystal_t = torch.from_numpy(crystal_pos).float().to(device)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(25, 45)

    frames = []
    for i, state in enumerate(traj):
        apos = state["atom_pos"].numpy()
        rmsd = ligand_rmsd(state["atom_pos"].to(device), crystal_t).item()
        render_frame(ax, apos, crystal_pos, prot_pos, bond_ei, elem,
                     i, args.num_steps, state["t"], pdb_id, rmsd,
                     xlim, ylim, zlim)
        frames.append(fig_to_image(fig))
        if i % 5 == 0:
            print(f"  step {i:3d}/{args.num_steps}: t={state['t']:.3f} RMSD={rmsd:.2f}A")

    plt.close(fig)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    durations = [100] * len(frames)
    durations[0] = 1000
    durations[-1] = 1500
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"\nSaved: {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
