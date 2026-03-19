"""Visualize ODE trajectory as animated GIF: prior → docked pose.

Layout: [ODE trajectory (3D)] | [molecule by fragments (3D)]
                               | [exploded fragments (3D)]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Element colors (CPK-ish)
ELEM_COLORS = {
    0: "#909090",  # C - grey
    1: "#3050F8",  # N - blue
    2: "#FF0D0D",  # O - red
    3: "#FFFF30",  # S - yellow
    4: "#FF8000",  # P - orange
    5: "#90E050",  # F - green
    6: "#1FF01F",  # Cl - green
    7: "#A62929",  # Br - brown
    8: "#940094",  # I - purple
}

# Distinct fragment colors
FRAG_COLORS = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
    "#42D4F4", "#F032E6", "#BFEF45", "#FABED4", "#469990",
    "#DCBEFF", "#9A6324", "#800000", "#AAFFC3", "#808000",
    "#FFD8B1", "#000075", "#A9A9A9", "#E6BEFF", "#1CE6FF",
]


def render_frame(
    ax: plt.Axes,
    atom_pos: np.ndarray,
    crystal_pos: np.ndarray,
    prot_pos: np.ndarray,
    bond_index: np.ndarray,
    elem: np.ndarray,
    step: int,
    total_steps: int,
    t_val: float,
    pdb_id: str,
    rmsd: float,
    xlim: tuple, ylim: tuple, zlim: tuple,
) -> None:
    ax.cla()

    # Protein CA atoms (small, grey, transparent)
    ax.scatter(
        prot_pos[:, 0], prot_pos[:, 1], prot_pos[:, 2],
        c="#CCCCCC", s=8, alpha=0.3, depthshade=True,
    )

    # Crystal ligand (transparent green wireframe)
    ax.scatter(
        crystal_pos[:, 0], crystal_pos[:, 1], crystal_pos[:, 2],
        c="#00CC00", s=30, alpha=0.25, edgecolors="#009900", linewidths=0.5,
    )
    for i, j in bond_index.T:
        ax.plot(
            [crystal_pos[i, 0], crystal_pos[j, 0]],
            [crystal_pos[i, 1], crystal_pos[j, 1]],
            [crystal_pos[i, 2], crystal_pos[j, 2]],
            c="#00CC00", alpha=0.2, lw=1,
        )

    # Current ligand pose (colored by element)
    colors = [ELEM_COLORS.get(int(e), "#909090") for e in elem]
    ax.scatter(
        atom_pos[:, 0], atom_pos[:, 1], atom_pos[:, 2],
        c=colors, s=50, alpha=0.9, edgecolors="black", linewidths=0.5,
        depthshade=True,
    )
    # Bonds
    for i, j in bond_index.T:
        ax.plot(
            [atom_pos[i, 0], atom_pos[j, 0]],
            [atom_pos[i, 1], atom_pos[j, 1]],
            [atom_pos[i, 2], atom_pos[j, 2]],
            c="#333333", alpha=0.7, lw=1.5,
        )

    ax.set_title(
        f"{pdb_id}  |  step {step}/{total_steps}  (t={t_val:.2f})  |  RMSD={rmsd:.2f}A",
        fontsize=11,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)
    ax.tick_params(labelsize=7)


def fig_to_image(fig: plt.Figure, dpi: int = 120) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return Image.open(buf).copy()


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _find_sdf(pdb_id: str, raw_dir: str = "/mnt/data/PLI/P-L") -> Path | None:
    """Find SDF file for a given pdb_id under raw_dir."""
    import glob
    hits = glob.glob(f"{raw_dir}/*/{pdb_id}/{pdb_id}_ligand.sdf")
    return Path(hits[0]) if hits else None


def render_mol_2d(
    sdf_path: Path,
    fragment_id: np.ndarray,
) -> Image.Image:
    """Render 2D molecule with atoms colored by fragment assignment."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=True)
    AllChem.Compute2DCoords(mol)

    n_frags = int(fragment_id.max()) + 1
    n_atoms = mol.GetNumAtoms()

    # Build highlight maps
    atom_colors = {}
    atom_radii = {}
    highlight_atoms = list(range(n_atoms))
    for i in range(n_atoms):
        fid = int(fragment_id[i]) if i < len(fragment_id) else 0
        atom_colors[i] = tuple(c / 255.0 for c in _hex_to_rgb(FRAG_COLORS[fid % len(FRAG_COLORS)]))
        atom_radii[i] = 0.4

    # Bond colors: same color if both atoms in same fragment, grey otherwise
    bond_colors = {}
    highlight_bonds = []
    for bond in mol.GetBonds():
        bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bid = bond.GetIdx()
        if bi < len(fragment_id) and bj < len(fragment_id) and fragment_id[bi] == fragment_id[bj]:
            fid = int(fragment_id[bi])
            bond_colors[bid] = tuple(c / 255.0 for c in _hex_to_rgb(FRAG_COLORS[fid % len(FRAG_COLORS)]))
        else:
            bond_colors[bid] = (0.3, 0.3, 0.3)
        highlight_bonds.append(bid)

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 400)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.5
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()

    # Add title
    img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(img)
    ax.set_title(f"Molecule ({n_frags} fragments)", fontsize=11)
    ax.axis("off")
    result = fig_to_image(fig)
    plt.close(fig)
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument(
        "--time_schedule",
        type=str,
        default="late",
        choices=("uniform", "late", "early"),
        help="Time grid schedule (default: late, denser near t=1 / crystal)",
    )
    parser.add_argument("--schedule_power", type=float, default=3.0)
    parser.add_argument("--complex_idx", type=int, default=0, help="Index in dataset")
    parser.add_argument("--split", type=str, default="train", help="Split to visualize when split_file is set")
    parser.add_argument("--prior_source", type=str, choices=("sample", "fresh"), default="sample")
    parser.add_argument("--out", type=str, default="outputs/trajectory.gif")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    from src.models.flowfrag import FlowFrag
    model = FlowFrag(**cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)

    # Load dataset
    from src.data.dataset import FlowFragDataset
    dcfg = cfg["data"]
    ds_kwargs = dict(
        root=cfg["data"]["data_dir"],
        translation_sigma=cfg["data"].get("prior_sigma", 10.0),
        max_atoms=cfg["data"].get("max_atoms", 80),
        max_frags=cfg["data"].get("max_frags", 20),
        min_atoms=cfg["data"].get("min_atoms", 5),
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
    else:
        ds = FlowFragDataset(**ds_kwargs)

    data = ds[args.complex_idx]
    pdb_id = data.pdb_id
    print(f"Visualizing: {pdb_id} ({data['atom'].x.shape[0]} atoms, "
          f"{data['fragment'].num_nodes} frags)")

    # Ground-truth atom positions in the dataset target gauge.
    data_dir = Path(cfg["data"]["data_dir"]) / pdb_id
    ligand = torch.load(data_dir / "ligand.pt", weights_only=True)
    protein = torch.load(data_dir / "protein.pt", weights_only=True)
    meta = torch.load(data_dir / "meta.pt", weights_only=True)
    pocket_center = meta["pocket_center"]

    from src.geometry.se3 import quaternion_to_matrix

    frag_id_t = data["atom"].fragment_id
    local_pos_t = data["atom"].local_pos
    T_target = data["fragment"].T_target
    q_target = getattr(data["fragment"], "q_target", None)
    if q_target is not None:
        R_target = quaternion_to_matrix(q_target)
        crystal_pos = (
            torch.einsum("nij,nj->ni", R_target[frag_id_t], local_pos_t)
            + T_target[frag_id_t]
        ).numpy()
    else:
        crystal_pos = (local_pos_t + T_target[frag_id_t]).numpy()

    prot_pos = (protein["res_coords"] - pocket_center).numpy()
    bond_index = ligand["bond_index"].numpy()
    elem = ligand["atom_element"].numpy()
    fragment_id = ligand["fragment_id"].numpy()

    # Render static 2D side panel
    print("Rendering 2D molecule panel...")
    sdf_path = _find_sdf(pdb_id)
    if sdf_path is not None:
        side_panel = render_mol_2d(sdf_path, fragment_id)
    else:
        # Fallback: blank panel with text
        side_panel = Image.new("RGB", (500, 400), "white")
        print(f"  Warning: SDF not found for {pdb_id}, skipping 2D panel")

    # Run ODE integration, saving frames
    from src.geometry.flow_matching import sample_prior_poses, integrate_se3_step
    from src.inference.sampler import build_time_grid
    from src.inference.metrics import ligand_rmsd

    data_dev = data.to(device)
    n_frags = data_dev["fragment"].num_nodes
    frag_sizes = data_dev["fragment"].size

    if args.prior_source == "sample" and hasattr(data_dev["fragment"], "T_prior") and hasattr(data_dev["fragment"], "q_prior"):
        T = data_dev["fragment"].T_prior.clone()
        q = data_dev["fragment"].q_prior.clone()
    else:
        torch.manual_seed(cfg.get("training", {}).get("seed", 42))
        sigma = cfg["data"].get("prior_sigma", 10.0)
        T, q = sample_prior_poses(
            n_frags, torch.zeros(3, device=device), sigma,
            frag_sizes=frag_sizes, device=device, dtype=torch.float32,
        )

    time_grid = build_time_grid(
        args.num_steps,
        schedule=args.time_schedule,
        power=args.schedule_power,
        device=device,
        dtype=torch.float32,
    )
    frames = []

    # Compute axis limits from crystal + prior
    R_init = quaternion_to_matrix(q)
    fid = data_dev["atom"].fragment_id
    local_pos = data_dev["atom"].local_pos
    init_pos = (
        torch.einsum("nij,nj->ni", R_init[fid], local_pos) + T[fid]
    ).cpu().numpy()

    # Focus on ligand region: use crystal + init positions (not full pocket)
    lig_pos = np.concatenate([crystal_pos, init_pos], axis=0)
    center = crystal_pos.mean(axis=0)
    spread = max(np.abs(lig_pos - center).max() + 3, 5)
    xlim = (center[0] - spread, center[0] + spread)
    ylim = (center[1] - spread, center[1] + spread)
    zlim = (center[2] - spread, center[2] + spread)

    # Filter protein to nearby residues only
    prot_dists = np.linalg.norm(prot_pos - center, axis=-1)
    prot_mask = prot_dists < spread + 3
    prot_pos = prot_pos[prot_mask]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=25, azim=45)

    crystal_pos_t = torch.from_numpy(crystal_pos).float().to(device)

    for step in range(args.num_steps + 1):
        t = float(time_grid[step].item())

        # Current atom positions
        R = quaternion_to_matrix(q)
        atom_pos = torch.einsum("nij,nj->ni", R[fid], local_pos) + T[fid]
        rmsd = ligand_rmsd(atom_pos, crystal_pos_t).item()
        atom_pos_np = atom_pos.cpu().numpy()

        render_frame(
            ax, atom_pos_np, crystal_pos, prot_pos, bond_index, elem,
            step, args.num_steps, t, pdb_id, rmsd, xlim, ylim, zlim,
        )
        traj_img = fig_to_image(fig)

        # Composite: [trajectory | side panels]
        # Resize side panel to match trajectory height
        sp = side_panel.resize(
            (int(side_panel.width * traj_img.height / side_panel.height), traj_img.height),
            Image.LANCZOS,
        )
        composite = Image.new("RGB", (traj_img.width + sp.width, traj_img.height), "white")
        composite.paste(traj_img, (0, 0))
        composite.paste(sp, (traj_img.width, 0))
        frames.append(composite)

        if step < args.num_steps:
            # Model forward
            data_dev["fragment"].T_frag = T
            data_dev["fragment"].q_frag = q
            data_dev.t = torch.tensor([t], device=device, dtype=torch.float32)
            data_dev["atom"].pos_t = atom_pos

            with torch.no_grad():
                out = model(data_dev)

            T, q = integrate_se3_step(
                T,
                q,
                out["v_pred"],
                out["omega_pred"],
                time_grid[step + 1] - time_grid[step],
                frag_sizes=frag_sizes,
            )

        if step % 10 == 0:
            print(f"  step {step:3d}/{args.num_steps}: RMSD={rmsd:.2f}A")

    plt.close(fig)

    # Save GIF (hold first/last frame longer)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    durations = [100] * len(frames)
    durations[0] = 1000   # hold initial frame
    durations[-1] = 1500  # hold final frame
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=True,
    )
    print(f"\nSaved trajectory GIF: {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
