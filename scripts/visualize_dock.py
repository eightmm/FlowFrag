#!/usr/bin/env python
"""Visualize docking trajectory from dock.py --save_traj output.

Reads results.pt + protein PDB to render an animated GIF showing
ligand fragments flowing from prior (t=0) to docked pose (t=1).

Usage:
    python scripts/visualize_dock.py \
        --results outputs/docked_traj_test/results.pt \
        --protein /mnt/data/PLI/P-L/1981-2000/10gs/10gs_pocket.pdb \
        --out outputs/docked_traj_test/traj.gif

    # With reference crystal ligand for RMSD overlay
    python scripts/visualize_dock.py \
        --results outputs/docked_traj_test/results.pt \
        --protein pocket.pdb \
        --ligand_ref ligand.sdf \
        --out traj.gif

    # Pick a specific sample from multi-sample run
    python scripts/visualize_dock.py \
        --results results.pt --protein pocket.pdb \
        --sample_idx 2 --out traj_sample2.gif
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Element colors (CPK)
ELEM_COLORS = {
    0: "#909090",  # C
    1: "#3050F8",  # N
    2: "#FF0D0D",  # O
    3: "#FFFF30",  # S
    4: "#FF8000",  # P
    5: "#90E050",  # F
    6: "#1FF01F",  # Cl
    7: "#A62929",  # Br
    8: "#940094",  # I
}
DEFAULT_COLOR = "#909090"

# Fragment palette
FRAG_COLORS = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
    "#42D4F4", "#F032E6", "#BFEF45", "#FABED4", "#469990",
    "#DCBEFF", "#9A6324", "#800000", "#AAFFC3", "#808000",
]


def load_protein_ca(pdb_path: Path) -> np.ndarray:
    """Extract CA coordinates from PDB."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
    return np.array(coords, dtype=np.float32)


def load_ref_ligand(sdf_path: Path) -> np.ndarray | None:
    """Load crystal ligand coords from SDF for RMSD overlay."""
    from rdkit import Chem
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=True)
    if mol is None:
        return None
    conf = mol.GetConformer()
    return np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )


def compute_rmsd(pos: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((pos - ref) ** 2, axis=-1))))


def fig_to_image(fig: plt.Figure, dpi: int = 120) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def render_frame(
    fig: plt.Figure,
    ax: plt.Axes,
    atom_pos: np.ndarray,
    frag_id: np.ndarray,
    prot_pos: np.ndarray,
    ref_pos: np.ndarray | None,
    step: int,
    total_steps: int,
    t_val: float,
    rmsd: float | None,
) -> Image.Image:
    ax.cla()

    # Protein CA (grey, transparent)
    if prot_pos is not None and len(prot_pos) > 0:
        ax.scatter(
            prot_pos[:, 0], prot_pos[:, 1], prot_pos[:, 2],
            c="#CCCCCC", s=8, alpha=0.25, depthshade=True,
        )

    # Reference ligand (transparent green)
    if ref_pos is not None:
        ax.scatter(
            ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2],
            c="#00CC00", s=30, alpha=0.2, edgecolors="#009900", linewidths=0.5,
        )

    # Current ligand (colored by fragment)
    n_atoms = atom_pos.shape[0]
    colors = [FRAG_COLORS[int(frag_id[i]) % len(FRAG_COLORS)] for i in range(n_atoms)]
    ax.scatter(
        atom_pos[:, 0], atom_pos[:, 1], atom_pos[:, 2],
        c=colors, s=60, alpha=0.9, edgecolors="black", linewidths=0.5,
        depthshade=True,
    )

    # Title
    rmsd_str = f"  RMSD={rmsd:.2f}A" if rmsd is not None else ""
    ax.set_title(f"step {step}/{total_steps}  (t={t_val:.3f}){rmsd_str}", fontsize=11)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)
    ax.tick_params(labelsize=7)

    return fig_to_image(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize docking trajectory as GIF")
    parser.add_argument("--results", type=str, required=True,
                        help="results.pt from dock.py --save_traj")
    parser.add_argument("--protein", type=str, required=True, help="Protein pocket PDB")
    parser.add_argument("--ligand_ref", type=str, default=None,
                        help="Reference crystal ligand SDF (for RMSD overlay)")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Which sample trajectory to visualize (default: 0)")
    parser.add_argument("--out", type=str, default=None, help="Output GIF path")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument("--elev", type=float, default=25, help="3D view elevation")
    parser.add_argument("--azim", type=float, default=45, help="3D view azimuth")
    parser.add_argument("--rotate", action="store_true",
                        help="Slowly rotate view during animation")
    args = parser.parse_args()

    results_path = Path(args.results)
    assert results_path.exists(), f"Not found: {results_path}"

    data = torch.load(results_path, weights_only=False)
    pocket_center = data["pocket_center"].numpy()

    assert "trajectories" in data, (
        "No trajectory data in results.pt. Re-run dock.py with --save_traj"
    )

    traj_data = data["trajectories"][args.sample_idx]
    traj_frames = traj_data["traj"]  # list of [N_atom, 3] tensors (pocket-centered)
    traj_times = traj_data["traj_times"]  # list of float
    n_frames = len(traj_frames)

    # Fragment IDs from frag_sizes
    frag_sizes = data["frag_sizes"].numpy()
    n_atoms = traj_frames[0].shape[0]
    frag_id = np.zeros(n_atoms, dtype=np.int64)
    # Reconstruct fragment_id from frag_sizes (atoms are ordered by fragment)
    # But we don't have the actual mapping — use a simpler approach: just color uniformly
    # Actually we can infer from the results data
    # For now, try to build from frag_centers count
    n_frags = len(frag_sizes)
    if n_frags == 1:
        frag_id[:] = 0
    else:
        # Use spatial clustering: assign each atom to nearest fragment center
        frag_centers = data["frag_centers"].numpy()  # crystal coords, not pocket-centered
        frag_centers_pc = frag_centers - pocket_center  # pocket-centered
        final_pos = traj_frames[-1].numpy() if isinstance(traj_frames[-1], torch.Tensor) else traj_frames[-1]
        dists = np.linalg.norm(
            final_pos[:, None, :] - frag_centers_pc[None, :, :], axis=-1
        )  # [N_atom, N_frag]
        frag_id = np.argmin(dists, axis=1)

    # Load protein CA
    prot_ca = load_protein_ca(Path(args.protein))  # absolute coords
    prot_ca_pc = prot_ca - pocket_center  # pocket-centered

    # Load reference ligand
    ref_pos = None
    if args.ligand_ref:
        ref_abs = load_ref_ligand(Path(args.ligand_ref))
        if ref_abs is not None:
            ref_pos = ref_abs - pocket_center

    # Compute axis limits from all frames + protein
    all_pos = np.concatenate(
        [f.numpy() if isinstance(f, torch.Tensor) else f for f in traj_frames], axis=0
    )
    center = all_pos.mean(axis=0)
    spread = max(np.abs(all_pos - center).max() + 2, 5)

    # Filter protein to nearby region
    prot_dists = np.linalg.norm(prot_ca_pc - center, axis=-1)
    prot_nearby = prot_ca_pc[prot_dists < spread + 5]

    # Render frames
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    images = []
    total_steps = n_frames - 1

    print(f"Rendering {n_frames} frames...")
    for i, (frame, t) in enumerate(zip(traj_frames, traj_times)):
        pos = frame.numpy() if isinstance(frame, torch.Tensor) else frame

        rmsd = compute_rmsd(pos, ref_pos) if ref_pos is not None else None

        azim = args.azim + (i * 180 / n_frames) if args.rotate else args.azim
        ax.view_init(elev=args.elev, azim=azim)
        ax.set_xlim(center[0] - spread, center[0] + spread)
        ax.set_ylim(center[1] - spread, center[1] + spread)
        ax.set_zlim(center[2] - spread, center[2] + spread)

        img = render_frame(
            fig, ax, pos, frag_id, prot_nearby, ref_pos,
            i, total_steps, t, rmsd,
        )
        images.append(img)

        if i % 5 == 0 or i == n_frames - 1:
            rmsd_str = f", RMSD={rmsd:.2f}A" if rmsd is not None else ""
            print(f"  frame {i}/{total_steps} (t={t:.3f}{rmsd_str})")

    plt.close(fig)

    # Save GIF
    out_path = Path(args.out) if args.out else results_path.parent / "traj.gif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ms_per_frame = 1000 // args.fps
    durations = [ms_per_frame] * len(images)
    durations[0] = 1500   # hold initial
    durations[-1] = 2000  # hold final

    images[0].save(
        out_path, save_all=True, append_images=images[1:],
        duration=durations, loop=0, optimize=True,
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"\nSaved: {out_path} ({n_frames} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
