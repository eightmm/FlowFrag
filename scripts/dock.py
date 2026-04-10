#!/usr/bin/env python
"""Dock a ligand into a protein pocket using a trained FlowFrag model.

Accepts raw PDB + ligand (SMILES, SDF, or MOL2) and outputs docked poses as SDF.

Usage:
    # From SMILES
    python scripts/dock.py \
        --protein data/example/pocket.pdb \
        --ligand "CCO" \
        --checkpoint outputs/checkpoints/latest.pt \
        --config configs/overfit_unified.yaml

    # From SDF file
    python scripts/dock.py \
        --protein data/example/pocket.pdb \
        --ligand data/example/ligand.sdf \
        --checkpoint outputs/checkpoints/latest.pt \
        --config configs/overfit_unified.yaml

    # From MOL2 file
    python scripts/dock.py \
        --protein data/example/pocket.pdb \
        --ligand data/example/ligand.mol2 \
        --checkpoint outputs/checkpoints/latest.pt \
        --config configs/overfit_unified.yaml

    # Multiple samples + custom ODE settings
    python scripts/dock.py \
        --protein pocket.pdb --ligand ligand.sdf \
        --checkpoint ckpt.pt --config cfg.yaml \
        --num_samples 10 --num_steps 50 --time_schedule late
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.geometry.flow_matching import integrate_se3_step, sample_prior_poses
from src.geometry.se3 import quaternion_to_matrix
from src.preprocess.fragments import decompose_fragments
from src.preprocess.graph import build_static_complex_graph
from src.preprocess.ligand import featurize_ligand, load_molecule
from src.preprocess.protein import parse_pocket_atoms


# ---------------------------------------------------------------------------
# Ligand loading: SMILES / SDF / MOL2
# ---------------------------------------------------------------------------

def _get_pocket_center(pdb_path: Path) -> torch.Tensor:
    """Quick CA-based pocket center from PDB (no RDKit, just CA lines)."""
    ca_coords = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            ca_coords.append([x, y, z])
    assert ca_coords, f"No CA atoms found in {pdb_path}"
    return torch.tensor(ca_coords, dtype=torch.float32).mean(dim=0)


def load_ligand(ligand_input: str) -> tuple[Chem.Mol, bool]:
    """Load ligand from SMILES string, SDF file, or MOL2 file.

    Returns (mol, has_pose):
        mol: RDKit Mol with 3D conformer, hydrogens removed.
        has_pose: True if ligand coords are meaningful (SDF/MOL2 with crystal
            coords), False if generated from SMILES (arbitrary coords).
    """
    path = Path(ligand_input)

    if path.suffix.lower() == ".sdf" and path.exists():
        mol, _ = load_molecule(path)
        assert mol is not None, f"Failed to parse SDF: {ligand_input}"
        return mol, True

    if path.suffix.lower() == ".mol2" and path.exists():
        mol = Chem.MolFromMol2File(str(path), sanitize=False)
        assert mol is not None, f"Failed to parse MOL2: {ligand_input}"
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        assert mol.GetNumConformers() > 0, "MOL2 has no 3D conformer"
        return mol, True

    # Try as SMILES (coords are arbitrary — not a real binding pose)
    mol = Chem.MolFromSmiles(ligand_input)
    assert mol is not None, f"Invalid SMILES: {ligand_input}"
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    assert status == 0, f"3D embedding failed for SMILES: {ligand_input}"
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    mol = Chem.RemoveHs(mol)
    return mol, False


# ---------------------------------------------------------------------------
# Preprocessing: raw inputs -> unified graph dict
# ---------------------------------------------------------------------------

def preprocess_complex(
    protein_pdb: Path,
    mol: Chem.Mol,
    ligand_has_pose: bool = True,
) -> tuple[dict[str, torch.Tensor], dict, dict]:
    """Preprocess a protein-ligand complex into unified graph tensors.

    Args:
        protein_pdb: Path to pocket PDB file.
        mol: RDKit Mol with 3D conformer (heavy atoms only).
        ligand_has_pose: If False (e.g. SMILES input), the ligand 3D coords
            are arbitrary. We translate them to the pocket center so the 8A
            cutoff can select the right residues. The actual starting pose
            will be sampled from the prior anyway.

    Returns (graph, lig_data, meta).
    """
    lig_data = featurize_ligand(mol)
    assert lig_data is not None, "Ligand featurization failed"

    # For SMILES input: center ligand at pocket center so cutoff works
    if not ligand_has_pose:
        pocket_ca = _get_pocket_center(protein_pdb)
        lig_centroid = lig_data["atom_coords"].mean(dim=0)
        shift = pocket_ca - lig_centroid
        lig_data["atom_coords"] = lig_data["atom_coords"] + shift
        # Update the conformer in the mol too
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            old = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, [old.x + shift[0].item(),
                                     old.y + shift[1].item(),
                                     old.z + shift[2].item()])

    frag_data = decompose_fragments(mol, lig_data["atom_coords"])
    assert frag_data is not None, "Fragment decomposition failed"

    # Merge fragment data into lig_data (same as build_fragment_flow_dataset.py)
    lig_data["fragment_id"] = frag_data["fragment_id"]
    lig_data["frag_centers"] = frag_data["frag_centers"]
    lig_data["frag_local_coords"] = frag_data["frag_local_coords"]
    lig_data["frag_sizes"] = frag_data["frag_sizes"]
    lig_data["tri_edge_index"] = frag_data["tri_edge_index"]
    lig_data["tri_edge_ref_dist"] = frag_data["tri_edge_ref_dist"]
    lig_data["fragment_adj_index"] = frag_data["fragment_adj_index"]
    lig_data["cut_bond_index"] = frag_data["cut_bond_index"]

    patom_data = parse_pocket_atoms(
        protein_pdb,
        ligand_coords=lig_data["atom_coords"],
        cutoff=8.0,
    )
    assert patom_data is not None, "Protein pocket parsing failed (no residues within 8A)"

    pocket_center = patom_data["pca_coords"].mean(dim=0)

    graph = build_static_complex_graph(lig_data, patom_data)

    meta = {
        "pocket_center": pocket_center,
        "num_frag": frag_data["n_frags"],
        "num_atom": lig_data["atom_coords"].shape[0],
    }

    return graph, lig_data, meta


# ---------------------------------------------------------------------------
# ODE sampler for unified model (flat dict format)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_unified(
    model: torch.nn.Module,
    graph: dict[str, torch.Tensor],
    lig_data: dict,
    meta: dict,
    *,
    num_steps: int = 25,
    translation_sigma: float = 1.0,
    time_schedule: str = "late",
    schedule_power: float = 3.0,
    device: torch.device = torch.device("cpu"),
    save_traj: bool = False,
) -> dict[str, torch.Tensor]:
    """Run ODE integration on a single complex using the unified model.

    Returns dict with T_pred, q_pred, atom_pos_pred.
    If save_traj=True, also returns trajectory (list of atom positions per step).
    """
    from src.inference.sampler import build_time_grid

    n_frags = meta["num_frag"]
    frag_sizes = lig_data["frag_sizes"]
    frag_id = lig_data["fragment_id"]
    local_pos = lig_data["frag_local_coords"]

    # Sample prior
    T, q = sample_prior_poses(
        n_frags,
        pocket_center=torch.zeros(3),
        translation_sigma=translation_sigma,
        frag_sizes=frag_sizes,
    )

    time_grid = build_time_grid(
        num_steps,
        schedule=time_schedule,
        power=schedule_power,
        device=device,
        dtype=torch.float32,
    )

    # Prepare batch dict (single sample, so batch index = all zeros)
    batch: dict[str, torch.Tensor] = {}
    for k, v in graph.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    n_nodes = batch["node_coords"].shape[0]
    batch["batch"] = torch.zeros(n_nodes, dtype=torch.long, device=device)
    batch["frag_batch"] = torch.zeros(n_frags, dtype=torch.long, device=device)

    # Slices for coordinate updates
    frag_slice = graph["lig_frag_slice"]
    frag_start, frag_end = frag_slice[0].item(), frag_slice[1].item()
    atom_slice = graph["lig_atom_slice"]
    atom_start = atom_slice[0].item()
    n_real_atoms = local_pos.shape[0]

    T = T.to(device)
    q = q.to(device)
    frag_sizes_d = frag_sizes.to(device)
    local_pos_d = local_pos.to(device)
    frag_id_d = frag_id.to(device)

    # Trajectory: record atom positions at each step
    traj: list[torch.Tensor] = []
    traj_times: list[float] = []

    for step_idx in range(num_steps):
        t = time_grid[step_idx]
        dt = time_grid[step_idx + 1] - time_grid[step_idx]

        # Update coordinates in batch
        node_coords = batch["node_coords"].clone()
        node_coords[frag_start:frag_end] = T

        R = quaternion_to_matrix(q)
        atom_pos_t = (
            torch.einsum("nij,nj->ni", R[frag_id_d], local_pos_d)
            + T[frag_id_d]
        )
        node_coords[atom_start:atom_start + n_real_atoms] = atom_pos_t
        batch["node_coords"] = node_coords

        if save_traj:
            traj.append(atom_pos_t.cpu())
            traj_times.append(t.item())

        batch["T_frag"] = T
        batch["q_frag"] = q
        batch["frag_sizes"] = frag_sizes_d
        batch["t"] = t.view(1, 1)
        batch["frag_id_for_atoms"] = frag_id_d

        out = model(batch)
        v_pred = out["v_pred"]
        omega_pred = out["omega_pred"]

        T, q = integrate_se3_step(
            T, q, v_pred, omega_pred, dt, frag_sizes=frag_sizes_d,
        )

    # Final atom positions
    R_final = quaternion_to_matrix(q)
    atom_pos_pred = (
        torch.einsum("nij,nj->ni", R_final[frag_id_d], local_pos_d)
        + T[frag_id_d]
    )

    if save_traj:
        traj.append(atom_pos_pred.cpu())
        traj_times.append(1.0)

    result = {
        "T_pred": T.cpu(),
        "q_pred": q.cpu(),
        "atom_pos_pred": atom_pos_pred.cpu(),
    }
    if save_traj:
        result["traj"] = traj          # list of [N_atom, 3]
        result["traj_times"] = traj_times  # list of float
    return result


# ---------------------------------------------------------------------------
# SDF output
# ---------------------------------------------------------------------------

def write_sdf(
    mol: Chem.Mol,
    atom_pos: torch.Tensor,
    pocket_center: torch.Tensor,
    out_path: Path,
    sample_idx: int = 0,
) -> None:
    """Write docked pose as SDF file.

    atom_pos is in pocket-centered coordinates, so we add pocket_center back.
    """
    mol_out = Chem.RWMol(mol)
    conf = mol_out.GetConformer()
    pos = atom_pos + pocket_center  # restore absolute coordinates

    assert pos.shape[0] == mol_out.GetNumAtoms(), (
        f"Atom count mismatch: model predicted {pos.shape[0]}, "
        f"mol has {mol_out.GetNumAtoms()}"
    )

    for i in range(mol_out.GetNumAtoms()):
        conf.SetAtomPosition(i, pos[i].tolist())

    mol_out.SetProp("_Name", f"docked_pose_{sample_idx}")

    writer = Chem.SDWriter(str(out_path))
    writer.write(mol_out)
    writer.close()


def write_multi_sdf(
    mol: Chem.Mol,
    all_poses: list[torch.Tensor],
    pocket_center: torch.Tensor,
    out_path: Path,
) -> None:
    """Write multiple docked poses to a single SDF file."""
    writer = Chem.SDWriter(str(out_path))
    for i, atom_pos in enumerate(all_poses):
        mol_out = Chem.RWMol(mol)
        conf = mol_out.GetConformer()
        pos = atom_pos + pocket_center

        for j in range(mol_out.GetNumAtoms()):
            conf.SetAtomPosition(j, pos[j].tolist())

        mol_out.SetProp("_Name", f"docked_pose_{i}")
        writer.write(mol_out)
    writer.close()


def write_traj_sdf(
    mol: Chem.Mol,
    traj: list[torch.Tensor],
    traj_times: list[float],
    pocket_center: torch.Tensor,
    out_path: Path,
) -> None:
    """Write ODE trajectory as multi-frame SDF (one frame per integration step)."""
    writer = Chem.SDWriter(str(out_path))
    for i, (atom_pos, t) in enumerate(zip(traj, traj_times)):
        mol_out = Chem.RWMol(mol)
        conf = mol_out.GetConformer()
        pos = atom_pos + pocket_center

        for j in range(mol_out.GetNumAtoms()):
            conf.SetAtomPosition(j, pos[j].tolist())

        mol_out.SetProp("_Name", f"frame_{i}")
        mol_out.SetProp("t", f"{t:.4f}")
        mol_out.SetProp("step", str(i))
        writer.write(mol_out)
    writer.close()


def write_traj_pdb(
    mol: Chem.Mol,
    traj: list[torch.Tensor],
    pocket_center: torch.Tensor,
    out_path: Path,
) -> None:
    """Write ODE trajectory as multi-MODEL PDB for visualization in PyMOL/VMD."""
    element_map = {6: "C", 7: "N", 8: "O", 16: "S", 15: "P", 9: "F",
                   17: "Cl", 35: "Br", 53: "I", 5: "B", 14: "Si", 34: "Se"}
    with open(out_path, "w") as f:
        for frame_idx, atom_pos in enumerate(traj):
            pos = atom_pos + pocket_center
            f.write(f"MODEL     {frame_idx + 1:4d}\n")
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                elem = element_map.get(atom.GetAtomicNum(), "X")
                name = f" {elem:<3s}"
                x, y, z = pos[i].tolist()
                f.write(
                    f"HETATM{i+1:5d} {name:4s} LIG A   1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                    f"          {elem:>2s}\n"
                )
            f.write("ENDMDL\n")
        f.write("END\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dock a ligand into a protein pocket using FlowFrag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--protein", type=str, required=True, help="Protein pocket PDB file")
    parser.add_argument("--ligand", type=str, required=True,
                        help="Ligand input: SMILES string, SDF file, or MOL2 file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--num_steps", type=int, default=25, help="ODE integration steps")
    parser.add_argument("--time_schedule", type=str, default="late",
                        choices=("uniform", "late", "early"))
    parser.add_argument("--schedule_power", type=float, default=3.0)
    parser.add_argument("--sigma", type=float, default=None,
                        help="Translation prior sigma (default: from config)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of docking poses to generate")
    parser.add_argument("--save_traj", action="store_true",
                        help="Save ODE trajectory as multi-frame SDF + PDB")
    parser.add_argument("--out_dir", type=str, default="outputs/docked", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    args = parser.parse_args()

    protein_pdb = Path(args.protein)
    assert protein_pdb.exists(), f"Protein PDB not found: {protein_pdb}"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- Load model ---
    model_cfg = dict(cfg["model"])  # copy to avoid mutating config
    model_cfg.pop("model_type", None)

    from src.models.unified import UnifiedFlowFrag
    model = UnifiedFlowFrag(**model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)
    step = ckpt.get("step", "?")
    print(f"Model loaded: {args.checkpoint} (step {step})")

    # --- Load ligand ---
    print(f"Loading ligand: {args.ligand}")
    mol, has_pose = load_ligand(args.ligand)
    print(f"  Atoms: {mol.GetNumAtoms()}, Formula: {rdMolDescriptors.CalcMolFormula(mol)}")
    if not has_pose:
        print("  (SMILES input: coords will be centered at pocket)")

    # --- Preprocess ---
    print("Preprocessing complex...")
    graph, lig_data, meta = preprocess_complex(protein_pdb, mol, ligand_has_pose=has_pose)
    n_frags = meta["num_frag"]
    n_atoms = meta["num_atom"]
    n_prot = graph["num_prot_atom"].item()
    n_ca = graph["num_prot_ca"].item()
    print(f"  Ligand: {n_atoms} atoms, {n_frags} fragments")
    print(f"  Protein pocket: {n_prot} atoms, {n_ca} residues")
    print(f"  Graph: {graph['num_nodes'].item()} nodes, {graph['edge_index'].shape[1]} edges")

    sigma = args.sigma if args.sigma is not None else cfg["data"].get("prior_sigma", 1.0)

    # --- Generate poses ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_poses = []
    all_trajs = []
    print(f"\nGenerating {args.num_samples} pose(s), {args.num_steps} ODE steps "
          f"(schedule={args.time_schedule}, power={args.schedule_power}, sigma={sigma})...")

    for i in range(args.num_samples):
        result = sample_unified(
            model, graph, lig_data, meta,
            num_steps=args.num_steps,
            translation_sigma=sigma,
            time_schedule=args.time_schedule,
            schedule_power=args.schedule_power,
            device=device,
            save_traj=args.save_traj,
        )
        all_poses.append(result["atom_pos_pred"])
        if args.save_traj:
            all_trajs.append(result)

        if args.num_samples > 1:
            print(f"  Sample {i+1}/{args.num_samples} done")

    # --- Write output ---
    pocket_center = meta["pocket_center"]

    if args.num_samples == 1:
        out_path = out_dir / "docked.sdf"
        write_sdf(mol, all_poses[0], pocket_center, out_path)
        print(f"\nDocked pose saved to {out_path}")
    else:
        out_path = out_dir / "docked_poses.sdf"
        write_multi_sdf(mol, all_poses, pocket_center, out_path)
        print(f"\n{args.num_samples} docked poses saved to {out_path}")

    # Write trajectories
    if args.save_traj:
        for i, res in enumerate(all_trajs):
            suffix = f"_{i}" if args.num_samples > 1 else ""
            traj_sdf = out_dir / f"traj{suffix}.sdf"
            traj_pdb = out_dir / f"traj{suffix}.pdb"
            write_traj_sdf(mol, res["traj"], res["traj_times"], pocket_center, traj_sdf)
            write_traj_pdb(mol, res["traj"], pocket_center, traj_pdb)
            n_frames = len(res["traj"])
            print(f"  Trajectory{suffix}: {n_frames} frames -> {traj_sdf}, {traj_pdb}")

    # Also save raw tensors
    save_data = {
        "pocket_center": pocket_center,
        "frag_centers": lig_data["frag_centers"],
        "frag_sizes": lig_data["frag_sizes"],
        "poses": [{"atom_pos_pred": p} for p in all_poses],
    }
    if args.save_traj:
        save_data["trajectories"] = [
            {"traj": [f for f in res["traj"]], "traj_times": res["traj_times"]}
            for res in all_trajs
        ]
    torch.save(save_data, out_dir / "results.pt")
    print(f"Raw tensors saved to {out_dir / 'results.pt'}")


if __name__ == "__main__":
    main()
