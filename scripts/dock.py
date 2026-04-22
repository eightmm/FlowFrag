#!/usr/bin/env python
"""Dock a ligand into a protein pocket using a trained FlowFrag model.

Mirrors the training data pipeline:
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
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import crop_to_pocket
from src.geometry.flow_matching import integrate_se3_step, sample_prior_poses
from src.geometry.se3 import quaternion_to_matrix
from src.preprocess.fragments import decompose_fragments
from src.preprocess.graph import build_static_complex_graph
from src.preprocess.ligand import featurize_ligand, load_molecule
from src.preprocess.protein import parse_pocket_atoms


# ---------------------------------------------------------------------------
# Ligand loading
# ---------------------------------------------------------------------------

def load_ligand(ligand_input: str) -> tuple[Chem.Mol, bool]:
    """Load ligand from SMILES / SDF / MOL2. Returns (mol, has_pose)."""
    path = Path(ligand_input)

    if path.suffix.lower() == ".sdf" and path.exists():
        mol, _, _ = load_molecule(path)
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

    # SMILES → ETKDG conformer
    mol = Chem.MolFromSmiles(ligand_input)
    assert mol is not None, f"Invalid SMILES: {ligand_input}"
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    assert status == 0, f"3D embedding failed: {ligand_input}"
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    mol = Chem.RemoveHs(mol)
    return mol, False


# ---------------------------------------------------------------------------
# Pocket center
# ---------------------------------------------------------------------------

def derive_pocket_center(
    prot_data: dict[str, torch.Tensor],
    ligand_coords: torch.Tensor,
    cutoff: float = 8.0,
) -> torch.Tensor:
    """Pocket center = centroid of residue virtual nodes within cutoff of ligand.

    Matches build_fragment_flow_dataset.py.
    """
    pres = prot_data["pres_coords"]
    dmat = torch.cdist(pres, ligand_coords)
    mask = dmat.min(dim=1).values <= cutoff
    if mask.any():
        return pres[mask].mean(dim=0)
    return ligand_coords.mean(dim=0)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_complex(
    protein_pdb: Path,
    mol: Chem.Mol,
    pocket_center: torch.Tensor | None = None,
    pocket_cutoff: float = 8.0,
) -> tuple[dict[str, torch.Tensor], dict, dict]:
    """Parse protein + ligand, crop pocket, build unified graph.

    Args:
        protein_pdb: Full protein or pocket PDB.
        mol: RDKit mol with 3D conformer.
        pocket_center: [3] reference. If None, derived from ligand coords.
        pocket_cutoff: Residue-aware cutoff (Å).
    """
    lig_data = featurize_ligand(mol)
    assert lig_data is not None, "Ligand featurization failed"

    frag_data = decompose_fragments(mol, lig_data["atom_coords"])
    assert frag_data is not None, "Fragment decomposition failed"

    for k in ("fragment_id", "frag_centers", "frag_local_coords", "frag_sizes",
              "tri_edge_index", "tri_edge_ref_dist", "fragment_adj_index", "cut_bond_index"):
        lig_data[k] = frag_data[k]

    # Full protein parse (no cutoff)
    prot_data = parse_pocket_atoms(protein_pdb)
    assert prot_data is not None, f"Protein parsing failed: {protein_pdb}"

    if pocket_center is None:
        pocket_center = derive_pocket_center(prot_data, lig_data["atom_coords"], cutoff=pocket_cutoff)
    else:
        pocket_center = pocket_center.to(torch.float32)

    cropped = crop_to_pocket(prot_data, pocket_center, cutoff=pocket_cutoff)
    assert cropped is not None, (
        f"No protein residues within {pocket_cutoff}Å of pocket_center={pocket_center.tolist()}"
    )

    graph = build_static_complex_graph(lig_data, cropped)

    meta = {
        "pocket_center": pocket_center,
        "num_frag": frag_data["n_frags"],
        "num_atom": lig_data["atom_coords"].shape[0],
    }
    return graph, lig_data, meta


# ---------------------------------------------------------------------------
# ODE sampler
# ---------------------------------------------------------------------------

def sample_unified(
    model: torch.nn.Module,
    graph: dict[str, torch.Tensor],
    lig_data: dict,
    meta: dict,
    *,
    num_steps: int = 25,
    translation_sigma: float = 5.0,
    time_schedule: str = "late",
    schedule_power: float = 3.0,
    device: torch.device = torch.device("cpu"),
    save_traj: bool = False,
    phys_guidance=None,
    phys_lambda_max: float = 0.0,
    phys_power: float = 2.0,
    phys_start_t: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Run ODE integration for a single complex."""
    from src.inference.sampler import build_time_grid

    n_frags = meta["num_frag"]
    pocket_center = meta["pocket_center"]
    frag_sizes = lig_data["frag_sizes"]
    frag_id = lig_data["fragment_id"]
    local_pos = lig_data["frag_local_coords"]

    # Prior in pocket-centered frame
    T, q = sample_prior_poses(
        n_frags, pocket_center=torch.zeros(3),
        translation_sigma=translation_sigma, frag_sizes=frag_sizes,
    )

    time_grid = build_time_grid(
        num_steps, schedule=time_schedule, power=schedule_power,
        device=device, dtype=torch.float32,
    )

    batch: dict[str, torch.Tensor] = {}
    for k, v in graph.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    n_nodes = batch["node_coords"].shape[0]
    batch["batch"] = torch.zeros(n_nodes, dtype=torch.long, device=device)
    batch["frag_batch"] = torch.zeros(n_frags, dtype=torch.long, device=device)

    # Center node coords on pocket_center (matches training frame)
    batch["node_coords"] = batch["node_coords"] - pocket_center.to(device)

    frag_slice = graph["lig_frag_slice"]
    frag_start, frag_end = frag_slice[0].item(), frag_slice[1].item()
    atom_slice = graph["lig_atom_slice"]
    atom_start = atom_slice[0].item()
    n_real_atoms = local_pos.shape[0]

    T, q = T.to(device), q.to(device)
    frag_sizes_d = frag_sizes.to(device)
    local_pos_d = local_pos.to(device)
    frag_id_d = frag_id.to(device)

    traj: list[torch.Tensor] = []
    traj_times: list[float] = []

    for step_idx in range(num_steps):
        t = time_grid[step_idx]
        dt = time_grid[step_idx + 1] - time_grid[step_idx]

        node_coords = batch["node_coords"].clone()
        node_coords[frag_start:frag_end] = T
        R = quaternion_to_matrix(q)
        atom_pos_t = torch.einsum("nij,nj->ni", R[frag_id_d], local_pos_d) + T[frag_id_d]
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

        with torch.no_grad():
            out = model(batch)

        v_use = out["v_pred"]
        omega_use = out["omega_pred"]

        if (
            phys_guidance is not None
            and phys_lambda_max > 0.0
            and t.item() >= phys_start_t
        ):
            lam = phys_lambda_max * (t.item() ** phys_power)
            v_phys, omega_phys = phys_guidance.compute_drift(
                atom_pos_t=atom_pos_t,
                T_frag=T,
                frag_id=frag_id_d,
                frag_sizes=frag_sizes_d,
            )
            v_use = v_use + lam * v_phys
            omega_use = omega_use + lam * omega_phys

        T, q = integrate_se3_step(
            T, q, v_use, omega_use, dt, frag_sizes=frag_sizes_d,
        )

    R_final = quaternion_to_matrix(q)
    atom_pos_pred = torch.einsum("nij,nj->ni", R_final[frag_id_d], local_pos_d) + T[frag_id_d]

    if save_traj:
        traj.append(atom_pos_pred.cpu())
        traj_times.append(1.0)

    result = {"T_pred": T.cpu(), "q_pred": q.cpu(), "atom_pos_pred": atom_pos_pred.cpu()}
    if save_traj:
        result["traj"] = traj
        result["traj_times"] = traj_times
    return result


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _set_coords(mol: Chem.Mol, pos: torch.Tensor) -> Chem.Mol:
    mol_out = Chem.RWMol(mol)
    conf = mol_out.GetConformer()
    assert pos.shape[0] == mol_out.GetNumAtoms()
    for i in range(mol_out.GetNumAtoms()):
        conf.SetAtomPosition(i, pos[i].tolist())
    return mol_out


def write_sdf(mol: Chem.Mol, atom_pos: torch.Tensor, pocket_center: torch.Tensor, out_path: Path) -> None:
    mol_out = _set_coords(mol, atom_pos + pocket_center)
    mol_out.SetProp("_Name", "docked_pose")
    writer = Chem.SDWriter(str(out_path))
    writer.write(mol_out)
    writer.close()


def write_multi_sdf(mol: Chem.Mol, all_poses: list[torch.Tensor], pocket_center: torch.Tensor, out_path: Path) -> None:
    writer = Chem.SDWriter(str(out_path))
    for i, atom_pos in enumerate(all_poses):
        mol_out = _set_coords(mol, atom_pos + pocket_center)
        mol_out.SetProp("_Name", f"docked_pose_{i}")
        writer.write(mol_out)
    writer.close()


def write_traj_sdf(
    mol: Chem.Mol, traj: list[torch.Tensor], traj_times: list[float],
    pocket_center: torch.Tensor, out_path: Path,
) -> None:
    writer = Chem.SDWriter(str(out_path))
    for i, (atom_pos, t) in enumerate(zip(traj, traj_times)):
        mol_out = _set_coords(mol, atom_pos + pocket_center)
        mol_out.SetProp("_Name", f"frame_{i}")
        mol_out.SetProp("t", f"{t:.4f}")
        writer.write(mol_out)
    writer.close()


def write_traj_pdb(
    mol: Chem.Mol, traj: list[torch.Tensor], pocket_center: torch.Tensor, out_path: Path,
) -> None:
    element_map = {6: "C", 7: "N", 8: "O", 16: "S", 15: "P", 9: "F",
                   17: "Cl", 35: "Br", 53: "I", 5: "B", 14: "Si", 34: "Se"}
    with open(out_path, "w") as f:
        for frame_idx, atom_pos in enumerate(traj):
            pos = atom_pos + pocket_center
            f.write(f"MODEL     {frame_idx + 1:4d}\n")
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                elem = element_map.get(atom.GetAtomicNum(), "X")
                x, y, z = pos[i].tolist()
                f.write(
                    f"HETATM{i+1:5d}  {elem:<3s} LIG A   1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n"
                )
            f.write("ENDMDL\n")
        f.write("END\n")


# ---------------------------------------------------------------------------
# Main
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
    parser.add_argument("--time_schedule", type=str, default="late", choices=("uniform", "late", "early"))
    parser.add_argument("--schedule_power", type=float, default=3.0)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--save_traj", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/docked")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--phys_guidance", action="store_true",
                        help="Enable Vina-gradient guidance during ODE sampling.")
    parser.add_argument("--phys_lambda_max", type=float, default=0.3,
                        help="Max guidance weight lambda_max (0 = off).")
    parser.add_argument("--phys_power", type=float, default=2.0,
                        help="lambda(t) = lambda_max * t^power.")
    parser.add_argument("--phys_start_t", type=float, default=0.3,
                        help="Disable guidance for t < phys_start_t.")
    parser.add_argument("--phys_max_force", type=float, default=10.0,
                        help="Per-atom force norm clip.")
    parser.add_argument("--phys_weight_preset", type=str, default="vina",
                        choices=("vina", "vinardo"))
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
    print(f"  Atoms: {mol.GetNumAtoms()}, Formula: {rdMolDescriptors.CalcMolFormula(mol)}, has_pose={has_pose}")

    # Pocket center
    pocket_center = _parse_center(args.pocket_center)
    if pocket_center is None and not has_pose:
        raise ValueError(
            "SMILES input requires explicit --pocket_center x,y,z "
            "(cannot derive from ligand without a pose)."
        )

    # Preprocess
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
            pocket_cutoff=args.pocket_cutoff,
            weight_preset=args.phys_weight_preset,
            max_force_per_atom=args.phys_max_force,
        )
        print(
            f"Physics guidance ON: lambda_max={args.phys_lambda_max}, "
            f"power={args.phys_power}, start_t={args.phys_start_t}, "
            f"preset={args.phys_weight_preset}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_poses = []
    all_trajs = []
    print(f"\nGenerating {args.num_samples} pose(s), {args.num_steps} ODE steps, sigma={sigma}...")

    for i in range(args.num_samples):
        result = sample_unified(
            model, graph, lig_data, meta,
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
        all_poses.append(result["atom_pos_pred"])
        if args.save_traj:
            all_trajs.append(result)
        if args.num_samples > 1:
            print(f"  Sample {i+1}/{args.num_samples} done")

    pc = meta["pocket_center"]

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
            write_traj_sdf(mol, res["traj"], res["traj_times"], pc, out_dir / f"traj{suffix}.sdf")
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
    }, out_dir / "results.pt")
    print(f"Raw tensors saved to {out_dir / 'results.pt'}")


if __name__ == "__main__":
    main()
