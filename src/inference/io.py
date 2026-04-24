"""Pose serialization helpers (SDF / PDB writers).

Reusable output formatters for docking CLIs and benchmarks.  Poses are given
in the *pocket-centered* frame (same frame used by the sampler); callers
pass the original pocket center so coordinates are written back in the
global frame.
"""
from __future__ import annotations

from pathlib import Path

import torch
from rdkit import Chem


def _set_coords(mol: Chem.Mol, pos: torch.Tensor) -> Chem.Mol:
    """Return a copy of ``mol`` with atom positions replaced by ``pos``."""
    mol_out = Chem.RWMol(mol)
    conf = mol_out.GetConformer()
    assert pos.shape[0] == mol_out.GetNumAtoms()
    for i in range(mol_out.GetNumAtoms()):
        conf.SetAtomPosition(i, pos[i].tolist())
    return mol_out


def write_sdf(
    mol: Chem.Mol, atom_pos: torch.Tensor,
    pocket_center: torch.Tensor, out_path: Path,
) -> None:
    mol_out = _set_coords(mol, atom_pos + pocket_center)
    mol_out.SetProp("_Name", "docked_pose")
    writer = Chem.SDWriter(str(out_path))
    writer.write(mol_out)
    writer.close()


def write_multi_sdf(
    mol: Chem.Mol, all_poses: list[torch.Tensor],
    pocket_center: torch.Tensor, out_path: Path,
) -> None:
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


_ELEMENT_SYMBOL = {
    6: "C", 7: "N", 8: "O", 16: "S", 15: "P", 9: "F",
    17: "Cl", 35: "Br", 53: "I", 5: "B", 14: "Si", 34: "Se",
}


def write_traj_pdb(
    mol: Chem.Mol, traj: list[torch.Tensor],
    pocket_center: torch.Tensor, out_path: Path,
) -> None:
    """Emit multi-model PDB for a docking trajectory (HETATM records per frame)."""
    with open(out_path, "w") as f:
        for frame_idx, atom_pos in enumerate(traj):
            pos = atom_pos + pocket_center
            f.write(f"MODEL     {frame_idx + 1:4d}\n")
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                elem = _ELEMENT_SYMBOL.get(atom.GetAtomicNum(), "X")
                x, y, z = pos[i].tolist()
                f.write(
                    f"HETATM{i+1:5d}  {elem:<3s} LIG A   1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n"
                )
            f.write("ENDMDL\n")
        f.write("END\n")


__all__ = ["write_sdf", "write_multi_sdf", "write_traj_sdf", "write_traj_pdb"]
