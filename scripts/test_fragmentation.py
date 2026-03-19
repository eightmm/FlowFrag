"""Compare current vs merged fragmentation on specific molecules.

Outputs 2D images: current fragments (left) vs merged fragments (right).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess.fragments import decompose_fragments, _get_rotatable_bonds, _assign_fragments

# Distinct fragment colors
FRAG_COLORS = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
    "#42D4F4", "#F032E6", "#BFEF45", "#FABED4", "#469990",
    "#DCBEFF", "#9A6324", "#800000", "#AAFFC3", "#808000",
    "#FFD8B1", "#000075", "#A9A9A9", "#E6BEFF", "#1CE6FF",
]


def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def merge_small_fragments(
    mol: Chem.Mol,
    fragment_id: torch.Tensor,
    rot_bonds: list[tuple[int, int]],
    min_size: int = 1,
) -> torch.Tensor:
    """Merge small fragments in two steps:

    1. Ring-absorb: 1-atom fragments directly bonded to a ring atom
       get absorbed into that ring's fragment (single pass, no cascading).
    2. Chain-merge: adjacent 1-atom fragments are merged together into
       one chain fragment (reduces trivial fragments).
    """
    fid = fragment_id.clone()

    # Step 1: chain-merge adjacent 1-atom fragments (max 3 atoms)
    # 3 atoms = no internal dihedral = rigid. 4+ atoms have internal torsion.
    # This prevents chain atoms (S-C-C) from being split by step 2.
    single_atoms = set()
    for i in range(mol.GetNumAtoms()):
        fi = fid[i].item()
        if (fid == fi).sum().item() == 1:
            single_atoms.add(i)

    # Build adjacency among single-atom fragments
    adj: dict[int, list[int]] = {a: [] for a in single_atoms}
    for bond in mol.GetBonds():
        bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if bi in single_atoms and bj in single_atoms:
            adj[bi].append(bj)
            adj[bj].append(bi)

    # Greedy merge: pick a seed, grow chain up to 3 atoms via BFS
    merged = set()
    for seed in sorted(single_atoms):
        if seed in merged:
            continue
        chain = [seed]
        merged.add(seed)
        # BFS but cap at 3
        queue = [seed]
        while queue and len(chain) < 3:
            cur = queue.pop(0)
            for nb in adj[cur]:
                if nb not in merged and len(chain) < 3:
                    chain.append(nb)
                    merged.add(nb)
                    queue.append(nb)
        # Assign all chain atoms to seed's fragment
        for atom_idx in chain[1:]:
            fid[atom_idx] = fid[chain[0]]

    _, fid = torch.unique(fid, return_inverse=True)

    # Step 2: absorb remaining 1-atom frags into rigid neighbors
    # Criteria: neighbor is in a ring OR has degree >= 3 (branching point)
    # Absorb into SMALLEST qualifying neighbor for balanced fragment sizes
    for i in range(mol.GetNumAtoms()):
        fi = fid[i].item()
        if (fid == fi).sum().item() > min_size:
            continue
        atom = mol.GetAtomWithIdx(i)
        best_frag, best_size = None, float("inf")
        for nb in atom.GetNeighbors():
            if not (nb.IsInRing() or nb.GetDegree() >= 3):
                continue
            fj = fid[nb.GetIdx()].item()
            if fj == fi:
                continue
            nsize = (fid == fj).sum().item()
            if nsize < best_size:
                best_size = nsize
                best_frag = fj
        if best_frag is not None:
            fid[fid == fi] = best_frag

    _, fid = torch.unique(fid, return_inverse=True)
    return fid


def draw_mol_with_fragments(
    mol: Chem.Mol,
    fragment_id: np.ndarray,
    title: str,
    width: int = 600,
    height: int = 500,
) -> Image.Image:
    """Draw 2D molecule with atoms/bonds colored by fragment."""
    AllChem.Compute2DCoords(mol)
    n_atoms = mol.GetNumAtoms()
    n_frags = int(fragment_id.max()) + 1

    atom_colors = {}
    atom_radii = {}
    highlight_atoms = list(range(n_atoms))
    for i in range(n_atoms):
        fid = int(fragment_id[i]) if i < len(fragment_id) else 0
        atom_colors[i] = hex_to_rgb(FRAG_COLORS[fid % len(FRAG_COLORS)])
        atom_radii[i] = 0.4

    bond_colors = {}
    highlight_bonds = []
    for bond in mol.GetBonds():
        bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bid = bond.GetIdx()
        fi = int(fragment_id[bi]) if bi < len(fragment_id) else 0
        fj = int(fragment_id[bj]) if bj < len(fragment_id) else 0
        if fi == fj:
            bond_colors[bid] = hex_to_rgb(FRAG_COLORS[fi % len(FRAG_COLORS)])
        else:
            bond_colors[bid] = (0.3, 0.3, 0.3)  # cut bond = grey
        highlight_bonds.append(bid)

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.5
    opts.addAtomIndices = True
    opts.annotationFontScale = 0.7
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText()))

    # Add title + legend using PIL
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width / 100, height / 100 + 0.8))
    ax.imshow(img)
    ax.set_title(f"{title}  ({n_frags} fragments)", fontsize=12, fontweight="bold")
    ax.axis("off")

    # Legend: colored bold fragment labels
    frag_sizes = [int((fragment_id == f).sum()) for f in range(n_frags)]
    x_start = 0.5 - (n_frags * 0.045)  # rough centering
    for f in range(n_frags):
        color = FRAG_COLORS[f % len(FRAG_COLORS)]
        ax.text(
            x_start + f * 0.09, -0.02,
            f"F{f}({frag_sizes[f]})",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, fontweight="bold", fontfamily="monospace",
            color=color,
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    result = Image.open(buf).copy()
    plt.close(fig)
    return result


def process_molecule(sdf_path: Path, pdb_id: str, out_dir: Path) -> None:
    """Generate before/after fragmentation comparison."""
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=True)
    assert mol is not None, f"Failed to load {sdf_path}"

    n_atoms = mol.GetNumAtoms()
    # Dummy coords (only need for centroid calc, not for 2D drawing)
    coords = torch.zeros(n_atoms, 3)
    conf = mol.GetConformer()
    for i in range(n_atoms):
        pos = conf.GetAtomPosition(i)
        coords[i] = torch.tensor([pos.x, pos.y, pos.z])

    # Current decomposition
    result = decompose_fragments(mol, coords)
    assert result is not None
    fid_current = result["fragment_id"].numpy()

    # Merged decomposition
    rot_bonds = _get_rotatable_bonds(mol)
    fid_raw = _assign_fragments(mol, rot_bonds, n_atoms)
    _, fid_raw = torch.unique(fid_raw, return_inverse=True)
    fid_merged = merge_small_fragments(mol, fid_raw, rot_bonds, min_size=1).numpy()

    print(f"\n{pdb_id}: {n_atoms} atoms")
    print(f"  Current: {int(fid_current.max()) + 1} fragments, sizes={np.bincount(fid_current).tolist()}")
    print(f"  Merged:  {int(fid_merged.max()) + 1} fragments, sizes={np.bincount(fid_merged).tolist()}")
    print(f"  Cut bonds: {rot_bonds}")

    # Draw comparison
    img_current = draw_mol_with_fragments(mol, fid_current, f"{pdb_id} - Current")
    img_merged = draw_mol_with_fragments(mol, fid_merged, f"{pdb_id} - Merged (ring+chain)")

    # Side by side
    total_w = img_current.width + img_merged.width + 20
    total_h = max(img_current.height, img_merged.height)
    combined = Image.new("RGB", (total_w, total_h), "white")
    combined.paste(img_current, (0, 0))
    combined.paste(img_merged, (img_current.width + 20, 0))

    out_path = out_dir / f"frag_compare_{pdb_id}.png"
    combined.save(out_path)
    print(f"  Saved: {out_path}")


def main() -> None:
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_ids", nargs="+", default=None, help="Specific PDB IDs")
    parser.add_argument("--raw_dir", default="/mnt/data/PLI/P-L", help="Raw data root")
    parser.add_argument("--out_dir", default="outputs/frag_compare")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pdb_ids is None:
        args.pdb_ids = ["10gs", "16pk", "2r2b", "1if7", "3a1t", "2oz5", "1lqf",
                        "4ih7", "5ta2", "3uxk"]

    for pdb_id in args.pdb_ids:
        hits = glob.glob(f"{args.raw_dir}/*/{pdb_id}/{pdb_id}_ligand.sdf")
        if not hits:
            print(f"  {pdb_id}: SDF not found, skipping")
            continue
        process_molecule(Path(hits[0]), pdb_id, out_dir)


if __name__ == "__main__":
    main()
