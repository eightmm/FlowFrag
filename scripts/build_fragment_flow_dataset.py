#!/usr/bin/env python
"""Build processed dataset from raw PDBbind complexes.

Usage:
    python scripts/build_fragment_flow_dataset.py \
        --raw_dir /mnt/data/PLI/P-L \
        --out_dir data/processed \
        --workers 8
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from src.preprocess.fragments import decompose_fragments, add_dummy_atoms
from src.preprocess.graph import build_static_complex_graph
from src.preprocess.ligand import featurize_ligand, load_molecule
from src.preprocess.protein import parse_pocket_atoms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def process_complex(complex_dir: Path, out_dir: Path, dummy: bool = False) -> dict:
    """Process a single protein-ligand complex.

    Returns a status dict for logging.
    """
    pdb_id = complex_dir.name
    status = {"pdb_id": pdb_id, "success": False, "reason": ""}

    # Find files
    pocket_pdb = complex_dir / f"{pdb_id}_pocket.pdb"
    ligand_sdf = complex_dir / f"{pdb_id}_ligand.sdf"
    ligand_mol2 = complex_dir / f"{pdb_id}_ligand.mol2"

    if not pocket_pdb.exists():
        status["reason"] = "missing_pocket_pdb"
        return status
    if not ligand_sdf.exists():
        status["reason"] = "missing_ligand_sdf"
        return status

    # Parse ligand
    mol2_path = ligand_mol2 if ligand_mol2.exists() else None
    mol, used_mol2 = load_molecule(ligand_sdf, mol2_path)
    if mol is None:
        status["reason"] = "ligand_parse_failed"
        return status

    # Featurize ligand
    lig_data = featurize_ligand(mol)
    if lig_data is None:
        status["reason"] = "ligand_featurize_failed"
        return status

    # Fragment decomposition
    frag_data = decompose_fragments(mol, lig_data["atom_coords"])
    if frag_data is None:
        status["reason"] = "fragment_decomposition_failed"
        return status

    # Optionally add dummy atoms at cut-bond boundaries
    if dummy and frag_data.get("rot_bonds"):
        frag_data = add_dummy_atoms(
            frag_data,
            lig_data["atom_coords"],
            lig_data,
            frag_data["rot_bonds"],
        )
        # Update lig_data with extended atom arrays from frag_data
        for key in (
            "atom_coords",
            "atom_element",
            "atom_charge",
            "atom_aromatic",
            "atom_hybridization",
            "atom_in_ring",
            "atom_degree",
            "atom_implicit_valence",
            "atom_explicit_valence",
            "atom_num_rings",
            "atom_chirality",
            "atom_is_donor",
            "atom_is_acceptor",
            "atom_is_positive",
            "atom_is_negative",
            "atom_is_hydrophobe",
            "atom_is_halogen",
        ):
            if key in frag_data:
                lig_data[key] = frag_data[key]

    # Parse protein heavy atoms (for atom-level pocket representation)
    patom_data = parse_pocket_atoms(
        pocket_pdb,
        ligand_coords=lig_data["atom_coords"],
        cutoff=8.0,
    )
    if patom_data is None:
        status["reason"] = "protein_atom_parse_failed"
        return status

    # Compute pocket center from CA virtual nodes.
    pocket_center = patom_data["pca_coords"].mean(dim=0)

    n_atoms = lig_data["atom_coords"].shape[0]

    # Build meta
    meta = {
        "pdb_id": pdb_id,
        "pocket_center": pocket_center,
        "num_res": torch.tensor(patom_data["pca_coords"].shape[0], dtype=torch.int64),
        "num_atom": torch.tensor(n_atoms, dtype=torch.int64),
        "num_frag": torch.tensor(frag_data["n_frags"], dtype=torch.int64),
        "num_prot_atom": torch.tensor(patom_data["patom_coords"].shape[0], dtype=torch.int64),
        "used_mol2_fallback": torch.tensor(used_mol2, dtype=torch.bool),
        "has_dummy_atoms": torch.tensor(dummy, dtype=torch.bool),
        "schema_version": SCHEMA_VERSION,
    }

    # Add fragment data to ligand dict
    lig_data["fragment_id"] = frag_data["fragment_id"]
    lig_data["frag_centers"] = frag_data["frag_centers"]
    lig_data["frag_local_coords"] = frag_data["frag_local_coords"]
    lig_data["frag_sizes"] = frag_data["frag_sizes"]
    lig_data["tri_edge_index"] = frag_data["tri_edge_index"]
    lig_data["tri_edge_ref_dist"] = frag_data["tri_edge_ref_dist"]
    lig_data["fragment_adj_index"] = frag_data["fragment_adj_index"]
    lig_data["cut_bond_index"] = frag_data["cut_bond_index"]
    if "is_dummy" in frag_data:
        lig_data["is_dummy"] = frag_data["is_dummy"]
        lig_data["dummy_to_real"] = frag_data["dummy_to_real"]

    static_graph = build_static_complex_graph(lig_data, patom_data)

    # Save
    complex_out = out_dir / pdb_id
    complex_out.mkdir(parents=True, exist_ok=True)
    torch.save(lig_data, complex_out / "ligand.pt")
    torch.save(static_graph, complex_out / "graph.pt")
    torch.save(meta, complex_out / "meta.pt")
    torch.save(patom_data, complex_out / "protein_atoms.pt")

    status["success"] = True
    status["num_res"] = patom_data["pca_coords"].shape[0]
    status["num_atom"] = lig_data["atom_coords"].shape[0]
    status["num_frag"] = frag_data["n_frags"]
    status["single_atom_frags"] = int((frag_data["frag_sizes"] == 1).sum())
    return status


def find_complexes(raw_dir: Path) -> list[Path]:
    """Find all complex directories under raw_dir."""
    complexes = []
    for year_dir in sorted(raw_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for complex_dir in sorted(year_dir.iterdir()):
            if not complex_dir.is_dir():
                continue
            complexes.append(complex_dir)
    return complexes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FlowFrag dataset")
    parser.add_argument("--raw_dir", type=Path, default=Path("/mnt/data/PLI/P-L"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--dummy", action="store_true", help="Add dummy atoms at cut-bond boundaries"
    )
    args = parser.parse_args()

    assert args.raw_dir.exists(), f"Raw data dir not found: {args.raw_dir}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    complexes = find_complexes(args.raw_dir)
    log.info("Found %d complexes", len(complexes))

    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_complex, c, args.out_dir, dummy=args.dummy): c for c in complexes
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "pdb_id": futures[future].name,
                    "success": False,
                    "reason": f"exception: {e}",
                }
            results.append(result)

    # Summary statistics
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    log.info("=" * 60)
    log.info("Processing complete")
    log.info(
        "  Success: %d / %d (%.1f%%)",
        len(successes),
        len(results),
        100 * len(successes) / max(len(results), 1),
    )
    log.info("  Failed:  %d", len(failures))

    if successes:
        n_res = [r["num_res"] for r in successes]
        n_atom = [r["num_atom"] for r in successes]
        n_frag = [r["num_frag"] for r in successes]
        n_single = [r["single_atom_frags"] for r in successes]
        log.info(
            "  Residues:  min=%d, median=%d, max=%d",
            min(n_res),
            sorted(n_res)[len(n_res) // 2],
            max(n_res),
        )
        log.info(
            "  Atoms:     min=%d, median=%d, max=%d",
            min(n_atom),
            sorted(n_atom)[len(n_atom) // 2],
            max(n_atom),
        )
        log.info(
            "  Fragments: min=%d, median=%d, max=%d",
            min(n_frag),
            sorted(n_frag)[len(n_frag) // 2],
            max(n_frag),
        )
        log.info(
            "  Single-atom frags: min=%d, median=%d, max=%d",
            min(n_single),
            sorted(n_single)[len(n_single) // 2],
            max(n_single),
        )

    # Failure breakdown
    if failures:
        reason_counts: dict[str, int] = {}
        for r in failures:
            reason = r.get("reason", "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        log.info("  Failure reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            log.info("    %s: %d", reason, count)

    # Save manifest
    manifest_path = args.out_dir / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "total": len(results),
        "success": len(successes),
        "failed": len(failures),
        "pdb_ids": sorted([r["pdb_id"] for r in successes]),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest saved to %s", manifest_path)


if __name__ == "__main__":
    main()
