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

from src.preprocess.fragments import decompose_fragments
from src.preprocess.ligand import compute_dg_bounds, featurize_ligand, load_molecule
from src.preprocess.protein import parse_pocket_atoms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_COVALENT_FILE = Path(__file__).resolve().parent.parent / "data" / "covalent_binders_v2020.txt"
COVALENT_IDS: set[str] = set()
if _COVALENT_FILE.exists():
    COVALENT_IDS = {
        tok.upper()
        for tok in _COVALENT_FILE.read_text().split()
        if tok.strip()
    }


def process_complex(complex_dir: Path, out_dir: Path) -> dict:
    """Process a single protein-ligand complex.

    Returns a status dict for logging.
    """
    pdb_id = complex_dir.name
    status = {"pdb_id": pdb_id, "success": False, "reason": ""}

    if pdb_id.upper() in COVALENT_IDS:
        status["reason"] = "covalent_binder"
        return status

    # Find files: prefer full protein, fall back to pocket-only
    protein_pdb = complex_dir / f"{pdb_id}_protein.pdb"
    pocket_pdb = complex_dir / f"{pdb_id}_pocket.pdb"
    ligand_sdf = complex_dir / f"{pdb_id}_ligand.sdf"
    ligand_mol2 = complex_dir / f"{pdb_id}_ligand.mol2"

    pdb_path = protein_pdb if protein_pdb.exists() else pocket_pdb
    if not pdb_path.exists():
        status["reason"] = "missing_protein_pdb"
        return status
    if not ligand_sdf.exists() and not ligand_mol2.exists():
        status["reason"] = "missing_ligand"
        return status

    # --- Ligand -----------------------------------------------------------
    sdf_arg = ligand_sdf if ligand_sdf.exists() else None
    mol2_arg = ligand_mol2 if ligand_mol2.exists() else None
    mol, used_mol2, sanitize_ok = load_molecule(sdf_arg, mol2_arg)
    if mol is None:
        status["reason"] = "ligand_parse_failed"
        return status

    lig_data = featurize_ligand(mol)
    if lig_data is None:
        status["reason"] = "ligand_featurize_failed"
        return status

    frag_data = decompose_fragments(mol, lig_data["atom_coords"])
    if frag_data is None:
        status["reason"] = "fragment_decomposition_failed"
        return status

    # DG bounds matrix (raw; margin applied at training time)
    dg_bounds = compute_dg_bounds(mol)
    if dg_bounds is not None:
        lig_data["dg_bounds"] = dg_bounds

    # Merge fragment data into ligand dict
    lig_data["fragment_id"] = frag_data["fragment_id"]
    lig_data["frag_centers"] = frag_data["frag_centers"]
    lig_data["frag_local_coords"] = frag_data["frag_local_coords"]
    lig_data["frag_sizes"] = frag_data["frag_sizes"]
    lig_data["tri_edge_index"] = frag_data["tri_edge_index"]
    lig_data["tri_edge_ref_dist"] = frag_data["tri_edge_ref_dist"]
    lig_data["fragment_adj_index"] = frag_data["fragment_adj_index"]
    lig_data["cut_bond_index"] = frag_data["cut_bond_index"]

    # --- Protein (full, no cutoff) ----------------------------------------
    prot_data = parse_pocket_atoms(pdb_path)
    if prot_data is None:
        status["reason"] = "protein_parse_failed"
        return status

    # Pocket center: centroid of pocket residue virtual nodes (CB/pseudo-CB)
    # within 8Å of crystal ligand. Protein-derived, consistent with what a
    # pocket predictor would output at inference time.
    pres_coords = prot_data["pres_coords"]
    pres_to_lig = torch.cdist(pres_coords, lig_data["atom_coords"])
    pocket_mask = pres_to_lig.min(dim=1).values <= 8.0
    if pocket_mask.any():
        pocket_center = pres_coords[pocket_mask].mean(dim=0)
    else:
        pocket_center = lig_data["atom_coords"].mean(dim=0)
    n_pocket_res = int(pocket_mask.sum().item())

    # --- Meta -------------------------------------------------------------
    n_atoms = lig_data["atom_coords"].shape[0]
    meta = {
        "pdb_id": pdb_id,
        "pocket_center": pocket_center,
        "num_pocket_res": torch.tensor(n_pocket_res, dtype=torch.int64),
        "used_full_protein": protein_pdb.exists(),
        "num_res": torch.tensor(prot_data["pres_coords"].shape[0], dtype=torch.int64),
        "num_atom": torch.tensor(n_atoms, dtype=torch.int64),
        "num_frag": torch.tensor(frag_data["n_frags"], dtype=torch.int64),
        "num_prot_atom": torch.tensor(prot_data["patom_coords"].shape[0], dtype=torch.int64),
        "used_mol2_fallback": torch.tensor(used_mol2, dtype=torch.bool),
        "ligand_sanitize_ok": torch.tensor(sanitize_ok, dtype=torch.bool),
        "schema_version": SCHEMA_VERSION,
    }

    # --- Save (protein + ligand separately; graph built at runtime) -------
    complex_out = out_dir / pdb_id
    complex_out.mkdir(parents=True, exist_ok=True)
    torch.save(prot_data, complex_out / "protein.pt")
    torch.save(lig_data, complex_out / "ligand.pt")
    torch.save(meta, complex_out / "meta.pt")

    status["success"] = True
    status["num_res"] = prot_data["pres_coords"].shape[0]
    status["num_prot_atom"] = prot_data["patom_coords"].shape[0]
    status["num_atom"] = n_atoms
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
    args = parser.parse_args()

    assert args.raw_dir.exists(), f"Raw data dir not found: {args.raw_dir}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    complexes = find_complexes(args.raw_dir)
    log.info("Found %d complexes", len(complexes))

    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_complex, c, args.out_dir): c for c in complexes
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

        def _stats(vals: list[int]) -> tuple[int, int, int]:
            s = sorted(vals)
            return s[0], s[len(s) // 2], s[-1]

        for label, key in [
            ("Prot residues", "num_res"),
            ("Prot atoms", "num_prot_atom"),
            ("Lig atoms", "num_atom"),
            ("Fragments", "num_frag"),
            ("Single-atom frags", "single_atom_frags"),
        ]:
            vals = [r[key] for r in successes]
            lo, med, hi = _stats(vals)
            log.info("  %-18s min=%d, median=%d, max=%d", label, lo, med, hi)

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
