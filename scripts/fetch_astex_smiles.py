#!/usr/bin/env python
"""Fetch canonical SMILES for each Astex Diverse Set ligand from RCSB.

For each PDB ID in the Astex dir, query RCSB REST to enumerate non-polymer
entities, pick the one whose heavy-atom count best matches the local mol2
file, and cache the SMILES to a JSON file.

Usage:
    python scripts/fetch_astex_smiles.py \
        --astex_dir /mnt/data/PLI/Astex-diverse-set \
        --out data/astex_smiles.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


RCSB_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{pdb}"
RCSB_NONPOLY = "https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb}/{eid}"
RCSB_CHEMCOMP = "https://data.rcsb.org/rest/v1/core/chemcomp/{cid}"

# Cofactors/buffers/ions typically not the ligand of interest
BLACKLIST = {
    "HOH", "WAT", "DOD", "NA", "K", "CA", "MG", "ZN", "FE", "MN", "CU", "CO",
    "NI", "CD", "CL", "BR", "I", "F", "SO4", "PO4", "NO3", "ACE", "NH2",
    "EDO", "GOL", "PEG", "PG4", "DMS", "SCN", "EPE", "MES", "TRS", "FMT",
    "ACT", "BME", "MPD", "IPA", "DTT", "GSH", "ATP", "ADP", "AMP", "GTP",
    "NAD", "NAP", "NDP", "FAD", "FMN", "COA", "SAM", "SAH", "HEM", "HEC",
    "CLA", "BCL", "CHL",
}


def fetch_json(url: str, retries: int = 3, timeout: float = 10.0) -> dict | None:
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:
            time.sleep(0.5)
    return None


def count_mol2_heavy_atoms(mol2_path: Path) -> int | None:
    txt = mol2_path.read_text()
    m = re.search(r"@<TRIPOS>MOLECULE\s*\n.*?\n\s*(\d+)\s+(\d+)", txt, re.DOTALL)
    if not m:
        return None
    # MOL2 atom count includes H; we want heavy. Parse ATOM section.
    atom_section = re.search(
        r"@<TRIPOS>ATOM\n(.*?)(?:@<TRIPOS>|\Z)", txt, re.DOTALL
    )
    if not atom_section:
        return None
    heavy = 0
    for line in atom_section.group(1).splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        atom_type = parts[5]
        element = atom_type.split(".")[0]
        if element != "H":
            heavy += 1
    return heavy


def chem_comp_heavy_count(smiles: str) -> int:
    # Simple element count; excludes H and brackets.
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 0
    return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() != 1)


def pick_best_ligand(pdb_id: str, target_heavy: int, log=print) -> tuple[str, str] | None:
    entry = fetch_json(RCSB_ENTRY.format(pdb=pdb_id))
    if entry is None:
        return None
    eids = entry.get("rcsb_entry_container_identifiers", {}).get(
        "non_polymer_entity_ids", []
    )
    if not eids:
        return None

    candidates: list[tuple[int, str, str]] = []  # (heavy_diff, comp_id, smiles)
    for eid in eids:
        np_ent = fetch_json(RCSB_NONPOLY.format(pdb=pdb_id, eid=eid))
        if np_ent is None:
            continue
        comp_id = np_ent.get("pdbx_entity_nonpoly", {}).get("comp_id")
        if comp_id is None or comp_id.upper() in BLACKLIST:
            continue
        cc = fetch_json(RCSB_CHEMCOMP.format(cid=comp_id))
        if cc is None:
            continue
        desc = cc.get("rcsb_chem_comp_descriptor", {}) or {}
        smiles = desc.get("SMILES_stereo") or desc.get("SMILES")
        if not smiles:
            continue
        heavy = chem_comp_heavy_count(smiles)
        candidates.append((abs(heavy - target_heavy), comp_id, smiles))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1], candidates[0][2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--astex_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    astex_dir = Path(args.astex_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())

    pdb_ids = sorted(d.name for d in astex_dir.iterdir() if d.is_dir())
    print(f"Fetching SMILES for {len(pdb_ids)} Astex complexes", file=sys.stderr)

    results = dict(existing)
    for i, pdb_id in enumerate(pdb_ids):
        if pdb_id in results and results[pdb_id].get("smiles"):
            print(f"[{i+1:3d}/{len(pdb_ids)}] {pdb_id}: cached", file=sys.stderr)
            continue

        mol2 = astex_dir / pdb_id / f"{pdb_id}_ligand.mol2"
        if not mol2.exists():
            print(f"[{i+1:3d}/{len(pdb_ids)}] {pdb_id}: no mol2", file=sys.stderr)
            results[pdb_id] = {"error": "no mol2"}
            continue

        heavy = count_mol2_heavy_atoms(mol2) or 0
        picked = pick_best_ligand(pdb_id, heavy)
        if picked is None:
            print(f"[{i+1:3d}/{len(pdb_ids)}] {pdb_id}: no ligand found (heavy={heavy})",
                  file=sys.stderr)
            results[pdb_id] = {"error": "no ligand", "mol2_heavy": heavy}
        else:
            comp_id, smiles = picked
            print(f"[{i+1:3d}/{len(pdb_ids)}] {pdb_id}: {comp_id}  {smiles}",
                  file=sys.stderr)
            results[pdb_id] = {
                "comp_id": comp_id,
                "smiles": smiles,
                "mol2_heavy": heavy,
            }

        # Flush every 10
        if (i + 1) % 10 == 0:
            out_path.write_text(json.dumps(results, indent=2))

    out_path.write_text(json.dumps(results, indent=2))
    ok = sum(1 for v in results.values() if v.get("smiles"))
    print(f"\nDone: {ok}/{len(pdb_ids)} SMILES fetched -> {out_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
