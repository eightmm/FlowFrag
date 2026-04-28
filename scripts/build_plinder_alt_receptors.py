#!/usr/bin/env python
"""Preprocess PLINDER alternate receptors (apo / predicted) into shared
``protein.pt`` files for receptor-augmentation training.

Reads ``data/plinder_alt_receptors.parquet`` (HQ-filtered links; one row per
``(reference_system_id, alt_id)`` pair), deduplicates by ``alt_id``, parses each
``.cif`` (apo from PDB or AF2-predicted), and saves a single ``protein.pt`` per
unique alt structure to ``{out_dir}/_alt_proteins/{alt_id}.pt``.

At training time, ``Dataset.__getitem__`` randomly swaps the holo receptor with
one of these alt receptors (with prob ``data.receptor_aug_prob``); the holo
ligand still defines the binding site, so we crop the alt receptor's pocket
using the holo pocket center.
"""

import argparse
import json
import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from Bio.PDB import MMCIFParser, PDBIO

from src.preprocess.protein import parse_pocket_atoms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _cif_to_temp_pdb(cif_path: Path) -> Path | None:
    """Convert mmCIF to a temporary PDB file. Returns None on parse failure."""
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("S", str(cif_path))
    except Exception:
        return None
    io = PDBIO()
    io.set_structure(structure)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.close()
    try:
        io.save(tmp.name)
        return Path(tmp.name)
    except Exception:
        Path(tmp.name).unlink(missing_ok=True)
        return None


def process_alt(alt_id: str, alt_root: Path, out_dir: Path) -> dict:
    status = {"alt_id": alt_id, "success": False, "reason": ""}
    cif_path = alt_root / f"{alt_id}.cif"
    out_path = out_dir / f"{alt_id}.pt"
    if out_path.exists():
        status["success"] = True
        status["reason"] = "already_done"
        return status
    if not cif_path.exists():
        status["reason"] = "missing_cif"
        return status

    pdb_tmp = _cif_to_temp_pdb(cif_path)
    if pdb_tmp is None:
        status["reason"] = "cif_parse_failed"
        return status

    try:
        prot_data = parse_pocket_atoms(pdb_tmp)
    finally:
        pdb_tmp.unlink(missing_ok=True)

    if prot_data is None or prot_data["patom_coords"].shape[0] == 0:
        status["reason"] = "empty_protein"
        return status

    # Save full protein (no crop — runtime crop_to_pocket handles it via holo center)
    torch.save(prot_data, out_path)
    status["success"] = True
    status["num_res"] = prot_data["pres_coords"].shape[0]
    status["num_prot_atom"] = prot_data["patom_coords"].shape[0]
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt_links_parquet", type=Path,
                    default=Path("data/plinder_alt_receptors.parquet"))
    ap.add_argument("--alt_root", type=Path,
                    default=Path("/home/jaemin/.local/share/plinder/2024-06/v2/linked_structures"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("data/plinder_processed/_alt_proteins"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.alt_links_parquet)
    unique_ids = df["id"].drop_duplicates().tolist()
    if args.limit:
        unique_ids = unique_ids[: args.limit]
    log.info("Processing %d unique alt receptors", len(unique_ids))
    log.info("alt_root: %s", args.alt_root)
    log.info("out_dir:  %s", args.out_dir)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_alt, aid, args.alt_root, args.out_dir): aid for aid in unique_ids}
        for i, fut in enumerate(as_completed(futures)):
            aid = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"alt_id": aid, "success": False, "reason": f"exc:{e}"}
            results.append(r)
            if (i + 1) % 500 == 0:
                ok = sum(1 for x in results if x["success"])
                log.info("  progress %d/%d, success=%d", i + 1, len(unique_ids), ok)

    success = sum(1 for r in results if r["success"])
    log.info("=" * 60)
    log.info("Done. success=%d / %d (%.1f%%)", success, len(results),
             100 * success / max(len(results), 1))
    if any(not r["success"] for r in results):
        from collections import Counter
        reasons = Counter(r["reason"] for r in results if not r["success"])
        for reason, c in sorted(reasons.items(), key=lambda x: -x[1]):
            log.info("  %s: %d", reason, c)

    # === Build system_id → [alt_id] mapping ===
    df_ok = df[df["id"].isin([r["alt_id"] for r in results if r["success"]])]
    mapping = (
        df_ok.groupby("reference_system_id")
        .apply(lambda g: g.sort_values("sort_score", ascending=False)[["id", "kind", "pocket_lddt"]].to_dict("records"))
        .to_dict()
    )
    json.dump({"mapping": mapping, "n_systems": len(mapping)},
              open(args.out_dir.parent / "alt_receptor_mapping.json", "w"))
    log.info("Mapping saved → %s (%d systems)", args.out_dir.parent / "alt_receptor_mapping.json", len(mapping))


if __name__ == "__main__":
    main()
