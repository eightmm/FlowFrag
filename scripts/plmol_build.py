#!/usr/bin/env python
"""All-in-one PLINDER preprocessor using plmol for chemistry/geometry features.

Single-file build pipeline:
  1. Iterate (system_id, ligand_chain) rows from data/plinder_train_filtered.parquet
  2. For each row:
       - LigandFeaturizer  → atom 98-d features + 37-d bond adjacency + fragments + 62-d molecule descriptors
       - ProteinFeaturizer → atom tokens/coords + residue node features + backbone dihedrals/curvature/torsion + edge features
       - Add fragment local frames (rigid) computed from plmol's atom_to_fragment + crystal coords
       - Add pocket center from holo ligand
  3. Save protein.pt / ligand.pt / meta.pt under data/plinder_processed_v3/{key}/

This is a v3 schema EXPLORATORY build; we run it on a small subset first to
inspect what plmol gives us, before deciding to migrate fully.

Run:
    uv run python scripts/plmol_build.py --limit 30 --workers 4 \
        --out_dir data/plinder_processed_v3_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# plmol entrypoints
from plmol import LigandFeaturizer, ProteinFeaturizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCHEMA_VERSION = 3  # plmol-based features


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def sample_key(system_id: str, instance_chain: str) -> str:
    return f"{system_id}__{instance_chain}".replace("/", "_")


def load_ligand_mol(sdf_path: Path) -> Chem.Mol | None:
    """Load and sanitize a single ligand SDF (returns largest fragment)."""
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    raw = next((m for m in suppl if m is not None), None)
    if raw is None:
        return None
    try:
        Chem.SanitizeMol(raw)
    except Exception:
        try:
            Chem.SanitizeMol(
                raw,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:
            return None
    try:
        Chem.AssignStereochemistryFrom3D(raw)
    except Exception:
        pass
    raw = Chem.RemoveHs(raw)
    # Largest connected component
    frags = Chem.rdmolops.GetMolFrags(raw, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        raw = max(frags, key=lambda m: m.GetNumAtoms())
    if raw.GetNumAtoms() < 2 or raw.GetNumConformers() == 0:
        return None
    return raw


# ---------------------------------------------------------------------------
# Fragment local frame (FlowFrag-specific: rigid local coords)
# ---------------------------------------------------------------------------

def build_fragment_geometry(
    coords: torch.Tensor,           # [N_atom, 3] crystal coords
    atom_to_fragment: torch.Tensor, # [N_atom] from plmol
) -> dict[str, torch.Tensor]:
    """Compute per-fragment centroid + rigid local coords.

    plmol gives us atom_to_fragment, but FlowFrag needs:
      - frag_centers [F, 3] = mean of crystal coords per fragment
      - frag_local_coords [N, 3] = crystal_coords - frag_centers[atom_to_fragment]

    These are the rigid invariants the SE(3) flow-matching head learns over.
    """
    n_frag = int(atom_to_fragment.max().item()) + 1
    frag_centers = torch.zeros(n_frag, 3, dtype=coords.dtype)
    frag_sizes = torch.zeros(n_frag, dtype=torch.int64)
    for fid in range(n_frag):
        mask = atom_to_fragment == fid
        if not mask.any():
            continue
        frag_centers[fid] = coords[mask].mean(0)
        frag_sizes[fid] = int(mask.sum())
    frag_local_coords = coords - frag_centers[atom_to_fragment]
    return {
        "frag_centers": frag_centers,
        "frag_sizes": frag_sizes,
        "frag_local_coords": frag_local_coords,
        "n_frags": n_frag,
    }


# ---------------------------------------------------------------------------
# Per-sample pipeline
# ---------------------------------------------------------------------------

def process_one(row: dict, plinder_root: Path, out_dir: Path) -> dict[str, Any]:
    system_id = row["system_id"]
    chain = row["ligand_instance_chain"]
    key = sample_key(system_id, chain)
    status: dict[str, Any] = {"key": key, "success": False, "reason": ""}

    sys_dir = plinder_root / "systems" / system_id
    pdb_path = sys_dir / "receptor.pdb"
    sdf_path = sys_dir / "ligand_files" / f"{chain}.sdf"
    if not pdb_path.exists():
        status["reason"] = "missing_receptor_pdb"; return status
    if not sdf_path.exists():
        status["reason"] = "missing_ligand_sdf"; return status

    # Idempotent skip
    out_complex = out_dir / key
    if (out_complex / "meta.pt").exists():
        status["success"] = True
        status["reason"] = "already_done"
        return status

    # ---- LIGAND via plmol -------------------------------------------------
    mol = load_ligand_mol(sdf_path)
    if mol is None:
        status["reason"] = "ligand_parse_failed"; return status

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lf = LigandFeaturizer(mol)
            lig = lf.featurize(mode="graph")["graph"]
    except Exception as e:
        status["reason"] = f"plmol_ligand_failed: {e}"; return status

    # Coerce numpy → torch for plmol outputs
    coords = _to_tensor(lig["coords"]).float()              # [N, 3]
    atom_to_frag = _to_tensor(lig["atom_to_fragment"]).long()
    bond_mask = _to_tensor(lig["bond_mask"]).bool()         # [N, N]
    n_atoms = coords.shape[0]
    if n_atoms < 2:
        status["reason"] = "too_few_atoms"; return status

    frag_geom = build_fragment_geometry(coords, atom_to_frag)

    # cut_bond_index: covalent bonds whose endpoints are in different fragments
    frag_diff = atom_to_frag[:, None] != atom_to_frag[None, :]
    cut_mask = bond_mask & frag_diff
    cut_bond_index = torch.triu(cut_mask, diagonal=1).nonzero().t().contiguous()

    # Sparse bond list + per-edge attributes (plmol gives dense [N, N, 37])
    bond_idx_pairs = torch.triu(bond_mask, diagonal=1).nonzero()  # [E, 2]
    if bond_idx_pairs.numel() == 0:
        bond_index = torch.zeros(2, 0, dtype=torch.int64)
        bond_attrs = torch.zeros(0, 0, dtype=torch.float32)
    else:
        src = torch.cat([bond_idx_pairs[:, 0], bond_idx_pairs[:, 1]])
        dst = torch.cat([bond_idx_pairs[:, 1], bond_idx_pairs[:, 0]])
        bond_index = torch.stack([src, dst])
        edge_attr = _to_tensor(lig["adjacency"]).float()
        bond_attrs = torch.cat([
            edge_attr[bond_idx_pairs[:, 0], bond_idx_pairs[:, 1]],
            edge_attr[bond_idx_pairs[:, 1], bond_idx_pairs[:, 0]],
        ], dim=0)

    ligand_data: dict[str, torch.Tensor] = {
        # Plmol features (rich)
        "atom_features": lig["node_features"],          # [N, 98]
        "atom_coords": coords,                           # [N, 3]
        "molecule_features": lig["molecule_features"],   # [62]
        "distance_bounds": lig["distance_bounds"],       # [N, N, 2]  (compact via masking later)
        # Bonds (sparse)
        "bond_index": bond_index,                        # [2, 2*E]
        "bond_attr": bond_attrs,                         # [2*E, 37]
        # Fragments (from plmol + our local frame)
        "fragment_id": atom_to_frag,                     # [N]
        "frag_centers": frag_geom["frag_centers"],       # [F, 3]
        "frag_local_coords": frag_geom["frag_local_coords"],  # [N, 3]
        "frag_sizes": frag_geom["frag_sizes"],           # [F]
        "fragment_adj_index": _adj_to_edge_index(lig["fragment_adjacency"]),
        "cut_bond_index": cut_bond_index,                # [2, n_cut]
    }

    # ---- PROTEIN via plmol ------------------------------------------------
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = ProteinFeaturizer(str(pdb_path), standardize=True, keep_hydrogens=False)

            # atom-level
            atom_tok, atom_coord = pf.get_atom_tokens_and_coords()

            # residue-level node features (multi-D vector per residue)
            res_nodes = pf.get_node_features()         # dict
            geom = pf.get_geometric_features()         # dict
            backbone = pf.get_backbone()               # dict
            edges = pf.get_edge_features()             # dict (residue kNN edges)
    except Exception as e:
        status["reason"] = f"plmol_protein_failed: {e}"; return status

    protein_data: dict[str, torch.Tensor] = {
        # Atom-level
        "patom_token": atom_tok,                         # [N_patom] int64
        "patom_coords": atom_coord,                      # [N_patom, 3]
        # Residue-level (plmol's hierarchical features)
        "res_features": _coerce_dict_to_tensor(res_nodes),  # [N_res, F_res]
        "res_dihedrals": geom.get("dihedrals"),          # [N_res, 8]  (phi/psi/omega + chi1-4 + ...)
        "res_has_chi": geom.get("has_chi_angles"),       # [N_res, 5]
        "res_backbone_curvature": geom.get("backbone_curvature"),  # [N_res]
        "res_backbone_torsion": geom.get("backbone_torsion"),       # [N_res]
        "res_self_distances": geom.get("self_distances"),  # [N_res, 10]
        "res_self_vectors": geom.get("self_vectors"),      # [N_res, 20, 3]
        "res_coords_full": geom.get("coords"),             # [N_res, MAX_ATOM, 3]
        # Backbone
        "backbone_features": _coerce_dict_to_tensor(backbone),
        # Edges (kNN / contact)
        "edge_index": edges.get("edge_index"),
        "edge_attr": _coerce_dict_to_tensor(edges, exclude=("edge_index",)),
    }

    # Pocket center: derived from THIS ligand's crystal coords
    crystal_pos = coords  # ligand is already at crystal pose
    res_ca = protein_data["res_coords_full"][:, 1, :] if protein_data["res_coords_full"] is not None else None
    if res_ca is not None and res_ca.shape[0] > 0:
        d = torch.cdist(res_ca, crystal_pos)
        pocket_mask = (d.min(dim=1).values <= 8.0)
        if pocket_mask.any():
            pocket_center = res_ca[pocket_mask].mean(0)
        else:
            pocket_center = crystal_pos.mean(0)
        n_pocket_res = int(pocket_mask.sum().item())
    else:
        pocket_center = crystal_pos.mean(0)
        n_pocket_res = 0

    # ---- META ------------------------------------------------------------
    meta = {
        "pdb_id": key,
        "plinder_system_id": system_id,
        "plinder_ligand_chain": chain,
        "plinder_ccd_code": str(row.get("ligand_unique_ccd_code", "")),
        "is_cofactor": bool(row.get("ligand_is_cofactor", False)),
        "is_kinase_inhibitor": bool(row.get("ligand_is_kinase_inhibitor", False)),
        "pocket_center": pocket_center,
        "num_pocket_res": torch.tensor(n_pocket_res, dtype=torch.int64),
        "num_atom": torch.tensor(n_atoms, dtype=torch.int64),
        "num_frag": torch.tensor(frag_geom["n_frags"], dtype=torch.int64),
        "num_res": torch.tensor(int(protein_data["res_coords_full"].shape[0]) if protein_data["res_coords_full"] is not None else 0, dtype=torch.int64),
        "num_prot_atom": torch.tensor(int(atom_tok.shape[0]), dtype=torch.int64),
        "schema_version": SCHEMA_VERSION,
        "source": "plinder_2024_06_v2",
        "featurizer": "plmol@0.2.1",
    }

    out_complex.mkdir(parents=True, exist_ok=True)
    torch.save(protein_data, out_complex / "protein.pt")
    torch.save(ligand_data, out_complex / "ligand.pt")
    torch.save(meta, out_complex / "meta.pt")

    status["success"] = True
    status["num_atom"] = n_atoms
    status["num_frag"] = frag_geom["n_frags"]
    status["num_res"] = int(meta["num_res"].item())
    status["num_prot_atom"] = int(meta["num_prot_atom"].item())
    status["is_cofactor"] = meta["is_cofactor"]
    return status


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _to_tensor(x: Any) -> torch.Tensor:
    """Coerce numpy/list/tensor to torch tensor (no-copy when possible)."""
    if x is None:
        return torch.zeros(0)
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.tensor(x)


def _adj_to_edge_index(adj: Any) -> torch.Tensor:
    """Convert dense [F, F] adjacency to [2, E] edge_index (bidirectional)."""
    if adj is None:
        return torch.zeros(2, 0, dtype=torch.int64)
    t = _to_tensor(adj)
    if t.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.int64)
    a = (t > 0).nonzero().t().contiguous()
    return a if a.shape[1] > 0 else torch.zeros(2, 0, dtype=torch.int64)


def _coerce_dict_to_tensor(d: Any, exclude: tuple[str, ...] = ()) -> Any:
    """Concatenate dict-of-tensors along last axis. Returns dict or tensor."""
    if not isinstance(d, dict):
        return d
    parts = []
    for k, v in d.items():
        if k in exclude or v is None:
            continue
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            parts.append(v if v.ndim >= 2 else v.unsqueeze(-1))
    if not parts:
        return d  # leave as-is
    try:
        return torch.cat(parts, dim=-1)
    except Exception:
        return d  # shape mismatch — preserve dict for inspection


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered_parquet", type=Path,
                    default=Path("data/plinder_train_filtered.parquet"))
    ap.add_argument("--plinder_root", type=Path,
                    default=Path("/home/jaemin/.local/share/plinder/2024-06/v2"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("data/plinder_processed_v3"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.filtered_parquet)
    if args.limit:
        df = df.head(args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Loaded %d (system,ligand) rows", len(df))
    log.info("PLINDER root: %s", args.plinder_root)
    log.info("Out:          %s", args.out_dir)
    log.info("Workers:      %d", args.workers)

    rows = df.to_dict("records")
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, r, args.plinder_root, args.out_dir): r for r in rows}
            for i, fut in enumerate(as_completed(futures)):
                try:
                    results.append(fut.result())
                except Exception as e:
                    row = futures[fut]
                    results.append({
                        "key": sample_key(row["system_id"], row["ligand_instance_chain"]),
                        "success": False, "reason": f"exception:{e}",
                    })
                if (i + 1) % 500 == 0:
                    ok = sum(1 for r in results if r["success"])
                    log.info("  progress %d/%d, success=%d", i + 1, len(rows), ok)
    else:
        for i, row in enumerate(rows):
            try:
                results.append(process_one(row, args.plinder_root, args.out_dir))
            except Exception as e:
                results.append({
                    "key": sample_key(row["system_id"], row["ligand_instance_chain"]),
                    "success": False, "reason": f"exception:{e}",
                })
            if (i + 1) % 50 == 0:
                ok = sum(1 for r in results if r["success"])
                log.info("  progress %d/%d, success=%d", i + 1, len(rows), ok)

    success = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]
    log.info("=" * 60)
    log.info("Done. success=%d / %d (%.1f%%)", len(success), len(results),
             100 * len(success) / max(len(results), 1))
    if fail:
        from collections import Counter
        for reason, c in sorted(Counter(r["reason"] for r in fail).items(), key=lambda x: -x[1]):
            log.info("  fail: %s: %d", reason, c)
    if success:
        for label, key in [("Lig atoms", "num_atom"), ("Fragments", "num_frag"),
                           ("Prot residues", "num_res"), ("Prot atoms", "num_prot_atom")]:
            vals = sorted([r[key] for r in success if key in r])
            if vals:
                log.info("  %-16s min=%d med=%d max=%d", label, vals[0],
                         vals[len(vals)//2], vals[-1])
        cof = sum(1 for r in success if r.get("is_cofactor"))
        log.info("  cofactors:    %d (%.1f%%)", cof, 100*cof/len(success))

    json.dump({
        "schema_version": SCHEMA_VERSION,
        "featurizer": "plmol@0.2.1",
        "source": "plinder_2024_06_v2",
        "filtered_parquet": str(args.filtered_parquet),
        "total": len(results),
        "success": len(success),
        "failed": len(fail),
        "keys": sorted([r["key"] for r in success]),
    }, open(args.out_dir / "manifest.json", "w"), indent=2)
    log.info("manifest -> %s", args.out_dir / "manifest.json")


if __name__ == "__main__":
    main()
