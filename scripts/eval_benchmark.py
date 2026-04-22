#!/usr/bin/env python
"""Evaluate FlowFrag on docking benchmarks (Astex / PoseBusters).

Auto-detects dataset format from file naming conventions.
Samples N poses once per complex, then evaluates all combinations of:
  - Refinement: none, mmff
  - Selection:  oracle (best-RMSD), cluster

Usage:
    # PoseBusters v2 (308)
    python scripts/eval_benchmark.py \
        --data_dir /mnt/data/PLI/PoseBusters/posebusters_benchmark_set \
        --checkpoint ckpt.pt --config cfg.yaml --subset v2

    # Astex Diverse (85)
    python scripts/eval_benchmark.py \
        --data_dir /mnt/data/PLI/Astex-diverse-set \
        --checkpoint ckpt.pt --config cfg.yaml

    # Any dataset directory with {id}_protein.pdb + {id}_ligand.sdf
    python scripts/eval_benchmark.py \
        --data_dir /path/to/dataset \
        --checkpoint ckpt.pt --config cfg.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, rdFMCS, rdMolAlign
from rdkit import RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dock import preprocess_complex, sample_unified
from src.scoring.ranking import select_by_clustering, rank_poses

V2_IDS_PATH = Path(__file__).resolve().parent.parent / "data" / "posebusters_v2_ids.txt"

REFINE_METHODS = ("none", "mmff")
SELECT_METHODS = ("oracle", "cluster", "vina")


# ---------------------------------------------------------------------------
# Dataset auto-detection
# ---------------------------------------------------------------------------

def detect_complex_files(complex_dir: Path, pdb_id: str) -> tuple[Path, Path, str] | None:
    """Auto-detect protein PDB and ligand file for a complex directory.

    Returns (pocket_pdb, ligand_file, format) or None if not found.
    Format is 'sdf' or 'mol2'.
    """
    # PoseBusters: {id}_protein.pdb + {id}_ligand.sdf
    prot_pdb = complex_dir / f"{pdb_id}_protein.pdb"
    lig_sdf = complex_dir / f"{pdb_id}_ligand.sdf"
    if prot_pdb.exists() and lig_sdf.exists():
        return prot_pdb, lig_sdf, "sdf"

    # Astex: {id}_pocket.pdb + {id}_ligand.mol2
    pocket_pdb = complex_dir / f"{pdb_id}_pocket.pdb"
    lig_mol2 = complex_dir / f"{pdb_id}_ligand.mol2"
    if pocket_pdb.exists() and lig_mol2.exists():
        return pocket_pdb, lig_mol2, "mol2"

    return None


def detect_dataset_name(data_dir: Path) -> str:
    """Infer dataset name from directory path."""
    name = data_dir.name.lower()
    if "posebusters" in name:
        return "posebusters"
    if "astex" in name:
        return "astex"
    return data_dir.name


# ---------------------------------------------------------------------------
# Ligand loading
# ---------------------------------------------------------------------------

def load_sdf_robust(sdf_path: Path) -> Chem.Mol:
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=True)
    mol = next(suppl)
    if mol is not None:
        frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        return mol

    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    mol = next(suppl)
    assert mol is not None, f"RDKit cannot parse {sdf_path}"
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    Chem.SanitizeMol(
        mol,
        sanitizeOps=(
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
        ),
    )
    mol = Chem.RemoveHs(mol, sanitize=False)
    frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumAtoms())
    assert mol.GetNumConformers() > 0, f"No 3D conformer in {sdf_path}"
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def load_mol2_robust(mol2_path: Path) -> Chem.Mol:
    mol = Chem.MolFromMol2File(str(mol2_path), sanitize=True)
    if mol is not None:
        mol = Chem.RemoveHs(mol)
        frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        return mol

    mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=False)
    assert mol is not None, f"RDKit cannot parse {mol2_path}"
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    Chem.SanitizeMol(
        mol,
        sanitizeOps=(
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
        ),
    )
    mol = Chem.RemoveHs(mol, sanitize=False)
    frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumAtoms())
    assert mol.GetNumConformers() > 0, f"No 3D conformer in {mol2_path}"
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def load_ligand(path: Path, fmt: str) -> Chem.Mol:
    if fmt == "sdf":
        return load_sdf_robust(path)
    elif fmt == "mol2":
        return load_mol2_robust(path)
    raise ValueError(f"Unknown ligand format: {fmt}")


# ---------------------------------------------------------------------------
# Refinement & selection
# ---------------------------------------------------------------------------

def mmff_refine(mol: Chem.Mol, pred_pos: torch.Tensor, pocket_center: torch.Tensor,
                max_iters: int = 200) -> torch.Tensor:
    """Relax a predicted pose with MMFF94s while tethering heavy atoms to keep
    the ligand in place; unconstrained minimization pulls the molecule back
    toward its gas-phase minimum and can translate it tens of Å.
    """
    try:
        # 1) Set heavy atom coords to the predicted pose BEFORE AddHs so the
        #    hydrogens are placed relative to the correct heavy geometry.
        mol_copy = Chem.RWMol(mol)
        mol_copy.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol_copy)
        conf = mol_copy.GetConformer()
        pos_abs = pred_pos + pocket_center
        n_heavy = mol_copy.GetNumAtoms()
        for j in range(n_heavy):
            conf.SetAtomPosition(j, pos_abs[j].tolist())

        mol_h = Chem.AddHs(mol_copy, addCoords=True, addResidueInfo=False)

        props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
        if props is None:
            return pred_pos
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=0)
        if ff is None:
            return pred_pos

        # 2) Tether heavy atoms with a harmonic restraint so MMFF only removes
        #    local geometric strain, not the binding-pose translation.
        for j in range(n_heavy):
            ff.MMFFAddPositionConstraint(j, 0.5, 50.0)

        ff.Minimize(maxIts=max_iters)

        refined = torch.zeros_like(pred_pos)
        conf = mol_h.GetConformer()
        for j in range(n_heavy):
            p = conf.GetAtomPosition(j)
            refined[j] = torch.tensor([p.x, p.y, p.z])
        return refined - pocket_center
    except Exception:
        return pred_pos


def compute_rmsd(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Index-wise RMSD assuming identical atom ordering (no symmetry handling)."""
    return (pred - ref).pow(2).sum(-1).mean().sqrt().item()


def compute_centroid_dist(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return (pred.mean(0) - ref.mean(0)).norm().item()


def compute_pose_rmsd(
    pose: torch.Tensor,
    ref_pos: torch.Tensor,
    pocket_center: torch.Tensor,
    dock_idx: list[int],
    mol_dock: Chem.Mol,
    mol_ref: Chem.Mol,
) -> float:
    """Symmetry-aware heavy-atom RMSD (no alignment).

    Uses RDKit rdMolAlign.CalcRMS when topology matches between mol_dock and
    mol_ref (full strict/charge-stripped match); otherwise falls back to
    tensor-based RMSD on the matched atom subset.
    """
    if len(dock_idx) == mol_dock.GetNumAtoms() == mol_ref.GetNumAtoms():
        try:
            mol_pose = Chem.RWMol(mol_dock)
            conf = mol_pose.GetConformer()
            pose_abs = pose + pocket_center
            for i in range(mol_dock.GetNumAtoms()):
                conf.SetAtomPosition(i, pose_abs[i].tolist())
            return rdMolAlign.CalcRMS(mol_pose, mol_ref)
        except Exception:
            pass
    dock_idx_t = torch.as_tensor(dock_idx, dtype=torch.long)
    return compute_rmsd(pose.index_select(0, dock_idx_t), ref_pos)


def _strip_charges(mol: Chem.Mol) -> Chem.Mol:
    m = Chem.RWMol(mol)
    for atom in m.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
    try:
        Chem.SanitizeMol(
            m,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
    except Exception:
        pass
    return m


def match_atoms(
    mol_ref: Chem.Mol, mol_dock: Chem.Mol
) -> tuple[list[int], list[int], str]:
    """Map atoms between crystal mol (mol_ref) and docking mol (mol_dock).

    Returns (dock_indices, ref_indices, method). Both index lists have the same
    length; dock_indices[i] corresponds to ref_indices[i]. Falls back from strict
    → charge-agnostic → MCS. Empty lists signal no overlap.
    """
    match = mol_ref.GetSubstructMatch(mol_dock)
    if len(match) == mol_dock.GetNumAtoms():
        return list(range(mol_dock.GetNumAtoms())), list(match), "strict"

    ref2, dock2 = _strip_charges(mol_ref), _strip_charges(mol_dock)
    match = ref2.GetSubstructMatch(dock2)
    if len(match) == dock2.GetNumAtoms():
        return list(range(mol_dock.GetNumAtoms())), list(match), "nocharges"

    mcs = rdFMCS.FindMCS(
        [ref2, dock2],
        timeout=5,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchValences=False,
        ringMatchesRingOnly=False,
    )
    if mcs.numAtoms == 0:
        return [], [], "fail"
    patt = Chem.MolFromSmarts(mcs.smartsString)
    ref_m = ref2.GetSubstructMatch(patt)
    dock_m = dock2.GetSubstructMatch(patt)
    if len(ref_m) == len(dock_m) == mcs.numAtoms:
        return list(dock_m), list(ref_m), f"mcs({mcs.numAtoms}/{mol_dock.GetNumAtoms()})"
    return [], [], "fail"


def apply_refinement(
    method: str,
    poses: list[torch.Tensor],
    mol: Chem.Mol,
    pocket_center: torch.Tensor,
) -> list[torch.Tensor]:
    if method == "none":
        return poses
    return [mmff_refine(mol, pos, pocket_center) for pos in poses]


def select_pose(
    method: str,
    poses: list[torch.Tensor],
    rmsds: list[float],
    cluster_threshold: float = 2.0,
) -> int:
    if method == "oracle":
        return int(np.argmin(rmsds))
    elif method == "cluster":
        return select_by_clustering(poses, threshold=cluster_threshold)
    raise ValueError(f"Unknown selection method: {method}")


def compute_stats(rmsds: np.ndarray) -> dict:
    return {
        "mean_rmsd": float(rmsds.mean()),
        "median_rmsd": float(np.median(rmsds)),
        "std_rmsd": float(rmsds.std()),
        "pct_lt_1A": float((rmsds < 1.0).mean() * 100),
        "pct_lt_2A": float((rmsds < 2.0).mean() * 100),
        "pct_lt_3A": float((rmsds < 3.0).mean() * 100),
        "pct_lt_5A": float((rmsds < 5.0).mean() * 100),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FlowFrag on docking benchmarks (Astex / PoseBusters / custom)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dataset directory (auto-detects Astex vs PoseBusters)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--subset", type=str, default="all", choices=("all", "v2"),
                        help="v2 = PoseBusters 308 subset, all = everything in data_dir")
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--time_schedule", type=str, default="late")
    parser.add_argument("--schedule_power", type=float, default=3.0)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=40)
    parser.add_argument("--cluster_threshold", type=float, default=2.0)
    parser.add_argument("--smiles_init", action="store_true",
                        help="Re-generate 3D conformer via ETKDGv3+MMFF (simulate SMILES input)")
    parser.add_argument("--smiles_map", type=str, default=None,
                        help="Optional JSON file mapping pdb_id -> {smiles, ...}. "
                             "Used with --smiles_init to build 3D from external SMILES "
                             "instead of round-tripping the crystal mol.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: outputs/eval_{dataset_name})")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"Data directory not found: {data_dir}"
    dataset_name = detect_dataset_name(data_dir)

    if args.out_dir is None:
        args.out_dir = f"outputs/eval_{dataset_name}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- Load model ---
    model_cfg = dict(cfg["model"])
    model_cfg.pop("model_type", None)

    from src.models.unified import UnifiedFlowFrag
    model = UnifiedFlowFrag(**model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)
    step = ckpt.get("step", "?")
    print(f"Model loaded: {args.checkpoint} (step {step})")
    print(f"Device: {device}")

    sigma = args.sigma if args.sigma is not None else cfg["data"].get("prior_sigma", 1.0)

    smiles_map: dict = {}
    if args.smiles_map is not None:
        smiles_map_path = Path(args.smiles_map)
        assert smiles_map_path.exists(), f"smiles_map not found: {smiles_map_path}"
        smiles_map = json.loads(smiles_map_path.read_text())
        ok = sum(1 for v in smiles_map.values() if isinstance(v, dict) and v.get("smiles"))
        print(f"Loaded SMILES map: {ok}/{len(smiles_map)} entries with SMILES "
              f"({smiles_map_path})")

    # --- Discover complexes ---
    all_dirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    if args.subset == "v2":
        assert V2_IDS_PATH.exists(), f"v2 ID list not found: {V2_IDS_PATH}"
        with open(V2_IDS_PATH) as f:
            v2_ids = {line.strip() for line in f if line.strip()}
        pdb_ids = [d for d in all_dirs if d in v2_ids]
        subset_label = f"v2 ({len(pdb_ids)})"
    else:
        pdb_ids = all_dirs
        subset_label = f"all ({len(pdb_ids)})"

    print(f"\n{dataset_name}: {len(pdb_ids)} complexes [{subset_label}]")
    smiles_tag = ", smiles_init=True" if args.smiles_init else ""
    print(f"Settings: {args.num_samples} samples, {args.num_steps} steps, "
          f"schedule={args.time_schedule}, sigma={sigma}{smiles_tag}")
    print(f"Evaluating: {len(REFINE_METHODS)} refinements x {len(SELECT_METHODS)} selections "
          f"= {len(REFINE_METHODS) * len(SELECT_METHODS)} combos\n")

    # --- Per-combo accumulators ---
    combo_results: dict[tuple[str, str], list[dict]] = {
        (r, s): [] for r in REFINE_METHODS for s in SELECT_METHODS
    }
    failures = []
    t_start = time.time()

    for idx, pdb_id in enumerate(pdb_ids):
        complex_dir = data_dir / pdb_id
        detected = detect_complex_files(complex_dir, pdb_id)

        if detected is None:
            print(f"[{idx+1:3d}/{len(pdb_ids)}] {pdb_id}: SKIP (missing files)")
            failures.append({"pdb_id": pdb_id, "error": "missing files"})
            continue

        pocket_pdb, ligand_file, fmt = detected

        try:
            mol = load_ligand(ligand_file, fmt)
            mol_ref = mol

            # --smiles_init: re-generate 3D conformer (simulate SMILES input).
            # Prefer external SMILES (from --smiles_map); fall back to round-tripping
            # the crystal mol through canonical SMILES.
            if args.smiles_init:
                RDLogger.DisableLog("rdApp.*")
                external = smiles_map.get(pdb_id, {}) if smiles_map else {}
                smi = external.get("smiles") if isinstance(external, dict) else None
                source = "external"
                if not smi:
                    smi = Chem.MolToSmiles(mol)
                    source = "roundtrip"
                mol_from_smi = Chem.MolFromSmiles(smi) if smi else None
                if mol_from_smi is None:
                    RDLogger.EnableLog("rdApp.*")
                    print(f"[{idx+1:3d}/{len(pdb_ids)}] {pdb_id}: SKIP (SMILES parse failed [{source}])")
                    failures.append({"pdb_id": pdb_id, "error": f"SMILES parse failed [{source}]"})
                    continue
                mol_h = Chem.AddHs(mol_from_smi)
                status = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
                RDLogger.EnableLog("rdApp.*")
                if status != 0:
                    print(f"[{idx+1:3d}/{len(pdb_ids)}] {pdb_id}: SKIP (ETKDGv3 embed failed [{source}])")
                    failures.append({"pdb_id": pdb_id, "error": f"ETKDGv3 embed failed [{source}]"})
                    continue
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
                mol_smiles = Chem.RemoveHs(mol_h)
                # Keep original mol for ref_pos, use mol_smiles for docking
                mol_ref = mol
                mol = mol_smiles

            poses_dir = out_dir / "poses"
            poses_dir.mkdir(exist_ok=True)
            poses_file = poses_dir / f"{pdb_id}.pt"

            # Resume: reuse saved poses if present and have enough samples
            resumed = False
            if poses_file.exists():
                saved = torch.load(poses_file, map_location="cpu", weights_only=False)
                if len(saved["raw_poses"]) >= args.num_samples:
                    raw_poses = saved["raw_poses"][: args.num_samples]
                    ref_pos = saved["ref_pos"]
                    pocket_center = saved["pocket_center"]
                    meta = {
                        "num_atom": saved["n_atoms"],
                        "num_frag": saved["n_frags"],
                    }
                    dock_idx = saved.get("dock_idx", list(range(int(saved["n_atoms"]))))
                    match_how = saved.get("match_how", "identity")
                    resumed = True

            if not resumed:
                if args.smiles_init:
                    # Derive pocket_center + ref_pos from crystal mol, dock with re-embedded mol.
                    # SMILES-sourced mol may differ in atom ordering or protonation;
                    # use strict → charge-stripped → MCS matching.
                    _, lig_ref, meta_ref = preprocess_complex(pocket_pdb, mol_ref)
                    pocket_center = meta_ref["pocket_center"]
                    dock_idx, ref_idx, match_how = match_atoms(mol_ref, mol)
                    if not dock_idx:
                        print(f"[{idx+1:3d}/{len(pdb_ids)}] {pdb_id}: SKIP (atom match failed)")
                        failures.append({"pdb_id": pdb_id, "error": "atom match failed"})
                        continue
                    ref_pos = lig_ref["atom_coords"][ref_idx] - pocket_center
                    graph, lig_data, meta = preprocess_complex(pocket_pdb, mol, pocket_center=pocket_center)
                else:
                    graph, lig_data, meta = preprocess_complex(pocket_pdb, mol)
                    pocket_center = meta["pocket_center"]
                    ref_pos = lig_data["atom_coords"] - pocket_center
                    dock_idx = list(range(int(meta["num_atom"])))
                    match_how = "identity"

                # --- Sample N poses (once) ---
                raw_poses = []
                for _ in range(args.num_samples):
                    result = sample_unified(
                        model, graph, lig_data, meta,
                        num_steps=args.num_steps,
                        translation_sigma=sigma,
                        time_schedule=args.time_schedule,
                        schedule_power=args.schedule_power,
                        device=device,
                    )
                    raw_poses.append(result["atom_pos_pred"])

                torch.save({
                    "pdb_id": pdb_id,
                    "raw_poses": raw_poses,
                    "ref_pos": ref_pos,
                    "pocket_center": pocket_center,
                    "n_atoms": meta["num_atom"],
                    "n_frags": meta["num_frag"],
                    "dock_idx": dock_idx,
                    "match_how": match_how,
                }, poses_file)

            # --- Apply each refinement, then each selection ---
            # Use symmetry-aware RDKit RMSD when topology matches; otherwise
            # fall back to subset tensor RMSD (for partial MCS).
            dock_idx_t = torch.as_tensor(dock_idx, dtype=torch.long)
            best_rmsds_str = []
            for refine in REFINE_METHODS:
                poses = apply_refinement(refine, raw_poses, mol, pocket_center)
                poses_matched = [p.index_select(0, dock_idx_t) for p in poses]
                rmsds = [
                    compute_pose_rmsd(p, ref_pos, pocket_center, dock_idx, mol, mol_ref)
                    for p in poses
                ]
                cdists = [compute_centroid_dist(p, ref_pos) for p in poses_matched]

                # Vina + validity ranking: pick top-scoring pose
                vina_best_idx = 0
                try:
                    ranked = rank_poses(
                        mol, poses, pocket_pdb, pocket_center, device=device,
                    )
                    vina_best_idx = ranked[0]["idx"]
                except Exception:
                    pass

                for select in SELECT_METHODS:
                    if select == "vina":
                        sel_idx = vina_best_idx
                    else:
                        sel_idx = select_pose(
                            select, poses_matched, rmsds, args.cluster_threshold,
                        )
                    entry = {
                        "pdb_id": pdb_id,
                        "rmsd": rmsds[sel_idx],
                        "centroid_dist": cdists[sel_idx],
                        "oracle_rmsd": min(rmsds),
                        "n_atoms": meta["num_atom"],
                        "n_frags": meta["num_frag"],
                        "match": match_how,
                    }
                    combo_results[(refine, select)].append(entry)

                best_rmsds_str.append(f"{refine}={min(rmsds):.2f}")

            tag = " [R]" if resumed else ""
            match_tag = "" if match_how in ("strict", "identity") else f" [{match_how}]"
            print(f"[{idx+1:3d}/{len(pdb_ids)}] {pdb_id}:  "
                  f"{'  '.join(best_rmsds_str)}  "
                  f"atoms={meta['num_atom']:3d}  frags={meta['num_frag']:2d}{tag}{match_tag}")

        except Exception as e:
            print(f"[{idx+1:3d}/{len(pdb_ids)}] {pdb_id}: FAIL ({e})")
            failures.append({"pdb_id": pdb_id, "error": str(e)})
            traceback.print_exc()

    elapsed = time.time() - t_start

    # --- Summary table ---
    print(f"\n{'='*75}")
    print(f"{dataset_name} [{subset_label}] — {len(pdb_ids)} complexes, "
          f"{args.num_samples} samples")
    print(f"Time: {elapsed:.0f}s ({elapsed/max(len(pdb_ids),1):.1f}s/complex)")
    print(f"Failures: {len(failures)}")
    print(f"{'='*75}")

    header = f"{'Refine':<8} {'Select':<8} {'Mean':>6} {'Med':>6} {'<1A':>6} {'<2A':>6} {'<5A':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    all_stats = {}
    for refine in REFINE_METHODS:
        for select in SELECT_METHODS:
            results = combo_results[(refine, select)]
            if not results:
                continue
            rmsds = np.array([r["rmsd"] for r in results])
            s = compute_stats(rmsds)
            all_stats[f"{refine}+{select}"] = s
            print(f"{refine:<8} {select:<8} {s['mean_rmsd']:6.2f} {s['median_rmsd']:6.2f} "
                  f"{s['pct_lt_1A']:5.1f}% {s['pct_lt_2A']:5.1f}% {s['pct_lt_5A']:5.1f}%")

    # --- Save full results ---
    summary = {
        "dataset": dataset_name,
        "data_dir": str(data_dir),
        "checkpoint": args.checkpoint,
        "step": step,
        "subset": args.subset,
        "num_complexes": len(pdb_ids),
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "sigma": sigma,
        "time_schedule": args.time_schedule,
        "elapsed_seconds": elapsed,
        "failures": failures,
        "stats": all_stats,
        "per_complex": {
            f"{r}+{s}": combo_results[(r, s)]
            for r in REFINE_METHODS for s in SELECT_METHODS
        },
    }

    out_file = out_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
