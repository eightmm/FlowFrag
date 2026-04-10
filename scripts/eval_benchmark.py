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
from rdkit.Chem import AllChem, rdmolops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dock import preprocess_complex, sample_unified
from src.scoring.clustering import select_by_clustering

V2_IDS_PATH = Path(__file__).resolve().parent.parent / "data" / "posebusters_v2_ids.txt"

REFINE_METHODS = ("none", "mmff")
SELECT_METHODS = ("oracle", "cluster")


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
    try:
        mol_h = Chem.RWMol(mol)
        mol_h.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol_h)
        mol_h = Chem.AddHs(mol_h, addCoords=True, addResidueInfo=False)
        conf = mol_h.GetConformer()
        pos_abs = pred_pos + pocket_center
        heavy_idx = [i for i in range(mol_h.GetNumAtoms())
                     if mol_h.GetAtomWithIdx(i).GetAtomicNum() != 1]
        for j, hi in enumerate(heavy_idx):
            conf.SetAtomPosition(hi, pos_abs[j].tolist())
        props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
        if props is None:
            return pred_pos
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=0)
        if ff is None:
            return pred_pos
        ff.Minimize(maxIts=max_iters)
        refined = torch.zeros_like(pred_pos)
        conf = mol_h.GetConformer()
        for j, hi in enumerate(heavy_idx):
            p = conf.GetAtomPosition(hi)
            refined[j] = torch.tensor([p.x, p.y, p.z])
        return refined - pocket_center
    except Exception:
        return pred_pos


def compute_rmsd(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return (pred - ref).pow(2).sum(-1).mean().sqrt().item()


def compute_centroid_dist(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return (pred.mean(0) - ref.mean(0)).norm().item()


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
    print(f"Settings: {args.num_samples} samples, {args.num_steps} steps, "
          f"schedule={args.time_schedule}, sigma={sigma}")
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
                    resumed = True

            if not resumed:
                graph, lig_data, meta = preprocess_complex(pocket_pdb, mol, ligand_has_pose=True)
                pocket_center = meta["pocket_center"]
                ref_pos = lig_data["atom_coords"] - pocket_center

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
                }, poses_file)

            # --- Apply each refinement, then each selection ---
            best_rmsds_str = []
            for refine in REFINE_METHODS:
                poses = apply_refinement(refine, raw_poses, mol, pocket_center)
                rmsds = [compute_rmsd(p, ref_pos) for p in poses]
                cdists = [compute_centroid_dist(p, ref_pos) for p in poses]

                for select in SELECT_METHODS:
                    sel_idx = select_pose(
                        select, poses, rmsds, args.cluster_threshold,
                    )
                    entry = {
                        "pdb_id": pdb_id,
                        "rmsd": rmsds[sel_idx],
                        "centroid_dist": cdists[sel_idx],
                        "oracle_rmsd": min(rmsds),
                        "n_atoms": meta["num_atom"],
                        "n_frags": meta["num_frag"],
                    }
                    combo_results[(refine, select)].append(entry)

                best_rmsds_str.append(f"{refine}={min(rmsds):.2f}")

            tag = " [R]" if resumed else ""
            print(f"[{idx+1:3d}/{len(pdb_ids)}] {pdb_id}:  "
                  f"{'  '.join(best_rmsds_str)}  "
                  f"atoms={meta['num_atom']:3d}  frags={meta['num_frag']:2d}{tag}")

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
