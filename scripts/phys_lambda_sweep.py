#!/usr/bin/env python
"""Lambda sweep for Vina-gradient guidance on a small set of Astex complexes.

For each (pdb_id, lambda_max), samples K poses with fixed per-sample seeds so
priors are shared across lambdas (the only variable is the guidance strength).
Reports per-complex oracle/median RMSD vs crystal.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.evaluation import load_mol2_robust, load_sdf_robust, match_atoms
from src.inference.preprocess import preprocess_complex
from src.inference.sampler import sample_unified
from src.models.unified import UnifiedFlowFrag


def compute_rmsd_symm(mol_dock: Chem.Mol, pose: torch.Tensor, pocket_center: torch.Tensor,
                       mol_ref: Chem.Mol, dock_idx, ref_idx) -> float:
    pose_abs = pose + pocket_center
    m = Chem.RWMol(mol_dock)
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, pose_abs[i].tolist())
    try:
        from rdkit.Chem import rdMolAlign
        if len(dock_idx) == mol_dock.GetNumAtoms() == mol_ref.GetNumAtoms():
            return float(rdMolAlign.CalcRMS(m, mol_ref))
    except Exception:
        pass
    # Fall back to index-wise RMSD on matched subset (in pocket-centered frame)
    ref_arr = np.array(mol_ref.GetConformer().GetPositions()) - pocket_center.numpy()
    pose_arr = pose.numpy()
    d = pose_arr[list(dock_idx)] - ref_arr[list(ref_idx)]
    return float(np.sqrt((d * d).sum(-1).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/mnt/data/PLI/Astex-diverse-set")
    ap.add_argument("--complexes", nargs="+", required=True)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.3, 0.6, 1.0])
    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--num_steps", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--phys_power", type=float, default=2.0)
    ap.add_argument("--phys_start_t", type=float, default=0.3)
    ap.add_argument("--base_seed", type=int, default=1000)
    ap.add_argument("--checkpoint", default="weights/best.pt")
    ap.add_argument("--config", default="configs/train_v3_b200.yaml")
    ap.add_argument("--smiles_map", default="data/astex_smiles.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    smiles_map = json.loads(Path(args.smiles_map).read_text()) if Path(args.smiles_map).exists() else {}

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = yaml.safe_load(open(args.config))
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "model_type"}
    model = UnifiedFlowFrag(**model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)

    results: dict[str, dict[float, list[float]]] = {}
    data_dir = Path(args.data_dir)
    t0 = time.time()

    for pid in args.complexes:
        d = data_dir / pid
        # Prefer full protein
        prot = d / f"{pid}_protein.pdb"
        if not prot.exists():
            prot = d / f"{pid}_pocket.pdb"
        lig_sdf = d / f"{pid}_ligand.sdf"
        lig_mol2 = d / f"{pid}_ligand.mol2"
        if lig_sdf.exists():
            mol_ref = load_sdf_robust(lig_sdf)
        else:
            mol_ref = load_mol2_robust(lig_mol2)
        ext = smiles_map.get(pid, {}) if smiles_map else {}
        smi = ext.get("smiles") if isinstance(ext, dict) else None
        if not smi:
            smi = Chem.MolToSmiles(mol_ref)
        from rdkit import RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            RDLogger.EnableLog("rdApp.*")
            print(f"{pid}: SMILES parse failed ({smi}), skipping")
            continue
        mol_h = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        if status != 0:
            RDLogger.EnableLog("rdApp.*")
            print(f"{pid}: ETKDG embed failed, skipping")
            continue
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        mol = Chem.RemoveHs(mol_h)
        RDLogger.EnableLog("rdApp.*")

        _, lig_ref, meta_ref = preprocess_complex(prot, mol_ref)
        pocket_center = meta_ref["pocket_center"]
        dock_idx, ref_idx, match_how = match_atoms(mol_ref, mol)
        if not dock_idx:
            print(f"{pid}: atom match failed, skipping")
            continue
        graph, lig_data, meta = preprocess_complex(prot, mol, pocket_center=pocket_center)

        phys = None  # built per lambda when > 0
        results[pid] = {}

        for lam in args.lambdas:
            rmsds = []
            if lam > 0.0:
                from src.scoring.physics_guidance import PhysicsGuidance
                phys = PhysicsGuidance(
                    mol=mol,
                    pocket_pdb=str(prot),
                    pocket_center=pocket_center,
                    device=device,
                    pocket_cutoff=cfg["data"].get("pocket_cutoff", 8.0),
                )
            else:
                phys = None
            torch.manual_seed(args.base_seed)
            batch_out = sample_unified(
                model, graph, lig_data, meta,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                translation_sigma=args.sigma,
                time_schedule="late", schedule_power=3.0, device=device,
                phys_guidance=phys, phys_lambda_max=lam,
                phys_power=args.phys_power, phys_start_t=args.phys_start_t,
            )
            for out in batch_out:
                rmsd = compute_rmsd_symm(mol, out["atom_pos_pred"], pocket_center, mol_ref, dock_idx, ref_idx)
                rmsds.append(rmsd)
            results[pid][lam] = rmsds
            arr = np.array(rmsds)
            print(f"  {pid} λ={lam:>4.2f}  oracle={arr.min():.3f}  median={np.median(arr):.3f}  mean={arr.mean():.3f}")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.0f}s\n")

    # Summary table
    hdr = f"{'pdb':<6}  " + "  ".join(f"λ={lam:>4.2f} oracle/median" for lam in args.lambdas)
    print(hdr)
    print("-" * len(hdr))
    agg = {lam: [] for lam in args.lambdas}
    for pid, d in results.items():
        parts = [f"{pid:<6}  "]
        for lam in args.lambdas:
            rs = np.array(d[lam])
            parts.append(f"{rs.min():6.2f}/{np.median(rs):6.2f}     ")
            agg[lam].append((float(rs.min()), float(np.median(rs))))
        print("".join(parts))

    print()
    print("Aggregate across complexes (oracle mean / median mean):")
    for lam in args.lambdas:
        a = np.array(agg[lam])
        print(f"  λ={lam:>4.2f}: oracle {a[:,0].mean():.3f}  median {a[:,1].mean():.3f}")


if __name__ == "__main__":
    main()
