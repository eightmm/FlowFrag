#!/usr/bin/env python
"""Evaluate saved confidence checkpoints on Astex / PoseBusters benchmark poses.

For each benchmark complex:
  1. Re-preprocess via preprocess_complex (matching eval_benchmark)
  2. Load cached raw_poses + ref_pos from outputs/eval_*/poses_phys/{pid}.pt
  3. Run extract_per_atom_features (atom_scalar/norms + pose_stats); atom_disp
     labels are ignored — actual RMSD is computed via compute_pose_rmsd.
  4. For every checkpoint in --ckpt_dir: load confidence head, score all poses,
     pick argmin(pred_pose_rmsd), record true RMSD of the selected pose.

Reports mean / median / <1Å / <2Å / <5Å per (dataset, checkpoint).
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
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.confidence_features import extract_per_atom_features
from src.inference.evaluation import (
    compute_pose_rmsd, detect_complex_files, load_ligand, match_atoms,
)
from src.inference.preprocess import preprocess_complex
from src.models.confidence import ConfidenceHead
from src.models.unified import UnifiedFlowFrag


POSE_STATS_KEYS = ["pose_v_mean", "pose_v_max", "pose_v_p95",
                   "pose_w_mean", "pose_w_max", "pose_w_p95"]


# ---------------------------------------------------------------------------
# Per-complex feature extraction (cached on disk)
# ---------------------------------------------------------------------------
def extract_benchmark_features(
    data_dir: Path, poses_dir: Path, smiles_map: dict,
    main_model, device: torch.device,
    cache_dir: Path,
) -> list[dict]:
    """Extract per-atom features + true per-pose RMSDs for each cached complex.

    Returns list of dicts: {pid, atom_scalar, atom_norms, pose_stats, true_rmsds}.
    Results are cached to `cache_dir/features.pt` for re-use across checkpoints.
    """
    cache_path = cache_dir / "features.pt"
    if cache_path.exists():
        print(f"  loading cached features from {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    pose_files = sorted(poses_dir.glob("*.pt"))
    print(f"  extracting features for {len(pose_files)} complexes ...")
    rows = []
    t0 = time.time()

    RDLogger.DisableLog("rdApp.*")
    for i, pf in enumerate(pose_files):
        pid = pf.stem
        complex_dir = data_dir / pid
        detected = detect_complex_files(complex_dir, pid)
        if detected is None:
            continue
        pocket_pdb, ligand_file, fmt = detected

        try:
            mol_ref = load_ligand(ligand_file, fmt)
            ext = smiles_map.get(pid, {})
            smi = ext.get("smiles") if isinstance(ext, dict) else None
            if smi:
                mol = Chem.MolFromSmiles(smi)
                mol = Chem.AddHs(mol)
                if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
                    continue
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                mol = Chem.RemoveHs(mol)
            else:
                mol = mol_ref

            dock_idx, ref_idx, match_how = match_atoms(mol_ref, mol)
            if not dock_idx:
                continue

            saved = torch.load(pf, map_location="cpu", weights_only=False)
            raw_poses = saved["raw_poses"]
            ref_pos = saved["ref_pos"]
            pocket_center = saved["pocket_center"]

            graph, lig_data, meta = preprocess_complex(
                pocket_pdb, mol, pocket_center=pocket_center,
            )

            # atom_disp labels are not needed here — pass dummy crystal (zeros).
            # We compute real RMSD afterwards via compute_pose_rmsd.
            n_atoms = meta["num_atom"]
            dummy_crystal = torch.zeros(n_atoms, 3)

            feats = extract_per_atom_features(
                main_model, graph, lig_data, meta, raw_poses,
                dummy_crystal, device=device, t_eval=1.0,
            )

            true_rmsds = [
                compute_pose_rmsd(p, ref_pos, pocket_center, dock_idx, mol, mol_ref)
                for p in raw_poses
            ]

            pose_stats = np.stack([feats[k] for k in POSE_STATS_KEYS], axis=1)
            rows.append({
                "pid": pid,
                "n_atoms": int(feats["pose_n_atoms"]),
                "n_frags": int(feats["pose_n_frags"]),
                "atom_scalar": feats["atom_scalar"],
                "atom_norms": feats["atom_norms"],
                "pose_stats": pose_stats.astype(np.float32),
                "atom_frag_id_local": lig_data["fragment_id"].numpy().astype(np.int32),
                "true_rmsds": np.asarray(true_rmsds, dtype=np.float32),
            })
        except Exception as e:
            print(f"    skip {pid}: {e}")
            continue

        if (i + 1) % 20 == 0:
            dt = time.time() - t0
            print(f"    {i+1}/{len(pose_files)} elapsed={dt:.0f}s")
    RDLogger.EnableLog("rdApp.*")

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(rows, cache_path)
    print(f"  saved features → {cache_path}  ({len(rows)} complexes)")
    return rows


# ---------------------------------------------------------------------------
# Checkpoint evaluation
# ---------------------------------------------------------------------------
def score_checkpoint(
    ckpt_path: Path, rows: list[dict], device: torch.device,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    head = ConfidenceHead(
        scalar_dim=ckpt["scalar_dim"], norms_dim=ckpt["norms_dim"],
        pose_stats_dim=ckpt["pose_stats_dim"],
        hidden=ckpt["hidden"], trunk_depth=ckpt["trunk_depth"],
        head_depth=ckpt["head_depth"], dropout=ckpt["dropout"],
        pool_mode=ckpt.get("pool_mode", "mean_max"),
        n_pool_queries=ckpt.get("n_pool_queries", 4),
    ).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.train(False)

    scalar_mu = torch.from_numpy(ckpt["scalar_mu"]).to(device)
    scalar_sd = torch.from_numpy(ckpt["scalar_sd"]).to(device)
    norms_mu = torch.from_numpy(ckpt["norms_mu"]).to(device)
    norms_sd = torch.from_numpy(ckpt["norms_sd"]).to(device)
    ps_mu = torch.from_numpy(ckpt["pose_stats_mu"]).to(device)
    ps_sd = torch.from_numpy(ckpt["pose_stats_sd"]).to(device)

    sel_rmsds = []
    with torch.no_grad():
        for r in rows:
            # Normalize features + move to device
            atom_scalar = (torch.from_numpy(r["atom_scalar"]).to(device) - scalar_mu) / scalar_sd
            atom_norms = (torch.from_numpy(r["atom_norms"]).to(device) - norms_mu) / norms_sd
            pose_stats = (torch.from_numpy(r["pose_stats"]).to(device) - ps_mu) / ps_sd

            B = r["pose_stats"].shape[0]
            n_atoms = r["n_atoms"]
            atom_pose_ptr = torch.arange(B + 1, device=device, dtype=torch.long) * n_atoms
            local_fid = torch.from_numpy(r["atom_frag_id_local"]).to(device).long()
            atom_frag_id = torch.cat([
                local_fid + b * r["n_frags"] for b in range(B)
            ])

            out = head(atom_scalar, atom_norms, atom_pose_ptr, atom_frag_id,
                       pose_stats=pose_stats)
            pred = out["pose_rmsd"].cpu().numpy()
            i = int(np.argmin(pred))
            sel_rmsds.append(r["true_rmsds"][i])

    sel = np.asarray(sel_rmsds)
    return {
        "n": len(sel),
        "mean": float(sel.mean()),
        "med": float(np.median(sel)),
        "lt1": float((sel < 1).mean() * 100),
        "lt2": float((sel < 2).mean() * 100),
        "lt5": float((sel < 5).mean() * 100),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs/conf_models")
    ap.add_argument("--ckpt_glob", default="head_16k_step*.pt")
    ap.add_argument("--main_config", default="configs/train_v3_b200.yaml")
    ap.add_argument("--main_checkpoint", default="weights/best.pt")
    ap.add_argument("--cache_dir", default="outputs/conf_benchmark_features",
                    help="Where to cache per-atom features per benchmark")
    ap.add_argument("--out_json", default="outputs/conf_benchmark_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load main model (for per-atom feature extraction)
    cfg = yaml.safe_load(open(args.main_config))
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "model_type"}
    main_model = UnifiedFlowFrag(**model_cfg).to(device)
    mc = torch.load(args.main_checkpoint, map_location=device, weights_only=False)
    main_model.load_state_dict(mc["model_state_dict"])
    main_model.train(False)
    print(f"Main model loaded: {args.main_checkpoint}  step={mc.get('step','?')}")

    # Benchmarks to evaluate
    benchmarks = [
        ("astex_stoch",
         Path("/mnt/data/PLI/Astex-diverse-set"),
         Path("outputs/eval_astex_phys_25s/poses_phys"),
         Path("data/astex_smiles.json")),
        ("astex_det",
         Path("/mnt/data/PLI/Astex-diverse-set"),
         Path("outputs/eval_astex_phys_25/poses_phys"),
         Path("data/astex_smiles.json")),
        ("pb_stoch",
         Path("/mnt/data/PLI/PoseBusters/posebusters_benchmark_set"),
         Path("outputs/eval_posebusters_phys_25s/poses_phys"),
         Path("data/pb_smiles.json")),
        ("pb_det",
         Path("/mnt/data/PLI/PoseBusters/posebusters_benchmark_set"),
         Path("outputs/eval_posebusters_phys_25/poses_phys"),
         Path("data/pb_smiles.json")),
    ]

    # Extract features for each benchmark (cached)
    bench_features: dict[str, list[dict]] = {}
    for name, data_dir, poses_dir, smiles_map_path in benchmarks:
        if not poses_dir.exists() or not data_dir.exists():
            print(f"SKIP {name}: missing {poses_dir} or {data_dir}")
            continue
        smiles_map = json.loads(smiles_map_path.read_text()) if smiles_map_path.exists() else {}
        print(f"\n[{name}] extracting features ...")
        cache_dir = Path(args.cache_dir) / name
        rows = extract_benchmark_features(
            data_dir, poses_dir, smiles_map, main_model, device, cache_dir,
        )
        bench_features[name] = rows

    # Evaluate each checkpoint
    ckpt_dir = Path(args.ckpt_dir)
    ckpts = sorted(ckpt_dir.glob(args.ckpt_glob))
    print(f"\nFound {len(ckpts)} checkpoints in {ckpt_dir}")

    all_results: dict[str, dict[str, dict]] = {}   # ckpt → dataset → metrics
    for ck in ckpts:
        print(f"\n=== {ck.name} ===")
        all_results[ck.name] = {}
        for name, rows in bench_features.items():
            r = score_checkpoint(ck, rows, device)
            all_results[ck.name][name] = r
            print(f"  {name:<14}  mean={r['mean']:.3f}  med={r['med']:.3f}  "
                  f"<1={r['lt1']:5.1f}%  <2={r['lt2']:5.1f}%  <5={r['lt5']:5.1f}%  "
                  f"(n={r['n']})")

    # Summary table
    print("\n\n=== Summary (val mean RMSD per ckpt × dataset) ===")
    if all_results:
        datasets = list(next(iter(all_results.values())).keys())
        header = f"{'ckpt':<30}  " + "  ".join(f"{d:<14}" for d in datasets)
        print(header)
        print("-" * len(header))
        for ck_name, by_ds in all_results.items():
            cells = "  ".join(f"{by_ds[d]['mean']:.3f} ({by_ds[d]['lt2']:.0f}%)" if d in by_ds else "-"
                              for d in datasets)
            print(f"{ck_name:<30}  {cells}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
