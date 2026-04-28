#!/usr/bin/env python
"""Autonomous experiment runner for the confidence head.

For each config in EXPERIMENTS:
  1. Train (scripts/train_confidence.py) with specified hyperparameters
  2. Pick best ckpt by CASF val sel_mean
  3. Evaluate best ckpt on Astex stoch/det + PB stoch/det
     (uses per-atom feature cache from outputs/conf_benchmark_features/)
  4. Compute Vina-only selection baseline on the same benchmark poses
  5. Sweep score-combination α weights:
       combined(α) = α * normalize(pred_rmsd) + (1-α) * normalize(-vina)
     Record best α + resulting metrics per dataset
  6. Append structured row to experiments.jsonl
  7. git add experiments.jsonl && git commit -m "exp[name]: ..."

Runs sequentially to avoid GPU contention.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.confidence_features import extract_per_atom_features
from src.inference.evaluation import (
    compute_pose_rmsd, detect_complex_files, load_ligand, match_atoms,
)
from src.inference.preprocess import preprocess_complex
from src.models.confidence import ConfidenceHead
from src.models.unified import UnifiedFlowFrag
from src.scoring.vina import (
    compute_pocket_features_from_pdb, compute_vina_features,
    precompute_interaction_matrices, vina_scoring,
)
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors


POSE_STATS_KEYS = ["pose_v_mean", "pose_v_max", "pose_v_p95",
                   "pose_w_mean", "pose_w_max", "pose_w_p95"]


EXPERIMENTS = [
    # Archite sweep
    dict(name="arch.big.mean_max",  hidden=512, trunk=4, head=3, pool="mean_max",  q=4,  dropout=0.2, lr=3e-4, muon_lr=0.02, w_rank=1.0, steps=8000),
    dict(name="arch.big.attention", hidden=512, trunk=4, head=3, pool="attention", q=4,  dropout=0.2, lr=3e-4, muon_lr=0.02, w_rank=1.0, steps=8000),
    dict(name="arch.big.both",      hidden=512, trunk=4, head=3, pool="both",      q=4,  dropout=0.2, lr=3e-4, muon_lr=0.02, w_rank=1.0, steps=8000),
    dict(name="arch.attn.q8",       hidden=512, trunk=4, head=3, pool="attention", q=8,  dropout=0.2, lr=3e-4, muon_lr=0.02, w_rank=1.0, steps=8000),
    dict(name="arch.attn.q16",      hidden=512, trunk=4, head=3, pool="attention", q=16, dropout=0.2, lr=3e-4, muon_lr=0.02, w_rank=1.0, steps=8000),
    dict(name="arch.xl.both",       hidden=1024,trunk=5, head=4, pool="both",      q=8,  dropout=0.2, lr=3e-4, muon_lr=0.02, w_rank=1.0, steps=8000),
    # Regularization
    dict(name="reg.dropout.1",      hidden=512, trunk=4, head=3, pool="both",      q=4,  dropout=0.1, lr=3e-4, muon_lr=0.02, w_rank=1.0, steps=8000),
    dict(name="reg.dropout.3",      hidden=512, trunk=4, head=3, pool="both",      q=4,  dropout=0.3, lr=3e-4, muon_lr=0.02, w_rank=1.0, steps=8000),
    # Loss weights
    dict(name="loss.rank.heavy",    hidden=512, trunk=4, head=3, pool="both",      q=4,  dropout=0.2, lr=3e-4, muon_lr=0.02, w_rank=3.0, steps=8000),
    dict(name="loss.pose.only",     hidden=512, trunk=4, head=3, pool="both",      q=4,  dropout=0.2, lr=3e-4, muon_lr=0.02, w_rank=1.0, steps=8000, w_atom=0.0, w_plddt=0.0, w_frag=0.0, w_frag_bad=0.0),
    # Training schedule
    dict(name="train.long",         hidden=512, trunk=4, head=3, pool="both",      q=4,  dropout=0.2, lr=3e-4, muon_lr=0.02, w_rank=1.0, steps=20000),
    dict(name="train.lr.high",      hidden=512, trunk=4, head=3, pool="both",      q=4,  dropout=0.2, lr=1e-3, muon_lr=0.04, w_rank=1.0, steps=8000),
    dict(name="train.lr.low",       hidden=512, trunk=4, head=3, pool="both",      q=4,  dropout=0.2, lr=1e-4, muon_lr=0.01, w_rank=1.0, steps=8000),
]


BENCHMARKS = [
    ("astex_stoch",
     Path("/mnt/data/PLI/Astex-diverse-set"),
     Path("outputs/eval_astex_phys_25s/poses_phys"),
     Path("data/external_test/astex_smiles.json")),
    ("astex_det",
     Path("/mnt/data/PLI/Astex-diverse-set"),
     Path("outputs/eval_astex_phys_25/poses_phys"),
     Path("data/external_test/astex_smiles.json")),
    ("pb_stoch",
     Path("/mnt/data/PLI/PoseBusters/posebusters_benchmark_set"),
     Path("outputs/eval_posebusters_phys_25s/poses_phys"),
     Path("data/external_test/pb_smiles.json")),
    ("pb_det",
     Path("/mnt/data/PLI/PoseBusters/posebusters_benchmark_set"),
     Path("outputs/eval_posebusters_phys_25/poses_phys"),
     Path("data/external_test/pb_smiles.json")),
]


# ---------------------------------------------------------------------------
def run_train(exp: dict, out_dir: Path) -> None:
    import os
    uv_bin = os.path.expanduser("~/.local/bin/uv")
    cmd = [
        uv_bin, "run", "python", "scripts/train_confidence.py",
        "--out_dir", str(out_dir),
        "--run_name", str(exp["name"]),
        "--hidden", str(exp["hidden"]),
        "--trunk_depth", str(exp["trunk"]),
        "--head_depth", str(exp["head"]),
        "--dropout", str(exp["dropout"]),
        "--pool_mode", str(exp["pool"]),
        "--n_pool_queries", str(exp["q"]),
        "--total_steps", str(exp["steps"]),
        "--val_every", "500",
        "--lr", str(exp["lr"]),
        "--muon_lr", str(exp["muon_lr"]),
        "--w_rank", str(exp["w_rank"]),
    ]
    for k in ("w_atom", "w_plddt", "w_frag", "w_frag_bad", "w_pose", "w_pose_prob"):
        if k in exp:
            cmd += [f"--{k}", str(exp[k])]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    with open(log_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)


def parse_best_ckpt(train_log: Path) -> tuple[Path, float] | None:
    """Return (best ckpt path, CASF val sel_mean)."""
    for line in reversed(train_log.read_text().splitlines()):
        if line.startswith("Best: step"):
            # "Best: step 1500 mean=1.779 → outputs/.../big_step001500.pt"
            parts = line.split()
            mean = float(parts[3].split("=")[1])
            path = Path(parts[-1])
            return path, mean
    return None


# ---------------------------------------------------------------------------
def extract_benchmark_features_and_vina(
    data_dir: Path, poses_dir: Path, smiles_map: dict,
    main_model, device: torch.device, cache_dir: Path,
) -> list[dict]:
    """Load/compute per-atom features + Vina scores + true RMSDs. Cached."""
    cache_path = cache_dir / "features_with_vina.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    pose_files = sorted(poses_dir.glob("*.pt"))
    rows = []
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

            dock_idx, _, _ = match_atoms(mol_ref, mol)
            if not dock_idx:
                continue

            saved = torch.load(pf, map_location="cpu", weights_only=False)
            raw_poses = saved["raw_poses"]
            ref_pos = saved["ref_pos"]
            pocket_center = saved["pocket_center"]

            graph, lig_data, meta = preprocess_complex(
                pocket_pdb, mol, pocket_center=pocket_center,
            )
            n_atoms = meta["num_atom"]
            feats = extract_per_atom_features(
                main_model, graph, lig_data, meta, raw_poses,
                torch.zeros(n_atoms, 3), device=device, t_eval=1.0,
            )
            true_rmsds = [
                compute_pose_rmsd(p, ref_pos, pocket_center, dock_idx, mol, mol_ref)
                for p in raw_poses
            ]

            # Vina per pose (cutoff 15Å)
            pocket_features, pocket_coords = compute_pocket_features_from_pdb(
                str(pocket_pdb), device, center=pocket_center, cutoff=15.0,
            )
            lig_features = compute_vina_features(mol, device)
            precomputed = precompute_interaction_matrices(lig_features, pocket_features, device)
            num_rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
            vina_scores = []
            for p in raw_poses:
                v = vina_scoring(
                    (p + pocket_center).to(device), pocket_coords, precomputed,
                    weight_preset="vina", num_rotatable_bonds=num_rb,
                ).item()
                vina_scores.append(v)

            pose_stats = np.stack([feats[k] for k in POSE_STATS_KEYS], axis=1).astype(np.float32)
            rows.append({
                "pid": pid,
                "n_atoms": int(feats["pose_n_atoms"]),
                "n_frags": int(feats["pose_n_frags"]),
                "atom_scalar": feats["atom_scalar"],
                "atom_norms": feats["atom_norms"],
                "pose_stats": pose_stats,
                "atom_frag_id_local": lig_data["fragment_id"].numpy().astype(np.int32),
                "true_rmsds": np.asarray(true_rmsds, dtype=np.float32),
                "vina_scores": np.asarray(vina_scores, dtype=np.float32),
            })
        except Exception as e:
            print(f"    skip {pid}: {e}")
            continue
    RDLogger.EnableLog("rdApp.*")

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(rows, cache_path)
    return rows


def score_ckpt_on_rows(ckpt_path: Path, rows: list[dict], device: torch.device) -> dict:
    """Return per-complex predicted pose RMSDs + pure-confidence selection summary."""
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

    preds_by_complex: list[np.ndarray] = []
    with torch.no_grad():
        for r in rows:
            atom_scalar = (torch.from_numpy(r["atom_scalar"]).to(device) - scalar_mu) / scalar_sd
            atom_norms = (torch.from_numpy(r["atom_norms"]).to(device) - norms_mu) / norms_sd
            pose_stats = (torch.from_numpy(r["pose_stats"]).to(device) - ps_mu) / ps_sd
            B = r["pose_stats"].shape[0]
            n_atoms = r["n_atoms"]
            atom_pose_ptr = torch.arange(B + 1, device=device, dtype=torch.long) * n_atoms
            local_fid = torch.from_numpy(r["atom_frag_id_local"]).to(device).long()
            atom_frag_id = torch.cat([local_fid + b * r["n_frags"] for b in range(B)])

            out = head(atom_scalar, atom_norms, atom_pose_ptr, atom_frag_id, pose_stats=pose_stats)
            preds_by_complex.append(out["pose_rmsd"].cpu().numpy())
    return {"preds": preds_by_complex}


def metrics_from_selection(true_rmsd_by_cx: list[np.ndarray], sel_idx_by_cx: list[int]) -> dict:
    sel = np.asarray([true_rmsd_by_cx[i][k] for i, k in enumerate(sel_idx_by_cx)])
    return {
        "n": int(len(sel)),
        "mean": float(sel.mean()),
        "med": float(np.median(sel)),
        "lt1": float((sel < 1).mean() * 100),
        "lt2": float((sel < 2).mean() * 100),
        "lt5": float((sel < 5).mean() * 100),
    }


def sweep_alpha(
    preds_by_cx: list[np.ndarray],
    vina_by_cx: list[np.ndarray],
    true_rmsd_by_cx: list[np.ndarray],
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[float, dict]:
    """Find α minimizing mean selection RMSD.  0=pure vina, 1=pure confidence."""
    best_alpha: float = 0.0
    best_mean: float = float("inf")
    best_metrics: dict = {}
    for a in alphas:
        sel = []
        for p, v, t in zip(preds_by_cx, vina_by_cx, true_rmsd_by_cx):
            # normalize per-complex (z-score)
            pz = (p - p.mean()) / (p.std() + 1e-6)
            vz = (v - v.mean()) / (v.std() + 1e-6)
            combined = a * pz + (1 - a) * vz   # lower is better (vina already lower=better)
            sel.append(t[int(np.argmin(combined))])
        sel = np.asarray(sel)
        m = float(sel.mean())
        if m < best_mean:
            best_mean = m
            best_alpha = a
            best_metrics = {
                "mean": m, "med": float(np.median(sel)),
                "lt1": float((sel < 1).mean() * 100),
                "lt2": float((sel < 2).mean() * 100),
                "lt5": float((sel < 5).mean() * 100),
            }
    return best_alpha, best_metrics


def git_commit_experiment(exp_name: str, metrics: dict) -> None:
    """Add experiments.jsonl and commit with concise summary."""
    subprocess.run(["git", "add", "experiments.jsonl"], check=False)
    casf = metrics.get("casf_val_best", float("nan"))
    asx_s = metrics.get("bench", {}).get("astex_stoch", {}).get("conf_mean", float("nan"))
    asx_d = metrics.get("bench", {}).get("astex_det", {}).get("conf_mean", float("nan"))
    pb_s = metrics.get("bench", {}).get("pb_stoch", {}).get("conf_mean", float("nan"))
    pb_d = metrics.get("bench", {}).get("pb_det", {}).get("conf_mean", float("nan"))
    msg = (
        f"exp[{exp_name}]: "
        f"CASF={casf:.3f}  "
        f"AsSt={asx_s:.3f} AsDt={asx_d:.3f}  "
        f"PbSt={pb_s:.3f} PbDt={pb_d:.3f}"
    )
    subprocess.run(["git", "commit", "-m", msg], check=False)


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs/auto_exp")
    ap.add_argument("--main_config", default="configs/train_v3_b200.yaml")
    ap.add_argument("--main_checkpoint", default="weights/best.pt")
    ap.add_argument("--cache_dir", default="outputs/conf_benchmark_features_vina")
    ap.add_argument("--results_jsonl", default="experiments.jsonl")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Run only first N experiments")
    ap.add_argument("--no_git_commit", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-extract benchmark features + Vina scores (shared across all experiments)
    cfg = yaml.safe_load(open(args.main_config))
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "model_type"}
    main_model = UnifiedFlowFrag(**model_cfg).to(device)
    mc = torch.load(args.main_checkpoint, map_location=device, weights_only=False)
    main_model.load_state_dict(mc["model_state_dict"])
    main_model.train(False)
    print(f"[auto] Main model loaded: {args.main_checkpoint}")

    bench_data: dict[str, list[dict]] = {}
    for name, data_dir, poses_dir, smiles_map_path in BENCHMARKS:
        if not poses_dir.exists():
            continue
        smiles_map = json.loads(smiles_map_path.read_text()) if smiles_map_path.exists() else {}
        print(f"[auto] Loading bench features for {name} ...")
        cache_dir = Path(args.cache_dir) / name
        rows = extract_benchmark_features_and_vina(
            data_dir, poses_dir, smiles_map, main_model, device, cache_dir,
        )
        bench_data[name] = rows
        print(f"   {len(rows)} complexes")

    # Done features; free main_model from GPU
    del main_model
    torch.cuda.empty_cache()

    exps = EXPERIMENTS
    if args.limit:
        exps = exps[: args.limit]

    out_root = Path(args.out_dir)
    results_jsonl = Path(args.results_jsonl)

    for exp in exps:
        exp_dir = out_root / exp["name"]
        if args.skip_existing and (exp_dir / "train.log").exists():
            print(f"\n[auto] SKIP {exp['name']} (already present)")
            continue

        print(f"\n{'='*60}\n[auto] RUN {exp['name']}\n{'='*60}")
        t0 = time.time()
        try:
            run_train(exp, exp_dir)
        except subprocess.CalledProcessError as e:
            print(f"   TRAIN FAILED: {e}")
            append_jsonl(results_jsonl, {"name": exp["name"], "error": str(e), "config": exp})
            continue

        parsed = parse_best_ckpt(exp_dir / "train.log")
        if parsed is None:
            print("   parse best ckpt failed")
            continue
        best_ckpt, casf_mean = parsed
        print(f"   best ckpt: {best_ckpt}  CASF val mean={casf_mean:.3f}")

        # Bench eval
        bench_metrics: dict[str, dict] = {}
        for bname, rows in bench_data.items():
            preds = score_ckpt_on_rows(best_ckpt, rows, device)["preds"]
            true_rmsds = [r["true_rmsds"] for r in rows]
            vina = [r["vina_scores"] for r in rows]

            # Pure confidence
            sel_c = [int(np.argmin(p)) for p in preds]
            m_c = metrics_from_selection(true_rmsds, sel_c)

            # Pure vina
            sel_v = [int(np.argmin(v)) for v in vina]
            m_v = metrics_from_selection(true_rmsds, sel_v)

            # Alpha sweep
            best_a, m_combo = sweep_alpha(preds, vina, true_rmsds)

            bench_metrics[bname] = {
                "conf_mean": m_c["mean"], "conf_lt2": m_c["lt2"],
                "vina_mean": m_v["mean"], "vina_lt2": m_v["lt2"],
                "combo_alpha": best_a, "combo_mean": m_combo["mean"], "combo_lt2": m_combo["lt2"],
                "full_conf": m_c, "full_vina": m_v, "full_combo": m_combo,
            }
            print(
                f"   {bname:<14}"
                f"  conf={m_c['mean']:.3f} (<2={m_c['lt2']:.0f}%)"
                f"  vina={m_v['mean']:.3f} (<2={m_v['lt2']:.0f}%)"
                f"  combo(α={best_a})={m_combo['mean']:.3f} (<2={m_combo['lt2']:.0f}%)"
            )

        dt = time.time() - t0
        row = {
            "name": exp["name"],
            "config": {k: v for k, v in exp.items() if k != "name"},
            "best_ckpt": str(best_ckpt),
            "casf_val_best": casf_mean,
            "bench": bench_metrics,
            "elapsed_seconds": dt,
        }
        append_jsonl(results_jsonl, row)
        print(f"   recorded → {results_jsonl}  ({dt/60:.1f} min total)")

        if not args.no_git_commit:
            git_commit_experiment(exp["name"], row)
            print("   git committed")


if __name__ == "__main__":
    main()
