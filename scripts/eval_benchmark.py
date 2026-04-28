#!/usr/bin/env python
"""Evaluate FlowFrag on docking benchmarks (Astex / PoseBusters).

Auto-detects dataset format from file naming conventions.
Samples N poses once per complex, then evaluates all combinations of:
  - Refinement: none, mmff
  - Selection:  oracle (best-RMSD), vina, optionally confidence

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
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.evaluation import (
    REFINE_METHODS, SELECT_METHODS, V2_IDS_PATH,
    apply_refinement, compute_centroid_dist, compute_pose_rmsd, compute_stats,
    detect_complex_files, detect_dataset_name,
    load_ligand, match_atoms, select_pose,
)
from src.inference.preprocess import preprocess_complex
from src.inference.sampler import sample_unified
from src.scoring.ranking import rank_poses



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
    parser.add_argument("--pocket_cutoff", type=float, default=8.0,
                        help="Residue-aware pocket cutoff (Å). Model trained at 8.0 with 6-10 noise.")
    parser.add_argument("--confidence_ckpt", type=str, default="",
                        help="Path to confidence head checkpoint. When set, adds 'confidence' selector.")
    parser.add_argument("--smiles_init", action="store_true",
                        help="Re-generate 3D conformer via ETKDGv3+MMFF (simulate SMILES input)")
    parser.add_argument("--smiles_map", type=str, default=None,
                        help="Optional JSON file mapping pdb_id -> {smiles, ...}. "
                             "Used with --smiles_init to build 3D from external SMILES "
                             "instead of round-tripping the crystal mol.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: outputs/eval_{dataset_name})")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--phys_guidance", action="store_true",
                        help="Enable Vina-gradient guidance during ODE sampling.")
    parser.add_argument("--phys_lambda_max", type=float, default=0.3)
    parser.add_argument("--phys_power", type=float, default=2.0)
    parser.add_argument("--phys_start_t", type=float, default=0.3)
    parser.add_argument("--phys_max_force", type=float, default=10.0)
    parser.add_argument("--phys_weight_preset", type=str, default="vina",
                        choices=("vina", "vinardo"))
    parser.add_argument("--stochastic_gamma", type=float, default=0.0,
                        help="Annealed noise scale γ for stochastic sampling (0 = deterministic).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"Data directory not found: {data_dir}"
    dataset_name = detect_dataset_name(data_dir)

    if args.out_dir is None:
        args.out_dir = f"outputs/eval_{dataset_name}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    select_methods = (
        SELECT_METHODS if args.confidence_ckpt
        else tuple(s for s in SELECT_METHODS if s != "confidence")
    )

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
    print(f"Evaluating: {len(REFINE_METHODS)} refinements x {len(select_methods)} selections "
          f"= {len(REFINE_METHODS) * len(select_methods)} combos\n")

    # --- Per-combo accumulators ---
    combo_results: dict[tuple[str, str], list[dict]] = {
        (r, s): [] for r in REFINE_METHODS for s in select_methods
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

            poses_subdir = "poses_phys" if args.phys_guidance else "poses"
            poses_dir = out_dir / poses_subdir
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
                    _, lig_ref, meta_ref = preprocess_complex(pocket_pdb, mol_ref, pocket_cutoff=args.pocket_cutoff)
                    pocket_center = meta_ref["pocket_center"]
                    dock_idx, ref_idx, match_how = match_atoms(mol_ref, mol)
                    if not dock_idx:
                        print(f"[{idx+1:3d}/{len(pdb_ids)}] {pdb_id}: SKIP (atom match failed)")
                        failures.append({"pdb_id": pdb_id, "error": "atom match failed"})
                        continue
                    ref_pos = lig_ref["atom_coords"][ref_idx] - pocket_center
                    graph, lig_data, meta = preprocess_complex(pocket_pdb, mol, pocket_center=pocket_center, pocket_cutoff=args.pocket_cutoff)
                else:
                    graph, lig_data, meta = preprocess_complex(pocket_pdb, mol, pocket_cutoff=args.pocket_cutoff)
                    pocket_center = meta["pocket_center"]
                    ref_pos = lig_data["atom_coords"] - pocket_center
                    dock_idx = list(range(int(meta["num_atom"])))
                    match_how = "identity"

                phys = None
                if args.phys_guidance:
                    from src.scoring.physics_guidance import PhysicsGuidance
                    phys = PhysicsGuidance(
                        mol=mol,
                        pocket_pdb=str(pocket_pdb),
                        pocket_center=pocket_center,
                        device=device,
                        # Vina scoring needs a wider pocket than the model's 8Å graph cutoff —
                        # convergence is reached around 12-15Å. Keep it decoupled from cfg.
                        pocket_cutoff=15.0,
                        weight_preset=args.phys_weight_preset,
                        max_force_per_atom=args.phys_max_force,
                    )

                # --- Sample N poses (batched into one forward per ODE step) ---
                results = sample_unified(
                    model, graph, lig_data, meta,
                    num_samples=args.num_samples,
                    num_steps=args.num_steps,
                    translation_sigma=sigma,
                    time_schedule=args.time_schedule,
                    schedule_power=args.schedule_power,
                    device=device,
                    phys_guidance=phys,
                    phys_lambda_max=args.phys_lambda_max if args.phys_guidance else 0.0,
                    phys_power=args.phys_power,
                    phys_start_t=args.phys_start_t,
                    stochastic_gamma=args.stochastic_gamma,
                )
                raw_poses = [r["atom_pos_pred"] for r in results]

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

            # --- Confidence-based selection (run once on raw flow output) ---
            sorted_idx_conf: list[int] = []
            if args.confidence_ckpt:
                if resumed:
                    # Rebuild graph/lig_data/meta — saved poses.pt only has the minimum needed for RMSD.
                    graph, lig_data, meta = preprocess_complex(
                        pocket_pdb, mol, pocket_center=pocket_center,
                        pocket_cutoff=args.pocket_cutoff,
                    )
                from src.inference.confidence_features import score_poses_with_confidence
                _, sorted_idx_conf = score_poses_with_confidence(
                    model, args.confidence_ckpt, graph, lig_data, meta, raw_poses, device,
                )

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

                for select in select_methods:
                    if select == "vina":
                        sel_idx = vina_best_idx
                    elif select == "confidence":
                        if not sorted_idx_conf:
                            continue
                        sel_idx = sorted_idx_conf[0]
                    else:
                        sel_idx = select_pose(select, rmsds)
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
        for select in select_methods:
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
            for r in REFINE_METHODS for s in select_methods
        },
    }

    out_file = out_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
