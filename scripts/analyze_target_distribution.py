"""Measure the empirical distribution of crystal fragment centers relative to
the pocket center, across the full training split.

T_1 in training = ligand["frag_centers"] - meta["pocket_center"]

We want to pick prior_sigma to match this target distribution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    root = Path("data/processed")
    split_file = Path("data/splits/pdbbind2020.json")

    with open(split_file) as f:
        pdb_ids = json.load(f)["train"]

    filters = dict(min_atoms=5, max_atoms=80, max_frags=20, min_protein_res=50)

    per_axis: list[np.ndarray] = []   # [K, 3] per-fragment offsets
    norms: list[float] = []           # ‖offset‖ per fragment
    n_skipped = 0
    for pid in pdb_ids:
        d = root / pid
        if not all((d / f).exists() for f in ("ligand.pt", "meta.pt")):
            n_skipped += 1
            continue
        meta = torch.load(d / "meta.pt", weights_only=True)
        n_atom = meta["num_atom"].item()
        n_frag = meta["num_frag"].item()
        n_res = meta["num_res"].item()
        if (n_atom < filters["min_atoms"] or n_atom > filters["max_atoms"]
                or n_frag > filters["max_frags"] or n_res < filters["min_protein_res"]):
            n_skipped += 1
            continue

        lig = torch.load(d / "ligand.pt", weights_only=True)
        pc = meta["pocket_center"]
        offsets = (lig["frag_centers"] - pc).numpy()   # [n_frag, 3]
        per_axis.append(offsets)
        norms.extend(np.linalg.norm(offsets, axis=1).tolist())

    arr = np.concatenate(per_axis, axis=0)   # [N_total_frag, 3]
    n = arr.shape[0]

    axis_std = arr.std(axis=0)          # per-axis std
    axis_std_iso = arr.reshape(-1).std() # treating all components as iid
    norm_arr = np.asarray(norms)

    print(f"Complexes scanned: {len(pdb_ids) - n_skipped} (skipped {n_skipped})")
    print(f"Total fragments: {n}")
    print()
    print("=== Per-axis std of T_1 (x, y, z) ===")
    print(f"  std  = [{axis_std[0]:.3f}, {axis_std[1]:.3f}, {axis_std[2]:.3f}]  Å")
    print(f"  iso  = {axis_std_iso:.3f}  Å  (all axes pooled)")
    print()
    print("=== ‖T_1‖ distribution (Euclidean, per fragment) ===")
    for p in [50, 75, 90, 95, 99]:
        print(f"  p{p:02d}: {np.percentile(norm_arr, p):.2f} Å")
    print(f"  max : {norm_arr.max():.2f} Å")
    print(f"  mean: {norm_arr.mean():.2f} Å")
    print()
    print("=== Prior comparison (N(0, σ²I)) ===")
    for sigma in [2.0, 3.0, 4.0, 5.0]:
        # For 3D Gaussian, E[‖X‖] = σ·sqrt(8/π) ≈ 1.596σ, median ≈ σ·sqrt(chi2(3,.5))≈1.538σ
        emean = sigma * np.sqrt(8 / np.pi)
        emed = sigma * np.sqrt(2.366)
        # Fraction of target fragments within typical prior mass (2σ)
        frac_inside_2sigma = float((norm_arr < 2 * sigma).mean())
        print(f"  σ={sigma:.1f}:  E[‖T_0‖]≈{emean:.2f}Å  median≈{emed:.2f}Å  "
              f"target‖T_1‖<2σ: {frac_inside_2sigma*100:.1f}%")


if __name__ == "__main__":
    main()
