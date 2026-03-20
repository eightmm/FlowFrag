#!/usr/bin/env python
"""Diagnose whether h_frag contains sufficient rotation information.

Loads a checkpoint, runs forward passes on the overfit set, and analyzes:
1. Magnitude of scalar (0e) vs vector (1o) vs pseudovector (1e) components
2. Per-axis correlation between pred and target
3. Gradient magnitude: how much gradient flows to omega vs v?
4. Variance collapse check
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import FlowFragDataset
from src.models.flowfrag import FlowFrag
from src.geometry.se3 import quaternion_to_matrix


def collect_predictions(model, dataset, device, max_samples=16):
    model.eval()
    all_v_pred, all_omega_pred = [], []
    all_v_target, all_omega_target = [], []
    all_frag_sizes, all_R_t = [], []

    with torch.no_grad():
        for i in range(min(len(dataset), max_samples)):
            data = dataset[i].to(device)
            out = model(data)
            all_v_pred.append(out["v_pred"])
            all_omega_pred.append(out["omega_pred"])
            all_v_target.append(data["fragment"].v_target)
            all_omega_target.append(data["fragment"].omega_target)
            all_frag_sizes.append(data["fragment"].size)
            all_R_t.append(quaternion_to_matrix(data["fragment"].q_frag))

    return {
        "v_pred": torch.cat(all_v_pred),
        "omega_pred": torch.cat(all_omega_pred),
        "v_target": torch.cat(all_v_target),
        "omega_target": torch.cat(all_omega_target),
        "frag_sizes": torch.cat(all_frag_sizes),
        "R_t": torch.cat(all_R_t),
    }


def analyze_predictions(data):
    cos = nn.functional.cosine_similarity
    multi = data["frag_sizes"] > 1
    if not multi.any():
        print("No multi-atom fragments!")
        return

    w_pred = data["omega_pred"][multi]
    w_tgt = data["omega_target"][multi]
    v_pred = data["v_pred"][multi]
    v_tgt = data["v_target"][multi]
    R_t = data["R_t"][multi]

    print("=" * 60)
    print("PREDICTION ANALYSIS (multi-atom fragments only)")
    print("=" * 60)
    n = multi.sum().item()
    print(f"N fragments: {n}")

    cos_v = cos(v_pred, v_tgt, dim=-1)
    cos_w = cos(w_pred, w_tgt, dim=-1)
    print(f"\ncos_v: mean={cos_v.mean():.3f}, std={cos_v.std():.3f}")
    print(f"cos_w: mean={cos_w.mean():.3f}, std={cos_w.std():.3f}")

    w_norm = w_tgt.norm(dim=-1)
    v_norm = v_tgt.norm(dim=-1)
    print(f"\n|omega_target|: mean={w_norm.mean():.3f}, std={w_norm.std():.3f}, max={w_norm.max():.3f}")
    print(f"|v_target|:     mean={v_norm.mean():.3f}, std={v_norm.std():.3f}, max={v_norm.max():.3f}")
    print(f"omega/v ratio:  {w_norm.mean() / (v_norm.mean() + 1e-8):.3f}")

    print(f"\nomega_target per-axis std: x={w_tgt[:,0].std():.3f} y={w_tgt[:,1].std():.3f} z={w_tgt[:,2].std():.3f}")
    print(f"omega_pred   per-axis std: x={w_pred[:,0].std():.3f} y={w_pred[:,1].std():.3f} z={w_pred[:,2].std():.3f}")

    for axis, name in zip(range(3), ["x", "y", "z"]):
        corr_v = torch.corrcoef(torch.stack([v_pred[:, axis], v_tgt[:, axis]]))[0, 1]
        corr_w = torch.corrcoef(torch.stack([w_pred[:, axis], w_tgt[:, axis]]))[0, 1]
        print(f"  axis {name}: corr_v={corr_v:.3f}, corr_w={corr_w:.3f}")

    # Body-frame
    Rt_inv = R_t.transpose(-1, -2)
    w_pred_body = torch.einsum("nij,nj->ni", Rt_inv, w_pred)
    w_tgt_body = torch.einsum("nij,nj->ni", Rt_inv, w_tgt)
    cos_w_body = cos(w_pred_body, w_tgt_body, dim=-1)
    print(f"\nBody-frame cos_w: mean={cos_w_body.mean():.3f}, std={cos_w_body.std():.3f}")

    # Variance collapse check
    pred_var = w_pred.var(dim=0)
    tgt_var = w_tgt.var(dim=0)
    print(f"\nomega_pred variance:   x={pred_var[0]:.4f} y={pred_var[1]:.4f} z={pred_var[2]:.4f}")
    print(f"omega_target variance: x={tgt_var[0]:.4f} y={tgt_var[1]:.4f} z={tgt_var[2]:.4f}")
    ratio = pred_var / (tgt_var + 1e-8)
    print(f"variance ratio:        x={ratio[0]:.3f} y={ratio[1]:.3f} z={ratio[2]:.3f}")

    mean_pred = w_pred.mean(dim=0)
    mean_tgt = w_tgt.mean(dim=0)
    print(f"\nomega_pred mean:   {mean_pred.tolist()}")
    print(f"omega_target mean: {mean_tgt.tolist()}")

    mse_total = ((w_pred - w_tgt) ** 2).mean()
    bias_sq = ((mean_pred - mean_tgt) ** 2).sum()
    print(f"\nMSE total: {mse_total:.4f}")
    print(f"Bias sq:   {bias_sq:.4f}")

    # Magnitude prediction
    w_pred_norm = w_pred.norm(dim=-1)
    corr_mag = torch.corrcoef(torch.stack([w_pred_norm, w_norm]))[0, 1]
    print(f"\nMagnitude correlation: {corr_mag:.3f}")
    print(f"|omega_pred|: mean={w_pred_norm.mean():.3f}, std={w_pred_norm.std():.3f}")


def analyze_gradient_flow(model, dataset, device):
    model.train()
    data = dataset[0].to(device)
    out = model(data)
    multi = data["fragment"].size > 1

    if not multi.any():
        return

    loss_w = ((out["omega_pred"][multi] - data["fragment"].omega_target[multi]) ** 2).mean()
    loss_w.backward(retain_graph=True)

    omega_grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            omega_grad_norms[name] = p.grad.norm().item()
    model.zero_grad()

    loss_v = ((out["v_pred"] - data["fragment"].v_target) ** 2).mean()
    loss_v.backward()

    print("\n" + "=" * 60)
    print("GRADIENT FLOW (top 15 layers by omega gradient)")
    print("=" * 60)

    rows = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            gv = p.grad.norm().item()
            gw = omega_grad_norms.get(name, 0)
            if gv > 1e-7 or gw > 1e-7:
                rows.append((name, gw, gv))

    rows.sort(key=lambda x: -x[1])
    print(f"{'Layer':<55} {'|grad_w|':>10} {'|grad_v|':>10} {'w/v':>8}")
    print("-" * 85)
    for name, gw, gv in rows[:15]:
        ratio = gw / (gv + 1e-10)
        short = name[-53:] if len(name) > 53 else name
        print(f"{short:<55} {gw:>10.6f} {gv:>10.6f} {ratio:>8.3f}")

    total_gw = sum(r[1] for r in rows)
    total_gv = sum(r[2] for r in rows)
    print(f"\n{'TOTAL':<55} {total_gw:>10.4f} {total_gv:>10.4f} {total_gw/(total_gv+1e-10):>8.3f}")
    model.zero_grad()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlowFrag(**cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: {args.checkpoint} (step {ckpt.get('step', '?')})")

    dcfg = cfg["data"]
    ds = FlowFragDataset(
        root=dcfg["data_dir"],
        translation_sigma=dcfg.get("prior_sigma", 1.0),
        max_atoms=dcfg.get("max_atoms", 80),
        max_frags=dcfg.get("max_frags", 20),
        min_atoms=dcfg.get("min_atoms", 5),
        rotation_augmentation=dcfg.get("rotation_augmentation", "none"),
        deterministic_augmentation=dcfg.get("deterministic_augmentation"),
        deterministic_prior=dcfg.get("deterministic_prior"),
        deterministic_time=dcfg.get("deterministic_time"),
        seed=cfg.get("training", {}).get("seed", 42),
    )
    print(f"Dataset: {len(ds)} samples")

    reps = collect_predictions(model, ds, device)
    analyze_predictions(reps)
    analyze_gradient_flow(model, ds, device)


if __name__ == "__main__":
    main()
