"""Probe per-GPU max batch size on B200.

Loads v2 model config, runs a few train-like forward+backward steps at
increasing batch sizes, reports peak memory.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_v3_b200.yaml")
    ap.add_argument("--bs", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    ap.add_argument("--steps", type=int, default=5, help="steps per BS")
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.data.dataset import UnifiedDataset, unified_collate
    from src.models.unified import UnifiedFlowFrag
    from src.training.losses import (
        flow_matching_loss, atom_position_auxiliary_loss, distance_geometry_loss,
    )
    from src.training.trainer import configure_optimizers

    device = torch.device("cuda:0")
    dcfg = cfg["data"]
    mcfg = {k: v for k, v in cfg["model"].items() if k != "model_type"}

    ds = UnifiedDataset(
        root=dcfg["data_dir"],
        split_file=dcfg.get("split_file"),
        split_key="train",
        pocket_cutoff=dcfg.get("pocket_cutoff", 8.0),
        pocket_jitter_sigma=dcfg.get("pocket_jitter_sigma", 0.0),
        pocket_cutoff_noise=dcfg.get("pocket_cutoff_noise", 0.0),
        translation_sigma=dcfg.get("prior_sigma", 5.0),
        max_atoms=dcfg.get("max_atoms", 80),
        max_frags=dcfg.get("max_frags", 20),
        min_atoms=dcfg.get("min_atoms", 5),
        min_protein_res=dcfg.get("min_protein_res", 50),
        rotation_augmentation=dcfg.get("rotation_augmentation", "none"),
        deterministic=False,
        seed=42,
        preload=args.preload,
    )
    print(f"dataset size: {len(ds)}")

    model = UnifiedFlowFrag(**mcfg).to(device)
    opts = configure_optimizers(model, lr=3e-4, muon_lr=0.02, use_muon=True)

    results: list[tuple[int, float, float, bool]] = []
    for bs in args.bs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=True, num_workers=args.workers,
            collate_fn=unified_collate, pin_memory=True, drop_last=True,
            persistent_workers=args.workers > 0,
            prefetch_factor=4 if args.workers > 0 else None,
        )
        try:
            it = iter(loader)
            step_times: list[float] = []
            t0 = time.time()
            for s in range(args.steps):
                t_s = time.time()
                batch = next(it)
                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                for opt in opts:
                    opt.zero_grad(set_to_none=True)
                out = model(batch)
                losses = flow_matching_loss(
                    out["v_pred"], out["omega_pred"],
                    batch["v_target"], batch["omega_target"],
                    batch["frag_sizes"],
                    omega_weight=cfg["training"].get("omega_weight", 1.0),
                    P_observable=out.get("P_observable"),
                )
                loss = losses["loss"]
                aw = cfg["training"].get("atom_aux_weight", 0.0)
                if aw > 0:
                    aux = atom_position_auxiliary_loss(
                        out["v_pred"], out["omega_pred"],
                        batch["v_target"], batch["omega_target"],
                        atom_pos_t=batch["atom_pos_t"],
                        T_frag=batch["T_frag"],
                        fragment_id=batch["frag_id_for_atoms"],
                        frag_sizes=batch["frag_sizes"],
                    )
                    loss = loss + aw * aux["loss_atom_aux"]
                dw = cfg["training"].get("dg_weight", 0.0)
                if dw > 0:
                    dg = distance_geometry_loss(
                        v_pred=out["v_pred"], omega_pred=out["omega_pred"],
                        T_t=batch["T_frag"], q_t=batch["q_frag"],
                        t_per_sample=batch["t"], frag_batch=batch["frag_batch"],
                        T_target=batch["T_target"], q_target=batch["q_target"],
                        local_pos=batch["local_pos"],
                        frag_id_for_atoms=batch["frag_id_for_atoms"],
                        atom_batch=batch["atom_batch"],
                        lig_atom_slice=batch["lig_atom_slice"],
                        lig_frag_slice=batch["lig_frag_slice"],
                    )
                    loss = loss + dw * dg["loss_dg"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for opt in opts:
                    opt.step()
                torch.cuda.synchronize()
                step_times.append(time.time() - t_s)
            dt = time.time() - t0
            peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            alloc_gb = torch.cuda.memory_allocated(device) / 1024**3
            steady = sum(step_times[2:]) / max(len(step_times) - 2, 1) if len(step_times) > 2 else step_times[-1]
            print(f"BS={bs:4d}  peak={peak_gb:6.1f}GB  alloc={alloc_gb:6.1f}GB  "
                  f"first={step_times[0]:.2f}s  steady={steady:.2f}s  OK")
            results.append((bs, peak_gb, steady, True))
        except torch.cuda.OutOfMemoryError as e:
            print(f"BS={bs:4d}  OOM: {str(e)[:80]}")
            results.append((bs, -1.0, -1.0, False))
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"BS={bs:4d}  FAIL: {type(e).__name__}: {str(e)[:200]}")
            results.append((bs, -1.0, -1.0, False))
            torch.cuda.empty_cache()

    print("\n== summary ==")
    for bs, peak, step_s, ok in results:
        status = f"peak={peak:.1f}GB  step={step_s:.2f}s" if ok else "FAIL"
        print(f"  BS={bs:4d}  {status}")


if __name__ == "__main__":
    main()
