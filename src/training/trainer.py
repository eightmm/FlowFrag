"""Training loop for FlowFrag with DDP, Muon+AdamW, and trapezoidal LR."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, Subset

from src.data.dataset import UnifiedDataset, unified_collate
from src.geometry.flow_matching import integrate_se3_step, sample_prior_poses
from src.models.unified import UnifiedFlowFrag
from src.training.losses import flow_matching_loss
from src.geometry.se3 import quaternion_to_matrix


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # When scripts/train.py masks CUDA_VISIBLE_DEVICES per rank (workaround for
    # cuEquivariance DDP), each process only sees one GPU as cuda:0.
    if world_size > 1:
        cuda_idx = 0 if torch.cuda.device_count() == 1 else local_rank
        torch.cuda.set_device(cuda_idx)
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def cleanup_ddp(world_size: int) -> None:
    if world_size > 1:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def configure_optimizers(
    model: nn.Module,
    lr: float = 3e-4,
    muon_lr: float = 0.02,
    weight_decay: float = 0.01,
    use_muon: bool = True,
) -> list:
    muon_params, adamw_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Muon's Newton-Schulz is degenerate for single-row matrices.
        # Require min(shape) >= 2 so [1, 64] out_linear goes to AdamW.
        if use_muon and param.ndim == 2 and min(param.shape) >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    optimizers = []
    if muon_params:
        from torch.optim import Muon
        optimizers.append(Muon(muon_params, lr=muon_lr, momentum=0.95))
    if adamw_params:
        optimizers.append(AdamW(adamw_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)))
    return optimizers


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def get_trapezoidal_scheduler(
    optimizer, total_steps: int, warmup_ratio: float = 0.1, cooldown_ratio: float = 0.3
) -> LambdaLR:
    warmup = int(total_steps * warmup_ratio)
    cooldown = int(total_steps * cooldown_ratio)
    stable = total_steps - warmup - cooldown

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        elif step < warmup + stable:
            return 1.0
        else:
            return max(1.0 - (step - warmup - stable) / max(cooldown, 1), 0.0)

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.rank, self.local_rank, self.world_size = setup_ddp()
        # cuda:0 inside each rank (per-rank CUDA_VISIBLE_DEVICES) or local_rank
        # when unmasked (e.g. single-process run with all GPUs visible).
        if torch.cuda.is_available():
            cuda_idx = 0 if torch.cuda.device_count() == 1 else self.local_rank
            self.device = torch.device(f"cuda:{cuda_idx}")
        else:
            self.device = torch.device("cpu")
        self.is_main = self.rank == 0

        torch.manual_seed(cfg["training"].get("seed", 42))

        # Dirs
        self.output_dir = Path(cfg["logging"]["output_dir"])
        self.ckpt_dir = self.output_dir / "checkpoints"
        if self.is_main:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self._build_dataloaders()

        # Model
        model_kwargs = {k: v for k, v in cfg["model"].items() if k != "model_type"}
        self.model = UnifiedFlowFrag(**model_kwargs).to(self.device)
        if self.world_size > 1:
            ddp_device = self.device.index
            self.model = DDP(
                self.model, device_ids=[ddp_device],
                output_device=ddp_device,
                find_unused_parameters=False, gradient_as_bucket_view=True, static_graph=True,
            )

        # Training params (needed before _total_steps)
        tcfg = cfg["training"]
        self.global_step = 0
        self.start_epoch = 0
        self.grad_accum = tcfg.get("gradient_accumulation_steps", 1)

        # Optimizers & schedulers
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        self.optimizers = configure_optimizers(
            raw_model, lr=tcfg["lr"], muon_lr=tcfg.get("muon_lr", 0.02),
            weight_decay=tcfg.get("weight_decay", 0.01), use_muon=tcfg.get("use_muon", True),
        )
        total_steps = self._total_steps()
        self.schedulers = [
            get_trapezoidal_scheduler(
                opt, total_steps,
                warmup_ratio=tcfg.get("warmup_ratio", 0.1),
                cooldown_ratio=tcfg.get("cooldown_ratio", 0.3),
            )
            for opt in self.optimizers
        ]
        # EMA (used for val/rollout only)
        self.use_ema = tcfg.get("use_ema", True)
        self.ema_decay = tcfg.get("ema_decay", 0.999)
        if self.use_ema:
            from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
            self.ema_model = AveragedModel(
                raw_model,
                multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay),
                use_buffers=True,
            )
        else:
            self.ema_model = None

        self.max_grad_norm = tcfg.get("max_grad_norm", 1.0)
        self.omega_weight = tcfg.get("omega_weight", 1.0)
        self.omega_loss_frame = tcfg.get("omega_loss_frame", "world")
        self.omega_loss_type = tcfg.get("omega_loss_type", "mse")
        self.omega_dir_weight = tcfg.get("omega_dir_weight", 1.0)
        self.omega_mag_weight = tcfg.get("omega_mag_weight", 0.1)
        self.atom_aux_weight = tcfg.get("atom_aux_weight", 0.0)
        self.dg_weight = tcfg.get("dg_weight", 0.0)

        # Mixed precision: disabled by default because cuEquivariance's
        # fused_tp kernels only support FP32 inputs (they raise on BF16).
        # TF32 is enabled globally in scripts/train.py and already accelerates
        # the scalar path of the model.
        self.use_amp = tcfg.get("use_amp", False)
        self.amp_dtype = torch.bfloat16
        self.boundary_weight = tcfg.get("boundary_weight", 0.0)
        self.dummy_weight = tcfg.get("dummy_weight", 0.0)
        self.use_time_weighting = tcfg.get("use_time_weighting", True)

        # Wandb (initialized lazily after potential checkpoint load)
        self.use_wandb = cfg["logging"].get("use_wandb", False) and self.is_main
        self.wandb_run_id: str | None = None
        self._wandb_initialized = False
        self._best_rmsd = float("inf")

    # ---- Data ----

    def _build_dataloaders(self) -> None:
        dcfg = self.cfg["data"]
        tcfg = self.cfg["training"]
        overfit_batches = int(tcfg.get("overfit_batches", 0))
        overfit_mode = overfit_batches > 0
        bs = tcfg["batch_size"]
        ds_kwargs = dict(
            root=dcfg["data_dir"],
            pocket_cutoff=dcfg.get("pocket_cutoff", 8.0),
            pocket_jitter_sigma=dcfg.get("pocket_jitter_sigma", 0.0),
            pocket_cutoff_noise=dcfg.get("pocket_cutoff_noise", 0.0),
            translation_sigma=dcfg.get("prior_sigma", 5.0),
            max_atoms=dcfg.get("max_atoms", 80),
            max_frags=dcfg.get("max_frags", 20),
            min_atoms=dcfg.get("min_atoms", 5),
            min_protein_res=dcfg.get("min_protein_res", 50),
            rotation_augmentation=dcfg.get("rotation_augmentation", "none"),
            deterministic=dcfg.get("deterministic", False),
            seed=tcfg.get("seed", 42),
        )

        split_file = dcfg.get("split_file")
        DatasetClass = UnifiedDataset

        # Val dataset: no augmentation, no jitter (deterministic crystal eval)
        val_kwargs = dict(ds_kwargs)
        val_kwargs["rotation_augmentation"] = "none"
        val_kwargs["pocket_jitter_sigma"] = 0.0
        val_kwargs["pocket_cutoff_noise"] = 0.0

        if split_file is not None:
            train_ds = DatasetClass(split_file=split_file, split_key="train", **ds_kwargs)
            val_ds = DatasetClass(split_file=split_file, split_key="val", **val_kwargs)
            if len(val_ds) == 0:
                val_ds = None
        else:
            full_ds = DatasetClass(**ds_kwargs)
            val_ratio = dcfg.get("val_split", 0.05)
            n_val = int(len(full_ds) * val_ratio)
            n_train = len(full_ds) - n_val
            if n_val > 0:
                train_ds, val_ds = torch.utils.data.random_split(
                    full_ds, [n_train, n_val],
                    generator=torch.Generator().manual_seed(self.cfg["training"].get("seed", 42)),
                )
            else:
                train_ds, val_ds = full_ds, None

        if overfit_mode:
            max_samples = min(len(train_ds), bs * overfit_batches)
            if max_samples <= 0:
                raise ValueError("overfit_batches > 0 but the training dataset is empty.")
            train_ds = Subset(train_ds, list(range(max_samples)))

        if self.is_main:
            train_msg = f"Dataset: train={len(train_ds)}, val={len(val_ds) if val_ds else 0}"
            if overfit_mode:
                train_msg += f" (strict overfit subset, {len(train_ds)} samples)"
            print(train_msg)

        nw = dcfg.get("num_workers", 4)
        collate_fn = unified_collate

        if self.world_size > 1:
            self.train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=not overfit_mode,
                seed=42,
            )
            shuffle = False
        else:
            self.train_sampler = None
            shuffle = not overfit_mode

        loader_kwargs = dict(
            collate_fn=collate_fn, pin_memory=True,
            persistent_workers=nw > 0,
            prefetch_factor=4 if nw > 0 else None,
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=shuffle, sampler=self.train_sampler,
            num_workers=nw, drop_last=True, **loader_kwargs,
        )
        if val_ds is not None:
            if self.world_size > 1:
                self.val_sampler = DistributedSampler(
                    val_ds, num_replicas=self.world_size, rank=self.rank,
                    shuffle=False, drop_last=False,
                )
            else:
                self.val_sampler = None
            self.val_loader = DataLoader(
                val_ds, batch_size=bs, shuffle=False, sampler=self.val_sampler,
                num_workers=nw, **loader_kwargs,
            )
        else:
            self.val_loader = None
            self.val_sampler = None

    @staticmethod
    def _dict_batch_to_device(batch: dict, device: torch.device) -> dict:
        """Move all tensors in a dict batch to device."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _compute_loss_unified(self, out: dict, batch: dict) -> dict:
        """Compute flow matching loss for unified model."""
        from src.training.losses import (
            compute_time_weight,
            atom_position_auxiliary_loss,
            distance_geometry_loss,
            boundary_alignment_loss,
        )

        R_t = None
        if self.omega_loss_frame == "body":
            R_t = quaternion_to_matrix(batch["q_frag"])

        # Per-fragment time weight
        if self.use_time_weighting:
            t_per_frag = batch["t"].view(-1)[batch["frag_batch"]]  # [N_frag]
            time_weight = compute_time_weight(t_per_frag)
        else:
            time_weight = None

        losses = flow_matching_loss(
            out["v_pred"], out["omega_pred"],
            batch["v_target"], batch["omega_target"],
            batch["frag_sizes"], omega_weight=self.omega_weight,
            R_t=R_t,
            omega_loss_frame=self.omega_loss_frame,
            omega_loss_type=self.omega_loss_type,
            omega_dir_weight=self.omega_dir_weight,
            omega_mag_weight=self.omega_mag_weight,
            time_weight=time_weight,
            P_observable=out.get("P_observable"),
        )

        # Atom-level auxiliary loss: v_atom = v_frag + omega × r
        if self.atom_aux_weight > 0:
            aux = atom_position_auxiliary_loss(
                out["v_pred"], out["omega_pred"],
                batch["v_target"], batch["omega_target"],
                atom_pos_t=batch["atom_pos_t"],
                T_frag=batch["T_frag"],
                fragment_id=batch["frag_id_for_atoms"],
                frag_sizes=batch["frag_sizes"],
            )
            losses["loss"] = losses["loss"] + self.atom_aux_weight * aux["loss_atom_aux"]
            losses["loss_atom_aux"] = aux["loss_atom_aux"].detach()

        # Distance geometry loss: one-step Euler → pairwise distance MSE
        # (weighted by t² — strict near t=1, loose near t=0)
        if self.dg_weight > 0:
            dg = distance_geometry_loss(
                v_pred=out["v_pred"],
                omega_pred=out["omega_pred"],
                T_t=batch["T_frag"],
                q_t=batch["q_frag"],
                t_per_sample=batch["t"],
                frag_batch=batch["frag_batch"],
                T_target=batch["T_target"],
                q_target=batch["q_target"],
                local_pos=batch["local_pos"],
                frag_id_for_atoms=batch["frag_id_for_atoms"],
                atom_batch=batch["atom_batch"],
                lig_atom_slice=batch["lig_atom_slice"],
                lig_frag_slice=batch["lig_frag_slice"],
            )
            losses["loss"] = losses["loss"] + self.dg_weight * dg["loss_dg"]
            losses["loss_dg"] = dg["loss_dg"].detach()

        # Boundary alignment loss: cut-bond atom velocities should agree
        if self.boundary_weight > 0 and "cut_bond_src" in batch:
            bnd = boundary_alignment_loss(
                v_pred=out["v_pred"],
                omega_pred=out["omega_pred"],
                atom_pos_t=batch["atom_pos_t"],
                T_frag=batch["T_frag"],
                fragment_id=batch["frag_id_for_atoms"],
                cut_src=batch["cut_bond_src"],
                cut_dst=batch["cut_bond_dst"],
            )
            losses["loss"] = losses["loss"] + self.boundary_weight * bnd["loss_boundary"]
            losses["loss_boundary"] = bnd["loss_boundary"].detach()

        return losses

    def _total_steps(self) -> int:
        """Total optimizer steps from config (step-based schedule)."""
        tcfg = self.cfg["training"]
        assert "max_steps" in tcfg, "config must specify training.max_steps"
        return int(tcfg["max_steps"])

    # ---- Checkpoint ----

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        if self.ema_model is not None and "ema_state_dict" in ckpt:
            self.ema_model.load_state_dict(ckpt["ema_state_dict"])
        for opt, opt_sd in zip(self.optimizers, ckpt.get("optimizer_state_dicts", [])):
            opt.load_state_dict(opt_sd)
        for sched, sched_sd in zip(self.schedulers, ckpt.get("scheduler_state_dicts", [])):
            sched.load_state_dict(sched_sd)
        self.global_step = ckpt.get("step", 0)
        self.start_epoch = ckpt.get("epoch", 0) + 1  # resume AFTER completed epoch
        self.wandb_run_id = ckpt.get("wandb_run_id")
        if self.is_main:
            print(f"Resumed from {path} (completed epoch {self.start_epoch - 1}, step {self.global_step})")

    def _build_checkpoint_state(self, epoch: int, metrics: dict) -> dict:
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        state = {
            "epoch": epoch,
            "step": self.global_step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dicts": [o.state_dict() for o in self.optimizers],
            "scheduler_state_dicts": [s.state_dict() for s in self.schedulers],
            "metrics": metrics,
            "wandb_run_id": self.wandb_run_id,
        }
        if self.ema_model is not None:
            state["ema_state_dict"] = self.ema_model.state_dict()
        return state

    def _save_latest(self, epoch: int, metrics: dict) -> None:
        """Save latest.pt (overwritten every val). Used for resume."""
        if not self.is_main:
            return
        state = self._build_checkpoint_state(epoch, metrics)
        torch.save(state, self.ckpt_dir / "latest.pt")

    def _save_rollout(self, epoch: int, metrics: dict) -> None:
        """Save a named rollout checkpoint + update best.pt if RMSD improved."""
        if not self.is_main:
            return
        state = self._build_checkpoint_state(epoch, metrics)
        path = self.ckpt_dir / f"rollout_step{self.global_step:07d}.pt"
        torch.save(state, path)

        # Update best.pt if median RMSD improved
        rmsd = metrics.get("rollout/rmsd_median", float("inf"))
        if rmsd < self._best_rmsd:
            self._best_rmsd = rmsd
            torch.save(state, self.ckpt_dir / "best.pt")
            print(f"  New best RMSD: {rmsd:.2f}A → saved best.pt")

    def _init_wandb(self) -> None:
        if not self.use_wandb or self._wandb_initialized:
            return
        try:
            import wandb

            init_kwargs: dict = dict(
                project=self.cfg["logging"].get("wandb_project", "flowfrag"),
                config=self.cfg,
            )
            run_name = self.cfg["logging"].get("wandb_run_name")
            if run_name is not None:
                init_kwargs["name"] = run_name
            # Resume existing run if we have a saved run_id
            if self.wandb_run_id is not None:
                init_kwargs["id"] = self.wandb_run_id
                init_kwargs["resume"] = "must"

            wandb.init(**init_kwargs)
            self.wandb_run_id = wandb.run.id  # type: ignore[union-attr]

            # Define metric groupings — hide internal step counter
            wandb.define_metric("global_step", hidden=True)
            wandb.define_metric("step/*", step_metric="global_step")
            wandb.define_metric("epoch/*", step_metric="global_step")
            wandb.define_metric("val/*", step_metric="global_step")
            wandb.define_metric("rollout/*", step_metric="global_step")
            wandb.define_metric("meta/*", step_metric="global_step")

            self._wandb_initialized = True
        except Exception as e:
            print(f"WARNING: wandb init failed: {e}")
            self.use_wandb = False

    # ---- Train ----

    def train(self) -> None:
        self._init_wandb()

        tcfg = self.cfg["training"]
        lcfg = self.cfg["logging"]
        log_every = lcfg.get("log_every", 50)
        val_every = lcfg.get("val_every", 0)
        rollout_every = lcfg.get("rollout_every", 0)
        overfit_mode = tcfg.get("overfit_batches", 0) > 0

        total_steps = self._total_steps()
        if self.is_main:
            print(f"Training: total_steps={total_steps}, {len(self.train_loader)} batches/data_pass")
            print(f"  grad_accum={self.grad_accum}, effective_bs={tcfg['batch_size'] * self.grad_accum}")

        # data_pass = one full iteration through the train dataset (was called
        # "epoch" before we moved to step-based). Used for DistributedSampler
        # seeding and log grouping.
        epoch = self.start_epoch
        epoch_loss = 0.0
        epoch_steps = 0
        avg_loss = 0.0

        try:
            while self.global_step < total_steps:
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)

                self.model.train()
                epoch_loss = 0.0
                epoch_loss_v = 0.0
                epoch_loss_w = 0.0
                epoch_cos_v = 0.0
                epoch_cos_w = 0.0
                epoch_steps = 0
                t0 = time.time()

                for opt in self.optimizers:
                    opt.zero_grad()

                max_batches = tcfg.get("overfit_batches", 0) if overfit_mode else 0
                for batch_idx, batch in enumerate(self.train_loader):
                    if overfit_mode and batch_idx >= max_batches:
                        break

                    batch = self._dict_batch_to_device(batch, self.device)
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                        out = self.model(batch)
                    losses = self._compute_loss_unified(out, batch)
                    raw_loss = losses["loss"]
                    if not torch.isfinite(raw_loss):
                        print(f"  WARNING: non-finite loss at E{epoch} B{batch_idx}, skipping")
                        for opt in self.optimizers:
                            opt.zero_grad()
                        continue
                    loss = raw_loss / self.grad_accum
                    loss.backward()

                    if (batch_idx + 1) % self.grad_accum == 0:
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm if self.max_grad_norm > 0 else float("inf"),
                        ).item()
                        for opt in self.optimizers:
                            opt.step()
                            opt.zero_grad()
                        for sched in self.schedulers:
                            sched.step()
                        self.global_step += 1
                        if self.ema_model is not None:
                            raw = self.model.module if isinstance(self.model, DDP) else self.model
                            self.ema_model.update_parameters(raw)
                        if self.global_step >= total_steps:
                            # Final val + rollout before exiting
                            if val_every > 0:
                                val_metrics = self._validate(epoch)
                                self._save_latest(epoch, {"train_loss": epoch_loss / max(epoch_steps, 1), **val_metrics})
                            if rollout_every > 0:
                                rollout_metrics = self._validate_rollout(epoch)
                                self._save_rollout(epoch, {"train_loss": epoch_loss / max(epoch_steps, 1), **rollout_metrics})
                            break
                    else:
                        grad_norm = None

                    step_loss = losses["loss"].item()
                    epoch_loss += step_loss
                    epoch_loss_v += losses["loss_v"].item()
                    epoch_loss_w += losses["loss_omega"].item()
                    epoch_cos_v += losses["cos_v"].item()
                    epoch_cos_w += losses["cos_omega"].item()
                    epoch_steps += 1

                    # Logging (triggered on optimizer step boundaries)
                    log_trigger = (
                        self.is_main
                        and log_every > 0
                        and grad_norm is not None  # optimizer step just happened
                        and self.global_step % log_every == 0
                    )
                    if log_trigger:
                        avg = epoch_loss / epoch_steps
                        lr_vals = [opt.param_groups[0]["lr"] for opt in self.optimizers]
                        print(f"  [S{self.global_step}] loss={step_loss:.4f} avg={avg:.4f} "
                              f"loss_v={losses['loss_v'].item():.4f} loss_w={losses['loss_omega'].item():.4f} "
                              f"lr={lr_vals}")
                        if self.use_wandb:
                            import wandb
                            log_dict = {
                                "step/loss": step_loss,
                                "step/loss_v": losses["loss_v"].item(),
                                "step/loss_omega": losses["loss_omega"].item(),
                                "step/cos_v": losses["cos_v"].item(),
                                "step/cos_omega": losses["cos_omega"].item(),
                                "meta/lr_adamw": lr_vals[-1],
                                "meta/epoch": epoch,
                            }
                            if len(lr_vals) > 1:
                                log_dict["meta/lr_muon"] = lr_vals[0]
                            if grad_norm is not None:
                                log_dict["step/grad_norm"] = grad_norm
                            for extra_key in ("loss_omega_dir", "loss_omega_mag", "loss_atom_aux", "loss_dg", "loss_boundary", "cos_omega_world"):
                                if extra_key in losses:
                                    log_dict[f"step/{extra_key}"] = losses[extra_key].item()
                            wandb.log(log_dict, step=self.global_step)

                    # Val (loss only) + save latest
                    if val_every > 0 and self.global_step > 0 and self.global_step % val_every == 0:
                        val_metrics = self._validate(epoch)
                        self._save_latest(epoch, {"train_loss": epoch_loss / max(epoch_steps, 1), **val_metrics})

                    # Rollout (ODE integration + RMSD) + save rollout checkpoint
                    if rollout_every > 0 and self.global_step > 0 and self.global_step % rollout_every == 0:
                        rollout_metrics = self._validate_rollout(epoch)
                        self._save_rollout(epoch, {"train_loss": epoch_loss / max(epoch_steps, 1), **rollout_metrics})

                # Flush leftover gradients at epoch end
                n_batches = batch_idx + 1 if epoch_steps > 0 else 0
                if n_batches > 0 and n_batches % self.grad_accum != 0:
                    if self.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    for opt in self.optimizers:
                        opt.step()
                        opt.zero_grad()
                    for sched in self.schedulers:
                        sched.step()
                    self.global_step += 1

                # Epoch summary
                elapsed = time.time() - t0
                if self.is_main:
                    if epoch_steps > 0:
                        avg_loss = epoch_loss / epoch_steps
                        avg_v = epoch_loss_v / epoch_steps
                        avg_w = epoch_loss_w / epoch_steps
                        cv = epoch_cos_v / epoch_steps
                        cw = epoch_cos_w / epoch_steps
                        print(f"Epoch {epoch} done: loss={avg_loss:.4f} v={avg_v:.4f} w={avg_w:.4f} "
                              f"cos_v={cv:.3f} cos_w={cw:.3f} ({elapsed:.1f}s)")
                        # Epoch-level wandb logging removed — step-level (every 50 steps)
                        # provides finer granularity; epoch averages add chart clutter.
                    else:
                        avg_loss = float("nan")
                        print(f"Epoch {epoch}: ALL BATCHES SKIPPED ({elapsed:.1f}s)")

                if overfit_mode and self.is_main and (epoch + 1) % 50 == 0:
                    self._save_latest(epoch, {"train_loss": avg_loss})

                epoch += 1

            # Final save
            if self.is_main:
                self._save_latest(epoch, {"train_loss": avg_loss})
                print("Training complete.")

        except KeyboardInterrupt:
            if self.is_main:
                print("Interrupted. Saving checkpoint...")
                self._save_latest(epoch, {"train_loss": epoch_loss / max(epoch_steps, 1)})
        finally:
            cleanup_ddp(self.world_size)
            if self.use_wandb:
                import wandb
                wandb.finish()

    # ---- Validation ----

    def _eval_model(self) -> nn.Module:
        """Return EMA model for evaluation if enabled, else unwrapped live model."""
        if self.ema_model is not None:
            return self.ema_model
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _validate(self, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            return {}

        em = self._eval_model()
        em.train(False)
        total_loss = 0.0
        total_v = 0.0
        total_w = 0.0
        n = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._dict_batch_to_device(batch, self.device)
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    out = em(batch)
                losses = self._compute_loss_unified(out, batch)
                total_loss += losses["loss"].item()
                total_v += losses["loss_v"].item()
                total_w += losses["loss_omega"].item()
                n += 1

        avg_loss = total_loss / max(n, 1)
        avg_v = total_v / max(n, 1)
        avg_w = total_w / max(n, 1)

        if self.world_size > 1:
            tensors = torch.tensor([avg_loss, avg_v, avg_w], device=self.device)
            dist.all_reduce(tensors)
            tensors /= self.world_size
            avg_loss, avg_v, avg_w = tensors.tolist()

        if self.is_main:
            print(f"  [Val E{epoch} S{self.global_step}] loss={avg_loss:.4f} v={avg_v:.4f} w={avg_w:.4f}")
            if self.use_wandb:
                import wandb
                wandb.log({
                    "val/loss": avg_loss,
                    "val/loss_v": avg_v,
                    "val/loss_omega": avg_w,
                }, step=self.global_step)

        self.model.train()
        return {"val_loss": avg_loss, "val_loss_v": avg_v, "val_loss_omega": avg_w}

    @torch.no_grad()
    def _rollout_single_unified(
        self,
        raw_model: nn.Module,
        sample: dict[str, torch.Tensor | str],
        *,
        sigma: float,
        num_steps: int,
        time_schedule: str,
        schedule_power: float,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from src.inference.sampler import build_time_grid

        n_frag = sample["num_lig_frag"].item()
        frag_sizes = sample["frag_sizes"].to(self.device)
        frag_id = sample["frag_id_for_atoms"].to(self.device)
        local_pos = sample["local_pos"].to(self.device)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        T, q = sample_prior_poses(
            n_frag,
            torch.zeros(3),
            sigma,
            frag_sizes=frag_sizes.cpu(),
            dtype=torch.float32,
            generator=gen,
        )
        T, q = T.to(self.device), q.to(self.device)

        time_grid = build_time_grid(
            num_steps,
            schedule=time_schedule,
            power=schedule_power,
            device=self.device,
            dtype=torch.float32,
        )

        batch = unified_collate([sample])
        batch_gpu = {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        frag_slice = batch_gpu["lig_frag_slice"][0]
        frag_start, frag_end = frag_slice[0].item(), frag_slice[1].item()
        atom_slice = batch_gpu["lig_atom_slice"][0]
        atom_start = atom_slice[0].item()

        for step_idx in range(num_steps):
            t_val = time_grid[step_idx]
            dt = time_grid[step_idx + 1] - time_grid[step_idx]

            R = quaternion_to_matrix(q)
            atom_pos = torch.einsum("nij,nj->ni", R[frag_id], local_pos) + T[frag_id]

            node_coords = batch_gpu["node_coords"].clone()
            node_coords[frag_start:frag_end] = T
            node_coords[atom_start:atom_start + atom_pos.shape[0]] = atom_pos

            batch_gpu["node_coords"] = node_coords
            batch_gpu["T_frag"] = T
            batch_gpu["q_frag"] = q
            batch_gpu["frag_sizes"] = frag_sizes
            batch_gpu["t"] = t_val.view(1, 1)

            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                out = raw_model(batch_gpu)
            T, q = integrate_se3_step(
                T, q, out["v_pred"], out["omega_pred"], dt, frag_sizes=frag_sizes,
            )

        R_final = quaternion_to_matrix(q)
        atom_pos_final = torch.einsum("nij,nj->ni", R_final[frag_id], local_pos) + T[frag_id]

        R_target = quaternion_to_matrix(sample["q_target"].to(self.device))
        true_pos = torch.einsum("nij,nj->ni", R_target[frag_id], local_pos) + sample["T_target"].to(self.device)[frag_id]

        return T, atom_pos_final, true_pos

    def _validate_rollout(self, epoch: int) -> dict[str, float]:
        """Run ODE rollout on val set and compute docking metrics."""
        if self.val_loader is None:
            return {}

        from src.inference.metrics import ligand_rmsd, centroid_distance, frag_centroid_rmsd

        raw_model = self._eval_model()
        raw_model.train(False)

        dcfg = self.cfg["data"]
        lcfg = self.cfg["logging"]
        num_steps = lcfg.get("rollout_steps", 20)
        time_schedule = lcfg.get("rollout_time_schedule", "uniform")
        schedule_power = lcfg.get("rollout_schedule_power", 3.0)
        max_samples = lcfg.get("rollout_max_samples", 0)  # 0 = full val set
        sigma = dcfg.get("prior_sigma", 5.0)
        seed_base = self.cfg["training"].get("seed", 42)

        rmsds, cent_dists, frag_rmsds = [], [], []
        n_done = 0

        # Iterate over val dataset (optionally capped by rollout_max_samples).
        # Under DDP each rank handles indices[rank::world_size] (disjoint
        # partition, no duplication) — then all_gather below assembles the
        # full metric arrays.
        val_ds = self.val_loader.dataset
        n_val = len(val_ds) if max_samples <= 0 else min(len(val_ds), max_samples)
        for i in range(self.rank, n_val, self.world_size):
            data = val_ds[i]
            T_pred, atom_pos_pred, true_pos = self._rollout_single_unified(
                raw_model,
                data,
                sigma=sigma,
                num_steps=num_steps,
                time_schedule=time_schedule,
                schedule_power=schedule_power,
                seed=seed_base + i,
            )
            T_target = data["T_target"].to(self.device)

            rmsds.append(ligand_rmsd(atom_pos_pred, true_pos).item())
            cent_dists.append(centroid_distance(atom_pos_pred, true_pos).item())
            frag_rmsds.append(frag_centroid_rmsd(T_pred, T_target).item())
            n_done += 1

        if self.world_size > 1:
            gathered: list[tuple[list, list, list]] = [None] * self.world_size  # type: ignore
            dist.all_gather_object(gathered, (rmsds, cent_dists, frag_rmsds))
            rmsds = [x for shard in gathered for x in shard[0]]
            cent_dists = [x for shard in gathered for x in shard[1]]
            frag_rmsds = [x for shard in gathered for x in shard[2]]
            n_done = len(rmsds)

        if n_done == 0:
            self.model.train()
            return {}

        rmsds_t = torch.tensor(rmsds)
        metrics = {
            "rollout/rmsd_median": rmsds_t.median().item(),
            "rollout/rmsd_mean": rmsds_t.mean().item(),
            "rollout/rmsd_p25": rmsds_t.quantile(0.25).item(),
            "rollout/rmsd_p75": rmsds_t.quantile(0.75).item(),
            "rollout/success_2A": (rmsds_t < 2.0).float().mean().item(),
            "rollout/success_5A": (rmsds_t < 5.0).float().mean().item(),
            "rollout/centroid_dist": torch.tensor(cent_dists).mean().item(),
            "rollout/frag_rmsd": torch.tensor(frag_rmsds).mean().item(),
        }

        if self.is_main:
            print(f"  [Rollout E{epoch} S{self.global_step}] "
                  f"RMSD={metrics['rollout/rmsd_median']:.2f}A (median) "
                  f"<2A={metrics['rollout/success_2A']:.1%} "
                  f"<5A={metrics['rollout/success_5A']:.1%} "
                  f"({n_done} samples, {num_steps} steps)")
            if self.use_wandb:
                import wandb
                wandb.log(metrics, step=self.global_step)

        self.model.train()
        return metrics


__all__ = ["Trainer"]
