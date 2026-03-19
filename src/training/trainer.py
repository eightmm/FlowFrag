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

from src.data.dataset import FlowFragDataset
from src.models.flowfrag import FlowFrag
from src.training.losses import flow_matching_loss, atom_velocity_loss, atom_position_auxiliary_loss, boundary_alignment_loss
from src.geometry.se3 import quaternion_to_matrix


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
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
# Collate
# ---------------------------------------------------------------------------

def hetero_collate(batch):
    """Collate HeteroData list via PyG Batch."""
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.rank, self.local_rank, self.world_size = setup_ddp()
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
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
        self.model = FlowFrag(**cfg["model"]).to(self.device)
        if self.world_size > 1:
            self.model = DDP(
                self.model, device_ids=[self.local_rank],
                output_device=self.local_rank,
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
        self.max_grad_norm = tcfg.get("max_grad_norm", 1.0)
        self.omega_weight = tcfg.get("omega_weight", 1.0)
        self.omega_loss_frame = tcfg.get("omega_loss_frame", "world")
        self.omega_loss_type = tcfg.get("omega_loss_type", "mse")
        self.omega_dir_weight = tcfg.get("omega_dir_weight", 1.0)
        self.omega_mag_weight = tcfg.get("omega_mag_weight", 0.1)
        self.atom_aux_weight = tcfg.get("atom_aux_weight", 0.0)
        self.boundary_weight = tcfg.get("boundary_weight", 0.0)

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
            translation_sigma=dcfg.get("prior_sigma", 10.0),
            max_atoms=dcfg.get("max_atoms", 80),
            max_frags=dcfg.get("max_frags", 20),
            min_atoms=dcfg.get("min_atoms", 5),
            rotation_augmentation=dcfg.get("rotation_augmentation", "none"),
            deterministic=dcfg.get("deterministic", False),
            deterministic_augmentation=dcfg.get("deterministic_augmentation"),
            deterministic_prior=dcfg.get("deterministic_prior"),
            deterministic_time=dcfg.get("deterministic_time"),
            prior_bank_size=dcfg.get("prior_bank_size", 1),
            time_bank_size=dcfg.get("time_bank_size", 1),
            seed=tcfg.get("seed", 42),
        )

        split_file = dcfg.get("split_file")

        if split_file is not None:
            # JSON or text split file → separate datasets
            train_ds = FlowFragDataset(split_file=split_file, split_key="train", **ds_kwargs)
            val_ds = FlowFragDataset(split_file=split_file, split_key="val", **ds_kwargs)
            if len(val_ds) == 0:
                val_ds = None
        else:
            # Fallback: random split
            full_ds = FlowFragDataset(**ds_kwargs)
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

        self.train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=shuffle, sampler=self.train_sampler,
            num_workers=nw, collate_fn=hetero_collate, pin_memory=True, drop_last=True,
        )
        if val_ds is not None:
            self.val_loader = DataLoader(
                val_ds, batch_size=bs, shuffle=False,
                num_workers=nw, collate_fn=hetero_collate, pin_memory=True,
            )
        else:
            self.val_loader = None

    def _total_steps(self) -> int:
        tcfg = self.cfg["training"]
        steps_per_epoch = max(len(self.train_loader) // self.grad_accum, 1)
        return steps_per_epoch * tcfg["epochs"]

    # ---- Checkpoint ----

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        raw_model.load_state_dict(ckpt["model_state_dict"])
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
        return {
            "epoch": epoch,
            "step": self.global_step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dicts": [o.state_dict() for o in self.optimizers],
            "scheduler_state_dicts": [s.state_dict() for s in self.schedulers],
            "metrics": metrics,
            "wandb_run_id": self.wandb_run_id,
        }

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

        if self.is_main:
            print(f"Training: {tcfg['epochs']} epochs, {len(self.train_loader)} batches/epoch")
            print(f"  grad_accum={self.grad_accum}, effective_bs={tcfg['batch_size'] * self.grad_accum}")

        epoch = self.start_epoch
        epoch_loss = 0.0
        epoch_steps = 0
        avg_loss = 0.0

        try:
            for epoch in range(self.start_epoch, tcfg["epochs"]):
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
                    batch = batch.to(self.device)
                    out = self.model(batch)

                    if "v_atom_pred" in out:
                        losses = atom_velocity_loss(
                            out["v_atom_pred"], batch["atom"].v_target,
                            out["v_pred"], out["omega_pred"],
                            batch["fragment"].v_target, batch["fragment"].omega_target,
                            batch["fragment"].size,
                        )
                    else:
                        R_t = None
                        if self.omega_loss_frame == "body":
                            R_t = quaternion_to_matrix(batch["fragment"].q_frag)
                        losses = flow_matching_loss(
                            out["v_pred"], out["omega_pred"],
                            batch["fragment"].v_target, batch["fragment"].omega_target,
                            batch["fragment"].size, omega_weight=self.omega_weight,
                            R_t=R_t,
                            omega_loss_frame=self.omega_loss_frame,
                            omega_loss_type=self.omega_loss_type,
                            omega_dir_weight=self.omega_dir_weight,
                            omega_mag_weight=self.omega_mag_weight,
                        )
                        # Atom-position auxiliary loss
                        if self.atom_aux_weight > 0:
                            aux = atom_position_auxiliary_loss(
                                out["v_pred"], out["omega_pred"],
                                batch["fragment"].v_target, batch["fragment"].omega_target,
                                batch["atom"].pos_t, batch["fragment"].T_frag,
                                batch["atom"].fragment_id, batch["fragment"].size,
                            )
                            losses["loss"] = losses["loss"] + self.atom_aux_weight * aux["loss_atom_aux"]
                            losses["loss_atom_aux"] = aux["loss_atom_aux"].detach()
                        # Boundary alignment loss (cut-bond velocity consistency)
                        if self.boundary_weight > 0:
                            try:
                                cut_edge = batch["atom", "cut", "atom"]
                                has_cut = hasattr(cut_edge, "edge_index") and cut_edge.edge_index.shape[1] > 0
                            except (KeyError, AttributeError):
                                has_cut = False
                            if has_cut:
                                bnd = boundary_alignment_loss(
                                    out["v_pred"], out["omega_pred"],
                                    batch["atom"].pos_t, batch["fragment"].T_frag,
                                    batch["atom"].fragment_id,
                                    cut_edge.edge_index[0], cut_edge.edge_index[1],
                                )
                                losses["loss"] = losses["loss"] + self.boundary_weight * bnd["loss_boundary"]
                                losses["loss_boundary"] = bnd["loss_boundary"].detach()
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
                    else:
                        grad_norm = None

                    step_loss = losses["loss"].item()
                    epoch_loss += step_loss
                    epoch_loss_v += losses["loss_v"].item()
                    epoch_loss_w += losses["loss_omega"].item()
                    epoch_cos_v += losses["cos_v"].item()
                    epoch_cos_w += losses["cos_omega"].item()
                    epoch_steps += 1

                    # Logging
                    if self.is_main and log_every > 0 and (batch_idx + 1) % log_every == 0:
                        avg = epoch_loss / epoch_steps
                        lr_vals = [opt.param_groups[0]["lr"] for opt in self.optimizers]
                        print(f"  [E{epoch} B{batch_idx+1}] loss={step_loss:.4f} avg={avg:.4f} "
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
                            for extra_key in ("loss_omega_dir", "loss_omega_mag", "loss_atom_aux", "cos_omega_world"):
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
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "epoch/loss": avg_loss,
                                "epoch/loss_v": avg_v,
                                "epoch/loss_omega": avg_w,
                                "epoch/cos_v": cv,
                                "epoch/cos_omega": cw,
                            }, step=self.global_step)
                    else:
                        avg_loss = float("nan")
                        print(f"Epoch {epoch}: ALL BATCHES SKIPPED ({elapsed:.1f}s)")

                if overfit_mode and self.is_main and (epoch + 1) % 50 == 0:
                    self._save_latest(epoch, {"train_loss": avg_loss})

            # Final save
            if self.is_main:
                self._save_latest(tcfg["epochs"] - 1, {"train_loss": avg_loss})
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

    def _validate(self, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_v = 0.0
        total_w = 0.0
        n = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                if "v_atom_pred" in out:
                    losses = atom_velocity_loss(
                        out["v_atom_pred"], batch["atom"].v_target,
                        out["v_pred"], out["omega_pred"],
                        batch["fragment"].v_target, batch["fragment"].omega_target,
                        batch["fragment"].size,
                    )
                else:
                    losses = flow_matching_loss(
                        out["v_pred"], out["omega_pred"],
                        batch["fragment"].v_target, batch["fragment"].omega_target,
                        batch["fragment"].size, omega_weight=self.omega_weight,
                    )
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

    def _validate_rollout(self, epoch: int) -> dict[str, float]:
        """Run ODE rollout on val set and compute docking metrics."""
        if self.val_loader is None:
            return {}

        from src.inference.sampler import FlowFragSampler
        from src.inference.metrics import ligand_rmsd, centroid_distance, frag_centroid_rmsd
        from src.geometry.se3 import quaternion_to_matrix

        self.model.eval()
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model

        dcfg = self.cfg["data"]
        num_steps = self.cfg["logging"].get("rollout_steps", 20)
        sigma = dcfg.get("prior_sigma", 5.0)

        sampler = FlowFragSampler(raw_model, num_steps=num_steps, translation_sigma=sigma)

        rmsds, cent_dists, frag_rmsds = [], [], []
        n_done = 0

        # Iterate over full val dataset
        val_ds = self.val_loader.dataset
        for i in range(len(val_ds)):
            data = val_ds[i]
            result = sampler.sample(data, device=self.device)

            # Ground-truth atom positions
            frag_id = data["atom"].fragment_id.to(self.device)
            local_pos = data["atom"].local_pos.to(self.device)
            T_target = data["fragment"].T_target.to(self.device)
            q_target = getattr(data["fragment"], "q_target", None)
            if q_target is not None:
                R_target = quaternion_to_matrix(q_target.to(self.device))
                true_pos = torch.einsum("nij,nj->ni", R_target[frag_id], local_pos) + T_target[frag_id]
            else:
                true_pos = local_pos + T_target[frag_id]

            rmsds.append(ligand_rmsd(result["atom_pos_pred"], true_pos).item())
            cent_dists.append(centroid_distance(result["atom_pos_pred"], true_pos).item())
            frag_rmsds.append(frag_centroid_rmsd(result["T_pred"], T_target).item())
            n_done += 1

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
