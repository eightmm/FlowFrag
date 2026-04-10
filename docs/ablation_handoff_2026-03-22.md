# Ablation Handoff

Snapshot time: 2026-03-22 16:18:36 KST

Branch: `feat/unified-model`

This note captures the unified-model ablation state that was being driven from
Claude Code on 2026-03-22.

## Current sequential run

- Wrapper script: `/tmp/run_ablations.sh`
- Nohup log: `outputs/logs/ablations.log`
- Process chain seen at snapshot:
  - wrapper: PID `56945`
  - `uv run`: PID `65722`
  - trainer: PID `65731`

The script order is:

1. `configs/abl_prune_caca.yaml`
2. `configs/abl_rt_initonly.yaml`
3. `configs/abl_layers2.yaml`

Log markers in `outputs/logs/ablations.log`:

- line 1: `=== P1b: Prune CA-CA ===`
- line 2028: `Epoch 999 done ... cos_v=0.898 cos_w=0.672 (1.6s)`
- line 2064: `=== P2: R_t init-only ===`

At this snapshot, `P1b` is finished and `P2` is actively running. The latest
observed log chunk was around `E679` of `P2`, with epoch time about `2.1s`.

## Baseline and recent runs

The quick ablations were launched against the current unified baseline family,
not the legacy FlowFrag configs.

### Baseline used for current ablation pass

- Config: `configs/overfit_unified_ne_contact.yaml`
- W&B local run: `wandb/run-20260322_141210-96emhwdk`
- Output: `outputs/overfit_unified_ne_contact`
- 2000-epoch summary:
  - `epoch/cos_v = 0.9260`
  - `epoch/cos_omega = 0.7058`
  - `epoch/loss = 1.4972`

### Finished quick ablation

- Config: `configs/abl_prune_caca.yaml`
- W&B local run: `wandb/run-20260322_152657-f60ml136`
- Output: `outputs/abl_prune_caca`
- 1000-epoch summary:
  - `epoch/cos_v = 0.8976`
  - `epoch/cos_omega = 0.6721`
  - `epoch/loss = 1.6928`
  - `~1.6s/epoch`

Claude session notes explicitly treated `P1b` as promising because it was
faster and looked non-harmful to slightly helpful at matched early epochs.

### Rollout quick compare done after this handoff

On 2026-03-22, a direct unified rollout comparison was run locally between:

- baseline: `configs/overfit_unified_ne_contact.yaml`
- prune: `configs/abl_prune_caca.yaml`

Comparison protocol:

- overfit first 16 complexes
- 3 prior seeds per complex
- 25 ODE steps
- `late` schedule with power `3`

Observed summary:

- baseline:
  - mean RMSD `1.875A`
  - median of sample means `1.440A`
  - `<2A` success `68.8%`
- prune CA-CA:
  - mean RMSD `2.356A`
  - median of sample means `1.936A`
  - `<2A` success `50.0%`

Head-to-head on sample-mean RMSD:

- `prune_caca` better on `0 / 16` complexes
- worst observed delta was `+0.884A`

Important caveat:

- This is not a matched-training-step comparison.
- baseline checkpoint is a `2000`-epoch run, while `prune_caca` checkpoint is a
  `1000`-epoch quick ablation run.
- Still, it is strong evidence that `P1b` should not be promoted based only on
  pointwise `cos/loss` without a longer rollout-validated rerun.

### Active quick ablation

- Config: `configs/abl_rt_initonly.yaml`
- W&B local run: `wandb/run-20260322_155356-85ln5lx6`
- Output: `outputs/abl_rt_initonly`
- Started at: 2026-03-22 15:53:56 KST

### Queued next

- Config: `configs/abl_layers2.yaml`
- Output: `outputs/abl_layers2`

## Claude Code context

Most relevant Claude session file:

- `~/.claude/projects/-home-jaemin-project-protein-ligand-flowfrag/fb47f7ee-9337-46d7-8a3b-de6bcff16196.jsonl`

Relevant Claude memory files:

- `~/.claude/projects/-home-jaemin-project-protein-ligand-flowfrag/memory/MEMORY.md`
- `~/.claude/projects/-home-jaemin-project-protein-ligand-flowfrag/memory/project_omega_progress.md`
- `~/.claude/projects/-home-jaemin-project-protein-ligand-flowfrag/memory/project_omega_analytic.md`

Important points recovered from the Claude session on 2026-03-22:

- The 3-run quick screen was intentionally set to `1000` epochs for fast
  directionality checks.
- Decision rule was: use these runs to decide whether a change is harmful or
  clearly helpful, then rerun the chosen combination for `2000+` epochs.
- Claude’s stated pending decisions after the 3-run screen:
  1. Keep or discard CA-CA pruning.
  2. Keep per-layer `R_t` injection or switch to `init_only`.
  3. Keep `4` layers or drop to `2`.

## Remaining work after P2 and P4 finish

1. Compare `P2` against the per-layer baseline.
2. Compare `P4` against the 4-layer baseline.
3. Decide the best combined config.
4. Optional follow-up: `P1c` prune `protein_bond` if `P1b` still looks good.
5. Rerun the chosen best config for `2000+` epochs.
6. Only after architecture is fixed:
   - `P5` t-decile diagnostics
   - `P6` omega-weight sweep
   - `P7` Muon
   - full training

## Notes

- `docs/CURRENT_STATUS.md` and `CLAUDE.md` are older and do not reflect this
  2026-03-22 ablation pass.
- The worktree currently contains both unified-model additions and cleanup of
  many legacy configs. Do not blindly revert the deletions without checking why
  they were removed.
