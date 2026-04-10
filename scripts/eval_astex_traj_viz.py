#!/usr/bin/env python
"""Evaluate FlowFrag on Astex Diverse Set and render per-complex trajectory GIFs.

Each GIF shows:
  - ODE trajectory frames (prior -> docked, fragment-colored)
  - MMFF-minimized final pose appended as the last frame

Also outputs a JSON summary comparing RMSD before/after MMFF.

Usage:
    python scripts/eval_astex_traj_viz.py \
        --astex_dir /mnt/data/PLI/Astex-diverse-set \
        --checkpoint outputs/train_unified_ne_contact_adamw_1000/checkpoints/latest.pt \
        --config configs/train_unified_ne_contact_adamw_1000.yaml \
        --out_dir outputs/eval_astex_viz

    # Subset for quick test
    python scripts/eval_astex_traj_viz.py \
        --astex_dir /mnt/data/PLI/Astex-diverse-set \
        --checkpoint outputs/train_unified_ne_contact_adamw_1000/checkpoints/latest.pt \
        --config configs/train_unified_ne_contact_adamw_1000.yaml \
        --pdb_ids 1a28 1b9v 1e66 \
        --out_dir outputs/eval_astex_viz
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dock import preprocess_complex, sample_unified
from scripts.eval_benchmark import load_mol2_robust, mmff_refine, compute_rmsd


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

FRAG_COLORS = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
    "#42D4F4", "#F032E6", "#BFEF45", "#FABED4", "#469990",
    "#DCBEFF", "#9A6324", "#800000", "#AAFFC3", "#808000",
]


def load_protein_ca(pdb_path: Path) -> np.ndarray:
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(coords, dtype=np.float32)


def fig_to_image(fig: plt.Figure, dpi: int = 100) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def render_frame(
    ax: plt.Axes,
    atom_pos: np.ndarray,
    frag_id: np.ndarray,
    prot_nearby: np.ndarray,
    ref_pos: np.ndarray | None,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    elev: float = 25,
    azim: float = 45,
    mmff_frame: bool = False,
) -> None:
    ax.cla()
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    # Protein CA (faint grey)
    if prot_nearby is not None and len(prot_nearby) > 0:
        ax.scatter(
            prot_nearby[:, 0], prot_nearby[:, 1], prot_nearby[:, 2],
            c="#CCCCCC", s=8, alpha=0.25, depthshade=True,
        )

    # Crystal ligand (transparent green)
    if ref_pos is not None:
        ax.scatter(
            ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2],
            c="#00CC00", s=30, alpha=0.2, edgecolors="#009900", linewidths=0.5,
        )

    # Predicted ligand: fragment-colored
    n_atoms = atom_pos.shape[0]
    colors = [FRAG_COLORS[int(frag_id[i]) % len(FRAG_COLORS)] for i in range(n_atoms)]
    edge_color = "white" if mmff_frame else "black"
    edge_lw = 1.0 if mmff_frame else 0.5
    size = 80 if mmff_frame else 60

    ax.scatter(
        atom_pos[:, 0], atom_pos[:, 1], atom_pos[:, 2],
        c=colors, s=size, alpha=0.95, edgecolors=edge_color, linewidths=edge_lw,
        depthshade=True,
    )

    title_color = "#8B0000" if mmff_frame else "black"
    ax.set_title(title, fontsize=10, color=title_color)
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.tick_params(labelsize=6)


def render_traj_gif(
    traj: list[torch.Tensor],
    traj_times: list[float],
    mmff_final: torch.Tensor | None,
    frag_id: np.ndarray,
    prot_ca: np.ndarray,
    pocket_center: np.ndarray,
    ref_pos_pc: np.ndarray | None,
    out_path: Path,
    rmsd_raw: float,
    rmsd_mmff: float | None,
    fps: int = 8,
    elev: float = 25,
    azim: float = 45,
) -> None:
    """Render ODE trajectory + MMFF final frame to GIF."""
    # Axis limits
    all_pos = np.concatenate(
        [f.numpy() if isinstance(f, torch.Tensor) else f for f in traj], axis=0
    )
    if mmff_final is not None:
        mmff_np = mmff_final.numpy() if isinstance(mmff_final, torch.Tensor) else mmff_final
        all_pos = np.concatenate([all_pos, mmff_np], axis=0)
    if ref_pos_pc is not None:
        all_pos = np.concatenate([all_pos, ref_pos_pc], axis=0)

    center = all_pos.mean(axis=0)
    spread = max(float(np.abs(all_pos - center).max()) + 2.0, 5.0)

    prot_pc = prot_ca - pocket_center
    dists = np.linalg.norm(prot_pc - center, axis=-1)
    prot_nearby = prot_pc[dists < spread + 5]

    xlim = (center[0] - spread, center[0] + spread)
    ylim = (center[1] - spread, center[1] + spread)
    zlim = (center[2] - spread, center[2] + spread)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    images: list[Image.Image] = []
    n_traj = len(traj)

    for i, (frame, t) in enumerate(zip(traj, traj_times)):
        pos = frame.numpy() if isinstance(frame, torch.Tensor) else frame
        rmsd_str = ""
        if ref_pos_pc is not None:
            r = float(np.sqrt(np.mean(np.sum((pos - ref_pos_pc) ** 2, axis=-1))))
            rmsd_str = f"  RMSD={r:.2f}\u00c5"
        title = f"ODE  step {i}/{n_traj - 1}  t={t:.3f}{rmsd_str}"
        render_frame(ax, pos, frag_id, prot_nearby, ref_pos_pc, title,
                     xlim, ylim, zlim, elev=elev, azim=azim, mmff_frame=False)
        images.append(fig_to_image(fig))

    # MMFF final frame
    if mmff_final is not None:
        mmff_np = mmff_final.numpy() if isinstance(mmff_final, torch.Tensor) else mmff_final
        mmff_rmsd_str = f"  RMSD={rmsd_mmff:.2f}\u00c5" if rmsd_mmff is not None else ""
        title = f"MMFF refined{mmff_rmsd_str}  (raw={rmsd_raw:.2f}\u00c5)"
        render_frame(ax, mmff_np, frag_id, prot_nearby, ref_pos_pc, title,
                     xlim, ylim, zlim, elev=elev, azim=azim, mmff_frame=True)
        images.append(fig_to_image(fig))

    plt.close(fig)

    ms = 1000 // fps
    durations = [ms] * len(images)
    durations[0] = 1500    # hold prior longer
    durations[-1] = 3000   # hold MMFF final longest
    if mmff_final is not None and len(images) >= 2:
        durations[-2] = 1500   # hold ODE final before MMFF

    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_path, save_all=True, append_images=images[1:],
        duration=durations, loop=0, optimize=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Astex eval + trajectory GIF visualization")
    parser.add_argument("--astex_dir", type=str, default="/mnt/data/PLI/Astex-diverse-set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--time_schedule", type=str, default="late")
    parser.add_argument("--schedule_power", type=float, default=3.0)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs/eval_astex_viz")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--no_gif", action="store_true", help="Skip GIF rendering (stats only)")
    parser.add_argument("--pdb_ids", nargs="*", default=None,
                        help="Subset of PDB IDs (default: all)")
    parser.add_argument("--gif_ids", nargs="*", default=None,
                        help="Subset of PDB IDs to render GIFs for (default: all)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    astex_dir = Path(args.astex_dir)
    out_dir = Path(args.out_dir)
    gif_dir = out_dir / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load model
    from src.models.unified import UnifiedFlowFrag
    model_cfg = dict(cfg["model"])
    model_cfg.pop("model_type", None)
    model = UnifiedFlowFrag(**model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    step = ckpt.get("step", "?")
    print(f"Loaded: {args.checkpoint}  (step={step}, device={device})")

    sigma = args.sigma if args.sigma is not None else cfg["data"].get("prior_sigma", 1.0)

    all_pdb_ids = sorted([d.name for d in astex_dir.iterdir() if d.is_dir()])
    pdb_ids = args.pdb_ids if args.pdb_ids else all_pdb_ids
    gif_ids = set(args.gif_ids) if args.gif_ids else None
    print(f"Processing {len(pdb_ids)} complexes  (sigma={sigma}, steps={args.num_steps})\n")

    results = []
    failures = []
    t_start = time.time()

    for idx, pdb_id in enumerate(pdb_ids):
        complex_dir = astex_dir / pdb_id
        pocket_pdb = complex_dir / f"{pdb_id}_pocket.pdb"
        ligand_mol2 = complex_dir / f"{pdb_id}_ligand.mol2"

        if not pocket_pdb.exists() or not ligand_mol2.exists():
            print(f"[{idx+1:3d}/{len(pdb_ids)}] {pdb_id}: SKIP (missing files)")
            failures.append({"pdb_id": pdb_id, "error": "missing files"})
            continue

        try:
            mol = load_mol2_robust(ligand_mol2)
            graph, lig_data, meta = preprocess_complex(pocket_pdb, mol, ligand_has_pose=True)
            pocket_center = meta["pocket_center"]
            ref_pos = lig_data["atom_coords"] - pocket_center

            do_gif = not args.no_gif and (gif_ids is None or pdb_id in gif_ids)

            result = sample_unified(
                model, graph, lig_data, meta,
                num_steps=args.num_steps,
                translation_sigma=sigma,
                time_schedule=args.time_schedule,
                schedule_power=args.schedule_power,
                device=device,
                save_traj=do_gif,
            )

            raw_pos = result["atom_pos_pred"]
            mmff_pos = mmff_refine(mol, raw_pos, pocket_center)

            rmsd_raw = compute_rmsd(raw_pos, ref_pos)
            rmsd_mmff = compute_rmsd(mmff_pos, ref_pos)

            status = "OK" if min(rmsd_raw, rmsd_mmff) < 2.0 else "  "
            delta = rmsd_mmff - rmsd_raw
            print(
                f"[{idx+1:3d}/{len(pdb_ids)}] {pdb_id}: "
                f"raw={rmsd_raw:.2f}  mmff={rmsd_mmff:.2f} ({delta:+.2f})  "
                f"atoms={meta['num_atom']:3d}  frags={meta['num_frag']:2d}  {status}"
            )

            results.append({
                "pdb_id": pdb_id,
                "rmsd_raw": rmsd_raw,
                "rmsd_mmff": rmsd_mmff,
                "rmsd_delta": rmsd_mmff - rmsd_raw,
                "n_atoms": meta["num_atom"],
                "n_frags": meta["num_frag"],
            })

            if do_gif:
                frag_id_np = lig_data["fragment_id"].numpy().astype(np.int64)
                prot_ca = load_protein_ca(pocket_pdb)
                ref_np = ref_pos.numpy() if isinstance(ref_pos, torch.Tensor) else ref_pos

                gif_path = gif_dir / f"{pdb_id}.gif"
                render_traj_gif(
                    traj=result["traj"],
                    traj_times=result["traj_times"],
                    mmff_final=mmff_pos,
                    frag_id=frag_id_np,
                    prot_ca=prot_ca,
                    pocket_center=pocket_center.numpy(),
                    ref_pos_pc=ref_np,
                    out_path=gif_path,
                    rmsd_raw=rmsd_raw,
                    rmsd_mmff=rmsd_mmff,
                    fps=args.fps,
                )
                size_kb = gif_path.stat().st_size / 1024
                print(f"           GIF: {gif_path.name} ({size_kb:.0f} KB)")

        except Exception as e:
            print(f"[{idx+1:3d}/{len(pdb_ids)}] {pdb_id}: FAIL ({e})")
            failures.append({"pdb_id": pdb_id, "error": str(e)})
            traceback.print_exc()

    elapsed = time.time() - t_start

    print(f"\n{'='*65}")
    print("Astex Evaluation — Raw ODE vs MMFF Refined")
    print(f"{'='*65}")

    if results:
        rmsds_raw = np.array([r["rmsd_raw"] for r in results])
        rmsds_mmff = np.array([r["rmsd_mmff"] for r in results])

        print(f"Complexes:  {len(results)} / {len(pdb_ids)}  ({len(failures)} failed)")
        print(f"Time:       {elapsed:.1f}s  ({elapsed/max(len(pdb_ids),1):.1f}s per complex)\n")

        print(f"{'':12s}  {'Raw ODE':>10s}  {'MMFF':>10s}  {'Delta':>10s}")
        print(f"{'Mean':12s}  {rmsds_raw.mean():10.3f}  {rmsds_mmff.mean():10.3f}  "
              f"{(rmsds_mmff - rmsds_raw).mean():+10.3f}")
        print(f"{'Median':12s}  {np.median(rmsds_raw):10.3f}  {np.median(rmsds_mmff):10.3f}  "
              f"{np.median(rmsds_mmff - rmsds_raw):+10.3f}")
        print(f"{'Std':12s}  {rmsds_raw.std():10.3f}  {rmsds_mmff.std():10.3f}")

        print(f"\nSuccess Rate:")
        print(f"{'Threshold':12s}  {'Raw ODE':>14s}  {'MMFF':>14s}")
        for thr in [1.0, 2.0, 3.0, 5.0]:
            r_raw = (rmsds_raw < thr).mean() * 100
            r_mmff = (rmsds_mmff < thr).mean() * 100
            n_raw = int((rmsds_raw < thr).sum())
            n_mmff = int((rmsds_mmff < thr).sum())
            print(f"< {thr:.0f}A{'':9s}  "
                  f"{r_raw:6.1f}% ({n_raw:2d}/{len(results)})  "
                  f"{r_mmff:6.1f}% ({n_mmff:2d}/{len(results)})")

        deltas = rmsds_mmff - rmsds_raw
        print(f"\nMMFF effect:  improved={int((deltas < -0.05).sum())}  "
              f"neutral={int((np.abs(deltas) <= 0.05).sum())}  "
              f"degraded={int((deltas > 0.05).sum())}")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for ff in failures:
            print(f"  {ff['pdb_id']}: {ff['error']}")

    summary = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "num_steps": args.num_steps,
        "sigma": sigma,
        "time_schedule": args.time_schedule,
        "results": results,
        "failures": failures,
    }
    if results:
        summary["stats"] = {
            "mean_rmsd_raw": float(rmsds_raw.mean()),
            "median_rmsd_raw": float(np.median(rmsds_raw)),
            "mean_rmsd_mmff": float(rmsds_mmff.mean()),
            "median_rmsd_mmff": float(np.median(rmsds_mmff)),
            "success_2A_raw": float((rmsds_raw < 2.0).mean()),
            "success_2A_mmff": float((rmsds_mmff < 2.0).mean()),
            "success_5A_raw": float((rmsds_raw < 5.0).mean()),
            "success_5A_mmff": float((rmsds_mmff < 5.0).mean()),
        }

    out_file = out_dir / "results.json"
    with open(out_file, "w") as ff:
        json.dump(summary, ff, indent=2)
    print(f"\nResults saved: {out_file}")
    if not args.no_gif:
        n_gifs = len(list(gif_dir.glob("*.gif")))
        print(f"GIFs saved:   {gif_dir}  ({n_gifs} files)")


if __name__ == "__main__":
    main()
