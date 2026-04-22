#!/usr/bin/env python
"""Render a FlowFrag trajectory as PNG frames + animated GIF.

Shows the protein pocket (backbone trace + pocket-residue side chains),
the evolving ligand pose, and an optional translucent crystal ligand for
reference. Per-frame RMSD is reported in the title when a crystal is given.

Usage:
    python scripts/viz_traj.py \
        --traj outputs/traj_1gkc/traj_0.sdf \
        --protein /mnt/data/PLI/Astex-diverse-set/1gkc/1gkc_pocket.pdb \
        --crystal_ligand /mnt/data/PLI/Astex-diverse-set/1gkc/1gkc_ligand.mol2 \
        --out_dir outputs/viz_1gkc
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem.Draw import rdMolDraw2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocess.fragments import decompose_fragments

RDLogger.DisableLog("rdApp.*")


ELEMENT_COLORS = {
    "C": "#303030", "N": "#3064e8", "O": "#e63946", "S": "#f4d35e",
    "P": "#ff8000", "F": "#90e050", "Cl": "#1ff01f", "Br": "#a62929",
    "I": "#940094", "H": "#ffffff",
}
ELEMENT_RADIUS = {
    "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8,
    "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98, "H": 1.2,
}


def parse_pdb(pdb_path: Path) -> dict:
    """Parse a PDB into CA backbone trace + per-residue heavy-atom dicts."""
    residues: dict[tuple[str, int], dict] = {}
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21].strip() or "A"
            res_id = int(line[22:26])
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            element = line[76:78].strip() or atom_name[0]
        except ValueError:
            continue
        if element.upper() == "H":
            continue
        key = (chain, res_id)
        if key not in residues:
            residues[key] = {"name": res_name, "atoms": [], "elements": []}
        residues[key]["atoms"].append([x, y, z])
        residues[key]["elements"].append(element)

    ca_trace: list[list[float]] = []
    for key in sorted(residues):
        for xyz, elem, atom_name in zip(
            residues[key]["atoms"],
            residues[key]["elements"],
            [a for a in residues[key].get("_names", [])] or [],
        ):
            pass
    # Re-parse just for CA with atom names retained
    cas: list[list[float]] = []
    prev_chain = None
    chain_traces: list[list[list[float]]] = []
    current: list[list[float]] = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        chain = line[21].strip() or "A"
        if atom_name != "CA":
            continue
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        if chain != prev_chain and current:
            chain_traces.append(current)
            current = []
        prev_chain = chain
        current.append([x, y, z])
    if current:
        chain_traces.append(current)

    for key, rec in residues.items():
        rec["atoms"] = np.asarray(rec["atoms"], dtype=np.float32)
    return {"residues": residues, "chain_traces": [np.asarray(c, dtype=np.float32) for c in chain_traces]}


def load_multimol_sdf(sdf_path: Path) -> list[Chem.Mol]:
    return [m for m in Chem.SDMolSupplier(str(sdf_path), removeHs=True) if m is not None]


def mol_xyz(mol: Chem.Mol) -> np.ndarray:
    c = mol.GetConformer()
    return np.asarray(
        [[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())], dtype=np.float32,
    )


def draw_ligand(ax, mol: Chem.Mol, *, fragment_id: np.ndarray | None = None,
                frag_palette: list[str] | None = None,
                atom_size=150, edge=True) -> None:
    """Draw ligand as ball-and-stick. When fragment_id is given, atoms are
    colored by fragment and inter-fragment bonds are dashed.
    """
    xs = mol_xyz(mol)
    n_atoms = mol.GetNumAtoms()

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        same_frag = fragment_id is None or fragment_id[i] == fragment_id[j]
        ax.plot([xs[i, 0], xs[j, 0]],
                [xs[i, 1], xs[j, 1]],
                [xs[i, 2], xs[j, 2]],
                color="#222" if same_frag else "#888",
                alpha=0.9 if same_frag else 0.55,
                linewidth=2.2 if same_frag else 1.4,
                linestyle="-" if same_frag else "--",
                solid_capstyle="round", zorder=3)

    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    sizes = [atom_size * (ELEMENT_RADIUS.get(s, 1.7) / 1.7) ** 2 for s in symbols]
    if fragment_id is not None and frag_palette is not None:
        colors = [frag_palette[fragment_id[i] % len(frag_palette)] for i in range(n_atoms)]
    else:
        colors = [ELEMENT_COLORS.get(s, "#ff69b4") for s in symbols]
    ax.scatter(
        xs[:, 0], xs[:, 1], xs[:, 2],
        c=colors, s=sizes, alpha=0.95,
        edgecolor="black" if edge else "none",
        linewidth=0.6 if edge else 0, depthshade=True, zorder=4,
    )


def render_2d_fragments(
    mol: Chem.Mol,
    fragment_id: np.ndarray,
    frag_palette: list[str],
    size: tuple[int, int] = (460, 400),
) -> Image.Image:
    """RDKit 2D depiction with atoms and in-fragment bonds colored by fragment."""
    mol_2d = Chem.Mol(mol)
    mol_2d.RemoveAllConformers()
    AllChem.Compute2DCoords(mol_2d)

    atom_colors = {
        i: matplotlib.colors.to_rgb(frag_palette[fragment_id[i] % len(frag_palette)])
        for i in range(mol_2d.GetNumAtoms())
    }
    highlight_bonds: list[int] = []
    bond_colors: dict[int, tuple[float, float, float]] = {}
    for b in mol_2d.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if fragment_id[i] == fragment_id[j]:
            highlight_bonds.append(b.GetIdx())
            bond_colors[b.GetIdx()] = atom_colors[i]

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.bondLineWidth = 2
    opts.baseFontSize = 0.7
    drawer.DrawMolecule(
        mol_2d,
        highlightAtoms=list(range(mol_2d.GetNumAtoms())),
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")


def draw_fragment_trails(ax, centroid_history: list[np.ndarray],
                         frag_palette: list[str], max_tail: int = 30) -> None:
    """Draw trailing polyline per fragment, with alpha fading into the past."""
    if len(centroid_history) < 2:
        return
    recent = centroid_history[-max_tail:]
    n_frag = recent[0].shape[0]
    recent_arr = np.stack(recent, axis=0)  # [T, n_frag, 3]
    T = recent_arr.shape[0]
    for k in range(n_frag):
        color = frag_palette[k % len(frag_palette)]
        for s in range(T - 1):
            alpha = 0.15 + 0.75 * (s / max(T - 1, 1))
            ax.plot(recent_arr[s:s+2, k, 0],
                    recent_arr[s:s+2, k, 1],
                    recent_arr[s:s+2, k, 2],
                    color=color, alpha=alpha, linewidth=2.0,
                    solid_capstyle="round", zorder=3)


def draw_crystal_ghost(ax, mol: Chem.Mol) -> None:
    xs = mol_xyz(mol)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ax.plot([xs[i, 0], xs[j, 0]],
                [xs[i, 1], xs[j, 1]],
                [xs[i, 2], xs[j, 2]],
                color="#2ecc71", alpha=0.55, linewidth=2.2,
                solid_capstyle="round", zorder=2)
    ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2],
               c="#27ae60", s=35, alpha=0.35, edgecolor="none", zorder=2)


def draw_protein(ax, protein: dict, lig_xyz: np.ndarray,
                 d_near: float = 2.5, d_far: float = 7.0) -> None:
    """Show every protein heavy atom, colored by distance to the current
    ligand pose. Close contacts glow hot (red/orange), far atoms fade to a
    faint grey — so pocket atoms "blink" as the ligand approaches them.
    """
    atoms = np.concatenate([r["atoms"] for r in protein["residues"].values()
                             if r["atoms"].size > 0], axis=0)
    if atoms.size == 0:
        return
    d = np.linalg.norm(atoms[:, None, :] - lig_xyz[None, :, :], axis=-1).min(axis=1)

    norm = np.clip((d - d_near) / (d_far - d_near), 0.0, 1.0)  # 0 close, 1 far
    cmap = matplotlib.colormaps["YlOrRd"]
    # Flip: close → hot end of colormap (1.0), far → light end (0.1)
    colors = cmap(1.0 - norm * 0.9)

    alphas = np.where(d < d_far, 0.35 + 0.55 * (1 - norm), 0.08)
    sizes = np.where(d < d_far, 12 + 55 * (1 - norm) ** 2, 6)

    colors[:, 3] = alphas
    ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2],
               c=colors, s=sizes, edgecolor="none",
               depthshade=True, zorder=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=str, required=True,
                    help="Multi-conformer SDF with trajectory frames")
    ap.add_argument("--protein", type=str, required=True, help="Pocket / full-protein PDB")
    ap.add_argument("--crystal_ligand", type=str, default=None,
                    help="Optional crystal ligand (mol2/sdf) — drawn as green ghost + RMSD in title")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--pad", type=float, default=4.0, help="Padding (Å) around ligand bbox")
    ap.add_argument("--contact_near", type=float, default=2.5,
                    help="Distance (Å) below which protein atoms glow brightest")
    ap.add_argument("--contact_far", type=float, default=7.0,
                    help="Distance (Å) above which protein atoms fade to faint")
    ap.add_argument("--trail_length", type=int, default=30,
                    help="Number of past frames to draw in fragment centroid trails")
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--append_mmff", action="store_true", default=True,
                    help="Append one MMFF-refined frame after the ODE trajectory "
                         "(position-restrained, local strain only)")
    ap.add_argument("--no_mmff", dest="append_mmff", action="store_false",
                    help="Disable the trailing MMFF frame")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    protein = parse_pdb(Path(args.protein))
    frames = load_multimol_sdf(Path(args.traj))
    assert frames, f"No frames in {args.traj}"

    crystal = None
    if args.crystal_ligand:
        cp = Path(args.crystal_ligand)
        if cp.suffix.lower() == ".mol2":
            crystal = Chem.MolFromMol2File(str(cp), removeHs=True)
            if crystal is None:
                crystal = Chem.MolFromMol2File(str(cp), sanitize=False, removeHs=False)
                crystal = Chem.RemoveHs(crystal, sanitize=False) if crystal else None
        else:
            crystal = next(iter(Chem.SDMolSupplier(str(cp), removeHs=True)), None)

    # Bounding box from all ligand frames + crystal
    all_xyz = np.concatenate(
        [mol_xyz(m) for m in frames] + ([mol_xyz(crystal)] if crystal is not None else []),
        axis=0,
    )
    center = (all_xyz.min(0) + all_xyz.max(0)) / 2
    half = (all_xyz.max(0) - all_xyz.min(0)).max() / 2 + args.pad
    xlim = (center[0] - half, center[0] + half)
    ylim = (center[1] - half, center[1] + half)
    zlim = (center[2] - half, center[2] + half)

    # --- Fragment decomposition from the first frame (topology-invariant) ---
    ref_frame = frames[0]
    try:
        frag_res = decompose_fragments(ref_frame, torch.tensor(mol_xyz(ref_frame)))
        fragment_id = frag_res["fragment_id"].numpy() if frag_res is not None else None
    except Exception:
        fragment_id = None
    n_frag = int(fragment_id.max()) + 1 if fragment_id is not None else 0

    base_palette = [
        "#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6",
        "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
    ]
    frag_palette = [base_palette[i % len(base_palette)] for i in range(max(n_frag, 1))]

    # --- Pre-compute per-frame RMSD and fragment centroids ---
    rmsd_history: list[float] = []
    if crystal is not None:
        for m in frames:
            try:
                rmsd_history.append(float(rdMolAlign.CalcRMS(m, crystal)))
            except Exception:
                rmsd_history.append(float("nan"))

    centroid_history: list[np.ndarray] = []
    if fragment_id is not None:
        for m in frames:
            xs = mol_xyz(m)
            centroids = np.zeros((n_frag, 3), dtype=np.float32)
            for k in range(n_frag):
                mask = fragment_id == k
                if mask.any():
                    centroids[k] = xs[mask].mean(0)
            centroid_history.append(centroids)

    # --- 2D ligand sketch (topology-invariant, render once) ---
    ligand_2d_img = None
    if fragment_id is not None:
        try:
            ligand_2d_img = render_2d_fragments(ref_frame, fragment_id, frag_palette)
        except Exception:
            ligand_2d_img = None

    # --- Optional MMFF-refined final frame (position-restrained, local strain only) ---
    mmff_frame: Chem.Mol | None = None
    if args.append_mmff:
        try:
            final_mol = Chem.RWMol(frames[-1])
            mol_h = Chem.AddHs(final_mol, addCoords=True)
            props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=0) if props is not None else None
            if ff is not None:
                for j in range(final_mol.GetNumAtoms()):
                    ff.MMFFAddPositionConstraint(j, 0.5, 50.0)
                ff.Minimize(maxIts=200)
                mmff_frame = Chem.RemoveHs(mol_h)
        except Exception:
            mmff_frame = None
    if mmff_frame is not None:
        frames.append(mmff_frame)
        if crystal is not None:
            try:
                rmsd_history.append(float(rdMolAlign.CalcRMS(mmff_frame, crystal)))
            except Exception:
                rmsd_history.append(rmsd_history[-1] if rmsd_history else float("nan"))
        if fragment_id is not None:
            xs = mol_xyz(mmff_frame)
            centroids = np.zeros((n_frag, 3), dtype=np.float32)
            for k in range(n_frag):
                mask = fragment_id == k
                if mask.any():
                    centroids[k] = xs[mask].mean(0)
            centroid_history.append(centroids)

    # --- Figure layout: 2x2 3D views + 2D inset + RMSD inset + progress bar ---
    views = [
        ("front",     args.elev, args.azim),
        ("side",      args.elev, args.azim + 90.0),
        ("top",       85.0,      args.azim),
        ("isometric", 35.0,      args.azim + 45.0),
    ]
    fig = plt.figure(figsize=(10.5, 10.5), dpi=120, facecolor="white")

    axes_3d: list = []
    positions = [(0.02, 0.50, 0.48, 0.45),
                 (0.50, 0.50, 0.48, 0.45),
                 (0.02, 0.05, 0.48, 0.45),
                 (0.50, 0.05, 0.48, 0.45)]
    for (name, elev, azim), pos in zip(views, positions):
        ax = fig.add_axes(pos, projection="3d")
        axes_3d.append((name, ax, elev, azim))

    ax_rmsd = fig.add_axes((0.80, 0.945, 0.18, 0.05)) if rmsd_history else None
    ax_2d = fig.add_axes((0.02, 0.94, 0.17, 0.055)) if ligand_2d_img is not None else None
    ax_title = fig.add_axes((0.22, 0.94, 0.55, 0.05))
    ax_title.set_xticks([]); ax_title.set_yticks([])
    ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
    for side in ("top", "right", "left", "bottom"):
        ax_title.spines[side].set_visible(False)
    ax_bar = fig.add_axes((0.08, 0.015, 0.84, 0.012))
    ax_bar.set_xlim(0, 1); ax_bar.set_ylim(0, 1)
    ax_bar.set_xticks([]); ax_bar.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax_bar.spines[side].set_visible(False)

    n_ode = len(frames) - (1 if mmff_frame is not None else 0)

    def draw_3d_panel(ax, m, i: int, title: str, show_trails: bool = True) -> None:
        ax.cla()
        ax.set_facecolor("white")
        for spine_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            spine_axis.pane.set_edgecolor("#e5e7eb")
            spine_axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        draw_protein(ax, protein, mol_xyz(m),
                     d_near=args.contact_near, d_far=args.contact_far)
        if crystal is not None:
            draw_crystal_ghost(ax, crystal)
        if show_trails and centroid_history:
            draw_fragment_trails(ax, centroid_history[: i + 1], frag_palette,
                                 max_tail=args.trail_length)
        draw_ligand(ax, m, fragment_id=fragment_id, frag_palette=frag_palette)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_box_aspect((1, 1, 1))
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
        ax.text2D(0.04, 0.94, title, transform=ax.transAxes,
                  fontsize=10, fontfamily="monospace", color="#6b7280",
                  verticalalignment="top")

    # 2D ligand inset (static)
    if ax_2d is not None:
        ax_2d.imshow(np.asarray(ligand_2d_img))
        ax_2d.set_xticks([]); ax_2d.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax_2d.spines[side].set_visible(False)
        ax_2d.set_title(f"ligand ({n_frag} frags)", fontsize=8, pad=2)

    pngs: list[Image.Image] = []
    for i, m in enumerate(frames):
        is_mmff = mmff_frame is not None and i == len(frames) - 1
        if is_mmff:
            t = 1.0
            top_label = f"MMFF refine (post-ODE)   RMSD = {rmsd_history[i]:.2f} Å"
            panel_accent = "#059669"
        else:
            t = i / max(n_ode - 1, 1)
            top_label = f"frame {i+1:>2d}/{n_ode}   t={t:.2f}"
            if rmsd_history:
                top_label += f"   RMSD = {rmsd_history[i]:.2f} Å"
            if n_frag:
                top_label += f"   frags = {n_frag}"
            panel_accent = None

        ax_title.cla()
        ax_title.set_xticks([]); ax_title.set_yticks([])
        ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
        for side in ("top", "right", "left", "bottom"):
            ax_title.spines[side].set_visible(False)
        ax_title.text(0.5, 0.5, top_label, ha="center", va="center",
                      fontsize=12, fontfamily="monospace",
                      color=panel_accent or "#111827")

        for (name, ax_v, elev, azim) in axes_3d:
            draw_3d_panel(ax_v, m, i, name)
            ax_v.view_init(elev=elev, azim=azim)
            if is_mmff:
                for spine_axis in (ax_v.xaxis, ax_v.yaxis, ax_v.zaxis):
                    spine_axis.pane.set_edgecolor("#10b981")

        # RMSD inset
        if ax_rmsd is not None:
            ax_rmsd.cla()
            xs = np.arange(len(rmsd_history)) / max(len(rmsd_history) - 1, 1)
            ax_rmsd.plot(xs, rmsd_history, color="#d1d5db", linewidth=1.0)
            ax_rmsd.plot(xs[: i + 1], rmsd_history[: i + 1],
                         color="#ef4444", linewidth=1.6)
            ax_rmsd.scatter([xs[i]], [rmsd_history[i]],
                            color="#ef4444" if not is_mmff else "#059669",
                            s=18, zorder=5, edgecolor="white", linewidth=0.8)
            ax_rmsd.set_xlim(0, 1)
            ax_rmsd.set_ylim(0, max(rmsd_history) * 1.1 or 1)
            ax_rmsd.set_xticks([0, 1])
            ax_rmsd.set_yticks([0, round(max(rmsd_history), 1)])
            ax_rmsd.tick_params(axis="both", labelsize=6, length=2, pad=1)
            ax_rmsd.set_title("RMSD", fontsize=7, pad=2)
            for side in ("top", "right"):
                ax_rmsd.spines[side].set_visible(False)
            ax_rmsd.set_facecolor((1, 1, 1, 0.85))

        # Progress bar
        ax_bar.cla()
        ax_bar.set_xlim(0, 1); ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks([]); ax_bar.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax_bar.spines[side].set_visible(False)
        bar_color = "#059669" if is_mmff else "#ef4444"
        ax_bar.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#e5e7eb",
                                        edgecolor="none"))
        ax_bar.add_patch(plt.Rectangle((0, 0), t, 1, facecolor=bar_color,
                                        edgecolor="none"))
        label = "MMFF" if is_mmff else f"t={t:.2f}"
        ax_bar.text(1.01, 0.5, label, fontsize=8,
                     va="center", ha="left", fontfamily="monospace",
                     transform=ax_bar.transAxes, color=bar_color)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        buf.seek(0)
        pngs.append(Image.open(buf).convert("RGB"))

    gif_path = out_dir / "trajectory.gif"
    per_frame_ms = int(1000 / args.fps)
    # Hold the last ODE frame + MMFF frame a few extra beats so the viewer
    # can see the "after" state rather than it flashing past.
    durations = [per_frame_ms] * len(pngs)
    if mmff_frame is not None and len(durations) >= 2:
        durations[-2] = per_frame_ms * 3   # last ODE frame
        durations[-1] = per_frame_ms * 6   # MMFF-refined frame
    elif durations:
        durations[-1] = per_frame_ms * 4   # hold final pose when no MMFF
    pngs[0].save(
        gif_path, save_all=True, append_images=pngs[1:],
        duration=durations, loop=0, optimize=True,
    )
    plt.close(fig)
    print(f"Rendered {len(frames)} frames -> {out_dir}")
    print(f"GIF: {gif_path}")


if __name__ == "__main__":
    main()
