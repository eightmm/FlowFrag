"""Visualize 3 best + 3 worst PB cases (pocket + crystal vs predicted)."""

import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DATA_ROOT = Path("/mnt/data/PLI/PoseBusters/posebusters_benchmark_set")
POSE_DIR  = Path("/home/jaemin/project/protein-ligand/flowfrag/outputs/eval_posebusters_phys_25s/poses_phys")
RESULTS   = Path("/home/jaemin/project/protein-ligand/flowfrag/outputs/conf_v3_eval/pb_det/results.json")
OUT       = Path("/home/jaemin/project/protein-ligand/flowfrag/outputs/viz_pb")
OUT.mkdir(parents=True, exist_ok=True)

BEST  = ["7ZU2_DHT", "7FB7_8NF", "7R59_I5F"]
WORST = ["7M6K_YRJ", "8F4J_PHO", "7ZZW_KKW"]


def load_protein_heavy_atoms(pdb_path: Path) -> np.ndarray:
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            elem = line[76:78].strip()
            if not elem:
                elem = line[12:16].strip()[0]
            if elem == "H":
                continue
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            coords.append((x, y, z))
    return np.asarray(coords, dtype=np.float32)


def load_ligand(sdf_path: Path):
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        return None, None
    conf = mol.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    return pos, bonds


def per_pose_rmsd(raw_poses, ref_pos, dock_idx) -> np.ndarray:
    ref = ref_pos.numpy() if hasattr(ref_pos, "numpy") else np.asarray(ref_pos)
    didx = np.asarray(dock_idx, dtype=np.int64)
    out = []
    for p in raw_poses:
        p = p.numpy() if hasattr(p, "numpy") else np.asarray(p)
        d = p[didx] - ref
        out.append(float(np.sqrt((d * d).sum(-1).mean())))
    return np.asarray(out)


def draw_complex(ax, prot_xyz, lig_crystal, lig_pred, bonds, title, success):
    # Center on crystal centroid
    center = lig_crystal.mean(0)
    # Pocket: protein atoms within 10 Å of any crystal ligand atom
    if prot_xyz.size > 0:
        d2 = ((prot_xyz[:, None, :] - lig_crystal[None, :, :]) ** 2).sum(-1)
        pocket_mask = d2.min(axis=1) < (10.0 ** 2)
        pocket = prot_xyz[pocket_mask]
        ax.scatter(pocket[:, 0], pocket[:, 1], pocket[:, 2], c="lightgray", s=4, alpha=0.35, depthshade=False)

    # Crystal ligand (blue)
    ax.scatter(lig_crystal[:, 0], lig_crystal[:, 1], lig_crystal[:, 2], c="#1f77b4", s=40, edgecolors="navy", linewidths=0.5, label="crystal")
    for i, j in bonds:
        if i < len(lig_crystal) and j < len(lig_crystal):
            ax.plot([lig_crystal[i,0], lig_crystal[j,0]], [lig_crystal[i,1], lig_crystal[j,1]], [lig_crystal[i,2], lig_crystal[j,2]], c="#1f77b4", lw=2.0)

    # Predicted (red for fail, green for success)
    pcol = "#2ca02c" if success else "#d62728"
    ax.scatter(lig_pred[:, 0], lig_pred[:, 1], lig_pred[:, 2], c=pcol, s=40, edgecolors="black", linewidths=0.5, label="predicted")
    for i, j in bonds:
        if i < len(lig_pred) and j < len(lig_pred):
            ax.plot([lig_pred[i,0], lig_pred[j,0]], [lig_pred[i,1], lig_pred[j,1]], [lig_pred[i,2], lig_pred[j,2]], c=pcol, lw=1.8, ls="--", alpha=0.85)

    # Bounds: zoom on ligand region
    all_lig = np.vstack([lig_crystal, lig_pred])
    pad = 4.0
    mn = all_lig.min(0) - pad
    mx = all_lig.max(0) + pad
    span = (mx - mn).max()
    cx, cy, cz = (mn + mx) / 2
    ax.set_xlim(cx - span/2, cx + span/2)
    ax.set_ylim(cy - span/2, cy + span/2)
    ax.set_zlim(cz - span/2, cz + span/2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=10)


def main():
    results = json.load(open(RESULTS))
    oracle_rows = {x["pdb_id"]: x for x in results["per_complex"]["none+oracle"]}
    conf_rows   = {x["pdb_id"]: x for x in results["per_complex"]["none+confidence"]}

    cases = [(pid, True) for pid in BEST] + [(pid, False) for pid in WORST]

    fig = plt.figure(figsize=(15, 10))
    for k, (pid, success) in enumerate(cases):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        try:
            pdb = DATA_ROOT / pid / f"{pid}_protein.pdb"
            sdf = DATA_ROOT / pid / f"{pid}_ligand.sdf"
            pose_pt = POSE_DIR / f"{pid}.pt"
            prot_xyz = load_protein_heavy_atoms(pdb)
            lig_crystal, bonds = load_ligand(sdf)
            d = torch.load(pose_pt, map_location="cpu", weights_only=False)
            ref_pos     = d["ref_pos"]
            dock_idx    = d["dock_idx"]
            pocket_cen  = d["pocket_center"].numpy()
            raw_poses   = d["raw_poses"]
            rmsds       = per_pose_rmsd(raw_poses, ref_pos, dock_idx)
            best_idx    = int(np.argmin(rmsds))
            # Convert pose to absolute coordinates: pred is in pocket-centered frame
            pose_abs = raw_poses[best_idx].numpy() + pocket_cen
            # Some atoms in pose may be ordered differently than the ligand SDF.
            # Use dock_idx to map ligand SDF atoms <-> pose atom indices.
            # ref_pos[i] aligns with pose[dock_idx[i]] (matched subset).
            # For visualization use the matched subset on both sides.
            didx = np.asarray(dock_idx, dtype=np.int64)
            if lig_crystal is None or lig_crystal.shape[0] != pose_abs.shape[0]:
                # fallback: use ref_pos and pose[dock_idx] subset; bonds based on index distance
                cry = ref_pos.numpy()
                pred = pose_abs[didx]
                # build bonds as the SDF bonds reindexed via inverse mapping if possible
                inv = {int(v): i for i, v in enumerate(didx)}
                if lig_crystal is not None and bonds is not None:
                    # bonds are over SDF atom indices; we don't have a SDF<->dock mapping here
                    new_bonds = []
                    # Heuristic: if dock_idx is identity over a permutation, retry
                else:
                    new_bonds = []
                # default to nearest-neighbor "skeleton" bonds for visualization
                # connect each atom to its 2 nearest neighbors
                from scipy.spatial.distance import cdist
                D = cdist(cry, cry)
                np.fill_diagonal(D, np.inf)
                nn = np.argsort(D, axis=1)[:, :3]
                seen = set()
                for i in range(len(cry)):
                    for j in nn[i]:
                        if cdist(cry[i:i+1], cry[j:j+1])[0,0] < 1.9:
                            a, b = (int(i), int(j)) if i < j else (int(j), int(i))
                            seen.add((a, b))
                bonds_use = list(seen)
                lig_c_use = cry
                lig_p_use = pred
            else:
                # SDF and pose have same atom count; assume same order (preprocess preserves order)
                lig_c_use = lig_crystal
                lig_p_use = pose_abs
                bonds_use = bonds

            oracle_r = oracle_rows[pid]["oracle_rmsd"]
            conf_r   = conf_rows[pid]["rmsd"]
            tag = "GOOD" if success else "FAIL"
            title = f"[{tag}] {pid}\norcl={oracle_r:.2f}Å  conf-pick={conf_r:.2f}Å  N_atom={d['n_atoms']}  N_frag={d['n_frags']}"
            draw_complex(ax, prot_xyz, lig_c_use, lig_p_use, bonds_use, title, success)
        except Exception as e:
            ax.text2D(0.05, 0.5, f"{pid}\nERR: {e}", transform=ax.transAxes, fontsize=8)
            ax.set_title(pid, fontsize=10)

    fig.suptitle("PB best 3 (top, GREEN=oracle pose) vs worst 3 (bottom, RED=oracle pose)\nblue=crystal  •  pred=oracle's best pose  •  gray=pocket atoms", fontsize=12)
    fig.tight_layout()
    out = OUT / "pb_best_worst.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
