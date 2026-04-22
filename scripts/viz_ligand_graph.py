"""Render the full static complex graph (graph.py output) in 2D.

All node types: ligand_atom, ligand_fragment, protein_atom, protein_res
All edge types: ligand_bond, ligand_tri, ligand_cut, ligand_atom_frag,
                ligand_frag_frag, protein_bond, protein_atom_res,
                protein_res_res, protein_res_frag

Layout: 3D coords → PCA 2D projection.

Usage:
    uv run python scripts/viz_ligand_graph.py --out outputs/viz/ligand_graph --n 8
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from src.preprocess.fragments import decompose_fragments
from src.preprocess.graph import EDGE_TYPES, NODE_TYPES, build_static_complex_graph
from src.preprocess.ligand import featurize_ligand, load_molecule
from src.preprocess.protein import parse_pocket_atoms

NODE_COLORS = {
    NODE_TYPES["ligand_atom"]: "#4477AA",
    NODE_TYPES["ligand_fragment"]: "#EE6677",
    NODE_TYPES["protein_atom"]: "#228833",
    NODE_TYPES["protein_res"]: "#CCBB44",
}
NODE_LABELS = {
    NODE_TYPES["ligand_atom"]: "lig atom",
    NODE_TYPES["ligand_fragment"]: "lig frag",
    NODE_TYPES["protein_atom"]: "prot atom",
    NODE_TYPES["protein_res"]: "prot res",
}
NODE_RADIUS = {
    NODE_TYPES["ligand_atom"]: 0.15,
    NODE_TYPES["ligand_fragment"]: 0.3,
    NODE_TYPES["protein_atom"]: 0.1,
    NODE_TYPES["protein_res"]: 0.25,
}

EDGE_STYLE: dict[int, dict] = {
    EDGE_TYPES["ligand_bond"]:     {"color": "black",   "ls": "-",  "lw": 1.5, "alpha": 0.8},
    EDGE_TYPES["ligand_tri"]:      {"color": "#00aacc", "ls": "--", "lw": 1.0, "alpha": 0.6},
    EDGE_TYPES["ligand_cut"]:      {"color": "red",     "ls": "-",  "lw": 2.2, "alpha": 0.8},
    EDGE_TYPES["ligand_atom_frag"]:{"color": "gray",    "ls": ":",  "lw": 0.6, "alpha": 0.3},
    EDGE_TYPES["ligand_frag_frag"]:{"color": "#AA3377", "ls": "--", "lw": 1.8, "alpha": 0.7},
    EDGE_TYPES["protein_bond"]:    {"color": "#228833", "ls": "-",  "lw": 1.0, "alpha": 0.5},
    EDGE_TYPES["protein_atom_res"]:{"color": "#228833", "ls": ":",  "lw": 0.5, "alpha": 0.25},
    EDGE_TYPES["protein_res_res"]: {"color": "#CCBB44", "ls": "--", "lw": 1.0, "alpha": 0.4},
    EDGE_TYPES["protein_res_frag"]:{"color": "#66CCEE", "ls": "--", "lw": 1.2, "alpha": 0.5},
}
EDGE_LABELS = {v: k for k, v in EDGE_TYPES.items()}


def _pca_2d(coords_3d: np.ndarray) -> np.ndarray:
    centered = coords_3d - coords_3d.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:2].T


def pick_complexes(raw_dir: Path, n: int, seed: int = 0) -> list[Path]:
    all_complexes: list[Path] = []
    for year in sorted(raw_dir.iterdir()):
        if not year.is_dir():
            continue
        for c in sorted(year.iterdir()):
            if not c.is_dir():
                continue
            sdf = c / f"{c.name}_ligand.sdf"
            mol2 = c / f"{c.name}_ligand.mol2"
            if (sdf.exists() or mol2.exists()) and (c / f"{c.name}_pocket.pdb").exists():
                all_complexes.append(c)
    rng = random.Random(seed)
    rng.shuffle(all_complexes)
    return all_complexes[:n]


def build_graph_for_complex(cdir: Path) -> dict | None:
    pdb_id = cdir.name
    sdf = cdir / f"{pdb_id}_ligand.sdf"
    mol2 = cdir / f"{pdb_id}_ligand.mol2"
    pocket = cdir / f"{pdb_id}_pocket.pdb"

    mol, _, _ = load_molecule(
        sdf if sdf.exists() else None, mol2 if mol2.exists() else None
    )
    if mol is None:
        return None
    feats = featurize_ligand(mol)
    if feats is None:
        return None
    frag = decompose_fragments(mol, feats["atom_coords"])
    if frag is None:
        return None

    for key in ("fragment_id", "frag_centers", "frag_local_coords", "frag_sizes",
                "tri_edge_index", "tri_edge_ref_dist", "fragment_adj_index", "cut_bond_index"):
        feats[key] = frag[key]

    from src.data.dataset import crop_to_pocket
    patom_data = parse_pocket_atoms(pocket)
    if patom_data is None:
        return None
    cropped = crop_to_pocket(patom_data, feats["atom_coords"], cutoff=8.0)
    if cropped is None:
        return None

    graph = build_static_complex_graph(feats, cropped)
    graph["pdb_id"] = pdb_id
    graph["mol"] = mol
    return graph


def render_one(graph: dict, out_dir: Path, dpi: int = 150) -> str | None:
    pdb_id = graph["pdb_id"]
    coords_3d = graph["node_coords"].numpy()
    pos2d = _pca_2d(coords_3d)

    node_type = graph["node_type"].numpy()
    edge_index = graph["edge_index"].numpy()
    edge_type = graph["edge_type"].numpy()
    n_nodes = pos2d.shape[0]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)

    # --- Draw edges (grouped by type, low zorder first) ---
    edge_seen: dict[int, set] = {et: set() for et in EDGE_STYLE}
    edge_counts: dict[int, int] = {et: 0 for et in EDGE_STYLE}

    draw_order = [
        EDGE_TYPES["protein_atom_res"],
        EDGE_TYPES["ligand_atom_frag"],
        EDGE_TYPES["protein_res_res"],
        EDGE_TYPES["protein_bond"],
        EDGE_TYPES["protein_res_frag"],
        EDGE_TYPES["ligand_frag_frag"],
        EDGE_TYPES["ligand_tri"],
        EDGE_TYPES["ligand_bond"],
        EDGE_TYPES["ligand_cut"],
    ]

    for z, et in enumerate(draw_order):
        style = EDGE_STYLE[et]
        for k in range(edge_index.shape[1]):
            if int(edge_type[k]) != et:
                continue
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            key = (min(i, j), max(i, j))
            if key in edge_seen[et]:
                continue
            edge_seen[et].add(key)
            edge_counts[et] += 1
            ax.plot(
                [pos2d[i, 0], pos2d[j, 0]],
                [pos2d[i, 1], pos2d[j, 1]],
                color=style["color"], linestyle=style["ls"],
                linewidth=style["lw"], alpha=style["alpha"],
                zorder=z + 1,
            )

    # --- Draw nodes ---
    for nt, color in NODE_COLORS.items():
        mask = node_type == nt
        r = NODE_RADIUS[nt]
        idxs = np.where(mask)[0]
        for i in idxs:
            ax.add_patch(Circle(
                (pos2d[i, 0], pos2d[i, 1]),
                radius=r, facecolor=color, edgecolor="white",
                linewidth=0.3, alpha=0.85, zorder=len(draw_order) + 1,
            ))

    # --- Stats ---
    type_counts = {nt: int((node_type == nt).sum()) for nt in NODE_COLORS}
    title_parts = [f"{NODE_LABELS[nt]}={type_counts[nt]}" for nt in NODE_COLORS]
    edge_parts = [f"{EDGE_LABELS[et]}={edge_counts[et]}" for et in draw_order if edge_counts[et] > 0]

    ax.set_title(
        f"{pdb_id}  |  nodes={n_nodes}  ({', '.join(title_parts)})\n"
        f"edges: {', '.join(edge_parts)}",
        fontsize=9,
    )
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Legend ---
    node_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=8, label=NODE_LABELS[nt])
        for nt, c in NODE_COLORS.items()
    ]
    edge_legend = [
        Line2D([0], [0], color=s["color"], ls=s["ls"], lw=s["lw"],
               alpha=s["alpha"], label=EDGE_LABELS[et])
        for et, s in EDGE_STYLE.items() if edge_counts.get(et, 0) > 0
    ]
    ax.legend(
        handles=node_legend + edge_legend,
        loc="lower left", fontsize=7, framealpha=0.9, ncol=2,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pdb_id}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=Path("/mnt/data/PLI/P-L"))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/viz/ligand_graph"))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    complexes = pick_complexes(args.raw_dir, args.n, args.seed)
    print(f"selected {len(complexes)} complexes")
    saved = 0
    for cdir in complexes:
        graph = build_graph_for_complex(cdir)
        if graph is None:
            print(f"[skip] {cdir.name}")
            continue
        p = render_one(graph, args.out, dpi=args.dpi)
        if p is None:
            print(f"[skip] {cdir.name}: render failed")
            continue
        print(f"[ok]   {cdir.name}  →  {p}")
        saved += 1
    print(f"saved {saved}/{len(complexes)} to {args.out}")


if __name__ == "__main__":
    main()
