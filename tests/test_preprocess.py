"""Tests for preprocessing pipeline: protein, ligand, fragments."""

from pathlib import Path

import pytest
import torch
from rdkit import Chem

from src.preprocess.fragments import decompose_fragments
from src.preprocess.ligand import featurize_ligand, load_molecule
from src.preprocess.protein import AA3_TO_IDX, parse_pocket_pdb


# ─── Fixtures ─────────────────────────────────────────────────────────

SAMPLE_PDB = """\
ATOM      1  N   ALA A  10      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A  10      11.000  10.000  10.000  1.00 20.00           C
ATOM      3  C   ALA A  10      12.000  10.000  10.000  1.00 20.00           C
ATOM      4  N   GLY A  15      15.000  12.000  10.000  1.00 20.00           N
ATOM      5  CA  GLY A  15      16.000  12.000  10.000  1.00 20.00           C
ATOM      6  C   GLY A  15      17.000  12.000  10.000  1.00 20.00           C
ATOM      7  N   MET A  20      20.000  15.000  10.000  1.00 20.00           N
ATOM      8  CA  MET A  20      21.000  15.000  10.000  1.00 20.00           C
HETATM    9  CA  MSE A  25      25.000  18.000  10.000  1.00 20.00          SE
HETATM   10  O   HOH A 100      30.000  30.000  30.000  1.00 20.00           O
END
"""


def _make_mol_manual_coords(smiles: str, coords_3d: list[list[float]]) -> Chem.Mol:
    """Create mol with manually specified 3D coordinates (no embedding needed)."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords_3d):
        conf.SetAtomPosition(i, (x, y, z))
    conf.Set3D(True)
    mol.AddConformer(conf, assignId=True)
    return mol


def _write_sdf(mol: Chem.Mol, path: Path) -> None:
    """Write mol to SDF file."""
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


# ─── Manual coordinate sets ──────────────────────────────────────────

# Benzene: 6 carbons in hexagonal ring
BENZENE_COORDS = [
    [1.4, 0.0, 0.0], [0.7, 1.2, 0.0], [-0.7, 1.2, 0.0],
    [-1.4, 0.0, 0.0], [-0.7, -1.2, 0.0], [0.7, -1.2, 0.0],
]

# Two phenyl rings connected by -CH2-CH2- linker (14 heavy atoms)
# Ring1(6) + CH2(1) + CH2(1) + Ring2(6)
DIPHENYLETHANE_COORDS = [
    # Ring 1
    [0.0, 1.4, 0.0], [1.2, 0.7, 0.0], [1.2, -0.7, 0.0],
    [0.0, -1.4, 0.0], [-1.2, -0.7, 0.0], [-1.2, 0.7, 0.0],
    # CH2 linker
    [0.0, 2.9, 0.0], [0.0, 4.4, 0.0],
    # Ring 2
    [0.0, 5.9, 0.0], [1.2, 6.6, 0.0], [1.2, 8.0, 0.0],
    [0.0, 8.7, 0.0], [-1.2, 8.0, 0.0], [-1.2, 6.6, 0.0],
]


# ─── Protein Tests ────────────────────────────────────────────────────


class TestProteinParsing:
    def test_parse_pocket_pdb(self, tmp_path: Path):
        pdb_file = tmp_path / "pocket.pdb"
        pdb_file.write_text(SAMPLE_PDB)

        result = parse_pocket_pdb(pdb_file)
        assert result is not None

        # Should find 4 CAs: ALA, GLY, MET, MSE(→MET)
        assert result["res_coords"].shape == (4, 3)
        assert result["res_type"].shape == (4,)

        # Check types
        assert result["res_type"][0].item() == AA3_TO_IDX["ALA"]
        assert result["res_type"][1].item() == AA3_TO_IDX["GLY"]
        assert result["res_type"][2].item() == AA3_TO_IDX["MET"]
        assert result["res_type"][3].item() == AA3_TO_IDX["MET"]  # MSE → MET

        # Check coords
        assert result["res_coords"][0, 0].item() == pytest.approx(11.0)  # ALA CA x

    def test_empty_pdb(self, tmp_path: Path):
        pdb_file = tmp_path / "empty.pdb"
        pdb_file.write_text("END\n")
        assert parse_pocket_pdb(pdb_file) is None

    def test_water_only_pdb(self, tmp_path: Path):
        pdb_file = tmp_path / "water.pdb"
        pdb_file.write_text(
            "HETATM    1  O   HOH A   1      10.0  10.0  10.0  1.00 20.00\nEND\n"
        )
        assert parse_pocket_pdb(pdb_file) is None

    def test_duplicate_residue_altloc(self, tmp_path: Path):
        """Only first CA per residue should be kept."""
        # PDB format: columns are fixed width
        # 1-6: record, 7-11: serial, 12: blank, 13-16: name, 17: altLoc, 18-20: resName
        pdb_text = (
            "ATOM      1  CA AALA A  10      10.000  10.000  10.000  1.00 20.00           C\n"
            "ATOM      2  CA BALA A  10      11.000  11.000  11.000  1.00 20.00           C\n"
            "END\n"
        )
        pdb_file = tmp_path / "altloc.pdb"
        pdb_file.write_text(pdb_text)

        result = parse_pocket_pdb(pdb_file)
        assert result is not None
        assert result["res_coords"].shape == (1, 3)
        assert result["res_coords"][0, 0].item() == pytest.approx(10.0)


# ─── Ligand Tests ─────────────────────────────────────────────────────


class TestLigandParsing:
    def test_load_molecule_sdf(self, tmp_path: Path):
        mol = _make_mol_manual_coords("c1ccccc1", BENZENE_COORDS)
        sdf_path = tmp_path / "test.sdf"
        _write_sdf(mol, sdf_path)

        loaded, used_fallback, sanitize_ok = load_molecule(sdf_path)
        assert loaded is not None
        assert not used_fallback
        assert sanitize_ok
        assert loaded.GetNumAtoms() == 6

    def test_featurize_benzene(self):
        mol = _make_mol_manual_coords("c1ccccc1", BENZENE_COORDS)
        result = featurize_ligand(mol)
        assert result is not None

        # 6 atoms
        assert result["atom_coords"].shape == (6, 3)
        assert result["atom_element"].shape == (6,)
        assert (result["atom_element"] == 0).all()  # all carbon
        assert (result["atom_aromatic"]).all()  # all aromatic
        assert (result["atom_in_ring"]).all()  # all in ring

        # 6 bonds → 12 directed edges
        assert result["bond_index"].shape[1] == 12
        assert result["bond_type"].shape == (12,)

    def test_featurize_charged_molecule(self):
        # Glycine zwitterion: [NH3+]CC([O-])=O → 5 heavy atoms
        coords = [[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [3.7, 1.2, 0], [3.7, -1.2, 0]]
        mol = _make_mol_manual_coords("[NH3+]CC([O-])=O", coords)
        result = featurize_ligand(mol)
        assert result is not None
        assert result["atom_charge"].abs().sum() > 0

    def test_single_atom_returns_none(self):
        mol = Chem.MolFromSmiles("[Na+]")
        assert featurize_ligand(mol) is None


# ─── Fragment Tests ───────────────────────────────────────────────────


class TestFragmentDecomposition:
    def test_benzene_single_fragment(self):
        """Benzene has no rotatable bonds → 1 fragment."""
        mol = _make_mol_manual_coords("c1ccccc1", BENZENE_COORDS)
        coords = torch.tensor(BENZENE_COORDS, dtype=torch.float32)

        result = decompose_fragments(mol, coords)
        assert result is not None
        assert result["n_frags"] == 1
        assert (result["fragment_id"] == 0).all()

    def test_diphenylethane_fragments(self):
        """c1ccc(CCc2ccccc2)cc1: two rings + CH2CH2 linker."""
        mol = _make_mol_manual_coords("c1ccc(CCc2ccccc2)cc1", DIPHENYLETHANE_COORDS)
        coords = torch.tensor(DIPHENYLETHANE_COORDS, dtype=torch.float32)

        result = decompose_fragments(mol, coords)
        assert result is not None
        assert result["n_frags"] >= 1
        assert result["fragment_id"].shape[0] == 14
        assert result["frag_centers"].shape == (result["n_frags"], 3)
        assert result["frag_local_coords"].shape == (14, 3)

    def test_local_coords_centroid_property(self):
        """Local coords should sum to ~0 per fragment."""
        mol = _make_mol_manual_coords("c1ccc(CCc2ccccc2)cc1", DIPHENYLETHANE_COORDS)
        coords = torch.tensor(DIPHENYLETHANE_COORDS, dtype=torch.float32)

        result = decompose_fragments(mol, coords)
        assert result is not None

        for f in range(result["n_frags"]):
            mask = result["fragment_id"] == f
            local = result["frag_local_coords"][mask]
            assert local.mean(dim=0).abs().max() < 1e-5

    def test_reconstruction_from_local_coords(self):
        """x_global = R_frag @ x_local + T_frag, with R=I at crystal pose."""
        mol = _make_mol_manual_coords("c1ccc(CCc2ccccc2)cc1", DIPHENYLETHANE_COORDS)
        coords = torch.tensor(DIPHENYLETHANE_COORDS, dtype=torch.float32)

        result = decompose_fragments(mol, coords)
        assert result is not None

        reconstructed = result["frag_local_coords"] + result["frag_centers"][result["fragment_id"]]
        assert torch.allclose(reconstructed, coords, atol=1e-5)

    def test_fragment_id_valid_range(self):
        """All fragment IDs should be in [0, n_frags)."""
        mol = _make_mol_manual_coords("c1ccc(CCc2ccccc2)cc1", DIPHENYLETHANE_COORDS)
        coords = torch.tensor(DIPHENYLETHANE_COORDS, dtype=torch.float32)

        result = decompose_fragments(mol, coords)
        assert result is not None
        assert result["fragment_id"].min() >= 0
        assert result["fragment_id"].max() < result["n_frags"]

    def test_frag_sizes_match(self):
        """Sum of frag_sizes should equal total atoms."""
        mol = _make_mol_manual_coords("c1ccc(CCc2ccccc2)cc1", DIPHENYLETHANE_COORDS)
        coords = torch.tensor(DIPHENYLETHANE_COORDS, dtype=torch.float32)

        result = decompose_fragments(mol, coords)
        assert result is not None
        assert result["frag_sizes"].sum().item() == 14

    def test_triangulation_edge_filtering(self):
        """Verify that neighbor-neighbor pairs (torsion-dependent) are excluded."""
        # Ethane-like: C1-C2-C3-C4 (actually 4 atoms)
        # 0-1 (cut) 1-2, 2-3... wait.
        # Let's use a simple 4-atom chain: A-B-C-D where B-C is rotatable.
        # Frag1: {A, B}, Frag2: {C, D}
        # Cut bond: (1, 2)
        # Neighbors of 1 in Frag1: {0}
        # Neighbors of 2 in Frag2: {3}
        # Triangulation edges should be: (1, 2), (1, 3), (0, 2)
        # (0, 3) should be EXCLUDED as it depends on torsion 0-1-2-3.
        coords = [[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 0, 0]]
        mol = _make_mol_manual_coords("CCCC", coords)
        # Ensure B-C is rotatable and cut
        # Decompose will cut B-C (1-2) if it's not in ring, etc.
        res = decompose_fragments(mol, torch.tensor(coords, dtype=torch.float32))
        assert res is not None
        assert res["n_frags"] == 2

        # Check tri_edge_index
        tri_edges = res["tri_edge_index"]
        # Convert to set of frozen sets for easy comparison
        edge_sets = {frozenset(tri_edges[:, k].tolist()) for k in range(tri_edges.shape[1])}

        # Expected:
        # (1, 2) - invariant (bond length)
        # (1, 3) - invariant (bond angle)
        # (0, 2) - invariant (bond angle)
        assert frozenset([1, 2]) in edge_sets
        assert frozenset([1, 3]) in edge_sets
        assert frozenset([0, 2]) in edge_sets

        # EXCLUDED:
        # (0, 3) - variant (torsion)
        assert frozenset([0, 3]) not in edge_sets


# ─── Integration Test ─────────────────────────────────────────────────

# Pocket PDB placed near diphenylethane ligand (coords ~0-9 Å) so the
# 8 Å residue-level cutoff keeps all residues.
E2E_POCKET_PDB = """\
ATOM      1  N   ALA A   1       3.000   3.000   0.500  1.00 20.00           N
ATOM      2  CA  ALA A   1       4.500   3.000   0.500  1.00 20.00           C
ATOM      3  C   ALA A   1       5.500   3.000   0.500  1.00 20.00           C
ATOM      4  O   ALA A   1       5.500   4.200   0.500  1.00 20.00           O
ATOM      5  CB  ALA A   1       4.500   1.500   0.500  1.00 20.00           C
ATOM      6  N   LEU A   2       6.700   3.000   0.500  1.00 20.00           N
ATOM      7  CA  LEU A   2       7.500   4.200   0.500  1.00 20.00           C
ATOM      8  C   LEU A   2       8.500   4.200   0.500  1.00 20.00           C
ATOM      9  O   LEU A   2       8.500   5.400   0.500  1.00 20.00           O
ATOM     10  CB  LEU A   2       7.500   5.700   0.500  1.00 20.00           C
ATOM     11  CG  LEU A   2       7.500   7.200   0.500  1.00 20.00           C
ATOM     12  CD1 LEU A   2       6.300   7.900   0.500  1.00 20.00           C
ATOM     13  CD2 LEU A   2       8.700   7.900   0.500  1.00 20.00           C
END
"""


class TestEndToEnd:
    def test_full_pipeline_single_complex(self, tmp_path: Path):
        """Test the full preprocessing pipeline on a synthetic complex."""
        pdb_id = "test"
        complex_dir = tmp_path / pdb_id
        complex_dir.mkdir()

        # Create pocket PDB (near ligand coords)
        (complex_dir / f"{pdb_id}_pocket.pdb").write_text(E2E_POCKET_PDB)

        # Create ligand SDF
        mol = _make_mol_manual_coords("c1ccc(CCc2ccccc2)cc1", DIPHENYLETHANE_COORDS)
        _write_sdf(mol, complex_dir / f"{pdb_id}_ligand.sdf")

        # Run pipeline
        from scripts.build_fragment_flow_dataset import process_complex

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        result = process_complex(complex_dir, out_dir)

        assert result["success"], f"Failed: {result.get('reason')}"
        assert result["num_res"] == 2
        assert result["num_atom"] == 14
        assert result["num_frag"] >= 1

        # Verify saved files (protein + ligand + meta, no graph.pt)
        assert (out_dir / pdb_id / "protein.pt").exists()
        assert (out_dir / pdb_id / "ligand.pt").exists()
        assert (out_dir / pdb_id / "meta.pt").exists()
        assert not (out_dir / pdb_id / "graph.pt").exists()

        # Load and verify shapes
        prot = torch.load(out_dir / pdb_id / "protein.pt", weights_only=True)
        lig = torch.load(out_dir / pdb_id / "ligand.pt", weights_only=False)
        meta = torch.load(out_dir / pdb_id / "meta.pt", weights_only=False)

        assert prot["pres_coords"].ndim == 2 and prot["pres_coords"].shape[1] == 3
        assert prot["patom_coords"].ndim == 2 and prot["patom_coords"].shape[1] == 3
        assert lig["atom_coords"].shape == (14, 3)
        assert lig["fragment_id"].shape[0] == 14
        assert lig["frag_centers"].shape[1] == 3
        assert meta["pdb_id"] == pdb_id
        assert meta["schema_version"] == 1
