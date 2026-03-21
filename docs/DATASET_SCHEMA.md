# Dataset Schema

This document is the practical spec for the rebuilt preprocessing output.

## Source

- raw source root: `/mnt/data/PLI/P-L`
- build entrypoint: `scripts/build_fragment_flow_dataset.py`

## Build Command

```bash
~/.local/bin/uv run python scripts/build_fragment_flow_dataset.py \
  --raw_dir /mnt/data/PLI/P-L \
  --out_dir data/processed_v2 \
  --workers 8 \
  --dummy
```

Use a fresh output directory when the schema changes.

## Per-Complex Files

| File | Required | Purpose |
|---|---|---|
| `ligand.pt` | yes | ligand atom, bond, fragment, dummy tensors |
| `protein_atoms.pt` | yes | protein atom graph and CA virtual metadata |
| `graph.pt` | yes | unified static graph for all non-contact edges |
| `meta.pt` | yes | counts, pocket center, schema version |
| `protein.pt` | legacy | residue-level file from the old pipeline |

## `meta.pt`

| Key | Meaning |
|---|---|
| `pdb_id` | complex id |
| `pocket_center` | reference center used for centering |
| `num_res` | number of CA virtual nodes kept in the pocket |
| `num_atom` | number of ligand atoms after optional dummy expansion |
| `num_frag` | number of rigid fragments |
| `num_prot_atom` | number of protein heavy atoms in the kept pocket |
| `used_mol2_fallback` | whether ligand load fell back to MOL2 |
| `has_dummy_atoms` | whether dummy atoms were added |
| `schema_version` | preprocessing schema version |

## Invariants

- `graph.pt` contains only static topology
- protein-ligand contact edges are not stored there
- ligand and protein atom chemistry fields share the same vocab/layout where
  possible
- dummy atoms copy chemistry from their real source atoms
- CA virtual nodes are per-residue and sit at the selected residue `CA`

## Migration Warning

At the moment, the preprocessing schema and the runtime loader are not fully in
sync.

- new preprocessing writes `graph.pt`
- current `src/data/dataset.py` still consumes legacy `protein.pt`

That means a rebuild alone does not switch experiments to the new graph path.
The loader must be updated as a separate step.

## Recommended Operational Policy

- never partially rebuild `data/processed/` after a schema change
- build into `data/processed_v2/` first
- validate a few samples
- then point training configs to the new root
