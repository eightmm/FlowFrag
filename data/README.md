# Data Directory

This directory stores processed protein-ligand complexes built from
`/mnt/data/PLI/P-L`.

## Current Rule

- Treat `data/processed/` as a generated artifact.
- Do not hand-edit `.pt` files.
- Rebuild the whole directory when the preprocessing schema changes.
- Prefer building into a fresh output directory such as `data/processed_v2/`
  instead of mixing old and new schemas in the same folder.

## Recommended Layout

- `processed/`
  - active processed dataset used by experiments
- `processed_v2/`
  - fresh full rebuild for the new atom-level unified graph schema
- `splits/`
  - train/val/test split files
- `cache/`
  - optional runtime caches

## Full Rebuild

Use `uv` and build from the raw source:

```bash
~/.local/bin/uv run python scripts/build_fragment_flow_dataset.py \
  --raw_dir /mnt/data/PLI/P-L \
  --out_dir data/processed_v2 \
  --workers 8 \
  --dummy
```

Important:

- `--dummy` is now part of the intended schema.
- The new build writes `graph.pt` and `protein_atoms.pt`.
- The old training dataset in `src/data/dataset.py` still reads `protein.pt`
  unless that loader is updated.

## Per-Complex Files

New schema targets the following files per complex:

- `ligand.pt`
  - ligand atom features, bond features, fragment decomposition, dummy metadata
- `protein_atoms.pt`
  - protein atom features, protein covalent bonds, residue/CA metadata
- `graph.pt`
  - unified static graph with all non-protein-ligand edges precomputed
- `meta.pt`
  - counts, pocket center, schema version, misc metadata

Legacy file:

- `protein.pt`
  - residue-level pocket tensor from the old pipeline
  - should be considered deprecated for the new atom-level path

## Docs

- `docs/DATA_GRAPH_TOPOLOGY.md`
- `docs/DATASET_SCHEMA.md`
