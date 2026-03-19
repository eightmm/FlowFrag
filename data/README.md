# data

This directory is reserved for the rebuilt dataset pipeline.

Planned layout:

- `raw_index/`
- `processed/`
- `splits/`
- `cache/`

Rules:

- do not mix legacy tensors from `legacy/data/` into this directory
- processed samples here should come only from `/mnt/data/PLI/P-L`
- keep the schema documented in `FRAGMENT_FLOW_REBUILD_PLAN.md`
