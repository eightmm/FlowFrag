#!/usr/bin/env python
"""Fetch canonical SMILES for each PoseBusters complex from RCSB.

PoseBusters directory names encode the ligand's HET code (e.g. `5S8I_2LY`
-> HET = `2LY`), so we can query /chemcomp/{het} directly.

Usage:
    python scripts/fetch_pb_smiles.py \
        --pb_dir /mnt/data/PLI/PoseBusters/posebusters_benchmark_set \
        --out data/pb_smiles.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


RCSB_CHEMCOMP = "https://data.rcsb.org/rest/v1/core/chemcomp/{cid}"


def fetch_json(url: str, retries: int = 3, timeout: float = 10.0) -> dict | None:
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:
            time.sleep(0.5)
    return None


def extract_het_code(dir_name: str) -> str | None:
    # Format: <PDB>_<HET> e.g. "5S8I_2LY"
    if "_" not in dir_name:
        return None
    return dir_name.split("_", 1)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    pb_dir = Path(args.pb_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())

    ids = sorted(d.name for d in pb_dir.iterdir() if d.is_dir())
    print(f"Fetching SMILES for {len(ids)} PoseBusters complexes", file=sys.stderr)

    results = dict(existing)
    cache: dict[str, dict] = {}  # HET -> fetched record (within-run)
    for i, name in enumerate(ids):
        if name in results and results[name].get("smiles"):
            continue
        het = extract_het_code(name)
        if het is None:
            results[name] = {"error": "no HET code"}
            continue

        if het in cache:
            cc = cache[het]
        else:
            cc = fetch_json(RCSB_CHEMCOMP.format(cid=het))
            cache[het] = cc or {}

        if not cc:
            print(f"[{i+1:3d}/{len(ids)}] {name}: chemcomp not found ({het})", file=sys.stderr)
            results[name] = {"error": "chemcomp not found", "het": het}
            continue

        desc = cc.get("rcsb_chem_comp_descriptor", {}) or {}
        smiles = desc.get("SMILES_stereo") or desc.get("SMILES")
        if not smiles:
            print(f"[{i+1:3d}/{len(ids)}] {name}: no SMILES in chemcomp ({het})", file=sys.stderr)
            results[name] = {"error": "no SMILES", "het": het}
            continue

        print(f"[{i+1:3d}/{len(ids)}] {name}: {het}  {smiles}", file=sys.stderr)
        results[name] = {"comp_id": het, "smiles": smiles}

        if (i + 1) % 20 == 0:
            out_path.write_text(json.dumps(results, indent=2))

    out_path.write_text(json.dumps(results, indent=2))
    ok = sum(1 for v in results.values() if v.get("smiles"))
    print(f"\nDone: {ok}/{len(ids)} SMILES fetched -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
