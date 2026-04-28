#!/usr/bin/env python
"""Bulk-download PLINDER system zips for our filtered training set.

Uses plinder.core.utils.unpack.get_zips_to_unpack to download + extract
``systems/{two_char_code}.zip`` files in parallel. Skips ``linked_structures``
(we do not need them — they triple disk usage).
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from plinder.core.utils.unpack import get_zips_to_unpack

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered_parquet", type=Path,
                    default=Path("data/plinder_train_filtered.parquet"))
    ap.add_argument("--also_val", action="store_true",
                    help="Also include codes from plinder_val_filtered.parquet")
    ap.add_argument("--limit_codes", type=int, default=None,
                    help="Process only the first N 2-char codes (testing).")
    args = ap.parse_args()

    df = pd.read_parquet(args.filtered_parquet)
    df["two_char_code"] = df["entry_pdb_id"].str[1:3]
    codes = sorted(df["two_char_code"].dropna().unique().tolist())

    if args.also_val:
        val_path = args.filtered_parquet.parent / "plinder_val_filtered.parquet"
        if val_path.exists():
            vdf = pd.read_parquet(val_path)
            vcodes = vdf["entry_pdb_id"].str[1:3].dropna().unique().tolist()
            codes = sorted(set(codes) | set(vcodes))

    if args.limit_codes:
        codes = codes[: args.limit_codes]
    log.info("Will download %d 2-char codes (e.g., %s)", len(codes), codes[:5])

    t0 = time.time()
    # get_zips_to_unpack downloads zips + extracts to systems/{system_id}/...
    # Internally uses thread_map for parallelism.
    zips = get_zips_to_unpack(kind="systems", two_char_codes=codes)
    log.info("Got %d zip paths", len(zips))
    log.info("Total wall time: %.1fs", time.time() - t0)

    # Save manifest of downloaded codes
    Path("data").mkdir(exist_ok=True)
    json.dump({"codes": codes, "n_zips": len(zips), "wall_seconds": time.time() - t0},
              open("data/plinder_download_manifest.json", "w"), indent=2)
    log.info("Manifest -> data/plinder_download_manifest.json")


if __name__ == "__main__":
    main()
