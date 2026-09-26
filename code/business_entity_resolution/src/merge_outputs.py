"""Merge per-country partial outputs written by `predict.py --countries ... --out-suffix ...`
(possibly on different machines) into the final submission files, in the original
Source-1 order, then run the organisers' validator.

    python merge_outputs.py --suffixes _india _us_fr [--dir ../../../output]
"""
import argparse
import subprocess
import sys

import pandas as pd

import config as C


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffixes", nargs="+", required=True)
    ap.add_argument("--dir", default=str(C.OUTPUT_DIR))
    ap.add_argument("--no-check", action="store_true", help="skip the completeness check and validator")
    args = ap.parse_args()
    out = C.ROOT / args.dir if not str(args.dir).startswith("/") else pd.io.common.Path(args.dir)

    order = pd.read_parquet(C.WORK_DIR / "test_s1.parquet", columns=["entity_id"]).entity_id
    for stem in ("matching_results", "candidate_pairs"):
        parts = [read_tsv(out / f"{stem}{s}.tsv") for s in args.suffixes]
        df = pd.concat(parts, ignore_index=True)
        dup = df.source1_entity_id.duplicated().sum()
        missing = int((~order.isin(df.source1_entity_id)).sum())
        print(f"{stem}: {len(df):,} rows from {len(parts)} files; duplicates={dup:,} missing={missing:,}")
        if not args.no_check and (dup or missing):
            sys.exit(f"{stem}: partial files do not cover every Source-1 entity exactly once")
        df = df.set_index("source1_entity_id")
        df = df.reindex(order[order.isin(df.index)]).reset_index()
        df.to_csv(out / f"{stem}.tsv", sep="\t", index=False, quoting=3, lineterminator="\n")
        print(f"  wrote {out / (stem + '.tsv')}")

    if not args.no_check:
        cmd = [sys.executable, str(C.ROOT / "student_resource" / "utils" / "validate_submission.py"),
               "--matching", str(out / "matching_results.tsv"),
               "--candidate", str(out / "candidate_pairs.tsv"),
               "--test-dir", str(C.TEST_DIR)]
        print(subprocess.run(cmd, capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main()
