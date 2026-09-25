"""Loading and writing the challenge TSV files.

All files are tab separated with NO quoting; business names may contain
double quotes, so csv.QUOTE_NONE is mandatory. dtype=str and
keep_default_na=False stop pandas turning a business called "NA" or "None"
into a missing value.
"""
import csv
from pathlib import Path

import pandas as pd

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: Path) -> pd.DataFrame:
    """Read one *_sourceN.tsv file exactly as provided."""
    df = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False,
        quoting=csv.QUOTE_NONE, encoding="utf-8",
    )
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[COLS]


def read_split(split_dir: Path, prefix: str):
    """Return (s1, pool) where pool = S2 rows followed by S3 rows."""
    s1 = read_source(split_dir / f"{prefix}_source1.tsv")
    s2 = read_source(split_dir / f"{prefix}_source2.tsv")
    s3 = read_source(split_dir / f"{prefix}_source3.tsv")
    pool = pd.concat([s2, s3], ignore_index=True)
    return s1, pool


def read_ground_truth(path: Path) -> dict:
    """Return {source1_entity_id: set(matched ids)}; empty set for singletons."""
    gt = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            rest = rest.strip()
            gt[s1] = set(x for x in rest.split(",") if x) if rest else set()
    return gt


def write_id_list_file(path: Path, s1_ids, mapping: dict, col: str) -> None:
    """Write a two-column TSV (source1_entity_id, <col>) with one row per S1 id.

    mapping: {s1_id: iterable of S2/S3 ids}. Order inside a list is preserved
    after de-duplication so the file is deterministic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s1 in s1_ids:
            ids = list(dict.fromkeys(mapping.get(s1, ())))
            f.write(f"{s1}\t{','.join(ids)}\n")
