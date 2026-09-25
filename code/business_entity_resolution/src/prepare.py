"""Step 1: load a split, learn/apply transliteration, normalize, cache parquet.

Usage:
    python prepare.py --split train      # learns translit dict from train GT
    python prepare.py --split test       # applies the learned dict

Writes work/<split>_s1.parquet and work/<split>_pool.parquet with columns
  entity_id, business_name, business_address, country, source (2/3 for pool),
  name_full, name_core, name_alias, is_domain, addr_norm
"""
import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

import config as C
from io_utils import read_ground_truth, read_split
from normalize import normalize_address, normalize_name, set_translit_dict
from translit import learn_translit, load_translit, save_translit

TRANSLIT_PATH = C.WORK_DIR / "translit_dict.json"


def _norm_names(chunk):
    return [normalize_name(x) for x in chunk]


def _norm_addrs(chunk):
    return [normalize_address(a, c) for a, c in chunk]


def _parallel(fn, items, n):
    if len(items) < 20000 or n <= 1:
        return fn(items)
    chunks = [items[i::n] for i in range(n)]
    ctx = mp.get_context("fork")
    with ctx.Pool(n) as pool:
        parts = pool.map(fn, chunks)
    out = [None] * len(items)
    for i, part in enumerate(parts):
        out[i::n] = part
    return out


def add_normalized(df: pd.DataFrame, log=print) -> pd.DataFrame:
    t0 = time.time()
    uniq = pd.unique(df.business_name)
    res = _parallel(_norm_names, list(uniq), C.N_THREADS)
    m = dict(zip(uniq, res))
    vals = [m[x] for x in df.business_name]
    df["name_full"] = [v[0] for v in vals]
    df["name_core"] = [v[1] for v in vals]
    df["name_alias"] = [v[2] for v in vals]
    df["is_domain"] = np.array([v[3] for v in vals], dtype=np.int8)
    log(f"  names normalized ({len(uniq):,} unique) in {time.time()-t0:.0f}s")
    t0 = time.time()
    pairs = list(zip(df.business_address, df.country))
    upairs = list(set(pairs))
    res = _parallel(_norm_addrs, upairs, C.N_THREADS)
    m = dict(zip(upairs, res))
    df["addr_norm"] = [m[p] for p in pairs]
    log(f"  addresses normalized ({len(upairs):,} unique) in {time.time()-t0:.0f}s")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    args = ap.parse_args()
    C.WORK_DIR.mkdir(parents=True, exist_ok=True)
    split_dir = C.TRAIN_DIR if args.split == "train" else C.TEST_DIR
    t0 = time.time()
    s1, pool = read_split(split_dir, args.split)
    pool["source"] = np.where(pool.entity_id.str.startswith("S2-"), 2, 3).astype(np.int8)
    print(f"loaded {args.split}: s1={len(s1):,} pool={len(pool):,} in {time.time()-t0:.0f}s")

    if args.split == "train":
        gt = read_ground_truth(split_dir / "train_ground_truth.tsv")
        t0 = time.time()
        d = learn_translit(s1, pool, gt)
        save_translit(d, TRANSLIT_PATH)
        print(f"learned transliteration dictionary: {len(d):,} tokens in {time.time()-t0:.0f}s")
    else:
        d = load_translit(TRANSLIT_PATH)
        print(f"loaded transliteration dictionary: {len(d):,} tokens")
    set_translit_dict(d)

    for name, df in (("s1", s1), ("pool", pool)):
        print(f"normalizing {name} ...")
        df = add_normalized(df)
        out = C.WORK_DIR / f"{args.split}_{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"  wrote {out} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
