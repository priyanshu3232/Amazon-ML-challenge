"""Step 3: run blocking + matcher on the TEST split and write the two
submission files:

    output/candidate_pairs.tsv     exact candidate set scored by the model
    output/matching_results.tsv    final matches after the decision layer

Usage:
    python predict.py                      # full test set
    python predict.py --limit 50000        # smoke test on the first N S1 rows
                                           # (writes to work/smoke/ instead)
"""
import argparse
import json
import subprocess
import sys
import time

import sparse_dot_topn  # noqa: F401  (must load before lightgbm: two OpenMP runtimes on macOS segfault otherwise)
import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C
import features as FT
from blocking import block
from evaluate import decide
from features import STAGE2_KEEP, add_relational, pair_features, stage2_features
from io_utils import write_id_list_file

T0 = time.time()
KEEP_COLS = ["entity_id", "country", "name_full", "name_core", "name_alias", "is_domain", "addr_norm"]


def log(msg):
    print(f"[{time.time()-T0:7.0f}s] {msg}", flush=True)


def chunk_bounds(q_sorted, target):
    """Yield (start, end) slices of ~target rows that never split a q group."""
    n = len(q_sorted)
    start = 0
    while start < n:
        end = min(n, start + target)
        if end < n:
            while end < n and q_sorted[end] == q_sorted[end - 1]:
                end += 1
        yield start, end
        start = end


def to_lists(s1_ids, pool_ids, q_idx, p_idx):
    out = {}
    order = np.argsort(q_idx, kind="stable")
    for qi, pi in zip(q_idx[order], p_idx[order]):
        out.setdefault(s1_ids[qi], []).append(pool_ids[pi])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--chunk", type=int, default=4_000_000)
    ap.add_argument("--unseen-thr-bump", type=float, default=None,
                    help="threshold increase for countries absent from training")
    args = ap.parse_args()

    with open(C.WORK_DIR / f"model_meta{args.tag}.json") as fh:
        meta = json.load(fh)
    m1 = lgb.Booster(model_file=str(C.WORK_DIR / f"model_stage1{args.tag}.txt"))
    m2 = lgb.Booster(model_file=str(C.WORK_DIR / f"model_stage2{args.tag}.txt"))
    dec = meta["decision"]
    FT.ALONE_FULL = bool(meta.get("alone_full", False))
    bump = args.unseen_thr_bump if args.unseen_thr_bump is not None else meta.get("unseen_thr_bump", 0.0)
    log(f"decision={dec} use_stage2={meta['use_stage2']} unseen_bump={bump}")

    s1 = pd.read_parquet(C.WORK_DIR / "test_s1.parquet", columns=KEEP_COLS)
    pool = pd.read_parquet(C.WORK_DIR / "test_pool.parquet", columns=KEEP_COLS + ["source"])
    out_dir = C.OUTPUT_DIR
    q_mask = np.ones(len(s1), bool)
    if args.limit:
        q_mask[args.limit:] = False
        out_dir = C.WORK_DIR / "smoke"
    s1_ids = s1.entity_id.values
    pool_ids = pool.entity_id.values
    log(f"loaded test: s1={len(s1):,} pool={len(pool):,}")

    cands = block(s1, pool, q_mask=q_mask, log=log, rev_cache=C.WORK_DIR / "rev_cache")
    cands = cands.sort_values(["q_idx", "p_idx"], kind="stable").reset_index(drop=True)
    add_relational(cands, cands.q_idx.values, cands.p_idx.values, "sim_full", dims=("q", "p"))
    log(f"candidates: {len(cands):,} pairs ({len(cands)/max(1, q_mask.sum()):.1f}/query)")
    write_id_list_file(out_dir / "candidate_pairs.tsv", s1_ids[q_mask],
                       to_lists(s1_ids, pool_ids, cands.q_idx.values, cands.p_idx.values),
                       "candidate_entity_ids")
    log("wrote candidate_pairs.tsv")

    # ---- stage 1 in query-aligned chunks ----
    feat_cols = meta["feat_cols"]
    q_all = cands.q_idx.values
    parts = []
    for a, b in chunk_bounds(q_all, args.chunk):
        sub = cands.iloc[a:b]
        F = pair_features(sub, s1, pool)
        p1 = m1.predict(F[feat_cols]).astype(np.float32)
        tab = sub[["q_idx", "p_idx"]].copy()
        tab[STAGE2_KEEP] = F[STAGE2_KEEP].values
        tab["p1"] = p1
        parts.append(tab)
        log(f"  scored rows {a:,}-{b:,}")
    tab = pd.concat(parts, ignore_index=True)
    del parts

    # ---- stage 2 ----
    if meta["use_stage2"]:
        G = stage2_features(tab)
        prob = m2.predict(G[meta["stage2_cols"]]).astype(np.float32)
        del G
    else:
        prob = tab.p1.values
    log("stage-2 probabilities ready")

    # ---- decision layer ----
    q_idx, p_idx = tab.q_idx.values, tab.p_idx.values
    thr = np.full(len(tab), dec["thr"], np.float32)
    if bump:
        train_countries = set(meta.get("train_countries", []))
        unseen = ~np.isin(s1.country.values, list(train_countries))
        thr[unseen[q_idx]] += bump
        log(f"  unseen-country rows: {int(unseen[q_idx].sum()):,} get thr+{bump}")
    keep = decide(q_idx, p_idx, prob, 0.0, one_parent=dec["one_parent"], rel_gap=dec["rel_gap"],
                  min_margin=dec.get("min_margin"))
    keep &= prob >= thr
    matches = to_lists(s1_ids, pool_ids, q_idx[keep], p_idx[keep])
    write_id_list_file(out_dir / "matching_results.tsv", s1_ids[q_mask], matches, "matched_entity_ids")
    n_nonempty = sum(1 for v in matches.values() if v)
    log(f"wrote matching_results.tsv: {int(q_mask.sum()):,} rows, {n_nonempty:,} non-empty, "
        f"{int(keep.sum()):,} matched pairs ({keep.sum()/max(1, q_mask.sum()):.2f}/entity)")
    pd.DataFrame({"s1": s1_ids[q_idx], "cand": pool_ids[p_idx], "prob": prob, "keep": keep}) \
        .to_parquet(out_dir / "test_pairs_scored.parquet", index=False)

    if not args.limit:
        cmd = [sys.executable, str(C.ROOT / "student_resource" / "utils" / "validate_submission.py"),
               "--matching", str(out_dir / "matching_results.tsv"),
               "--candidate", str(out_dir / "candidate_pairs.tsv"),
               "--test-dir", str(C.TEST_DIR)]
        log("running validator ...")
        print(subprocess.run(cmd, capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main()
