"""Step 3: run blocking + matcher on the TEST split and write the two
submission files:

    output/candidate_pairs.tsv     exact candidate set scored by the model
    output/matching_results.tsv    final matches after the decision layer

Memory design (fits a 16 GB laptop for the full 1.7M x 10M test set):
  * everything is processed one country at a time (blocking, features,
    stage 2 and the decision layer never need to look across countries);
  * the two TSVs are streamed from sorted integer arrays, never from Python
    lists of id strings;
  * model inference runs on row chunks so LightGBM never sees a huge matrix;
  * the per-country candidate set is cached in work/cands_cache so a re-score
    with a retrained model skips the ~75 minute blocking stage.

Usage:
    python predict.py --tag _v2                 # full test set
    python predict.py --limit 3000 --tag _v2    # smoke test -> work/smoke/
"""
import argparse
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import sparse_dot_topn  # noqa: F401  (must load before lightgbm: two OpenMP runtimes on macOS segfault otherwise)
import lightgbm as lgb

import config as C
import features as FT
from blocking import block
from evaluate import decide
from features import STAGE2_KEEP, add_name_freq, add_relational, pair_features

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


def write_groups(fh, q_rows, q_sorted, p_sorted, s1_ids, pool_ids):
    """Append one line per query in q_rows: '<s1 id>\\t<comma ids>' where the
    ids are pool_ids[p_sorted] of the rows with q_sorted == that query.
    q_sorted must be sorted ascending; rows for a query are contiguous."""
    lo = np.searchsorted(q_sorted, q_rows, side="left")
    hi = np.searchsorted(q_sorted, q_rows, side="right")
    lines = []
    for qi, a, b in zip(q_rows, lo, hi):
        ids = ",".join(pool_ids[p_sorted[a:b]]) if b > a else ""
        lines.append(f"{s1_ids[qi]}\t{ids}\n")
        if len(lines) >= 100_000:
            fh.writelines(lines); lines = []
    fh.writelines(lines)


def predict_chunks(model, X, cols, chunk=2_000_000):
    out = np.empty(len(X), np.float32)
    for a in range(0, len(X), chunk):
        out[a:a + chunk] = model.predict(X.iloc[a:a + chunk][cols])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--chunk", type=int, default=1_500_000)
    ap.add_argument("--unseen-thr-bump", type=float, default=None,
                    help="threshold increase for countries absent from training")
    ap.add_argument("--no-cands-cache", action="store_true")
    ap.add_argument("--countries", default="",
                    help="comma-separated subset of countries to score (for splitting the test set "
                         "across machines); the partial outputs are merged with merge_outputs.py")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for the output file names, e.g. _india -> matching_results_india.tsv")
    args = ap.parse_args()
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]

    with open(C.WORK_DIR / f"model_meta{args.tag}.json") as fh:
        meta = json.load(fh)
    m1 = lgb.Booster(model_file=str(C.WORK_DIR / f"model_stage1{args.tag}.txt"))
    m2 = lgb.Booster(model_file=str(C.WORK_DIR / f"model_stage2{args.tag}.txt"))
    dec = meta["decision"]
    FT.ALONE_FULL = bool(meta.get("alone_full", False))
    bump = args.unseen_thr_bump if args.unseen_thr_bump is not None else meta.get("unseen_thr_bump", 0.0)
    train_countries = set(meta.get("train_countries", []))
    log(f"model{args.tag}: decision={dec} use_stage2={meta['use_stage2']} alone_full={FT.ALONE_FULL} unseen_bump={bump}")

    s1 = pd.read_parquet(C.WORK_DIR / "test_s1.parquet", columns=KEEP_COLS)
    pool = pd.read_parquet(C.WORK_DIR / "test_pool.parquet", columns=KEEP_COLS + ["source"])
    out_dir = C.OUTPUT_DIR
    limit_mask = np.ones(len(s1), bool)
    if args.limit:
        limit_mask[args.limit:] = False
        out_dir = C.WORK_DIR / "smoke"
    if countries:
        limit_mask &= np.isin(s1.country.values, countries)
        log(f"scoring only {countries}: {int(limit_mask.sum()):,} entities")
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_ids = s1.entity_id.values.astype(object)
    pool_ids = pool.entity_id.values.astype(object)
    log(f"loaded test: s1={len(s1):,} pool={len(pool):,}")
    add_name_freq(s1, pool)

    cache_dir = C.WORK_DIR / "cands_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sfx = args.out_suffix
    f_c = open(out_dir / f"candidate_pairs{sfx}.tsv", "w", encoding="utf-8", newline="\n")
    f_m = open(out_dir / f"matching_results{sfx}.tsv", "w", encoding="utf-8", newline="\n")
    f_c.write("source1_entity_id\tcandidate_entity_ids\n")
    f_m.write("source1_entity_id\tmatched_entity_ids\n")
    tot_pairs = tot_matches = tot_nonempty = 0

    for ctry in pd.unique(s1.country.values[limit_mask]):
        q_mask = limit_mask & (s1.country.values == ctry)
        q_rows = np.flatnonzero(q_mask)
        tc = time.time()
        cache = cache_dir / f"cands_test_{ctry}_{int(q_mask.sum())}_{len(pool)}.parquet"
        if cache.exists() and not args.no_cands_cache:
            cands = pd.read_parquet(cache)
            log(f"[{ctry}] candidates loaded from cache: {len(cands):,} pairs")
        else:
            cands = block(s1, pool, q_mask=q_mask, log=log, rev_cache=C.WORK_DIR / "rev_cache")
            cands = cands.sort_values(["q_idx", "p_idx"], kind="stable").reset_index(drop=True)
            if not args.no_cands_cache:
                cands.to_parquet(cache, index=False)
        q_all = cands.q_idx.values
        write_groups(f_c, q_rows, q_all, cands.p_idx.values, s1_ids, pool_ids)
        f_c.flush()
        tot_pairs += len(cands)
        log(f"[{ctry}] {len(q_rows):,} entities, {len(cands):,} candidates ({len(cands)/max(1,len(q_rows)):.1f}/entity); candidate lines written")
        add_relational(cands, q_all, cands.p_idx.values, "sim_full", dims=("q", "p"))

        # ---- stage 1 in query-aligned chunks ----
        feat_cols = meta["feat_cols"]
        tab = cands[["q_idx", "p_idx"]].copy()
        for col in STAGE2_KEEP:
            tab[col] = np.zeros(len(tab), np.float32)
        tab["p1"] = np.zeros(len(tab), np.float32)
        for a, b in chunk_bounds(q_all, args.chunk):
            F = pair_features(cands.iloc[a:b], s1, pool)
            tab.iloc[a:b, tab.columns.get_indexer(STAGE2_KEEP)] = F[STAGE2_KEEP].values.astype(np.float32)
            tab.iloc[a:b, tab.columns.get_loc("p1")] = m1.predict(F[feat_cols]).astype(np.float32)
            del F
            log(f"[{ctry}]   stage-1 scored rows {a:,}-{b:,}")
        del cands

        # ---- stage 2 (in place, chunked inference) ----
        if meta["use_stage2"]:
            add_relational(tab, tab.q_idx.values, tab.p_idx.values, "p1", dims=("q", "p"))
            df = pd.DataFrame({"q": tab.q_idx.values, "p1": tab.p1.values})
            tab["p1_sum_q"] = df.groupby("q")["p1"].transform("sum").values.astype(np.float32)
            tab["p1_n50_q"] = (df.p1 >= 0.5).groupby(df.q).transform("sum").values.astype(np.int16)
            tab["p1_n80_q"] = (df.p1 >= 0.8).groupby(df.q).transform("sum").values.astype(np.int16)
            del df
            prob = predict_chunks(m2, tab, meta["stage2_cols"])
        else:
            prob = tab.p1.values.astype(np.float32)
        q_idx, p_idx = tab.q_idx.values, tab.p_idx.values
        log(f"[{ctry}]   stage-2 probabilities ready")

        # ---- decision layer ----
        thr = dec["thr"] + (bump if (bump and ctry not in train_countries) else 0.0)
        keep = decide(q_idx, p_idx, prob, thr, one_parent=dec["one_parent"],
                      rel_gap=dec["rel_gap"], min_margin=dec.get("min_margin"))
        write_groups(f_m, q_rows, q_idx[keep], p_idx[keep], s1_ids, pool_ids)
        f_m.flush()
        n_match = int(keep.sum())
        n_nonempty = len(np.unique(q_idx[keep]))
        tot_matches += n_match; tot_nonempty += n_nonempty
        pd.DataFrame({"q_idx": q_idx, "p_idx": p_idx, "prob": prob, "keep": keep}) \
            .to_parquet(out_dir / f"scored_{ctry}.parquet", index=False)
        log(f"[{ctry}] thr={thr:.2f}: {n_match:,} matches, {n_nonempty:,}/{len(q_rows):,} entities non-empty "
            f"({n_match/max(1,len(q_rows)):.2f}/entity) in {time.time()-tc:.0f}s")
        del tab, prob, keep, q_idx, p_idx

    f_c.close(); f_m.close()
    n_ent = int(limit_mask.sum())
    log(f"done: {n_ent:,} entities, {tot_pairs:,} candidates, {tot_matches:,} matches, "
        f"{tot_nonempty:,} non-empty ({tot_nonempty/max(1,n_ent):.3f})")

    if not args.limit and not countries:
        cmd = [sys.executable, str(C.ROOT / "student_resource" / "utils" / "validate_submission.py"),
               "--matching", str(out_dir / f"matching_results{sfx}.tsv"),
               "--candidate", str(out_dir / f"candidate_pairs{sfx}.tsv"),
               "--test-dir", str(C.TEST_DIR)]
        log("running validator ...")
        print(subprocess.run(cmd, capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main()
