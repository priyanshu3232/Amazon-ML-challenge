"""Candidate generation (blocking).

For every country label present in the query set we build sparse TF-IDF
matrices over the Source-2/3 pool of that country and take top-K neighbours
for each Source-1 query with sparse_dot_topn. Passes:

  full : word TF-IDF over normalized name + address   (top-K per S1)
  name : word TF-IDF over normalized name only        (top-K per S1)
  char : char 3-gram TF-IDF over normalized name      (top-K per S1)
  rev  : word TF-IDF (name+address), top-K S1 per POOL record (reverse
         direction). Source 1 is deduplicated, so every pool record has at
         most one parent; the reverse pass makes sure that parent is a
         candidate even when the S1 side's list is crowded.

Very frequent tokens / n-grams are pruned (max_df) because they carry almost
no IDF weight and would make each query scan millions of postings.
The union of all passes is the candidate set that is written to
candidate_pairs.tsv and scored by the model.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

import config as C


def _csr32(m):
    m = sp.csr_matrix(m, dtype=np.float32)
    m.indices = m.indices.astype(np.int32)
    m.indptr = m.indptr.astype(np.int32)
    return m


def _topn(A, B_T, k, thr, row_chunk=200_000):
    """Row-wise top-k of A @ B_T -> (rows, cols, vals), computed in row chunks
    of A to keep peak memory bounded."""
    if A.shape[0] == 0 or B_T.shape[1] == 0:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    rows, cols, vals = [], [], []
    for a in range(0, A.shape[0], row_chunk):
        sub = A[a:a + row_chunk]
        if sub.nnz == 0:
            continue
        Cm = sp_matmul_topn(sub, B_T, top_n=k, threshold=thr, sort=True,
                            n_threads=C.N_THREADS).tocoo()
        rows.append(Cm.row.astype(np.int64) + a)
        cols.append(Cm.col.astype(np.int64))
        vals.append(Cm.data.astype(np.float32))
    if not rows:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def _word_vec(max_df):
    return TfidfVectorizer(analyzer="word", token_pattern=r"[a-z0-9]+",
                           min_df=2, max_df=max_df, sublinear_tf=True,
                           dtype=np.float32, lowercase=False)


def _char_vec(max_df):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                           max_df=max_df, sublinear_tf=True, dtype=np.float32,
                           lowercase=False)


def block(s1: pd.DataFrame, pool: pd.DataFrame, q_mask=None, log=print,
          rev_cache=None, ctx=False) -> pd.DataFrame:
    """Return candidate pairs with per-pass similarity columns.

    s1 / pool must carry columns: country, name_full, addr_norm.
    q_mask: optional boolean array over s1 rows; forward passes run only for
    masked queries, while the reverse pass always uses ALL s1 rows (so that a
    sub-sample sees the same candidate density as a full run) and keeps only
    pairs whose query is masked.
    rev_cache: optional directory; the (sample-independent) reverse pass result
    per country is stored there and reused by later runs on the same split.
    ctx: when True (training on a sample), reverse-pass pairs of UNMASKED
    queries are also kept for every pool record that is a candidate of a masked
    query, flagged ctx=1. They give pool-side relational features the same
    "competing parents" a full run sees; they are never trained or scored on.
    Output columns: q_idx (row in s1), p_idx (row in pool),
                    sim_full, sim_name, sim_char, sim_rev, n_pass
    """
    if q_mask is None:
        q_mask = np.ones(len(s1), bool)
    q_mask = np.asarray(q_mask, bool)
    pieces = []
    n_pool = len(pool)
    for ctry in s1.country[q_mask].unique():
        q_all = np.flatnonzero(s1.country.values == ctry)          # for reverse pass
        q_rows = q_all[q_mask[q_all]]                               # forward queries
        p_rows = np.flatnonzero(pool.country.values == ctry)
        t0 = time.time()
        if len(p_rows) == 0:
            log(f"[block] {ctry}: no pool records -> no candidates")
            continue
        qs, qa, ps = s1.iloc[q_rows], s1.iloc[q_all], pool.iloc[p_rows]
        q_full = (qs.name_full + " " + qs.addr_norm).values
        qa_full = (qa.name_full + " " + qa.addr_norm).values
        p_full = (ps.name_full + " " + ps.addr_norm).values
        keys, scores = [], {}

        def add(name, qidx, pidx, v):
            k = qidx.astype(np.int64) * n_pool + pidx
            keys.append(k)
            scores[name] = (k, v.astype(np.float32))

        # pass: full (name + address), forward
        t1 = time.time()
        vec = _word_vec(C.BLOCK_WORD_MAX_DF)
        P = _csr32(vec.fit_transform(p_full))
        PT = _csr32(P.T)
        Q = _csr32(vec.transform(q_full))
        r, c, v = _topn(Q, PT, C.BLOCK_TOPK_WORD, C.BLOCK_MIN_SIM)
        add("sim_full", q_rows[r], p_rows[c], v)
        del Q, PT
        log(f"[block] {ctry}: forward full pass {len(v):,} pairs in {time.time()-t1:.0f}s")
        # pass: reverse (pool -> all S1 of this country), keep masked queries
        t1 = time.time()
        cache_file = None
        if rev_cache is not None:
            cache_file = Path(rev_cache) / f"rev_{ctry}_{len(s1)}_{n_pool}_{C.BLOCK_TOPK_REVERSE}_{C.BLOCK_REV_MIN_SIM}.parquet"
        if cache_file is not None and cache_file.exists():
            rc = pd.read_parquet(cache_file)
            rq, rp, v2 = rc.q.values, rc.p.values, rc.s.values.astype(np.float32)
            log(f"[block] {ctry}: reverse pass loaded from cache ({len(rq):,} pairs)")
        else:
            QA = _csr32(vec.transform(qa_full))
            QAT = _csr32(QA.T)
            pr, qc, v2 = _topn(P, QAT, C.BLOCK_TOPK_REVERSE, C.BLOCK_REV_MIN_SIM)
            rq, rp = q_all[qc], p_rows[pr]
            del QA, QAT
            if cache_file is not None:
                Path(rev_cache).mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"q": rq.astype(np.int32), "p": rp.astype(np.int32), "s": v2}).to_parquet(cache_file, index=False)
            log(f"[block] {ctry}: reverse pass {len(rq):,} pairs in {time.time()-t1:.0f}s")
        keep = q_mask[rq]
        add("sim_rev", rq[keep], rp[keep], v2[keep])
        ctx_keys = None
        if ctx:
            # pool records that are candidates of masked queries (so far: forward + reverse)
            cand_p = np.unique(np.concatenate([k % n_pool for k in keys]))
            sel = (~keep) & np.isin(rp, cand_p)
            ctx_keys = rq[sel].astype(np.int64) * n_pool + rp[sel]
            ctx_vals = v2[sel]
        del P
        # pass: name only (words)
        t1 = time.time()
        vec = _word_vec(C.BLOCK_WORD_MAX_DF)
        P = _csr32(vec.fit_transform(ps.name_full.values))
        Q = _csr32(vec.transform(qs.name_full.values))
        r, c, v = _topn(Q, _csr32(P.T), C.BLOCK_TOPK_NAME, C.BLOCK_MIN_SIM)
        add("sim_name", q_rows[r], p_rows[c], v)
        del P, Q
        log(f"[block] {ctry}: name pass {len(v):,} pairs in {time.time()-t1:.0f}s")
        # pass: name char 3-grams
        t1 = time.time()
        if C.BLOCK_TOPK_CHAR > 0:
            vec = _char_vec(C.BLOCK_CHAR_MAX_DF)
            P = _csr32(vec.fit_transform(ps.name_full.values))
            Q = _csr32(vec.transform(qs.name_full.values))
            r, c, v = _topn(Q, _csr32(P.T), C.BLOCK_TOPK_CHAR, C.BLOCK_MIN_SIM)
            add("sim_char", q_rows[r], p_rows[c], v)
            del P, Q
            log(f"[block] {ctry}: char pass {len(v):,} pairs in {time.time()-t1:.0f}s")
        else:
            add("sim_char", np.array([], np.int64), np.array([], np.int64), np.array([], np.float32))
        # pass: bare domain names ("manishadvisory.com") vs space-less S1 core names
        t1 = time.time()
        dom_rows = np.flatnonzero(ps.is_domain.values == 1) if "is_domain" in ps.columns else np.array([], int)
        if len(dom_rows) and C.BLOCK_TOPK_DOMAIN > 0:
            vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1, sublinear_tf=True,
                                  dtype=np.float32, lowercase=False)
            P = _csr32(vec.fit_transform(ps.name_core.values[dom_rows]))
            Q = _csr32(vec.transform([x.replace(" ", "") for x in qs.name_core.values]))
            r, c, v = _topn(Q, _csr32(P.T), C.BLOCK_TOPK_DOMAIN, C.BLOCK_DOMAIN_MIN_SIM)
            add("sim_dom", q_rows[r], p_rows[dom_rows[c]], v)
            del P, Q
            log(f"[block] {ctry}: domain pass {len(v):,} pairs ({len(dom_rows):,} domain records) in {time.time()-t1:.0f}s")
        else:
            add("sim_dom", np.array([], np.int64), np.array([], np.int64), np.array([], np.float32))

        main_keys = np.unique(np.concatenate(keys))
        if ctx and ctx_keys is not None and len(ctx_keys):
            # ctx rows: competitor parents for pool records that are candidates
            # of masked queries (recomputed on the final candidate set)
            cand_p = np.unique(main_keys % n_pool)
            sel = np.isin(ctx_keys % n_pool, cand_p)
            ctx_keys, ctx_vals = ctx_keys[sel], ctx_vals[sel]
            scores["sim_rev"] = (np.concatenate([scores["sim_rev"][0], ctx_keys]),
                                 np.concatenate([scores["sim_rev"][1], ctx_vals]))
            allk = np.unique(np.concatenate([main_keys, ctx_keys]))
            is_ctx = ~np.isin(allk, main_keys)
        else:
            allk = main_keys
            is_ctx = np.zeros(len(allk), bool)
        out = pd.DataFrame({
            "q_idx": (allk // n_pool).astype(np.int32),
            "p_idx": (allk % n_pool).astype(np.int32),
        })
        n_pass = np.zeros(len(allk), np.int8)
        for name in ("sim_full", "sim_name", "sim_char", "sim_rev", "sim_dom"):
            k, v = scores[name]
            col = np.zeros(len(allk), np.float32)
            if len(k):
                pos = np.searchsorted(allk, k)
                np.maximum.at(col, pos, v)
            out[name] = col
            n_pass += (col > 0).astype(np.int8)
        out["n_pass"] = n_pass
        if ctx:
            out["ctx"] = is_ctx.astype(np.int8)
        pieces.append(out)
        log(f"[block] {ctry}: {len(q_rows):,} queries x {len(p_rows):,} pool "
            f"-> {len(out):,} pairs ({len(out)/max(1,len(q_rows)):.1f}/query) "
            f"in {time.time()-t0:.0f}s")
    if not pieces:
        return pd.DataFrame(columns=["q_idx", "p_idx", "sim_full", "sim_name",
                                     "sim_char", "sim_rev", "sim_dom", "n_pass"])
    return pd.concat(pieces, ignore_index=True)


def blocking_recall(cands: pd.DataFrame, s1_ids, pool_ids, gt: dict) -> dict:
    """Pair-level recall of the candidate set against ground truth."""
    have = set(zip(s1_ids[cands.q_idx.values], pool_ids[cands.p_idx.values]))
    pool_set = set(pool_ids)
    s1_set = set(s1_ids)
    total = hit = 0
    for s1, ids in gt.items():
        if s1 not in s1_set:
            continue
        for m in ids:
            if m in pool_set:
                total += 1
                hit += (s1, m) in have
    return {"gt_pairs": total, "found": hit, "recall": hit / max(1, total)}


def per_pass_recall(cands: pd.DataFrame, s1_ids, pool_ids, gt: dict, log=print) -> None:
    """Log the recall each blocking pass achieves alone and the union."""
    cols = ("sim_full", "sim_name", "sim_char", "sim_rev", "sim_dom")
    for col in cols:
        sub = cands[cands[col] > 0]
        r = blocking_recall(sub, s1_ids, pool_ids, gt)
        others = [c for c in cols if c != col]
        without = cands[(cands[others] > 0).any(axis=1)]
        r2 = blocking_recall(without, s1_ids, pool_ids, gt)
        log(f"  pass {col}: alone recall={r['recall']:.4f} pairs={len(sub):,} | "
            f"union without it: recall={r2['recall']:.4f} pairs={len(without):,}")
