"""Pairwise features for (Source-1, Source-2/3) candidate pairs.

All string features work on the normalized, ASCII, lower-cased text produced
by normalize.py, so they behave the same for every country and script.
Relational features (rank / gap inside a query's candidate list and inside a
pool record's list of possible parents) encode the structural fact that each
Source-2/3 record has at most one Source-1 parent.
"""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from normalize import LEGAL_SET

_PIN_RE = re.compile(r"^\d{5,6}$")

# When True, a row that is alone in its group gets margin = its own score
# (best_other = 0) instead of 0. Recorded in model meta; set by train/predict.
ALONE_FULL = False


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _tokset(s):
    return set(s.split()) if s else set()


def _jacc(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cover(a, b):
    """fraction of tokens of a present in b"""
    return len(a & b) / len(a) if a else 0.0


def _tri(s):
    s = f" {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _suffix_state(a_full, b_full):
    sa = {t for t in a_full.split() if t in LEGAL_SET}
    sb = {t for t in b_full.split() if t in LEGAL_SET}
    if not sa and not sb:
        return 0
    if sa == sb:
        return 1
    if not sa or not sb:
        return 2
    return 4 if (sa & sb) else 3


def _num_tokens(addr):
    return [t for t in addr.split() if any(c.isdigit() for c in t)]


def _house_state(n1, n2):
    """0 both missing, 1 equal, 2 one missing, 3 different, 4 one contains other."""
    if not n1 and not n2:
        return 0
    if not n1 or not n2:
        return 2
    if n1 == n2:
        return 1
    d1 = "".join(c for c in n1 if c.isdigit())
    d2 = "".join(c for c in n2 if c.isdigit())
    if d1 and d2 and (d1 == d2 or d1.startswith(d2) or d2.startswith(d1)):
        return 4
    return 3


def pair_features(cands: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Compute features for every row of cands (q_idx, p_idx, sim_*)."""
    q = cands.q_idx.values
    p = cands.p_idx.values
    F = pd.DataFrame(index=cands.index)

    n1 = s1.name_full.values[q]; n2 = pool.name_full.values[p]
    c1 = s1.name_core.values[q]; c2 = pool.name_core.values[p]
    al2 = pool.name_alias.values[p]
    a1 = s1.addr_norm.values[q]; a2 = pool.addr_norm.values[p]

    # ---- name similarity (C-implemented, multi-threaded) ----
    F["name_ratio"] = _cp(n1, n2, fuzz.ratio)
    F["name_tsr"] = _cp(n1, n2, fuzz.token_sort_ratio)
    F["name_tset"] = _cp(n1, n2, fuzz.token_set_ratio)
    F["name_partial"] = _cp(n1, n2, fuzz.partial_ratio)
    F["name_jw"] = _cp(n1, n2, JaroWinkler.normalized_similarity)
    F["core_ratio"] = _cp(c1, c2, fuzz.ratio)
    F["core_tsr"] = _cp(c1, c2, fuzz.token_sort_ratio)
    F["core_tset"] = _cp(c1, c2, fuzz.token_set_ratio)
    F["core_partial"] = _cp(c1, c2, fuzz.partial_ratio)
    F["core_jw"] = _cp(c1, c2, JaroWinkler.normalized_similarity)
    ns1 = [x.replace(" ", "") for x in c1]; ns2 = [x.replace(" ", "") for x in c2]
    F["nospace_ratio"] = _cp(ns1, ns2, fuzz.ratio)
    F["nospace_jw"] = _cp(ns1, ns2, JaroWinkler.normalized_similarity)
    alias_ratio = _cp(c1, al2, fuzz.token_set_ratio)
    alias_ratio[np.asarray(al2) == ""] = 0
    F["alias_tset"] = alias_ratio
    F["best_core"] = np.maximum(F.core_tset.values, alias_ratio)

    # ---- token-level name features (python, cheap sets) ----
    ts1 = [_tokset(x) for x in c1]; ts2 = [_tokset(x) for x in c2]
    F["core_jacc"] = np.fromiter((_jacc(a, b) for a, b in zip(ts1, ts2)), np.float32, len(q))
    F["core_cover1"] = np.fromiter((_cover(a, b) for a, b in zip(ts1, ts2)), np.float32, len(q))
    F["core_cover2"] = np.fromiter((_cover(b, a) for a, b in zip(ts1, ts2)), np.float32, len(q))
    F["core_shared"] = np.fromiter((len(a & b) for a, b in zip(ts1, ts2)), np.int16, len(q))
    F["core_ntok1"] = np.fromiter((len(a) for a in ts1), np.int16, len(q))
    F["core_ntok2"] = np.fromiter((len(b) for b in ts2), np.int16, len(q))
    F["core_ntok_diff"] = (F.core_ntok1 - F.core_ntok2).astype(np.int16)
    F["first_tok_eq"] = np.fromiter(
        ((a.split(" ", 1)[0] == b.split(" ", 1)[0]) if a and b else False for a, b in zip(c1, c2)),
        np.int8, len(q))
    F["tri_jacc"] = np.fromiter((_jacc(_tri(a), _tri(b)) for a, b in zip(c1, c2)), np.float32, len(q))
    F["name_len1"] = np.fromiter((len(x) for x in n1), np.int16, len(q))
    F["name_len2"] = np.fromiter((len(x) for x in n2), np.int16, len(q))
    F["suffix_state"] = np.fromiter((_suffix_state(a, b) for a, b in zip(n1, n2)), np.int8, len(q))
    F["is_domain2"] = pool.is_domain.values[p].astype(np.int8)
    F["has_alias2"] = (np.asarray(al2) != "").astype(np.int8)

    # ---- address similarity ----
    F["addr_ratio"] = _cp(a1, a2, fuzz.ratio)
    F["addr_tsr"] = _cp(a1, a2, fuzz.token_sort_ratio)
    F["addr_tset"] = _cp(a1, a2, fuzz.token_set_ratio)
    F["addr_partial"] = _cp(a1, a2, fuzz.partial_ratio)
    as1 = [_tokset(x) for x in a1]; as2 = [_tokset(x) for x in a2]
    F["addr_jacc"] = np.fromiter((_jacc(a, b) for a, b in zip(as1, as2)), np.float32, len(q))
    F["addr_cover1"] = np.fromiter((_cover(a, b) for a, b in zip(as1, as2)), np.float32, len(q))
    F["addr_cover2"] = np.fromiter((_cover(b, a) for a, b in zip(as1, as2)), np.float32, len(q))
    F["addr_shared"] = np.fromiter((len(a & b) for a, b in zip(as1, as2)), np.int16, len(q))
    F["addr_ntok1"] = np.fromiter((len(a) for a in as1), np.int16, len(q))
    F["addr_ntok2"] = np.fromiter((len(b) for b in as2), np.int16, len(q))
    F["addr_empty1"] = (F.addr_ntok1 == 0).astype(np.int8)
    F["addr_empty2"] = (F.addr_ntok2 == 0).astype(np.int8)
    nums1 = [_num_tokens(x) for x in a1]; nums2 = [_num_tokens(x) for x in a2]
    F["house_state"] = np.fromiter(
        (_house_state(x[0] if x else "", y[0] if y else "") for x, y in zip(nums1, nums2)), np.int8, len(q))
    F["num_jacc"] = np.fromiter((_jacc(set(x), set(y)) for x, y in zip(nums1, nums2)), np.float32, len(q))
    F["num_shared"] = np.fromiter((len(set(x) & set(y)) for x, y in zip(nums1, nums2)), np.int16, len(q))
    F["num_cnt1"] = np.fromiter((len(x) for x in nums1), np.int16, len(q))
    F["num_cnt2"] = np.fromiter((len(y) for y in nums2), np.int16, len(q))
    pin1 = [next((t for t in x if _PIN_RE.match(t)), "") for x in nums1]
    pin2 = [next((t for t in y if _PIN_RE.match(t)), "") for y in nums2]
    F["pin_state"] = np.fromiter((_house_state(a, b) for a, b in zip(pin1, pin2)), np.int8, len(q))
    # alpha-only address tokens (street/city words) overlap
    aw1 = [{t for t in s if t.isalpha()} for s in as1]; aw2 = [{t for t in s if t.isalpha()} for s in as2]
    F["addr_word_jacc"] = np.fromiter((_jacc(a, b) for a, b in zip(aw1, aw2)), np.float32, len(q))
    F["addr_word_shared"] = np.fromiter((len(a & b) for a, b in zip(aw1, aw2)), np.int16, len(q))
    # last token (state / region) agreement
    F["last_tok_eq"] = np.fromiter(
        ((a.rsplit(" ", 1)[-1] == b.rsplit(" ", 1)[-1]) if a and b else False for a, b in zip(a1, a2)),
        np.int8, len(q))

    # ---- blocking similarities, global relational columns & misc ----
    for col in cands.columns:
        if col not in ("q_idx", "p_idx"):
            F[col] = cands[col].values
    F["source"] = pool.source.values[p].astype(np.int8)

    # ---- relational features on a heuristic score (within query only; a
    # feature chunk always contains complete query groups) ----
    F["heur"] = (0.5 * F.best_core + 0.3 * F.addr_tset + 20 * F.sim_full).astype(np.float32)
    add_relational(F, q, p, "heur", dims=("q",))
    return F


def add_relational(F: pd.DataFrame, q, p, col: str, dims=("q", "p")) -> None:
    """Rank / gap / count / margin of `col` inside the S1 group (q) and/or the
    pool-record group (p). Vectorized: one lexsort per dimension."""
    s = np.asarray(F[col].values, dtype=np.float32)
    for name, key in (("q", q), ("p", p)):
        if name not in dims:
            continue
        key = np.asarray(key)
        order = np.lexsort((-s, key))
        ks, ss = key[order], s[order]
        start = np.ones(len(order), bool)
        start[1:] = ks[1:] != ks[:-1]
        grp_id = np.cumsum(start) - 1
        first_pos = np.flatnonzero(start)
        size = np.diff(np.append(first_pos, len(order)))
        mx = ss[first_pos][grp_id]
        pos_in_grp = np.arange(len(order)) - first_pos[grp_id]
        second_pos = np.minimum(first_pos + 1, first_pos + size - 1)
        second = ss[second_pos][grp_id]
        if ALONE_FULL:
            second = np.where(size[grp_id] > 1, second, 0.0).astype(np.float32)
        best_other = np.where(pos_in_grp == 0, second, mx)
        gap = np.empty(len(order), np.float32); gap[order] = mx - ss
        rank = np.empty(len(order), np.int16); rank[order] = pos_in_grp + 1
        cnt = np.empty(len(order), np.int16); cnt[order] = size[grp_id]
        margin = np.empty(len(order), np.float32); margin[order] = ss - best_other
        F[f"{col}_gap_{name}"] = gap
        F[f"{col}_rank_{name}"] = rank
        F[f"{col}_cnt_{name}"] = cnt
        F[f"{col}_margin_{name}"] = margin


STAGE2_KEEP = ["best_core", "core_tset", "name_tset", "addr_tset", "addr_jacc",
               "house_state", "num_shared", "sim_full", "sim_rev", "n_pass", "source",
               "heur_gap_q", "heur_rank_q", "core_ntok1", "addr_empty2"]


def stage2_features(tab: pd.DataFrame) -> pd.DataFrame:
    """Second-stage features: stage-1 probability plus its relational statistics.

    tab must contain q_idx, p_idx, p1 and the STAGE2_KEEP columns.
    """
    F = tab[STAGE2_KEEP + ["p1"]].copy()
    add_relational(F, tab.q_idx.values, tab.p_idx.values, "p1", dims=("q", "p"))
    df = pd.DataFrame({"q": tab.q_idx.values, "p1": tab.p1.values})
    F["p1_sum_q"] = df.groupby("q")["p1"].transform("sum").values.astype(np.float32)
    F["p1_n50_q"] = (df.p1 >= 0.5).groupby(df.q).transform("sum").values.astype(np.int16)
    F["p1_n80_q"] = (df.p1 >= 0.8).groupby(df.q).transform("sum").values.astype(np.int16)
    return F
