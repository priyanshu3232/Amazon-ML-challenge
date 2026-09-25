"""Step 2: build candidate pairs for a sample of training S1 entities, train the
two-stage LightGBM matcher, tune the decision layer on a held-out set of S1
entities, and save everything needed by predict.py.

Usage:
    python train.py --n-queries 60000     # quick dev run
    python train.py --n-queries 400000    # full-strength run (default)

Validation is split BY S1 ENTITY (never by pair) and both splits are blocked
against the complete Source-2/3 pool, exactly like test time.
"""
import argparse
import json
import time

import sparse_dot_topn  # noqa: F401  (must load before lightgbm: two OpenMP runtimes on macOS segfault otherwise)
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import config as C
import features as FT
from blocking import block, blocking_recall, per_pass_recall
from evaluate import breakdown, decide, tune
from features import STAGE2_KEEP, add_relational, pair_features, stage2_features
from io_utils import read_ground_truth

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.0f}s] {msg}", flush=True)


def label_pairs(cands, s1_ids, pool_ids, gt):
    parent = {}
    for s1, ids in gt.items():
        for m in ids:
            parent[m] = s1
    pid = pool_ids[cands.p_idx.values]
    par = np.array([parent.get(x, "") for x in pid], dtype=object)
    return (par == s1_ids[cands.q_idx.values]).astype(np.int8)


def fit_lgb(Xtr, ytr, Xva, yva, rounds, log=log):
    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=False)
    m = lgb.train(C.LGB_PARAMS, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(C.LGB_EARLY_STOP, verbose=False),
                             lgb.log_evaluation(0)])
    log(f"  best_iter={m.best_iteration} val_logloss={m.best_score['valid_0']['binary_logloss']:.5f}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=400000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--train-countries", default="",
                    help="comma-separated country labels used for TRAINING pairs "
                         "(validation still covers all sampled countries); "
                         "leave-one-country-out proxy for the unseen test country")
    args = ap.parse_args()
    C.WORK_DIR.mkdir(parents=True, exist_ok=True)

    KEEP = ["entity_id", "country", "name_full", "name_core", "name_alias", "is_domain", "addr_norm"]
    s1 = pd.read_parquet(C.WORK_DIR / "train_s1.parquet", columns=KEEP)
    pool = pd.read_parquet(C.WORK_DIR / "train_pool.parquet", columns=KEEP + ["source"])
    gt = read_ground_truth(C.TRAIN_DIR / "train_ground_truth.tsv")
    log(f"loaded s1={len(s1):,} pool={len(pool):,}")

    # sample of S1 entities for forward blocking / training; val split by entity
    rng = np.random.RandomState(C.SEED)
    take = rng.permutation(len(s1))[: args.n_queries]
    q_mask = np.zeros(len(s1), bool); q_mask[take] = True
    is_val = np.zeros(len(s1), bool); is_val[take] = rng.rand(len(take)) < C.VAL_FRACTION
    s1_ids = s1.entity_id.values
    pool_ids = pool.entity_id.values
    n_gt = np.array([len(gt.get(x, ())) for x in s1_ids])
    log(f"sampled {q_mask.sum():,} S1 queries ({is_val.sum():,} val); "
        f"mean GT matches={n_gt[q_mask].mean():.2f}")

    FT.ALONE_FULL = True
    cands = block(s1, pool, q_mask=q_mask, log=log, rev_cache=C.WORK_DIR / "rev_cache", ctx=True)
    is_ctx = cands.ctx.values.astype(bool)
    cands = cands.drop(columns=["ctx"])
    log(f"candidate rows: {len(cands):,} of which context rows (competing parents): {is_ctx.sum():,}")
    gt_s = {k: gt[k] for k in s1_ids[q_mask]}
    main = cands[~is_ctx]
    rec = blocking_recall(main, s1_ids, pool_ids, gt_s)
    log(f"blocking recall={rec['recall']:.4f} ({rec['found']:,}/{rec['gt_pairs']:,}); "
        f"pairs={len(main):,} ({len(main)/q_mask.sum():.1f}/query)")
    for ctry in s1.country[q_mask].unique():
        m = s1.country.values[main.q_idx.values] == ctry
        sub = main[m]
        ids_c = set(s1_ids[q_mask & (s1.country.values == ctry)])
        r = blocking_recall(sub, s1_ids, pool_ids, {k: v for k, v in gt_s.items() if k in ids_c})
        log(f"  {ctry}: recall={r['recall']:.4f} pairs/query={len(sub)/max(1,len(ids_c)):.1f}")
    per_pass_recall(main, s1_ids, pool_ids, gt_s, log=log)
    del main
    add_relational(cands, cands.q_idx.values, cands.p_idx.values, "sim_full", dims=("q", "p"))

    y = label_pairs(cands, s1_ids, pool_ids, gt)
    log(f"labels: {y.sum():,} positives / {len(y):,} pairs")
    F = pair_features(cands, s1, pool)
    feat_cols = list(F.columns)
    log(f"features: {len(feat_cols)} columns")

    qv = is_val[cands.q_idx.values] & ~is_ctx
    tr_sel = ~qv & ~is_ctx
    if args.train_countries:
        allowed = [c.strip() for c in args.train_countries.split(",") if c.strip()]
        tr_sel &= np.isin(s1.country.values[cands.q_idx.values], allowed)
        log(f"training pairs restricted to countries {allowed}: {tr_sel.sum():,} pairs")
    Xtr, ytr, Xva, yva = F[tr_sel], y[tr_sel], F[qv], y[qv]

    # ---- stage 1 ----
    log("stage 1: LightGBM on pairwise features")
    m1 = fit_lgb(Xtr, ytr, Xva, yva, C.LGB_ROUNDS)
    imp = pd.Series(m1.feature_importance("gain"), index=feat_cols).sort_values(ascending=False)
    log("  top features: " + ", ".join(f"{k}={v:.0f}" for k, v in imp.head(12).items()))
    p1_va = m1.predict(Xva, num_iteration=m1.best_iteration)
    # out-of-fold stage-1 predictions for the training pairs
    p1_tr = np.zeros(len(Xtr), np.float32)
    gkf = GroupKFold(n_splits=4)
    qtr = cands.q_idx.values[tr_sel]
    for k, (a, b) in enumerate(gkf.split(Xtr, ytr, groups=qtr)):
        mk = fit_lgb(Xtr.iloc[a], ytr[a], Xtr.iloc[b], ytr[b], m1.best_iteration + 50)
        p1_tr[b] = mk.predict(Xtr.iloc[b], num_iteration=mk.best_iteration)
        log(f"  oof fold {k} done")

    # ---- stage 2 ----
    log("stage 2: LightGBM on stage-1 probability + relational statistics")
    tab = cands[["q_idx", "p_idx"]].copy()
    tab[STAGE2_KEEP] = F[STAGE2_KEEP].values
    p1_all = m1.predict(F, num_iteration=m1.best_iteration).astype(np.float32)  # val + ctx rows
    p1_all[tr_sel] = p1_tr                                                     # OOF for train rows
    tab["p1"] = p1_all
    G = stage2_features(tab)
    G_tr, G_va = G[tr_sel], G[qv]
    m2 = fit_lgb(G_tr, ytr, G_va, yva, C.LGB_ROUNDS)
    p2_va = m2.predict(G_va, num_iteration=m2.best_iteration)

    # ---- decision layer tuned on validation S1 entities ----
    n_q = len(s1)
    q_va = cands.q_idx.values[qv]; p_va = cands.p_idx.values[qv]
    val_mask_q = is_val
    n_gt_val = np.where(val_mask_q, n_gt, 0)

    def macro_on_val(prob, keep):
        from evaluate import f05_per_entity
        n_pred = np.bincount(q_va[keep], minlength=n_q)
        tp = np.bincount(q_va[keep & (yva == 1)], minlength=n_q)
        f = f05_per_entity(tp, n_pred, n_gt_val)
        return f[val_mask_q].mean()

    log("tuning decision layer on stage-1 probabilities")
    best1 = tune(q_va, p_va, p1_va, yva, n_q, n_gt_val, log=log)
    best1["f05"] = macro_on_val(p1_va, decide(q_va, p_va, p1_va, best1["thr"], best1["one_parent"], rel_gap=best1["rel_gap"], min_margin=best1["min_margin"]))
    log(f"  stage-1 best: {best1}")
    log("tuning decision layer on stage-2 probabilities")
    best2 = tune(q_va, p_va, p2_va, yva, n_q, n_gt_val, log=log)
    best2["f05"] = macro_on_val(p2_va, decide(q_va, p_va, p2_va, best2["thr"], best2["one_parent"], rel_gap=best2["rel_gap"], min_margin=best2["min_margin"]))
    log(f"  stage-2 best: {best2}")

    use2 = best2["f05"] >= best1["f05"]
    best = best2 if use2 else best1
    prob = p2_va if use2 else p1_va
    keep = decide(q_va, p_va, prob, best["thr"], best["one_parent"], rel_gap=best["rel_gap"], min_margin=best["min_margin"])
    log(f"validation macro F0.5 = {best['f05']:.4f} (stage {'2' if use2 else '1'})")
    from evaluate import f05_per_entity
    n_pred = np.bincount(q_va[keep], minlength=n_q)
    tp = np.bincount(q_va[keep & (yva == 1)], minlength=n_q)
    f = f05_per_entity(tp, n_pred, n_gt_val)
    vm = val_mask_q
    bucket = np.where(n_gt == 0, "0", np.where(n_gt == 1, "1", np.where(n_gt <= 3, "2-3", "4+")))
    log(f"  singletons: {((n_gt==0)&(n_pred==0)&vm).sum()}/{((n_gt==0)&vm).sum()} correct")
    n_gt = np.where(q_mask, n_gt, 0)
    for gname, garr in (("country", s1.country.values), ("gt_bucket", bucket)):
        for val in pd.unique(garr[vm]):
            mm = vm & (garr == val)
            P = tp[mm].sum() / max(1, n_pred[mm].sum()); R = tp[mm].sum() / max(1, n_gt[mm].sum())
            log(f"  {gname}={val}: n={mm.sum():,} f05={f[mm].mean():.4f} pairP={P:.3f} pairR={R:.3f}")

    # per-country threshold sensitivity: F0.5 at the global threshold vs the
    # country-optimal threshold (tells us how much an unseen country may shift)
    from evaluate import macro_f05
    ctry_q = s1.country.values
    for val in pd.unique(ctry_q[vm]):
        rows_c = ctry_q[q_va] == val
        ent_c = vm & (ctry_q == val)
        best_c = (None, -1)
        for thr in np.round(np.arange(0.20, 0.96, 0.02), 2):
            kp = decide(q_va[rows_c], p_va[rows_c], prob[rows_c], thr, best["one_parent"], rel_gap=best["rel_gap"], min_margin=best["min_margin"])
            n_pred_c = np.bincount(q_va[rows_c][kp], minlength=n_q)
            tp_c = np.bincount(q_va[rows_c][kp & (yva[rows_c] == 1)], minlength=n_q)
            fc = f05_per_entity(tp_c, n_pred_c, n_gt_val)[ent_c].mean()
            if fc > best_c[1]:
                best_c = (thr, fc)
        log(f"  threshold sensitivity {val}: global thr={best['thr']:.2f} f05={f[ent_c].mean():.4f} | "
            f"country-optimal thr={best_c[0]:.2f} f05={best_c[1]:.4f}")

    # ---- persist ----
    tag = args.tag
    m1.save_model(str(C.WORK_DIR / f"model_stage1{tag}.txt"), num_iteration=m1.best_iteration)
    m2.save_model(str(C.WORK_DIR / f"model_stage2{tag}.txt"), num_iteration=m2.best_iteration)
    meta = {"feat_cols": feat_cols, "stage2_cols": list(G_tr.columns), "use_stage2": bool(use2),
            "decision": best, "stage1_decision": best1, "stage2_decision": best2,
            "blocking_recall": rec, "n_queries": int(q_mask.sum()),
            "train_countries": sorted(s1.country.unique().tolist()),
            "alone_full": True}
    with open(C.WORK_DIR / f"model_meta{tag}.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    val_out = pd.DataFrame({"s1": s1_ids[q_va], "cand": pool_ids[p_va], "label": yva,
                            "p1": p1_va.astype(np.float32), "p2": p2_va.astype(np.float32), "keep": keep})
    val_out.to_parquet(C.WORK_DIR / f"val_pairs{tag}.parquet", index=False)
    log("saved models, meta and validation pairs")


if __name__ == "__main__":
    main()
