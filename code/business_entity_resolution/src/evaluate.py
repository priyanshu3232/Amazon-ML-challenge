"""Macro F0.5 scoring and decision-layer tuning.

Metric (from the problem statement): F0.5 per Source-1 entity, averaged over
ALL Source-1 entities. A singleton scores 1 when predicted empty, else 0. An
entity with matches but an empty prediction scores 0.
"""
import numpy as np
import pandas as pd


def f05_per_entity(tp, n_pred, n_gt):
    """Vectorized per-entity F0.5."""
    f = np.zeros(len(tp), np.float64)
    single = n_gt == 0
    f[single & (n_pred == 0)] = 1.0
    ok = (~single) & (n_pred > 0) & (tp > 0)
    P = tp[ok] / n_pred[ok]
    R = tp[ok] / n_gt[ok]
    f[ok] = 1.25 * P * R / (0.25 * P + R)
    return f


def one_parent_mask(p_idx, prob):
    """True for the row with the highest prob among rows sharing a pool record."""
    order = np.lexsort((-prob, p_idx))          # sort by p_idx, then prob desc
    first = np.ones(len(order), bool)
    sp = p_idx[order]
    first[1:] = sp[1:] != sp[:-1]
    mask = np.zeros(len(order), bool)
    mask[order[first]] = True
    return mask


def parent_margin(p_idx, prob):
    """For each row: prob minus the best OTHER candidate parent of the same
    pool record (1.0 when the record has a single candidate parent)."""
    order = np.lexsort((-prob, p_idx))
    ps, ss = p_idx[order], prob[order]
    start = np.ones(len(order), bool)
    start[1:] = ps[1:] != ps[:-1]
    first = np.flatnonzero(start)
    size = np.diff(np.append(first, len(order)))
    gid = np.cumsum(start) - 1
    mx = ss[first][gid]
    pos = np.arange(len(order)) - first[gid]
    second = np.where(size[gid] > 1, ss[np.minimum(first + 1, first + size - 1)][gid], -1.0)
    best_other = np.where(pos == 0, second, mx)
    margin = np.empty(len(order), np.float32)
    margin[order] = np.where(best_other < 0, 1.0, ss - best_other)
    return margin


def decide(q_idx, p_idx, prob, thr, one_parent=True, max_per_q=None, rel_gap=None,
           min_margin=None):
    """Return boolean keep-mask implementing the decision layer."""
    keep = prob >= thr
    if one_parent:
        keep &= one_parent_mask(p_idx, prob)
    if min_margin:
        # ambiguity rule: a pool record whose two best candidate parents are
        # nearly tied is more likely a wrong merge than a right one
        keep &= parent_margin(p_idx, prob) >= min_margin
    if rel_gap is not None:
        # drop candidates far below the best candidate of the same query
        df = pd.DataFrame({"q": q_idx, "s": np.where(keep, prob, -1.0)})
        best = df.groupby("q")["s"].transform("max").values
        keep &= prob >= best - rel_gap
    if max_per_q is not None:
        df = pd.DataFrame({"q": q_idx, "s": np.where(keep, prob, -1.0)})
        rank = df.groupby("q")["s"].rank(ascending=False, method="first").values
        keep &= rank <= max_per_q
    return keep


def macro_f05(q_idx, label, keep, n_q, n_gt_per_q):
    """Macro F0.5 over n_q entities given per-pair labels and keep mask."""
    n_pred = np.bincount(q_idx[keep], minlength=n_q)
    tp = np.bincount(q_idx[keep & (label == 1)], minlength=n_q)
    return f05_per_entity(tp, n_pred, n_gt_per_q).mean()


def tune(q_idx, p_idx, prob, label, n_q, n_gt_per_q, log=print):
    """Grid-search threshold and decision options; return best config + score."""
    best = None
    rows = []
    for one_parent in (True, False):
        for thr in np.round(np.arange(0.20, 0.96, 0.02), 2):
            keep = decide(q_idx, p_idx, prob, thr, one_parent=one_parent)
            f = macro_f05(q_idx, label, keep, n_q, n_gt_per_q)
            rows.append((one_parent, thr, f))
            if best is None or f > best[2]:
                best = (one_parent, thr, f)
    tab = pd.DataFrame(rows, columns=["one_parent", "thr", "f05"])
    for op in (True, False):
        sub = tab[tab.one_parent == op].sort_values("f05", ascending=False).head(3)
        log(f"  one_parent={op}: " + ", ".join(f"thr={r.thr:.2f} f05={r.f05:.4f}" for r in sub.itertuples()))
    op, thr, f = best
    # refine with a relative-gap rule around the best threshold
    for gap in (0.1, 0.2, 0.3, 0.4, 0.5):
        keep = decide(q_idx, p_idx, prob, thr, one_parent=op, rel_gap=gap)
        f2 = macro_f05(q_idx, label, keep, n_q, n_gt_per_q)
        log(f"  rel_gap={gap}: f05={f2:.4f}")
        if f2 > f + 1e-5:
            best = (op, thr, f2, gap)
    if len(best) == 3:
        best = best + (None,)
    op, thr, f, gap = best
    best_m = None
    for mm in (0.05, 0.1, 0.2, 0.3, 0.5):
        keep = decide(q_idx, p_idx, prob, thr, one_parent=op, rel_gap=gap, min_margin=mm)
        f2 = macro_f05(q_idx, label, keep, n_q, n_gt_per_q)
        log(f"  min_margin={mm}: f05={f2:.4f}")
        if f2 > f + 1e-5:
            f, best_m = f2, mm
    return {"one_parent": op, "thr": float(thr), "rel_gap": gap, "min_margin": best_m, "f05": float(f)}


def breakdown(q_idx, label, keep, n_q, n_gt_per_q, groups: dict, log=print):
    """Report macro F0.5 by sub-population (country, match-count bucket...)."""
    n_pred = np.bincount(q_idx[keep], minlength=n_q)
    tp = np.bincount(q_idx[keep & (label == 1)], minlength=n_q)
    f = f05_per_entity(tp, n_pred, n_gt_per_q)
    P = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), np.nan)
    R = np.where(n_gt_per_q > 0, tp / np.maximum(n_gt_per_q, 1), np.nan)
    log(f"  overall: f05={f.mean():.4f}  pair-precision={np.nansum(tp)/max(1,n_pred.sum()):.4f} "
        f"pair-recall={tp.sum()/max(1,n_gt_per_q.sum()):.4f}  "
        f"singletons correct={((n_gt_per_q==0)&(n_pred==0)).sum()}/{(n_gt_per_q==0).sum()}")
    for gname, garr in groups.items():
        for val in pd.unique(garr):
            m = garr == val
            log(f"  {gname}={val}: n={m.sum():,} f05={f[m].mean():.4f} "
                f"meanP={np.nanmean(P[m]) if np.any(~np.isnan(P[m])) else float('nan'):.3f} "
                f"meanR={np.nanmean(R[m]) if np.any(~np.isnan(R[m])) else float('nan'):.3f}")
    return f
