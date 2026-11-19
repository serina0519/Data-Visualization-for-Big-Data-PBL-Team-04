"""Paired, block-bootstrap comparisons of weekly metrics between methods.

Prompts are not independent (repeated templates, temporal autocorrelation), so we:
 1. pair methods by evaluation week (same prompts, same week),
 2. resample contiguous blocks of weeks (circular moving-block bootstrap),
 3. report differences in pooled Recall / FPR / F1 with percentile 95% CIs,
 4. give a two-sided bootstrap p-value and Holm-correct within each comparison family,
 5. repeat for several block lengths as a sensitivity analysis.
Seed variability (C1/R1) is handled by averaging weekly confusion counts over seeds
before bootstrapping, and is reported separately in setting_summary.csv.
"""
import numpy as np
import pandas as pd

METRICS = ['recall', 'fpr', 'f1']


def _rates_from_counts(c):
    tp, fp, tn, fn = c[...,0], c[...,1], c[...,2], c[...,3]
    with np.errstate(divide='ignore', invalid='ignore'):
        recall = tp/(tp+fn); fpr = fp/(fp+tn); prec = tp/(tp+fp)
        f1 = np.where(prec+recall>0, 2*prec*recall/(prec+recall), 0.0)
    return dict(recall=recall, fpr=fpr, f1=f1)


def weekly_counts(weekly, setting):
    """Mean over seeds of weekly TP/FP/TN/FN -> array (weeks, 4)."""
    g = weekly[weekly.setting==setting].groupby('week')[['TP','FP','TN','FN']].mean().sort_index()
    return g.index.values, g.values


def block_bootstrap_diff(a, b, block=4, n_boot=2000, seed=12345):
    """a, b: (weeks,4) counts aligned by week. Returns dict of point diff, CI, p per metric."""
    assert a.shape == b.shape
    W = len(a); rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(W/block))
    starts = rng.integers(0, W, size=(n_boot, n_blocks))
    idx = (starts[:,:,None] + np.arange(block)[None,None,:]) % W   # circular blocks
    idx = idx.reshape(n_boot, -1)[:, :W]
    A = a[idx].sum(axis=1); B = b[idx].sum(axis=1)   # (n_boot,4) pooled counts
    ra, rb = _rates_from_counts(A), _rates_from_counts(B)
    pa, pb = _rates_from_counts(a.sum(0)[None]), _rates_from_counts(b.sum(0)[None])
    out = {}
    for m in METRICS:
        diff = ra[m]-rb[m]; point = float(pa[m][0]-pb[m][0])
        lo, hi = np.nanpercentile(diff, [2.5, 97.5])
        # two-sided p: fraction of bootstrap differences on the other side of 0 (doubled, capped)
        p = 2*min((diff<=0).mean(), (diff>=0).mean()); p = min(1.0, max(p, 1/n_boot))
        out[m] = dict(diff=point, ci_low=float(lo), ci_high=float(hi), p=float(p),
                      a=float(pa[m][0]), b=float(pb[m][0]))
    return out


def holm(pvals):
    p = np.asarray(pvals, dtype=float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n-rank)*p[i]); adj[i] = min(1.0, running)
    return adj


def compare(weekly, targets, reference, family, block=4, n_boot=2000):
    rows = []
    wr, cr = weekly_counts(weekly, reference)
    for t in targets:
        if t == reference: continue
        wt, ct = weekly_counts(weekly, t)
        assert np.array_equal(wr, wt)
        res = block_bootstrap_diff(ct, cr, block=block, n_boot=n_boot)
        for m, r in res.items():
            rows.append(dict(family=family, reference=reference, setting=t, metric=m, block=block, **r))
    df = pd.DataFrame(rows)
    if len(df):
        df['p_holm'] = np.nan
        for m, g in df.groupby('metric'):
            df.loc[g.index, 'p_holm'] = holm(g.p.values)
    return df


def run_all(weekly, settings_v1, settings_v2, settings_abl, out, blocks=(2,4,8), n_boot=2000):
    from pathlib import Path
    out = Path(out)/'statistics'; out.mkdir(parents=True, exist_ok=True)
    all_main = [s for s in settings_v1+settings_v2]
    frames = []
    for b in blocks:
        frames.append(compare(weekly, all_main, 'A1_p0_b00', 'vs_A1', b, n_boot))
        for p in [1,2,4]:
            same = [s for s in all_main if s.endswith(f'_p{p}_b00') or f'_p{p}_b' in s]
            frames.append(compare(weekly, same, f'A2_p{p}_b00', f'vs_A2_p{p}', b, n_boot))
        # R1 vs C1 / C2 at identical period & budget
        for s in settings_v2:
            tail = s.split('R1')[1]
            for c in ['C1','C2']:
                frames.append(compare(weekly, [s], c+tail, f'R1_vs_{c}', b, n_boot))
        # ablations vs full R1
        for s in settings_abl:
            full = '_'.join(s.split('_')[:3])   # R1_p1_b10_no_novelty -> R1_p1_b10
            frames.append(compare(weekly, [s], full, 'ablation_vs_R1', b, n_boot))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out/'paired_block_bootstrap.csv', index=False)
    df[df.block==4].to_csv(out/'paired_block_bootstrap_block4.csv', index=False)
    return df
