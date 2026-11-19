"""Evaluation: truth is joined to frozen predictions only here, after training.

Produces, per run: weekly confusion counts and rates, pooled (micro) metrics over the
whole evaluation window, mean-of-weekly metrics, and stratified metrics by source family,
length bin, and novelty (first exposure of a text vs. repeated exposure).
"""
from pathlib import Path
import numpy as np
import pandas as pd

THRESHOLD = 0.5


def confusion(y, s, thr=THRESHOLD):
    yp = (np.asarray(s) >= thr).astype(int); y = np.asarray(y).astype(int)
    return dict(TP=int(((yp==1)&(y==1)).sum()), FP=int(((yp==1)&(y==0)).sum()),
                TN=int(((yp==0)&(y==0)).sum()), FN=int(((yp==0)&(y==1)).sum()))


def rates(c):
    tp, fp, tn, fn = c['TP'], c['FP'], c['TN'], c['FN']
    recall = tp/(tp+fn) if tp+fn else np.nan
    fpr = fp/(fp+tn) if fp+tn else np.nan
    precision = tp/(tp+fp) if tp+fp else np.nan
    f1 = 2*precision*recall/(precision+recall) if (tp+fp and tp+fn and precision+recall) else (0.0 if tp+fn else np.nan)
    return dict(recall=recall, fpr=fpr, precision=precision, f1=f1, n=tp+fp+tn+fn, positives=tp+fn)


def _metrics(y, s):
    c = confusion(y, s); return {**c, **rates(c)}


def evaluate_run(pred, d, audits, result):
    """pred: row_id, week, score (frozen). d: frame with label/family/length_bin/novel_week."""
    p = pred.merge(d[['label','family','length_bin','novel_week','first_occurrence','missing_prompt']],
                   left_on='row_id', right_index=True, how='left')
    assert p.label.notna().all()
    name, seed = result['setting'], result['seed']
    # cumulative human labels per week from the acquisition audit
    aud = pd.DataFrame(audits)
    lab = {}
    if len(aud):
        for w, t in zip(aud.week, aud.truth_total): lab[int(w)] = int(t)
    weekly = []
    running = result['initial_labels']
    for w, g in p.groupby('week'):
        m = _metrics(g.label, g.score)
        weekly.append(dict(setting=name, seed=seed, week=int(w), **m,
                           initial_labels=result['initial_labels'],
                           human_labels_cum=running, added_labels_cum=running-result['initial_labels'],
                           updated=int(w in lab)))
        running = lab.get(int(w), running)   # update happens after evaluation of week w
    weekly = pd.DataFrame(weekly)
    pooled = _metrics(p.label, p.score)
    # Threshold-free supplements (pooled over weeks 5+): disentangle ranking quality from
    # the fixed 0.5 operating point, which is sensitive to class-prior drift.
    from sklearn.metrics import roc_auc_score, roc_curve
    fpr_c, tpr_c, _ = roc_curve(p.label, p.score)
    pooled['auroc'] = float(roc_auc_score(p.label, p.score))
    pooled['recall_at_fpr01'] = float(tpr_c[fpr_c <= .01].max())
    pooled['recall_at_fpr05'] = float(tpr_c[fpr_c <= .05].max())
    pooled['pred_positive_rate'] = float((p.score >= THRESHOLD).mean())
    mean_weekly = weekly[['recall','fpr','precision','f1']].mean().add_suffix('_weekmean').to_dict()
    strata = []
    for col in ['family','length_bin','novel_week','missing_prompt']:
        for v, g in p.groupby(col):
            strata.append(dict(setting=name, seed=seed, stratum=col, value=str(v), **_metrics(g.label, g.score)))
    # sensitivity: drop missing prompts entirely
    q = p[~p.missing_prompt]
    strata.append(dict(setting=name, seed=seed, stratum='exclude_missing', value='all', **_metrics(q.label, q.score)))
    summary = dict(**result, **pooled, **mean_weekly)
    return weekly, summary, pd.DataFrame(strata)


def write_outputs(out, weekly_all, summary_all, strata_all, audits_all):
    out = Path(out)/'metrics'; out.mkdir(parents=True, exist_ok=True)
    pd.concat(weekly_all, ignore_index=True).to_csv(out/'weekly_metrics.csv', index=False)
    s = pd.DataFrame(summary_all); s.to_csv(out/'run_summary.csv', index=False)
    pd.concat(strata_all, ignore_index=True).to_csv(out/'stratified_metrics.csv', index=False)
    pd.DataFrame(audits_all).to_csv(out/'update_audit.csv', index=False)
    # Seed aggregation: mean ± sd across seeds (seed variability only, NOT a temporal CI)
    cols = ['recall','fpr','precision','f1','recall_weekmean','fpr_weekmean','precision_weekmean','f1_weekmean',
            'auroc','recall_at_fpr01','recall_at_fpr05','pred_positive_rate',
            'post_initial_labels','human_labels_total']
    agg = s.groupby(['setting','method','period','budget','version','variant'])[cols].agg(['mean','std','count'])
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg.reset_index().to_csv(out/'setting_summary.csv', index=False)
    return s
