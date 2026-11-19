"""Publication-style figures: fixed palette, fixed axes, 300 DPI PNG + vector PDF."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
                     'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 110})
COLORS = {'A1': '#7f7f7f', 'A2': '#000000', 'B1': '#1f77b4', 'B2': '#17becf', 'B3': '#9467bd',
          'B4': '#8c564b', 'C1': '#ff7f0e', 'C2': '#d62728', 'R1': '#2ca02c'}
LABELS = {'A1': 'A1 non-adaptive', 'A2': 'A2 all labels', 'B1': 'B1 self-training', 'B2': 'B2 conf. pseudo-label',
          'B3': 'B3 delayed replay', 'B4': 'B4 kNN memory', 'C1': 'C1 random labels', 'C2': 'C2 uncertainty labels',
          'R1': 'R1 hybrid (proposed)'}


def _save(fig, out, name):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def fig1_data(qc_dir, out):
    weekly = pd.read_csv(Path(qc_dir)/'weekly_counts.csv')
    src = pd.read_csv(Path(qc_dir)/'weekly_source_counts.csv', index_col=0)
    fig, ax = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True)
    ax[0].bar(weekly.week, weekly.n, color='#4c72b0'); ax[0].set_ylabel('Prompts / week')
    ax[0].axvspan(0.5, 4.5, color='#dddddd', alpha=.6, label='Initial training (weeks 1-4)'); ax[0].legend(loc='upper right')
    ax[1].plot(weekly.week, weekly.positives/weekly.n, 'o-', color='#d62728', ms=3); ax[1].set_ylabel('Jailbreak fraction'); ax[1].set_ylim(0, .5)
    bottom = np.zeros(len(src))
    for col, c in zip(src.columns, ['#999999', '#d62728', '#ff7f0e']):
        ax[2].bar(src.index, src[col]/src.sum(1), bottom=bottom, color=c, label=col); bottom += (src[col]/src.sum(1)).values
    ax[2].set_ylabel('Source share'); ax[2].set_xlabel('Week (7-day bins from 2023-02-28 UTC)'); ax[2].legend(ncol=3, loc='lower right')
    ax[2].set_xlim(0.5, 43.5)
    fig.suptitle('Figure 1. Weekly sample size, jailbreak fraction and source composition')
    _save(fig, out, 'fig1_data_overview')


def fig2_weekly(weekly, settings, out, period=1, budget=.10):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, m in zip(axes.ravel(), ['recall', 'fpr', 'precision', 'f1']):
        for s in settings:
            meth = s.split('_')[0]
            g = weekly[weekly.setting==s].groupby('week')[m].mean()
            ax.plot(g.index, g.values, color=COLORS[meth], label=LABELS[meth], lw=1.3 if meth in ('R1','A1','A2') else 1)
        ax.set_title(m.upper() if m in ('fpr','f1') else m.capitalize()); ax.set_xlim(4.5, 43.5)
        ax.set_ylim(0, 1 if m!='fpr' else .25)
        ax.set_xlabel('Week') if ax in axes[1] else None
    axes[0,0].legend(ncol=2, loc='lower left')
    fig.suptitle(f'Figure 2. Weekly test-then-train metrics (period={period} wk, label budget={int(budget*100)}%, seed mean)')
    _save(fig, out, f'fig2_weekly_metrics_p{period}_b{int(budget*100):02d}')


def fig3_cost(summary, out):
    s = summary.groupby(['setting','method','period','budget'])[['recall','f1','fpr','post_initial_labels']].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, m in zip(axes, ['recall', 'f1']):
        for meth in ['A1','A2','B1','B2','B3','B4','C1','C2','R1']:
            g = s[s.method==meth]
            for i, (_, r) in enumerate(g.iterrows()):
                ax.scatter(r.post_initial_labels, r[m], color=COLORS[meth], s=28, zorder=3,
                           marker={1:'o',2:'s',4:'^',0:'D'}[int(r.period)], label=LABELS[meth] if i==0 else None)
        ax.set_xscale('symlog', linthresh=100); ax.set_xlabel('Additional human labels after week 4 (symlog)'); ax.set_ylabel(m.upper() if m=='f1' else 'Recall')
        ax.set_ylim(0.6, 1.0)
    axes[0].legend(fontsize=6.5, loc='lower right', ncol=2)
    fig.suptitle('Figure 3. Labeling cost vs. pooled performance (marker: o=1wk, s=2wk, ^=4wk, D=none)')
    _save(fig, out, 'fig3_labels_vs_performance')


def fig4_paired(stats, out, block=4):
    d = stats[(stats.block==block) & (stats.family=='vs_A1')]
    d = d[~d.setting.str.contains('no_')]
    fig, axes = plt.subplots(1, 3, figsize=(11, 7), sharey=True)
    order = sorted(d.setting.unique(), key=lambda s: (s.split('_')[0], s))
    y = np.arange(len(order))
    for ax, m in zip(axes, ['recall','fpr','f1']):
        g = d[d.metric==m].set_index('setting').loc[order]
        cols = [COLORS[s.split('_')[0]] for s in order]
        ax.errorbar(g['diff'], y, xerr=[g['diff']-g.ci_low, g.ci_high-g['diff']], fmt='o', ms=3.5, ecolor='#555555', elinewidth=.8, color='none')
        ax.scatter(g['diff'], y, c=cols, s=22, zorder=3)
        ax.axvline(0, color='k', lw=.8); ax.set_title(f'Δ {m.upper() if m!="recall" else "Recall"} vs A1 (95% block-bootstrap CI, block={block})')
        sig = g.p_holm < .05
        for yi, sg in zip(y, sig):
            if sg: ax.text(ax.get_xlim()[1]*0.98 if False else g['diff'].iloc[yi], yi+.35, '*', ha='center', fontsize=8)
    axes[0].set_yticks(y); axes[0].set_yticklabels(order, fontsize=6.5)
    fig.suptitle('Figure 4. Paired weekly differences against the non-adaptive baseline (* Holm-adjusted p<0.05)')
    fig.tight_layout()
    _save(fig, out, 'fig4_paired_differences_vs_A1')


def fig5_source(strata, out):
    """Per-source recall (jailbreak_chat / jailbreak_llms) and WildChat FPR.
    Note: every repeated text in weeks 5-43 is benign, so a novel-vs-repeat recall plot is
    degenerate; the exact-string novelty split is reported in stratified_metrics.csv instead."""
    s = strata[(strata.stratum=='family')].copy()
    s['method'] = s.setting.str.split('_').str[0]
    s = s[~s.setting.str.contains('no_')]
    g = s.groupby(['setting','method','value'])[['recall','fpr']].mean().reset_index()
    rec = g.pivot(index=['setting','method'], columns='value', values='recall').reset_index().sort_values(['method','setting'])
    fpr = g[g.value=='wildchat'].set_index('setting').loc[rec.setting, 'fpr'].values
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw=dict(height_ratios=[2,1]))
    x = np.arange(len(rec))
    ax[0].bar(x-.2, rec['jailbreak_chat'], .4, color='#d62728', label='Recall on JailbreakChat (n=2,720)')
    ax[0].bar(x+.2, rec['jailbreak_llms'], .4, color='#ff7f0e', label='Recall on JailbreakHub / jailbreak_llms (n=725)')
    ax[0].set_ylabel('Recall'); ax[0].set_ylim(0, 1); ax[0].legend(loc='lower left')
    ax[1].bar(x, fpr, .6, color='#999999'); ax[1].set_ylabel('FPR on WildChat'); ax[1].set_ylim(0, .08)
    ax[1].set_xticks(x); ax[1].set_xticklabels(rec.setting, rotation=90, fontsize=6)
    fig.suptitle('Figure 5. Per-source recall and benign-source FPR (weeks 5-43, seed mean)')
    _save(fig, out, 'fig5_per_source_metrics')


def fig6_ablation(summary, out):
    s = summary[summary.method=='R1'].groupby(['setting','variant','period','budget'])[['recall','fpr','f1']].agg(['mean','std']).reset_index()
    s.columns = ['_'.join(c).strip('_') for c in s.columns]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    variants = ['full','no_random','no_novelty','no_decay']; vc = ['#2ca02c','#ff7f0e','#1f77b4','#9467bd']
    keys = sorted(set(zip(s.period, s.budget)))
    x = np.arange(len(keys))
    for ax, m in zip(axes, ['recall','fpr','f1']):
        for i, (v, c) in enumerate(zip(variants, vc)):
            g = s[s.variant==v].set_index(['period','budget']).reindex(keys)
            ax.bar(x+(i-1.5)*.2, g[f'{m}_mean'], .2, yerr=g[f'{m}_std'], color=c, label=v, capsize=1.5, error_kw=dict(lw=.6))
        ax.set_xticks(x); ax.set_xticklabels([f'p{p}\nb{int(b*100)}%' for p,b in keys], fontsize=7); ax.set_title(m.upper() if m!='recall' else 'Recall')
        if m=='fpr': ax.set_ylim(0, .05)
        else: ax.set_ylim(0.7, 1.0)
    axes[0].legend(fontsize=7)
    fig.suptitle('Figure 6. R1 ablations (mean ± sd over 5 seeds; sd reflects seed variation only)')
    _save(fig, out, 'fig6_r1_ablations')


def make_all(out_root, weekly, summary, strata, stats):
    out = Path(out_root)/'figures'
    fig1_data(Path(out_root)/'qc', out)
    for p in [1,2,4]:
        sel = ['A1_p0_b00', f'A2_p{p}_b00', f'B1_p{p}_b00', f'B2_p{p}_b00', f'B3_p{p}_b00', f'B4_p{p}_b00',
               f'C1_p{p}_b10', f'C2_p{p}_b10', f'R1_p{p}_b10']
        fig2_weekly(weekly, [s for s in sel if s in set(weekly.setting)], out, period=p)
    fig3_cost(summary, out)
    fig4_paired(stats, out)
    fig5_source(strata, out)
    fig6_ablation(summary, out)
