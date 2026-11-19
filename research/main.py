#!/usr/bin/env python3
"""JailbreaksOverTime continual-adaptation study: end-to-end orchestrator.

Stages: (1) data validation + QC, (2) frozen initial-only embedding, (3) prequential
runs for V1 (34 settings / 70 runs), V2 R1 (9 settings / 45 runs) and R1 ablations
(27 settings / 135 runs), (4) evaluation, (5) paired block-bootstrap statistics,
(6) figures. Every stage writes to <out>/ and can be re-run independently.

Usage:
  python main.py --data data/JailbreaksOverTime_train.csv --out outputs --workers 2
  python main.py --data ... --out outputs_char --embedding char --versions main   # robustness
"""
import argparse, json, sys, time
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import data as data_mod, experiment, evaluation, statistics, visualization


def _run(d, X, setting, seed, out):
    pred, result, audits = experiment.run_one(d, X, setting, seed, out)
    weekly, summary, strata = evaluation.evaluate_run(pred, d, audits, result)
    return weekly, summary, strata, audits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default=str(ROOT/'outputs'))
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--embedding', choices=['word','char'], default='word')
    ap.add_argument('--versions', choices=['all','main','V1','V2'], default='all',
                    help='all = V1+V2+ablations; main = V1+V2; V1 / V2 only')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--skip-runs', action='store_true', help='reuse metrics on disk; redo stats/figures')
    a = ap.parse_args()
    out = Path(a.out)
    for sub in ['qc','predictions','acquisition','metrics','statistics','figures']: (out/sub).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    d, qc = data_mod.load_data(a.data, out/'qc')
    X = data_mod.embed(d, out/'qc', kind=a.embedding)
    print(f'[data] rows={qc["rows"]} weeks={qc["calendar_weeks"]} initial={qc["initial_rows"]} embed={X.shape} ({time.time()-t0:.1f}s)')

    v1 = experiment.grid('V1') if a.versions in ('all','main','V1') else []
    v2 = experiment.grid('V2') if a.versions in ('all','main','V2') else []
    abl = experiment.ablation_grid() if a.versions == 'all' else []
    runs = experiment.runs_for(v1 + v2 + abl)
    print(f'[grid] V1={len(v1)} V2={len(v2)} ablation={len(abl)} settings -> {len(runs)} runs')

    if not a.skip_runs:
        t1 = time.time()
        # B4 (kNN) is the slow one; run it first so it overlaps with the many cheap LR runs.
        runs.sort(key=lambda r: 0 if r[0].method=='B4' else 1)
        res = Parallel(n_jobs=a.workers, verbose=5)(delayed(_run)(d, X, s, seed, out) for s, seed in runs)
        weekly_all = [r[0] for r in res]; summary_all = [r[1] for r in res]
        strata_all = [r[2] for r in res]; audits_all = [x for r in res for x in r[3]]
        evaluation.write_outputs(out, weekly_all, summary_all, strata_all, audits_all)
        print(f'[runs] {len(runs)} runs in {time.time()-t1:.1f}s')

    weekly = pd.read_csv(out/'metrics'/'weekly_metrics.csv')
    summary = pd.read_csv(out/'metrics'/'run_summary.csv')
    strata = pd.read_csv(out/'metrics'/'stratified_metrics.csv')
    summary['variant'] = summary['variant'].fillna('full')
    names = lambda ss: [s.name for s in ss]
    stats = statistics.run_all(weekly, names(v1), names(v2), names(abl), out, n_boot=a.n_boot)
    print(f'[stats] {len(stats)} comparisons written')
    visualization.make_all(out, weekly, summary, strata, stats)
    print(f'[figures] written to {out/"figures"}')
    manifest = dict(data=str(a.data), embedding=a.embedding, versions=a.versions, runs=len(runs),
                    input_sha256=qc['input_sha256'], wall_seconds=time.time()-t0,
                    settings=[s.name for s in v1+v2+abl])
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
