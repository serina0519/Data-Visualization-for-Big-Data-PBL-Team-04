"""Prequential experiment with a label-access firewall and full acquisition ledger."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json, math, time
import numpy as np
import pandas as pd
from .models import LogisticDetector, CosineMemory


@dataclass(frozen=True)
class Setting:
    method: str
    period: int = 0
    budget: float = 0.0
    version: str = 'V1'
    variant: str = 'full'   # R1 ablations: full | no_novelty | no_decay | no_random
    @property
    def name(self):
        base = f'{self.method}_p{self.period}_b{int(round(100*self.budget)):02d}'
        return base if self.variant == 'full' else f'{base}_{self.variant}'


# Methods whose outcome depends on a random seed (random acquisition).
STOCHASTIC = {'C1', 'R1'}
SEEDS = [0, 1, 2, 3, 4]


def grid(versions='both'):
    """34 requested settings, 70 runs; 9 recommended settings, 45 runs;
    ablations (3 variants x 9 settings, 5 seeds) are returned separately."""
    settings = [Setting('A1')]
    settings += [Setting(m,p) for m in ['A2','B1','B2','B3','B4'] for p in [1,2,4]]
    settings += [Setting(m,p,b) for m in ['C1','C2'] for b in [.01,.05,.10] for p in [1,2,4]]
    assert len(settings)==34
    if versions in ('both', 'V2'):
        if versions == 'V2': settings = []
        settings += [Setting('R1',p,b,'V2') for b in [.01,.05,.10] for p in [1,2,4]]
    return settings


def ablation_grid():
    return [Setting('R1',p,b,'V2',v) for v in ['no_novelty','no_decay','no_random']
            for b in [.01,.05,.10] for p in [1,2,4]]


def runs_for(settings):
    """Expand settings into (setting, seed) runs: 5 seeds for stochastic methods, 1 otherwise."""
    return [(s, seed) for s in settings for seed in (SEEDS if s.method in STOCHASTIC else [0])]


class LabelOracle:
    """Training code obtains truth ONLY through reveal(), never evaluator arrays."""
    def __init__(self, truth, initial, method):
        self.__truth = np.asarray(truth)
        self.initial = set(map(int,initial)); self.seen = set(); self.method = method
        self.ledger = []
    def reveal(self, ids, week, reason):
        ids = np.asarray(ids,dtype=int)
        if self.method in ['A1','B1','B2','B3','B4']:
            assert set(ids).issubset(self.initial), 'Unsupervised label leak'
        for i in ids:
            if int(i) not in self.seen:
                self.ledger.append({'row_id':int(i),'reveal_week':int(week),'reason':reason})
                self.seen.add(int(i))
        return self.__truth[ids].copy()


ALLOCATION = {  # (random, uncertainty) fractions; remainder goes to novelty
    'full': (.2, .4), 'no_novelty': (.2, .8), 'no_decay': (.2, .4), 'no_random': (0., .5)}


def select_hybrid(pool, scores, X, known, n, rng, variant='full'):
    """R1: 20% random exploration, 40% uncertainty, remainder geometric novelty.
    All scores/features available before any current-cycle label is revealed.
    Disjoint allocations, no pseudo-labels; budget exactly matches C1/C2.
    Ablations: no_novelty -> 20/80/0; no_random -> 0/50/50; no_decay -> same
    selection as full but no time weighting (handled in run_one).
    """
    fr, fu = ALLOCATION[variant]
    n_random = int(np.floor(fr*n)); n_uncertain = int(np.floor(fu*n))
    if variant == 'no_novelty': n_uncertain = n - n_random  # remainder also to uncertainty
    random_ids = rng.choice(pool,size=n_random,replace=False)
    rest = pool[~np.isin(pool,random_ids)]
    lookup = dict(zip(pool,scores))
    u = np.array([abs(lookup[i]-.5) for i in rest])
    uncertain = rest[np.argsort(u,kind='stable')[:n_uncertain]]
    rest = rest[~np.isin(rest,uncertain)]
    novelty = []
    for s in range(0,len(rest),256):
        novelty.extend(1-(X[rest[s:s+256]]@X[known].T).max(axis=1))
    diverse = rest[np.argsort(-np.array(novelty),kind='stable')[:n-n_random-n_uncertain]]
    return np.r_[random_ids,uncertain,diverse].astype(int)


def run_one(d, X, setting, seed, out, tau=.9, k=10, half_life=8):
    """At week w: evaluate ALL w rows, THEN (on a complete boundary) update.
    Last incomplete cycle is evaluated but not used for a final off-schedule fit.
    """
    out=Path(out); name=setting.name; method=setting.method; period=setting.period
    initial = np.flatnonzero(d.week.values<=4)
    oracle=LabelOracle(d.label.values,initial,method)
    known=initial.copy(); known_y=oracle.reveal(initial,4,'initial')
    model=LogisticDetector(X.shape[1]).fit(X[initial],known_y)
    initial_fit=model.last_fit.copy()
    memory=CosineMemory(X,initial,known_y,k) if method=='B4' else None
    # B3 is strict one-cycle delayed replay. First replay buffer is the initial
    # model's own predictions on weeks 1-4, NOT their true labels.
    delayed_ids=initial.copy(); delayed_y=(model.predict(X[initial])>=.5).astype(int)
    pseudo_ids=np.array([],dtype=int); pseudo_y=np.array([],dtype=int)
    rng=np.random.default_rng(seed)
    pending=[]; predictions=[]; audits=[]
    started=time.time(); update_count=0
    for week in range(5,int(d.week.max())+1):
        ids=np.flatnonzero(d.week.values==week)
        if len(ids):
            p=memory.predict_ids(ids) if memory else model.predict(X[ids])
            predictions.append(pd.DataFrame({'row_id':ids,'week':week,'score':p}))
        pending.extend(ids.tolist())
        if method=='A1' or (week-4)%period: continue
        pool=np.array(pending,dtype=int); pending=[]
        if not len(pool): continue
        assert (d.week.values[pool]<=week).all()
        # The period model is frozen: these are the same predictions used at test time.
        pp=memory.predict_ids(pool) if memory else model.predict(X[pool])
        pseudo=(pp>=.5).astype(int)
        acquired=np.array([],dtype=int); trained=np.array([],dtype=int)
        accepted=0; fit={}; train_latest_week=None
        if method=='A2':
            acquired=pool
            new_y=oracle.reveal(acquired,week,'all_truth')
            known=np.r_[known,acquired]; known_y=np.r_[known_y,new_y]
            trained=known
            model.fit(X[trained],known_y)
        elif method in ['B1','B2']:
            mask=np.ones(len(pool),dtype=bool) if method=='B1' else np.maximum(pp,1-pp)>=tau
            accepted=int(mask.sum())
            pseudo_ids=np.r_[pseudo_ids,pool[mask]]; pseudo_y=np.r_[pseudo_y,pseudo[mask]]
            trained=np.r_[initial,pseudo_ids]
            model.fit(X[trained],np.r_[known_y,pseudo_y])
        elif method=='B3':
            trained=delayed_ids.copy(); accepted=len(delayed_ids)
            model.fit(X[trained],delayed_y,warm=True)
            delayed_ids,delayed_y=pool.copy(),pseudo.copy()
        elif method=='B4':
            memory.append(pool,pseudo); accepted=len(pool)
        elif method in ['C1','C2','R1']:
            # Floor is explicit: actual budgets <= nominal k%; no hidden minimum-1 rule.
            n=int(math.floor(setting.budget*len(pool)))
            if method=='C1': acquired=rng.choice(pool,size=n,replace=False)
            elif method=='C2': acquired=pool[np.argsort(np.abs(pp-.5),kind='stable')[:n]]
            else: acquired=select_hybrid(pool,pp,X,known,n,rng,setting.variant)
            assert len(np.unique(acquired))==n and set(acquired).issubset(pool)
            new_y=oracle.reveal(acquired,week,'selected_truth')
            known=np.r_[known,acquired]; known_y=np.r_[known_y,new_y]
            trained=known
            weights=None
            if method=='R1' and setting.variant!='no_decay':
                weights=np.exp2(-(week-d.week.values[known])/half_life)
                # Anchors are never down-weighted; later human labels age gradually.
                weights[np.isin(known,initial)]=1.0
            if n: model.fit(X[trained],known_y,weights=weights)
        else: raise ValueError(method)
        if len(trained):
            train_latest_week=int(d.week.values[trained].max())
            assert train_latest_week<=week
            fit=model.last_fit.copy()
        update_count+=1
        audits.append(dict(setting=name,seed=seed,week=week,pool_n=len(pool),
            truth_new=len(acquired),truth_total=len(oracle.seen),pseudo_accepted=accepted,
            train_n=len(trained),train_latest_week=train_latest_week,
            next_eligible_test_week=week+1,**fit))
    if method in ['B1','B2','B3','B4','A1']: assert oracle.seen==set(initial)
    pred=pd.concat(predictions,ignore_index=True)
    assert len(pred)==int((d.week>4).sum()) and pred.row_id.is_unique
    # Truth is appended only in evaluation, outside this training loop.
    pred.to_csv(out/'predictions'/f'{name}_s{seed}.csv.gz',index=False,compression='gzip')
    pd.DataFrame(oracle.ledger).to_csv(out/'acquisition'/f'{name}_s{seed}.csv.gz',index=False,compression='gzip')
    result=dict(**asdict(setting),setting=name,seed=seed,initial_labels=len(initial),
        post_initial_labels=len(oracle.seen)-len(initial),human_labels_total=len(oracle.seen),
        updates=update_count,wall_seconds=time.time()-started,
        initial_fit_converged=initial_fit['converged'])
    if memory:
        result.update(knn_zero_denominator=memory.zero_denominator,
            knn_negative_neighbor_queries=memory.negative_neighbor_queries,
            knn_out_of_range=memory.out_of_range)
    return pred, result, audits
