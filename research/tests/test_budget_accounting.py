"""R1 allocation and C1/C2 budgets: disjoint, exact, and within the current pool."""
import sys, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import experiment


def test_hybrid_allocation_disjoint_and_exact():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 8)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    pool = np.arange(100, 500); known = np.arange(0, 100)
    scores = rng.uniform(size=len(pool))
    for variant in ['full','no_novelty','no_random','no_decay']:
        sel = experiment.select_hybrid(pool, scores, X, known, 40, rng, variant)
        assert len(sel) == 40 and len(np.unique(sel)) == 40 and set(sel) <= set(pool)
    sel = experiment.select_hybrid(pool, scores, X, known, 40, rng, 'full')
    # 20% random (8), 40% uncertainty (16), 40% novelty (16): the uncertainty slice is the 16
    # closest-to-0.5 scores among the non-random remainder.
    rand, unc = sel[:8], sel[8:24]
    lookup = dict(zip(pool, scores))
    rest = np.array([i for i in pool if i not in set(rand)])
    expected = rest[np.argsort(np.abs(np.array([lookup[i] for i in rest])-.5), kind='stable')[:16]]
    assert set(unc) == set(expected)


def test_holm_monotone():
    from src.statistics import holm
    p = np.array([.01, .04, .03, .5])
    adj = holm(p)
    assert np.all(adj >= p) and np.all(adj <= 1)
    assert adj[0] == pytest.approx(.04)
