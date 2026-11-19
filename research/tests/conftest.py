import numpy as np, pandas as pd


def toy(weeks=12, per_week=60, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    n = weeks*per_week
    week = np.repeat(np.arange(1, weeks+1), per_week)
    label = rng.integers(0, 2, n)
    X = rng.normal(size=(n, dim)) + 1.5*label[:, None]
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    d = pd.DataFrame(dict(week=week, label=label, uid=np.arange(n)))
    return d, X
