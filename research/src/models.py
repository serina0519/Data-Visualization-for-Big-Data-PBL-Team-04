"""Identical convex logistic objective for all LR-based adaptation rules."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class LogisticDetector:
    """Mean weighted log loss + lambda/2 ||w||^2; intercept not penalised.
    Own implementation supports a single-class pseudo-labelled cycle, unlike
    sklearn LogisticRegression. B3 changes only data and initialization, not loss.
    """
    def __init__(self, dim, l2=.001, maxiter=200):
        self.theta = np.zeros(dim+1, dtype=float)
        self.l2, self.maxiter = l2, maxiter
        self.last_fit = {}

    def fit(self, X, y, weights=None, warm=False):
        if len(y)==0: return self
        y = np.asarray(y, dtype=float)
        weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
        weights = weights/weights.sum()
        def objective(t):
            z = X@t[:-1]+t[-1]
            residual = weights*(expit(z)-y)
            value = np.dot(weights, np.logaddexp(0,z)-y*z) + self.l2/2*np.dot(t[:-1],t[:-1])
            grad = np.r_[X.T@residual+self.l2*t[:-1], residual.sum()]
            return value, grad
        start = self.theta.copy() if warm else np.zeros_like(self.theta)
        result = minimize(objective, start, jac=True, method='L-BFGS-B',
            options={'maxiter':self.maxiter, 'ftol':1e-10, 'gtol':1e-6})
        self.theta = result.x
        self.last_fit = dict(converged=bool(result.success), iterations=int(result.nit),
            objective=float(result.fun), gradient_max=float(np.max(np.abs(result.jac))),
            status=str(result.message))
        return self

    def predict(self, X): return expit(X@self.theta[:-1]+self.theta[-1])


class CosineMemory:
    """Exact cosine-weighted kNN, no logistic retraining (explicit B4 exception).
    Uses the user's raw similarity ratio. Zero denominator falls back to unweighted
    neighbour votes. Negative similarities and out-of-range scores are audited.
    """
    def __init__(self, X, ids, labels, k=10):
        self.X, self.ids, self.labels = X, np.array(ids), np.array(labels)
        self.k = k
        self.zero_denominator = self.negative_neighbor_queries = self.out_of_range = 0

    def predict_ids(self, query):
        scores = []
        for start in range(0,len(query),256):
            sim = self.X[query[start:start+256]]@self.X[self.ids].T
            k = min(self.k,len(self.ids))
            # Stable sorting gives deterministic index-order tie breaking.
            order = np.argsort(-sim, axis=1, kind='stable')[:,:k]
            vals = np.take_along_axis(sim,order,axis=1)
            labs = self.labels[order]
            den = vals.sum(axis=1)
            ok = np.abs(den)>1e-12
            p = np.divide((vals*labs).sum(axis=1), den, out=labs.mean(axis=1), where=ok)
            self.zero_denominator += int((~ok).sum())
            self.negative_neighbor_queries += int((vals<0).any(axis=1).sum())
            self.out_of_range += int(((p<0)|(p>1)).sum())
            scores.append(p)
        return np.concatenate(scores) if scores else np.array([])

    def append(self, ids, labels):
        self.ids = np.r_[self.ids,ids]
        self.labels = np.r_[self.labels,labels]
