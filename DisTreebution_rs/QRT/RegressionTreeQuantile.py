import numpy as np
from distreebu_rs import RegressionTreeQuantile as _Rs

class RegressionTreeQuantile:
    def __init__(self, quantiles, max_depth=None, min_samples_split=2, IG_biais_correction=None):
        loo = IG_biais_correction in ("Mallows", "LOO")
        self._tree = _Rs(quantiles=list(quantiles), max_depth=max_depth,
                         min_samples_split=min_samples_split, loo=loo)
        self.quantiles = quantiles; self.max_depth = max_depth
        self.min_samples_split = min_samples_split; self.IG_biais_correction = IG_biais_correction
    def fit(self, X, y, depth=0, ref_tree=None, max_depth_ref_tree=-1):
        self._tree.fit(np.asarray(X, dtype=float).tolist(), np.asarray(y, dtype=float).tolist())
    def get_values_leaf(self, X, indexes):
        r = self._tree.get_values_leaf(np.asarray(X, dtype=float).tolist(), list(np.asarray(indexes).astype(int)))
        return [[idxs, np.array(yv)] for idxs, yv in r]
    def save(self, path):
        self._tree.save(path)
    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        obj._tree = _Rs.load(path)
        obj.quantiles = None; obj.max_depth = None
        obj.min_samples_split = None; obj.IG_biais_correction = None
        return obj
    def to_json(self):
        return self._tree.to_json()
    @classmethod
    def from_json(cls, data):
        obj = cls.__new__(cls)
        obj._tree = _Rs.from_json(data)
        obj.quantiles = None; obj.max_depth = None
        obj.min_samples_split = None; obj.IG_biais_correction = None
        return obj
