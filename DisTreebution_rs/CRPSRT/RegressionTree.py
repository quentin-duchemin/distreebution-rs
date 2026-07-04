import numpy as np
from distreebu_rs import RegressionTreeCRPS as _Rs

class RegressionTree:
    """Drop-in for DisTreebution.CRPSRT.RegressionTree.
    Note: limit_use_CRPS hybrid mode is not supported by the Rust backend;
    use the pure-Python RegressionTree for that.
    """
    def __init__(self, max_depth=None, min_samples_split=2, limit_use_CRPS=None,
                 quantiles=None, IG_biais_correction=None):
        if limit_use_CRPS is not None:
            raise NotImplementedError(
                "limit_use_CRPS not supported in the Rust backend. "
                "Use DisTreebution.CRPSRT.RegressionTree instead.")
        loo = IG_biais_correction in ("Mallows", "LOO")
        self._tree = _Rs(max_depth=max_depth, min_samples_split=min_samples_split, loo=loo)
        self.max_depth = max_depth; self.min_samples_split = min_samples_split
        self.IG_biais_correction = IG_biais_correction
    def fit(self, X, y, depth=0, ref_tree=None, max_depth_ref_tree=-1):
        self._tree.fit(np.asarray(X, dtype=float).tolist(), np.asarray(y, dtype=float).tolist())
    def get_values_leaf(self, X, indexes):
        r = self._tree.get_values_leaf(np.asarray(X, dtype=float).tolist(), list(np.asarray(indexes).astype(int)))
        return [[idxs, np.array(yv)] for idxs, yv in r]
