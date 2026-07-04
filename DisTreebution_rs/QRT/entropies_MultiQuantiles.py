import numpy as np
from distreebu_rs import entropies_multi_quantiles as _rs

def entropies_MultiQuantiles(order, y, quantiles, IG_biais_correction=None):
    loo = IG_biais_correction in ("Mallows", "LOO")
    return np.array(_rs(list(order), list(y), list(quantiles), loo))
