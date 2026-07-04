import numpy as np
from distreebu_rs import entropies_crps as _rs

def entropies_CRPS(order, y, IG_biais_correction=None):
    loo = IG_biais_correction in ("Mallows", "LOO")
    return np.array(_rs(list(order), list(y), loo))
