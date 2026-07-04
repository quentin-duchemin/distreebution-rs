import numpy as np
from distreebu_rs import entropies_quadratic as _rs

def entropies_Quadratic(order, y, IG_biais_correction=None):
    loo = IG_biais_correction in ("Mallows", "LOO")
    return np.array(_rs(list(order), list(y), loo))
