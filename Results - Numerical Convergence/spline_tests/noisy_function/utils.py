import sys
sys.path.insert(0, '../../..')

import Approximators
import Approximators.MultivariateBernstein.CauchySimplex as MultivariateBernstein

import numpy as np
import pygam


def get_approximators(n):
    bernstein_df = 2 * (n + 1) * (n + 1) - 1
    n_splines = int(np.ceil(np.sqrt(bernstein_df)))

    return {'Bernstein': MultivariateBernstein([n, n], [n, n], max_iter=100, stopping_tol=1e-8, gamma=0.9, 
                                               early_stopping=False, train_proportion=0.85),
            'Spline': pygam.LinearGAM(pygam.te(0, 1, n_splines=n_splines, lam=0))}

