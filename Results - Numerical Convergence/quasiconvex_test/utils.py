import sys
sys.path.insert(0, '../..')

import Approximators
import Approximators.Bernstein.CauchySimplex as Bernstein
from Approximators import LinProgApproximator


def get_approximators(n):
    return {'Bernstein': Bernstein(n, n, max_iter=1000, stopping_tol=0, gamma=0.9),
            'Polynomial': Bernstein(2 * n, 0),
            'Quasiconvex': LinProgApproximator(n, n, stopping_tol=1e-13, denominator_lb=0.1, denominator_ub=50)}


