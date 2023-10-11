import sys
sys.path.insert(0, '../../..')

import Approximators
import Approximators.Bernstein.CauchySimplex as Bernstein
from Approximators import AAAApproximator


def get_approximators(n):
    return {'Bernstein': Bernstein(n, n, max_iter=1000, stopping_tol=0, gamma=0.9),
            'Polynomial': Bernstein(2 * n, 0),
            'AAA': AAAApproximator(n + 1, n + 1, cleanup=True, cleanup_tol=1e-13)}


