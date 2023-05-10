import numpy as np

from .Optimizer import Optimizer
from .ArmijoSearch import ArmijoSearch
from .Bernstein import Bernstein


class EGD(Bernstein, ArmijoSearch, Optimizer):
    """ Rational function approximation using Legendre polynomials on the numerator and Bernstein polynomials
        on the denominator. Here we only iteratively change the Bernstein coefficients and the Legendre coefficients
        are found using projection
    """
    def __init__(self, target_function, m, n=None, num_integration_points=100, tol=1e-10, spacing_type='linear'):
        """ m is the degree of the denominator
            n is the degree of the numerator
        """
        Bernstein.__init__(self, target_function, m, n=n,
                           num_integration_points=num_integration_points, spacing_type=spacing_type)
        self.tol = tol

    def update(self, x, d, step_size):
        z = x * np.exp(-step_size * d)
        return z / np.sum(z)

    def search(self, x, step_size=1, c1=1e-4, c2=0.5, max_iter=100):
        d = self.f(x, grad=True)
        step_size = self.backtracking_armijo_line_search(x, d, step_size,
                                                         c1=c1, c2=c2, max_iter=max_iter)

        return self.update(x, d, step_size)