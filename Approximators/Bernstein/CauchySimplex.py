import numpy as np

from .Optimizer import Optimizer
from .ArmijoSearch import ArmijoSearch
from .Bernstein import Bernstein


class CauchySimplex(Bernstein, ArmijoSearch, Optimizer):
    """ Rational function approximation using Legendre polynomials on the numerator and Bernstein polynomials
        on the denominator. Here we only iteratively change the Bernstein coefficients and the Legendre coefficients
        are found using projection
    """
    def __init__(self, target_function, m, n=None, num_integration_points=100, tol=1e-10):
        """ m is the degree of the denominator
            n is the degree of the numerator
        """
        Bernstein.__init__(self, target_function, m, n=n,
                           num_integration_points=num_integration_points)
        self.tol = tol

    def update(self, x, d, step_size):
        z = x - step_size * d
        z[x < self.tol] = 0

        return z / np.sum(z)

    def search(self, x, c1=1e-4, c2=0.5, max_iter=100, gamma=1):
        grad = self.f(x, grad=True)
        d = x * (grad - grad @ x)

        max_step_size = self.max_step_size(x, grad, tol=self.tol) * gamma

        step_size = self.backtracking_armijo_line_search(x, d, max_step_size,
                                                         c1=c1, c2=c2, max_iter=max_iter)

        return self.update(x, d, step_size)

    @staticmethod
    def max_step_size(x, grad, tol=1e-10):
        support = x > tol

        diff = np.max(grad[support]) - x @ grad
        return 1 / diff if diff > 1e-6 else 1e6
