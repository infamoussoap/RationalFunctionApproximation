import numpy as np

from .ArmijoSearch import ArmijoSearch
from .Optimizer import Optimizer
from .BernsteinLegendre import BernsteinLegendre


class CauchySimplex(BernsteinLegendre, ArmijoSearch, Optimizer):
    """ Rational function approximation using Legendre polynomials on the numerator and Bernstein polynomials
        on the denominator. Here we iteratively change the Bernstein coefficients and the Legendre coefficients
    """
    def __init__(self, target_function, m, n=None, num_integration_points=100, tol=1e-10):
        BernsteinLegendre.__init__(self, target_function, m, n=n,
                                   num_integration_points=num_integration_points)
        self.tol = tol

    def update(self, x, d, step_size):
        w, c = x[:self.m + 1], x[self.m + 1:]
        dw_dt, dL_dc = d[:self.m + 1], d[self.m + 1:]

        w = w - step_size * dw_dt
        w[w < self.tol] = 0
        w = w / np.sum(w)

        c = c - step_size * dL_dc

        return np.concatenate([w, c])

    def search(self, x, c1=1e-4, c2=0.5, max_iter=100, gamma=1):
        grad = self.f(x, grad=True)

        w = x[:self.m + 1]
        dL_dw = grad[:self.m + 1]

        max_step_size = self.max_step_size(w, dL_dw, tol=self.tol) * gamma

        grad[:self.m + 1] = w * (dL_dw - w @ dL_dw)
        step_size = self.backtracking_armijo_line_search(x, grad, max_step_size,
                                                         c1=c1, c2=c2, max_iter=max_iter)

        return self.update(x, grad, step_size)

    @staticmethod
    def max_step_size(w, dL_dw, tol=1e-10):
        support = w > tol

        diff = np.max(dL_dw[support]) - w @ dL_dw
        return 1 / diff if diff > 1e-6 else 1e6
