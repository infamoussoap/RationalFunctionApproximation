import numpy as np

from functools import partial

from .Approximator import Approximator
from .BernsteinLegendre import BernsteinLegendre

from .utils import check_x

from ..WriteToScreen import WriterToScreen
import warnings


class EGD(Approximator, BernsteinLegendre):
    """ Rational function approximation using Legendre polynomials on the numerator and Bernstein polynomials
        on the denominator. Here we iteratively change the Bernstein coefficients and the Legendre coefficients
    """
    def __init__(self, m, n=None, num_integration_points=100, tol=1e-10, spacing_type='linear'):
        Approximator.__init__(self)
        BernsteinLegendre.__init__(self, m, n=n,
                                   num_integration_points=num_integration_points, spacing_type=spacing_type)
        self.tol = tol
        self.x = None
        self.n_iter_ = None

        self.writer = WriterToScreen()

    def _update(self, x, d, step_size):
        w, c = self.w, self.legendre_coef
        dw_dt, dL_dc = d[:self.m + 1], d[self.m + 1:]

        w = w * np.exp(-step_size * dw_dt)
        w = w / np.sum(w)

        c = c - step_size * dL_dc

        return np.concatenate([w, c])

    def _search(self, target_function, c1=1e-4, c2=0.5, line_search_iter=100, step_size=1):
        f = partial(self.f, target_function)
        grad = f(self.x, grad=True)

        step_size = self.backtracking_armijo_line_search(f, self.x, grad, step_size,
                                                         c1=c1, c2=c2, max_iter=line_search_iter)

        return self._update(self.x, grad, step_size)

    def fit(self, target_function, max_iter=100, stopping_tol=1e-6, x=None,
            c1=1e-4, c2=0.5, line_search_iter=100, step_size=1, verbose=False):
        self.x = check_x(x, self.m + 1, self.n + 1)

        w_old = 1  # Needs to be large enough so the while loop starts
        self.n_iter_ = 0
        while self.n_iter_ < max_iter and np.linalg.norm(w_old - self.w) > stopping_tol:
            w_old = self.w.copy()

            self.x = self._search(target_function, c1=c1, c2=c2, line_search_iter=line_search_iter, step_size=step_size)

            self.n_iter_ += 1

            if verbose:
                self.writer.write(f"{self.n_iter_}: {self.f(target_function, self.x)}", header='\r')

        if verbose:
            print()

        if self.n_iter_ == max_iter:
            warnings.warn("Maximum number of iterations has been reached and convergence is not guaranteed. "
                          "Try increasing `max_iter` or increasing `stopping_tol`.")

        return self

    @property
    def w(self):
        return self.x[:self.m + 1]

    @property
    def legendre_coef(self):
        return self.x[self.m + 1:]