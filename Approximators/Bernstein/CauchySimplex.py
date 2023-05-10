import numpy as np

from functools import partial

from .Bernstein import Bernstein
from .Approximator import Approximator

from .utils import check_w


class CauchySimplex(Approximator, Bernstein):
    """ Rational function approximation using Legendre polynomials on the numerator and Bernstein polynomials
        on the denominator. Here we only iteratively change the Bernstein coefficients and the Legendre coefficients
        are found using projection

        n_iter_
            Number of iterations run by the coordinate descent solver to reach the specified tolerance
    """
    def __init__(self, m, n=None, num_integration_points=100, tol=1e-10, spacing_type='linear'):
        """
            n is the degree of the numerator
            m is the degree of the denominator
        """
        Approximator.__init__(self)
        Bernstein.__init__(self, m, n=n,
                           num_integration_points=num_integration_points, spacing_type=spacing_type)
        self.tol = tol

        self.w = None
        self.legendre_coef = None

        self.n_iter_ = None

    def _update(self, x, d, step_size):
        z = x - step_size * d
        z[x < self.tol] = 0

        return z / np.sum(z)

    def _search(self, target_function, c1=1e-4, c2=0.5, line_search_iter=100, gamma=1):
        f = partial(self.f, target_function)

        grad = f(self.w, grad=True)
        d = self.w * (grad - grad @ self.w)

        max_step_size = self._max_step_size(self.w, grad, tol=self.tol) * gamma

        step_size = self.backtracking_armijo_line_search(f, self.w, d, max_step_size,
                                                         c1=c1, c2=c2, max_iter=line_search_iter)

        return self._update(self.w, d, step_size)

    def fit(self, target_function, max_iter=100, stopping_tol=1e-6, w=None,
            c1=1e-4, c2=0.5, line_search_iter=100, gamma=1, verbose=False):
        self.w = check_w(w, self.m + 1)

        w_old = 1  # Needs to be large enough so the while loop starts
        self.n_iter_ = 0
        while self.n_iter_ < max_iter and np.linalg.norm(w_old - self.w) > stopping_tol:
            w_old = self.w.copy()

            self.w = self._search(target_function, c1=c1, c2=c2, line_search_iter=line_search_iter, gamma=gamma)
            self.legendre_coef = self._legendre_coef(target_function, self.w)

            self.n_iter_ += 1

            if verbose:
                print(f"{self.n_iter_}: {self.f(target_function, self.w)}")

        return self

    @staticmethod
    def _max_step_size(x, grad, tol=1e-10):
        support = x > tol

        diff = np.max(grad[support]) - x @ grad
        return 1 / diff if diff > 1e-6 else 1e6
