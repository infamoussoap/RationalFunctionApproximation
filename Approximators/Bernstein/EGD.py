import numpy as np

from functools import partial

from .Optimizer import Optimizer
from .ArmijoSearch import ArmijoSearch
from .Bernstein import Bernstein
from .Approximator import Approximator

from .utils import check_w


class EGD(Approximator, Bernstein, ArmijoSearch, Optimizer):
    """ Rational function approximation using Legendre polynomials on the numerator and Bernstein polynomials
        on the denominator. Here we only iteratively change the Bernstein coefficients and the Legendre coefficients
        are found using projection
    """
    def __init__(self, m, n=None, num_integration_points=100, tol=1e-10, spacing_type='linear'):
        """ m is the degree of the denominator
            n is the degree of the numerator
        """
        Approximator.__init__(self)
        Bernstein.__init__(self, m, n=n,
                           num_integration_points=num_integration_points, spacing_type=spacing_type)
        self.tol = tol

        self.w = None
        self.legendre_coef = None

        self.n_iter_ = None

    def update(self, x, d, step_size):
        z = x * np.exp(-step_size * d)
        return z / np.sum(z)

    def search(self, target_function, c1=1e-4, c2=0.5, line_search_iter=100, step_size=1):
        f = partial(self.f, target_function)

        d = f(self.w, grad=True)
        
        step_size = self.backtracking_armijo_line_search(f, self.w, d, step_size,
                                                         c1=c1, c2=c2, max_iter=line_search_iter)

        return self.update(self.w, d, step_size)

    def fit(self, target_function, max_iter=100, stopping_tol=1e-6, w=None,
            c1=1e-4, c2=0.5, line_search_iter=100, step_size=1, verbose=False):
        self.w = check_w(w, self.m + 1)

        w_old = 1  # Needs to be large enough so the while loop starts
        self.n_iter_ = 0
        while self.n_iter_ < max_iter and np.linalg.norm(w_old - self.w) > stopping_tol:
            w_old = self.w.copy()

            self.w = self.search(target_function, c1=c1, c2=c2, line_search_iter=line_search_iter, step_size=step_size)
            self.legendre_coef = self._legendre_coef(target_function, self.w)

            self.n_iter_ += 1

            if verbose:
                print(f"{self.n_iter_}: {self.f(target_function, self.w)}")

        return self
