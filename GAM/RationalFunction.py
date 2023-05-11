import numpy as np

from functools import partial

from .Approximator import Approximator
from .Bernstein import Bernstein

from .utils import check_bernstein_w

import warnings


class Zeros:
    def __call__(self, x):
        return np.zeros_like(x)


class RationalFunction(Approximator, Bernstein):
    def __init__(self, n, m, evaluation_points, tol=1e-10):
        Approximator.__init__(self)
        Bernstein.__init__(self, n, m, evaluation_points)

        self.tol = tol

        self.w = None
        self.legendre_coef = None

        self.n_iter_ = None

        self.intercept_ = 0

    def _update(self, x, d, step_size):
        """ Perform a step in the update direction

            Parameters
            ----------
            x : (m + 1, ) np.ndarray
                The starting point
            d : (m + 1, ) np.ndarray
                The descent direction
            step_size : float
                The step size to be taken

            Returns
            -------
            (m + 1, ) np.ndarray
                The point once the step has been taken
        """
        z = x - step_size * d
        z[x < self.tol] = 0

        return z / np.sum(z)

    def _search(self, y, c1=1e-4, c2=0.5, line_search_iter=100, gamma=0.9):
        """ Perform a step using a line search

            Parameters
            ----------
            y : (k, ) np.ndarray
                Target values
            c1 : float, default=1e-4
                Parameter for the armijo line search
            c2 : float, default=0.5
                Parameter for the armijo line search
            line_search_iter : int, default=100
                Number of iterations for the line search
            gamma : float, default=0.9
                Expected to be a float between [0, 1]. It represents the percent of the maximum step size
                to be taken.

            Returns
            -------
            (m + 1, ) np.ndarray
                The new iteration point once the optimal step has been taken
        """
        f = partial(self.f, y)

        grad = f(self.w, grad=True)
        d = self.w * (grad - grad @ self.w)

        max_step_size = self._max_step_size(self.w, grad, tol=self.tol) * gamma

        step_size = self.backtracking_armijo_line_search(f, self.w, d, max_step_size,
                                                         c1=c1, c2=c2, max_iter=line_search_iter)

        return self._update(self.w, d, step_size)

    def fit(self, y, max_iter=100, stopping_tol=1e-6, w=None,
            c1=1e-4, c2=0.5, line_search_iter=100, gamma=0.9):
        """ Fits a rational polynomial to the given datapoints

            Parameters
            ----------
            y : (k, ) np.ndarray
                Target values to be fitted against
            max_iter : int, default=100
                The number of iterations to perform the optimization
            stopping_tol : float, default=1e-6
                The tolerance for the stopping criteria. If |w_prev - w_new| < stopping_tol
                then the iteration will stop
            w : (m + 1, ) np.ndarray, default=None
                The starting point for optimization. If None is given then it will default
                to (np.ones(m + 1) / (m + 1).
            c1 : float, default=1e-4
                Parameter for the armijo line search
            c2 : float, default=0.5
                Parameter for the armijo line search
            line_search_iter : int, default=100
                Number of iterations for the line search
            gamma : float, default=0.9
                Expected to be a float between [0, 1]. It represents the percent of the maximum step size
                to be taken

            Returns
            -------
            self : object
                Fitted rational polynomial
        """
        self.w = check_bernstein_w(w, self.m + 1)

        w_old = 1  # Needs to be large enough so the while loop starts
        self.n_iter_ = 0
        while self.n_iter_ < max_iter and np.linalg.norm(w_old - self.w) > stopping_tol:
            w_old = self.w.copy()

            self.w = self._search(y, c1=c1, c2=c2, line_search_iter=line_search_iter, gamma=gamma)
            self.legendre_coef = self._legendre_coef(y, self.w)

            self.n_iter_ += 1

        self.intercept_ = -np.mean(self(self.evaluation_points))

        if self.n_iter_ == max_iter:
            warnings.warn("Maximum number of iterations has been reached and convergence is not guaranteed. "
                          "Try increasing `max_iter` or increasing `stopping_tol`.")

        return self

    @staticmethod
    def _max_step_size(x, grad, tol=1e-10):
        """ Compute the maximum step size

            Parameters
            ----------
            x : (m + 1, ) np.ndarray
                A point in the probability simplex
            grad : (m + 1, ) np.ndarray
                Gradient at the point `x`
            tol : float, default=1e-10
                Tolerance for the zero set

            Returns
            -------
            float
                The maximum step size
        """
        support = x > tol

        diff = np.max(grad[support]) - x @ grad
        return 1 / diff if diff > 1e-6 else 1e6

    def __call__(self, evaluation_points):
        if self.w is None:
            return np.zeros_like(evaluation_points)

        return self.numerator(evaluation_points) / self.denominator(evaluation_points) + self.intercept_
