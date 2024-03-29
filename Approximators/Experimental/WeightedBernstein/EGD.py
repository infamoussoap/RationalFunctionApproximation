import numpy as np

from functools import partial

from .Bernstein import Bernstein
from .BernsteinApproximator import BernsteinApproximator

from ..validation_checks import check_bernstein_w, check_target_ys, check_X_in_range

from ..WriteToScreen import WriterToScreen

import warnings
from ..CustomWarnings import ConvergenceWarning


class EGD(BernsteinApproximator, Bernstein):
    """ Rational function approximation using Legendre polynomials on the numerator and Bernstein polynomials
        on the denominator. We iteratively change the Bernstein coefficients using an EGD scheme and
        the Legendre coefficients are found using a projection.

        Attributes
        ----------
        n : int
            The degree of the numerator
        m : int
            The degree of the denominator
        tol : float (ignored)
            Tolerance for the zero set
        w : (m + 1, ) np.ndarray
            The coefficients for the Bernstein polynomials (denominator)
        _legendre_coef : (n + 1, ) np.ndarray
            The coefficients for the Legendre polynomials (numerator)
        n_iter_ : int
            Number of iterations run by the coordinate descent solver to reach the specified tolerance
        _writer : WriterToScreen
            Used to write to screen for verbose
    """
    def __init__(self, n, m=None, tol=1e-10, max_iter=100, stopping_tol=1e-6, w=None,
                 c1=1e-4, c2=0.5, line_search_iter=100, max_step_size=10, verbose=False):
        """ Initialize EGD Optimizer

            Parameters
            ----------
            n : int
                The degree of the numerator
            m : int, default=None
                The degree of the denominator, if not given it will default to n
            tol : float, default=1e-10
                Tolerance for the zero set
            max_iter : int, default=100
                The number of iterations to perform the optimization
            stopping_tol : float, default=1e-6
                The tolerance for the stopping criteria. If |w_prev - w_new| < stopping_tol
                then the iteration will stop
            w : (m + 1, ) np.ndarray, default=None
                The starting point for optimization. If None is given then it will default
                to np.ones(m + 1) / (m + 1).
            c1 : float, default=1e-4
                Parameter for the armijo line search
            c2 : float, default=0.5
                Parameter for the armijo line search
            line_search_iter : int, default=100
                Number of iterations for the line search
            max_step_size : float, default=10
                Maximum candidate step size
            verbose : bool, default=False
                If set to true then the result of each step will be printed.
        """
        BernsteinApproximator.__init__(self)
        Bernstein.__init__(self, n, m=m)

        self.tol = tol

        self.max_iter = max_iter
        self.stopping_tol = stopping_tol

        self.c1 = c1
        self.c2 = c2
        self.line_search_iter = line_search_iter
        self.max_step_size = max_step_size

        self.verbose = verbose

        self.w_start = w
        self.w = None

        self._legendre_coef = None

        self.n_iter_ = None

        self._writer = WriterToScreen()

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
        z = x * np.exp(-step_size * d)
        return z / np.sum(z)

    def _search(self, X, target_ys):
        """ Perform a step using a line search

            Parameters
            ----------
            X : np.ndarray
            target_ys : list of np.ndarray
                The functions to be fitted against. Must be able to take np.ndarray

            Returns
            -------
            (m + 1, ) np.ndarray
                The new iteration point once the optimal step has been taken
        """
        f = partial(self.f, X, target_ys)

        d = f(self.w, grad=True)

        step_size = self.backtracking_armijo_line_search(f, self.w, d, self.max_step_size,
                                                         c1=self.c1, c2=self.c2, max_iter=self.line_search_iter)

        return self._update(self.w, d, step_size)

    def fit(self, X, y):
        """ Fit the rational polynomial coefficients to the target function

            Parameters
            ----------
            X : np.ndarray
            y : np.ndarray or list of np.ndarray
                The target values to be fitted against

            Returns
            -------
            self : object
                Fitted rational polynomial
        """
        check_X_in_range(X, 0, 1)

        self.w = check_bernstein_w(self.w_start, self.m + 1)
        target_ys = check_target_ys(y)

        if len(self.w) == 1:
            self._legendre_coef = [self._compute_legendre_coef(X, y) for y in target_ys]
            return self

        w_old = 1  # Needs to be large enough so the while loop starts
        self.n_iter_ = 0
        while self.n_iter_ < self.max_iter and np.linalg.norm(w_old - self.w) > self.stopping_tol:
            w_old = self.w.copy()

            self.w = self._search(X, target_ys)

            denominator = self.denominator(X)
            self._legendre_coef = [self._compute_legendre_coef(X, y * denominator) for y in target_ys]

            self.n_iter_ += 1

            if self.verbose:
                diff = self(X) - y
                self._writer.write_norms(self.n_iter_, diff, norms=[2, np.inf])

        if self.verbose:
            print()

        if self.n_iter_ == self.max_iter:
            warnings.warn("Maximum number of iterations has been reached and convergence is not guaranteed. "
                          "Try increasing `max_iter` or increasing `stopping_tol`.", category=ConvergenceWarning)

        return self
