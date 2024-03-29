import numpy as np
from sklearn.model_selection import ParameterGrid, train_test_split

from functools import partial

from .BernsteinApproximator import BernsteinApproximator

from Approximators.Polynomials import LegendrePolynomial

from Approximators.validation_checks import check_bernstein_w, check_target_ys, check_X_in_range

from Approximators.WriteToScreen import WriterToScreen

import warnings
from Approximators.CustomWarnings import ConvergenceWarning


class CauchySimplex(BernsteinApproximator):
    """ Rational function approximation using Legendre polynomials on the numerator and Bernstein polynomials
        on the denominator. We iteratively change the Bernstein coefficients using a Cauchy-Simplex scheme and
        the Legendre coefficients are found using a projection.

        Attributes
        ----------
        n : int
            The degree of the numerator
        m : int
            The degree of the denominator
        tol : float
            Tolerance for the zero set
        w : (m + 1, ) np.ndarray
            The coefficients for the Bernstein polynomials (denominator)
        _legendre_coef : (n + 1, ) np.ndarray
            The coefficients for the Legendre polynomials (numerator)
        outer_n_iter_ : int
            Number of iterations run by the coordinate descent solver to reach the specified tolerance
        _writer : WriterToScreen
            Used to write to screen for verbose
    """
    def __init__(self, n, m=None, tol=1e-10, outer_max_iter=50, inner_max_iter=50, stopping_tol=1e-6, w=None,
                 c1=1e-4, c2=0.5, line_search_iter=100, gamma=1, verbose=False,
                 numerator_smoothing_penalty=None):
        """ Initialize Cauchy Simplex Optimizer

            Parameters
            ----------
            n : int
                The degree of the numerator
            m : int, default=None
                The degree of the denominator, if not given it will default to n
            tol : float, default=1e-10
                Tolerance for the zero set
            outer_max_iter : int, default=100
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
            gamma : float, default=1
                Expected to be a float between [0, 1]. It represents the percent of the maximum step size
                to be taken.
            verbose : bool, default=False
                If set to true then the result of each step will be printed.
            numerator_smoothing_penalty : float, default=None
                Degree of smoothing to apply. If None then no smoothing is applied
        """
        self.m_start = m
        BernsteinApproximator.__init__(self, n, m=m, numerator_smoothing_penalty=numerator_smoothing_penalty)

        self.tol = tol

        self.outer_max_iter = outer_max_iter
        self.inner_max_iter = inner_max_iter
        self.stopping_tol = stopping_tol

        self.c1 = c1
        self.c2 = c2
        self.line_search_iter = line_search_iter
        self.gamma = gamma

        self.verbose = verbose

        self.w_start = w
        self.w = None

        self._legendre_coef = None

        self.outer_n_iter_ = None

        self._writer = WriterToScreen()

    def get_params(self):
        return {
            'n': self.n,
            'm': self.m_start,
            'tol': self.tol,
            'outer_max_iter': self.outer_max_iter,
            'inner_max_iter': self.inner_max_iter,
            'stopping_tol': self.stopping_tol,
            'w': self.w_start,
            'c1': self.c1,
            'c2': self.c2,
            'line_search_iter': self.line_search_iter,
            'gamma': self.gamma,
            'verbose': self.verbose,
            'numerator_smoothing_penalty': self.numerator_smoothing_penalty
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

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

        grad = f(self.w, grad=True)
        d = self.w * (grad - grad @ self.w)

        max_step_size = self._max_step_size(self.w, grad, tol=self.tol) * self.gamma

        step_size = self.backtracking_armijo_line_search(f, self.w, d, max_step_size,
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

        evaluated_legendre = LegendrePolynomial(self.n, X, grad=False)

        self._legendre_coef = [self._compute_legendre_coef(np.ones(len(X)), y, evaluated_legendre,
                                                           self.numerator_smoothing_penalty, self.n)
                               for y in target_ys]

        if len(self.w) == 1:
            return self

        outer_w_old = 1  # Needs to be large enough so the while loop starts
        self.outer_n_iter_ = 0
        while self.outer_n_iter_ < self.outer_max_iter and np.linalg.norm(outer_w_old - self.w) > self.stopping_tol:
            outer_w_old = self.w.copy()

            inner_n_iter = 0
            inner_w_old = np.ones_like(self.w)
            while inner_n_iter < self.inner_max_iter and np.linalg.norm(inner_w_old - self.w) > self.stopping_tol:
                inner_w_old[:] = self.w
                self.w = self._search(X, target_ys)
                inner_n_iter += 1

            denominator = self.denominator(X)
            self._legendre_coef = [self._compute_legendre_coef(denominator, y, evaluated_legendre,
                                                               self.numerator_smoothing_penalty, self.n)
                                   for y in target_ys]

            self.outer_n_iter_ += 1

            if self.verbose:
                diffs = [y - self._numerator(X, coef) / denominator
                         for y, coef in zip(target_ys, self._legendre_coef)]
                self._writer.write_norms(self.outer_n_iter_, diffs, norms=[2, np.inf])

        if self.verbose:
            print()

        if self.outer_n_iter_ == self.outer_max_iter:
            warnings.warn("Maximum number of iterations has been reached and convergence is not guaranteed. "
                          "Try increasing `max_iter` or increasing `stopping_tol`.", category=ConvergenceWarning)

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

    def gridsearch(self, X, y, return_scores=False, keep_best=True, **param_grids):
        """ Performs a grid search over the values given in param_grids

            Parameters
            ----------
            X : (N,) np.ndarray
                The input

        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
        loss = []

        default_param_combination = self.get_params()
        param_combination_generator = ParameterGrid(param_grids)
        for param_combination in param_combination_generator:
            new_param_combinations = default_param_combination.copy()
            new_param_combinations.update(param_combination)

            model = CauchySimplex(**new_param_combinations)
            model.fit(X_train, y_train)

            mse = np.mean((model(X_test) - y_test) ** 2)
            loss.append((mse, model))

        if keep_best:
            sorted_loss = sorted(loss)
            best_param = sorted_loss[0][1].get_params()

            self.set_params(**best_param)
            if self.m is None:
                self.m = self.n

            self.fit(X, y)

        if return_scores:
            return loss
