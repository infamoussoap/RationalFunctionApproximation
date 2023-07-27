import numpy as np

import warnings

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import Approximators
from Approximators.Bernstein import CauchySimplex as Bernstein

from .WriteToScreen import WriteToScreen

from .utils import check_rational_degrees

WRITER = WriteToScreen()
Approximators.ignore_warnings()


class GAMRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, rational_degrees=(2, 2), stopping_tol=1e-6, max_iter=10, bernstein_max_iter=10,
                 bernstein_stopping_tol=1e-10, w=None, c1=1e-4, c2=0.5, line_search_iter=100, gamma=0.1, verbose=False):
        """ Initialize a generalized additive error model where each function is a rational function

        Parameters
        ----------
        rational_degrees : 2-tuple or list of 2-tuple, default (2, 2)
            Defines the rational degrees for each function. If a 2-tuple, then it is applied to each function.
        stopping_tol : float, default 1e-6
            The stopping tolerance to stop the backfitting procedure
        max_iter : int, default=10
            Number of rounds to perform backfitting
        bernstein_max_iter : int, default=10
            The number of iterations to perform the optimization when fitting each function. Ideally this would be
            as big as you can make it, but it seems to work best when this is small. This might be so each function
            doesn't over fit.
        bernstein_stopping_tol : float, default=1e-10
            The tolerance for the stopping criteria when performing the optimization when fitting each function.
            If |w_prev - w_new| < stopping_tol then the iteration will stop
        w : (m + 1, ) np.ndarray, default=None
            The starting point for optimization. If None is given then it will default to (np.ones(m + 1) / (m + 1).
        c1 : float, default=1e-4
            Parameter for the armijo line search
        c2 : float, default=0.5
            Parameter for the armijo line search
        line_search_iter : int, default=100
            Number of iterations for the line search
        gamma : float, default=0.1
            Expected to be a float between [0, 1]. It represents the percent of the maximum step size
            to be taken
        verbose : bool, default=False
            If set to true then the result of each step will be printed.
        """
        self.rational_degrees = rational_degrees

        self.stopping_tol = stopping_tol
        self.max_iter = max_iter

        self.bernstein_max_iter = bernstein_max_iter
        self.bernstein_stopping_tol = bernstein_stopping_tol

        self.w = w

        self.c1 = c1
        self.c2 = c2
        self.line_search_iter = line_search_iter
        self.gamma = gamma

        self.verbose = verbose

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return {
            'rational_degrees': self.rational_degrees,
            'stopping_tol': self.stopping_tol,
            'num_rounds': self.max_iter,
            'bernstein_max_iter': self.bernstein_max_iter,
            'bernstein_stopping_tol': self.bernstein_stopping_tol,
            'w': self.w,
            'c1': self.c1,
            'c2': self.c2,
            'line_search_iter': self.line_search_iter,
            'gamma': self.gamma,
            'verbose': self.verbose
        }

    def fit(self, X, y):
        """ Fits a rational polynomial to the given datapoints

            Parameters
            ----------
            X : (k, p) np.ndarray
                The input data
            y : (k, ) np.ndarray
                Target values to be fitted against

            Returns
            -------
            self : object
                Fitted GAM
        """
        X, y = check_X_y(X, y)

        self.n_features_in_ = X.shape[1]
        self.intercept_ = np.mean(y)

        rational_degrees = check_rational_degrees(self.rational_degrees, self.n_features_in_)
        self._learner_functions = [Bernstein(n, m, max_iter=self.bernstein_max_iter,
                                             stopping_tol=self.bernstein_stopping_tol,
                                             w=self.w, c1=self.c1, c2=self.c2,
                                             line_search_iter=self.line_search_iter, gamma=self.gamma)
                                   for (n, m) in rational_degrees]
        self._learner_functions_constants = np.zeros(self.n_features_in_)

        prev_loss = 0
        current_loss = np.inf
        self.n_iter_ = 0
        while self.n_iter_ < self.max_iter and abs(current_loss - prev_loss) > self.stopping_tol:
            self._fit(X, y)

            prev_loss = current_loss

            residuals = y - self.predict(X)
            current_loss = np.mean(residuals ** 2)

            if self.verbose:
                WRITER.write(f"{self.n_iter_ + 1}: {current_loss}", header='\r')

            self.n_iter_ += 1

        if self.verbose:
            print()

        if self.n_iter_ == self.max_iter:
            warnings.warn("Maximum number of iterations has been reached and convergence is not guaranteed. "
                          "Try increasing `max_iter` or increasing `stopping_tol`.")

        return self

    def _fit(self, X, y):
        for i in range(self.n_features_in_):
            learner_function_predictions = [f(X[:, j]) + c if f.w is not None else np.zeros_like(X[:, j])
                                            for j, (f, c) in
                                            enumerate(zip(self._learner_functions, self._learner_functions_constants))
                                            if j != i]
            target_y = y - self.intercept_ - np.sum(learner_function_predictions, axis=0)

            self._learner_functions[i].fit(X[:, i], target_y)
            self._learner_functions_constants[i] = -np.mean(self._learner_functions[i](X[:, i]))

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        learner_function_predictions = [f(X[:, j]) + c if f.w is not None else np.zeros_like(X[:, j])
                                        for j, (f, c) in
                                        enumerate(zip(self._learner_functions, self._learner_functions_constants))]
        return self.intercept_ + np.sum(learner_function_predictions, axis=0)
