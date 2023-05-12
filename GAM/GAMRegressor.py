import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .RationalFunction import RationalFunction
from .WriteToScreen import WriteToScreen

from .utils import check_rational_degrees

WRITER = WriteToScreen()


class GAMRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, rational_degrees=(2, 2), tol=1e-10, num_rounds=10, max_iter=100, stopping_tol=1e-6, w=None,
                 c1=1e-4, c2=0.5, line_search_iter=100, gamma=0.9, verbose=False):
        """

        :param rational_degrees:
        :param tol:
        num_rounds : int, default=10
            Number of rounds to perform backfitting
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
        verbose : bool, default=False
            If set to true then the result of each step will be printed.
        """
        self.rational_degrees = rational_degrees

        self.tol = tol
        self.num_rounds = num_rounds
        self.max_iter = max_iter
        self.stopping_tol = stopping_tol

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
            'tol': self.tol,
            'num_rounds': self.num_rounds,
            'max_iter': self.max_iter,
            'stopping_tol': self.stopping_tol,
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
        self._learner_functions = [RationalFunction(n, m, X[:, i])
                                   for i, (n, m) in enumerate(rational_degrees)]

        for count in range(self.num_rounds):
            self._fit(X, y)

            if self.verbose:
                residuals = y - self.predict(X)
                mse = np.mean(residuals ** 2)
                WRITER.write(f"{count + 1}: {mse}", header='\r')

        if self.verbose:
            print()

        return self

    def _fit(self, X, y):
        for i in range(self.n_features_in_):
            target_y = y - self.intercept_ - np.sum([f(X[:, j])
                                                     for j, f in enumerate(self._learner_functions) if j != i], axis=0)

            self._learner_functions[i].fit(target_y, max_iter=self.max_iter, stopping_tol=self.stopping_tol, w=self.w,
                                           c1=self.c1, c2=self.c2, line_search_iter=self.line_search_iter,
                                           gamma=self.gamma)

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)
        return np.sum([f(X[:, j]) for j, f in enumerate(self._learner_functions)], axis=0) + self.intercept_
