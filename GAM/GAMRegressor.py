import numpy as np

from .RationalFunction import Zeros, RationalFunction
from .WriteToScreen import WriteToScreen


class GAMRegressor:
    def __init__(self, n, m=None, tol=1e-10):
        self.n = n
        self.m = n if m is None else m

        self.tol = tol

        self.n_features_in_ = None
        self.intercept_ = None

        self._learner_functions = []

        self._writer = WriteToScreen()

    def fit(self, X, y, num_rounds=10, max_iter=100, stopping_tol=1e-6, w=None,
            c1=1e-4, c2=0.5, line_search_iter=100, gamma=0.9, verbose=False):
        """ Fits a rational polynomial to the given datapoints

            Parameters
            ----------
            X : (k, p) np.ndarray
                The input data
            y : (k, ) np.ndarray
                Target values to be fitted against
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

            Returns
            -------
            self : object
                Fitted GAM
        """
        self.n_features_in_ = X.shape[1]
        self.intercept_ = np.mean(y)

        self._learner_functions = [RationalFunction(self.n, self.m, X[:, i]) for i in range(self.n_features_in_)]

        for count in range(num_rounds):
            self._fit(X, y, max_iter=max_iter, stopping_tol=stopping_tol, w=w, c1=c1, c2=c2,
                      line_search_iter=line_search_iter, gamma=gamma)

            if verbose:
                residuals = y - self.predict(X)
                self._writer.write(f"{count + 1}: {residuals @ residuals}", header='\r')

        if verbose:
            print()

    def _fit(self, X, y, max_iter=100, stopping_tol=1e-6, w=None, c1=1e-4, c2=0.5, line_search_iter=100, gamma=0.9):
        for i in range(self.n_features_in_):
            target_y = y - self.intercept_ - np.sum([f(X[:, j])
                                                     for j, f in enumerate(self._learner_functions) if j != i], axis=0)

            self._learner_functions[i].fit(target_y, max_iter=max_iter, stopping_tol=stopping_tol, w=w, c1=c1, c2=c2,
                                           line_search_iter=line_search_iter, gamma=gamma)

    def predict(self, X):
        return np.sum([f(X[:, j]) for j, f in enumerate(self._learner_functions)], axis=0) + self.intercept_
