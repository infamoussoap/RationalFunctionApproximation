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
        self.n_features_in_ = X.shape[1]
        self.intercept_ = np.mean(y)

        self._learner_functions = [Zeros() for _ in range(self.n_features_in_)]

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

            learner_function = RationalFunction(self.n, self.m, X[:, i])
            learner_function.fit(target_y, max_iter=max_iter, stopping_tol=stopping_tol, w=w, c1=c1, c2=c2,
                                 line_search_iter=line_search_iter, gamma=gamma)

            self._learner_functions[i] = learner_function

    def predict(self, X):
        return np.sum([f(X[:, j]) for j, f in enumerate(self._learner_functions)], axis=0) + self.intercept_
