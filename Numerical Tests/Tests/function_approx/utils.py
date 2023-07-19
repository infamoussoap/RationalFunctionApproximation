import sys
sys.path.insert(0, '../..')

import Approximators
import Approximators.Bernstein.CauchySimplex as Bernstein
import Approximators.LinearizedBernstein.CauchySimplex as LinearizedBernstein
from Approximators import SKApproximator, AAAApproximator, PolelessBarycentric

from sklearn.linear_model import LinearRegression
from patsy import dmatrix, build_design_matrices


def get_approximators(n):
    return {'Bernstein': Bernstein(n, n, max_iter=1000, stopping_tol=0, gamma=0.9),
            'Linearized Bernstein': LinearizedBernstein(n, n, max_iter=1000, stopping_tol=0, gamma=0.9),
            'Polynomial': Bernstein(2 * n, 0),
            'Natural Spline': NaturalCubic(2 * n),
            'SK': SKApproximator(n, n),
            'AAA': AAAApproximator(n + 1, n + 1),
            'Poleless Barycentric': PolelessBarycentric(n + 1, n + 1, d=3) if n >= 4 else None}


class NaturalCubic:
    def __init__(self, df):
        self.df = df

        self.design_matrix = None
        self.ols_model = None

    def fit(self, X, y):
        self.design_matrix = dmatrix(f'cr(variable, df={self.df}) - 1', {'variable': X.reshape(-1, 1)})
        self.ols_model = LinearRegression(fit_intercept=True).fit(self.design_matrix, y)

        return self

    def predict(self, X):
        design_matrix = build_design_matrices([self.design_matrix.design_info], {'variable': X.reshape(-1, 1)})[0]
        return self.ols_model.predict(design_matrix)

    def __call__(self, X):
        return self.predict(X)

