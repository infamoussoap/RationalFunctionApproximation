import numpy as np

from ..RationalApproximator import RationalApproximator
from ..utils import LegendrePolynomial


class SKApproximator(RationalApproximator):
    def __init__(self, n, m=None, evaluation_points=None):
        self.n = n
        self.m = n if m is None else m

        self.evaluation_points = np.linspace(0, 1, 100) if evaluation_points is None else evaluation_points

        self.a = np.ones(self.n + 1)  # Numerator Coefficients
        self.b = np.zeros(self.m)  # Denominator Coefficients

        self.n_iter_ = 0

    def fit(self, target_function, max_iter=100, stopping_tol=1e-6):
        self._reset_params()

        F = target_function(self.evaluation_points)

        P_legendre = LegendrePolynomial(self.n, self.evaluation_points)
        Q_legendre = LegendrePolynomial(self.m, self.evaluation_points)[1:]

        x_old = 2  # Needs to be large enough for the while loop to start

        while self.n_iter_ < max_iter and np.linalg.norm(x_old - np.concatenate([self.a, self.b])):
            Q = (1 + self.b @ Q_legendre)
            y = F / Q

            design_matrix = np.vstack([P_legendre / Q, - y * Q_legendre]).T

            coef, *_ = np.linalg.lstsq(design_matrix, y, rcond=None)

            self.a = coef[:self.n + 1]
            self.b = coef[self.n + 1:]

            self.n_iter_ += 1

        return self

    def numerator(self, x):
        return self.a @ LegendrePolynomial(self.n, x)

    def denominator(self, x):
        return self.b @ LegendrePolynomial(self.m, x)[1:] + 1

    def _reset_params(self):
        self.a = np.ones(self.n + 1)
        self.b = np.zeros(self.m)

        self.n_iter_ = 0
