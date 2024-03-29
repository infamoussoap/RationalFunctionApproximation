import numpy as np
from numpy.polynomial.legendre import Legendre

from ..RationalApproximator import RationalApproximator
from ..Polynomials import LegendrePolynomial
from ..validation_checks import check_X_in_range


class SKApproximator(RationalApproximator):
    def __init__(self, n, m=None, max_iter=100, stopping_tol=1e-6):
        self.n = n
        self.m = n if m is None else m

        self.max_iter = max_iter
        self.stopping_tol = stopping_tol

        self.a = np.ones(self.n + 1)  # Numerator Coefficients
        self.b = np.zeros(self.m)  # Denominator Coefficients

        self.n_iter_ = 0

    def fit(self, X, y):
        check_X_in_range(X, 0, 1)

        self._reset_params()

        P_legendre = LegendrePolynomial(self.n, X)
        Q_legendre = LegendrePolynomial(self.m, X)[1:]  # Ignoring the first Legendre polynomial

        x_old = 2  # Needs to be large enough for the while loop to start

        while self.n_iter_ < self.max_iter and \
                np.linalg.norm(x_old - np.concatenate([self.a, self.b])) > self.stopping_tol:
            Q = (1 + self.b @ Q_legendre)

            support = Q != 0

            weighted_y = y[support] / Q[support]
            design_matrix = np.vstack([P_legendre[:, support] / Q[support], - weighted_y * Q_legendre[:, support]]).T

            coef, *_ = np.linalg.lstsq(design_matrix, weighted_y, rcond=None)

            self.a = coef[:self.n + 1]
            self.b = coef[self.n + 1:]

            self.n_iter_ += 1

        return self

    def numerator(self, x):
        check_X_in_range(x, 0, 1)
        return self.a @ LegendrePolynomial(self.n, x)

    def denominator(self, x):
        check_X_in_range(x, 0, 1)
        return self.b @ LegendrePolynomial(self.m, x)[1:] + 1

    def _reset_params(self):
        self.a = np.ones(self.n + 1)
        self.b = np.zeros(self.m)

        self.n_iter_ = 0

    def poles(self):
        coef = np.ones(len(self.b) + 1)
        coef[1:] = self.b

        denominator = Legendre(coef, domain=[0, 1])

        roots = denominator.roots()
        real_roots = np.real(roots[np.isreal(roots)])

        return real_roots[(0 <= real_roots) * (real_roots <= 1)].copy()
