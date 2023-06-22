import numpy as np
from numpy.polynomial.legendre import Legendre

from ..Polynomials import LegendrePolynomial, BernsteinPolynomial
from ..validation_checks import check_X_in_range


class Bernstein:
    def __init__(self, n, m=None):
        """

            Parameters
            ----------
            m : int
                Degree of the denominator
            n : int
                Degree of the numerator
        """
        self.n = n
        self.m = n if m is None else m

        self.domain = [0, 1]

    def f(self, w_prev, evaluated_bernstein, evaluated_legendre, target_ys, w, grad=False):
        denominator = w @ evaluated_bernstein
        weighted_target_ys = [y * denominator for y in target_ys]

        weights = 1 / (w_prev @ evaluated_bernstein)
        legendre_coefs = [self._weighted_least_squares(weights, evaluated_legendre.T, y) for y in weighted_target_ys]

        difference = [(y - coef @ evaluated_legendre) * weights
                      for y, coef in zip(weighted_target_ys, legendre_coefs)]

        if grad:
            grads = [evaluated_bernstein @ (z * y * weights) for (z, y) in zip(difference, target_ys)]
            return np.mean(grads, axis=0)

        return sum([np.mean(z ** 2) for z in difference])

    def _denominator(self, X, w):
        B = BernsteinPolynomial(self.m, X)
        return w @ B

    def _numerator(self, X, legendre_coef):
        P = LegendrePolynomial(self.n, X)
        return legendre_coef @ P

    @staticmethod
    def _weighted_least_squares(w, X, y):
        """ Returns the vector `a` such that it minimizes
            (w ** 2) @ ((y - X @ a) ** 2)
        """
        a, *_ = np.linalg.lstsq(X * w[:, None], w * y, rcond=None)
        return a

    def _compute_legendre_coef(self, X, y):
        model = Legendre.fit(X, y, deg=self.n, domain=self.domain)
        return model.coef
