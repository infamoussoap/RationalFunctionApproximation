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

        self.w = np.ones(m + 1) / (m + 1)

    def f(self, X, target_ys, w, grad=False):
        """

            Parameters
            ----------
            target_ys : list of np.ndarray
            w : np.ndarray
            grad : bool
        """
        check_X_in_range(X, 0, 1)

        denominator = self._denominator(X, w)
        weighted_target_ys = [y * denominator for y in target_ys]
        legendre_coefs = [self._compute_legendre_coef(X, y) for y in weighted_target_ys]

        difference = np.array([y - self._numerator(X, coef) for y, coef in zip(weighted_target_ys, legendre_coefs)])
        difference = difference / self._denominator(X, self.w)

        if grad:
            B = BernsteinPolynomial(self.m, X)
            grads = [B @ (z * y) for (z, y) in zip(difference, target_ys)]
            return np.mean(grads, axis=0)

        return sum([np.mean(z ** 2) for z in difference])

    def _denominator(self, X, w):
        B = BernsteinPolynomial(self.m, X)
        return w @ B

    def _numerator(self, X, legendre_coef):
        P = LegendrePolynomial(self.n, X)
        return legendre_coef @ P

    def _compute_legendre_coef(self, X, y):
        model = Legendre.fit(X, y, deg=self.n, domain=self.domain)
        return model.coef
