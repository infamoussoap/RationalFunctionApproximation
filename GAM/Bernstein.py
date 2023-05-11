import numpy as np
from numpy.polynomial.legendre import Legendre

from .utils import BernsteinPolynomial, LegendrePolynomial


class Bernstein:
    def __init__(self, n, m, evaluation_points):
        """ m is the degree of the denominator
            n is the degree of the numerator
        """
        self.n = n
        self.m = n if m is None else m

        self.evaluation_points = evaluation_points
        self.domain = [0, 1]

        self.B = BernsteinPolynomial(m, self.evaluation_points)
        self.P = LegendrePolynomial(n, self.evaluation_points)

    def f(self, y, w, grad=False):
        target_y = y * self._denominator(w)
        legendre_coef = self._legendre_coef(y, w, target_y=target_y)

        z = target_y - self._numerator(legendre_coef)

        if grad:
            return self.B @ (z * y)
        
        return np.sum(z ** 2)

    def _denominator(self, w):
        return w @ self.B

    def _numerator(self, legendre_coef):
        return legendre_coef @ self.P

    def _legendre_coef(self, y, w, target_y=None):
        if target_y is None:
            target_y = y * self._denominator(w)

        model = Legendre.fit(self.evaluation_points, target_y, deg=self.n, domain=self.domain)

        return model.coef
