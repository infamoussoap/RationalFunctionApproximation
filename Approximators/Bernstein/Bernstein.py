import numpy as np
from numpy.polynomial.legendre import Legendre

from .utils import spacing, BernsteinPolynomial, LegendrePolynomial


class Bernstein:
    def __init__(self, m, n=None, num_integration_points=100, spacing_type='linear'):
        """ m is the degree of the denominator
            n is the degree of the numerator
        """
        self.m = m
        self.n = m if n is None else n

        integration_points = spacing(spacing_type=spacing_type, n_points=num_integration_points)
        self.integration_points = integration_points[:-1]
        self.domain = [0, 1]

        self.dx = integration_points[1:] - integration_points[:-1]

        self.B = BernsteinPolynomial(m, self.integration_points)
        self.P = LegendrePolynomial(n, self.integration_points)

    def f(self, target_function, w, grad=False):
        target_y = target_function(self.integration_points) * self._denominator(w)
        legendre_coef = self._legendre_coef(target_function, w, target_y=target_y)

        z = target_y - self._numerator(legendre_coef)

        if grad:
            return self.B @ (z * target_function(self.integration_points))

        return (z ** 2) @ self.dx

    def _denominator(self, w):
        return w @ self.B

    def _numerator(self, legendre_coef):
        return legendre_coef @ self.P

    def _legendre_coef(self, target_function, w, target_y=None):
        if target_y is None:
            target_y = target_function(self.integration_points) * self._denominator(w)

        model = Legendre.fit(self.integration_points, target_y, deg=self.n, domain=self.domain)

        return model.coef
