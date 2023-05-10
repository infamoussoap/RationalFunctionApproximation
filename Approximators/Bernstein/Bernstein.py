import numpy as np
from numpy.polynomial.legendre import Legendre

from .utils import combination, spacing, BernsteinPolynomial


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

    def f(self, target_function, w, grad=False):
        legendre_coef = self._legendre_coef(target_function, w)
        target_y = target_function(self.integration_points) * self._denominator(w, self.integration_points)

        z = target_y - self._numerator(legendre_coef, self.integration_points)

        if grad:
            return self.B @ (z * target_function(self.integration_points))

        return (z ** 2) @ self.dx

    def _denominator(self, w, eval_points):
        return w @ BernsteinPolynomial(self.m, eval_points)

    @staticmethod
    def _numerator(legendre_coef, eval_points):
        return Legendre(legendre_coef, domain=[0, 1])(eval_points)

    def _legendre_coef(self, target_function, w):
        target_y = target_function(self.integration_points) * self._denominator(w, self.integration_points)
        model = Legendre.fit(self.integration_points, target_y, deg=self.n, domain=self.domain)

        return model.coef
