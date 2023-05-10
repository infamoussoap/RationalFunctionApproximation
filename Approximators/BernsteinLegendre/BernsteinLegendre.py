import numpy as np
from numpy.polynomial.legendre import Legendre

from .utils import spacing
from .utils import BernsteinPolynomial, LegendrePolynomial


class BernsteinLegendre:
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

    def f(self, target_function, x, grad=False):
        target_values = target_function(self.integration_points)
        R = target_values * self._denominator(x) - self._numerator(x)

        if grad:
            dL_dw = (self.dx * self.B) @ (R * target_values)  # Might need to be scaled by dx
            dL_dc = (self.dx * -self.P) @ R
            return np.concatenate([dL_dw, dL_dc])

        return (R ** 2) @ self.dx

    def _denominator(self, x):
        w = x[:self.m + 1]
        return w @ self.B

    def _numerator(self, x):
        legendre_coef = x[self.m + 1:]
        return legendre_coef @ self.P
