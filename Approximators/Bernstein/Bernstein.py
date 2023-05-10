import numpy as np
from numpy.polynomial.legendre import Legendre

from .utils import combination, spacing, BernsteinPolynomial


class Bernstein:
    def __init__(self, target_function, m, n=None, num_integration_points=100, spacing_type='linear'):
        """ m is the degree of the denominator
            n is the degree of the numerator
        """
        self.target_function = target_function

        self.m = m
        self.n = m if n is None else n

        integration_points = spacing(spacing_type=spacing_type, n_points=num_integration_points)
        self.integration_points = integration_points[:-1]
        self.domain = [0, 1]

        self.dx = integration_points[1:] - integration_points[:-1]

        self.B = BernsteinPolynomial(m, self.integration_points)

    def f(self, w, grad=False):
        target_y = self.target_function(self.integration_points) * self.denominator(w)
        z = target_y - self.numerator(w)

        if grad:
            return self.B @ (z * self.target_function(self.integration_points))

        return (z ** 2) @ self.dx

    def denominator(self, w):
        return w @ self.B

    def numerator(self, w):
        target_y = self.target_function(self.integration_points) * (w @ self.B)
        P = Legendre.fit(self.integration_points, target_y, deg=self.n, domain=self.domain)

        return P(self.integration_points)
