import numpy as np

from ..utils import spacing_grid
from ..Polynomials import LegendrePolynomial, BernsteinPolynomial


class BernsteinLegendre:
    def __init__(self, n, m=None, evaluation_points=None):
        """ m is the degree of the denominator
            n is the degree of the numerator
        """
        self.n = n
        self.m = n if m is None else m

        if self.m == 0 or self.n == 0:
            raise ValueError("Bernstein Legendre doesn't support numerator or denominators with 0-degree. Use"
                             " Bernstein instead.")

        evaluation_points = np.linspace(0, 1, 100) if evaluation_points is None else evaluation_points
        self.evaluation_points = evaluation_points[:-1]
        self.domain = [0, 1]

        self.dx = evaluation_points[1:] - evaluation_points[:-1]

        self.B = BernsteinPolynomial(m, self.evaluation_points)
        self.P = LegendrePolynomial(n, self.evaluation_points)

    def f(self, target_function, x, grad=False):
        target_values = target_function(self.evaluation_points)
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
