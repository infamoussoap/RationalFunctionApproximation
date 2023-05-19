import numpy as np
from numpy.polynomial.legendre import Legendre

from ..utils import spacing_grid, BernsteinPolynomial, LegendrePolynomial


class Bernstein:
    def __init__(self, n, m=None, evaluation_points=None):
        """ m is the degree of the denominator
            n is the degree of the numerator
        """
        self.n = n
        self.m = n if m is None else m

        evaluation_points = np.linspace(0, 1, 100) if evaluation_points is None else evaluation_points

        self.evaluation_points = evaluation_points[:-1]
        self.domain = [0, 1]

        self.dx = evaluation_points[1:] - evaluation_points[:-1]

        self.B = BernsteinPolynomial(m, self.evaluation_points)
        self.P = LegendrePolynomial(n, self.evaluation_points)

    def f(self, target_functions, w, grad=False):
        target_y = [f(self.evaluation_points) * self._denominator(w) for f in target_functions]
        legendre_coefs = [self._compute_legendre_coef(f, w, target_y=y)
                          for y, f in zip(target_y, target_functions)]

        difference = [y - self._numerator(coef) for y, coef in zip(target_y, legendre_coefs)]

        if grad:
            grads = [(self.dx[None, :] * self.B) @ (z * f(self.evaluation_points))
                     for (z, f) in zip(difference, target_functions)]
            return np.sum(grads, axis=0)

        return sum([(z ** 2) @ self.dx for z in difference])

    def _denominator(self, w):
        return w @ self.B

    def _numerator(self, legendre_coef):
        return legendre_coef @ self.P

    def _compute_legendre_coef(self, target_function, w, target_y=None):
        if target_y is None:
            target_y = target_function(self.evaluation_points) * self._denominator(w)

        model = Legendre.fit(self.evaluation_points, target_y, deg=self.n, domain=self.domain)

        return model.coef
