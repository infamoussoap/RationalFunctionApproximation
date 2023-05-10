import numpy as np
from scipy.special import eval_legendre

from .utils import BernsteinPolynomial


class BernsteinLegendre:
    def __init__(self, target_function, m, n=None, num_integration_points=100):
        """ m is the degree of the denominator
            n is the degree of the numerator
        """
        self.target_function = target_function

        self.m = m
        self.n = m if n is None else n

        self.integration_points = np.linspace(0, 1, num_integration_points + 1)
        self.domain = [self.integration_points[0], self.integration_points[-1]]

        self.dx = self.integration_points[1] - self.integration_points[0]

        self.B = BernsteinPolynomial(m, self.integration_points)
        self.P = self.Legendre(n, self.integration_points)

    def f(self, x, grad=False):
        w, c = x[:self.m + 1], x[self.m + 1:]

        target_values = self.target_function(self.integration_points)
        R = target_values * self.denominator(w)(self.integration_points) - self.numerator(c)(self.integration_points)

        if grad:
            dL_dw = self.B @ (R * target_values) * self.dx
            dL_dc = -self.P @ R * self.dx
            return np.concatenate([dL_dw, dL_dc])

        return np.sum(R ** 2) * self.dx

    def denominator(self, w):
        def f(eval_points):
            return w @ BernsteinPolynomial(self.m, eval_points)
        return f

    def numerator(self, c):
        def f(eval_points):
            return c @ self.Legendre(self.n, eval_points)
        return f

    @staticmethod
    def Legendre(n, integration_points):
        return np.vstack([eval_legendre(i, 2 * integration_points - 1) for i in range(n + 1)])
