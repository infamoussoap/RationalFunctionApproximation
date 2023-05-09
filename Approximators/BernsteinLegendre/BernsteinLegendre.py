import numpy as np
from numpy.polynomial.legendre import Legendre

from utils import combination


class BernsteinLegendre:
    def __init__(self, target_function, n, deg=None, num_integration_points=100):
        """ n is the degree of the denominator
            deg is the degree of the numerator
        """
        self.target_function = target_function

        self.n = n
        self.deg = n if deg is None else deg

        self.integration_points = np.linspace(0, 1, num_integration_points)
        self.domain = [self.integration_points[0], self.integration_points[-1]]

        self.dx = self.integration_points[1] - self.integration_points[0]

        self.B = self.Bernstein(n, self.integration_points)

    def f(self, w, grad=False):
        target_y = self.target_function(self.integration_points) * self.denominator(w)
        z = target_y - self.numerator(w)

        if grad:
            return self.B @ (z * self.target_function(self.integration_points))

        return np.sum(z ** 2) * self.dx

    def denominator(self, w):
        return w @ self.B

    def numerator(self, w):
        target_y = self.target_function(self.integration_points) * (w @ self.B)
        P = Legendre.fit(self.integration_points, target_y, deg=self.deg, domain=self.domain)

        return P(self.integration_points)

    @staticmethod
    def Bernstein(n, x):
        assert x[0] == 0 and x[-1] == 1

        k = np.arange(0, n + 1)
        log_B = np.zeros((n + 1, len(x)))

        log_B += combination(n, k, as_log=True)[:, None]
        log_B[:, 1:] += k[:, None] * np.log(x[None, 1:])
        log_B[:, :-1] += (n - k[:, None]) * np.log(1 - x[None, :-1])

        log_B[1:, 0] = -np.inf
        log_B[:-1, -1] = -np.inf

        B = np.exp(log_B)
        return B