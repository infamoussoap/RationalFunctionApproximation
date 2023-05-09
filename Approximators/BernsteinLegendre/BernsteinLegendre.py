import numpy as np
from scipy.special import eval_legendre

from .utils import combination


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

        self.denominator = Denominator(m, self.integration_points)
        self.numerator = Numerator(n, self.integration_points)

    def f(self, x, grad=False):
        w, c = x[:self.m + 1], x[self.m + 1:]

        target_values = self.target_function(self.integration_points)
        R = target_values * self.denominator(w) - self.numerator(c)

        if grad:
            dL_dw = self.denominator.B @ (R * target_values) * self.dx
            dL_dc = -self.numerator.P @ R * self.dx
            return np.concatenate([dL_dw, dL_dc])

        return np.sum(R ** 2) * self.dx


class Denominator:
    def __init__(self, m, integration_points):
        self.m = m
        self.integration_points = integration_points
        self.B = self.Bernstein(m, integration_points)

    def __call__(self, w):
        return w @ self.B

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


class Numerator:
    def __init__(self, n, integration_points):
        self.n = n
        self.integration_points = integration_points
        self.P = self.Legendre(n, integration_points)

    def __call__(self, c):
        return c @ self.P

    @staticmethod
    def Legendre(n, integration_points):
        assert integration_points[0] == 0 and integration_points[-1] == 1
        return np.vstack([eval_legendre(i, 2 * integration_points - 1) for i in range(n + 1)])
