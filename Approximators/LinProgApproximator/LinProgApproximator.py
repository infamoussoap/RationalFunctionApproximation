import numpy as np
from numpy.polynomial.legendre import Legendre

from scipy.optimize import linprog

from ..Polynomials import LegendrePolynomial


class LinProgApproximator:
    """ Implementation of the paper
            Rational approximation and its application to improving deep learning classifiers
            by V. Peiris, N. Sharon, N. Sukhorukova J. Ugon

        This approximator reformulates the problem as a linear programming problem, with an extra constraint that
        the denominator at the evaluation points must be greater than delta, and thus positive at those points
    """
    def __init__(self, n, m, stopping_tol=1e-10, denominator_lb=0.1, denominator_ub=50):
        self.stopping_tol = stopping_tol
        self.denominator_lb = denominator_lb
        self.denominator_ub = denominator_ub

        self.n = n
        self.m = m

        self.z = None

        self.numerator_coef = None
        self.denominator_coef = None

    def fit(self, x, y):
        P = LegendrePolynomial(self.n, x)
        Q = LegendrePolynomial(self.m, x)

        l = 0
        # u = self._legendre_fit_max_error(x, y, self.n + self.m)
        u = (y.max() - y.min()) / 2

        self.z = (u + l) / 2
        x_valid = None
        while u - l > self.stopping_tol:
            success, x = self.solve_polynomial_approximation(y, self.z, P, Q, self.denominator_lb, self.denominator_ub)
            if success:
                u = self.z
                x_valid = x.copy()
            else:
                l = self.z

            self.z = (u + l) / 2

        if x_valid is None:
            raise ValueError("Linear Program did not converge")

        self.numerator_coef = x_valid[:self.n + 1]
        self.denominator_coef = x_valid[self.n + 1: -1]
        return self

    @staticmethod
    def solve_polynomial_approximation(y, z, P, Q, denominator_lb, denominator_ub):
        N = len(y)

        # Max polynomial degree
        n = len(P) - 1
        m = len(Q) - 1

        A = np.vstack([
            np.hstack([-P.T, (y - z)[:, None] * Q.T, -np.ones((N, 1))]),
            np.hstack([P.T, -(y + z)[:, None] * Q.T, -np.ones((N, 1))]),
            np.hstack([np.zeros((N, n + 1)), -Q.T, np.zeros((N, 1))])
        ])

        b = np.vstack([np.zeros((2 * N, 1)), -denominator_lb * np.ones((N, 1))])

        if denominator_ub is not None:
            A = np.vstack([A, np.hstack([np.zeros((N, n + 1)), Q.T, np.zeros((N, 1))])])
            b = np.vstack([b, denominator_ub * np.ones((N, 1))])

        c = np.zeros(n + m + 3)
        c[-1] = 1

        lb = -np.inf * np.ones(n + m + 3)
        lb[n + 1] = 0
        lb[-1] = -1

        ub = np.inf * np.ones(n + m + 3)

        result = linprog(c, A_ub=A, b_ub=b, bounds=np.stack([lb, ub], axis=1), method='highs')

        if result.success:
            return (result.x[-1] < 1e-13), result.x
        return False, None

    @staticmethod
    def _legendre_fit_max_error(x, y, degree):
        P = LegendrePolynomial(degree, x)
        coef, *_ = np.linalg.lstsq(P.T, y, rcond=None)

        return np.max(abs(coef @ P - y))

    def __call__(self, x):
        return self.numerator(x) / self.denominator(x)

    def numerator(self, x):
        return self.numerator_coef @ LegendrePolynomial(self.n, x)

    def denominator(self, x):
        return self.denominator_coef @ LegendrePolynomial(self.m, x)

    def poles(self):
        denominator = Legendre(self.denominator_coef, domain=[0, 1])

        roots = denominator.roots()
        real_roots = np.real(roots[np.isreal(roots)])

        return real_roots[(0 <= real_roots) * (real_roots <= 1)].copy()
