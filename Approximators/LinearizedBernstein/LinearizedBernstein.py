import numpy as np
from numpy.polynomial.legendre import Legendre

import cvxopt

from ..Polynomials import LegendrePolynomial, BernsteinPolynomial
from ..RationalApproximator import RationalApproximator


class LinearizedBernstein(RationalApproximator):
    def __init__(self, n, m=None):
        self.n = n
        self.m = n if m is None else m

        self.legendre_coef = None
        self.w = None

    def fit(self, x, y):
        self.legendre_coef, self.w = self.solve_as_qp(x, y, self.n, self.m)

    @staticmethod
    def solve_as_qp(x, y, n, m):
        P = LegendrePolynomial(n, x)
        B = BernsteinPolynomial(m, x)

        R = np.hstack([P.T, -y[:, None] * B.T])
        Q = cvxopt.matrix(R.T @ R)
        c = cvxopt.matrix(np.zeros(n + m + 2))

        A_ub = cvxopt.matrix(np.hstack([np.zeros((m + 1, n + 1)), -np.eye(m + 1)]))
        b_ub = cvxopt.matrix(np.zeros(m + 1))

        A_eq = cvxopt.matrix(np.zeros((1, n + m + 2)))
        A_eq[n + 1:] = 1
        b_eq = cvxopt.matrix(1.0)

        sol = cvxopt.solvers.qp(Q, c, A_ub, b_ub, A_eq, b_eq)

        x_sol = np.array(sol['x']).flatten()

        legendre_coef = x_sol[:n + 1]
        w = x_sol[n + 1:]

        return legendre_coef, w

    def __call__(self, x):
        return self.numerator(x) / self.denominator(x)

    def numerator(self, x):
        return self.legendre_coef @ LegendrePolynomial(self.n, x)

    def denominator(self, x):
        return self.w @ BernsteinPolynomial(self.m, x)

    def poles(self):
        denominator = Legendre(self.w, domain=[0, 1])

        roots = denominator.roots()
        real_roots = np.real(roots[np.isreal(roots)])

        return real_roots[(0 <= real_roots) * (real_roots <= 1)].copy()
