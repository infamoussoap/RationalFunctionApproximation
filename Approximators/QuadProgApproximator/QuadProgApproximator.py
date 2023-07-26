import numpy as np
from numpy.polynomial.legendre import Legendre

import cvxopt

from ..Polynomials import LegendrePolynomial

from ..RationalApproximator import RationalApproximator


class QuadProgApproximator(RationalApproximator):
    """ Implementation of the paper
            Rational approximation and its application to improving deep learning classifiers
            by V. Peiris, N. Sharon, N. Sukhorukova, and J. Ugon
            https://arxiv.org/abs/2002.11330
        But we reformulate it as a quadratic programming problem, with a constraint that
        the denominator at the evaluation points must be greater than delta
    """
    def __init__(self, n, m, denominator_lb=0.1, denominator_ub=50):
        self.denominator_lb = denominator_lb
        self.denominator_ub = denominator_ub

        self.n = n
        self.m = m

        self.numerator_coef = None
        self.denominator_coef = None

    def fit(self, x, y):
        P = LegendrePolynomial(self.n, x)
        Q = LegendrePolynomial(self.m, x)

        x_sol = self.solve_polynomial_approximation(y, P, Q, self.denominator_lb, self.denominator_ub)

        self.numerator_coef = x_sol[:self.n + 1]
        self.denominator_coef = x_sol[self.n + 1:]
        return self

    @staticmethod
    def solve_polynomial_approximation(y, P, B, denominator_lb, denominator_ub):
        N = len(y)

        # Max polynomial degree
        n = len(P) - 1
        m = len(B) - 1

        R = np.hstack([P.T, -y[:, None] * B.T])
        Q = cvxopt.matrix(R.T @ R)
        c = cvxopt.matrix(np.zeros(n + m + 2))

        A_ub = np.hstack([np.zeros((N, n + 1)), -B.T])
        b_ub = -denominator_lb * np.ones((N, 1))

        if denominator_ub is not None:
            A_ub = np.vstack([A_ub, np.hstack([np.zeros((N, n + 1)), B.T])])
            b_ub = np.vstack([b_ub, denominator_ub * np.ones((N, 1))])

        sol = cvxopt.solvers.qp(Q, c, cvxopt.matrix(A_ub), cvxopt.matrix(b_ub))

        x_sol = np.array(sol['x']).flatten()
        return x_sol

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
