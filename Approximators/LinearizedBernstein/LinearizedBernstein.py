import numpy as np
from numpy.polynomial.legendre import Legendre

from ..Polynomials import LegendrePolynomial, BernsteinPolynomial
from ..RationalApproximator import RationalApproximator

from ..validation_checks import check_target_ys, check_X_in_range, check_numerator_degrees

import cvxopt


class LinearizedBernstein(RationalApproximator):
    def __init__(self, n, m=None, max_iter=10, stopping_tol=1e-6, cvxopt_maxiter=500):
        self.n = n
        self.m = n if m is None else m

        self._legendre_coef = None
        self.w = None

        self.max_iter = max_iter
        self.n_iter_ = 0

        self.stopping_tol = stopping_tol

        self.sol = None

        cvxopt.solvers.options['maxiters'] = cvxopt_maxiter

    def fit(self, x, y):
        check_X_in_range(x, 0, 1)
        target_ys = check_target_ys(y)

        n_vals = check_numerator_degrees(self.n, len(target_ys))

        weight = np.ones_like(x)
        w_old = 2
        self.w = np.ones(self.m + 1)
        while self.n_iter_ < self.max_iter and np.linalg.norm(w_old - self.w) > self.stopping_tol:
            w_old = self.w.copy()

            self._legendre_coef, self.w, self.sol = self.solve_as_qp(x, target_ys, n_vals, self.m, weight)
            weight = self.w @ BernsteinPolynomial(self.m, x)

            self.n_iter_ += 1

        return self

    @staticmethod
    def solve_as_qp(x, target_y, n_vals, m, weight):
        B = BernsteinPolynomial(m, x)

        N = sum(n_vals) + len(n_vals)

        Q = np.zeros((N + (m + 1), N + (m + 1)))
        for i, (n, y) in enumerate(zip(n_vals, target_y)):
            P = LegendrePolynomial(n, x)

            temp_matrix_list = [(P / weight).T if i == j else np.zeros((len(x), k + 1))
                                for j, k in enumerate(n_vals)]
            temp_matrix_list += [-(y / weight)[:, None] * B.T]

            R = np.hstack(temp_matrix_list)

            Q += R.T @ R

        Q = cvxopt.matrix(Q)
        c = cvxopt.matrix(np.zeros(N + (m + 1)))

        A_ub = cvxopt.matrix(np.hstack([np.zeros((m + 1, k + 1)) for k in n_vals] + [-np.eye(m + 1)]))
        b_ub = cvxopt.matrix(np.zeros(m + 1))

        A_eq = cvxopt.matrix(np.zeros((1, N + (m + 1))))
        A_eq[N:] = 1
        b_eq = cvxopt.matrix(1.0)

        sol = cvxopt.solvers.qp(Q, c, A_ub, b_ub, A_eq, b_eq)

        x_sol = np.array(sol['x']).flatten()

        legendre_coefs = x_sol[:N]
        w = x_sol[N:]

        start_index = [0] + list(np.cumsum(np.array(n_vals) + 1))[:-1]
        end_index = list(np.cumsum(np.array(n_vals) + 1))

        legendre_coefs = [legendre_coefs[start:end] for start, end in zip(start_index, end_index)]

        return legendre_coefs, w, sol

    def __call__(self, x):
        numerator_vals = self.numerator(x)
        denominator_vals = self.denominator(x)

        if isinstance(numerator_vals, np.ndarray):
            return numerator_vals / denominator_vals
        return [num / denominator_vals for num in numerator_vals]

    def numerator(self, x):
        n_vals = check_numerator_degrees(self.n, len(self._legendre_coef))

        numerator_vals = [coef @ LegendrePolynomial(n, x) for coef, n in zip(self._legendre_coef, n_vals)]

        if len(numerator_vals) == 1:
            return numerator_vals[0]
        return numerator_vals

    def denominator(self, x):
        return self.w @ BernsteinPolynomial(self.m, x)

    def poles(self):
        denominator = Legendre(self.w, domain=[0, 1])

        roots = denominator.roots()
        real_roots = np.real(roots[np.isreal(roots)])

        return real_roots[(0 <= real_roots) * (real_roots <= 1)].copy()

    @property
    def legendre_coef(self):
        if len(self._legendre_coef) == 1:
            return self._legendre_coef[0]
        return self._legendre_coef
