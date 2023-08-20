from abc import ABC

import numpy as np

from .ArmijoSearch import ArmijoSearch

from ..utils import bernstein_to_legendre_matrix, bernstein_to_chebyshev_matrix
from ..Polynomials import BernsteinPolynomial, LegendrePolynomial
from ..validation_checks import check_bernstein_w, check_X_in_range
from ..RationalApproximator import RationalApproximator


class BernsteinApproximator(ArmijoSearch, RationalApproximator, ABC):
    def __init__(self, n, m=None, numerator_smoothing_penalty=None):
        self.w = None

        self.n = n
        self.m = n if m is None else m

        self.numerator_smoothing_penalty = numerator_smoothing_penalty

        self.domain = [0, 1]

        self._legendre_coef = None

    def f(self, X, target_ys, w, grad=False):
        """

            Parameters
            ----------
            target_ys : list of np.ndarray
            w : np.ndarray
            grad : bool
        """
        check_X_in_range(X, 0, 1)

        evaluated_legendre = LegendrePolynomial(self.n, X, grad=False)

        denominator = self._denominator(X, w)

        if np.any(denominator == 0):
            return np.inf

        legendre_coefs = [self._compute_legendre_coef(denominator, y, evaluated_legendre,
                                                      self.numerator_smoothing_penalty, self.n)
                          for y in target_ys]
        numerators = [coef @ evaluated_legendre for coef in legendre_coefs]

        difference = [y - numerator / denominator for y, numerator in zip(target_ys, numerators)]

        if grad:
            B = BernsteinPolynomial(self.m, X)
            grads = [B @ (diff * (numerator / (denominator ** 2)))
                     for (numerator, diff, y) in zip(numerators, difference, target_ys)]
            return np.mean(grads, axis=0)

        return sum([np.mean(z ** 2) for z in difference])

    def denominator(self, x, grad=False):
        check_X_in_range(x, 0, 1)

        if len(self.w) == 1:
            return np.ones_like(x)

        return self._denominator(x, self.w, grad=grad)

    def _denominator(self, X, w, grad=False):
        return w @ BernsteinPolynomial(self.m, X, grad=grad)

    def numerator(self, x, grad=False):
        check_X_in_range(x, 0, 1)

        numerator_vals = self._eval_numerator(x, grad=grad)

        if len(numerator_vals) == 1:
            return numerator_vals[0]
        return numerator_vals

    def _eval_numerator(self, x, grad=False):
        if grad:
            return self._numerator_grad(x)

        numerator_vals = [self._numerator(x, coef) for coef in self._legendre_coef]
        return numerator_vals

    def _numerator(self, X, legendre_coef):
        if len(legendre_coef) == 1:
            return np.ones_like(X)

        P = LegendrePolynomial(self.n, X)
        return legendre_coef @ P

    def _numerator_grad(self, x):
        numerator_grads = [c @ LegendrePolynomial(self.n, x, grad=True) for c in self._legendre_coef]
        return numerator_grads

    def reset(self, w=None):
        self.w = check_bernstein_w(w, self.m + 1)

    def __call__(self, x, grad=False):
        check_X_in_range(x, 0, 1)

        if grad:
            return self._grad(x)

        denominator = self.denominator(x)
        numerator = self.numerator(x)

        if isinstance(numerator, np.ndarray):
            return numerator / denominator
        else:
            return [num / denominator for num in numerator]

    def _grad(self, x):
        numerator_vals = self._eval_numerator(x, grad=False)
        numerator_grads = self._eval_numerator(x, grad=True)

        denominator_val = self.denominator(x, grad=False)
        denominator_grad = self.denominator(x, grad=True)

        grads = [(numerator_grad * denominator_val - numerator_val * denominator_grad) / (denominator_val ** 2)
                 for (numerator_val, numerator_grad) in zip(numerator_vals, numerator_grads)]

        if len(grads) == 1:
            return grads[0]
        return grads

    @property
    def legendre_coef(self):
        return self._legendre_coef[0] if len(self._legendre_coef) == 1 else self._legendre_coef

    def w_as_legendre_coef(self):
        M = bernstein_to_legendre_matrix(self.m)
        return M @ self.w

    def w_as_chebyshev_coef(self):
        M = bernstein_to_chebyshev_matrix(self.m)
        return M @ self.w

    def poles(self):
        roots = []

        if self.w[0] == 0:
            roots.append(0)

        if self.w[-1] == 0:
            roots.append(1)

        return np.array(roots)

    @staticmethod
    def _compute_legendre_coef(denominator, y, evaluated_legendre, smoothing_penalty, n):
        support = denominator > 0
        design_matrix = evaluated_legendre[:, support] / denominator[None, support]

        if smoothing_penalty is None:
            coef, *_ = np.linalg.lstsq(design_matrix.T, y[support], rcond=None)
        else:
            coef_weight = BernsteinApproximator.get_smoothing_penalty(n)
            coef = np.linalg.inv(design_matrix @ design_matrix.T \
                                 + smoothing_penalty * np.diag(coef_weight)) @ design_matrix @ y[support]

        return coef

    @staticmethod
    def get_smoothing_penalty(n):
        return np.arange(n + 1) ** np.arange(n + 1)
