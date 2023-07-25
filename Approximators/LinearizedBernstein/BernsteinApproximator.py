from abc import ABC

import numpy as np
from numpy.polynomial.legendre import Legendre

from .ArmijoSearch import ArmijoSearch

from ..utils import bernstein_to_legendre_matrix, bernstein_to_chebyshev_matrix
from ..Polynomials import BernsteinPolynomial, LegendrePolynomial
from ..validation_checks import check_bernstein_w, check_X_in_range
from ..RationalApproximator import RationalApproximator


class BernsteinApproximator(ArmijoSearch, RationalApproximator, ABC):
    def __init__(self):
        self.w = None

        self.n = None
        self.m = None

        self._legendre_coef = None

    def denominator(self, x, grad=False):
        check_X_in_range(x, 0, 1)

        if len(self.w) == 1:
            return np.ones_like(x)

        return self.w @ BernsteinPolynomial(self.m, x, grad=grad)

    def numerator(self, x, grad=False):
        check_X_in_range(x, 0, 1)

        numerator_vals = self._eval_numerator(x, grad=grad)

        if len(numerator_vals) == 1:
            return numerator_vals[0]
        return numerator_vals

    def _eval_numerator(self, x, grad=False):
        if grad:
            return self._numerator_grad(x)

        numerator_vals = [np.ones_like(x) if len(coef) == 1 else Legendre(coef, domain=[0, 1])(x)
                          for coef in self._legendre_coef]
        return numerator_vals

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
