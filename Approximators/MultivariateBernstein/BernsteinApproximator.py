from abc import ABC

import numpy as np
from numpy.polynomial.legendre import Legendre

from .ArmijoSearch import ArmijoSearch

from ..utils import bernstein_to_legendre_matrix, bernstein_to_chebyshev_matrix
from ..Polynomials import MultivariateBernsteinPolynomial, MultivariateLegendrePolynomial
from ..validation_checks import check_bernstein_w, check_X_in_range
from ..RationalApproximator import RationalApproximator


class BernsteinApproximator(ArmijoSearch, RationalApproximator, ABC):
    def __init__(self):
        self.n_vals = None
        self.m_vals = None

        self.w = None
        self._legendre_coef = None

    def denominator(self, x):
        """ x assumed to be a numpy array of shape (# data points, # variables) """
        check_X_in_range(x, 0, 1)

        if len(self.w) == 1:
            return np.ones(len(x))

        B = MultivariateBernsteinPolynomial(self.m_vals, x)
        B = B.reshape(-1, len(x))
        return self.w @ B

    def numerator(self, x):
        check_X_in_range(x, 0, 1)

        numerator_vals = self._eval_numerator(x)

        if len(numerator_vals) == 1:
            return numerator_vals[0]
        return numerator_vals

    def _eval_numerator(self, x):
        """ x assumed to be a numpy array of shape (# data points, # variables) """
        numerator_vals = []
        for coef in self._legendre_coef:
            if len(coef) == 1:
                numerator_vals.append(np.ones(len(x)) * coef)
            else:
                P = MultivariateLegendrePolynomial(self.n_vals, x)
                P = P.reshape(-1, len(x))
                numerator_vals.append(coef @ P)

        return numerator_vals

    def reset(self, w=None):
        self.w = check_bernstein_w(w, int(np.prod([m + 1 for m in self.m_vals])))

    def __call__(self, x):
        check_X_in_range(x, 0, 1)

        denominator = self.denominator(x)
        numerator = self.numerator(x)

        if isinstance(numerator, np.ndarray):
            return numerator / denominator
        else:
            return [num / denominator for num in numerator]

    @property
    def legendre_coef(self):
        return self._legendre_coef[0] if len(self._legendre_coef) == 1 else self._legendre_coef
