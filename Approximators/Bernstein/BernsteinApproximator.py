from abc import ABC

import numpy as np
from numpy.polynomial.legendre import Legendre

from .ArmijoSearch import ArmijoSearch

from ..utils import BernsteinPolynomial, bernstein_to_legendre_matrix
from ..utils import check_bernstein_w, check_X_in_range
from ..RationalApproximator import RationalApproximator


class BernsteinApproximator(ArmijoSearch, RationalApproximator, ABC):
    def __init__(self):
        self.w = None
        self.m = None
        self._legendre_coef = None

    def denominator(self, x):
        check_X_in_range(x, 0, 1)

        if len(self.w) == 1:
            return np.ones_like(x)

        return self.w @ BernsteinPolynomial(self.m, x)

    def numerator(self, x):
        check_X_in_range(x, 0, 1)

        numerator_vals = [np.ones_like(x) if len(coef) == 1 else Legendre(coef, domain=[0, 1])(x)
                          for coef in self._legendre_coef]

        if len(numerator_vals) == 1:
            return numerator_vals[0]
        return numerator_vals

    def reset(self, w=None):
        self.w = check_bernstein_w(w, self.m + 1)

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

    @property
    def w_as_legendre_coef(self):
        M = bernstein_to_legendre_matrix(self.m)
        return M @ self.w
