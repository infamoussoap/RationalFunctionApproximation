from abc import ABC

import numpy as np
from numpy.polynomial.legendre import Legendre

from .ArmijoSearch import ArmijoSearch

from ..utils import BernsteinPolynomial, check_bernstein_w


class Approximator(ArmijoSearch, ABC):
    def __init__(self):
        self.w = None
        self.m = None
        self._legendre_coef = None

    @property
    def denominator(self):
        if len(self.w) == 1:
            return lambda x: np.ones_like(x)

        def f(eval_points):
            return self.w @ BernsteinPolynomial(self.m, eval_points)

        return f

    @property
    def numerator(self):
        numerator_functions = [(lambda x: np.ones_like(x)) if len(coef) == 1 else Legendre(coef, domain=[0, 1])
                               for coef in self._legendre_coef]

        if len(numerator_functions) == 1:
            return numerator_functions[0]
        return numerator_functions

    def reset(self, w=None):
        self.w = check_bernstein_w(w, self.m + 1)

    def __call__(self, eval_points):
        denominator = self.denominator(eval_points)
        try:
            numerator = self.numerator(eval_points)
        except:
            return [f(eval_points) / denominator for f in self.numerator]
        else:
            return numerator / denominator

    @property
    def legendre_coef(self):
        return self._legendre_coef[0] if len(self._legendre_coef) == 1 else self._legendre_coef
