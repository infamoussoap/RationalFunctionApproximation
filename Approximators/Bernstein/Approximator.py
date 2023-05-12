from abc import ABC

import numpy as np
from numpy.polynomial.legendre import Legendre

from .ArmijoSearch import ArmijoSearch

from ..utils import BernsteinPolynomial, check_bernstein_w


class Approximator(ArmijoSearch, ABC):
    def __init__(self):
        self.w = None
        self.m = None
        self.legendre_coef = None

    @property
    def denominator(self):
        if len(self.w) == 1:
            return lambda x: np.ones_like(x)

        def f(eval_points):
            return self.w @ BernsteinPolynomial(self.m, eval_points)

        return f

    @property
    def numerator(self):
        if len(self.legendre_coef) == 1:
            return lambda x: np.ones_like(x)

        return Legendre(self.legendre_coef, domain=[0, 1])

    def reset(self, w=None):
        self.w = check_bernstein_w(w, self.m + 1)
