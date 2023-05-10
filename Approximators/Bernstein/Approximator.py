import numpy as np
from numpy.polynomial.legendre import Legendre

from .utils import BernsteinPolynomial, check_w


class Approximator:
    def __init__(self):
        self.w = None
        self.m = None
        self.legendre_coef = None

    @property
    def denominator(self):
        def f(eval_points):
            return self.w @ BernsteinPolynomial(self.m, eval_points)

        return f

    @property
    def numerator(self):
        return Legendre(self.legendre_coef, domain=[0, 1])

    def reset(self, w=None):
        self.w = check_w(w, self.m + 1)
