from abc import ABC, abstractmethod

from numpy.polynomial.legendre import Legendre

from .ArmijoSearch import ArmijoSearch

from .utils import BernsteinPolynomial, check_x


class Approximator(ArmijoSearch, ABC):
    def __init__(self):
        self.x = None
        self.m = None
        self.n = None

    @property
    def denominator(self):
        def f(eval_points):
            return self.w @ BernsteinPolynomial(self.m, eval_points)

        return f

    @property
    def numerator(self):
        return Legendre(self.legendre_coef, domain=[0, 1])

    def reset(self, x=None):
        self.x = check_x(x, self.m + 1, self.n + 1)

    @property
    @abstractmethod
    def w(self):
        pass

    @property
    @abstractmethod
    def legendre_coef(self):
        pass
