from abc import ABC, abstractmethod

from numpy.polynomial.legendre import Legendre

from .ArmijoSearch import ArmijoSearch

from ..utils import BernsteinPolynomial
from ..validation_checks import check_bernstein_legendre_x
from ..RationalApproximator import RationalApproximator


class BernsteinLegendreApproximator(ArmijoSearch, RationalApproximator, ABC):
    def __init__(self):
        self.x = None
        self.m = None
        self.n = None

    def denominator(self, x):
        return self.w @ BernsteinPolynomial(self.m, x)

    def numerator(self, x):
        return Legendre(self.legendre_coef, domain=[0, 1])(x)

    def reset(self, x=None):
        self.x = check_bernstein_legendre_x(x, self.m + 1, self.n + 1)

    @property
    @abstractmethod
    def w(self):
        pass

    @property
    @abstractmethod
    def legendre_coef(self):
        pass

    def __call__(self, x):
        return self.numerator(x) / self.denominator(x)
