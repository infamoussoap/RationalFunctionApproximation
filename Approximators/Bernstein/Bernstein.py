import numpy as np
from numpy.polynomial.legendre import Legendre

from ..Polynomials import LegendrePolynomial, BernsteinPolynomial
from ..validation_checks import check_X_in_range


class Bernstein:
    def __init__(self, n, m=None):
        """

            Parameters
            ----------
            m : int
                Degree of the denominator
            n : int
                Degree of the numerator
        """
        self.n = n
        self.m = n if m is None else m

        self.domain = [0, 1]

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

        legendre_coefs = [self._compute_legendre_coef(denominator, y, evaluated_legendre) for y in target_ys]
        numerators = [coef @ evaluated_legendre for coef in legendre_coefs]

        difference = [y - numerator / denominator for y, numerator in zip(target_ys, numerators)]

        if grad:
            B = BernsteinPolynomial(self.m, X)
            grads = [B @ (diff * (numerator / (denominator ** 2)))
                     for (numerator, diff, y) in zip(numerators, difference, target_ys)]
            return np.mean(grads, axis=0)

        return sum([np.mean(z ** 2) for z in difference])

    def _denominator(self, X, w):
        B = BernsteinPolynomial(self.m, X)
        return w @ B

    def _numerator(self, X, legendre_coef):
        P = LegendrePolynomial(self.n, X)
        return legendre_coef @ P

    @staticmethod
    def _compute_legendre_coef(denominator, y, evaluated_legendre):
        support = denominator > 0

        design_matrix = evaluated_legendre[:, support] / denominator[None, support]
        try:
            coef, *_ = np.linalg.lstsq(design_matrix.T, y[support], rcond=None)
        except np.linalg.LinAlgError:
            print(np.sum(denominator == 0))
            raise ValueError("BREAK")

        return coef
