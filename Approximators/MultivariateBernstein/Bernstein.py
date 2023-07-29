import numpy as np

from ..Polynomials import MultivariateLegendrePolynomial, MultivariateBernsteinPolynomial
from ..validation_checks import check_X_in_range


class Bernstein:
    def __init__(self, n_vals, m_vals=None):
        """

            Parameters
            ----------
            m : int
                Degree of the denominator
            n : int
                Degree of the numerator
        """
        self.n_vals = n_vals
        self.m_vals = n_vals.copy() if m_vals is None else m_vals

        self.domain = [0, 1]

    def f(self, X, target_ys, w, grad=False):
        """

            Parameters
            ----------
            X : np.array
                Assumed to be of the shape (# datapoints, # variables)
            target_ys : list of np.ndarray
            w : np.ndarray
            grad : bool
        """
        check_X_in_range(X, 0, 1)

        evaluated_legendre = MultivariateLegendrePolynomial(self.n_vals, X)
        evaluated_legendre = evaluated_legendre.reshape(-1, len(X))

        denominator = self._denominator(X, w)

        if np.any(denominator == 0):
            return np.inf

        legendre_coefs = [self._compute_legendre_coef(denominator, y, evaluated_legendre) for y in target_ys]
        numerators = [coef @ evaluated_legendre for coef in legendre_coefs]

        difference = [y - numerator / denominator for y, numerator in zip(target_ys, numerators)]

        if grad:
            B = MultivariateBernsteinPolynomial(self.m_vals, X)
            B = B.reshape(-1, len(X))
            grads = [B @ (diff * (numerator / (denominator ** 2)))
                     for (numerator, diff, y) in zip(numerators, difference, target_ys)]
            return np.mean(grads, axis=0)

        return sum([np.mean(z ** 2) for z in difference])

    def _denominator(self, X, w):
        B = MultivariateBernsteinPolynomial(self.m_vals, X)
        B = B.reshape(-1, len(X))
        return w @ B

    def _numerator(self, X, legendre_coef):
        P = MultivariateLegendrePolynomial(self.n_vals, X)
        P = P.reshape(-1, len(X))
        return legendre_coef @ P

    @staticmethod
    def _compute_legendre_coef(denominator, y, evaluated_legendre):
        support = denominator > 0
        design_matrix = evaluated_legendre[:, support] / denominator[None, support]
        coef, *_ = np.linalg.lstsq(design_matrix.T, y[support], rcond=None)

        return coef
