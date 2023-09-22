from abc import ABC
import itertools

import numpy as np

from .ArmijoSearch import ArmijoSearch

from ..Polynomials import MultivariateBernsteinPolynomial, MultivariateLegendrePolynomial
from ..validation_checks import check_bernstein_w, check_X_in_range
from ..RationalApproximator import RationalApproximator


class BernsteinApproximator(ArmijoSearch, RationalApproximator, ABC):
    def __init__(self, n_vals, m_vals=None, numerator_smoothing_penalty=None):
        self.n_vals = n_vals
        self.m_vals = n_vals if m_vals is None else m_vals

        self.numerator_smoothing_penalty = numerator_smoothing_penalty

        self.w = None
        self._legendre_coef = None

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

        legendre_coefs = [self._compute_legendre_coef(denominator, y, evaluated_legendre,
                                                      self.numerator_smoothing_penalty, self.n_vals)
                          for y in target_ys]
        numerators = [coef @ evaluated_legendre for coef in legendre_coefs]

        difference = [y - numerator / denominator for y, numerator in zip(target_ys, numerators)]

        if grad:
            B = MultivariateBernsteinPolynomial(self.m_vals, X)
            B = B.reshape(-1, len(X))
            grads = [B @ (diff * (numerator / (denominator ** 2)))
                     for (numerator, diff, y) in zip(numerators, difference, target_ys)]
            return np.mean(grads, axis=0)

        return sum([np.mean(z ** 2) for z in difference])

    def denominator(self, x):
        """ x assumed to be a numpy array of shape (# data points, # variables) """
        check_X_in_range(x, 0, 1)

        if len(self.w) == 1:
            return np.ones(len(x))

        return self._denominator(x, self.w)

    def _denominator(self, X, w):
        B = MultivariateBernsteinPolynomial(self.m_vals, X)
        B = B.reshape(-1, len(X))
        return w @ B

    def numerator(self, x):
        check_X_in_range(x, 0, 1)

        numerator_vals = self._eval_numerator(x)

        if len(numerator_vals) == 1:
            return numerator_vals[0]
        return numerator_vals

    def _eval_numerator(self, x):
        """ x assumed to be a numpy array of shape (# data points, # variables) """
        numerator_vals = [self._numerator(x, coef) for coef in self._legendre_coef]
        return numerator_vals

    def _numerator(self, X, legendre_coef):
        if len(legendre_coef) == 1:
            return np.ones(len(X)) * legendre_coef

        P = MultivariateLegendrePolynomial(self.n_vals, X)
        P = P.reshape(-1, len(X))
        return legendre_coef @ P

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

    @staticmethod
    def _compute_legendre_coef(denominator, y, evaluated_legendre, smoothing_penalty, n_vals=None):
        support = denominator > 0
        design_matrix = evaluated_legendre[:, support] / denominator[None, support]

        if smoothing_penalty is None:
            coef, *_ = np.linalg.lstsq(design_matrix.T, y[support], rcond=None)
        else:
            assert n_vals is not None, "Smoothing penalty requires n_vals to not be none"

            try:
                len(smoothing_penalty)
            except TypeError:
                smoothing_penalty = [smoothing_penalty] * 3
            else:
                assert len(smoothing_penalty) == len(n_vals), "Smoothing penalty must have the same length as numerator"

            coef_weight = BernsteinApproximator.get_smoothing_penalty(n_vals, smoothing_penalty).flatten()
            coef = np.linalg.inv(design_matrix @ design_matrix.T \
                                 + np.diag(coef_weight)) @ design_matrix @ y[support]

        return coef

    @staticmethod
    def get_smoothing_penalty(n_vals, gammas):
        index_generator = itertools.product(*[np.arange(n + 1) for n in n_vals])

        penalty = np.zeros([n + 1 for n in n_vals])
        for n_index in index_generator:

            total = 0
            for i in range(len(n_index)):
                log_summand = np.log(gammas[i]) - sum([np.log(2 * n + 1) for j, n in enumerate(n_index) if j != i])

                if n_index[i] > 0:
                    log_summand += 2 * n_index[i] * np.log(n_index[i])

                total += np.exp(log_summand)

            penalty[n_index] = total

        return penalty
