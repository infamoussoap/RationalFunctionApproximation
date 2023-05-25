import numpy as np
import warnings

from ..RationalApproximator import RationalApproximator
from ..validation_checks import check_X_in_range


class AAAApproximator(RationalApproximator):
    def __init__(self, n, m=None):
        if m is not None and m != n:
            warnings.warn('m is an ignored parameter, as '
                          'AAA assumes the numerator and denominator are of the same degree')
        self.n = n
        self.m = n

        self.w = None

        self.target_at_poles = None
        self.poles = None

    def fit(self, X, y):
        """ Fit the AAA Aproximator

            Parameters
            ----------
            X : np.ndarray
                (n, )
            y : np.ndarray
                (n, )
        """
        self._reset_params()

        check_X_in_range(X, 0, 1)

        support = np.ones(len(y)).astype(bool)
        for i in range(self.n):
            out = AAAApproximator._fit(X, y, self.w, support)
            self.w, support, self.target_at_poles, self.poles = out

        return self

    def _reset_params(self):
        self.w = None
        self.target_at_poles = None
        self.poles = None

    @staticmethod
    def _fit(X, y, w, support):
        """ The assumption is that X has unique values """
        if w is None:
            R = np.mean(y)
        else:
            R = AAAApproximator.eval_aaa(X[support], y[~support], w, X[~support])

        # This can be made faster
        j = np.argmax(abs(y[support] - R))
        true_j = np.arange(len(support))[support][j]

        support[true_j] = False

        A = (y[support][:, None] - y[~support][None, :]) \
            / (X[support][:, None] - X[~support][None, :])

        u, s, vh = np.linalg.svd(A)

        w = vh[-1]
        return w, support, y[~support], X[~support]

    def numerator(self, x):
        check_X_in_range(x, 0, 1)
        return AAAApproximator.eval_aaa_numerator(x, self.target_at_poles, self.w, self.poles)

    def denominator(self, x):
        check_X_in_range(x, 0, 1)
        return AAAApproximator.eval_aaa_denominator(x, self.w, self.poles)

    def __call__(self, x, tol=1e-10):
        check_X_in_range(x, 0, 1)
        return AAAApproximator.eval_aaa(x, self.target_at_poles, self.w, self.poles, tol=tol)

    @staticmethod
    def eval_aaa_numerator(x, target_at_poles, w, poles):
        check_X_in_range(x, 0, 1)
        numerator_matrix = target_at_poles[None, :] \
                           / (x[:, None] - poles[None, :])
        return numerator_matrix @ w

    @staticmethod
    def eval_aaa_denominator(x, w, poles):
        check_X_in_range(x, 0, 1)
        denominator_matrix = 1 / (x[:, None] - poles[None, :])
        return denominator_matrix @ w

    @staticmethod
    def eval_aaa(x, target_at_poles, w, poles, tol=1e-10):
        check_X_in_range(x, 0, 1)
        out = np.zeros_like(x)

        difference = abs(x[:, None] - poles[None, :])
        min_difference = np.min(difference, axis=1)
        mask = min_difference > tol

        if np.all(mask):
            return AAAApproximator.eval_aaa_numerator(x, target_at_poles, w, poles) \
                   / AAAApproximator.eval_aaa_denominator(x, w, poles)

        min_index = np.argmin(difference[~mask], axis=1)

        masked_x = x[mask]
        out[mask] = AAAApproximator.eval_aaa_numerator(masked_x, target_at_poles, w, poles) \
                    / AAAApproximator.eval_aaa_denominator(masked_x, w, poles)

        out[~mask] = target_at_poles[min_index]

        return out
