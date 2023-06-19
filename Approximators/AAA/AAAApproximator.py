import numpy as np
import scipy
import warnings

from ..RationalApproximator import RationalApproximator


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
        return AAAApproximator.eval_aaa_numerator(x, self.target_at_poles, self.w, self.poles)

    def denominator(self, x):
        return AAAApproximator.eval_aaa_denominator(x, self.w, self.poles)

    def __call__(self, x, tol=1e-10):
        return AAAApproximator.eval_aaa(x, self.target_at_poles, self.w, self.poles, tol=tol)

    @staticmethod
    def eval_aaa_numerator(x, target_at_poles, w, poles):
        numerator_matrix = target_at_poles[None, :] \
                           / (x[:, None] - poles[None, :])
        return numerator_matrix @ w

    @staticmethod
    def eval_aaa_denominator(x, w, poles):
        denominator_matrix = 1 / (x[:, None] - poles[None, :])
        return denominator_matrix @ w

    @staticmethod
    def eval_aaa(x, target_at_poles, w, poles, tol=1e-10):
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

    def pole_locations(self):
        N = len(self.w)

        top_row = np.concatenate([[0], self.w])
        bottom_row = np.hstack((np.ones((N, 1)), np.diag(self.poles)))

        E = np.vstack([top_row, bottom_row])

        B = np.eye(N + 1)
        B[0, 0] = 0

        V, *_ = scipy.linalg.eig(E, B, right=True)
        pol = V[~(np.isinf(V) + np.isnan(V))]

        return pol

    def residuals(self):
        dz = 1e-5 * np.exp(2j * np.pi * np.arange(1, 5) / 4)

        temp = self.pole_locations()[:, None] + dz[None, :]
        out = self(temp.flatten())
        return out.reshape(temp.shape).dot(dz) / 4
