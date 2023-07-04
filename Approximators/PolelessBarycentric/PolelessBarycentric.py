import numpy as np
import warnings

from ..RationalApproximator import RationalApproximator


class PolelessBarycentric(RationalApproximator):
    """ Implementation of the algorithm as described in
            Barycentric rational interpolation with no poles and high rates of approximation
            by Michael S. Floater and Kai Hormann
            DOI: 10.1007/s00211-007-0093-y
    """
    def __init__(self, n, m=None, d=3):
        if m is not None and m != n:
            warnings.warn('m is an ignored parameter, as '
                          'AAA assumes the numerator and denominator are of the same degree')

        assert 0 <= d <= n, "d must be less than or equal to n"

        self.n = n
        self.m = n

        self.d = d

        self.w = None

        self.poles = None
        self.target_at_poles = None

    def fit(self, x, y):
        self.w = None
        self.poles = None
        self.target_at_poles = None

        sorted_index = np.argsort(x)
        x = x[sorted_index]
        y = y[sorted_index]

        support = np.ones(len(x)).astype(bool)

        if len(x) == self.n:
            w, support, target_at_poles, poles = self._fit(x, y, self.w, ~support, self.n, self.d)

            self.w = w
            self.target_at_poles = target_at_poles
            self.poles = poles

            return self

        for i in range(self.d, self.n + 1):
            w, support, target_at_poles, poles = self._fit(x, y, self.w, support, i, self.d)

            self.w = w
            self.target_at_poles = target_at_poles
            self.poles = poles

        return self

    @staticmethod
    def _fit(x, y, w, support, n, d, tol=1e-10):
        if w is None:
            R = np.mean(y)

            sorted_index = np.argsort(abs(y - R))
            true_j = sorted_index[-d:]
        else:
            R = PolelessBarycentric.eval_barycentric(x[support], y[~support], w, x[~support], tol=tol)

            j = np.argmax(abs(y[support] - R))
            true_j = np.arange(len(support))[support][j]

        support[true_j] = False

        poles = x[~support]
        target_at_poles = y[~support]

        w = np.array([PolelessBarycentric._compute_w(k, n, d, poles) for k in range(0, n)])
        max_w = np.max(abs(w))
        if max_w > 1e-6:
            w = w / max_w

        return w, support, target_at_poles, poles

    @staticmethod
    def _compute_w(k, n, d, poles):
        min_index = max(0, k - d)
        max_index = min(n - d, k + 1)

        total = 0
        for i in range(min_index, max_index):
            sign = 1 if i % 2 == 0 else -1
            denominator = np.prod([poles[k] - poles[j] for j in range(i, i + d + 1)
                                   if j != k])

            total += sign / denominator
        return total

    def numerator(self, x):
        return PolelessBarycentric.eval_numerator(x, self.target_at_poles, self.w, self.poles)

    def denominator(self, x):
        return PolelessBarycentric.eval_denominator(x, self.w, self.poles)

    def __call__(self, x, tol=1e-10):
        return PolelessBarycentric.eval_barycentric(x, self.target_at_poles, self.w, self.poles, tol=tol)

    @staticmethod
    def eval_numerator(x, target_at_poles, w, poles):
        numerator_matrix = target_at_poles[None, :] / (x[:, None] - poles[None, :])
        return numerator_matrix @ w

    @staticmethod
    def eval_denominator(x, w, poles):
        denominator_matrix = 1 / (x[:, None] - poles[None, :])
        return denominator_matrix @ w

    @staticmethod
    def eval_barycentric(x, target_at_poles, w, poles, tol=1e-10):
        out = np.zeros_like(x)

        if np.all(w == 0):
            out[:] = np.mean(target_at_poles)
            return out

        difference = abs(x[:, None] - poles[None, :])
        min_difference = np.min(difference, axis=1)
        mask = min_difference > tol

        if np.all(mask):
            return PolelessBarycentric.eval_numerator(x, target_at_poles, w, poles) \
                   / PolelessBarycentric.eval_denominator(x, w, poles)

        min_index = np.argmin(difference[~mask], axis=1)

        masked_x = x[mask]
        out[mask] = PolelessBarycentric.eval_numerator(masked_x, target_at_poles, w, poles) \
                    / PolelessBarycentric.eval_denominator(masked_x, w, poles)

        out[~mask] = target_at_poles[min_index]

        return out
