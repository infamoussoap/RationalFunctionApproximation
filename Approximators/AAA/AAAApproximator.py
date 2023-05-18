import numpy as np

from ..RationalApproximator import RationalApproximator


class AAAApproximator(RationalApproximator):
    def __init__(self, n, evaluation_points=None):
        self.n = n

        self.evaluation_points = np.linspace(0, 1, 100) if evaluation_points is None else evaluation_points

        self.w = None
        self.support = np.ones(len(evaluation_points)).astype(bool)
        self.target_at_poles = None
        self.poles = None

    def fit(self, target_function):
        self._reset_params()

        F = target_function(self.evaluation_points)

        for i in range(self.n):
            out = AAAApproximator._fit(F, self.evaluation_points, self.w, self.support)
            self.w, self.support, self.target_at_poles, self.poles = out

    def _reset_params(self):
        self.w = None
        self.support = np.ones(len(self.support)).astype(bool)
        self.target_at_poles = None
        self.poles = None

    @staticmethod
    def _fit(F, evaluation_points, w, support):
        if w is None:
            R = np.mean(F)
        else:
            R = AAAApproximator.eval_aaa(evaluation_points[support], F[~support], w, evaluation_points[~support])

        j = np.argmax(abs(F[support] - R))
        true_j = np.arange(len(support))[support][j]

        support[true_j] = False

        A = (F[support][:, None] - F[~support][None, :]) \
            / (evaluation_points[support][:, None] - evaluation_points[~support][None, :])

        u, s, vh = np.linalg.svd(A)

        w = vh[-1]
        return w, support, F[~support], evaluation_points[~support]

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
