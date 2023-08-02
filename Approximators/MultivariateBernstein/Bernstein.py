import numpy as np
from sklearn.linear_model import Ridge

from ..Polynomials import MultivariateLegendrePolynomial, MultivariateBernsteinPolynomial, LegendrePolynomial
from ..validation_checks import check_X_in_range


class Bernstein:
    def __init__(self, n_vals, m_vals=None, numerator_smoothing_penalty=None):
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

        self.numerator_smoothing_penalty = numerator_smoothing_penalty

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

    def _denominator(self, X, w):
        B = MultivariateBernsteinPolynomial(self.m_vals, X)
        B = B.reshape(-1, len(X))
        return w @ B

    def _numerator(self, X, legendre_coef):
        P = MultivariateLegendrePolynomial(self.n_vals, X)
        P = P.reshape(-1, len(X))
        return legendre_coef @ P

    @staticmethod
    def _compute_legendre_coef(denominator, y, evaluated_legendre, smoothing_penalty, n_vals=None):
        support = denominator > 0
        design_matrix = evaluated_legendre[:, support] / denominator[None, support]

        if smoothing_penalty is None:
            coef, *_ = np.linalg.lstsq(design_matrix.T, y[support], rcond=None)
        else:
            assert n_vals is not None, "Smoothing penalty requires n_vals to not be none"
            coef_weight = Bernstein.get_smoothing_penalty(n_vals).flatten()
            coef = np.linalg.inv(design_matrix @ design_matrix.T \
                                 + smoothing_penalty * np.diag(coef_weight)) @ design_matrix @ y[support]

        return coef

    @staticmethod
    def get_smoothing_penalty(n_vals, integration_points=100):
        x = np.linspace(0, 1, integration_points)
        penalty = np.zeros([n + 1 for n in n_vals])
        for i in range(len(n_vals)):
            for j in range(i + 1):
                P_temp = np.ones(penalty.shape + (len(x),))

                derivative_indices = {i, j}
                nonderivative_indices = set(range(len(n_vals))) - derivative_indices

                for k in nonderivative_indices:
                    target_shape = [1 if l != k else (n_vals[k] + 1) for l in range(len(n_vals))] + [len(x)]

                    P = LegendrePolynomial(n_vals[k], x, grad=False)
                    P_temp *= P.reshape(target_shape)

                if len(derivative_indices) == 1:
                    target_shape = [1 if l != i else (n_vals[i] + 1) for l in range(len(n_vals))] + [len(x)]

                    P = LegendrePolynomial(n_vals[i], x, grad=True, d=2)
                    P_temp *= P.reshape(target_shape)

                else:
                    for k in derivative_indices:
                        target_shape = [1 if l != k else (n_vals[k] + 1) for l in range(len(n_vals))] + [len(x)]

                        P = LegendrePolynomial(n_vals[k], x, grad=True, d=2)
                        P_temp *= P.reshape(target_shape)

                penalty += np.mean(P_temp ** 2, axis=-1)
        return penalty
