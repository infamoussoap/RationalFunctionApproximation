import itertools

import numpy as np

from ..Polynomials import MultivariateBernsteinPolynomial, MultivariateLegendrePolynomial
from ..validation_checks import check_bernstein_w, check_X_in_range
from ..RationalApproximator import RationalApproximator

from ..utils import bernstein_to_legendre_matrix
from ..Polynomials import bernstein_edge_locations

from .. import ConvexHull


class MultivariateStepwiseBernstein(RationalApproximator):
    def __init__(self, n_vals, m_vals=None, numerator_smoothing_penalty=None,
                 max_projection_iter=100, max_fit_iter=10, max_hull_projection_iter=50, gamma=0.5):
        self.n_vals = n_vals
        self.m_vals = n_vals if m_vals is None else m_vals

        self.numerator_smoothing_penalty = numerator_smoothing_penalty

        self.w = None
        self.legendre_coef = None

        self.domain = [0, 1]

        self.max_projection_iter = max_projection_iter
        self.max_fit_iter = max_fit_iter
        self.max_hull_projection_iter = max_hull_projection_iter

        self.projection_n_iter_ = []
        self.fit_n_iter_ = None
        self.hull_projection_iter_ = []

        self.gamma = gamma

    def fit(self, X, y):
        weight = np.ones(len(X))

        self.fit_n_iter_ = 0
        while self.fit_n_iter_ < self.max_fit_iter:
            weight = np.clip(weight, 1e-6, None)

            requires_projection = self._fit(X, y, weight)
            if requires_projection:
                self._project_numerator_and_denominator(X, y, weight=weight)
            else:
                self.projection_n_iter_.append(0)
                self.hull_projection_iter_.append([0])

            self.fit_n_iter_ += 1
            weight = self.denominator(X)

        return self

    def _fit(self, X, y, weight):
        numerator_coef_legendre, denominator_coef_legendre = self._fit_as_legendre(X, y, weight)
        denominator_coef_bernstein = self._convert_legendre_coefficients_to_bernstein(denominator_coef_legendre,
                                                                                      self.m_vals)

        if np.all(denominator_coef_bernstein >= 0):
            c = np.sum(denominator_coef_bernstein)

            self.w = denominator_coef_bernstein / c
            self.legendre_coef = numerator_coef_legendre / c

            return False

        self.w = denominator_coef_bernstein
        self.legendre_coef = numerator_coef_legendre

        return True

    def _fit_as_legendre(self, X, y, weight):
        evaluated_numerator_legendre = MultivariateLegendrePolynomial(self.n_vals, X).reshape(-1, len(X))
        evaluated_denominator_legendre = MultivariateLegendrePolynomial(self.m_vals, X).reshape(-1, len(X))

        design_matrix = np.vstack([evaluated_numerator_legendre / weight[None, :],
                                   -y * evaluated_denominator_legendre[1:] / weight[None, :]])

        numerator_penalty = self.get_smoothing_penalty(self.n_vals).flatten()
        denominator_penalty = self.get_smoothing_penalty(self.m_vals).flatten()

        R = np.zeros_like(design_matrix)
        # R[:len(penalty), :len(penalty)] = np.diag(np.sqrt(penalty))
        R[:, :R.shape[0]] = np.diag(np.concatenate([np.sqrt(numerator_penalty), np.sqrt(denominator_penalty[1:])]))

        if self.numerator_smoothing_penalty is not None:
            temp = design_matrix + self.numerator_smoothing_penalty * R
        else:
            temp = design_matrix
        coef = np.linalg.inv(temp @ temp.T) @ design_matrix @ (y / weight)

        numerator_coef_legendre = coef[:np.prod(np.array(self.n_vals) + 1)]
        denominator_coef_legendre = np.concatenate([[1], coef[np.prod(np.array(self.n_vals) + 1):]])

        return numerator_coef_legendre, denominator_coef_legendre

    def _project_numerator_and_denominator(self, x, y, weight, stopping_tol=1e-6):
        w_old = np.zeros_like(self.w)

        count = 0
        hull_projections_niter = []
        while np.linalg.norm(w_old - self.w) > stopping_tol and \
                count < self.max_projection_iter:
            w_old[:] = self.w[:]

            hull_projections_niter.append(self._project_denominator(x, y, weight))
            self._project_numerator(x, y, weight)

            count += 1

        self.hull_projection_iter_.append(hull_projections_niter)
        self.projection_n_iter_.append(count)

    def reset(self, w=None):
        self.w = check_bernstein_w(w, int(np.prod([m + 1 for m in self.m_vals])))

    def _project_numerator(self, X, y, weight):
        evaluated_legendre = MultivariateLegendrePolynomial(self.n_vals, X).reshape(-1, len(X)) / weight[None, :]

        denominator = self.denominator(X)
        target = y * denominator / weight

        if self.numerator_smoothing_penalty is None:
            coef, *_ = np.linalg.lstsq(evaluated_legendre.T, y, rcond=None)

        else:
            coef_weight = np.diag(MultivariateStepwiseBernstein.get_smoothing_penalty(self.n_vals).flatten())
            coef = np.linalg.inv(evaluated_legendre @ evaluated_legendre.T
                                 + self.numerator_smoothing_penalty * coef_weight) @ evaluated_legendre @ target

        return coef

    def _project_denominator(self, X, y, weight):
        A = MultivariateBernsteinPolynomial(self.m_vals, X).reshape(-1, len(X)) * (y / weight)[None, :]
        target = self.numerator(X) / weight

        optimizer = ConvexHull.CauchySimplexHull(A, target)

        self.w = np.ones(len(self.w)) / len(self.w)
        w_old = np.zeros_like(self.w)
        for i in range(self.max_hull_projection_iter):
            if np.linalg.norm(w_old - self.w) < 1e-10:
                break
            w_old[:] = self.w[:]
            self.w = optimizer.search(self.w, gamma=self.gamma)

        return i

    @staticmethod
    def get_smoothing_penalty(n_vals):
        penalty = np.ones([n + 1 for n in n_vals])

        for i, n in enumerate(n_vals):
            target_shape = [1 if l != i else (n + 1) for l in range(len(n_vals))]

            temp_penalty = np.arange(n + 1) ** np.arange(n + 1)
            temp_penalty = temp_penalty.reshape(target_shape)

            penalty *= temp_penalty

        return penalty

    @staticmethod
    def _convert_legendre_coefficients_to_bernstein(coefs, n_vals):
        coefs = coefs.flatten()
        L_2_B_matrices = [np.linalg.inv(bernstein_to_legendre_matrix(n)) for n in n_vals]

        index_generator = itertools.product(*[np.arange(n + 1) for n in n_vals])

        out = np.zeros([n + 1 for n in n_vals])
        summand = np.ones_like(out)
        count = 0
        for index_tuple in index_generator:
            summand[:] = coefs[count]
            for i, (L_2_b, index) in enumerate(zip(L_2_B_matrices, index_tuple)):
                target_shape = [-1 if i == j else 1 for j in range(len(n_vals))]
                summand *= L_2_b[:, index].reshape(target_shape)

            out += summand
            count += 1

        return out.flatten()

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

        return self.legendre_coef @ MultivariateLegendrePolynomial(self.n_vals, x).reshape(-1, len(x))

    def __call__(self, x):
        check_X_in_range(x, 0, 1)

        return self.numerator(x) / self.denominator(x)

    def poles(self):
        """ Warning: This doesn't return the poles """
        edge_locations = bernstein_edge_locations(self.m_vals).flatten()

        edge_w = self.w[edge_locations]
        return edge_w[edge_w < 1e-10]
