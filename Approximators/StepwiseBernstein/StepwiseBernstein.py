import numpy as np

from ..RationalApproximator import RationalApproximator
from ..Polynomials import LegendrePolynomial, BernsteinPolynomial
from ..validation_checks import check_X_in_range

from ..utils import bernstein_to_legendre_matrix

from . import ConvexHull


class StepwiseBernstein(RationalApproximator):
    def __init__(self, n, m, max_projection_iter=100, max_fit_iter=10, max_hull_projection_iter=50):
        self.n = n
        self.m = m

        self.legendre_coef = None
        self.w = np.ones(m + 1) / (m + 1)

        self.sol = None

        self.max_projection_iter = max_projection_iter
        self.max_fit_iter = max_fit_iter
        self.max_hull_projection_iter = max_hull_projection_iter

        self.projection_n_iter = 0
        self.fit_n_iter_ = 0

    def fit(self, x, y):
        weight = np.ones_like(x)

        self.fit_n_iter_ = 0
        while self.fit_n_iter_ < self.max_fit_iter:
            requires_projection = self._fit(x, y, weight)
            if requires_projection:
                self._project_numerator_and_denominator(x, y, weight=weight)

            self.fit_n_iter_ += 1
            weight = self.denominator(x)

        return self

    def _project_numerator_and_denominator(self, x, y, weight, stopping_tol=1e-6):
        w_old = np.ones_like(self.w)
        legendre_coef_old = np.zeros_like(self.legendre_coef.copy())

        self.projection_n_iter = 0
        while (np.linalg.norm(w_old - self.w) > stopping_tol or
               np.linalg.norm(legendre_coef_old - self.legendre_coef) > stopping_tol) and \
                self.projection_n_iter < self.max_projection_iter:

            w_old[:] = self.w[:]
            legendre_coef_old = self.legendre_coef[:]

            self._project_denominator(x, y, weight)
            self._project_numerator(x, y, weight)

            self.projection_n_iter += 1

    def _fit(self, x, y, weight):
        unnormalized_legendre_coef, w_as_legendre = self._fit_as_legendre(x, y, weight)

        B_2_M = bernstein_to_legendre_matrix(self.m)
        w_as_bernstein = np.linalg.inv(B_2_M) @ w_as_legendre

        if np.all(w_as_bernstein > 0):
            c = np.sum(w_as_bernstein)

            self.w = w_as_bernstein / c
            self.legendre_coef = unnormalized_legendre_coef / c

            return False

        self.w = w_as_bernstein
        self.legendre_coef = unnormalized_legendre_coef

        return True

    def _fit_as_legendre(self, X, y, weight):
        P_legendre = LegendrePolynomial(self.n, X)
        Q_legendre = LegendrePolynomial(self.m, X)[1:]  # Ignoring the first Legendre polynomial

        support = weight != 0

        weighted_y = y[support] / weight[support]
        design_matrix = np.vstack([P_legendre[:, support] / weight[support], - weighted_y * Q_legendre[:, support]]).T

        coef, *_ = np.linalg.lstsq(design_matrix, weighted_y, rcond=None)

        a = coef[:self.n + 1]
        b = np.hstack([[1], coef[self.n + 1:]])

        return a, b

    def _project_numerator(self, x, y, weight):
        support = weight != 0

        P = LegendrePolynomial(self.n, x)
        self.legendre_coef, *_ = np.linalg.lstsq(P.T[support, :] / weight[support, None],
                                                 (y * self.denominator(x))[support] / weight[support],
                                                 rcond=None)

    def _project_denominator(self, x, y, weight):
        support = weight != 0

        B = BernsteinPolynomial(self.m, x)
        optimizer = ConvexHull.CauchySimplexHull((y * B)[:, support] / weight[None, support],
                                                 self.numerator(x)[support] / weight[support])

        self.w = np.ones(self.m + 1) / (self.m + 1)
        w_old = np.zeros_like(self.w)
        for i in range(self.max_hull_projection_iter):
            if np.linalg.norm(w_old - self.w) < 1e-6:
                break
            w_old[:] = self.w[:]
            self.w = optimizer.search(self.w)

    def numerator(self, x):
        return self.legendre_coef @ LegendrePolynomial(self.n, x)

    def denominator(self, x):
        return self.w @ BernsteinPolynomial(self.m, x)

    def __call__(self, x):
        return self.numerator(x) / self.denominator(x)

    def poles(self):
        """ Returns the poles inside [0, 1] """
        roots = []

        if self.w[0] == 0:
            roots.append(0)

        if self.w[-1] == 0:
            roots.append(1)

        return np.array(roots)
