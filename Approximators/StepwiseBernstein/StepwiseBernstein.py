import numpy as np

from ..RationalApproximator import RationalApproximator
from ..Polynomials import LegendrePolynomial, BernsteinPolynomial
from ..validation_checks import check_X_in_range, check_target_ys

from ..utils import bernstein_to_legendre_matrix

from . import ConvexHull


class StepwiseBernstein(RationalApproximator):
    def __init__(self, n, m, max_projection_iter=100, max_fit_iter=10, max_hull_projection_iter=50):
        self.n = n
        self.m = m

        self._legendre_coef = None
        self.w = np.ones(m + 1) / (m + 1)

        self.sol = None

        self.max_projection_iter = max_projection_iter
        self.max_fit_iter = max_fit_iter
        self.max_hull_projection_iter = max_hull_projection_iter

        self.projection_n_iter = 0
        self.fit_n_iter_ = 0

    def fit(self, x, target_ys):
        weight = np.ones_like(x)

        target_ys = check_target_ys(target_ys)

        self.fit_n_iter_ = 0
        while self.fit_n_iter_ < self.max_fit_iter:
            requires_projection = self._fit(x, target_ys, weight)
            if requires_projection:
                self._project_numerator_and_denominator(x, target_ys, weight=weight)

            self.fit_n_iter_ += 1
            weight = self.denominator(x)

        return self

    def _project_numerator_and_denominator(self, x, y, weight, stopping_tol=1e-6):
        w_old = np.ones_like(self.w)

        self.projection_n_iter = 0
        while np.linalg.norm(w_old - self.w) > stopping_tol and \
                self.projection_n_iter < self.max_projection_iter:

            w_old[:] = self.w[:]
            legendre_coef_old = self.legendre_coef[:]

            self._project_denominator(x, y, weight)
            self._project_numerator(x, y, weight)

            self.projection_n_iter += 1

    def _fit(self, x, target_ys, weight):
        unnormalized_legendre_coef, w_as_legendre = self._fit_as_legendre(x, target_ys, weight)

        B_2_M = bernstein_to_legendre_matrix(self.m)
        w_as_bernstein = np.linalg.inv(B_2_M) @ w_as_legendre

        if np.all(w_as_bernstein > 0):
            c = np.sum(w_as_bernstein)

            self.w = w_as_bernstein / c
            self._legendre_coef = [coef / c for coef in unnormalized_legendre_coef]

            return False

        self.w = w_as_bernstein
        self._legendre_coef = unnormalized_legendre_coef

        return True

    def _fit_as_legendre(self, X, target_ys, weight):
        P = LegendrePolynomial(self.n, X)
        Q = LegendrePolynomial(self.m, X)[1:]  # Ignoring the first Legendre polynomial

        A_list = []
        for i, y in enumerate(target_ys):
            A_list.append(np.hstack([P.T if i == j else np.zeros_like(P.T) for j in range(len(target_ys))] \
                                    + [-(y * Q).T]) / weight[:, None])

        A = np.vstack(A_list)
        coef, *_ = np.linalg.lstsq(A, np.hstack([y / weight for y in target_ys]), rcond=None)

        unnormalized_legendre_coefs = [coef[i * (self.n + 1): (i + 1) * (self.n + 1)] for i in range(len(target_ys))]
        w_as_legendre = np.hstack([[1], coef[-self.m:]])

        return unnormalized_legendre_coefs, w_as_legendre

    def _project_numerator(self, x, target_ys, weight):
        P = LegendrePolynomial(self.n, x)

        A_list = []
        for i, y in enumerate(target_ys):
            A_list.append(np.hstack([P.T if i == j else np.zeros_like(P.T) for j in range(len(target_ys))]) \
                          / weight[:, None])

        denominator = self.denominator(x)

        A = np.vstack(A_list)
        target = np.hstack([y * denominator / weight for y in target_ys])

        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        self._legendre_coef = [coef[i * (self.n + 1): (i + 1) * (self.n + 1)] for i in range(len(target_ys))]

    def _project_denominator(self, x, target_ys, weight):
        B = BernsteinPolynomial(self.m, x)

        A = np.hstack([y * B / weight for y in target_ys])
        target = np.hstack([num / weight for num in self._eval_numerator(x)])

        optimizer = ConvexHull.CauchySimplexHull(A, target)

        self.w = np.ones(self.m + 1) / (self.m + 1)
        w_old = np.zeros_like(self.w)
        for i in range(self.max_hull_projection_iter):
            if np.linalg.norm(w_old - self.w) < 1e-6:
                break
            w_old[:] = self.w[:]
            self.w = optimizer.search(self.w)

    def denominator(self, x):
        return self.w @ BernsteinPolynomial(self.m, x)

    def __call__(self, x):
        numerator = self.numerator(x)
        denominator = self.denominator(x)

        if isinstance(numerator, np.ndarray):
            return numerator / denominator
        else:
            return [num / denominator for num in numerator]

    def poles(self):
        """ Returns the poles inside [0, 1] """
        roots = []

        if self.w[0] == 0:
            roots.append(0)

        if self.w[-1] == 0:
            roots.append(1)

        return np.array(roots)

    def numerator(self, x):
        numerator_vals = self._eval_numerator(x)

        if len(numerator_vals) == 1:
            return numerator_vals[0]
        return numerator_vals

    def _eval_numerator(self, x):
        numerator_vals = [self._numerator(x, coef) for coef in self._legendre_coef]
        return numerator_vals

    def _numerator(self, X, legendre_coef):
        if len(legendre_coef) == 1:
            return np.ones_like(X)

        P = LegendrePolynomial(self.n, X)
        return legendre_coef @ P

    @property
    def legendre_coef(self):
        if len(self._legendre_coef) == 1:
            return self._legendre_coef[0]
        return self._legendre_coef
