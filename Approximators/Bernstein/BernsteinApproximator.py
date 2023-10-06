from abc import ABC
import warnings

import numpy as np

from .ArmijoSearch import ArmijoSearch

from ..utils import bernstein_to_legendre_matrix, bernstein_to_chebyshev_matrix
from ..Polynomials import BernsteinPolynomial, LegendrePolynomial, ChebyshevPolynomial
from ..validation_checks import check_bernstein_w, check_X_in_range
from ..RationalApproximator import RationalApproximator

from .. import StepwiseBernstein, LinearizedBernstein


class BernsteinApproximator(ArmijoSearch, RationalApproximator, ABC):
    def __init__(self, n, m=None, numerator_smoothing_penalty=None, hot_start=False,
                 **hot_start_kwargs):
        """ Initialize base class for Bernstein Approximator

            Parameters
            ----------
            n : int
                Degree of the numerator
            m : int, default=None
                Degree of the denominator. If None then is set to the degree of the numerator
            numerator_smoothing_penalty : float, default=None
                Degree of smoothing to apply. If None then no smoothing is applied.
        """
        self.w = None

        self.n = n
        self.m = n if m is None else m

        self.numerator_smoothing_penalty = numerator_smoothing_penalty

        self.domain = [0, 1]

        self._legendre_coef = None

        self.hot_start = hot_start
        self.hot_start_kwargs = hot_start_kwargs

    def f(self, X, target_ys, w, grad=False):
        """ Loss function to be minimized with respect to w

            Parameters
            ----------
            X : (N,) np.ndarray
                The input, expected to be a vector with values inside [0, 1]
            target_ys : list[(N,) np.ndarray]
                The target(s) to be fitted
            w : np.ndarray
                Evaluates the loss function at this value of w
            grad : bool, default=False
                If True, returns df/dw

            Returns
            -------
            float or np.ndarray
                If grad=False then a float will be returned (the loss). Otherwise, df/dw will be returned,
                that is, the gradient.
        """
        check_X_in_range(X, 0, 1)

        evaluated_legendre = LegendrePolynomial(self.n, X, grad=False)

        denominator = self._denominator(X, w)

        if np.any(denominator == 0):
            return np.inf

        legendre_coefs = [self._compute_legendre_coef(denominator, y, evaluated_legendre,
                                                      self.numerator_smoothing_penalty, self.n)
                          for y in target_ys]
        numerators = [coef @ evaluated_legendre for coef in legendre_coefs]

        difference = [y - numerator / denominator for y, numerator in zip(target_ys, numerators)]

        if grad:
            B = BernsteinPolynomial(self.m, X)
            grads = [B @ (diff * (numerator / (denominator ** 2)))
                     for (numerator, diff, y) in zip(numerators, difference, target_ys)]
            return np.mean(grads, axis=0)

        return sum([np.mean(z ** 2) for z in difference])

    def denominator(self, x, grad=False):
        """ Evaluates the denominator

            Parameters
            ----------
            x : (N,) np.ndarray
                The input, expected to be a vector with values inside [0, 1]
            grad : bool, default=False
                If true then returns the gradient of the denominator with respect to x

            Returns
            -------
            np.ndarray
        """
        check_X_in_range(x, 0, 1)

        if len(self.w) == 1:
            return np.ones_like(x)

        return self._denominator(x, self.w, grad=grad)

    def _denominator(self, X, w, grad=False):
        return w @ BernsteinPolynomial(self.m, X, grad=grad)

    def numerator(self, x, grad=False):
        """ Evaluates the numerator

            Parameters
            ----------
            x : (N,) np.ndarray
                The input, expected to be a vector with values inside [0, 1]
            grad : bool, default=False
                If true then returns the gradient of the numerator with respect to x

            Returns
            -------
            np.ndarray or list[np.ndarray]
                Will return list[np.ndarray] if multiple functions are being approximated
        """
        check_X_in_range(x, 0, 1)

        numerator_vals = self._eval_numerator(x, grad=grad)

        if len(numerator_vals) == 1:
            return numerator_vals[0]
        return numerator_vals

    def _eval_numerator(self, x, grad=False):
        if grad:
            return self._numerator_grad(x)

        numerator_vals = [self._numerator(x, coef) for coef in self._legendre_coef]
        return numerator_vals

    def _numerator(self, X, legendre_coef):
        if len(legendre_coef) == 1:
            return np.ones_like(X)

        P = LegendrePolynomial(self.n, X)
        return legendre_coef @ P

    def _numerator_grad(self, x):
        numerator_grads = [c @ LegendrePolynomial(self.n, x, grad=True) for c in self._legendre_coef]
        return numerator_grads

    def reset(self, w=None):
        self.w = check_bernstein_w(w, self.m + 1)

    def __call__(self, x, grad=False):
        """ Returns the fitted rational function evaluated at x

            Parameters
            ----------
            x : (N,) np.ndarray
                The input, expected to be a vector with values inside [0, 1]
            grad : bool, default=False
                If True then will return the derivative of the rational function with respect to x

            Returns
            -------
            np.ndarray or list[np.ndarray]
                Will return list[np.ndarray] if multiple functions are being approximated
        """
        check_X_in_range(x, 0, 1)

        if grad:
            return self._grad(x)

        denominator = self.denominator(x)
        numerator = self.numerator(x)

        if isinstance(numerator, np.ndarray):
            return numerator / denominator
        else:
            return [num / denominator for num in numerator]

    def _grad(self, x):
        numerator_vals = self._eval_numerator(x, grad=False)
        numerator_grads = self._eval_numerator(x, grad=True)

        denominator_val = self.denominator(x, grad=False)
        denominator_grad = self.denominator(x, grad=True)

        if len(self.w) == 1:
            grads = numerator_grads
        else:
            grads = [(numerator_grad * denominator_val - numerator_val * denominator_grad) / (denominator_val ** 2)
                     for (numerator_val, numerator_grad) in zip(numerator_vals, numerator_grads)]

        if len(grads) == 1:
            return grads[0]
        return grads

    @property
    def legendre_coef(self):
        return self._legendre_coef[0] if len(self._legendre_coef) == 1 else self._legendre_coef

    def w_as_legendre_coef(self):
        """ Converts the denominator Bernstein coefficients into Legendre coefficients """
        M = bernstein_to_legendre_matrix(self.m)
        return M @ self.w

    def w_as_chebyshev_coef(self):
        """ Converts the denominator Bernstein coefficients into Chebyshev coefficients """
        M = bernstein_to_chebyshev_matrix(self.m)
        return M @ self.w

    def poles(self):
        """ Returns the poles inside [0, 1] """
        roots = []

        if self.w[0] == 0:
            roots.append(0)

        if self.w[-1] == 0:
            roots.append(1)

        return np.array(roots)

    @staticmethod
    def _compute_legendre_coef(denominator, y, evaluated_legendre, smoothing_penalty, n):
        support = denominator > 0
        design_matrix = evaluated_legendre[:, support] / denominator[None, support]

        if smoothing_penalty is None:
            coef, *_ = np.linalg.lstsq(design_matrix.T, y[support], rcond=None)
        else:
            coef_weight = BernsteinApproximator.get_smoothing_penalty(n)
            coef = np.linalg.inv(design_matrix @ design_matrix.T \
                                 + smoothing_penalty * np.diag(coef_weight)) @ design_matrix @ y[support]

        return coef

    @staticmethod
    def get_smoothing_penalty(n):
        return np.arange(n + 1) ** np.arange(n + 1)

    def extrapolate_predict(self, x, basis='Legendre'):
        numerator_val = self._eval_numerator(x, grad=False)

        if basis.lower() == 'legendre':
            denominator_coef = self.w_as_legendre_coef()
            P = LegendrePolynomial(self.m, x, False)
        elif basis.lower() == 'chebyshev':
            denominator_coef = self.w_as_chebyshev_coef()
            P = ChebyshevPolynomial(self.m, x, False)
        else:
            raise ValueError("Only accepted basis are 'legendre' or 'chebyshev'")
        denominator_val = denominator_coef @ P

        if len(numerator_val) == 1:
            return numerator_val[0] / denominator_val

        return [num / denominator_val for num in numerator_val]

    def get_hotstart_w(self, x, target_ys, max_projection_iter=100,
                       max_fit_iter=2, max_hull_projection_iter=1000):

        stepwise_has_poles, stepwise_w, stepwise_error = self._get_stepwise_hotstart(x, target_ys, max_projection_iter,
                                                                                     max_fit_iter,
                                                                                     max_hull_projection_iter)

        linearized_has_poles, linearized_w, linearized_error = self._get_linearized_hotstart(x, target_ys,
                                                                                             max_fit_iter)

        if (not stepwise_has_poles) and (not linearized_has_poles):
            return stepwise_w if stepwise_error < linearized_error else linearized_w
        elif (not stepwise_has_poles) or (not linearized_has_poles):
            return stepwise_w if linearized_has_poles else linearized_w

        warnings.warn("Hotstart returned results with poles and so will not be used.")
        w = np.ones(self.m + 1) / (self.m + 1)

        return w

    def _get_stepwise_hotstart(self, x, target_ys, max_projection_iter=100,
                               max_fit_iter=2, max_hull_projection_iter=1000):
        stepwise_approximator = StepwiseBernstein(self.n, self.m, max_projection_iter=max_projection_iter,
                                                  max_fit_iter=max_fit_iter,
                                                  max_hull_projection_iter=max_hull_projection_iter).fit(x, target_ys)

        error = self.f(x, target_ys, stepwise_approximator.w)
        return len(stepwise_approximator.poles()) > 0, stepwise_approximator.w, error

    def _get_linearized_hotstart(self, x, target_ys, max_iter):
        try:
            linearized_approximator = LinearizedBernstein(self.n, self.m, max_iter=max_iter).fit(x, target_ys)
        except:
            # Cvxopt will raise an error if the problem isn't well conditioned
            return True, None, None
        else:
            error = self.f(x, target_ys, linearized_approximator.w)
            return len(linearized_approximator.poles()) > 0, linearized_approximator.w, error
