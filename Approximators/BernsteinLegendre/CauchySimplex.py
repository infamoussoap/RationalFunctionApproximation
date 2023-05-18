import numpy as np

from functools import partial

from .BernsteinLegendre import BernsteinLegendre
from .BernsteinLegendreApproximator import BernsteinLegendreApproximator

from ..utils import check_bernstein_legendre_x

from ..WriteToScreen import WriterToScreen
import warnings


class CauchySimplex(BernsteinLegendreApproximator, BernsteinLegendre):
    """ Rational function approximation using Legendre polynomials on the numerator and Bernstein polynomials
        on the denominator. We iteratively change the Bernstein coefficients and the Legendre coefficients using
        a Cauchy-Simplex scheme for the Bernstein coefficients and normal descent for Legendre coefficients.

        Attributes
        ----------
        n : int
            The degree of the numerator
        m : int
            The degree of the denominator
        x : (n + m + 2, ) np.ndarray
            Weights for the denominator concatenated with the weights for the numerator
        tol : float
            Tolerance for the zero set
        n_iter_ : int
            Number of iterations run by the coordinate descent solver to reach the specified tolerance
        _writer : WriterToScreen
            Used to write to screen for verbose
    """
    def __init__(self, n, m=None, num_integration_points=100, tol=1e-10, spacing='linear'):
        """ Initialize Cauchy Simplex Optimizer

            Parameters
            ----------
            n : int
                The degree of the numerator
            m : int, default=None
                The degree of the denominator, if not given it will default to n
            num_integration_points : int, default=100
                The number of points to evaluate the integrand at
            tol : float, default=1e-10
                Tolerance for the zero set
            spacing : {'linear', 'chebyshev'} or np.ndarray, default='linear'
                How the discretization of the integral is to be made.
        """
        BernsteinLegendreApproximator.__init__(self)
        BernsteinLegendre.__init__(self, n, m=m,
                                   num_integration_points=num_integration_points, spacing=spacing)
        self.tol = tol
        self.x = None
        self.n_iter_ = None

        self._writer = WriterToScreen()

    def _update(self, x, d, step_size):
        """ Perform a step in the update direction

            Parameters
            ----------
            x : (n + m + 2, ) np.ndarray
                The starting point
            d : (n + m + 2, ) np.ndarray
                The descent direction
            step_size : float
                The step size to be taken

            Returns
            -------
            (n + m + 2, ) np.ndarray
                The point once the step has been taken
        """
        w, c = self.w, self.legendre_coef
        dw_dt, dL_dc = d[:self.m + 1], d[self.m + 1:]

        w = w - step_size * dw_dt
        w[w < self.tol] = 0
        w = w / np.sum(w)

        c = c - step_size * dL_dc

        return np.concatenate([w, c])

    def _search(self, target_function, c1=1e-4, c2=0.5, line_search_iter=100, gamma=1):
        """ Perform a step using a line search

            Parameters
            ----------
            target_function : callable
                The function to be fitted against. Must be able to take np.ndarray
            c1 : float, default=1e-4
                Parameter for the armijo line search
            c2 : float, default=0.5
                Parameter for the armijo line search
            line_search_iter : int, default=100
                Number of iterations for the line search
            gamma : float, default=1
                Expected to be a float between [0, 1]. It represents the percent of the maximum step size
                to be taken.

            Returns
            -------
            (n + m + 2, ) np.ndarray
                The new iteration point once the optimal step has been taken
        """
        f = partial(self.f, target_function)

        grad = f(self.x, grad=True)

        dL_dw = grad[:self.m + 1]

        max_step_size = self._max_step_size(self.w, dL_dw, tol=self.tol) * gamma

        grad[:self.m + 1] = self.w * (dL_dw - self.w @ dL_dw)
        step_size = self.backtracking_armijo_line_search(f, self.x, grad, max_step_size,
                                                         c1=c1, c2=c2, max_iter=line_search_iter)

        return self._update(self.x, grad, step_size)

    def fit(self, target_function, max_iter=100, stopping_tol=1e-6, x=None,
            c1=1e-4, c2=0.5, line_search_iter=100, gamma=1, verbose=False):
        """ Fit the rational polynomial coefficients to the target function

            Parameters
            ----------
            target_function : callable
                The function to be fitted against. Must be able to take np.ndarray
            max_iter : int, default=100
                The number of iterations to perform the optimization
            stopping_tol : float, default=1e-6
                The tolerance for the stopping criteria. If |w_prev - w_new| < stopping_tol
                then the iteration will stop
            x : (n + m + 2, ) np.ndarray, default=None
                The starting point for optimization. If None is given then it will default
                to np.concatenate((np.ones(m + 1) / (m + 1), np.ones(n  + 1))).
            c1 : float, default=1e-4
                Parameter for the armijo line search
            c2 : float, default=0.5
                Parameter for the armijo line search
            line_search_iter : int, default=100
                Number of iterations for the line search
            gamma : float, default=1
                Expected to be a float between [0, 1]. It represents the percent of the maximum step size
                to be taken.
            verbose : bool, default=False
                If set to true then the result of each step will be printed.

            Returns
            -------
            self : object
                Fitted rational polynomial
        """

        self.x = check_bernstein_legendre_x(x, self.m + 1, self.n + 1)

        w_old = 1  # Needs to be large enough so the while loop starts
        self.n_iter_ = 0
        while self.n_iter_ < max_iter and np.linalg.norm(w_old - self.w) > stopping_tol:
            w_old = self.w.copy()

            self.x = self._search(target_function, c1=c1, c2=c2, line_search_iter=line_search_iter, gamma=gamma)

            self.n_iter_ += 1

            if verbose:
                self._writer.write(f"{self.n_iter_}: {self.f(target_function, self.x)}", header='\r')

        if verbose:
            print()

        if self.n_iter_ == max_iter:
            warnings.warn("Maximum number of iterations has been reached and convergence is not guaranteed. "
                          "Try increasing `max_iter` or increasing `stopping_tol`.")

        return self

    @staticmethod
    def _max_step_size(w, dL_dw, tol=1e-10):
        """ Compute the maximum step size

            Parameters
            ----------
            w : (n, ) np.ndarray
                A point in the probability simplex
            dL_dw : (n, ) np.ndarray
                Gradient at the point `x`
            tol : float, default=1e-10
                Tolerance for the zero set

            Returns
            -------
            float
                The maximum step size
        """
        support = w > tol

        diff = np.max(dL_dw[support]) - w @ dL_dw
        return 1 / diff if diff > 1e-6 else 1e6

    @property
    def w(self):
        """ The weights for the denominator (Bernstein Polynomials)

            Returns
            -------
            (m + 1, ) np.ndarray
                The coefficients for the Bernstein polynomials
        """
        return self.x[:self.m + 1]

    @property
    def legendre_coef(self):
        """ The weights for the numerator (Legendre Polynomials)

            Returns
            -------
            (n + 1, ) np.ndarray
                The coefficients for the Legendre polynomials
        """
        return self.x[self.m + 1:]
