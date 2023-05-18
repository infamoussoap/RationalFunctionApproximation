import numpy as np

from functools import partial

from .Bernstein import Bernstein
from .BernsteinApproximator import BernsteinApproximator

from ..utils import check_bernstein_w, check_target_functions

from ..WriteToScreen import WriterToScreen
import warnings


class CauchySimplex(BernsteinApproximator, Bernstein):
    """ Rational function approximation using Legendre polynomials on the numerator and Bernstein polynomials
        on the denominator. We iteratively change the Bernstein coefficients using a Cauchy-Simplex scheme and
        the Legendre coefficients are found using a projection.

        Attributes
        ----------
        n : int
            The degree of the numerator
        m : int
            The degree of the denominator
        tol : float
            Tolerance for the zero set
        w : (m + 1, ) np.ndarray
            The coefficients for the Bernstein polynomials (denominator)
        _legendre_coef : (n + 1, ) np.ndarray
            The coefficients for the Legendre polynomials (numerator)
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
        BernsteinApproximator.__init__(self)
        Bernstein.__init__(self, n, m=m,
                           num_integration_points=num_integration_points, spacing=spacing)
        self.tol = tol

        self.w = None
        self._legendre_coef = None

        self.n_iter_ = None

        self._writer = WriterToScreen()

    def _update(self, x, d, step_size):
        """ Perform a step in the update direction

            Parameters
            ----------
            x : (m + 1, ) np.ndarray
                The starting point
            d : (m + 1, ) np.ndarray
                The descent direction
            step_size : float
                The step size to be taken

            Returns
            -------
            (m + 1, ) np.ndarray
                The point once the step has been taken
        """
        z = x - step_size * d
        z[x < self.tol] = 0

        return z / np.sum(z)

    def _search(self, target_functions, c1=1e-4, c2=0.5, line_search_iter=100, gamma=1):
        """ Perform a step using a line search

            Parameters
            ----------
            target_functions : list(callable)
                The functions to be fitted against. Must be able to take np.ndarray
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
            (m + 1, ) np.ndarray
                The new iteration point once the optimal step has been taken
        """
        f = partial(self.f, target_functions)

        grad = f(self.w, grad=True)
        d = self.w * (grad - grad @ self.w)

        max_step_size = self._max_step_size(self.w, grad, tol=self.tol) * gamma

        step_size = self.backtracking_armijo_line_search(f, self.w, d, max_step_size,
                                                         c1=c1, c2=c2, max_iter=line_search_iter)

        return self._update(self.w, d, step_size)

    def fit(self, target_functions, max_iter=100, stopping_tol=1e-6, w=None,
            c1=1e-4, c2=0.5, line_search_iter=100, gamma=1, verbose=False):
        """ Fit the rational polynomial coefficients to the target function

            Parameters
            ----------
            target_functions : {callable, list(callable)}
                The function(s) to be fitted against. Must be able to take np.ndarray
            max_iter : int, default=100
                The number of iterations to perform the optimization
            stopping_tol : float, default=1e-6
                The tolerance for the stopping criteria. If |w_prev - w_new| < stopping_tol
                then the iteration will stop
            w : (m + 1, ) np.ndarray, default=None
                The starting point for optimization. If None is given then it will default
                to np.ones(m + 1) / (m + 1).
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
        self.w = check_bernstein_w(w, self.m + 1)
        target_functions = check_target_functions(target_functions)

        if len(self.w) == 1:
            self._legendre_coef = [self._compute_legendre_coef(f, self.w) for f in target_functions]
            return self

        w_old = 1  # Needs to be large enough so the while loop starts
        self.n_iter_ = 0
        while self.n_iter_ < max_iter and np.linalg.norm(w_old - self.w) > stopping_tol:
            w_old = self.w.copy()

            self.w = self._search(target_functions, c1=c1, c2=c2, line_search_iter=line_search_iter, gamma=gamma)
            self._legendre_coef = [self._compute_legendre_coef(f, self.w) for f in target_functions]

            self.n_iter_ += 1

            if verbose:
                self._writer.write(f"{self.n_iter_}: {self.f(target_functions, self.w)}", header='\r')

        if verbose:
            print()

        if self.n_iter_ == max_iter:
            warnings.warn("Maximum number of iterations has been reached and convergence is not guaranteed. "
                          "Try increasing `max_iter` or increasing `stopping_tol`.")

        return self

    @staticmethod
    def _max_step_size(x, grad, tol=1e-10):
        """ Compute the maximum step size

            Parameters
            ----------
            x : (m + 1, ) np.ndarray
                A point in the probability simplex
            grad : (m + 1, ) np.ndarray
                Gradient at the point `x`
            tol : float, default=1e-10
                Tolerance for the zero set

            Returns
            -------
            float
                The maximum step size
        """
        support = x > tol

        diff = np.max(grad[support]) - x @ grad
        return 1 / diff if diff > 1e-6 else 1e6
