import numpy as np

from functools import partial

from .BernsteinLegendreApproximator import BernsteinLegendreApproximator
from .BernsteinLegendre import BernsteinLegendre

from ..utils import check_bernstein_legendre_x

from ..WriteToScreen import WriterToScreen
import warnings


class EGD(BernsteinLegendreApproximator, BernsteinLegendre):
    """ Rational function approximation using Legendre polynomials on the numerator and Bernstein polynomials
        on the denominator. We iteratively change the Bernstein coefficients and the Legendre coefficients using
        an EGD for the Bernstein coefficients and normal descent for Legendre coefficients.

        Attributes
        ----------
        n : int
            The degree of the numerator
        m : int
            The degree of the denominator
        x : (n + m + 2, ) np.ndarray
            Weights for the denominator concatenated with the weights for the numerator
        tol : float (ignored)
            Tolerance for the zero set
        n_iter_ : int
            Number of iterations run by the coordinate descent solver to reach the specified tolerance
        _writer : WriterToScreen
            Used to write to screen for verbose
    """
    def __init__(self, n, m=None, tol=1e-10, evaluation_points=None):
        """ Initialize EGD Optimizer

            Parameters
            ----------
            n : int
                The degree of the numerator
            m : int, default=None
                The degree of the denominator, if not given it will default to n
            evaluation_points : None or np.ndarray, default=None
                Locations to evaluate the integral. Must be values between 0 and 1
            tol : float, default=1e-10
                Tolerance for the zero set
        """
        BernsteinLegendreApproximator.__init__(self)
        BernsteinLegendre.__init__(self, n, m=m, evaluation_points=evaluation_points)
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

        w = w * np.exp(-step_size * dw_dt)
        w = w / np.sum(w)

        c = c - step_size * dL_dc

        return np.concatenate([w, c])

    def _search(self, target_function, c1=1e-4, c2=0.5, line_search_iter=100, step_size=1):
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
            step_size : float, default=1
                The maximum candidate step size for the line search

            Returns
            -------
            (n + m + 2, ) np.ndarray
                The new iteration point once the optimal step has been taken
        """
        f = partial(self.f, target_function)
        grad = f(self.x, grad=True)

        step_size = self.backtracking_armijo_line_search(f, self.x, grad, step_size,
                                                         c1=c1, c2=c2, max_iter=line_search_iter)

        return self._update(self.x, grad, step_size)

    def fit(self, target_function, max_iter=100, stopping_tol=1e-6, x=None,
            c1=1e-4, c2=0.5, line_search_iter=100, step_size=1, verbose=False):
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
                to np.concatenate((np.ones(n + 1) / (n + 1), np.ones(m  + 1))).
            c1 : float, default=1e-4
                Parameter for the armijo line search
            c2 : float, default=0.5
                Parameter for the armijo line search
            line_search_iter : int, default=100
                Number of iterations for the line search
            step_size : float, default=1
                The maximum candidate step size for the line search
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

            self.x = self._search(target_function, c1=c1, c2=c2, line_search_iter=line_search_iter, step_size=step_size)

            self.n_iter_ += 1

            if verbose:
                self._writer.write(f"{self.n_iter_}: {self.f(target_function, self.x)}", header='\r')

        if verbose:
            print()

        if self.n_iter_ == max_iter:
            warnings.warn("Maximum number of iterations has been reached and convergence is not guaranteed. "
                          "Try increasing `max_iter` or increasing `stopping_tol`.")

        return self

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
