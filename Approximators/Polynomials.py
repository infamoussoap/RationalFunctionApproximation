import numpy as np
from scipy.special import eval_legendre, eval_chebyt
from scipy.special import legendre, chebyt

from .utils import combination


def LegendrePolynomial(n, x, grad=False, d=1):
    if grad:
        return legendre_derivative(n, x, d=d)

    return np.vstack([eval_legendre(i, 2 * x - 1) for i in range(n + 1)])


def legendre_derivative(n, x, d=1):
    return np.array([legendre(i).deriv(d)(2 * x - 1) for i in range(n + 1)]) * 2


def ChebyshevPolynomial(n, x, grad=False):
    if grad:
        return chebyshev_derivative(n, x)

    return np.vstack([eval_chebyt(i, 2 * x - 1) for i in range(n + 1)])


def chebyshev_derivative(n, x):
    return np.array([chebyt(i).deriv()(2 * x - 1) for i in range(n + 1)])


def BernsteinPolynomial(n, x, grad=False):
    if grad:
        return bernstein_derivative(n, x)

    working_mask = (x != 0) * (x != 1)

    k = np.arange(0, n + 1)
    log_B = np.zeros((n + 1, len(x)))

    log_B[:, working_mask] += combination(n, k, as_log=True)[:, None]
    log_B[:, working_mask] += k[:, None] * np.log(x[None, working_mask])
    log_B[:, working_mask] += (n - k[:, None]) * np.log(1 - x[None, working_mask])

    B = np.exp(log_B) * working_mask[None, :]

    # Edge cases
    B[0, x == 0] = 1
    B[n, x == 1] = 1

    return B


def bernstein_derivative(n, x):
    log_derivative = np.zeros((n + 1, len(x)))

    a = np.arange(n + 1)

    working_mask = (0 < x) * (x < 1)

    log_derivative += combination(n, a, as_log=True)[:, None]

    log_derivative[:-1, working_mask] += (n - a[:-1, None] - 1) * np.log(1 - x[None, working_mask])
    log_derivative[1:, working_mask] += (a[1:, None] - 1) * np.log(x[None, working_mask])

    derivative = np.exp(log_derivative)

    derivative[:-2, x == 1] = 0
    derivative[2:, x == 0] = 0

    derivative[0, :] *= -n
    derivative[-1, :] *= n
    derivative[1:-1, :] *= (a[1:-1, None] - n * x[None, :])

    return derivative


def MultivariateBernsteinPolynomial(n_vals, x_vals):
    return MultivariatePolynomial(n_vals, x_vals, BernsteinPolynomial)


def MultivariateLegendrePolynomial(n_vals, x_vals):
    return MultivariatePolynomial(n_vals, x_vals, LegendrePolynomial)


def MultivariatePolynomial(n_vals, x_vals, polynomial):
    """ x_vals assumed to be np.ndarray or shape (# datapoints, # variables) """
    out = np.ones([n + 1 for n in n_vals] + [len(x_vals)])
    for i in range(len(n_vals)):
        P = polynomial(n_vals[i], x_vals[:, i])

        broadcast_shape = [n_vals[i] + 1 if j == i else 1 for j in range(len(n_vals))] + [-1]
        out *= P.reshape(broadcast_shape)

    return out
