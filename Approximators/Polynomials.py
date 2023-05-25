import numpy as np
from scipy.special import eval_legendre, eval_chebyt

from .utils import combination


def LegendrePolynomial(n, x):
    return np.vstack([eval_legendre(i, 2 * x - 1) for i in range(n + 1)])


def ChebyshevPolynomial(n, x):
    return np.vstack([eval_chebyt(i, 2 * x - 1) for i in range(n + 1)])


def BernsteinPolynomial(n, x):
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
