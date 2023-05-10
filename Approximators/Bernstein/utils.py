import numpy as np
from scipy.special import gammaln


def BernsteinPolynomial(n, x):
    # assert x[0] == 0 and x[-1] == 1

    k = np.arange(0, n + 1)
    log_B = np.zeros((n + 1, len(x)))

    log_B += combination(n, k, as_log=True)[:, None]
    log_B[:, 1:] += k[:, None] * np.log(x[None, 1:])
    log_B[:, :-1] += (n - k[:, None]) * np.log(1 - x[None, :-1])

    log_B[1:, 0] = -np.inf
    log_B[:-1, -1] = -np.inf

    B = np.exp(log_B)
    return B


def combination(n, k, as_log=False):
    if as_log:
        return gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)

    return np.exp(gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1))


def safe_log(x):
    if not isinstance(x, np.ndarray):
        if x == 0:
            return 0
        return np.log(x)

    mask = x > 0

    out = np.zeros_like(x)
    out[mask] = np.log(x[mask])

    return out


def spacing(spacing_type='linear', n_points=100):
    if spacing_type.lower() == 'linear':
        return np.linspace(0, 1, n_points)
    elif spacing_type.lower() == 'chebyshev':
        return (np.cos(np.linspace(-np.pi, np.pi, n_points)) + 1) / 2

    raise ValueError("Invalid spacing_type. Select one from ['linear', 'chebyshev']")
