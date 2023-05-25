import numpy as np
from scipy.special import gammaln


def combination(n, k, as_log=False):
    if as_log:
        return gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)

    return np.exp(gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1))


def bernstein_to_legendre_matrix(n):
    """ Formula taken from 'Legendre–Bernstein basis transformations' by Rida T. Farouki 1999 """
    i = np.arange(0, n + 1)
    j = np.arange(0, n + 1)
    k = np.arange(0, n + 1)

    log_summand_numerator = combination(j[None, :, None], i[:, None, None], as_log=True) * 2
    log_summand_denominator = combination(n + j[None, :, None], k[None, None, :] + i[:, None, None], as_log=True)

    mask = log_summand_denominator == -np.inf
    log_summand_denominator[mask] = 0

    log_summand = log_summand_numerator - log_summand_denominator
    log_summand[mask] = -np.inf

    sign = np.power(-np.ones((n + 1, n + 1)), j[None, :] + i[:, None])
    sum_ = np.sum(sign[:, :, None] * np.exp(log_summand), axis=0)
    transform_matrix = ((2 * j[:, None] + 1) / (n + j[:, None] + 1)) \
                       * combination(n * np.ones((n + 1, n + 1)), k[None, :]) * sum_

    return transform_matrix


def bernstein_to_chebyshev_matrix(n):
    i = np.arange(0, n + 1)[:, None, None]
    j = np.arange(0, n + 1)[None, :, None]
    k = np.arange(0, n + 1)[None, None, :]

    with np.errstate(invalid='ignore'):
        # The last combination line throws a warning error
        log_summand_numerator = combination(2 * j, 2 * i, as_log=True) \
                                + combination(2 * (k + i), k + i, as_log=True) \
                                + combination(2 * (n + j - k - i), (n + j - k - i), as_log=True)

    log_summand_denominator = combination(n + j, k + i, as_log=True)

    sign = np.power(-np.ones((n + 1, n + 1)), j - i)

    summand = sign * np.exp(log_summand_numerator - log_summand_denominator)
    summand[np.isnan(summand)] = 0

    sum_ = np.sum(summand, axis=0)

    j = np.arange(0, n + 1)[:, None]
    k = np.arange(0, n + 1)[None, :]

    delta = (j == 0).astype(int)
    log_scaling_factor = np.log((delta + 2 * (1 - delta))) - (n + j) * np.log(4) + combination(n, k, as_log=True)

    M = np.exp(log_scaling_factor) * sum_

    return M


def safe_log(x):
    if not isinstance(x, np.ndarray):
        if x == 0:
            return 0
        return np.log(x)

    mask = x > 0

    out = np.zeros_like(x)
    out[mask] = np.log(x[mask])

    return out


def spacing_grid(spacing='linear', n_points=100):
    if isinstance(spacing, np.ndarray):
        assert len(spacing) > 1, "Grid spacing must have at least length 2"

        spacing.sort()
        assert 0 <= spacing[0] and spacing[-1] <= 1, "The evaluation points of the integral must be between [0, 1]"

        return spacing

    if spacing.lower() == 'linear':
        return np.linspace(0, 1, n_points)
    elif spacing.lower() == 'chebyshev':
        return (np.cos(np.linspace(np.pi, 0, n_points)) + 1) / 2

    raise ValueError("Invalid spacing_type. Select one from ['linear', 'chebyshev']")
