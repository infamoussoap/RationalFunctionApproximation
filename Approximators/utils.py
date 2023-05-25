import numpy as np
from scipy.special import gammaln, eval_legendre


def LegendrePolynomial(n, x):
    return np.vstack([eval_legendre(i, 2 * x - 1) for i in range(n + 1)])


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


def combination(n, k, as_log=False):
    if as_log:
        return gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)

    return np.exp(gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1))


def bernstein_to_legendre_matrix(n):
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
