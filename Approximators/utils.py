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
        return (np.cos(np.linspace(np.pi, 0, n_points)) + 1) / 2

    raise ValueError("Invalid spacing_type. Select one from ['linear', 'chebyshev']")


def check_bernstein_w(w, correct_length):
    if w is None:
        return np.ones(correct_length) / correct_length

    assert np.sum(w) == 1 and np.all(w > 0), "w must sum to one and have all positive indices"
    assert len(w) == correct_length, f"w must have length {correct_length}"

    return w


def check_bernstein_legendre_x(x, correct_w_length, correct_coef_length):
    if x is None:
        w = np.ones(correct_w_length) / correct_w_length
        coef = np.ones(correct_coef_length)
        return np.concatenate((w, coef))

    w = x[:correct_w_length]

    assert np.sum(w) == 1 and np.all(w > 0), "w must sum to one and have all positive indices"
    assert len(x) == correct_w_length + correct_coef_length, \
        f"x must have length {correct_w_length + correct_coef_length}"

    return x