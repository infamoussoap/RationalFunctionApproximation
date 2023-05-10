import numpy as np
from scipy.special import gammaln


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
        return (np.cos(np.linspace(-np.pi, np.pi, n_points)) + 1) / 2

    raise ValueError("Invalid spacing_type. Select one from ['linear', 'chebyshev']")


def check_w(w, correct_length):
    assert np.sum(w) == 1 and np.all(w > 0), "w must sum to one and have all positive indices"
    assert len(w) == correct_length, f"w must have length {correct_length}"

    return w
