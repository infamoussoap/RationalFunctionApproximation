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


def is_numpy_callable(f):
    x = np.linspace(0, 1, 10)
    try:
        f(x)
    except:
        return False
    return True


def check_target_functions(target_functions):
    if is_numpy_callable(target_functions):
        return [target_functions, ]

    try:
        valid_format = all([is_numpy_callable(f) for f in target_functions])
    except:
        pass
    else:
        if valid_format:
            return target_functions

    raise ValueError("target_functions is of invalid format. It must either be a callable that accepts Numpy "
                     "arrays or a list of callables that accepts Numpy arrays.")


def check_target_ys(target_ys):
    if isinstance(target_ys, np.ndarray):
        target_ys = [target_ys, ]

    if isinstance(target_ys, list):
        if all([is_numpy_array(y, dim=1) for y in target_ys]):
            return target_ys

    raise ValueError("Input expected to be either a 1-D np.ndarray or a list of 1-D np.ndarray")


def is_numpy_array(val, dim=1):
    if isinstance(val, np.ndarray) and len(val.shape) == dim:
        return True
    return False
