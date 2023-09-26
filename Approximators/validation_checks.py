import numpy as np


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


def check_numerator_degrees(n_vals, length):
    try:
        len(n_vals)
    except TypeError:
        assert isinstance(n_vals, (np.int64, np.int32, int)), "Numerator degree must be of type int or a list of int"
        return [n_vals] * length
    else:
        assert len(n_vals) == length, "Number of numerator degrees doesn't match up with expected number of degrees"

        for n in n_vals:
            assert isinstance(n, (np.int64, np.int32, int)), "Numerator degree must be of type int or a list of int"

    return n_vals


def is_numpy_array(val, dim=1):
    if isinstance(val, np.ndarray) and len(val.shape) == dim:
        return True
    return False


def check_X_in_range(X, min_val=0, max_val=1):
    assert np.all((min_val <= X) * (X <= max_val)), f"X must contain values between {min_val} and {max_val}"
