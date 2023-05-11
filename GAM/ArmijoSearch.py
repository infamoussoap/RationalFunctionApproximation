from abc import ABC
from abc import abstractmethod


class ArmijoSearch(ABC):

    def backtracking_armijo_line_search(self, f, x, d, step_size, c1=1e-4, c2=0.5, max_iter=100):
        """ Returns the step_size that satisfies the Armijo condition

            Parameters
            ----------
            f : callable
                The function to be optimized. Expected to have signature f(x, grad=False)
            x : np.ndarray
                The current position of the search
            d : np.ndarray
                The descent direction
            step_size : float
                The initial step_size to try
            c1 : float
                The constant in the Armijo condition
            c2 : float
                The amount to decrease the step_size at each iteration
            max_iter : int
                The maximum number of steps to try
        """
        f0 = f(x)
        grad0 = f(x, grad=True)

        count = 0
        x_new = self._update(x, d, step_size)

        while (not armijo_condition(f, x, x_new, f_old=f0, grad_old=grad0, c1=c1)) and count < max_iter:
            step_size = step_size * c2
            x_new = self._update(x, d, step_size)

            count += 1

        if count == max_iter:
            if f0 < f(x_new):
                return 0

        return step_size

    @abstractmethod
    def _update(self, x, d, step_size):
        pass

    @abstractmethod
    def _search(self, x, step_size=1, c1=1e-4, c2=0.5, max_iter=100):
        pass


def armijo_condition(f, x_old, x_new, f_old=None, grad_old=None, c1=1e-4):
    """ Returns True if the Armijo condition is satisfied

        Parameters
        ----------
        f : callable
            The function to be optimized. Expected to have signature f(x, grad=False)
        x_old : np.ndarray
            The current position of the search
        x_new : np.ndarray
            The suggested position of the search
        c1 : float
            The constant in the Armijo condition
        f_old : float, optional
            The function evaluated at x_old
        grad_old : np.ndarray, optional
            The gradient of the function evaluated at x_old
    """

    f_old = f(x_old) if f_old is None else f_old
    grad_old = f(x_old, grad=True) if grad_old is None else grad_old

    return f(x_new) <= f_old + c1 * grad_old @ (x_new - x_old)