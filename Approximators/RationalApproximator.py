import abc

import numpy as np

from .WriteToScreen import WriterToScreen


DEFAULT_WRITER = WriterToScreen()


class RationalApproximator(abc.ABC):
    @abc.abstractmethod
    def __init__(*args, **kwargs):
        pass

    @abc.abstractmethod
    def numerator(self, x):
        pass

    @abc.abstractmethod
    def denominator(self, x):
        pass

    def __call__(self, x):
        return self.numerator(x) / self.denominator(x)

    def predict(self, x):
        return self.__call__(x)

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    def evaluate(self, X, y):
        predicted_y = self(X)

        if isinstance(predicted_y, np.ndarray):
            diffs = [predicted_y - y]
        else:
            diffs = [pred_y - true_y for pred_y, true_y in zip(predicted_y, y)]

        DEFAULT_WRITER.write_norms("Errors", diffs, norms=[2, np.inf])
        print()

