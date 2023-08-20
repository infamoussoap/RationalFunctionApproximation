import abc


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
