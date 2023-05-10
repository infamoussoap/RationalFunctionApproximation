import numpy as np
from sklearn.linear_model import LinearRegression


class Chebyshev:
    def __init__(self, target_function, m, n=None, num_integration_points=100):
        """ m is the degree of the denominator
            n is the degree of the numerator
        """
        self.target_function = target_function

        self.m = m
        self.n = m if n is None else n

        self.integration_points = np.linspace(0, 1, num_integration_points)
        self.domain = [self.integration_points[0], self.integration_points[-1]]

        self.target_values = self.target_function(self.integration_points)

        self.dx = self.integration_points[1] - self.integration_points[0]

        x = self.integration_points

        self.numerator_matrix = np.vstack([x ** i for i in range(n + 1)])
        self.denominator_matrix = np.vstack([x ** i for i in range(1, m + 1)])

        self.numerator_coef = None
        self.denominator_coef = None

        self.X = np.vstack([self.numerator_matrix, -self.denominator_matrix * self.target_values]).T

        self.r = 0

    def f(self):
        return np.sum((self.target_values * self.denominator - self.numerator) ** 2)

    def search(self):
        model = LinearRegression(fit_intercept=False).fit(self.X, self.target_values - self.r)

        self.numerator_coef = model.coef_[:self.n + 1]
        self.denominator_coef = model.coef_[self.n + 1:]

        deviation = self.numerator / self.denominator - self.target_values
        self.r = -np.sign(deviation) * np.mean(abs(deviation))  # Not sure if there should be a negative here or not

    @property
    def denominator(self):
        return 1 + self.denominator_coef @ self.denominator_matrix

    @property
    def numerator(self):
        return self.numerator_coef @ self.numerator_matrix
