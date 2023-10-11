import numpy as np
import pandas as pd

from utils import get_approximators

import sys


FUNCTIONS = {'arctan': lambda x: np.arctan(50 * (x - 0.5)),
             'abs': lambda x: np.abs(2 * (x - 0.5)),
             'exp': lambda x: np.exp(-x) * np.sin(16 * x ** 2)}


if __name__ == '__main__':
    function_name = sys.argv[1]

    if function_name != 'wage':
        np.random.seed(sum([ord(x) for x in function_name]) % 10)
        x = np.random.rand(1000)
        y_true = FUNCTIONS[function_name](x)
        y = y_true + np.random.normal(0, 0.01 + x / 10, len(x))

    else:
        df = pd.read_csv("Wage.csv")
        x = df['age'].values
        y = df['wage'].values
        y_true = y.copy()

        x = (x - x.min()) / (x.max() - x.min())

    n_vals = np.arange(2, 20 + 1)

    columns = list(get_approximators(2).keys())

    mse_results = pd.DataFrame(0.0, index=n_vals, columns=columns)
    max_results = pd.DataFrame(0.0, index=n_vals, columns=columns)
    pole_counts = pd.DataFrame(-1.0, index=n_vals, columns=columns)

    for n in n_vals:
        approximators = get_approximators(n)

        for key, model in approximators.items():
            model.fit(x, y)

            y_pred = model(x)

            mse_results.loc[n, key] = np.mean((y_true - y_pred) ** 2)
            max_results.loc[n, key] = np.max(abs(y_true - y_pred))
            pole_counts.loc[n, key] = len(model.poles())

        mse_results.to_csv(f"csv_files/{function_name}_mse_error.csv")
        max_results.to_csv(f"csv_files/{function_name}_max_error.csv")
        pole_counts.to_csv(f"csv_files/{function_name}_poles.csv")

