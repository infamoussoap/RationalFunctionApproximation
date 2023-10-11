import numpy as np
import pandas as pd

from utils import get_approximators

import sys


FUNCTIONS = {'arctan': lambda x: np.arctan(50 * (x - 0.5)),
             'abs': lambda x: np.abs(2 * (x - 0.5)),
             'exp': lambda x: np.sin(16 * x ** 2) * np.exp(-x)}


if __name__ == '__main__':
    function_name = sys.argv[1]

    x = np.linspace(0, 1, 1001)
    y = FUNCTIONS[function_name](x)

    n_vals = np.arange(2, 20 + 1)

    columns = list(get_approximators(2).keys())

    mse_results = pd.DataFrame(0.0, index=n_vals, columns=columns)
    max_results = pd.DataFrame(0.0, index=n_vals, columns=columns)
    pole_counts = pd.DataFrame(-1.0, index=n_vals, columns=columns)

    for n in n_vals:
        approximators = get_approximators(n)

        for key, model in approximators.items():
            try:
                model.fit(x, y)
            except:
                mse_results.loc[n, key] = None
                max_results.loc[n, key] = None
                pole_counts.loc[n, key] = None
            else:
                y_pred = model(x)

                mse_results.loc[n, key] = np.mean((y - y_pred) ** 2)
                max_results.loc[n, key] = np.max(abs(y - y_pred))

                poles = model.poles()
                if len(poles) == 0:
                    pole_counts.loc[n, key] = 0
                else:
                    real_poles = np.real(poles[np.isreal(poles)])
                    pole_counts.loc[n, key] = np.sum((0 <= real_poles) * (real_poles <= 1))

        mse_results.to_csv(f"csv_files/{function_name}_mse_error.csv")
        max_results.to_csv(f"csv_files/{function_name}_max_error.csv")
        pole_counts.to_csv(f"csv_files/{function_name}_pole_count.csv")

