import numpy as np
import pandas as pd

import pygam

import sys
sys.path.insert(0, '../..')

from utils import get_approximators

import time

UNSCALED_FUNCTIONS = {'gaussian': lambda x, y: np.exp(-(x ** 2 + y ** 2)),
                      'radial': lambda x, y: np.sin(x ** 2 + y ** 2),
                      'sin': lambda x, y: np.sin(x) * np.sin(y),
                      'exp_sin': lambda x, y: np.exp(x) * np.sin(y)}

FUNCTIONS = {'gaussian': lambda x, y: UNSCALED_FUNCTIONS['gaussian'](4 * (x - 0.5), 4 * (y - 0.5)),
             'radial': lambda x, y: UNSCALED_FUNCTIONS['radial'](4 * (x - 0.5), 4 * (y - 0.5)),
             'sin': lambda x, y: UNSCALED_FUNCTIONS['sin'](10 * (x - 0.5), 10 * (y - 0.5)),
             'exp_sin': lambda x, y: UNSCALED_FUNCTIONS['exp_sin'](4 * (x - 0.5), 4 * (y - 0.5)),
             'wood': lambda x, z: 1.9 * (1.45 + np.exp(x) * np.sin(13 * (x - 0.6) ** 2)) * np.exp(-z) * np.sin(7 * z),
             'nonlinear_sin': lambda x, y: np.sin(8 * x ** 2) * np.sin(8 * y ** 2)}


if __name__ == '__main__':
    start = time.time()
    
    function_name = sys.argv[1]
    f = FUNCTIONS[function_name]

    n_vals = np.arange(2, 20 + 1)

    approximators = get_approximators(2)
    index = n_vals
    max_error = pd.DataFrame(0.0, index=index, columns=list(approximators.keys()))
    mse_error = pd.DataFrame(0.0, index=index, columns=list(approximators.keys()))
    best_params = pd.DataFrame("", index=index, columns=['Spline lam', 'Spline n_splines',
                                                         'Bernstein smoothing', 'Bernstein n_vals'])

    lam_grid = [10 ** i for i in range(-6, 1 + 1)] + [0]
    smoothing_grid = [10 ** i for i in range(-6, 1 + 1)] + [None]
    
    x_vals = np.linspace(0, 1, 51)
    X, Y = np.meshgrid(x_vals, x_vals)

    x = np.array([X.flatten(), Y.flatten()]).T
    y = f(x[:, 0], x[:, 1])

    x_valid = x.copy()

    for n in n_vals:
        dfs = 2 * (n + 1) ** 2 - 1
        n_splines = int(np.ceil(np.sqrt(dfs)))

        approximators = get_approximators(n)

        for name, model in approximators.items():
            if name == 'Spline':
                model.gridsearch(x, y, progress=False, keep_best=True, return_scores=False, 
                                 lam=lam_grid)
                y_pred = model.predict(x_valid)

                best_params.loc[n, 'Spline lam'] = model.terms[0].lam
                best_params.loc[n, 'Spline n_splines'] = model.terms[0].n_splines
            else:
                model.gridsearch(x, y, return_scores=False, keep_best=True, 
                                 numerator_smoothing_penalty=smoothing_grid)
                y_pred = model(x_valid)
                
                smoothing = model.numerator_smoothing_penalty
                best_params.loc[n, 'Bernstein smoothing'] = smoothing
                best_params.loc[n, 'Bernstein n_vals'] = model.n_vals

            diff = f(x_valid[:, 0], x_valid[:, 1]) - y_pred
            max_error.loc[n, name] = np.max(abs(diff))
            mse_error.loc[n, name] = np.mean(diff ** 2)
        
        max_error.to_csv(f'csv_files/{function_name}_max_error.csv')
        mse_error.to_csv(f'csv_files/{function_name}_mse_error.csv')
        best_params.to_csv(f'csv_files/{function_name}_best_params.csv')

    print(time.time() - start)

