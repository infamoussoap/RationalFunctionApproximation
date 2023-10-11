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

    n_vals = [4, 5, 6, 7, 8, 9, 10]
    n_vals_grid = [(n, n) for n in n_vals]

    bernstein_dfs = [2 * (n + 1) ** 2 - 1 for n in n_vals]
    n_splines_grid = [int(np.ceil(np.sqrt(n))) for n in bernstein_dfs]

    n_trials = 200

    N = 1000
    
    approximators = get_approximators(2)
    # index = pd.MultiIndex.from_product([n_vals, np.arange(n_trials)], names=['n', 'trial'])
    index = np.arange(n_trials)
    max_error = pd.DataFrame(0.0, index=index, columns=list(approximators.keys()))
    mse_error = pd.DataFrame(0.0, index=index, columns=list(approximators.keys()))
    best_params = pd.DataFrame("", index=index, columns=['Spline lam', 'Spline n_splines',
                                                         'Bernstein smoothing', 'Bernstein n_vals'])

    lam_grid = [10 ** i for i in range(-6, 1 + 1)]
    smoothing_grid = [10 ** i for i in range(-6, 1 + 1)]
    
    np.random.seed(sum([ord(x) for x in function_name]) % 10)
    for trial_count in range(n_trials):
        x = np.random.rand(N, 2)
        y = f(x[:, 0], x[:, 1]) + np.random.normal(0, 0.1, len(x))

        approximators = get_approximators(2)

        for name, model in approximators.items():
            if name == 'Spline':
                model.gridsearch(x, y, progress=False, keep_best=True, return_scores=False, 
                                 lam=lam_grid, n_splines=n_splines_grid)
                y_pred = model.predict(x)

                best_params.loc[trial_count, 'Spline lam'] = model.terms[0].lam
                best_params.loc[trial_count, 'Spline n_splines'] = model.terms[0].n_splines
            else:
                model.gridsearch(x, y, return_scores=False, keep_best=True, 
                                 numerator_smoothing_penalty=smoothing_grid, 
                                 n_vals=n_vals_grid)
                y_pred = model(x)
                
                smoothing = model.numerator_smoothing_penalty
                best_params.loc[trial_count, 'Bernstein smoothing'] = smoothing
                best_params.loc[trial_count, 'Bernstein n_vals'] = model.n_vals

            diff = f(x[:, 0], x[:, 1]) - y_pred
            max_error.loc[trial_count, name] = np.max(abs(diff))
            mse_error.loc[trial_count, name] = np.mean(diff ** 2)
        
        max_error.to_csv(f'csv_files/{function_name}_max_error.csv')
        mse_error.to_csv(f'csv_files/{function_name}_mse_error.csv')
        best_params.to_csv(f'csv_files/{function_name}_best_params.csv')

    print(time.time() - start)

