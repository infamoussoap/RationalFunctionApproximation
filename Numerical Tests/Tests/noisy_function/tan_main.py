import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '../..')

import Approximators
from utils import get_approximators

Approximators.ignore_warnings()

random_seed = sum([ord(x) for x in 'arctan']) % 10
np.random.seed(random_seed)


f = lambda x: np.arctan(50 * (x - 0.5))


noise_level = sys.argv[1]

x = np.random.rand(1000)
y = f(x) + np.random.normal(loc=0, scale=float(noise_level), size=len(x))


approximators = get_approximators(5)
n_vals = np.arange(2, 40 + 1)

mse_errors = pd.DataFrame(0.0, index=n_vals, columns=list(approximators.keys()))
max_errors = pd.DataFrame(0.0, index=n_vals, columns=list(approximators.keys()))

print("STARTING")

import time
start = time.time()

for i, n in enumerate(n_vals):
	approximators = get_approximators(n)
	    
	for name, approximator in approximators.items():
		if approximator is not None:
			approximator.fit(x, y)

			diff = approximator(x) - y

			mse_errors.loc[n, name] = np.mean(diff ** 2)
			max_errors.loc[n, name] = np.max(abs(diff))
		else:
			mse_errors.loc[n, name] = np.nan
			max_errors.loc[n, name] = np.nan

		mse_errors.to_csv(f'csv_files/tan_mse_{noise_level}.csv')
		max_errors.to_csv(f'csv_files/tan_max_{noise_level}.csv')

	print(f"Finihsed degree {n}")

end = time.time()
print(f"FINISHED {(end - start) / 60} min")
