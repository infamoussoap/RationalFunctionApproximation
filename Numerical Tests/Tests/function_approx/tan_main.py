import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '../..')

import Approximators
from utils import get_approximators

Approximators.ignore_warnings()

f = lambda x: np.arctan(50 * (x - 0.5))

x = np.linspace(0, 1, 1001)
y = f(x)


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

		mse_errors.to_csv('csv_files/tan_mse.csv')
		max_errors.to_csv('csv_files/tan_max.csv')

	print(f"Finihsed degree {n}")

end = time.time()
print(f"FINISHED {(end - start) / 60} min")
