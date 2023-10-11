import numpy as np
import os


if __name__ == '__main__':
    working_files = [x for x in os.listdir() if 'mse' in x]

    for filename in working_files:
        arr = np.genfromtxt(filename, delimiter=",", dtype=str)
        arr = np.sqrt(arr[1:, 1:].astype(np.float64))
    
        mask = np.sum(arr, axis=1) > 0
        arr = arr[mask, :]
        
        mean = np.mean(arr, axis=0)
        median = np.median(arr, axis=0)
        print(f"{filename} ({len(arr)})")
        print(f"Mean = {mean[0]:.6f} {mean[1]:.6f}")
        print(f"Mode = {median[0]:.6f} {median[1]:.6f}")
        print("=" * 32)
