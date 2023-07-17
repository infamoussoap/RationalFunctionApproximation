import sys
import numpy as np


class WriterToScreen:
    def __init__(self):
        self.max_string_length = 0

    def write(self, s, header='', end=''):
        if len(s) >= self.max_string_length:
            self.max_string_length = len(s)
            sys.stdout.write(header + s + end)
        else:
            buffer_length = self.max_string_length - len(s)
            sys.stdout.write(header + s + " " * buffer_length + end)

        sys.stdout.flush()

    def write_norms(self, n_iter, x, norms=None, header='\r'):
        if norms is None:
            norms = [2, np.inf]

        s = f"{n_iter}"
        for norm in norms:
            s += f": L-{norm} {np.linalg.norm(x, ord=norm):.6e}"
        self.write(s, header=header)
