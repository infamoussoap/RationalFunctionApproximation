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

    def write_norms(self, n_iter, vectors, norms=None, header='\r'):
        assert isinstance(vectors, list), "Input is assumed to be a list of vectors"

        if norms is None:
            norms = [2, np.inf]

        s = f"{n_iter}"
        for norm in norms:
            if len(vectors) == 1:
                formatted_norms = f"{np.linalg.norm(vectors[0], ord=norm):.6e}"
            else:
                formatted_norms = "(" + ", ".join([f"{np.linalg.norm(x, ord=norm):.6e}" for x in vectors]) + ")"

            s += f": L-{norm} {formatted_norms}"
        self.write(s, header=header)
