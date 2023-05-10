import sys


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
