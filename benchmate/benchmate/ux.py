from threading import Thread
from contextlib import contextmanager


def spinner():
    spinner = [["."] * 8] * 8
    for i in range(8):
        spinner[i][i] = "*"

    spinner = [
        "".join(spinner[i]) for i in range(8)
    ]

    i = 0
    while True:
        yield spinner[i % len(spinner)]
        i += 1


class Spinner(Thread):
    def __init__(self, msg, interval=1):
        super().__init__()
        self.running = True
        self.msg = msg
        self.interval = interval

    def run(self):
        import time
        start = time.time()
        s = spinner()

        while self.running:
            time.sleep(self.interval)
            print(f"\r{self.msg} {time.time() - start:10.2f} {next(s)}", end="")


@contextmanager
def long_action(msg, interval=1):
    t = Spinner(msg, interval)
    t.start()
 
    yield

    t.running = False
    t.join()