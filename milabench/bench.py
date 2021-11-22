import time
from contextlib import contextmanager
from queue import Empty, Queue
from types import SimpleNamespace

from giving import give, given


class StopProgram(Exception):
    """Raise to stop the benchmark."""


class BenchmarkRunner:
    def __init__(self, fn, config, bridge, instruments):
        self.fn = fn
        self.config = SimpleNamespace(**config) if isinstance(config, dict) else config
        self.bridge = bridge or self._null_bridge
        self.instruments = instruments
        self._queue = Queue()

    def give(self, **data):
        data["#queued"] = time.time()
        self._queue.put(data)

    @contextmanager
    def _null_bridge(self, _, __):
        yield

    def __call__(self):
        try:
            with given() as gv:

                @gv.where("!#queued").subscribe
                def _(_):
                    while True:
                        try:
                            data = self._queue.get_nowait()
                            give(**data)
                        except Empty:
                            break

                with self.bridge(self, gv):
                    for instrument in self.instruments:
                        instrument(self, gv)
                    with give.wrap("run"):
                        self.fn()

        except StopProgram:
            pass
