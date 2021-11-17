from contextlib import nullcontext
from types import SimpleNamespace

from giving import give, given


class StopProgram(Exception):
    """Raise to stop the benchmark."""


class BenchmarkRunner:
    def __init__(self, fn, config, bridge, instruments):
        self.fn = fn
        self.config = SimpleNamespace(**config) if isinstance(config, dict) else config
        self.bridge = bridge or nullcontext
        self.instruments = instruments

    def __call__(self):
        try:
            with given() as gv:
                with self.bridge(self, gv):
                    for instrument in self.instruments:
                        instrument(self, gv)
                    with give.wrap("run"):
                        self.fn()

        except StopProgram:
            pass
