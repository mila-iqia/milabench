import os
import sys
import time
from contextlib import contextmanager
from queue import Empty, Queue
from types import ModuleType, SimpleNamespace

from giving import give, given

from .utils import exec_node, split_script


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


def make_runner(script, field, args, bridge, instruments):
    node, mainsection = split_script(script)
    mod = ModuleType("__main__")
    glb = vars(mod)
    glb["__file__"] = script
    sys.modules["__main__"] = mod
    code = compile(node, script, "exec")
    exec(code, glb, glb)
    glb["__main__"] = exec_node(script, mainsection, glb)

    sys.argv = [script, *args]

    return BenchmarkRunner(
        fn=glb[field],
        config={},
        bridge=bridge,
        instruments=instruments,
    )
