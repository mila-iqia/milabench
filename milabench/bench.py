import os
import sys
import time
from contextlib import ExitStack, contextmanager
from queue import Empty, Queue
from types import ModuleType, SimpleNamespace

from giving import give, given

from .utils import exec_node, split_script


class StopProgram(Exception):
    """Raise to stop the benchmark."""


class BenchmarkRunner:
    def __init__(self, fn, bridges, instruments):
        self.fn = fn
        self.bridges = bridges
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

                with ExitStack() as stack:
                    for fn in [*self.bridges, *self.instruments]:
                        ctx = fn(self, gv)
                        if hasattr(ctx, "__enter__"):
                            stack.enter_context(ctx)

                    with give.wrap("run"):
                        self.fn()

        except StopProgram:
            pass


def make_runner(script, field, args, bridges, instruments):
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
        bridges=bridges,
        instruments=instruments,
    )
