import json
import sys
import os
import time
from contextlib import contextmanager


from voir.instruments.utils import monitor as generic_monitor
from voir.smuggle import SmuggleWriter
from voir.tools import instrument_definition
from voir.instruments.cpu import cpu_monitor, process_monitor
from voir.instruments.gpu import gpu_monitor as gpu_monitor_fun, select_backend
from voir.instruments.io import io_monitor
from voir.instruments.network import network_monitor
from voir.instruments.monitor import monitor



@instrument_definition
def monitor_monogpu(ov, poll_interval=10, arch=None):
    return monitor(
        ov,
        poll_interval=poll_interval,
        gpudata=gpu_monitor_fun(),
        process=process_monitor(os.getpid()),
        worker_init=lambda: select_backend(arch, force=True),
    )


@instrument_definition
def monitor_node(ov, poll_interval=10, arch=None):
    return monitor(
        ov,
        poll_interval=poll_interval,
        gpudata=gpu_monitor_fun(),
        iodata=io_monitor(),
        netdata=network_monitor(),
        cpudata=cpu_monitor(),
        worker_init=lambda: select_backend(arch, force=True),
    )


def _smuggle_monitor(poll_interval=10, worker_init=None, **monitors):
    data_file = SmuggleWriter(sys.stdout)
    def mblog(data):
        nonlocal data_file
        if data_file is not None:
            print(json.dumps(data), file=data_file)

    def get():
        t = time.time()
        entries = []
        for k, v in monitors.items():
            values = {
                "task": "main",
                "time": t,
                k: v(),
            }
            entries.append(values)
        return entries

    def push(data):
        for entry in data:
            mblog(entry)

    mon = generic_monitor(
        poll_interval,
        get,
        push,
        process=False,
        worker_init=worker_init,
    )
    mon.start()
    
    return mblog, mon


@contextmanager
def smuggle_monitor(poll_interval=10, worker_init=None, enabled=True, **monitors):
    if enabled:
        # rank == 0
        mblog, mon = _smuggle_monitor(poll_interval, worker_init, **monitors)

        try:
            yield mblog
        finally:
            mon.stop()
        
    else:
        # rank > 0
        yield


def _monitors(monogpu=True):
    if monogpu:
        monitors = [
            ("gpudata", gpu_monitor_fun()),
            ("process", process_monitor(os.getpid())),
            ("worker_init", lambda: select_backend(None, True)),
        ]
    else:
        monitors = [
            ("gpudata", gpu_monitor_fun()),
            ("iodata", io_monitor()),
            ("netdata", network_monitor()),
            ("cpudata", cpu_monitor()),
            ("worker_init", lambda: select_backend(None, True)),
        ]
    return dict(monitors)


@contextmanager
def multigpu_monitor(*args, **kwargs):
    with smuggle_monitor(*args, **kwargs, **_monitors(False)) as log:
        yield log
        

@contextmanager
def monogpu_monitor(*args, **kwargs):
    with smuggle_monitor(*args, **kwargs, **_monitors(True)) as log:
        yield log


#
# Legacy compatibility
#
def setupvoir(monogpu=True, enabled=True):
    return _smuggle_monitor(
        poll_interval=3, 
        **_monitors(monogpu)
    )


def milabench_sys_monitor(monogpu=False):
    return setupvoir(monogpu)


