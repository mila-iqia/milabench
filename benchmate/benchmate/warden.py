import logging
import os
import re
import signal
import subprocess
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

from milabench.syslog import syslog
from voir.instruments.gpu import get_gpu_info

log = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    pid: int
    gpu_name: int = None
    type: str = None
    process_name: str = None
    memory: int = None
    used_memory: str = None
    gpu_bus_id: str = None
    gpu_serial: str = None
    gpu_uuid: str = None


def _hpu_parse_processes():
    output = subprocess.check_output(["hl-smi"], text=True)

    line_format = re.compile(
        r"\|(\s+)(?P<gpu_name>\d+)(\s+)(?P<pid>\d+)(\s+)(?P<type>\w+)(\s+)(?P<process_name>\w+)(\s+)(?P<memory>\d+)((?P<used_memory>\w+))(\s+)"
    )

    info = []
    for line in output.split("\n"):
        if match := line_format.match(line):
            info.append(ProcessInfo(**match.groupdict()))

    return info


def _cuda_parse_processes():
    metrics = [
        "pid",
        "gpu_name",
        "gpu_bus_id",
        "gpu_serial",
        "gpu_uuid",
        "process_name",
        "used_memory",
    ]
    query = ",".join(metrics)
    cmd = ["nvidia-smi", f"--query-compute-apps={query}", "--format=csv,noheader"]
    output = subprocess.check_output(cmd, text=True)

    info = []
    for line in output.split("\n"):
        frags = line.splti(",")
        info.append(ProcessInfo(**dict(*zip(metrics, frags))))
    return info


def _default():
    return []


backends = {
    "hpu": _hpu_parse_processes,
    "cuda": _cuda_parse_processes,
    # ROCM
    # XPU
    "cpu": _default,
}


class GPUProcessWarden:
    """Ensure all the process using the GPU are killed before & after the bench"""

    def __init__(self, kill_on_start=True, kill_on_end=True):
        self.gpus = get_gpu_info()
        self.arch = self.gpus["arch"]
        self.fetch_fun = backends.get(self.arch, _default)
        self.kill_on_start = kill_on_start
        self.kill_on_end = kill_on_end
        self.dead_processes = []

    def __enter__(self):
        if self.kill_on_start:
            self.ensure_free()

        return self

    def __exit__(self, *args):
        if self.kill_on_end:
            self.ensure_free()

        return None

    def fetch_processes(self):
        try:
            return self.fetch_fun()
        except:
            traceback.print_exc()
            return []

    def kill(self, pid, signal):
        if pid in self.dead_processes:
            return

        try:
            os.kill(pid, signal)
        except ProcessLookupError:
            self.dead_processes.append(pid)

    def ensure_free(self):
        processes = self.fetch_processes()
        if len(processes) == 0:
            return

        syslog("Found {0} still using devices after bench ended", len(processes))

        for process in processes:
            self.kill(process.pid, signal.SIGTERM)
        
        for process in processes:
            self.kill(process.pid, signal.SIGKILL)


class Protected:
    """Prevent a signal to be raised during the execution of some code"""

    def __init__(self):
        self.signal_received = False
        self.handlers = {}
        self.start = 0
        self.delayed = 0
        self.signal_installed = False

    def __enter__(self):
        """Override the signal handlers with our delayed handler"""
        self.signal_received = False

        try:
            self.handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self.handler)
            self.handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self.handler)
            self.signal_installed = True

        except ValueError:  # ValueError: signal only works in main thread
            warnings.warn(
                "SIGINT/SIGTERM protection hooks could not be installed because "
                "Runner is executing inside a thread/subprocess, results could get lost "
                "on interruptions"
            )

        return self

    def stop(self):
        pass

    def handler(self, sig, frame):
        """Register the received signal for later"""

        log.warning("Delaying signal %d to finish operations", sig)
        log.warning(
            "Press CTRL-C again to terminate the program now  (You may lose results)"
        )

        self.start = time.time()
        self.signal_received = (sig, frame)

        # if CTRL-C is pressed again the original handlers will handle it
        # and make the program stop
        self.restore_handlers()

    def restore_handlers(self):
        """Restore old signal handlers"""
        if not self.signal_installed:
            return

        signal.signal(signal.SIGINT, self.handlers[signal.SIGINT])
        signal.signal(signal.SIGTERM, self.handlers[signal.SIGTERM])

        self.signal_installed = False

        # Cleanup now
        self.stop()

    def maybe_stop(self):
        """Raise the delayed signal if any or restore the old signal handlers"""

        if not self.signal_received:
            self.restore_handlers()

        else:
            self.delayed = time.time() - self.start

            log.warning("Termination was delayed by %.4f s", self.delayed)
            handler = self.handlers[self.signal_received[0]]

            if callable(handler):
                handler(*self.signal_received)

    def __exit__(self, *args):
        self.maybe_stop()


def destroy(*processes, step=1, timeout=30):
    def kill(proc, signal):
        try:
            if getattr(proc, "did_setsid", False):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                os.kill(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    for proc in processes:
        kill(proc, signal.SIGTERM)

    # Wait a total amout of time, not per process
    elapsed = 0

    def wait(proc):
        nonlocal elapsed

        while (ret := proc.poll()) is None and elapsed < timeout:
            time.sleep(step)
            elapsed += step

        return ret is None

    for proc in processes:
        if wait(proc):
            kill(proc, signal.SIGKILL)


class SignalProtected(Protected):
    """Delay event handling until all the processes are killed"""

    def __init__(self):
        super().__init__()
        self.processes = []

    def add_process(self, *processes):
        self.processes.extend(processes)

    def stop(self):
        destroy(*self.processes)


@contextmanager
def process_cleaner():
    """Delay signal handling until all the processes have been killed"""

    with SignalProtected() as warden:
        yield warden
