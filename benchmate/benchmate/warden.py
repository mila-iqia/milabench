import logging
import os
import re
import signal
import subprocess
import time
import traceback
import warnings
import json
from collections import defaultdict
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
        if line:
            frags = line.split(",")
            info.append(ProcessInfo(**dict(zip(metrics, frags))))
    return info


def _default():
    return []



def _rocm_parse_processes():
    cmd = ["rocm-smi", f"--showpids", "--json"]
    output = subprocess.check_output(cmd, text=True)
    
    info = []

    try:
        data = json.loads(output)
    except json.decoder.JSONDecodeError:
        return info

    for key, data in data.get("system", {}).items():
        process_name, ngpu, vram, sdma, cu_occupancy = data.split(",")
        pid = key[3:]
        info.append(ProcessInfo(pid=pid, process_name=process_name, used_memory=vram))
    return info


backends = {
    "hpu": _hpu_parse_processes,
    "cuda": _cuda_parse_processes,
    "rocm": _rocm_parse_processes,
    # XPU
    "cpu": _default,
}


def safe_get_gpu_info():
    try:
        return get_gpu_info()
    except:
        traceback.print_exc()
        return {}


class BaseWarden:
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass

    def terminate(self):
        pass

    def kill(self):
        pass


@contextmanager
def gpu_warden(*args, enabled=True, **kwargs):
    cls = BaseWarden
    if enabled:
        cls = GPUProcessWarden

    with cls(*args, **kwargs) as warden:
        yield warden


@contextmanager
def pipe_warden(enabled=True):
    if not enabled:
        yield
        return

    fd_folder = f"/proc/{os.getpid()}/fd/" 

    before = set(os.listdir(fd_folder))

    yield

    after = set(os.listdir(fd_folder))

    new_fds = after - before

    if len(new_fds) > 0:
        syslog("detected {} new file descriptors after running the benchmark", len(new_fds))
        closed = 0

        for new_fd in new_fds:
            try:
                os.close(int(new_fd))
                closed += 1
            except:
                pass

        syslog("closed {}/{} fd after bench", closed, len(new_fds))


@contextmanager
def children_warden(enabled=True):
    if not enabled:
        yield
        return

    yield

    pid = os.getpid()

    def get_children():
        with open(f"/proc/{pid}/task/{pid}/children", "r") as f:
            return [int(c) for c in f.read().strip().split()]

    def wait_for_children():
        children = get_children()
        start_time = time.time()

        while children and time.time()  - start_time < 15:
            children = get_children()
            time.sleep(0)

        return children

    children = get_children()
    if children:
        syslog(f"{len(children)} still alive after end of benchmark")

    for child in children:
        os.kill(child, signal.SIGTERM)

    children = wait_for_children()
    if children:
        syslog(f"{len(children)} still alive after sigterm")

    for child in children:
        os.kill(child, signal.SIGKILL)

    children = wait_for_children()
    if children:
        syslog(f"{len(children)} still alive after sigkill")

    for child in children:
        os.kill(child, signal.SIGCONT)


class GPUProcessWarden(BaseWarden):
    """Ensure all the processes using the GPU are killed before & after the bench"""

    def __init__(self, kill_on_start=True, kill_on_end=True):
        self.gpus = safe_get_gpu_info()
        self.arch = self.gpus.get("arch", "cpu")
        self.fetch_fun = backends.get(self.arch, _default)
        self.kill_on_start = kill_on_start
        self.kill_on_end = kill_on_end
        self.dead_processes = []

    def __enter__(self):
        if self.kill_on_start:
            self.terminate(True)

        return self

    def __exit__(self, *args):
        if self.kill_on_end:
            self.kill(False)

        return None

    def fetch_processes(self):
        try:
            return self.fetch_fun()
        except:
            traceback.print_exc()
            return []

    def _kill(self, pid, signal):
        if pid in self.dead_processes:
            return False

        try:
            os.kill(int(pid), signal)
            return True
        except PermissionError:
            syslog("PermissionError: Could not kill process {0}", pid)
            self.dead_processes.append(pid)
            return False
        except ProcessLookupError:
            self.dead_processes.append(pid)
            return False

    def terminate(self, start=False):
        self.ensure_free(signal.SIGTERM, start)

    def kill(self, start=True):
        self.ensure_free(signal.SIGKILL, start)

    def ensure_free(self, signal_code, start):
        try:
            processes = self.fetch_processes()
            if len(processes) == 0:
                return

            sig = "KILL" if signal_code == signal.SIGKILL else "TERM"
            syslog("warden: {0}: Found {1} processes still using devices: {2}", 
                sig, len(processes), [p.pid for p in processes]
            )
 
            # Those processes might not be known by milabench
            # Depending on whose the parent the reaping might be happening later
            # and we cannot wait for the process to die
            # we could try something like os.waitpid but it might interfere with `SignalProtected`
            # for the processes we do know
            for process in processes:
                self._kill(process.pid, signal_code)
        except:
            traceback.print_exc()


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
    processes = list(processes)

    def kill(proc, sig):
        try:
            if getattr(proc, "did_setsid", False):
                os.killpg(os.getpgid(proc.pid), sig)
            else:
                os.kill(proc.pid, sig)
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

    k = 0
    start = - time.time()
    stats = defaultdict(int)
    while processes and (start + time.time()) < timeout:
        finished = []

        for i, proc in enumerate(processes):
            if wait(proc):
                kill(proc, signal.SIGKILL)
                stats["alive"] += 1
            else:
                stats["dead"] += 1
                finished.append(i)

        for i in reversed(finished):
            del processes[i]

        k += 1
    
    if stats["alive"] != 0:
        syslog("{0} processes needed to be killed, retried {1} times, waited {2} s", stats["alive"], k, elapsed)


@contextmanager
def process_cleaner(timeout=30, with_gpu_warden=True):
    """Delay signal handling until all the processes have been killed"""

    def kill_everything(processes):
        def _():
            destroy(*processes, timeout=timeout)

        return _

    with Protected() as signalhandler:
        # Makes sure we are not leaking pipes/fd
        with pipe_warden(enabled=True):
            # Makes sure we are the the only one using the GPUs
            with gpu_warden(enabled=with_gpu_warden) as warden:  # => SIGTERM all processes using GPUs
                # Makes sure all our children processes die after each benchmark
                with children_warden(enabled=True):
                    processes = []
        
                    # when a signal is received kill the known processes first
                    # then handle the signal
                    signalhandler.stop = kill_everything(processes)

                    try:                            # NOTE: we have not waited much between both signals
                        warden.kill()               # => SIGKILL all processes using GPUs

                        yield processes             # => Run milabench, spawning processes for the benches

                    finally:
                        warden.terminate()          # => SIGTERM all processes using GPUs

                        destroy(*processes, timeout=timeout) # => SIGTERM+SIGKILL milabench processes

                                                    # destroy waited 30s
                                                
                        # warden.__exit__           # => SIGKILL all  processes still using GPUs


def subreaper():
    # Make sure milabench children stays milabench children
    # and don't get re-parented
    #
    # So in the case of torchrun were it spawns one children per GPUs
    # if torchrun dies then the children get reparented to milabench instead of PID:1
    #
    # Thanks to this we can rely on `children_warden` to kill all milabench children
    #
    from ctypes import CDLL, c_int
    import ctypes.util
    libc_path = ctypes.util.find_library("c")

    if libc_path:
        libc = ctypes.CDLL(libc_path)
        PR_SET_CHILD_SUBREAPER = 36 
        libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)

subreaper()
