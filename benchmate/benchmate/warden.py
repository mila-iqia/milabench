from dataclasses import dataclass
import re
import os
import subprocess
import traceback
import signal

from voir.instruments.gpu import get_gpu_info
from milabench.syslog import syslog

@dataclass
class ProcessInfo:
    gpu: int
    pid: int
    type: str
    process_name: str
    memory: int
    unit: str


def _hpu_parse_processes():
    output = subprocess.check_output(["hl-smi"], text=True)

    line_format = re.compile(
        r"\|(\s+)(?P<gpu>\d+)(\s+)(?P<pid>\d+)(\s+)(?P<type>\w+)(\s+)(?P<process_name>\w+)(\s+)(?P<memory>\d+)((?P<unit>\w+))(\s+)"
    )
    
    info = []
    for line in output.split("\n"):
        if match := line_format.match(line):
            info.append(ProcessInfo(**match.groupdict()))
    
    return info



def _default():
    return []

backends = {
    "hpu": _hpu_parse_processes,
    "cpu": _default
}


class GPUProcessWarden:
    """Ensure all the process using the GPU are killed before & after the bench"""
    def __init__(self, kill_on_start=True, kill_on_end=True):
        self.gpus = get_gpu_info()
        self.arch = self.gpus['arch']
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
        except :
            traceback.print_exc()
            return []

    def kill(self, pid, signal):
        if pid in self.dead_processes:
            return

        try:
            os.kill(pid, signal):
        except ProcessLookupError:
            self.dead_processes.append(pid)

    def ensure_free(self):
        processes = self.fetch_processes()
        if len(processes) == 0:
            return
    
        syslog("Found {0} still using devices after bench ended", len(processes))

        # Keyboard interrupt
        for process in processes:
            self.kill(process.pid, signal.SIGINT)

        # Sig Term, please close now
        for process in processes:
            self.kill(process.pid, signal.SIGTERM)

        # Sig Kill, just die
        for process in processes:
            self.kill(process.pid, signal.SIGKILL)

