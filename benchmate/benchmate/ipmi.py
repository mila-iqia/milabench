"""Monitor IPMI Sensors"""

import fnmatch
import time
import subprocess
import os
from contextlib import contextmanager
import multiprocessing
import json

command = "ipmitool -I lanplus -C 3 -H {ip} -U {user} -P {password} sdr"


def match(name, subset):
    if subset is None:
        return True

    if isinstance(subset, str):
        return fnmatch.fnmatch(name, subset)

    for s in subset:
        if match := fnmatch.fnmatch(name, s):
            return match
    
    return False


def replace_unit(unit):
    match unit:
        case "degrees C":
            return "C"
        case "Readable":
            return None
        case "reading":
            return None
        case _:
            return unit

def process_value(value):
    match value:
        case "no reading":
            return None
        case "Not":
            return None
        case _ :
            return value


def ipmi_monitor(ip=None, user="admin", password="password", sensor_subset=None, extra=None):
    """Use IPMI to monitor a system, IPMI has access to sensors outside of the linux system"""

    if extra is None:
        extra = {}

    ip = os.getenv("VOIR_IPMI_IP", ip)
    user = os.getenv("VOIR_IPMI_USER", user)
    password = os.getenv("VOIR_IPMI_PASSWORD", password)

    assert ip is not None

    ipmi_cmd = command.format(ip=ip, user=user, password=password)

    def monitor():
        t = time.time()
        proc = subprocess.run(
            ipmi_cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if proc.returncode != 0:
            return {}

        metrics = {"time": { "value": t, "unit": "s"}, **extra}
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split("|")]
            name = parts[0].strip()

            if match(name, sensor_subset):
                try:
                    value, unit = parts[1].strip().split(" ", maxsplit=1)
                    if (value := process_value(value)) and (unit := replace_unit(unit)):
                        metrics[name] = {"value":  value, "unit": unit} 
        
                except ValueError:
                    pass

        return metrics

    return monitor


def _ipmi_worker(monitor, path, interval, stop_event):
    with open(path, "a") as fp:
        while not stop_event.is_set():
            try:
                start = time.time()
                json.dump(monitor(), fp)
                fp.write("\n")
                fp.flush()

                elapsed = time.time() - start
                stop_event.wait(max(interval - elapsed, 0))  # interruptible sleep
            except:
                import traceback
                traceback.print_exc()

def save_path():
    from milabench.config import get_run_folder
    import os
    run_folder = get_run_folder()

    run_folder.mkdir(parents=True, exist_ok=True)

    return run_folder / "ipmi.jsonl"


class IPMIParallelMonitor:
    def __init__(self, path=None, interval=10, **kwargs):
        if path is None:
            path = save_path()

        self.stop_event = None
        self.proc = None
        self.path = path
        self.interval = interval
        self.kwargs = kwargs
        self.manager = multiprocessing.Manager()
        self.extra = self.manager.dict()
        self.extra["bench"] = ""
        self.stopped = True

    def __enter__(self):
        self.stop_event = multiprocessing.Event()
        monitor = ipmi_monitor(**self.kwargs, extra=self.extra)

        self.proc = multiprocessing.Process(
            target=_ipmi_worker,
            args=(monitor, self.path, self.interval, self.stop_event),
        )
        self.manager.__enter__()
        self.proc.start()
        self.stopped = False
        return self

    def __exit__(self, *args, **kwargs):
        self.stop_event.set()
        self.proc.join(timeout=5)

        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join()

        self.manager.__exit__(*args, **kwargs)
        self.extra = dict()
        self.stopped = True
    #
    # Store the benchmark we are working on
    #
    def __call__(self, entry):
        return self.on_event(entry)
        
    def on_event(self, entry):
        method = getattr(self, f"on_{entry.event}", None)

        if method is not None:
            method(entry)

    def on_start(self, entry):
        name, _ = entry.tag.split(".", maxsplit=1)

        if not self.stopped:
            self.extra["bench"] = name

    def on_end(self, entry):
        if not self.stopped:
            self.extra["bench"] = ""



def main():
    with IPMIParallelMonitor(
        path="ipmi.jsonl",
        interval=2.0,
        sensor_subset=["PWR_PSU_*_POUT"],
    ) as mon:

        mon.extra["phase"] = "warmup"
        time.sleep(10)

        mon.extra["phase"] = "benchmark"
        time.sleep(10)

        mon.extra["phase"] = "cooldown"
        time.sleep(10)


if __name__ == "__main__":
    main()
