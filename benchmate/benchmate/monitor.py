import json
import multiprocessing
import sys
import time

from voir.instruments.gpu import get_gpu_info
from voir.instruments.utils import Monitor
from voir.smuggle import SmuggleWriter


def milabench_sys_monitor():
    data_file = SmuggleWriter(sys.stdout)

    def mblog(data):
        if data_file is not None:
            print(json.dumps(data), file=data_file)

    def monitor_fn():
        data = {
            gpu["device"]: {
                "memory": [gpu["memory"]["used"], gpu["memory"]["total"]],
                "load": gpu["utilization"]["compute"],
                "temperature": gpu["temperature"],
            }
            for gpu in get_gpu_info()["gpus"].values()
        }
        mblog({"task": "train", "gpudata": data})

    monitor_fn()
    monitor = Monitor(3, monitor_fn)
    monitor.start()


def _worker(state, queue, func, delay):
    while state["running"]:
        queue.put(func())
        time.sleep(delay)


class CustomMonitor:
    def __init__(self, delay, func):
        self.manager = multiprocessing.Manager()
        self.state = self.manager.dict()
        self.state["running"] = True
        self.results = multiprocessing.Queue()
        self.process = multiprocessing.Process(
            target=_worker,
            args=(self.state, self.results, func, delay),
        )

    def start(self):
        self.process.start()

    def stop(self):
        self.state["running"] = False
        self.process.join()


def setupvoir():
    # wtf this do
    data_file = SmuggleWriter(sys.stdout)
    # data_file = sys.stdout

    def monitor_fn():
        data = {
            gpu["device"]: {
                "memory": [
                    gpu["memory"]["used"],
                    gpu["memory"]["total"],
                ],
                "load": gpu["utilization"]["compute"],
                "temperature": gpu["temperature"],
                "power": gpu["power"],
            }
            for gpu in get_gpu_info()["gpus"].values()
        }
        return {"task": "train", "gpudata": data, "time": time.time(), "units": "s"}

    monitor = CustomMonitor(0.5, monitor_fn)

    def log(data):
        nonlocal monitor

        if data_file is not None:
            data["t"] = time.time()
            print(json.dumps(data), file=data_file)

            while not monitor.results.empty():
                print(json.dumps(monitor.results.get()), file=data_file)

    monitor.start()
    return log, monitor
