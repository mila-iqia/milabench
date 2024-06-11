import json
import time
import sys
import multiprocessing

from voir.smuggle import SmuggleWriter
from voir.instruments.gpu import get_gpu_info
from voir.instruments.utils import Monitor


#
# TODO: this use deprecated things
#

def _worker(state, queue, func, delay):
    import time

    while state["running"]:
        queue.put(func())
        time.sleep(delay)


class _Monitor:
    def __init__(self, delay, func):
        self.manager = multiprocessing.Manager()
        self.state = self.manager.dict()
        self.state["running"] = True
        self.results = multiprocessing.Queue()
        self.process = multiprocessing.Process(
            target=_worker, args=(self.state, self.results, func, delay),
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

    def log(data):
        if data_file is not None:
            data["t"] = time.time()
            print(json.dumps(data), file=data_file)

            while not monitor.results.empty():
                print(json.dumps(monitor.results.get()), file=data_file)

    def monitor_fn():
        data = {
            gpu["device"]: {
                "memory": [gpu["memory"]["used"], gpu["memory"]["total"],],
                "load": gpu["utilization"]["compute"],
                "temperature": gpu["temperature"],
                "power": gpu["power"],
            }
            for gpu in get_gpu_info()["gpus"].values()
        }
        return {"task": "main", "gpudata": data, "t": time.time()}

    monitor = _Monitor(0.5, monitor_fn)
    monitor.start()
    return log, monitor


def opt_voir():
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
        mblog({"task": "main", "gpudata": data})

    monitor_fn()
    monitor = Monitor(3, monitor_fn)
    monitor.start()
    return monitor