import GPUtil
from threading import Thread
import time


class GPUMonitor(Thread):
    def __init__(self, delay):
        super().__init__()
        self.stopped = False
        self.delay = delay
        self.data = {
            g.id: dict(
                load=[],
                memory=[],
                temperature=[]
            )
            for g in GPUtil.getGPUs()
        }

    def run(self):
        while not self.stopped:
            for g in GPUtil.getGPUs():
                data = self.data[g.id]
                data["load"].append(g.load)
                data["memory"].append(g.memoryUsed)
                data["temperature"].append(g.temperature)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
