import time
from threading import Thread

from voir.tools import gated, parametrized


@gated("--display", "Display given")
def display(ov):
    ov.given.display()


@parametrized("--stop", type=int, default=0, help="Number of iterations to run for")
def stop(ov):
    yield ov.phases.load_script
    stop = ov.options.stop
    if stop:
        ov.given.where("step").skip(stop) >> ov.stop


class GPUMonitor(Thread):
    def __init__(self, ov, delay):
        super().__init__(daemon=True)
        self.ov = ov
        self.stopped = False
        self.delay = delay

    def run(self):
        import GPUtil

        while not self.stopped:
            self.ov.give(gpudata=GPUtil.getGPUs())
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


@gated("--gpu", "Profile GPU usage.")
def profile_gpu(ov):
    yield ov.phases.load_script
    monitor = GPUMonitor(ov, 100)
    monitor.start()
    yield ov.phases.run_script
    monitor.stop()
