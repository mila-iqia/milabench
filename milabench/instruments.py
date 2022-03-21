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


@gated("--rates")
def rates(ov):
    yield ov.phases.load_script

    sync = None

    def setsync(use_cuda):
        if use_cuda:
            nonlocal sync
            import torch

            sync = torch.cuda.synchronize

    ov.given["?use_cuda"].first_or_default(False) >> setsync

    times = (
        ov.given.keep("step", "batch_size")
        .augment(time=lambda: time.time_ns())
        .pairwise()
        .buffer_with_time(1.0)
    )

    @times.subscribe
    def _(elems):
        t = 0
        if sync is not None:
            t0 = time.time_ns()
            sync()
            t1 = time.time_ns()
            t += t1 - t0

        t += sum(e2["time"] - e1["time"] for e1, e2 in elems)
        n = sum(e1["batch_size"] for e1, e2 in elems)
        t /= 1_000_000_000

        ov.give(rate=n / t)


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
