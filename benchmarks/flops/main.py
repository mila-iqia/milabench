#!/usr/bin/env python

from argparse import ArgumentParser
import json
import time
import sys
import multiprocessing

import torch
HAS_XPU = False
try:
    import intel_extension_for_pytorch as ipex
    HAS_XPU = True
except ImportError as err:
    raise

from voir.smuggle import SmuggleWriter
from voir.instruments.gpu import get_gpu_info
from voir.instruments.utils import Monitor

KILO = 1e3
MEGA = 1e6
GIGA = 1e9
TERA = 1e12
EXA = 1e18

def has_xpu():
    try:
        import intel_extension_for_pytorch as ipex
        return torch.xpu.is_available()
    except ImportError as err:
        return True
    
def has_gaudi():
    try:
        # Intel Gaudi
        import habana_frameworks.torch.core as htcore
        return True
    except ImportError:
        return False

def has_cuda():
    return torch.cuda.is_available()
    

devices = [has_xpu, has_gaudi, has_cuda]

device = "cpu"
if torch.cuda.is_available():
    # Nvidia & AMD
    device = "cuda"

if HAS_XPU and torch.xpu.is_available():
    # Intel GPU Max
    device = "xpu"


print(HAS_XPU, torch.xpu.is_available())


print(f"Deduced, {device}")

def empty_cache():
    if device == "cuda":
        torch.cuda.empty_cache()

    if device == "xpu":
        torch.xpu.empty_cache()


def synchronize():
    if device == "cuda":
        torch.cuda.synchronize()
    
    if device == "xpu":
        torch.xpu.synchronize()


def _worker(state, queue, func, delay):
    import time

    while state["running"]:
        queue.put(func())
        time.sleep(delay)


class Monitor:
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


def modelflops(
    model: torch.nn.Module, shape, repeat=10, dtype=torch.float32, unit=TERA
):
    # Not sure how much thop is correct in its computation
    # it says it return MAC but I feel its methods is wrong
    from thop import profile

    # MAC: Multiply–accumulate operation
    batch = torch.randn(*shape, dtype=dtype, device=f"{device}:0")

    flops, _ = profile(model, inputs=(batch,))

    with torch.no_grad():
        # Prepare
        empty_cache()

        batch = batch.to(f"{device}:0")
        model = model.to(dtype=dtype, device=f"{device}:0")

        synchronize()

        # Start
        start = time.time()

        for i in range(repeat):
            _ = model(batch)

        synchronize()
        end = time.time()
        # --

    return (flops * repeat) / (end - start) / unit


def f(N, R=30, m=5000000, n=256, unit=TERA, dtype=torch.float32, log=None):
    empty_cache()
    a = torch.eye(n, dtype=dtype, device=f"{device}:0")
    x = torch.randn((m, n), dtype=dtype, device=f"{device}:0")
    y = torch.zeros_like(x)

    F = N * (2 * m * n * n + 2 * m * n * n)

    for i in range(R):
        synchronize()
        ts = -time.time()

        for _ in range(N):
            # No allocation in main loop using dual-out strategy
            y = torch.mm(x, a, out=y)
            x = torch.mm(y, a, out=x)

        synchronize()
        ts += time.time()

        if log is not None:
            log({"task": "train", "rate": F / ts / unit, "units": "Tflops"})

    empty_cache()


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
        return {"task": "main", "gpudata": data, "t": time.time()}

    monitor = Monitor(0.5, monitor_fn)
    monitor.start()
    return log, monitor


def main():
    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    parser = ArgumentParser()
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--number", type=int, default=100)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="fp32", choices=dtypes.keys())
    parser.add_argument("--tf32", action="store_true", default=False)

    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    log, monitor = setupvoir()

    f(args.number, args.repeat, args.m, args.n, TERA, dtypes[args.dtype], log)

    monitor.stop()


if __name__ == "__main__":
    main()
    print("done")
