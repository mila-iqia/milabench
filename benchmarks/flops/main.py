from argparse import ArgumentParser
import json
import time
import sys

import torch

from voir.smuggle import SmuggleWriter
from voir.instruments.gpu import get_gpu_info
from voir.instruments.utils import Monitor

KILO = 1e3
MEGA = 1e6
GIGA = 1e9
TERA = 1e12
EXA = 1e18


def modelflops(model: torch.nn.Module, shape, repeat=10, dtype=torch.float32, unit=TERA):
    # Not sure how much thop is correct in its computation
    # it says it return MAC but I feel its methods is wrong
    from thop import profile
    
    # MAC: Multiply–accumulate operation
    batch = torch.randn(*shape, dtype=dtype, device="cuda:0")

    flops, _ = profile(model, inputs=(batch,))

    with torch.no_grad():
        # Prepare
        torch.cuda.empty_cache()

        batch = batch.cuda()
        model = model.to(dtype=dtype, device="cuda:0")

        torch.cuda.synchronize()

        # Start
        start = time.time()

        for i in range(repeat):
            _ = model(batch)

        torch.cuda.synchronize()
        end = time.time()
        # --

    return (flops * repeat) / (end - start) / unit



def f(N, m=5000000, n=256, unit=TERA, dtype=torch.float32):
    torch.cuda.empty_cache()
    a = torch.eye(n, dtype=dtype, device="cuda:0")
    x = torch.randn((m, n), dtype=dtype, device="cuda:0")
    y = torch.zeros_like(x)

    torch.cuda.synchronize()
    ts = -time.time()
    
    for _ in range(N):
        # No allocation in main loop using dual-out strategy
        y = torch.mm(x, a, out=y)
        x = torch.mm(y, a, out=x)
    
    torch.cuda.synchronize()
    ts += time.time()
    torch.cuda.empty_cache()
    F = N * (2 * m * n * n + 2 * m * n * n)
    return F / ts / unit



def setupvoir():
    data_file = SmuggleWriter(sys.stdout)
    def log(data):
        if data_file is not None:
            print(json.dumps(data), file=data_file)
        
    def monitor_fn():
        data = {
            gpu["device"]: {
                "memory": [
                    gpu["memory"]["used"], 
                    gpu["memory"]["total"],
                ],
                "load": gpu["utilization"]["compute"],
                "temperature": gpu["temperature"],
            }
            for gpu in get_gpu_info()["gpus"].values()
        }
        log({"task": "main", "gpudata": data})
        
    monitor = Monitor(3, monitor_fn)
    monitor.start()
    return log, monitor

    

def main():
    dtypes = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32,
    }
        
    parser = ArgumentParser()
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--m', type=int, default=256)
    parser.add_argument('--n', type=int, default=256)
    parser.add_argument('--dtype', type=str, default='fp32', choices=dtypes.keys())
    parser.add_argument('--unit', default='TERA')
    parser.add_argument('--tf32', action='store_true', default=False)
    
    args = parser.parse_args()
    
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    log, monitor = setupvoir()

    flops = f(
        args.repeat,
        args.m,
        args.n,
        args.unit,
        dtypes[args.dtype]
    )

    log({
        "task": "train",
        "rate": flops,
        "units": "Tflops"
    })
    
    monitor.stop()
    




