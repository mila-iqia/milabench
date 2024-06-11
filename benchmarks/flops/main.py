#!/usr/bin/env python

from argparse import ArgumentParser
import time

import torch
import torchcompat.core as accelerator

from benchmate.common import setupvoir

KILO = 1e3
MEGA = 1e6
GIGA = 1e9
TERA = 1e12
EXA = 1e18


print(f"Using, {accelerator.device_type}")


def empty_cache():
    accelerator.empty_cache()


def synchronize():
    accelerator.synchronize()



def modelflops(
    model: torch.nn.Module, shape, repeat=10, dtype=torch.float32, unit=TERA
):
    # Not sure how much thop is correct in its computation
    # it says it return MAC but I feel its methods is wrong
    from thop import profile

    device = accelerator.fetch_device(0)

    # MAC: Multiply–accumulate operation
    batch = torch.randn(*shape, dtype=dtype, device=device)

    flops, _ = profile(model, inputs=(batch,))

    with torch.no_grad():
        # Prepare
        empty_cache()

        batch = batch.to(device)
        model = model.to(dtype=dtype, device=device)

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
    device = accelerator.fetch_device(0)

    empty_cache()

    a = torch.eye(n, dtype=dtype, device=device)
    x = torch.randn((m, n), dtype=dtype, device=device)
    y = torch.zeros_like(x)

    F = N * (2 * m * n * n + 2 * m * n * n)

    for i in range(R):
        synchronize()
        ts = -time.time()

        for _ in range(N):
            # No allocation in main loop using dual-out strategy
            y = torch.mm(x, a, out=y)
            x = torch.mm(y, a, out=x)
            accelerator.mark_step()

        synchronize()
        ts += time.time()

        if log is not None:
            log({"task": "train", "rate": F / ts / unit, "units": "Tflops"})

    empty_cache()




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

    accelerator.set_enable_tf32(args.tf32)

    log, monitor = setupvoir()

    f(args.number, args.repeat, args.m, args.n, TERA, dtypes[args.dtype], log)

    monitor.stop()


if __name__ == "__main__":
    main()
    print("done")
