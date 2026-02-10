#!/usr/bin/env python

from argparse import ArgumentParser
import time

import torch
import torch.nn as nn
import torchcompat.core as accelerator

from benchmate.monitor import setupvoir

KILO = 1e3
MEGA = 1e6
GIGA = 1e9
TERA = 1e12
EXA = 1e18

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



def default_mm(x, a, out, device):
    return torch.mm(
        x,
        a,
        out=out,
    )


def mxfp_mm(x, a, out, device):
    # Works
    return torch._scaled_mm(
        x,
        a.t(),
        out=out,
        scale_a=torch.tensor(1.0, device=device),
        scale_b=torch.tensor(1.0, device=device)
    )

def mxfp_tensor_scaling_mm(x, a, out, device):
    #  For TensorWise scaling, a and b should be float8, scales should be float and singletons.
    return torch._scaled_mm(
        x,
        a.t(),
        # out=out,
        out_dtype=torch.bfloat16,
        scale_a=torch.tensor(1.0, device=device),
        scale_b=torch.tensor(1.0, device=device)
    )

def mxfp_rowwise_mm(x, a, out, device):
    # For RowWise scaling, a and b should be float8, 
    #   scales should be float, scale_a should be (8192, 1) and scale_b should be (1, 8192), and both should be contiguous.
    _, n = x.shape()

    return torch._scaled_mm(
        x,
        a.t(),
        # out=out,
        out_dtype=torch.bfloat16,
        scale_a=torch.ones((1, n), device=device),
        scale_b=torch.ones((n, 1), device=device)
    )

def mxfp_blockwise_fp4_mm(x, a, out, device):
    # For RowWise scaling, a and b should be float8, 
    #   scales should be float, scale_a should be (8192, 1) and scale_b should be (1, 8192), and both should be contiguous.
    _, n = x.shape()

    n = n * n / 8

    scale_a = torch.ones(n, device=device, dtype=torch.float8_e4m3fn),
    scale_b = torch.ones(n, device=device, dtype=torch.float8_e4m3fn),

    return torch._scaled_mm(
        x,          # float4
        a.t(),      # float4
        out_dtype=torch.bfloat16,
        scale_a=scale_a,
        scale_b=scale_b
    )



def f(N, R=30, m=5000000, n=256, unit=TERA, dtype_mm=torch.float32, log=None):
    device = accelerator.fetch_device(0)

    empty_cache()
    dtype, mm = dtype_mm

    a = torch.eye(n, dtype=torch.float32, device=device).to(dtype)
    x = torch.randn((m, n), dtype=torch.float32, device=device).to(dtype)
    y = torch.zeros_like(x)

    F = N * (2 * m * n * n + 2 * m * n * n)

    for i in range(R):
        synchronize()
        ts = -time.time()

        for _ in range(N):
            # No allocation in main loop using dual-out strategy
            
            y = mm(x, a, out=y, device=device)
            x = mm(y, a, out=x, device=device)
            
            accelerator.mark_step()

        synchronize()
        ts += time.time()

        if log is not None:
            log({"task": "train", "rate": F / ts / unit, "units": "Tflops", "time": time.time()})

    empty_cache()


def main():
    dtypes = {
        "bf16": (torch.bfloat16, default_mm),
        "fp16": (torch.float16, default_mm),
        "fp32": (torch.float32, default_mm),
        "fp8": (torch.float8_e4m3fn , mxfp_mm),
        
        "fp4": (torch.float4_e2m1fn_x2, mxfp_blockwise_fp4_mm),
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

    log, monitor = setupvoir(interval=0.1)

    f(args.number, args.repeat, args.m, args.n, TERA, dtypes[args.dtype], log)

    monitor.stop()


if __name__ == "__main__":
    main()
    print("done")
