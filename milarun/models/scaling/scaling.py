
import torch
import json
import os
import sys
import subprocess
import numpy as np

from coleo import Argument, default
from milarun.lib import coleo_main, init_torch, iteration_wrapper, dataloop


def main(exp, argv):
    rdir = exp.results_directory()
    device_count = torch.cuda.device_count()
    processes = []
    for i in range(device_count):
        outfile = os.path.join(rdir, f"results-{i}.json")
        cmd = [
            "milarun",
            "run",
            "--out", outfile,
            "--extra", json.dumps({"sub_job": True}),
            "milarun.models.scaling.micro_bench:main",
            "--",
            "--distributed-dataparallel",
            "--rank", str(i),
            "--world-size", str(device_count),
            "--dist-backend", "nccl",
            "--dist-url", "tcp://localhost:8181",
            *argv
        ]
        process = subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)})
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            return

    for i in range(device_count):
        results = json.load(open(os.path.join(rdir, f"results-{i}.json")))
        exp.timings[f"train_{i}"] = results["timings"]["train"]
