
import json
import shutil
import subprocess
import sys
import os


arch = os.environ.get("MILABENCH_GPU_ARCH", None)


def _get_info(requires, command, parse_function):
    if not shutil.which(requires):
        return None
    proc_results = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        return parse_function(json.loads(proc_results.stdout))
    except json.JSONDecodeError:
        print(f"There was a problem with {requires}:")
        print("=" * 80)
        print(proc_results.stderr, file=sys.stderr)
        print("=" * 80)
        return None


def get_cuda_info():
    return _get_info("nvidia-smi", "nvidia-smi -x -q | xml2json", parse_cuda)
    

def get_rocm_info():
    return _get_info("rocm-smi", "rocm-smi -a --showmeminfo vram --json", parse_rocm)


def get_gpu_info():
    global arch

    if arch is None:
        cuda = get_cuda_info()
        rocm = get_rocm_info()
        if cuda and rocm:
            raise Exception(
                "Milabench found both CUDA and ROCM-compatible GPUs and does not"
                " know which kind to use. Please set $MILABENCH_GPU_ARCH to 'cuda',"
                " 'rocm' or 'cpu'."
            )
        elif cuda:
            arch = "cuda"
            return cuda
        elif rocm:
            arch = "rocm"
            return rocm
        else:
            arch = "cpu"
            return {}

    if arch == "cuda":
        return get_cuda_info()
    elif arch == "rocm":
        return get_rocm_info()
    elif arch == "cpu":
        return {}
    else:
        raise ValueError(
            f"Invalid value for $MILABENCH_GPU_ARCH: '{arch}'."
            " It should be one of 'cuda', 'rocm' or 'cpu'."
        )

def parse_cuda(info):
    def parse_num(n):
        n, units = n.split(" ")
        if units == "MiB":
            return int(n)
        elif units == "C" or units == "W":
            return float(n)
        elif units == "%":
            return float(n) / 100
        else:
            raise ValueError(n)

    def parse_gpu(gpu, gid):
        mem = gpu["fb_memory_usage"]
        used = parse_num(mem["used"])
        total = parse_num(mem["total"])
        return {
            "device": gid,
            "product": gpu["product_name"],
            "memory": {
                "used": used,
                "total": total,
            },
            "utilization": {
                "compute": parse_num(gpu["utilization"]["gpu_util"]),
                "memory": used / total,
            },
            "temperature": parse_num(gpu["temperature"]["gpu_temp"]),
            "power": parse_num(gpu["power_readings"]["power_draw"]),
            "selection_variable": "CUDA_VISIBLE_DEVICES",
        }

    data = info["nvidia_smi_log"]
    gpus = data["gpu"]
    if not isinstance(gpus, list):
        gpus = [gpus]

    return {
        i: parse_gpu(g, i)
        for i, g in enumerate(gpus)
    }

def parse_rocm(info):
    def parse_gpu(gpu, gid):
        used = int(gpu["VRAM Total Used Memory (B)"])
        total = int(gpu["VRAM Total Memory (B)"])
        return {
            "device": gid,
            "product": "ROCm Device",
            "memory": {
                "used": used // (1024**2),
                "total": total // (1024**2),
            },
            "utilization": {
                "compute": float(gpu["GPU use (%)"]) / 100,
                "memory": used / total,
            },
            "temperature": float(gpu["Temperature (Sensor edge) (C)"]),
            "power": float(gpu["Average Graphics Package Power (W)"]),
            "selection_variable": "ROCR_VISIBLE_DEVICES",
        }

    results = {}
    for k, v in info.items():
        x, y, n = k.partition("card")
        if x != "" or y != "card":
            continue
        cnum = int(n)
        results[cnum] = parse_gpu(v, cnum)

    return results
