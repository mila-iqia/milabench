from dataclasses import dataclass
import os
import os
import yaml
import contextvars

import numpy as np

from .config import system_global


ROOT = os.path.dirname(__file__)

default_scaling_config = os.path.join(ROOT, "..", "config", "scaling.yaml")


@dataclass
class SizerOptions:
    size: int = os.getenv("MILABENCH_SIZER_BATCH_SIZE")
    autoscale: bool = os.getenv("MILABENCH_SIZER_AUTO", False)
    enabled: bool = os.getenv("MILABENCH_SIZER_ENABLED", False)
    multiple: int = os.getenv("MILABENCH_SIZER_MULTIPLE")
    optimized: bool = os.getenv("MILABENCH_SIZER_OPTIMIZED")


metric_prefixes = {
    "T": (12, 4),
    "G": (9, 3),
    "M": (6, 2),
    "k": (3, 1),
    "h": (2, None),
    "da": (1, None),
    "d": (-1, None),
    "c": (-2, None),
    "m": (-3, None),
    "u": (-6, None),
    "n": (-9, None),
    "p": (-12, None),
}


def to_octet(value: str) -> float:
    for p, (vm, vb) in metric_prefixes.items():
        if f"{p}iB" in value or f"{p}io" in value:
            return float(value[: -(len(p) + 2)]) * 1024**vb

        if f"{p}B" in value or f"{p}o" in value:
            return float(value[: -(len(p) + 1)]) * 10**vm

    return float(value[:-1])


class Sizer:
    """Automatically scale the batch size to match GPU spec"""

    def __init__(self, options=SizerOptions(), scaling_config=None):
        self.options = options

        if scaling_config is None:
            scaling_config = default_scaling_config

        with open(scaling_config, "r") as sconf:
            self.scaling_config = yaml.safe_load(sconf)

    def benchscaling(self, benchmark):
        # key
        if isinstance(benchmark, str):
            return self.scaling_config.get(benchmark)

        # benchmark config
        if isinstance(benchmark, dict) and "name" in benchmark:
            return benchmark

        # pack
        return self.scaling_config.get(benchmark.config["name"])

    def auto_size(self, benchmark, capacity):
        if isinstance(capacity, str):
            capacity = to_octet(capacity)

        config = self.benchscaling(benchmark)

        mem = [to_octet(v) for v in config["model"].values()]
        size = [float(v) for v in config["model"].keys()]

        # This does not extrapolate
        # int(np.interp(capacity, mem, size))

        # Use polynomial of degree 1 so it is essentially linear interpolation
        model = np.poly1d(np.polyfit(mem, size, deg=1))

        newsize_f = model(capacity)
        newsize_i = int(newsize_f)

        if (newsize_f - newsize_i) > 0.5:
            newsize_i += 1

        if self.options.multiple is not None:
            return (newsize_i // self.options.multiple) * self.options.multiple

        return newsize_i

    def size(self, benchmark, capacity):
        config = self.benchscaling(benchmark)

        if self.options.optimized:
            return config["optimized"]

        if self.options.autoscale:
            return self.auto_size(benchmark, capacity)

        return self.options.size

    def argv(self, benchmark, capacity, argv):
        """Find the batch size and override it with a new value"""

        newsize = self.size(benchmark, capacity)

        if newsize is None:
            return argv

        # <param> <value>
        argv = list(argv)
        config = self.benchscaling(benchmark)

        for i, arg in enumerate(argv):
            if arg == config["arg"]:
                break
        else:
            # add the new argument
            return argv + [config["arg"], str(newsize)]

        argv[i + 1] = str(newsize)
        return argv


sizer_global = contextvars.ContextVar("sizer_global")
sizer_global.set(Sizer())


def scale_argv(pack, argv):
    sizer = sizer_global.get()
    system = system_global.get()

    if not sizer.options.enabled:
        return argv

    capacity = system["gpu"]["capacity"]

    return sizer.argv(pack, capacity, argv)
