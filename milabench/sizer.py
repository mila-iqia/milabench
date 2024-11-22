import contextvars
import multiprocessing
import os
from copy import deepcopy
import multiprocessing

import numpy as np
import yaml
from voir.instruments.gpu import get_gpu_info

from .merge import merge
from .system import CPUOptions, SizerOptions, system_global, option
from .validation.validation import ValidationLayer

ROOT = os.path.dirname(__file__)

default_scaling_config = os.path.join(ROOT, "..", "config", "scaling.yaml")


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

    if "io" in value:
        return float(value.replace("io", ""))

    if "o" in value:
        return float(value.replace("o", ""))

    return float(value)


class Sizer:
    """Automatically scale the batch size to match GPU spec"""

    def __init__(self, sizer=None, scaling_config=None):
        self.path = scaling_config
        self.sizer_override = sizer
        
        if scaling_config is None:
            scaling_config = default_scaling_config

        with open(scaling_config, "r") as sconf:
            self.scaling_config = yaml.safe_load(sconf)
            
    @property
    def options(self):
        if self.sizer_override:
            return self.sizer_override
        return SizerOptions()

    def benchscaling(self, benchmark):
        # key
        if isinstance(benchmark, str):
            return self.scaling_config.get(benchmark)

        # benchmark config
        if isinstance(benchmark, dict) and "name" in benchmark:
            return self.scaling_config.get(benchmark["name"])

        # pack
        return self.scaling_config.get(benchmark.config["name"])

    def get_capacity(self, capacity):
        if self.options.capacity is not None:
            capacity = self.options.capacity

        if isinstance(capacity, str):
            capacity = to_octet(capacity)

        return capacity

    def auto_size(self, benchmark, capacity):
        capacity = self.get_capacity(capacity)

        if capacity is None:
            return None

        config = self.benchscaling(benchmark)
        model = config.get("model", None)

        if model is None:
            print(f"Missing batch-size model for {benchmark.config['name']}")
            return 1

        data = list(sorted(config["model"].items(), key=lambda x: x[0]))
        mem = [to_octet(v[1]) for v in data]
        size = [float(v[0]) for v in data]

        if len(mem) == 1:
            print(f"Not enough data for {benchmark.config['name']}")
            return 1
        # This does not extrapolate
        # int(np.interp(capacity, mem, size))

        # Use polynomial of degree 1 so it is essentially linear interpolation
        model = np.poly1d(np.polyfit(mem, size, deg=1))

        newsize_f = model(capacity)
        newsize_i = int(newsize_f)

        if newsize_i <= 0:
            return 1

        if (newsize_f - newsize_i) > 0.5:
            newsize_i += 1

        final_size = newsize_i

        if self.options.multiple:
            final_size = (newsize_i // self.options.multiple) * self.options.multiple

        if self.options.power:
            final_size = int(self.options.power) ** int(np.log2(newsize_i))

        return max(final_size, 1)

    def size(self, benchmark, capacity):
        config = self.benchscaling(benchmark)

        if self.options.size is not None:
            return self.options.size

        if self.options.optimized:
            return config["optimized"]

        if self.options.autoscale:
            return self.auto_size(benchmark, capacity)

        return None

    def find_batch_size(self, benchmark, event):
        config = self.benchscaling(benchmark)

        if config is None:
            return None

        argname = config.get("arg")
        if argname is None:
            return -1

        if "event" in event:
            event = event["data"]

        argv = event["command"]

        for i, arg in enumerate(argv):
            if str(arg).endswith(argname):
                return int(argv[i + 1])

        return -1

    def argv(self, benchmark, capacity, argv):
        newargv = self._argv(benchmark, capacity, argv)
        return newargv
        
    def _argv(self, benchmark, capacity, argv):
        """Find the batch size and override it with a new value"""

        config = self.benchscaling(benchmark)
        if config is None:
            return argv

        newsize = self.size(benchmark, capacity)

        if newsize is None:
            return argv

        # <param> <value>
        argv = list(argv)
        argname = config.get("arg")
        if argname is None:
            return argv

        # placeholder replace
        #   train.batch_size_per_gpu={batch_size}
        placeholder = "{batch_size}"
        if placeholder in argname:
            newval = argname.format(batch_size=str(newsize))

            for i, arg in enumerate(argv):
                if str(arg).startswith(argname[0:-len(placeholder)]):
                    break
            else:
                return argv + [newval]

            argv[i] = newval
            return argv

        # positional argument replace
        #   --argname {batch_size}
        for i, arg in enumerate(argv):
            if str(arg).endswith(argname):
                break
        else:
            # add the new argument
            return argv + [argname, str(newsize)]

        argv[i + 1] = str(newsize)
        return argv


sizer_global = contextvars.ContextVar("sizer_global", default=None)


def batch_sizer() -> Sizer:
    return Sizer()
    # sizer = sizer_global.get()
    # if sizer is None:
    #     sizer_global.set(Sizer())
    #     return batch_sizer()
    # return sizer


def get_batch_size(config, start_event):
    sizer = batch_sizer()
    return sizer.find_batch_size(config, start_event)


def scale_argv(pack, argv):
    sizer = batch_sizer()

    system = system_global.get()

    if system:
        capacity = system.get("gpu", dict()).get("capacity")
        return sizer.argv(pack, capacity, argv)
    else:
        return argv


class MemoryUsageExtractor(ValidationLayer):
    """Extract max memory usage per benchmark to populate the memory model"""

    def __init__(self):
        
        self.filepath = option("sizer.save", str, None)
        sizer = batch_sizer()
        self.memory = deepcopy(sizer.scaling_config)
        self.scaling = None
        self.benchname = None
        self.batch_size = 0
        self.max_usage = float("-inf")  # Usage from the gpu monitor
        self.peak_usage = float("-inf") # Usage provided by the bench itself (for jax)
        self.early_stopped = False

    def on_start(self, entry):
        if self.filepath is None:
            return

        argv = entry.data["command"]
        self.benchname = entry.pack.config["name"]
        self.batch_size = None
        self.max_usage = float("-inf")
        self.peak_usage = float("-inf")

        config = self.memory.setdefault(self.benchname, dict())
        template = config.get("arg", None)

        if template is None:
            self.benchname = None
            return

        placeholder = "{batch_size}"
        argstart = template.replace(placeholder, "")

        is_template = False
        found = None
        for i, arg in enumerate(argv):
            if arg.endswith(template):
                found = i
                break

            #
            if arg.startswith(argstart):
                found = i
                is_template = True
                break

        if found:
            if is_template:
                arg = argv[found]
                value = arg.replace(argstart, "")
                self.batch_size = int(value)
            else:
                self.batch_size = int(argv[found + 1])
        else:
            print("Count not find batch_size argument")

    def on_data(self, entry):
        if self.filepath is None:
            return

        if entry.data is None:
            return

        memorypeak = entry.data.get("memory_peak")
        if memorypeak is not None:
            self.peak_usage = max(memorypeak, self.peak_usage)
            return

        gpudata = entry.data.get("gpudata")
        if gpudata is not None:
            current_usage = []
            for device, data in gpudata.items():
                usage, total = data.get("memory", [0, 1])
                current_usage.append(usage)

            self.max_usage = max(*current_usage, self.max_usage)

    def on_stop(self, entry):
        self.early_stopped = True

    def max_memory_usage(self):
        if self.peak_usage != float("-inf"):
            return self.peak_usage
        return self.max_usage

    def on_end(self, entry):
        if self.filepath is None:
            return

        if (
            self.benchname is None
            or self.batch_size is None
            or self.max_memory_usage() == float("-inf")
        ):
            return

        # Only update is successful
        rc = entry.data["return_code"]
        if rc == 0 or self.early_stopped:
            config = self.memory.setdefault(self.benchname, dict())
            model = config.setdefault("model", dict())
            model[self.batch_size] = f"{self.max_memory_usage()} MiB"
            config["model"] = dict(sorted(model.items(), key=lambda x: x[0]))

        self.benchname = None
        self.batch_size = None
        self.max_usage = float("-inf")
        self.peak_usage = float("-inf")

    def report(self, *args):
        if self.filepath is not None:
            newdata = self.memory

            if os.path.exists(self.filepath):
                with open(self.filepath, "r") as fp:
                    previous_data = yaml.safe_load(fp)
                newdata = merge(previous_data, self.memory)

            with open(self.filepath, "w") as file:
                yaml.dump(newdata, file)


def arch_to_device(arch):
    device_types = [
        "cpu",
        "cuda",
        "ipu",
        "xpu",
        "mkldnn",
        "opengl", "opencl", "ideep", "hip", "ve",
        "fpga", "maia", "xla", "lazy", "vulkan", "mps", "meta",
        "hpu", "mtia", "privateuseone"
    ]
    arch_to_device = {t:t for t in device_types}
    arch_to_device["rocm"] = "cuda"
    return arch_to_device.get(arch, "cpu")



def new_argument_resolver(pack):
    system_config = system_global.get()
    if system_config is None:
        system_config = {}

    context = deepcopy(system_config)

    arch = context.get("arch", "cpu")
    device_count_used = 1
    device_count_system = len(get_gpu_info()["gpus"])

    if hasattr(pack, "config"):
        device_count_used = len(pack.config.get("devices", [0]))

    if device_count_used <= 0:
        device_count_used = 1

    ccl = {"hpu": "hccl", "cuda": "nccl", "rocm": "rccl", "xpu": "ccl", "cpu": "gloo"}

    cpu_opt = CPUOptions()
    def auto(value, default):
        if cpu_opt.enabled:
            return value
        return default

    def clamp(x, mn=cpu_opt.cpu_min, mx=cpu_opt.cpu_max):
        return min(max(x, mn), mx)

    total_cpu = cpu_opt.total_count or multiprocessing.cpu_count()
    total_available = total_cpu - cpu_opt.reserved_cores

    context["cpu_count"] = total_available
    context["cpu_per_gpu"] = total_available // max(device_count_system, 1)
    context["n_worker"] = clamp(context["cpu_per_gpu"])

    if cpu_opt.n_workers is not None:
        context["n_worker"] = cpu_opt.n_workers

    context["arch"] = arch
    context["device_name"] = arch_to_device(arch)
    context["ccl"] = ccl.get(arch, "gloo")

    context["milabench_base"] = option("base", str, default="")
    dirs = vars(pack.dirs)
    context["milabench_venv"] = dirs.get('venv', "")
    context["milabench_code"] = dirs.get('code', "")
    context["milabench_extra"] = dirs.get('extra', "")
    context["milabench_data"] = dirs.get('data', "")
    context["milabench_runs"] = dirs.get('runs', "")
    context["milabench_cache"] = dirs.get('cache', "")
    context["milabench_name"] = pack.config.get("name", None)
    context["benchmark_folder"] = pack.config.get('definition', None)

    def auto_eval(arg):
        newvalue = str(arg).format(**context)
        if newvalue.startswith("auto"):
            newvalue = str(eval(newvalue, {"auto": auto}, {}))
        return newvalue

    return auto_eval


def resolve_placeholder(pack, value):
    resolver = new_argument_resolver(pack)
    return resolver(value)


def resolve_argv(pack, argv):
    resolver = new_argument_resolver(pack)
    argv = list(argv)
    for i, arg in enumerate(argv):
        argv[i] = resolver(arg)
    return argv
