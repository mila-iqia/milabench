from collections import defaultdict
import contextvars
import multiprocessing
import os
from copy import deepcopy
import time

import numpy as np
import yaml
from voir.instruments.gpu import get_gpu_info
from cantilever.core.statstream import StatStream

from .syslog import syslog
from .system import CPUOptions, SizerOptions, system_global, option
from .validation.validation import ValidationLayer

ROOT = os.path.dirname(__file__)



default_scaling_folder = os.path.join(ROOT, "..", "config", "scaling")
default_scaling_config = os.path.join(default_scaling_folder, "default.yaml")


def gpu_name():
    try:
        info = get_gpu_info()
        values = list(info["gpus"].values())
        return values[0]["product"]
    except:
        return None


def get_scaling_config():
    name = gpu_name()

    specialized = os.path.join(default_scaling_folder, f"{name}.yaml")

    if name is None or not os.path.exists(specialized):
        return default_scaling_config
    
    return specialized


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

    def __init__(self, sizer=None, scaling_config=option("sizer.config", etype=str)):
        self.path = scaling_config
        self.sizer_override = sizer
        
        if scaling_config is None:
            scaling_config = get_scaling_config()

        if os.path.exists(scaling_config):
            with open(scaling_config, "r") as sconf:
                self.scaling_config = yaml.safe_load(sconf)
        else:
            print(scaling_config, "does not exist")

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

    def _scaling_v1(self, config):
        data = list(sorted(config["model"].items(), key=lambda x: x[0]))

        mem = [to_octet(v[1]) for v in data]
        size = [float(v[0]) for v in data]

        return mem, size
    
    def _scaling_v2(self, config):
        data = config["observations"]

        data = list(sorted(data, key=lambda x: x["batch_size"]))

        mem = [to_octet(v["memory"]) for v in data]
        size = [float(v["batch_size"]) for v in data]

        return mem, size

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

        if "model" in config:
            mem, size = self._scaling_v1(config)
        else:
            mem, size = self._scaling_v2(config)

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


def suggested_batch_size(pack):
    sizer = batch_sizer()

    system = system_global.get()
    capacity = system.get("gpu", dict()).get("capacity")

    return sizer.size(pack, capacity)


def compact_dump():
    # This is to create a compact yaml that is still readable
    from yaml.representer import SequenceNode, ScalarNode

    class CustomDumper(yaml.SafeDumper):
        
        def represent_sequence(self, tag, sequence, flow_style=None):
            value = []
            node = SequenceNode(tag, value, flow_style=flow_style)

            if self.alias_key is not None:
                self.represented_objects[self.alias_key] = node
            best_style = True

            for item in sequence:
                node_item = self.represent_data(item)
                node_item.flow_style = True

                if not (isinstance(node_item, ScalarNode) and not node_item.style):
                    best_style = False

                value.append(node_item)

            if flow_style is None:
                if self.default_flow_style is not None:
                    node.flow_style = self.default_flow_style
                else:
                    node.flow_style = best_style
            return node

    return CustomDumper


class MemoryUsageExtractor(ValidationLayer):
    """Extract max memory usage per benchmark to populate the memory model"""

    def __init__(self):
        sizer = Sizer()

        self.filepath = SizerOptions.instance().save
        
        if self.filepath and os.path.exists(self.filepath):
            with open(self.filepath, "r") as fp:
                self.memory = yaml.safe_load(fp) or {}
        else:
            self.memory = deepcopy(sizer.scaling_config)

        if self.memory.get("version", 1.0) <= 1.0:
            self.convert()
            self.memory["version"] = 2.0

        self.scaling = None
        self.stats = defaultdict(lambda: StatStream(drop_first_obs=0))
        self.benchname = None
        self.batch_size = None
        self.active_count = defaultdict(int)
        self.rc = defaultdict(int)
        self.max_usage = float("-inf")  # Usage from the gpu monitor
        self.peak_usage = float("-inf") # Usage provided by the bench itself (for jax)
        self.early_stopped = False

        global on_batch_size_set, on_cpu_count_set

        # TODO: currently this is okay but we might have to find a way to make
        # this class only remove its callback
        on_batch_size_set = [self.on_batch_size_set]
        on_cpu_count_set = [self.on_cpu_count_set]

    def on_cpu_count_set(self, pack, _, value):
        self.stats["cpu"] += value

    def on_batch_size_set(self, pack, _, value):
        self.batch_size = value
        self.stats["batch_size"] += value

    def convert(self):
        # TODO: this could be handled seemlessly on the loading part
        for bench, config in self.memory.items():
            if bench == "version":
                continue
        
            model = config.pop("model", None)

            if model is not None:
                obs = []
                for k, v in model.items():
                    obs.append({"batch_size": k, "memory": v})
    
                config["observations"] = obs

    def on_start(self, entry):
        if self.filepath is None:
            return

        self.benchname = entry.pack.config["name"]
        self.max_usage = float("-inf")
        self.peak_usage = float("-inf")
        self.active_count[self.benchname] += 1

    def on_data(self, entry):
        if self.filepath is None:
            return

        if entry.data is None:
            return

        # This is for jax
        if memorypeak := entry.data.get("memory_peak"):
            self.stats["memorypeak"] += memorypeak
            return
        
        if gpudata := entry.data.get("gpudata"):
            for device, data in gpudata.items():
                usage, total = data.get("memory", [0, 1])
                self.stats["memory"] += usage

        if rate := entry.data.get("rate"):
            self.stats["perf"] += rate
 
    def on_stop(self, entry):
        self.early_stopped = True

    def max_memory_usage(self):
        if self.stats["memorypeak"].current_count != 0:
            return self.stats["memorypeak"].max
        return self.stats["memory"].max

    def on_end(self, entry):
        if self.filepath is None:
            return

        if self.benchname is None:
            syslog("MemoryUsageExtractor: Skipping missing benchmark {}", entry)
            return

        if self.batch_size is None:
            syslog("MemoryUsageExtractor: Skipping missing batch_size {}", entry)
            return

        if self.max_memory_usage() == float("-inf"):
            syslog("MemoryUsageExtractor: Missing memory info {}", entry)
            return
    
        # Only update is successful
        rc = entry.data["return_code"]

        if rc == 0 or self.early_stopped:
            rc = 0

        self.rc[self.benchname] += rc
        self.active_count[self.benchname] -= 1


        if self.active_count[self.benchname] <= 0:
            print(self.rc)
            if self.rc[self.benchname] == 0:
                self.push_observation()
            else:
                syslog("MemoryUsageExtractor: Could not add scaling data because of a failure {}", self.benchname)

            self.benchname = None
            self.batch_size = None
            self.stats = defaultdict(lambda: StatStream(drop_first_obs=0))
            self.max_usage = float("-inf")
            self.peak_usage = float("-inf")
            # avoid losing results
            try:
                self.save()
            except Exception as err:
                print(f"MemoryUsageExtractor: Could not save scaling file because of {err}")

    def push_observation(self):
        config = self.memory.setdefault(self.benchname, dict())
        observations = config.setdefault("observations", [])

        obs = {
            "cpu": int(self.stats["cpu"].avg),
            "batch_size": int(self.stats["batch_size"].avg),
            "memory": f"{int(self.stats['memory'].max)} MiB",
            "perf": float(f"{self.stats['perf'].avg:.2f}"),
            "time": int(time.time())
        }

        if memorypeak := self.stats.pop("memorypeak", None):
            if memorypeak.current_count != 0:
                obs["memory"] = f"{int(memorypeak.max)} MiB",

        observations.append(obs)
        config["observations"] = list(sorted(observations, key=lambda x: x["batch_size"]))

    def save(self):
        if self.filepath is not None:
            with open(self.filepath, "w") as file:
                yaml.dump(self.memory, file, Dumper=compact_dump())

    def report(self, *args):
        self.save()

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


on_cpu_count_set = []
on_batch_size_set = []

def broadcast(delegates, *args, **kwargs):
    for fun in delegates:
        try:
            fun(*args, **kwargs)
        except Exception as err:
            print(f"Error during broadcasting {fun} {err}")


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

    cpu_opt = CPUOptions.instance()

    def cpu(value, default):
        newvalue = default

        if cpu_opt.enabled:
            newvalue = value
        
        broadcast(on_cpu_count_set, pack, default, newvalue)
        return newvalue
    
    gpu_opt = SizerOptions.instance()
    def batch_resize(default):
        val = default

        if gpu_opt.enabled:
            if (gpu_opt.add is not None or gpu_opt.mult is not None):
                val = max(1, int(default * (gpu_opt.mult or 1)) + (gpu_opt.add or 0))
            else:
                val = suggested_batch_size(pack)
                assert val is not None

        broadcast(on_batch_size_set, pack, default, val)
        return val

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
        newvalue: str = str(arg).format(**context)

        # Handles the case where argument=value
        finalize_val = lambda x: x
        if "=" in newvalue:
            name, newvalue = newvalue.split("=", maxsplit=1)
            finalize_val = lambda x: f"{name}={x}"

        if newvalue.startswith("auto"):
            newvalue = str(eval(newvalue, {"auto": cpu, "auto_batch": batch_resize}, {}))
        
        return finalize_val(newvalue)

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




def deduplicate_observation(scaling):
    deduplicated_scaling = {}

    for bench, data in scaling.items():
        if bench == "version":
            deduplicated_scaling[bench] = data
            continue

        observations = data.get("observations", [])
        duplicate_sets = defaultdict(list)

        for obs in observations:
            index = (obs["batch_size"], obs["cpu"])
            duplicate_sets[index].append(obs)
        
        newobs = []

        # Add back unique observation
        for key in list(duplicate_sets.keys()):
            if len(duplicate_sets[key]) == 1:
                data = duplicate_sets.pop(key)[0]

                if data["perf"] > 0:
                    newobs.append(data)

        # Merge duplicates
        while len(duplicate_sets) > 0:
            key, data = duplicate_sets.popitem()

            memory_stat = StatStream(0)
            perf_stat = StatStream(0)
            lastest_time = 0

            for obs in data:
                perf = obs["perf"]

                if perf > 0:
                    memory_stat += int(obs["memory"].split(" ")[0])
                    perf_stat += perf
                    lastest_time = max(lastest_time, obs["time"])

            should_generate_aggregate = (
                (perf_stat.count > 1 and memory_stat.count > 1) and 
                (memory_stat.avg > 0 and memory_stat.sd / memory_stat.avg < 0.1) and 
                (perf_stat.avg > 0   and perf_stat.sd   / perf_stat.avg   < 0.1)
            )
            should_generate_single = perf_stat.count == 1 and perf_stat.avg > 0 and memory_stat.count == 1

            if should_generate_aggregate or should_generate_single:
                # If observation are similar-ish merge them into one
                newobs.append({
                    "batch_size": key[0], 
                    "cpu": key[1],
                    "memory": f"{int(memory_stat.avg)} MiB",
                    "perf": int(perf_stat.avg * 100) / 100,
                    "time": int(lastest_time)
                })
            else:
                if (not should_generate_aggregate) and perf_stat.avg > 0:
                    syslog("Could not merge observation, significant differences because (Mem: {:.2f} < 0.1) and (Perf: {:.2f} < 0.1)",
                         memory_stat.sd / memory_stat.avg, perf_stat.sd / perf_stat.avg)
                
                for obs in data:
                    if obs["perf"] > 0:
                        newobs.append(obs)

        # make sure observations are sorted
        newobs = list(sorted(newobs, key=lambda x: x["batch_size"]))

        deduplicated_scaling[bench] = {
            "observations": newobs
        }

    return deduplicated_scaling


def deduplicate_scaling_file(filepath):
    with open(filepath, "r") as fp:
        memory = yaml.safe_load(fp) or {}

    newmem = deduplicate_observation(memory)

    with open(f"{filepath}.new.yml", "w") as fp:
        yaml.dump(newmem, fp, Dumper=compact_dump())



if __name__ == "__main__":
    filepath = "/home/testroot/milabench/config/scaling/MI325.yaml"
    deduplicate_scaling_file(filepath)
