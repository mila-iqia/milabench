import contextvars
from copy import deepcopy
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field

import yaml
from voir.instruments.gpu import get_gpu_info

from .fs import XPath
from .merge import merge
from .network import resolve_addresses

system_global = contextvars.ContextVar("system", default=None)
multirun_global = contextvars.ContextVar("multirun", default=None)

def get_gpu_capacity(strict=False):
    try:
        capacity = 1e24

        for k, v in get_gpu_info()["gpus"].items():
            capacity = min(v["memory"]["total"], capacity)

        return int(capacity)
    except:
        print("GPU not available, defaulting to 0 MiB")
        if strict:
            raise
        return 0


def getenv(name, expected_type):
    value = os.getenv(name)

    if value is not None:
        try:
            return expected_type(value)
        except TypeError:
            print(f"{name}={value} expected type {expected_type} got {type(value)}")
            return None
    return value


def print_once(*args, **kwargs):
    printed = 0
    def _print():
        nonlocal printed
        if printed == 0:
            print(*args, **kwargs, file=sys.stderr)
            printed += 1

    return _print


warn_no_config = print_once("No system config found, using defaults")


_global_options = {}

def _track_options(name, type, default, value):
    """This is just a helper so command line can display the options"""
    global _global_options

    try:
        _global_options[name] = {
            "type": type,
            "default": default,
            "value": value
        } 
    except:
        pass


def as_environment_variable(name):
    frags = name.split(".")
    return "MILABENCH_" + "_".join(map(str.upper, frags))


def _resumable_multirun(multirun_cache):
    import json

    done = {}

    if os.path.exists(multirun_cache):
        with open(multirun_cache, "r") as fp:
            for line in fp.readlines():
                run = json.loads(line)
                unique_name = run["name"].split(".")[0]
                done[unique_name] = run


    def mark_run_as_done(name, config):
        with open(multirun_cache, "a") as f:
            f.write(json.dumps({"name": name, "run": config}) + "\n")

    for run_name, run in _multirun():
        unique_name = run_name.split(".")[0]

        if unique_name in done:
            print(f"skipping run {unique_name} because it already ran")
            continue

        yield run_name, run

        mark_run_as_done(run_name, run)




def _multirun():
    multirun = multirun_global.get()
    
    if multirun is None or len(multirun) == 0:
        yield None, dict()
        
    runs = multirun.get("runs", dict())
    
    from .config import combine_args
    import time
    from types import SimpleNamespace
    
    def unflatten(dct):
        result = {}
        for k, v in dct.items():
            l = result
            frags = k.split(".")
            for frag in frags[:-1]:
                l = l.setdefault(frag, SimpleNamespace())
            setattr(l, frags[-1], v)
            
        return result
                
    for run_matrix in runs:
        arguments = run_matrix["matrix"]

        for run in combine_args(arguments, dict()):
            template_name = run_matrix["name"]
            
            ctx = unflatten(run)
            ctx['time'] = int(time.time())
            
            run_name = template_name.format(**ctx)
            
            yield run_name, run


@contextmanager
def apply_system(config: dict):
    system = system_global.get()
    old = deepcopy(system)
    
    if system is None:
        system = dict()
        system_global.set(system)
        system = system_global.get()
    
    for k, v in config.items():
        frags = k.split(".")
        
        lookup = system.setdefault("options", {})
        for f in frags[:-1]:
            lookup = lookup.setdefault(f, {})
        lookup[frags[-1]] = v
        

    yield    
    system_global.set(old)


def select(*args):
    # This handles the case where 0 is right value and None is not
    prev = []

    for val in args:
        if val is not None:
            prev.append(val)

        if val:
            return val
    
    if prev:
        return prev[0]

    return None


def option(name, etype, default=None):
    options = dict()
    system = system_global.get()
    if system:
        options = system.get("options", dict())

    frags = name.split(".")
    env_name = as_environment_variable(name)
    env_value = getenv(env_name, etype)

    lookup = options
    for frag in frags[:-1]:
        lookup = lookup.get(frag, dict())

    system_value = lookup.get(frags[-1], None)
    final_value = select(env_value, system_value, default)

    _track_options(name, etype, default, final_value)

    if final_value is None:
        return None
    try:
        value = etype(final_value)
        lookup[frags[-1]] = value
        return value
    except ValueError:
        print(f"{name}={value} expected type {etype} got {type(value)}")
        return None


def multirun(resumable=option("multirun.cache", str, None)):
    if resumable is not None:
        yield from _resumable_multirun(resumable)
    else:
        yield from _multirun()


def defaultfield(name, type, default=None):
    return field(default_factory=lambda: option(name, type, default))


def is_autoscale_enabled():
    return option("sizer.auto", int, 0) > 0


def default_save_location():
    from pathlib import Path
    return Path.home() / "new_scaling.yaml"


@dataclass
class SizerOptions:
    # overrides the batch size to use for all benchmarks
    batch_size: int = defaultfield("sizer.batch_size", int, None)

    # Add a fixed number to the current batch size
    add: int = defaultfield("sizer.add", int, None)

    # Add a fixed number to the current batch size
    mult: int = defaultfield("sizer.mult", float, None)

    # Enables auto batch resize
    auto: bool = defaultfield("sizer.auto", int, 0)

    # Constraint the batch size to be a multiple of a number
    multiple: int = defaultfield("sizer.multiple", int, 8)

    # Constraint the batch size to be a power of a specified base (usually 2)
    power: int = defaultfield("sizer.power", int)

    # Use the optimized batch size
    optimized: bool = defaultfield("sizer.optimized", int)

    # Set a target VRAM capacity to use
    capacity: str = defaultfield("sizer.capacity", str, None)

    # Save the batch size, VRM usage data to a scaling file
    save: str = defaultfield("sizer.save", str, None)

    # Configuration for batch scaling
    config: str = defaultfield("sizer.config", str, None)

    @property
    def autoscale(self):
        return self.enabled and self.multiple or self.capacity

    @property
    def enabled(self):
        return self.auto > 0

    @staticmethod
    def instance():
        system_config = system_global.get() or {}
        instance = SizerOptions(**system_config.get("options", {}).get("sizer", {}))
        return instance

    @property
    def size(self):
        return self.batch_size

@dataclass
class CPUOptions:
    enabled: bool = defaultfield("cpu.enabled", bool, False)

    total_count: bool = defaultfield("cpu.total_count", int, None)

    # max number of CPU per GPU
    max: int = defaultfield("cpu.max", int, 16)

    # min number of CPU per GPU
    min: int = defaultfield("cpu.min", int, 2)

    # reserved CPU cores (i.e not available for the benchmark)
    reserved_cores: int = defaultfield("cpu.reserved_cores", int, 0)

    # Number of workers (ignores cpu_max and cpu_min)
    n_workers: int = defaultfield("cpu.n_workers", int)

    @staticmethod
    def instance():
        system_config = system_global.get() or {}
        instance =  CPUOptions(**system_config.get("options", {}).get("cpu", {}))
        return instance

    @property
    def cpu_max(self):
        return self.max

    @property
    def cpu_min(self):
        return self.min
    

@dataclass
class DatasetConfig:
    # If use buffer is true then datasets are copied to the buffer before running the benchmark
    use_buffer: bool = defaultfield("data.use_buffer", bool, default=False)

    # buffer location to copy the datasets bfore running the benchmarks
    buffer: str = defaultfield("data.buffer", str, default="${dirs.base}/buffer")


def default_docker_args():
    return [
       "-it", "--rm", "--ipc=host", "--gpus=all",
        "--network", "host",
        "--privileged",
       "-e", f"MILABENCH_HF_TOKEN={os.getenv('MILABENCH_HF_TOKEN', 'Undefined')}",
       "-v", "/tmp/workspace/data:/milabench/envs/data",
       "-v", "/tmp/workspace/runs:/milabench/envs/runs",
    ]


@dataclass
class DockerConfig:
    executable: str = defaultfield("docker.executable", str, "podman")
    image: str  = defaultfield("docker.image", str, None)
    base: str = defaultfield("docker.base", str, "/tmp/workspace")
    args: list = defaultfield("docker.args", list, default_docker_args())

    def command(self, extra_args):
        return [
            self.executable,
            "run",
            *self.args,
            *extra_args,
            self.image
        ]


@dataclass
class Dirs:
    """Common directories used by milabench. This can be used to override
    location in case compute node do not have internet access."""
    venv: str = defaultfield("dirs.venv", str, default="${dirs.base}/venv/${install_group}")
    data: str = defaultfield("dirs.data", str, default="${dirs.base}/data")
    runs: str = defaultfield("dirs.runs", str, default="${dirs.base}/runs")
    extra: str = defaultfield("dirs.extra", str, default="${dirs.base}/extra/${group}")
    cache: str = defaultfield("dirs.cache", str, default="${dirs.base}/cache")


@dataclass 
class Torchrun:
    port: int = defaultfield("torchrun.port", int, default=29400)
    backend: str = defaultfield("torchrun.backend", str, default="c10d")

@dataclass
class Report:
    lean: bool = defaultfield("report.lean", int, default=0)


@dataclass
class Options:
    sizer: SizerOptions = field(default_factory=SizerOptions)
    cpu: CPUOptions = field(default_factory=CPUOptions)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dirs: Dirs = field(default_factory=Dirs)
    torchrun: Torchrun = field(default_factory=Torchrun)


@dataclass
class GPUConfig:
    arch: str = defaultfield("gpu.arch", str, None)
    capacity: str = defaultfield("gpu.capacity", str, str(get_gpu_capacity()))


@dataclass
class Nodes:
    name: str
    ip: str
    port: int
    main: bool
    user: str


@dataclass
class Github:
    pat: str = defaultfield("github.path", str, None)


def default_device():
    try:
        gpu_info = get_gpu_info()
        return gpu_info["arch"]
    except:
        return "cpu"


@dataclass
class SystemConfig:
    """This is meant to be an exhaustive list of all the environment overrides"""

    arch: str = defaultfield("gpu.arch", str, default_device())
    sshkey: str = defaultfield("ssh", str, "~/.ssh/id_rsa")
    docker_image: str = None
    nodes: list[Nodes] = field(default_factory=list)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    options: Options = field(default_factory=Options)

    base: str = defaultfield("base", str, None)
    config: str = defaultfield("config", str, None)
    dash: bool = defaultfield("dash", bool, 1)
    noterm: bool = defaultfield("noterm", bool, 0)
    github: Github = field(default_factory=Github)

    use_uv: bool = defaultfield("use_uv", bool, 0)

def check_node_config(nodes):
    mandatory_fields = ["name", "ip", "user"]

    for node in nodes:
        name = node.get("name", None)

        for field in mandatory_fields:
            assert field in node, f"The `{field}` of the node `{name}` is missing"


def build_system_config(config_file, defaults=None, gpu=True):
    """Load the system configuration, verify its validity and resolve ip addresses

    Notes
    -----
    * node['local'] true when the code is executing on the machine directly
    * node["main"] true when the machine is in charge of distributing the workload
    """

    if config_file is None:
        config = {"system": {}}
    else:
        config_file = XPath(config_file).absolute()
        with open(config_file) as cf:
            config = yaml.safe_load(cf)

    if defaults:
        config = merge(defaults, config)

    system = config.get("system", {})
    multirun = config.get("multirun", {})
    
    multirun_global.set(multirun)
    system_global.set(system)

    # capacity is only required if batch resizer is enabled
    if (gpu or is_autoscale_enabled()) and not "gpu" not in system:
        system["gpu"] = {"capacity": f"{int(get_gpu_capacity())} MiB"}

    if system.get("sshkey") is not None:
        system["sshkey"] = str(XPath(system["sshkey"]).resolve())

    check_node_config(system["nodes"])

    self = resolve_addresses(system["nodes"])
    system["self"] = self

    return config

def overrides_snapshot():
    return {name: value["value"] for name, value in _global_options.items()}


def show_overrides(to_json=False):
    import json
    import copy
    config = {}

    for name, value in _global_options.items():
        frags = name.split('.')

        dct = config
        for p in frags[:-1]:
            dct = dct.setdefault(p, dict())
            
        val_name = frags[-1]
        val = copy.deepcopy(value)

        val["type"] = str(val["type"].__name__)
        dct[val_name] = val
        val["env_name"] = as_environment_variable(name)

    def compact(d, depth):
        for k, v in d.items():
            idt = "    " * depth

            if "env_name" in v:
                value = v["value"]
                default = v["default"]
                if value != default:
                    print(f"{idt}{k:<{30 - len(idt)}}: {str(value):<40} (default={default})")
                else:
                    print(f"{idt}{k:<{30 - len(idt)}}: {str(value):<40} {v['env_name']}")
            else:
                print(f"{idt}{k}:")
                compact(v, depth + 1)
    
    if to_json:
        print(json.dumps(config, indent=2))
    else:
        compact(config, 0)
