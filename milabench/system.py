import contextvars
import os
import socket
from dataclasses import dataclass, field
import sys
import subprocess
from contextlib import contextmanager
import ipaddress

import psutil
import yaml
from voir.instruments.gpu import get_gpu_info

from .fs import XPath
from .merge import merge

system_global = contextvars.ContextVar("system", default=None)


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
    final_value = env_value or system_value or default

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
    size: int = defaultfield("sizer.batch_size", int, None)

    # Enables auto batch resize
    autoscale: bool = defaultfield("sizer.auto", int, 0)

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


@dataclass
class CPUOptions:
    enabled: bool = defaultfield("cpu.auto", bool, False)

    total_count: bool = defaultfield("cpu.total_count", int, None)

    # max number of CPU per GPU
    cpu_max: int = defaultfield("cpu.max", int, 16)

    # min number of CPU per GPU
    cpu_min: int = defaultfield("cpu.min", int, 2)

    # reserved CPU cores (i.e not available for the benchmark)
    reserved_cores: int = defaultfield("cpu.reserved_cores", int, 0)

    # Number of workers (ignores cpu_max and cpu_min)
    n_workers: int = defaultfield("cpu.n_workers", int)


@dataclass
class DatasetConfig:
    # If use buffer is true then datasets are copied to the buffer before running the benchmark
    use_buffer: bool = defaultfield("data.use_buffer", bool, default=False)

    # buffer location to copy the datasets bfore running the benchmarks
    buffer: str = defaultfield("data.buffer", str, default="${dirs.base}/buffer")


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
class Options:
    sizer: SizerOptions = SizerOptions()
    cpu: CPUOptions = CPUOptions() 
    dataset: DatasetConfig = DatasetConfig()
    dirs: Dirs = Dirs()
    torchrun: Torchrun = Torchrun()


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
    gpu: GPUConfig = GPUConfig()
    options: Options = Options()

    base: str = defaultfield("base", str, None)
    config: str = defaultfield("config", str, None)
    dash: bool = defaultfield("dash", bool, 1)
    noterm: bool = defaultfield("noterm", bool, 0)
    github: Github = Github()


def check_node_config(nodes):
    mandatory_fields = ["name", "ip", "user"]

    for node in nodes:
        name = node.get("name", None)

        for field in mandatory_fields:
            assert field in node, f"The `{field}` of the node `{name}` is missing"


def get_remote_ip():
    """Get all the ip of all the network interfaces"""
    if offline:
        return set()

    addresses = psutil.net_if_addrs()
    stats = psutil.net_if_stats()

    result = []

    for interface, address_list in addresses.items():
        for address in address_list:
            if interface in stats and getattr(stats[interface], "isup"):
                result.append(address.address)

    return set(result)


def is_loopback(address: str) -> bool:
    try:
        # Create an IP address object
        ip = ipaddress.ip_address(address)
        # Check if the address is a loopback address
        return ip.is_loopback
    except ValueError:
        # If the address is invalid, return False
        return False



def _resolve_ip(ip):
    hostname = ip
    aliaslist = []
    ipaddrlist = [ip]
    lazy_raise = None

    if not offline:
        # Resolve the IP
        try:
            hostname, aliaslist, ipaddrlist = socket.gethostbyaddr(ip)
            lazy_raise = None
        
        except socket.herror as err:
            lazy_raise = err

        except socket.gaierror as err:
            # Get Addr Info (GAI) Error
            #
            # When we are connecting to a node through a ssh proxy jump
            # the node IPs/Hostnames are not available until we reach
            # the first node inside the cluster
            #
            lazy_raise = err

    return hostname, aliaslist, ipaddrlist, lazy_raise


def _fix_weird(hostname):
    if hostname.endswith(".server.mila.quebec.server.mila.quebec"):
        print()
        print("Hostname was extra long for no reason")
        print(hostname, socket.gethostname())
        print()

        # why is this happening
        hostname = hostname[: -len(".server.mila.quebec")]
    
    return hostname


# If true that means we cannot resolve the ip addresses
# so we ignore errors
offline = False


@contextmanager
def enable_offline(enabled):
    global offline
    old = offline

    offline = enabled
    yield
    offline = old


def _resolve_addresses(nodes):
    # Note: it is possible for self to be none
    # if we are running milabench on a node that is not part of the system
    # in that case it should still work; the local is then going to
    # ssh into the main node which will dispatch the work to the other nodes
    self = None
    lazy_raise = None
    ip_list = get_remote_ip()

    for node in nodes:
        hostname, aliaslist, ipaddrlist, lazy_raise = _resolve_ip(node["ip"])

        hostname = _fix_weird(hostname)

        node["hostname"] = hostname
        node["aliaslist"] = aliaslist
        node["ipaddrlist"] = ipaddrlist

        is_local = (
            ("127.0.0.1" in ipaddrlist)
            or (hostname in ("localhost", socket.gethostname(), "127.0.0.1"))
            or (socket.gethostname().startswith(hostname))
            or len(ip_list.intersection(ipaddrlist)) > 0
            or any([is_loopback(ip) for ip in ipaddrlist])
        )

        # cn-g005 cn-g005.server.mila.quebec
        # print(hostname, socket.gethostname())
        node["local"] = is_local

        if is_local:
            self = node
            node["ipaddrlist"] = list(set(list(ip_list) + list(ipaddrlist)))

    # if self is node we might be outisde the cluster
    # which explains why we could not resolve the IP of the nodes
    if not offline:
        if self is not None and lazy_raise:
            raise RuntimeError("Could not resolve node ip") from lazy_raise

    return self


def gethostname(host):
    try:
        return subprocess.check_output(["ssh", host, "cat", "/etc/hostname"], text=True).strip()
    except:
        print("Could not resolve hostname")
        return host


def resolve_hostname(ip):
    try:
        hostname, _, iplist = socket.gethostbyaddr(ip)

        for ip in iplist:
            if is_loopback(ip):
                return hostname, True

        return hostname, False

    except:
        if offline:
            return ip, False

        raise

def resolve_node_address(node):
    hostname, local = resolve_hostname(node["ip"])

    node["hostname"] = hostname
    node["local"] = local

    if local:
        # `gethostbyaddr` returns `cn-d003` but we want `cn-d003.server.mila.quebec`
        # else torchrun does not recognize the main node
        node["hostname"] = socket.gethostname()
        
    return local


def resolve_addresses(nodes):
    self = None
    
    for node in nodes:
        if resolve_node_address(node):
            self = node

    return self


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


if __name__ == "__main__":
    show_overrides()
