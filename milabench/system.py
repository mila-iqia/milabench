import contextvars
import os
import socket
from dataclasses import dataclass, field

import psutil
import yaml
from voir.instruments.gpu import get_gpu_info

from .fs import XPath
from .merge import merge

system_global = contextvars.ContextVar("system", default=None)


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
            print(*args, **kwargs)
            printed += 1

    return _print


warn_no_config = print_once("No system config found, using defaults")


def option(name, etype, default=None):
    options = dict()
    system = system_global.get()

    if system:
        options = system.get("options", dict())
    else:
        warn_no_config()

    frags = name.split(".")
    env_name = "MILABENCH_" + "_".join(map(str.upper, frags))
    env_value = getenv(env_name, etype)

    lookup = options
    for frag in frags[:-1]:
        lookup = lookup.get(frag, dict())

    system_value = lookup.get(frags[-1], None)
    final_value = env_value or system_value or default

    if final_value is None:
        return None

    try:
        value = etype(final_value)
        lookup[frags[-1]] = value
        return value
    except ValueError:
        print(f"{name}={value} expected type {etype} got {type(value)}")
        return None


def is_autoscale_enabled():
    return option("sizer.auto", int, 0) > 0


def default_save_location():
    from pathlib import Path

    return Path.home() / "new_scaling.yaml"


@dataclass
class SizerOptions:
    size: int = option("sizer.batch_size", int)
    autoscale: bool = option("sizer.auto", int, 0)
    multiple: int = option("sizer.multiple", int, 8)
    power: int = option("sizer.power", int)
    optimized: bool = option("sizer.optimized", int)
    capacity: str = option("sizer.capacity", str)
    save: str = option("sizer.save", str, None)


@dataclass
class CPUOptions:
    enabled: bool = option("cpu.auto", bool, False)

    # max number of CPU per GPU
    cpu_max: int = option("cpu.max", int, 16)

    # min number of CPU per GPU
    cpu_min: int = option("cpu.min", int, 2)

    # reserved CPU cores (i.e not available for the benchmark)
    reserved_cores: int = option("cpu.reserved_cores", int, 0)

    # Number of workers (ignores cpu_max and cpu_min)
    n_workers: int = option("cpu.n_workers", int)


@dataclass
class Options:
    sizer: SizerOptions
    cpu: CPUOptions


@dataclass
class GPUConfig:
    capacity: str = None


@dataclass
class Nodes:
    name: str
    ip: str
    port: int
    main: bool
    user: str


@dataclass
class SystemConfig:
    arch: str = getenv("MILABENCH_GPU_ARCH", str)
    sshkey: str = None
    docker_image: str = None
    nodes: list[Nodes] = field(default_factory=list)
    gpu: GPUConfig = None
    options: Options = None


def check_node_config(nodes):
    mandatory_fields = ["name", "ip", "user"]

    for node in nodes:
        name = node.get("name", None)

        for field in mandatory_fields:
            assert field in node, f"The `{field}` of the node `{name}` is missing"


def get_remote_ip():
    """Get all the ip of all the network interfaces"""
    addresses = psutil.net_if_addrs()
    stats = psutil.net_if_stats()

    result = []

    for interface, address_list in addresses.items():
        for address in address_list:
            if interface in stats and getattr(stats[interface], "isup"):
                result.append(address.address)

    return set(result)


def _resolve_ip(ip):
    # Resolve the IP
    try:
        hostname, aliaslist, ipaddrlist = socket.gethostbyaddr(ip)
        lazy_raise = None
    except socket.gaierror as err:
        # Get Addr Info (GAI) Error
        #
        # When we are connecting to a node through a ssh proxy jump
        # the node IPs/Hostnames are not available until we reach
        # the first node inside the cluster
        #
        hostname = ip
        aliaslist = []
        ipaddrlist = []
        lazy_raise = err

    return hostname, aliaslist, ipaddrlist, lazy_raise


def resolve_addresses(nodes):
    # Note: it is possible for self to be none
    # if we are running milabench on a node that is not part of the system
    # in that case it should still work; the local is then going to
    # ssh into the main node which will dispatch the work to the other nodes
    self = None
    lazy_raise = None
    ip_list = get_remote_ip()

    for node in nodes:
        hostname, aliaslist, ipaddrlist, lazy_raise = _resolve_ip(node["ip"])

        node["hostname"] = hostname
        node["aliaslist"] = aliaslist
        node["ipaddrlist"] = ipaddrlist

        if hostname.endswith(".server.mila.quebec.server.mila.quebec"):
            print()
            print("Hostname was extra long for no reason")
            print(hostname, socket.gethostname())
            print()

            # why is this happening
            hostname = hostname[: -len(".server.mila.quebec")]

        is_local = (
            ("127.0.0.1" in ipaddrlist)
            or (hostname in ("localhost", socket.gethostname()))
            or len(ip_list.intersection(ipaddrlist)) > 0
        )
        node["local"] = is_local

        if is_local:
            self = node
            node["ipaddrlist"] = list(ip_list)

    # if self is node we might be outisde the cluster
    # which explains why we could not resolve the IP of the nodes
    if self is not None and lazy_raise:
        raise RuntimeError("Could not resolve node ip") from lazy_raise

    return self


def get_gpu_capacity(strict=False):
    try:
        capacity = 0

        for k, v in get_gpu_info()["gpus"].items():
            capacity = min(v["memory"]["total"], capacity)

        return capacity
    except:
        print("GPU not available, defaulting to 0 MiB")
        if strict:
            raise
        return 0


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
