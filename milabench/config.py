import socket

import yaml
from omegaconf import OmegaConf
import psutil

from .fs import XPath
from .merge import merge


def relative_to(pth, cwd):
    pth = XPath(pth).expanduser()
    if not pth.is_absolute():
        pth = (XPath(cwd) / pth).resolve()
    return pth


def _config_layers(config_files):
    for config_file in config_files:
        if isinstance(config_file, dict):
            yield config_file
        else:
            config_file = XPath(config_file).absolute()
            config_base = config_file.parent
            with open(config_file) as cf:
                config = yaml.safe_load(cf)
                includes = config.pop("include", [])
                if isinstance(includes, str):
                    includes = [includes]
                yield from _config_layers(
                    relative_to(incl, config_base) for incl in includes
                )
                for v in config.values():
                    assert isinstance(v, dict)
                    v.setdefault("config_base", str(config_base))
                    v.setdefault("config_file", str(config_file))
                    v.setdefault("dirs", {})
                yield config


def resolve_inheritance(bench_config, all_configs):
    while inherit := bench_config.pop("inherits", None):
        parent = all_configs[inherit]
        tags = {*parent.get("tags", []), *bench_config.get("tags", [])}
        bench_config = merge(parent, bench_config)
        bench_config["tags"] = sorted(tags)

    if "*" in all_configs:
        bench_config = merge(bench_config, all_configs["*"])

    return bench_config


def finalize_config(name, bench_config):
    bench_config["name"] = name
    if "definition" in bench_config:
        pack = XPath(bench_config["definition"]).expanduser()
        if not pack.is_absolute():
            pack = (XPath(bench_config["config_base"]) / pack).resolve()
            bench_config["definition"] = str(pack)

    bench_config["tag"] = [bench_config["name"]]

    bench_config = OmegaConf.to_object(OmegaConf.create(bench_config))
    return bench_config


def build_config(*config_files):
    all_configs = {}
    for layer in _config_layers(config_files):
        all_configs = merge(all_configs, layer)
    for name, bench_config in all_configs.items():
        all_configs[name] = resolve_inheritance(bench_config, all_configs)
    for name, bench_config in all_configs.items():
        all_configs[name] = finalize_config(name, bench_config)
    return all_configs


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
            # why is this happening
            hostname = hostname[:-len(".server.mila.quebec")]

        is_local = (
            ("127.0.0.1" in ipaddrlist)
            or (hostname in ("localhost", socket.gethostname()))
            or len(ip_list.intersection(ipaddrlist)) > 0
        )
        print()
        print("HERE", hostname, socket.gethostname())
        print()
        node["local"] = is_local

        if is_local:
            self = node
            node["ipaddrlist"] = list(ip_list)

    # if self is node we might be outisde the cluster
    # which explains why we could not resolve the IP of the nodes
    if self is not None and lazy_raise:
        raise RuntimeError("Could not resolve node ip") from lazy_raise

    return self


def build_system_config(config_file, defaults=None):
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

    if system.get("sshkey") is not None:
        system["sshkey"] = str(XPath(system["sshkey"]).resolve())

    check_node_config(system["nodes"])

    self = resolve_addresses(system["nodes"])
    system["self"] = self

    return config
