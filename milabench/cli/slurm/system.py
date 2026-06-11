"""Generate a system configuration file from Slurm environment."""

import getpass
import os
import socket
import subprocess
from dataclasses import dataclass
from typing import Optional

from argklass.command import Command

from ...system import get_gpu_capacity
from ...network import resolve_ip


def gethostname(host):
    try:
        return subprocess.check_output([
            "ssh",
            "-oCheckHostIP=no",
            "-oPasswordAuthentication=no",
            "-oStrictHostKeyChecking=no", host, "cat", "/etc/hostname"], text=True).strip()
    except:
        print("Could not resolve hostname")
        return host


def make_node_list_from_slurm(node_list):
    def make_node(i, ip):
        hostname, real_ip, local = resolve_ip(ip)

        node = {
            "name": ip,
            "ip": real_ip,
            "hostname": gethostname(ip),
            "user": getpass.getuser(),
            "main": local,
            "sshport": 22,
        }

        if i == 0:
            node["port"] = 8123

        return node

    nodes = [make_node(i, ip) for i, ip in enumerate(node_list)]

    for node in nodes:
        if node.get("main", False):
            break
    else:
        nodes[0]["main"] = True

    return nodes


def expand_range(s):
    numbers = []
    count = 0

    for i in s.split(","):
        if "-" not in i:
            count = len(i)
            numbers.append(i)
        else:
            start, end = i.split("-")
            count = len(start)

            for n in range(int(start), int(end) + 1):
                numbers.append(f"{n:0{count}d}")

    return numbers


def expand_node_list(node_list):
    nodes = []
    s = 0

    while s < len(node_list):
        if node_list[s] == ",":
            s += 1

        next = node_list.find(",", s)
        range_start = node_list.find("[", s)
        range_end = node_list.find("]", s)

        if range_start != -1 and (next == -1 or range_start < next):
            node_name = node_list[s:range_start]

            range = node_list[range_start + 1 : range_end]

            for i in expand_range(range):
                nodes.append(f"{node_name}{i}")

            s = range_end + 1

        else:
            if next == -1:
                next = len(node_list)

            node_name = node_list[s:next]
            nodes.append(node_name)

            s = next + 1

    return nodes


def cli_slurm_system(args):
    """Generate a system file based of slurm environment variables"""

    node_list = expand_node_list(os.getenv("SLURM_JOB_NODELIST", ""))

    if len(node_list) > 0:
        nodes = make_node_list_from_slurm(node_list)
    else:
        self = socket.gethostname()
        nodes = [{
            "name": self,
            "ip": self,
            "hostname": self,
            "user": getpass.getuser(),
            "main": True,
            "sshport": 22,
        }]

    from milabench.network import resolve_addresses
    resolve_addresses(nodes)

    system = {
        "arch": "cuda",
        "nodes": nodes,
    }

    capacity = get_gpu_capacity()
    if capacity > 0:
        system["gpu"] = {
            "capacity": f"{capacity} MiB"
        }

    import yaml

    print(yaml.dump({"system": system}))

    if args.output:
        with open(args.output, "w") as f:
            yaml.dump({"system": system}, f)


class System(Command):
    """Generate a system configuration from Slurm environment."""

    name = "system"

    # fmt: off
    @dataclass
    class Arguments:
        """Generate a system configuration from Slurm environment."""
        output : Optional[str] = None  # Output file for the generated configuration
    # fmt: on

    @staticmethod
    def execute(args):
        cli_slurm_system(args)


COMMANDS = System
