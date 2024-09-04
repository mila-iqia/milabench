import getpass
import os
import socket
import subprocess
from coleo import tooled

from ..system import get_gpu_capacity, is_loopback, resolve_hostname, gethostname




def make_node_list_from_slurm(node_list):
    def make_node(i, ip):
        hostname, local = resolve_hostname(ip)

        node = {
            "name": ip,
            "ip": hostname,
            "hostname": gethostname(ip),
            "user": getpass.getuser(),
            "main": local,
            "sshport": 22,
        }

        if i == 0:
            node["port"] = 8123

        return node

    # nvidia-smi --query-gpu=memory.total --format=csv

    nodes = [make_node(i, ip) for i, ip in enumerate(node_list)]

    # ensure there is a main
    # either it is the local node or first node
    for node in nodes:
        if node.get("main", False):
            break
    else:
        nodes[0]["main"] = True

    return nodes


@tooled
def cli_slurm_system():
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


    from milabench.system import resolve_addresses
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

        # Found a range
        if range_start != -1 and (next == -1 or range_start < next):
            node_name = node_list[s:range_start]

            range = node_list[range_start + 1 : range_end]

            for i in expand_range(range):
                nodes.append(f"{node_name}{i}")

            # eat the ]
            s = range_end + 1

        else:
            if next == -1:
                next = len(node_list)

            node_name = node_list[s:next]
            nodes.append(node_name)

            # eat the ,
            s = next + 1

    return nodes
