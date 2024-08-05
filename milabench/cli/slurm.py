import getpass
import os
import socket
import subprocess
from coleo import tooled

from ..system import get_gpu_capacity


def gethostname(host):
    try:
        return subprocess.check_output(["ssh", host, "cat", "/etc/hostname"], text=True).strip()
    except:
        print("Could not resolve hostname")
        return host


def getip(ip):
    # This does get a good IP for everything except the local node
    
    hostname, _, iplist = socket.gethostbyaddr(ip)
    if len(iplist) > 1:
        print("Multiple IP found")


    from milabench.system import get_remote_ip

    resolved = iplist[0]
    if resolved.startswith("127.0"):
        ips = get_remote_ip()
        for ip in ips:
            if "." in ip and not ip.startswith("127.0"):
                return ip
    
        return resolved
    
    return resolved


@tooled
def cli_slurm_system():
    """Generate a system file based of slurm environment variables"""

    node_list = expand_node_list(os.getenv("SLURM_JOB_NODELIST", ""))

    def make_node(i, ip):
        node = {
            "name": ip,
            "ip": getip(ip),
            "hostname": gethostname(ip),
            "user": getpass.getuser(),
            "main": i == 0,
            "sshport": 22,
        }

        if i == 0:
            node["port"] = 8123

        return node

    # nvidia-smi --query-gpu=memory.total --format=csv
    system = {
        "arch": "cuda",
        "nodes": [make_node(i, ip) for i, ip in enumerate(node_list)],
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
