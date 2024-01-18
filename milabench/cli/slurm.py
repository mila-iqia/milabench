import getpass
import os

from coleo import tooled
from voir.instruments.gpu import get_gpu_info

from ..slurm import expand_node_list


@tooled
def cli_slurm_system():
    """Generate a system file based of slurm environment variables"""

    node_list = expand_node_list(os.getenv("SLURM_JOB_NODELIST", ""))

    def make_node(i, ip):
        node = {
            "name": ip,
            "ip": ip,
            "user": getpass.getuser(),
            "main": i == 0,
        }

        if i == 0:
            node["port"] = 8123

        return node

    capacity = float("+inf")

    for _, v in get_gpu_info("cuda")["gpus"].items():
        capacity = min(v["memory"]["total"], capacity)

    # nvidia-smi --query-gpu=memory.total --format=csv
    system = {
        "arch": "cuda",
        "gpu": {"capacity": f"{int(capacity)} MiB"},
        "nodes": [make_node(i, ip) for i, ip in enumerate(node_list)],
    }

    import yaml

    print(yaml.dump({"system": system}))