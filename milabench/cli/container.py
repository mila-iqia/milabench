from __future__ import annotations

from dataclasses import dataclass, field
import subprocess
import getpass
import os


from coleo import Option, tooled
import yaml

from ..commands import DockerRunCommand, CmdCommand, ForeachNode
from ..pack import Package
from ..system import DockerConfig
from ..network import resolve_addresses
from ..common import run_with_loggers, TerminalFormatter
from ..utils import validation_layers


NIGHTLY_IMAGE = "ghcr.io/mila-iqia/milabench:cuda-nightly"
DEFAULT_TOKEN = "-"
DEFAULT_SSH_KEY = os.path.expanduser("~/.ssh/id_rsa")


class Failed(Exception):
    pass


def defaults_container(base, sshkey, token=DEFAULT_TOKEN, image=NIGHTLY_IMAGE):
    return {
        "executable": "podman",
        "image": image,
        "base": base,
        "args": [
            "--rm", "--ipc=host", "--network=host",
            "--device", "nvidia.com/gpu=all", 
            "--security-opt=label=disable",
            "-e", f"MILABENCH_HF_TOKEN={token}",
            "-e", f"HF_TOKEN={token}",
            "-v", f"{sshkey}:/root/.ssh/id_rsa:Z",
            "-v", f"{base}/data:/milabench/envs/data",
            "-v", f"{base}/cache:/milabench/envs/cache",
            "-v", f"{base}/runs:/milabench/envs/runs",
        ]
    }


def top_level_pack(name, base, system) -> Package:
    return Package(
        {
            "name": name,
            "tag": [name],
            "definition": ".",
            "run_name": "",
            "num_machines": len(system["nodes"]),
            "dirs": {
                "extra": f"{base}/extra",
                "cache": f"{base}/cache",
                "venv": f"{base}/env",
                "base": base
            },
            "system": {
                **system,
            }
        }
    )



def system_file(system_overrides, base, sshkey, token, image):
    return {
        **system_overrides,
        "docker": defaults_container(base, sshkey, token, image)
    }



@dataclass
class Arguments:
    base: str = "/tmp"
    token: str = DEFAULT_TOKEN
    image: str = NIGHTLY_IMAGE
    sshkey: str = DEFAULT_SSH_KEY
    args: str = field(default_factory=list)
    nodes: list = field(default_factory=list)
    dry: bool = False


@tooled
def arguments() -> Arguments:
    base: Option & str = "/tmp"

    token: Option & str = DEFAULT_TOKEN

    image: Option & str = NIGHTLY_IMAGE

    sshkey: Option & str = DEFAULT_SSH_KEY
    
    # [nargs: +]
    node: Option & str = []

    # [nargs: +]
    args: Option & str = []

    dry: Option & bool = False

    return Arguments(base, token, image, sshkey, args, node, dry)


def make_node(i, node):
    return {
        "name": "main" if i == 0 else f"worker-{i}",
        "ip": node,
        "user": getpass.getuser(),
        "main": i == 0,
    }


def verify_setup(args: Arguments):
    """Check everything that could have been missed before running milabench.
    
    Each check should analyze the error message and recommend steps to resolve the issue.
    """

    args.sshkey = os.path.expanduser(args.sshkey)

    # Verify environments/arguments ?
    #
    #
    # Verify disk space
    #   du > required size
    #
    # Verify Podman
    #   podman --version
    #   pdoman run --rm ubuntu echo "hello"
    #
    # Verify SSH key file
    #   cat SSH_FILE
    #
    # Verify we can SSH
    #   ssh node1
    #   ssh node2
    #
    # Verify gated models are accessible
    #   ?
    verify_cuda_setup(Arguments)


def verify_rocm_setup(args: Arguments):
    # Verify that user DOES belong to the render and video group
    #
    # Verify driver
    #   rocm-smi
    #
    pass


def verify_cuda_setup(args: Arguments):
    PACKAGES = {
        "podman": "",
        "dkms": "",
        "build-essential": "",
        "nvidia-container-toolkit": "",
        "nvidia-driver-580-server": "",
    }

    #
    # Verify NVIDIA driver
    #   nvidia-smi
    #
    # Verify nvidia-container-toolkit
    #   nvidia-ctk  --version
    #
    # Verify required files
    #   Check for CDI files `/etc/cdi/nvidia.yaml`
    #       cat /etc/cdi/nvidia.yaml
    #
    # Verify NVIDIA container+podman
    #   podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable ubuntu nvidia-smi -L
    #

def run_plan(plan, dry):
    if dry:
        for _, argv, _ in plan.commands():
            print(" ".join(argv))
        print()
        return 0
    else:
        # This is a corountine
        return_code = run_with_loggers(
            plan.execute(),
            loggers = {
                TerminalFormatter(dump_config=False),
                *validation_layers("error")
            }
        )
        if return_code != 0:
            raise Failed()


@tooled
def cli_docker(args=None):
    """Builds the docker command to execute from the system configuration"""

    # pip install git+https://github.com/mila-iqia/milabench.git
    # milabench container                                       \
    #       --base ~/output                                     \
    #       --image ghcr.io/mila-iqia/milabench:cuda-nightly    \
    #       --node 192.168.0.1 192.168.0.2                      \
    #       --token iowhoiwehoiwehfoiwehfoiwehf                 \
    #       --sshkey ~/.ssh/id_rsa                              \
    #       --args "--select multinode"                         \
    #       --dry
    

    try:
        if args is None:
            args = arguments()

        verify_setup(args)

        system = system_file({}, args.base, args.sshkey, args.token, args.image)
        container_config = system.get("docker")
        config = DockerConfig(**container_config)

        #
        # START
        # =====

        # Generate System config
        # ----------------------
        system["nodes"] = [
            make_node(i, node) for i, node in enumerate(args.nodes)
        ]

        system["self"] = resolve_addresses(system["nodes"])

        # Create the file in 
        os.makedirs(f"{args.base}/runs", exist_ok=True)
        with open(f"{args.base}/runs/system.yaml", "w") as fp:
            yaml.safe_dump({"system": system}, fp)
        
        # Make directories
        # ----------------
        mkdir = top_level_pack("mkdir", args.base, system)

        run_plan(ForeachNode(CmdCommand(mkdir, "mkdir", "-p", f"{args.base}/runs"), use_docker=False), args.dry)
        run_plan(ForeachNode(CmdCommand(mkdir, "mkdir", "-p", f"{args.base}/data"), use_docker=False), args.dry)
        run_plan(ForeachNode(CmdCommand(mkdir, "mkdir", "-p", f"{args.base}/cache"), use_docker=False), args.dry)

        # PULL
        # ---
        pull = top_level_pack("pull", args.base, system)
        run_plan(
            ForeachNode(
                CmdCommand(pull, container_config.get("executable", "podman"), "pull", container_config.get("image")), use_docker=False
            ),
            args.dry
        )

        # PREPARE
        # -------
        prepare = top_level_pack("prepare", args.base, system)
        more_args = []
        for arg in args.args:
            more_args.extend(arg.split(" "))
        more_args = list(filter(lambda x: x.strip() != "", more_args))

        run_plan(
            ForeachNode(
                DockerRunCommand(CmdCommand(prepare, "/milabench/.env/bin/milabench", "prepare", *more_args), config), use_docker=False
            ),
            args.dry
        )
        
        # RUN
        # ---
        run = top_level_pack("run", args.base, system)

        extra_args = [
            "--system", "/milabench/envs/runs/system.yaml",
        ]

        run_plan(
            DockerRunCommand(CmdCommand(run, "/milabench/.env/bin/milabench", "run", *extra_args, *more_args), config),
            args.dry,
        )

        # ARCHIVE
        # -------
        zip = top_level_pack("zip", args.base, system)
        run_plan(
            CmdCommand(zip, "zip", "-r", "runs.zip", f"{args.base}/runs"),
            args.dry
        )
    except Failed:
        pass