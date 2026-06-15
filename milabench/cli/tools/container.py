from __future__ import annotations

import getpass
import os
from dataclasses import dataclass

from argklass.command import Command
from argklass.arguments import argument
import yaml

from ...commands import DockerRunCommand, CmdCommand, ForeachNode
from ...pack import Package
from ...system import DockerConfig
from ...network import resolve_addresses
from ...common import run_with_loggers, TerminalFormatter
from ...utils import validation_layers


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
        ],
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
                "base": base,
            },
            "system": {**system},
        }
    )


def system_file(system_overrides, base, sshkey, token, image):
    return {
        **system_overrides,
        "docker": defaults_container(base, sshkey, token, image),
    }


def make_node(i, node):
    return {
        "name": "main" if i == 0 else f"worker-{i}",
        "ip": node,
        "user": getpass.getuser(),
        "main": i == 0,
    }


def verify_setup(args):
    args.sshkey = os.path.expanduser(args.sshkey)
    verify_cuda_setup(args)


def verify_rocm_setup(args):
    pass


def verify_cuda_setup(args):
    pass


def run_plan(plan, dry):
    if dry:
        for _, argv, _ in plan.commands():
            print(" ".join(argv))
        print()
        return 0
    else:
        def pretty_print(obj):
            print(obj)

        return_code = run_with_loggers(
            plan.execute(),
            loggers={
                TerminalFormatter(dump_config=False, pretty_print=pretty_print),
                *validation_layers("error"),
            },
        )
        if return_code != 0:
            raise Failed()


class Container(Command):
    """Builds the docker/podman command to execute from the system configuration."""

    name = "container"

    # fmt: off
    @dataclass
    class Arguments:
        """Builds the docker/podman command to execute from the system configuration."""
        base   : str       = "/tmp"             # Base path for output
        token  : str       = DEFAULT_TOKEN      # HuggingFace token
        image  : str       = NIGHTLY_IMAGE      # Container image to use
        sshkey : str       = DEFAULT_SSH_KEY    # Path to SSH private key
        node   : list[str] = argument(default=[], nargs="+")  # Node IP addresses
        args   : list[str] = argument(default=[], nargs="+")  # Extra arguments to pass to milabench
        dry    : bool      = False              # Dry run (print commands without executing)
    # fmt: on

    @staticmethod
    def execute(args):
        try:
            verify_setup(args)

            system = system_file({}, args.base, args.sshkey, args.token, args.image)
            container_config = system.get("docker")
            config = DockerConfig(**container_config)

            system["nodes"] = [make_node(i, node) for i, node in enumerate(args.node)]
            system["self"] = resolve_addresses(system["nodes"])

            os.makedirs(f"{args.base}/runs", exist_ok=True)
            with open(f"{args.base}/runs/system.yaml", "w") as fp:
                yaml.safe_dump({"system": system}, fp)

            mkdir = top_level_pack("mkdir", args.base, system)
            run_plan(ForeachNode(CmdCommand(mkdir, "mkdir", "-p", f"{args.base}/runs"), use_docker=False), args.dry)
            run_plan(ForeachNode(CmdCommand(mkdir, "mkdir", "-p", f"{args.base}/data"), use_docker=False), args.dry)
            run_plan(ForeachNode(CmdCommand(mkdir, "mkdir", "-p", f"{args.base}/cache"), use_docker=False), args.dry)

            pull = top_level_pack("pull", args.base, system)
            run_plan(
                ForeachNode(
                    CmdCommand(pull, container_config.get("executable", "podman"), "pull", container_config.get("image")),
                    use_docker=False,
                ),
                args.dry,
            )

            prepare = top_level_pack("prepare", args.base, system)
            more_args = []
            for arg in args.args:
                more_args.extend(arg.split(" "))
            more_args = list(filter(lambda x: x.strip() != "", more_args))

            run_plan(
                ForeachNode(
                    DockerRunCommand(CmdCommand(prepare, "/milabench/.env/bin/milabench", "prepare", *more_args), config),
                    use_docker=False,
                ),
                args.dry,
            )

            run = top_level_pack("run", args.base, system)
            extra_args = ["--system", "/milabench/envs/runs/system.yaml"]

            run_plan(
                DockerRunCommand(CmdCommand(run, "/milabench/.env/bin/milabench", "run", *extra_args, *more_args), config),
                args.dry,
            )

            zip_pack = top_level_pack("zip", args.base, system)
            run_plan(
                CmdCommand(zip_pack, "zip", "-r", "runs.zip", f"{args.base}/runs"),
                args.dry,
            )
        except Failed:
            pass


COMMANDS = Container
