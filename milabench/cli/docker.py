from __future__ import annotations

from coleo import Option, tooled


from ..commands import DockerRunCommand, CmdCommand
from ..pack import Package
from ..common import get_multipack
from ..system import DockerConfig


def dummy_pack(packs) -> Package:
    pack = list(packs.values())[0]
    name = "setup"

    return Package(
        {
            "name": name,
            "tag": [name],
            "definition": ".",
            "run_name": pack.config["run_name"],
            "dirs": pack.config["dirs"],
            "config_base": pack.config["config_base"],
            "config_file": pack.config["config_file"],
            "system": pack.config["system"],
        }
    )


@tooled
def cli_docker(args=None):
    """Builds the docker command to execute from the system configuration"""
    from .run import arguments

    if args is None:
        args = arguments()

    mp = get_multipack(run_name=args.run_name)

    pack = dummy_pack(mp.packs)

    system = pack.config["system"]
    config = DockerConfig(**system.get("docker"))

    # TODO: how can we generate this
    extra_args = [
        "--system", "/milabench/envs/data/system.yaml"
    ]
    print()
    print()

    # milabench prepare
    plan = DockerRunCommand(CmdCommand(pack, "milabench", "prepare", *extra_args), config)

    for pack, argv, _ in plan.commands():
        print(" ".join(argv))

    print()
    # milabench run

    for name, pack in mp.packs.items():
        plan = DockerRunCommand(CmdCommand(pack, "milabench", "run", "--select", name, *extra_args), config)

        for pack, argv, _ in plan.commands():
            print(" ".join(argv))

    print()