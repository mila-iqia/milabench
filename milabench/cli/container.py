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


defaults_container = {
    "executable": "podman",
    "image": "ghcr.io/mila-iqia/milabench:cuda-nightly",
    "base": "/tmp/workspace",
    "args": [
       "--rm", "--ipc=host", "--network=host",
       "--device", "nvidia.com/gpu=all", 
       "--security-opt=label=disable",
       "-e", "MILABENCH_HF_TOKEN=$MILABENCH_HF_TOKEN",
       "-v", "$SSH_KEY_FILE:/root/.ssh/id_rsa:Z",
       "-v", "$MILABENCH_BASE/data:/milabench/envs/data",
       "-v", "$MILABENCH_BASE/cache:/milabench/envs/cache",
       "-v", "$MILABENCH_BASE/runs:/milabench/envs/runs",
    ]
}


def verify_setup():
    # Check expected environment variables are there
    # Check we can ssh to each nodes
    # ?
    ENV = {
        "MILABENCH_BASE",
        "SSH_KEY_FILE",
        "MILABENCH_HF_TOKEN",
    }


@tooled
def cli_docker(args=None):
    """Builds the docker command to execute from the system configuration"""
    from .run import arguments

    if args is None:
        args = arguments()

    verify_setup()

    mp = get_multipack(run_name=args.run_name)
    pack = dummy_pack(mp.packs)

    system = pack.config["system"]
    container_config = system.get("docker", defaults_container)
    config = DockerConfig(**container_config)

    working_directory = "..."

    #
    #   Auto GENERATE
    #

    # Create the file in 
    # with open("$MILABENCH_BASE/runs/system.yaml", "w") as fp:
    #     pass

    print()


    # mkdir -p $MILABENCH_BASE/runs
    # mkdir -p $MILABENCH_BASE/data
    # mkdir -p $MILABENCH_BASE/cache

    #
    # START
    # =====

    # PULL
    # ---

    plan = CmdCommand(pack, container_config.get("executable", "podman"), "pull", container_config.get("image"))
    for pack, argv, _ in plan.commands():
        print(" ".join(argv))

    print()

    # PREPARE
    # -------

    # This doesnot work because the prepare tries to rsync to the other node
    # but the other node is not the same layout as the container
    # so we have to execute prepare independently on all nodes
    plan = DockerRunCommand(CmdCommand(pack, "milabench", "prepare"), config)
    for pack, argv, _ in plan.commands():
        print(" ".join(argv))
    print()

    # RUN
    # ---
    extra_args = [
        "--system", "/milabench/envs/runs/system.yaml"
    ]

    plan = DockerRunCommand(CmdCommand(pack, "milabench", "run", *extra_args), config)
    for pack, argv, _ in plan.commands():
        print(" ".join(argv))
    print()

    # ARCHIVE
    # -------

    plan = CmdCommand(pack, "zip", "-r", "runs.zip", "$MILABENCH_BASE/runs")
    for pack, argv, _ in plan.commands():
        print(" ".join(argv))

    print()
