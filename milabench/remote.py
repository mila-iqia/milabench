import os
import sys

from .executors import (
    CmdExecutor,
    SSHExecutor,
    ListExecutor,
    SequenceExecutor,
    VoidExecutor,
    Executor,
)


INSTALL_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def scp(node, folder, dest=None) -> list:
    """Copy a folder from local node to remote node"""
    host = node["ip"]
    user = node["user"]
    port = node.get("port", 22)

    if dest is None:
        dest = folder

    return [
        "scp",
        "-CBr",
        "-P",
        str(port),
        folder,
        f"{user}@{host}:{dest}",
    ]


def rsync(node, folder, dest=None) -> list:
    """Copy a folder from local node to remote node"""
    host = node["ip"]
    user = node["user"]
    port = node.get("port", 22)

    if dest is None:
        dest = os.path.abspath(os.path.join(folder, ".."))

    return [
        "rsync",
        "-av",
        "-e",
        f"ssh -p {port}",
        folder,
        f"{user}@{host}:{dest}",
    ]


def pip_install_milabench(pack, node, folder) -> SSHExecutor:
    host = node["ip"]
    user = node["user"]

    cmd = ["pip", "install", "-e", folder]
    plan = CmdExecutor(pack, *cmd)
    return SSHExecutor(plan, host=host, user=user)


def milabench_remote_setup_plan(pack) -> SequenceExecutor:
    """Copy milabench source files to remote

    Notes
    -----
    Assume that the filesystem of remote node mirror local system.
    """

    nodes = pack.config["system"]["nodes"]
    copy = []
    node_packs = []

    for worker in nodes:
        node_pack = None

        if not worker["main"]:
            node_pack = worker_pack(pack, worker)
            copy.append(CmdExecutor(node_pack, *rsync(worker, INSTALL_FOLDER)))

        node_packs.append(node_pack)

    install = []
    for i, worker in enumerate(nodes):
        if not worker["main"]:
            install.append(pip_install_milabench(node_packs[i], worker, INSTALL_FOLDER))

    return SequenceExecutor(
        ListExecutor(*copy),
        ListExecutor(*install),
    )


def worker_pack(pack, worker):
    name = worker.get("name", worker.get("ip", "REMOTE"))

    return pack.copy(
        {
            "tag": dict(append=[f"{name}"]),
        }
    )


def milabench_remote_command(pack, *command) -> ListExecutor:
    nodes = pack.config["system"]["nodes"]
    cmds = []

    config = pack.config["config_file"]
    base = pack.config["dirs"]["base"]
    arch = pack.config["system"]["arch"]

    env = f"MILABENCH_CONFIG=${config} MILABENCH_BASE={base} MILABENCH_GPU_ARCH={arch} MILABENCH_REMOTE=1"

    for worker in nodes:
        if not worker["main"]:
            host = worker["ip"]
            user = worker["user"]
            port = worker.get("port", 22)

            cmds.append(
                SSHExecutor(
                    CmdExecutor(
                        worker_pack(pack, worker), f"{env} milabench", *command
                    ),
                    host=host,
                    user=user,
                    port=port,
                )
            )

    return ListExecutor(*cmds)


def is_multinode(pack):
    """Return true if we have multiple nodes"""
    count = 0
    nodes = pack.config["system"]["nodes"]
    for node in nodes:
        if not node["main"]:
            count += 1
    return count > 0


def is_main_local(pack):
    """Only the local main can send remote commands to remote"""
    self = pack.config["system"]["self"]
    return self["local"] and self["main"]


def milabench_remote_install(pack) -> SequenceExecutor:
    """Copy milabench code, install milabench, execute milabench install"""

    if not is_multinode(pack) or not is_main_local(pack):
        return VoidExecutor(pack)

    argv = sys.argv[2:]

    return SequenceExecutor(
        milabench_remote_setup_plan(pack),
        milabench_remote_command(pack, "install", *argv),
    )


def milabench_remote_prepare(pack) -> Executor:
    """Execute milabench prepare"""
    if not is_multinode(pack) or not is_main_local(pack):
        return VoidExecutor(pack)

    argv = sys.argv[2:]
    return milabench_remote_command(pack, "prepare", *argv)


def milabench_remote_run(pack) -> Executor:
    """Execute milabench run"""

    # already on the main node, the regular flow
    # will be executed
    if is_main_local():
        return VoidExecutor(pack)

    argv = sys.argv[2:]
    return milabench_remote_command(pack, "run", *argv)
