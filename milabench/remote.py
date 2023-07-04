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
        f"ssh -oCheckHostIP=no -oStrictHostKeyChecking=no -p {port}",
        folder,
        f"{user}@{host}:{dest}",
    ]


def pip_install_milabench(pack, node, folder) -> SSHExecutor:
    host = node["ip"]
    user = node["user"]

    cmd = ["pip", "install", "-e", folder]
    plan = CmdExecutor(pack, *cmd)
    return SSHExecutor(plan, host=host, user=user)


def milabench_remote_sync(pack, worker):
    setup_for = "worker"

    # If we are outside the system prepare main only
    # main will take care of preparing the workers
    if is_remote(pack):
        setup_for = "main"

    return milabench_remote_setup_plan(pack, setup_for)


def should_run_for(worker, setup_for):
    if setup_for == "worker":
        return not worker["main"]

    return worker["main"]


def milabench_remote_setup_plan(pack, setup_for="worker") -> SequenceExecutor:
    """Copy milabench source files to remote

    Notes
    -----
    Assume that the filesystem of remote node mirror local system.
    """

    nodes = pack.config["system"]["nodes"]
    copy = []
    node_packs = []

    for node in nodes:
        node_pack = None

        if should_run_for(node, setup_for):
            node_pack = worker_pack(pack, node)
            copy.append(CmdExecutor(node_pack, *rsync(node, INSTALL_FOLDER)))

        node_packs.append(node_pack)

    install = []
    for i, node in enumerate(nodes):
        if should_run_for(node, setup_for):
            install.append(pip_install_milabench(node_packs[i], node, INSTALL_FOLDER))

    return SequenceExecutor(
        ListExecutor(*copy),
        ListExecutor(*install),
    )


def worker_pack(pack, worker):
    if is_remote(pack):
        return pack.copy({})

    name = worker.get("name", worker.get("ip", "REMOTE"))
    return pack.copy(
        {
            "tag": dict(append=[f"{name}"]),
        }
    )


def milabench_remote_command(pack, *command, run_for="worker") -> ListExecutor:
    nodes = pack.config["system"]["nodes"]
    cmds = []

    for worker in nodes:
        if should_run_for(worker, run_for):
            host = worker["ip"]
            user = worker["user"]
            port = worker.get("port", 22)

            cmds.append(
                SSHExecutor(
                    CmdExecutor(worker_pack(pack, worker), f"milabench", *command),
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


def is_remote(pack):
    self = pack.config["system"]["self"]
    return self is None


def is_main_local(pack):
    """Only the local main can send remote commands to remote"""
    self = pack.config["system"]["self"]
    return self is not None and self["local"] and self["main"]


def is_worker(pack):
    self = pack.config["system"]["self"]
    return self is not None and (not self["main"])


def _sanity(pack, setup_for):
    if setup_for == "worker":
        assert is_main_local(pack), "Only main node can setup workers"

    if setup_for == "main":
        assert is_remote(pack), "Only a remote node can setup the main node"


def milabench_remote_install(pack, setup_for="worker") -> SequenceExecutor:
    """Copy milabench code, install milabench, execute milabench install"""
    _sanity(pack, setup_for)

    if is_worker(pack):
        return VoidExecutor(pack)

    argv = sys.argv[2:]

    return SequenceExecutor(
        milabench_remote_setup_plan(pack, setup_for),
        milabench_remote_command(pack, "install", *argv, run_for=setup_for),
    )


def milabench_remote_prepare(pack, run_for="worker") -> Executor:
    """Execute milabench prepare"""
    _sanity(pack, run_for)

    if is_worker(pack):
        return VoidExecutor(pack)

    argv = sys.argv[2:]
    return milabench_remote_command(pack, "prepare", *argv, run_for=run_for)


def milabench_remote_run(pack) -> Executor:
    """Execute milabench run"""

    # already on the main node, the regular flow
    # will be executed
    if is_main_local(pack):
        return VoidExecutor(pack)

    argv = sys.argv[2:]
    return milabench_remote_command(pack, "run", *argv)
