import os
import sys

from .commands import (
    CmdCommand,
    Command,
    ListCommand,
    SequenceCommand,
    SSHCommand,
    VoidCommand,
)

INSTALL_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def scp(node, folder, dest=None) -> list:
    """Copy a folder from local node to remote node"""
    host = node["ip"]
    user = node["user"]

    if dest is None:
        dest = folder

    return [
        "scp",
        "-CBr",
        "-P",
        folder,
        f"{user}@{host}:{dest}",
    ]


def rsync(node, folder, dest=None) -> list:
    """Copy a folder from local node to remote node"""
    host = node["ip"]
    user = node["user"]

    if dest is None:
        dest = os.path.abspath(os.path.join(folder, ".."))

    return [
        "rsync",
        "-av",
        "-e",
        "ssh -oCheckHostIP=no -oStrictHostKeyChecking=no",
        folder,
        f"{user}@{host}:{dest}",
    ]


def pip_install_milabench(pack, node, folder) -> SSHCommand:
    host = node["ip"]
    user = node["user"]

    cmd = ["pip", "install", "-e", folder]
    plan = CmdCommand(pack, *cmd)
    return SSHCommand(plan, host=host, user=user)


def milabench_remote_sync(pack, worker):
    setup_for = "worker"

    # If we are outside the system prepare main only
    # main will take care of preparing the workers
    if is_remote(pack):
        setup_for = "main"

    return milabench_remote_setup_plan(pack, setup_for)


def should_run_for(worker, setup_for):
    if setup_for == "worker":
        return not worker.get("main", False)

    return worker.get("main", False)


def worker_commands(pack, worker_plan, setup_for="worker"):
    nodes = pack.config["system"]["nodes"]
    copy = []
    node_packs = []

    for node in nodes:
        node_pack = None

        if should_run_for(node, setup_for):
            node_pack = worker_pack(pack, node)

            cmds = worker_plan(node_pack, node)

            if not isinstance(cmds, list):
                cmds = [cmds]
            copy.extend(cmds)

        node_packs.append(node_pack)

    return ListCommand(*copy)


def sshnode(node, cmd):
    host = node["ip"]
    user = node["user"]
    port = node["sshport"]
    return SSHCommand(cmd, user=user, host=host, port=port)


def copy_folder(pack, folder, setup_for="worker"):
    def copy_to_worker(nodepack, node):
        return [
             sshnode(node, CmdCommand(nodepack, "mkdir", "-p", folder)),
             CmdCommand(nodepack, *rsync(node, folder))
        ]
    return worker_commands(pack, copy_to_worker, setup_for=setup_for)



def milabench_remote_setup_plan(pack, setup_for="worker") -> SequenceCommand:
    """Copy milabench source files to remote

    Notes
    -----
    Assume that the filesystem of remote node mirror local system.
    """

    nodes = pack.config["system"]["nodes"]
    copy = []
    node_packs = []

    copy_source = copy_folder(pack, INSTALL_FOLDER, setup_for)

    install = []

    for i, node in enumerate(nodes):
        if should_run_for(node, setup_for):
            install.append(pip_install_milabench(node_packs[i], node, INSTALL_FOLDER))

    return SequenceCommand(
        copy_source,
        ListCommand(*install),
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


def milabench_remote_command(pack, *command, run_for="worker") -> ListCommand:
    nodes = pack.config["system"]["nodes"]
    key = pack.config["system"].get("sshkey")
    cmds = []

    for worker in nodes:
        if should_run_for(worker, run_for):
            host = worker["ip"]
            user = worker["user"]

            cmds.append(
                SSHCommand(
                    CmdCommand(worker_pack(pack, worker), "milabench", *command),
                    host=host,
                    user=user,
                    key=key,
                )
            )

    return ListCommand(*cmds)


def is_multinode(pack):
    """Return true if we have multiple nodes"""
    count = 0
    nodes = pack.config["system"]["nodes"]
    for node in nodes:
        if not node.get("main", False):
            count += 1
    return count > 0


def is_remote(pack):
    self = pack.config["system"]["self"]
    return self is None


def is_main_local(pack):
    """Only the local main can send remote commands to remote"""
    self = pack.config["system"]["self"]
    return self is not None and self["local"] and self.get("main", False)


def is_worker(pack):
    self = pack.config["system"]["self"]
    return self is not None and (not self.get("main", False))


def _sanity(pack, setup_for):
    if setup_for == "worker":
        assert is_main_local(pack), "Only main node can setup workers"

    if setup_for == "main":
        assert is_remote(pack), "Only a remote node can setup the main node"


def milabench_remote_install(pack, setup_for="worker") -> SequenceCommand:
    """Copy milabench code, install milabench, execute milabench install"""
    _sanity(pack, setup_for)

    if is_worker(pack):
        return VoidCommand(pack)

    argv = sys.argv[2:]

    return SequenceCommand(
        milabench_remote_setup_plan(pack, setup_for),
        milabench_remote_command(pack, "install", *argv, run_for=setup_for),
    )


def milabench_remote_prepare(pack, run_for="worker") -> Command:
    """Execute milabench prepare"""
    _sanity(pack, run_for)

    if is_worker(pack):
        return VoidCommand(pack)

    argv = sys.argv[2:]
    return milabench_remote_command(pack, "prepare", *argv, run_for=run_for)


def milabench_remote_run(pack) -> Command:
    """Execute milabench run"""

    # already on the main node, the regular flow
    # will be executed
    if is_main_local(pack):
        return VoidCommand(pack)

    argv = sys.argv[2:]
    return milabench_remote_command(pack, "run", *argv)
