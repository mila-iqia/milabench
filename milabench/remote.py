import os

from .executors import CmdExecutor, SSHExecutor, ListExecutor, SequenceExecutor


INSTALL_FOLDER = os.path.abspath(os.path.dirname(__file__))


def scp(node, folder, dest=None):
    """Copy a folder from local node to remote node"""
    host = node["ip"]
    user = node["user"]
    
    if dest is None:
        dest = folder
            
    return [
        "scp",
        "-r",
        folder,
        f"{user}@{host}:{dest}",
    ]


def pip_install_milabench(pack, node, folder):
    host = node["ip"]
    user = node["user"]
    
    cmd = [
        "pip", "install", "-e", folder
    ]

    plan = CmdExecutor(pack, *cmd)
    return SSHExecutor(plan, host=host, user=user)
    

def milabench_remote_setup_plan(pack):
    """Copy milabench source files to remote
    
    Notes
    -----
    Assume that the filesystem of remote node mirror local system.
    """

    nodes = pack.config["system"]["nodes"]
    copy = []
    
    for worker in nodes:
        copy.append(CmdExecutor(pack, *scp(worker, INSTALL_FOLDER)))
        
    install = []
    for worker in nodes:
        install.append(pip_install_milabench(pack, worker, INSTALL_FOLDER))

    return SequenceExecutor(
        ListExecutor(*copy),
        ListExecutor(*install),
    )
    
def milabench_remote_command(pack, *command):
    nodes = pack.config["system"]["nodes"]
    cmds = []
    
    for worker in nodes:
        host = worker["ip"]
        user = worker["user"]
        
        cmds.append(SSHExecutor(CmdExecutor(pack, "milabench", *command), host=host, user=user))
        
    return ListExecutor(*cmds)


def milabench_remote_install(pack):
    return milabench_remote_command(pack, "install")


def milabench_remote_prepare(pack):
    return milabench_remote_command(pack, "prepare")
