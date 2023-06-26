import os

from .executors import CmdExecutor, SSHExecutor, ListExecutor, SequenceExecutor


INSTALL_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# docker pull kabirbaidhya/fakeserver
# docker run -d -p 2222:22 \
#             -v "/tmp/milabench/tests/config:/etc/authorized_keys/tester" \
#             -e SSH_USERS="tester:1001:1001" \
#             --name=fakeserver kabirbaidhya/fakeserver


def scp(node, folder, dest=None):
    """Copy a folder from local node to remote node"""
    host = node["ip"]
    user = node["user"]
    port = node.get("port", 22)
    
    if dest is None:
        dest = folder
            
    return [
        "scp",
        "-CBr",
        "-vvv",
        "-P", str(port),
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
        if not worker["main"]:
            copy.append(CmdExecutor(pack, *scp(worker, INSTALL_FOLDER)))
        
    install = []
    for worker in nodes:
        if not worker["main"]:
            install.append(pip_install_milabench(pack, worker, INSTALL_FOLDER))

    return (
        ListExecutor(*copy),
        ListExecutor(*install),
    )

def milabench_remote_command(pack, *command):
    nodes = pack.config["system"]["nodes"]
    cmds = []
    
    for worker in nodes:
        if not worker["main"]:
            host = worker["ip"]
            user = worker["user"]
            port = worker.get("port", 22)
            
            cmds.append(SSHExecutor(CmdExecutor(pack, "milabench", *command), host=host, user=user, port=port))
            
    return ListExecutor(*cmds)


def milabench_remote_install(pack):
    """Copy milabench code, install milabench, execute milabench install"""
    return SequenceExecutor(
        *milabench_remote_setup_plan(pack),
        milabench_remote_command(pack, "install")
    )


def milabench_remote_prepare(pack):
    """Execute milabench prepare"""
    return milabench_remote_command(pack, "prepare")
