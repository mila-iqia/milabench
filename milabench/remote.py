import os
import sys

from .executors import CmdExecutor, SSHExecutor, ListExecutor, SequenceExecutor, VoidExecutor


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
        "-P", str(port),
        folder,
        f"{user}@{host}:{dest}",
    ]


def rsync(node, folder, dest=None):
    """Copy a folder from local node to remote node"""
    host = node["ip"]
    user = node["user"]
    port = node.get("port", 22)
    
    if dest is None:
        dest = os.path.abspath(os.path.join(folder, ".."))
            
    return [
        "rsync",
        "-av",
        "-e", f"ssh -p {port}",
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
    node_packs = []
    
    for worker in nodes:
        node_pack = None
        
        if not worker["main"]:
            host = worker["ip"]
            
            # Make a pack per node
            node_pack = pack.copy({
                "name": f"W{host}",
                "tag": [f"W{host}"],
            })
            copy.append(CmdExecutor(node_pack, *rsync(worker, INSTALL_FOLDER)))

        node_packs.append(node_pack)
        
    install = []
    for i, worker in enumerate(nodes):
        if not worker["main"]:
            install.append(pip_install_milabench(node_packs[i], worker, INSTALL_FOLDER))

    return (
        ListExecutor(*copy),
        ListExecutor(*install),
    )

def milabench_remote_command(pack, *command):
    nodes = pack.config["system"]["nodes"]
    cmds = []
    
    config = os.getenv("MILABENCH_CONFIG", "")
    base = os.getenv("MILABENCH_BASE", "")
    arch = os.getenv("MILABENCH_GPU_ARCH", "")
    
    env = f"MILABENCH_CONFIG=${config} MILABENCH_BASE={base} MILABENCH_GPU_ARCH={arch} MILABENCH_REMOTE=1"
    
    for worker in nodes:
        if not worker["main"]:
            host = worker["ip"]
            user = worker["user"]
            port = worker.get("port", 22)
            
            # Make a pack per node
            node_pack = pack.copy({
                "name": f"W{host}",
                "tag": [f"W{host}"],
            })
            cmds.append(SSHExecutor(CmdExecutor(node_pack, f"{env} milabench", *command), host=host, user=user, port=port))
            
    return ListExecutor(*cmds)


def is_main():
    return int(os.getenv("MILABENCH_REMOTE", "0")) == 0


def is_remote():
    return not is_main()

def has_nodes(pack):
    count = 0
    nodes = pack.config["system"]["nodes"]
    for node in nodes:
        if not node["main"]:
            count += 1
    return count > 0


def milabench_remote_install(pack):
    """Copy milabench code, install milabench, execute milabench install"""
    if not has_nodes(pack) or is_remote():
        return VoidExecutor(pack)
    
    argv = sys.argv[2:]
    
    return SequenceExecutor(
        *milabench_remote_setup_plan(pack),
        milabench_remote_command(pack, "install", *argv)
    )


def milabench_remote_prepare(pack):
    """Execute milabench prepare"""
    if not has_nodes(pack) or is_remote():
        return VoidExecutor(pack)
    
    argv = sys.argv[2:]
    return milabench_remote_command(pack, "prepare", *argv)
