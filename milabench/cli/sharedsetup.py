from dataclasses import dataclass
import os
import subprocess

from coleo import Option, tooled


@dataclass
class Arguments:
    network: str
    local: str
    remote_data: str = None


@tooled
def arguments():
    network: Option & str

    local: Option & str = "/tmp/workspace"

    remote_data: Option & str = None

    return Arguments(network, local, remote_data)


@tooled
def cli_shared_setup(args = None):
    
    if args is None:
        args = arguments()
    
    remote_code = os.path.join(args.network, "venv")
    local_code  = os.path.join(args.local, "venv")
    remote_data = os.path.join(args.network, "data")
    local_data  = os.path.join(args.local, "data")

    assert os.path.exists(remote_code), "missing venv, was milabench install run ?"
    assert os.path.exists(remote_data), "missing data, was milabench prepare run ?"
    
    os.makedirs(args.local, exist_ok=True)

    # rsync datasets & checkpoints to local disk
    subprocess.check_call(["rsync", "-azh", "--info=progress2", "--partial", remote_data, args.local])

    # create a soft link for the code
    try:
        os.symlink(remote_code, local_code, target_is_directory=True)
    except:
        pass

    print("use for local excution of milabench")
    print("")
    print(f"    export MILABENCH_BASE={args.local}")
