from dataclasses import dataclass
import os
import subprocess
import shutil
import sys

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


def is_interactive():
    return sys.stdout.isatty() and sys.stderr.isatty()


def is_installed(command):
    return shutil.which(command) is not None


def sync_folder(src, dst, folder):
    rsync_interactive_flags = []
    if is_interactive():
        rsync_interactive_flags = ["--info=progress2"]

    if is_installed("rclone"):
        # --multi-thread-streams=32                     => Elapsed time:     17m29.2s | 428.337 GiB
        # --multi-thread-streams=32 --transfers=16      => Elapsed time:      4m50.8s | 428.337 GiB
        # --multi-thread-streams=32 --transfers=32      => Elapsed time:      2m36.8s | 428.337 GiB
        # --multi-thread-streams=32 --transfers=64      => Elapsed time:      2m51.3s | 428.337 GiB
        rsync = ["rclone", "copy", "--multi-thread-streams=32", "--transfers=32",  os.path.join(dst, folder)]
    else:
        rsync = ["rsync", "-azh"] + rsync_interactive_flags + ["--partial", src, dst]

    # Parallel rsync
    rsync = [f"find {src} -type f -print0 | xargs -0 -n100 -P8  'rsync -ah --whole-file --ignore-times --inplace --no-compress -R \"$@\" {dst}'"]

    cmd = " ".join(rsync)
    print(cmd)
    subprocess.check_call(cmd, shell=True)


@tooled
def cli_shared_setup(args = None):
    #
    # TODO: Do this for each node
    #
    if args is None:
        args = arguments()
    
    remote_code = os.path.join(args.network, "venv")
    local_code  = os.path.join(args.local, "venv")

    remote_data = os.path.join(args.network, "data")
    remote_cache = os.path.join(args.network, "cache")

    assert os.path.exists(remote_code), "missing venv, was milabench install run ?"
    assert os.path.exists(remote_data), "missing data, was milabench prepare run ?"
    
    os.makedirs(args.local, exist_ok=True)

    if args.network.endswith(".tar.gz"):
        untar = ["tar", "-xf", args.network, "-C", args.local]
        print(" ".join(untar))
        subprocess.check_call(untar)
    else:
        # rsync datasets & checkpoints to local disk
        sync_folder(remote_data, args.local, "data")

        sync_folder(remote_cache, args.local, "cache")

    # create a soft link for the code
    try:
        os.symlink(remote_code, local_code, target_is_directory=True)
    except:
        pass

    print("use for local excution of milabench")
    print("")
    print(f"    export MILABENCH_BASE={args.local}")




# mkdir -p $SLURM_TMPDIR/imagenet/train
# cd       $SLURM_TMPDIR/imagenet/train
# tar  -xf /network/datasets/imagenet/ILSVRC2012_img_train.tar --to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'