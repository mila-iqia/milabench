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
    local_data  = os.path.join(args.local, "data")

    remote_cache = os.path.join(args.network, "cache")
    local_cache  = os.path.join(args.local, "cache")

    assert os.path.exists(remote_code), "missing venv, was milabench install run ?"
    assert os.path.exists(remote_data), "missing data, was milabench prepare run ?"
    
    os.makedirs(args.local, exist_ok=True)

    rsync_interactive_flags = []
    if is_interactive():
        rsync_interactive_flags = ["--info=progress2"]

    if args.network.endswith(".tar.gz"):
        untar = ["tar", "-xf", args.network, "-C", args.local]
        print(" ".join(untar))
        subprocess.check_call(untar)
    else:
        # rsync datasets & checkpoints to local disk

        if is_installed("rclone"):
            # --multi-thread-streams=32                     => Elapsed time:     17m29.2s | 428.337 GiB
            # --multi-thread-streams=32 --transfers=16      => Elapsed time:      4m50.8s | 428.337 GiB
            # --multi-thread-streams=32 --transfers=32      => Elapsed time:      2m36.8s | 428.337 GiB
            # --multi-thread-streams=32 --transfers=64      => Elapsed time:      2m51.3s | 428.337 GiB
            rsync = ["rclone", "copy", "--multi-thread-streams=32", "--transfers=32",  remote_data, local_data]
        else:
            rsync = ["rsync", "-azh"] + rsync_interactive_flags + ["--partial", remote_data, args.local]

        print(" ".join(rsync))
        subprocess.check_call(rsync)

        if is_installed("rclone"):
            rsync = ["rclone", "copy", "--multi-thread-streams=32", "--transfers=32", "--copy-links", remote_cache, local_cache]
        else:
            rsync = ["rsync", "-ah", "--inplace", "--whole-file", "--no-compress"] + rsync_interactive_flags + ["--partial", remote_cache, args.local]

        rsync = f"find {remote_cache} -type f -print0 | xargs -0 -n100 -P8 rsync -aR {{}} {args.local}"

        # Parallel rsync
        # find /src/dir -type f | parallel -j8 rsync -aR {} user@host:/dst/dir

        cmd = " ".join(rsync)
        print(cmd)
        subprocess.check_call(cmd, shell=True)
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