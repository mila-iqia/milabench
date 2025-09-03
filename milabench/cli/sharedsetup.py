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
    nproc = 32
    
    tar_compressed_archive = src + ".tar.gz"
    tar_archive = src + ".tar"
    archive = None
    
    # Prefer the uncompressed tar archive if both are available
    if os.path.exists(tar_archive):
        archive = tar_archive
        compressed = False
    
    elif os.path.exists(tar_compressed_archive):
        compressed = True
        archive = tar_compressed_archive
    # --
    
    if archive is not None:
        if False:
            untar = ["tar", "-xf", archive, "-C", dst]
            print(" ".join(untar))
            subprocess.check_call(untar)
            return
        else:
            # Rsync first and then untar the local tar archive
            rsync = ["rsync", "-azh", archive, dst]
            print(" ".join(rsync))
            subprocess.check_call(rsync)
            
            local_tar = os.path.join(dst, folder + ".tar")
            untar = ["tar", "-xf", local_tar, "-C", dst]
            
            print(" ".join(untar))
            subprocess.check_call(untar)
            subprocess.check_call(["rm", "-f", local_tar])
            return

    rsync_interactive_flags = []
    if is_interactive():
        rsync_interactive_flags = ["--info=progress2"]

    if is_installed("rclone"):
        # --multi-thread-streams=32                     => Elapsed time:     17m29.2s | 428.337 GiB
        # --multi-thread-streams=32 --transfers=16      => Elapsed time:      4m50.8s | 428.337 GiB
        # --multi-thread-streams=32 --transfers=32      => Elapsed time:      2m36.8s | 428.337 GiB
        # --multi-thread-streams=32 --transfers=64      => Elapsed time:      2m51.3s | 428.337 GiB
        rsync = ["rclone", "copy", "--multi-thread-streams=32", f"--transfers={nproc}",  os.path.join(dst, folder)]
    else:
        rsync = ["rsync", "-azh"] + rsync_interactive_flags + ["--partial", src, dst]

    # Parallel rsync
    if False:
        rsync = [
            f"cd {src} && find . -type f -print0 | xargs -0 -n100 -P{nproc} sh -c 'rsync -ah --whole-file --ignore-times --inplace --no-compress -R \"$@\" {os.path.join(dst, folder)}' _"
        ]

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

