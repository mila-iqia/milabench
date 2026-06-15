"""Restore data from a shared/network location to local disk."""

from dataclasses import dataclass
import os
import subprocess
import shutil
import sys
import tempfile
import traceback

from cantilever.core.timer import timeit, show_timings
from argklass.command import Command

from ...system import option


COPY_METHOD = option("copy.method", str, default=None)
COPY_NPROC = option("copy.nproc", int, default=32)
PREFER_COMPRESSED = option("copy.compressed", int, default=0)


def is_interactive():
    return sys.stdout.isatty() and sys.stderr.isatty()


def is_installed(command):
    return shutil.which(command) is not None


def simple_rsync(src, dst, folder):
    """Rsync the network folder to local"""

    rsync_interactive_flags = []
    if is_interactive():
        rsync_interactive_flags = ["--info=progress2"]

    rsync = ["rsync", "-ah"] + rsync_interactive_flags + ["--partial", src, dst]
    
    print(" ".join(rsync))
    subprocess.check_call(rsync)


def simple_rclone(src, dst, folder):
    # --multi-thread-streams=32                     => Elapsed time:     17m29.2s | 428.337 GiB
    # --multi-thread-streams=32 --transfers=16      => Elapsed time:      4m50.8s | 428.337 GiB
    # --multi-thread-streams=32 --transfers=32      => Elapsed time:      2m36.8s | 428.337 GiB
    # --multi-thread-streams=32 --transfers=64      => Elapsed time:      2m51.3s | 428.337 GiB
    rsync = ["rclone", "copy", "--multi-thread-streams=32", f"--transfers={COPY_NPROC}", 
             src,
             os.path.join(dst, folder)        
    ]
    
    print(" ".join(rsync))
    subprocess.check_call(rsync)
    
    
def find_xargs_rsync(src, dst, folder):
    nproc = COPY_NPROC
    
    rsync = [
        f"cd {src} && find . -type f -print0 | xargs -0 -n100 -P{nproc} sh -c 'rsync -ah --whole-file --ignore-times --inplace --no-compress -R \"$@\" {os.path.join(dst, folder)}' _"
    ]
    
    cmd = " ".join(rsync)
    print(cmd)
    subprocess.check_call(cmd, shell=True)
    

def archive_name(src):
    a1 = src + ".tar"
    a2 = src + ".tar.gz"

    a1 = a1 if os.path.exists(a1) else None
    a2 = a2 if os.path.exists(a2) else None

    if PREFER_COMPRESSED:
        return a2 or a1

    return a1 or a2


def parallel_untar(src, dst, folder):
    archive = archive_name(src)
    
    if archive is None:
        print(f"No archive found for {src}")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            list_cmd = ["tar", "-tf", archive]
            print(" ".join(list_cmd))
            result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
            
            with open("files.list", "w") as f:
                f.write(result.stdout)
            
            if False:
                shuf_cmd = ["shuf", "files.list"]
                print(" ".join(shuf_cmd))
                result = subprocess.run(shuf_cmd, capture_output=True, text=True, check=True)

                with open("files.list", "w") as f:
                    f.write(result.stdout)

            split_cmd = ["split", "-n", f"l/{COPY_NPROC}", "files.list", "file.chunk."]
            print(" ".join(split_cmd))
            subprocess.run(split_cmd, check=True)
            
            chunk_files = [f for f in os.listdir(".") if f.startswith("file.chunk.")]
            chunk_files.sort()

            os.makedirs(dst, exist_ok=True)
            
            processes = []
            for chunk_file in chunk_files:
                extract_cmd = ["tar", "-xf", archive, "-C", dst, "-T", chunk_file]
                print(" ".join(extract_cmd))
                process = subprocess.Popen(extract_cmd)
                processes.append(process)
            
            elapsed_time = 0
            while len(processes) > 0:
                pending = []
                if elapsed_time > 60:
                    elapsed_time = 0
                    print("Still working ...")

                for i, process in enumerate(processes):
                    try:
                        returncode = process.wait(timeout=1)
                    
                        if returncode != 0:
                            print(f"Warning: Process {i} (chunk {chunk_files[i]}) failed with return code {returncode}")

                    except subprocess.TimeoutExpired:
                        elapsed_time += 1
                        pending.append(process)
                
                processes = pending

        except Exception:
            traceback.print_exc()
            simple_untar(src, dst, folder)
        finally:
            os.chdir(old_cwd)


def simple_unzip(src, dst, folder):
    archive = src + ".zip"

    os.makedirs(dst, exist_ok=True)
    
    untar = ["unzip", "-q", archive, "-d", dst]
    print(" ".join(untar))
    subprocess.check_call(untar)
    return

def parallel_unzip(src, dst, folder):
    archive = src + ".zip"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            list_cmd = ["unzip", "-Z1", archive]
            print(" ".join(list_cmd))
            result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
            
            with open("files.list", "w") as f:
                f.write(result.stdout)

            split_cmd = ["split", "-n", f"l/{COPY_NPROC}", "files.list", "file.chunk."]
            print(" ".join(split_cmd))
            subprocess.run(split_cmd, check=True)
            
            chunk_files = [f for f in os.listdir(".") if f.startswith("file.chunk.")]
            chunk_files.sort()

            os.makedirs(dst, exist_ok=True)
            
            processes = []
            for chunk_file in chunk_files:
                extract_cmd = ["unzip", "-q", archive, "-d", dst, "-@"]
                print(" ".join(extract_cmd))

                with open(chunk_file, "r") as fp:
                    process = subprocess.Popen(" ".join(extract_cmd) + f" < {chunk_file}", shell=True, stdin=fp)
                    processes.append(process)
            
            elapsed_time = 0
            while len(processes) > 0:
                pending = []
                if elapsed_time > 60:
                    elapsed_time = 0
                    print("Still working ...")

                for i, process in enumerate(processes):
                    try:
                        returncode = process.wait(timeout=1)
                    
                        if returncode != 0:
                            print(f"Warning: Process {i} (chunk {chunk_files[i]}) failed with return code {returncode}")

                    except subprocess.TimeoutExpired:
                        elapsed_time += 1
                        pending.append(process)
                
                processes = pending

        except Exception:
            traceback.print_exc()
            simple_unzip(src, dst, folder)
        finally:
            os.chdir(old_cwd)


def simple_untar(src, dst, folder):
    archive = archive_name(src)
    
    untar = ["tar", "-xf", archive, "-C", dst]
    print(" ".join(untar))
    subprocess.check_call(untar)
    return


def rsync_untar(src, dst, folder):
    archive = archive_name(src)
    
    rsync = ["rsync", "-ah", archive, dst]
    print(" ".join(rsync))
    subprocess.check_call(rsync)
    
    local_tar = os.path.join(dst, folder + ".tar")
    untar = ["tar", "-xf", local_tar, "-C", dst]
    
    print(" ".join(untar))
    subprocess.check_call(untar)
    subprocess.check_call(["rm", "-f", local_tar])
    return


COPY_METHODS = {
    "RSYNC": simple_rsync,
    "RCLONE": simple_rclone,
    "FIND_XARGS_RSYNC": find_xargs_rsync,
    "UNTAR": simple_untar,
    "PARALLEL_UNTAR": parallel_untar,
    "RSYNC_UNTAR": rsync_untar,
    "UNZIP": simple_unzip,
    "PARALLEL_UNZIP": parallel_unzip,
}


def sync_folder(src, dst, folder, archive_only=False):
    # unzip         2550.70
    # rclone        2587.07
    # untar         1067.72  <===
    # rsync_untar   2220.23
    # rsync         3390.26
    if COPY_METHOD is not None:
        return COPY_METHODS[COPY_METHOD.upper()](src, dst, folder)

    else:
        if archive_name(src) is not None:
            return simple_untar(src, dst, folder)

        if archive_only:
            return
        
        if is_installed("rclone"):
            return simple_rclone(src, dst, folder)
        
        return simple_rsync(src, dst, folder)


class SharedSetup(Command):
    """Restore data from a shared/network location to local disk."""

    name = "sharedsetup"

    # fmt: off
    @dataclass
    class Arguments:
        """Restore data from a shared/network location to local disk."""
        network: str = None         # Shared Drive archive or path
        local: str = None           # Local Drive
        remote_data: str = None
        archive: bool = False       # Only look for tar archives, do not fall back to rsync/rclone
        maybe: bool = False         # Do not fail if the archive or data is not available
    # fmt: on

    @staticmethod
    def execute(args):
        """Restore data from a shared/network location to local disk."""

        remote_code = os.path.join(args.network, "venv")
        local_code  = os.path.join(args.local, "venv")

        remote_data = os.path.join(args.network, "data")
        remote_cache = os.path.join(args.network, "cache")

        has_data_dir = os.path.exists(remote_data)
        has_data_archive = archive_name(remote_data) is not None

        if not has_data_dir and not has_data_archive:
            if args.maybe:
                print(f"Data not found at {remote_data}[.tar[.gz]], skipping (--maybe)")
                return
            assert False, "missing data, was milabench prepare run ?"
        
        os.makedirs(args.local, exist_ok=True)

        with timeit(f"sync_{COPY_METHOD}_{COPY_NPROC}"):
            with timeit("sync_data"):
                sync_folder(remote_data, args.local, "data", archive_only=args.archive)

            with timeit("sync_cache"):
                sync_folder(remote_cache, args.local, "cache", archive_only=args.archive)

        if os.path.exists(remote_code):
            try:
                os.symlink(remote_code, local_code, target_is_directory=True)
            except:
                pass

        try:
            show_timings(force=True)
        except (IndexError, Exception):
            pass

        print("use for local excution of milabench")
        print("")
        print(f"    export MILABENCH_BASE={args.local}")


COMMANDS = SharedSetup


# mkdir -p $SLURM_TMPDIR/imagenet/train
# cd       $SLURM_TMPDIR/imagenet/train
# tar  -xf /network/datasets/imagenet/ILSVRC2012_img_train.tar --to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'
