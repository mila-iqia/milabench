from dataclasses import dataclass
import os
import subprocess
import shutil
import sys
import tempfile
import traceback

from cantilever.core.timer import timeit, show_timings
from coleo import Option, tooled

from ..system import option


COPY_METHOD = option("copy.method", str, default=None)
COPY_NPROC = option("copy.nproc", int, default=32)
PREFER_COMPRESSED = option("copy.compressed", int, default=0)


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
    
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Get the list of files in the archive
            list_cmd = ["tar", "-tf", archive]
            print(" ".join(list_cmd))
            result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
            
            # Save the file list to a file
            with open("files.list", "w") as f:
                f.write(result.stdout)
            
            # Shuffle a little bit so the parallel processes get different files
            if False:
                shuf_cmd = ["shuf", "files.list"]
                print(" ".join(shuf_cmd))
                result = subprocess.run(shuf_cmd, capture_output=True, text=True, check=True)

                with open("files.list", "w") as f:
                    f.write(result.stdout)

            # Split the file list into chunks for parallel processing
            split_cmd = ["split", "-n", f"l/{COPY_NPROC}", "files.list", "file.chunk."]
            print(" ".join(split_cmd))
            subprocess.run(split_cmd, check=True)
            
            # Get list of chunk files
            chunk_files = [f for f in os.listdir(".") if f.startswith("file.chunk.")]
            chunk_files.sort()

            os.makedirs(dst, exist_ok=True)
            
            # Run multiple tar processes in parallel
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

                # Wait for all processes to complete
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
            # Change back to original directory
            os.chdir(old_cwd)


def simple_untar(src, dst, folder):
    archive = archive_name(src)
    
    untar = ["tar", "-xf", archive, "-C", dst]
    print(" ".join(untar))
    subprocess.check_call(untar)
    return


def rsync_untar(src, dst, folder):
    archive = archive_name(src)
    
    # Rsync first and then untar the local tar archive
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
    "RSYNC": simple_rsync,                  # Rsync the network folder to local
    "RCLONE": simple_rclone,                # Rclone the network folder to local
    "FIND_XARGS_RSYNC": find_xargs_rsync,   # Find the files in the network folder, make batches and rsync them to local
    "UNTAR": simple_untar,                  # untar the network archive to local
    "PARALLEL_UNTAR": parallel_untar,       # parallel untar the network archive to local
    "RSYNC_UNTAR": rsync_untar,             # rsync the network archive to local and untar it to local
}


def sync_folder(src, dst, folder):
    # rclone        2587.07
    # untar         1067.72  <===
    # rsync_untar   2220.23
    # rsync         3390.26
    if COPY_METHOD is not None:
        return COPY_METHODS[COPY_METHOD.upper()](src, dst, folder)

    else:
        if archive_name(src) is not None:
            return simple_untar(src, dst, folder)
        
        if is_installed("rclone"):
            return simple_rclone(src, dst, folder)
        
        return simple_rsync(src, dst, folder)


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

    with timeit(f"sync_{COPY_METHOD}_{COPY_NPROC}"):
        # rsync datasets & checkpoints to local disk
        with timeit("sync_data"):
            sync_folder(remote_data, args.local, "data")

        with timeit("sync_cache"):
            sync_folder(remote_cache, args.local, "cache")

    show_timings(force=True)

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

