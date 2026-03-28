import os
import subprocess
from dataclasses import dataclass

from coleo import Option, tooled


TAR_FLAGS = [
    "--sort=name",
    "--mtime=UTC 2020-01-01",
    "--owner=0",
    "--group=0",
    "--numeric-owner",
]


@dataclass
class Arguments:
    local: str
    network: str = None


@tooled
def arguments():
    # Path to the local milabench results directory (MILABENCH_BASE)
    local: Option & str = os.environ.get("MILABENCH_BASE", "/tmp/workspace")

    # Path to the shared/network directory to rsync archives to (optional)
    network: Option & str = None

    return Arguments(local, network)


def create_deterministic_tar(base_dir, folder):
    """Create a deterministic tar archive from a folder.

    Uses fixed metadata so that identical content produces byte-identical archives,
    allowing rsync to skip unchanged files.
    """
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        print(f"Skipping {folder}: directory does not exist at {folder_path}")
        return None

    archive_path = os.path.join(base_dir, f"{folder}.tar")
    cmd = ["tar"] + TAR_FLAGS + ["-cf", archive_path, "-C", base_dir, folder]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    return archive_path


@tooled
def cli_archive(args=None):
    """Create deterministic tar archives of data and cache for sharing."""
    if args is None:
        args = arguments()

    processes = []
    archives = []

    for folder in ("data", "cache"):
        folder_path = os.path.join(args.local, folder)
        if not os.path.isdir(folder_path):
            print(f"Skipping {folder}: directory does not exist")
            continue

        archive_path = os.path.join(args.local, f"{folder}.tar")
        cmd = ["tar"] + TAR_FLAGS + ["-cf", archive_path, "-C", args.local, folder]
        print(" ".join(cmd))
        proc = subprocess.Popen(cmd)
        processes.append((folder, proc))
        archives.append(archive_path)

    for folder, proc in processes:
        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(f"tar failed for {folder} with return code {returncode}")

    if args.network is not None:
        os.makedirs(args.network, exist_ok=True)
        for archive_path in archives:
            rsync = ["rsync", "--inplace", archive_path, args.network + "/"]
            print(" ".join(rsync))
            subprocess.check_call(rsync)

    print("Archive complete")
