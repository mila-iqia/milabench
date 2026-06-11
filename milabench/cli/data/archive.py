"""Create deterministic tar archives of data and cache for sharing."""

import os
import subprocess
from dataclasses import dataclass
from typing import Optional

from argklass.command import Command


TAR_FLAGS = [
    "--sort=name",
    "--mtime=UTC 2020-01-01",
    "--owner=0",
    "--group=0",
    "--numeric-owner",
]


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


def cli_archive(args):
    """Create deterministic tar archives of data and cache for sharing."""

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


class Archive(Command):
    """Create deterministic tar archives of data and cache for sharing."""

    name = "archive"

    # fmt: off
    @dataclass
    class Arguments:
        """Create deterministic tar archives of data and cache for sharing."""
        local   : str           = None  # Path to the local milabench results directory (MILABENCH_BASE)
        network : Optional[str] = None  # Path to the shared/network directory to rsync archives to
    # fmt: on

    @staticmethod
    def execute(args):
        if args.local is None:
            args.local = os.environ.get("MILABENCH_BASE", "/tmp/workspace")
        cli_archive(args)


COMMANDS = Archive
