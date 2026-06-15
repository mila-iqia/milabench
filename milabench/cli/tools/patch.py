"""Apply a global patch to all benchmark environments."""

import shutil
from dataclasses import dataclass
from pathlib import Path

from argklass.command import Command


def find_site_packages(venv_path: Path) -> Path:
    """Guess site-packages path inside a venv."""

    if not venv_path.exists():
        raise FileNotFoundError(f"{venv_path} does not exist")

    for sub in ["lib/python*/site-packages", "Lib/site-packages"]:
        for p in venv_path.glob(sub):
            return p

    raise RuntimeError(f"Could not locate site-packages under {venv_path}")


def cli_global_patch(args):
    """Copy benchmate.progress into sitecustomize.py of the target venv."""

    import benchmate.progress

    source_file = Path(benchmate.progress.__file__).resolve()

    site_packages = find_site_packages(Path(args.venv))
    target_file = site_packages / "sitecustomize.py"

    shutil.copy2(source_file, target_file)


class Patch(Command):
    """Apply a global patch to all benchmark environments."""

    name = "patch"

    # fmt: off
    @dataclass
    class Arguments:
        """Apply a global patch to all benchmark environments."""
        venv : str = None  # Path to the venv to patch
    # fmt: on

    @staticmethod
    def execute(args):
        cli_global_patch(args)


COMMANDS = Patch
