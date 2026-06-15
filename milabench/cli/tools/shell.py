import os
import subprocess
import sys
from dataclasses import dataclass

from argklass.command import Command
from argklass.arguments import group

from ...common import CommonArguments, get_multipack, selection_keys


class Shell(Command):
    """Open a shell in the environment of a selected benchmark."""

    name = "shell"

    # fmt: off
    @dataclass
    class Arguments:
        """Open a shell in the environment of a selected benchmark."""
        shared : CommonArguments = group(CommonArguments)
    # fmt: on

    @staticmethod
    def execute(args):
        mp = get_multipack(args, run_name="dev")

        for pack in mp.packs.values():
            if args.select in selection_keys(pack.config):
                break
        else:
            sys.exit(f"Cannot find a benchmark with selector {args.select}")

        subprocess.run(
            [os.environ["SHELL"]],
            env=pack.full_env(),
            cwd=pack.dirs.code,
        )


COMMANDS = Shell
