"""Run pip in the environments of selected benchmarks."""

from dataclasses import dataclass

from argklass.command import Command
from argklass.arguments import argument, group

from ...common import CommonArguments, get_multipack, run_sync


class Pip(Command):
    """Run a pip command in the environments of selected benchmarks."""

    name = "pip"

    # fmt: off
    @dataclass
    class Arguments:
        """Run a pip command in the environments of selected benchmarks."""
        shared   : CommonArguments = group(CommonArguments)
        pip_args : list[str]       = argument(default=[], nargs="*")  # Arguments to pass to pip
    # fmt: on

    @staticmethod
    def execute(args):
        mp = get_multipack(args, run_name="pip")

        for pack in mp.packs.values():
            run_sync(pack.pip_install(*args.pip_args))


COMMANDS = Pip
