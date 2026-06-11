"""Run milabench multiple times with different configurations."""

from dataclasses import dataclass

from argklass.command import Command
from argklass.arguments import group

from ..common import CommonArguments, get_multipack


class Multirun(Command):
    """Run milabench multiple times with different configurations."""

    name = "multirun"

    # fmt: off
    @dataclass
    class Arguments:
        """Run milabench multiple times with different configurations."""
        shared : CommonArguments = group(CommonArguments)
        repeat : int             = 1  # Number of times to repeat
    # fmt: on

    @staticmethod
    def execute(args):
        from ..system import multirun, apply_system

        for name, conf in multirun():
            print(f"Multirun: {name}")
            mp = get_multipack(args, run_name=name or "multirun")

            with apply_system(conf):
                mp.do_run(repeat=args.repeat)


COMMANDS = Multirun
