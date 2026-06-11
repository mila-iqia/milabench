from dataclasses import dataclass
from typing import Optional

from argklass.command import Command
from argklass.arguments import group

from ...common import CommonArguments, get_multipack, run_with_loggers
from ...log import DataReporter, TerminalFormatter, TextReporter
from ...utils import validation_layers


class PrepareRun(Command):
    """Prepare then run the benchmarks."""

    name = "prepare-run"

    # fmt: off
    @dataclass
    class Arguments:
        """Prepare then run the benchmarks."""
        shared     : CommonArguments = group(CommonArguments)
        run_name   : Optional[str]   = None   # Name of the run
        repeat     : int             = 1      # Number of times to repeat
        shorttrace : bool            = False  # On error show short stacktrace
    # fmt: on

    @staticmethod
    def execute(args):
        mp = get_multipack(args, run_name="prepare.{time}")
        run_with_loggers(
            mp.do_prepare(),
            loggers=[
                TerminalFormatter(),
                TextReporter("stdout"),
                TextReporter("stderr"),
                DataReporter(),
                *validation_layers("error", short=args.shorttrace),
            ],
            mp=mp,
        )

        mp = get_multipack(args, run_name=args.run_name or "run.{time}")
        run_with_loggers(
            mp.do_run(repeat=args.repeat),
            loggers=[
                TerminalFormatter(),
                TextReporter("stdout"),
                TextReporter("stderr"),
                DataReporter(),
                *validation_layers("error", short=args.shorttrace),
            ],
            mp=mp,
        )


COMMANDS = PrepareRun
