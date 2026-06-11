from dataclasses import dataclass

from argklass.command import Command
from argklass.arguments import group

from milabench.utils import validation_layers

from ..common import CommonArguments, get_multipack, run_with_loggers
from ..log import DataReporter, TerminalFormatter, TextReporter


class Prepare(Command):
    """Prepare a benchmark: download datasets, weights etc."""

    name = "prepare"

    # fmt: off
    @dataclass
    class Arguments:
        """Prepare a benchmark: download datasets, weights etc."""
        shared        : CommonArguments = group(CommonArguments)
        shortrace     : bool            = False    # On error show short stacktrace
        github_issues : bool            = False    # Generate GitHub issue links for failures
    # fmt: on

    @staticmethod
    def execute(args):
        mp = get_multipack(args, run_name="prepare.{time}")

        return run_with_loggers(
            mp.do_prepare(),
            loggers=[
                TerminalFormatter(),
                TextReporter("stdout"),
                TextReporter("stderr"),
                DataReporter(),
                *validation_layers("error", short=args.shortrace),
            ],
            mp=mp,
            short=args.shortrace,
            github_issues=args.github_issues,
        )


COMMANDS = Prepare
