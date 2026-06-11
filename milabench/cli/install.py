from dataclasses import dataclass
from typing import Optional

from argklass.command import Command
from argklass.arguments import group

from milabench.utils import validation_layers

from ..common import CommonArguments, get_multipack, run_with_loggers
from ..log import DataReporter, TerminalFormatter, TextReporter


class Install(Command):
    """Install the benchmarks' dependencies."""

    name = "install"

    # fmt: off
    @dataclass
    class Arguments:
        """Install the benchmarks' dependencies."""
        shared        : CommonArguments = group(CommonArguments)
        force         : bool            = False    # Force install (remove venvs)
        update        : bool            = False    # Update packages
        shorttrace    : bool            = False    # On error show short stacktrace
        variant       : Optional[str]   = None     # Install variant (unpinned, cuda, hpu, xpu, rocm)
        github_issues : bool            = False    # Generate GitHub issue links for failures
    # fmt: on

    @staticmethod
    def execute(args):
        overrides = {"*": {"install_variant": args.variant}} if args.variant else {}

        mp = get_multipack(args, run_name="install.{time}", overrides=overrides)
        for pack in mp.packs.values():
            if args.force or args.update:
                pack.install_mark_file.rm()

            if args.force:
                pack.dirs.venv.rm()

        mp = get_multipack(args, run_name="install.{time}", overrides=overrides)

        rc = run_with_loggers(
            mp.do_install(),
            loggers=[
                TerminalFormatter(),
                TextReporter("stdout"),
                TextReporter("stderr"),
                DataReporter(),
                *validation_layers("error", short=args.shorttrace),
            ],
            mp=mp,
            short=args.shorttrace,
            github_issues=args.github_issues,
        )

        from .data.gated import _print_gated_info
        _print_gated_info(args)

        return rc


COMMANDS = Install
