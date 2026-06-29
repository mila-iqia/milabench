from dataclasses import dataclass, field
from typing import Optional

from argklass.command import Command
from argklass.arguments import argument, group

from milabench.utils import validation_layers

from ..common import CommonArguments, get_multipack, run_with_loggers
from ..log import DataReporter, TerminalFormatter, TextReporter


def parse_version_overrides(args_list: list[str]) -> dict[str, str]:
    """Parse key=value pairs from positional args.

    Examples:
        ["cuda=130", "torch=2.12.0"] → {"cuda": "130", "torch": "2.12.0"}
    """
    overrides = {}
    for arg in args_list:
        if "=" in arg:
            key, value = arg.split("=", 1)
            overrides[key.strip()] = value.strip()
    return overrides


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
        set           : list[str]       = argument(default=[], nargs="*")  # Version overrides: cuda=130 torch=2.12.0
    # fmt: on

    @staticmethod
    def execute(args):
        overrides = {"*": {"install_variant": args.variant}} if args.variant else {}

        # Parse version overrides (e.g., --set cuda=130 torch=2.12.0)
        version_overrides = parse_version_overrides(args.set)
        if version_overrides:
            overrides.setdefault("*", {})["version_overrides"] = version_overrides

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
