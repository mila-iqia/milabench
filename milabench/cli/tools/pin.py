import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from argklass.command import Command
from argklass.arguments import argument, group

from ...common import CommonArguments, get_multipack, run_with_loggers
from ...log import TerminalFormatter, TextReporter


class Pin(Command):
    """Pin the benchmarks' dependencies."""

    name = "pin"

    # fmt: off
    @dataclass
    class Arguments:
        """Pin the benchmarks' dependencies."""
        shared       : CommonArguments = group(CommonArguments)
        pip_compile  : list[str]       = argument(default=[], nargs="*")   # Extra args to pass to pip-compile
        constraints  : list[str]       = argument(default=[], nargs="*")   # Constraints files
        variant      : Optional[str]   = None                              # Install variant
        from_scratch : bool            = False                             # Do not use previous pins if they exist
    # fmt: on

    @staticmethod
    def execute(args):
        overrides = {"*": {"install_variant": args.variant}} if args.variant else {}

        if "-h" in args.pip_compile or "--help" in args.pip_compile:
            out = (
                subprocess.check_output(["python3", "-m", "piptools", "compile", "--help"])
                .decode("utf-8")
                .split("\n")
            )
            for i in range(len(out)):
                if out[i].startswith("Usage:"):
                    bin = os.path.basename(sys.argv[0])
                    out[i] = out[i].replace(
                        "Usage: python -m piptools compile",
                        f"usage: {bin} pin [...] --pip-compile",
                    )
            print("\n".join(out))
            exit(0)

        mp = get_multipack(args, run_name="pin", overrides=overrides)

        return run_with_loggers(
            mp.do_pin(
                pip_compile_args=args.pip_compile,
                constraints=args.constraints,
                from_scratch=args.from_scratch,
            ),
            loggers=[
                TerminalFormatter(),
                TextReporter("stdout"),
                TextReporter("stderr"),
            ],
            mp=mp,
        )


COMMANDS = Pin
