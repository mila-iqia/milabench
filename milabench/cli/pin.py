import os
import subprocess
import sys
from dataclasses import dataclass

from coleo import Option, tooled

from ..common import get_multipack, run_with_loggers
from ..log import TerminalFormatter, TextReporter


# fmt: off
@dataclass
class Arguments:
    pip_compile: tuple = tuple()
    constraints: tuple = tuple()
    variant: str = None
    from_scratch: bool = False
# fmt: on


@tooled
def arguments():
    # Extra args to pass to pip-compile
    # [nargs: --]
    pip_compile: Option = tuple()

    # Constraints files
    # [options: -c]
    # [nargs: *]
    constraints: Option = tuple()

    # Install variant
    variant: Option & str = None

    # Do not use previous pins if they exist
    from_scratch: Option & bool = False

    return Arguments(pip_compile, constraints, variant, from_scratch)


@tooled
def cli_pin(args=None):
    """Pin the benchmarks' dependencies."""
    if args is None:
        args = arguments()
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

    mp = get_multipack(run_name="pin", overrides=overrides)

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
