from dataclasses import dataclass

from coleo import Option, tooled

from milabench.utils import validation_layers

from ..common import get_multipack, run_with_loggers
from ..log import DataReporter, TerminalFormatter, TextReporter


# fmt: off
@dataclass
class Arguments:
    force: bool = False
    shorttrace:  bool = False
    variant: str = None
# fmt: on


@tooled
def arguments():
    # Force install
    force: Option & bool = False

    # On error show full stacktrace
    shorttrace: Option & bool = False

    # Install variant
    variant: Option & str = None

    return Arguments(force, shorttrace, variant)


@tooled
def cli_install(args=None):
    """Install the benchmarks' dependencies."""
    if args is None:
        args = arguments()

    overrides = {"*": {"install_variant": args.variant}} if args.variant else {}

    if args.force:
        mp = get_multipack(run_name="install.{time}", overrides=overrides)
        for pack in mp.packs.values():
            pack.install_mark_file.rm()
            pack.dirs.venv.rm()

    mp = get_multipack(run_name="install.{time}", overrides=overrides)

    return run_with_loggers(
        mp.do_install(),
        loggers=[
            TerminalFormatter(),
            TextReporter("stdout"),
            TextReporter("stderr"),
            DataReporter(),
            *validation_layers("error", short=args.shorttrace),
        ],
        mp=mp,
    )
