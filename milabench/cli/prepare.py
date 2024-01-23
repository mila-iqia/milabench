from dataclasses import dataclass

from coleo import Option, tooled

from milabench.utils import validation_layers

from ..common import Option, get_multipack, run_with_loggers
from ..log import DataReporter, TerminalFormatter, TextReporter


# fmt: off
@dataclass
class Arguments:
    shortrace: bool = False
# fmt: on


@tooled
def arguments():
    # On error show full stacktrace
    shortrace: Option & bool = False

    return Arguments(shortrace)


@tooled
def cli_prepare(args=None):
    """Prepare a benchmark: download datasets, weights etc."""
    if args is None:
        args = arguments()

    mp = get_multipack(run_name="prepare.{time}")

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
    )
