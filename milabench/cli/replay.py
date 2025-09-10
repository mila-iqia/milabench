from dataclasses import dataclass, field
import os

from coleo import Option, tooled

from ..testing import replay_validation_scenario
from ..log import (
    DataReporter,
    LongDashFormatter,
    ShortDashFormatter,
    TerminalFormatter,
    TextReporter,
)
from ..sizer import MemoryUsageExtractor
from ..utils import validation_layers
from ..common import validation_names


# fmt: off
@dataclass
class Arguments:
    folder      : str
    validations : str  = None
    dash        : str  = os.getenv("MILABENCH_DASH", "long")
    noterm      : bool = os.getenv("MILABENCH_NOTERM", "0") == "1"
    fulltrace   : bool = False
# fmt: on


@tooled
def arguments() -> Arguments:
    # Directory(ies) containing the run data
    # [positional]
    folder      : Option & str

    validations : Option & str = None
    dash        : Option & str = os.getenv("MILABENCH_DASH", "long")
    noterm      : Option & bool = os.getenv("MILABENCH_NOTERM", "0") == "1"
    fulltrace   : Option & bool = False

    return Arguments(folder, validations, dash, noterm, fulltrace)

@tooled
def cli_replay(args: Arguments = None):
    if args is None:
        args = arguments()

    layers = validation_names(args.validations)

    dash_class = {
        "short": ShortDashFormatter,
        "long": LongDashFormatter,
        "no": None,
    }.get(args.dash, None)

    print(args.folder)

    replay_validation_scenario(
        args.folder,

        # TerminalFormatter() if not args.noterm else None,
        # dash_class and dash_class(),
        # TextReporter("stdout"),
        # TextReporter("stderr"),
        # DataReporter(),
        # MemoryUsageExtractor(),
        *validation_layers(*layers, short=not args.fulltrace)
    )