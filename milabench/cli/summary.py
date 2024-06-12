import json
from dataclasses import dataclass, field

from coleo import Option, tooled

from ..common import _read_reports
from ..summary import make_summary


# fmt: off
@dataclass
class Arguments:
    runs: list = field(default_factory=list)
    out: str = None
# fmt: on


@tooled
def arguments():
    # Directory(ies) containing the run data
    # [positional: +]
    runs: Option = []

    # Output file
    # [alias: -o]
    out: Option = None
    return Arguments(runs, out)


@tooled
def cli_summary(args=None):
    """Produce a JSON summary of a previous run."""
    if args is None:
        args = arguments()

    all_data = _read_reports(*args.runs)
    summary = make_summary(all_data.values())

    if args.out is not None:
        with open(args.out, "w") as file:
            json.dump(summary, file, indent=4)
    else:
        print(json.dumps(summary, indent=4))
