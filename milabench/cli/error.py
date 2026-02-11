import argparse
from dataclasses import dataclass, field

from ..testing import replay_run
from ..common import validation_names
from ..utils import multilogger, validation_layers


# fmt: off
@dataclass
class Arguments:
    runs: str
    tags: list = field(default_factory=list)
# fmt: on


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=str,
        help="Run folder",
        default="/home/mila/d/delaunap/batch_x_worker/",
    )

    return parser.parse_args()


def cli_error(args=None):
    if args is None:
        args = arguments()

    layer_names = validation_names("")
    layers = validation_layers(*layer_names, short=True)

    with multilogger(*layers) as log:
        for val in layers:
            if hasattr(val, "start_new_run"):
                val.start_new_run()
                
        for msg in replay_run(args.run):
            log(msg)


if __name__ == "__main__":
    cli_error()