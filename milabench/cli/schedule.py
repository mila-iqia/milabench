from dataclasses import dataclass, field

from coleo import Option, tooled

from ..schedule import launch_milabench


# fmt: off
@dataclass
class Arguments:
    sync: bool = False
    dry : bool = False
    args: list = field(default_factory=list)
# fmt: on


@tooled
def arguments():
    # tail -f on the slurm job
    sync: Option & bool = False

    # Print the command and return without running it
    dry: Option & bool = False

    # pip arguments
    # [remainder]
    args: Option = []

    return Arguments(sync, dry, args)


@tooled
def cli_schedule(args=None):
    """Launch a slurm job to run milabench"""
    # milabench schedule --sync -- --select resnet50
    if args is None:
        args = arguments()

    launch_milabench(args.args, sbatch_args=None, dry=args.dry, sync=args.sync)
