import os
from dataclasses import dataclass

from coleo import Option, tooled

from ..common import _read_reports
from ..compare import compare, fetch_runs
from ..summary import make_summary


# fmt: off
@dataclass
class Arguments:
    folder  : str = None
    last    : int = None
    metric  : str = "train_rate"
    stat    : str = "median"
    filter  : str = None
# fmt: on


@tooled
def arguments():
    # [positional: ?]
    folder: Option = None

    filter: Option & str = None

    last: Option & int = None

    metric: Option & str = "train_rate"

    stat: Option & str = "median"

    return Arguments(folder, last, metric, stat, filter)


@tooled
def cli_compare(args=None):
    """Compare all runs with each other."""
    if args is None:
        args = arguments()

    # Parameters
    # ----------
    # folder: str
    #     Folder where milabench results are stored
    # last: int
    #     Number of runs to compare i.e 3 means the 3 latest runs only
    # metric: str
    #     Metric to compare
    # stat: str
    #     statistic name to compare
    # Examples
    # --------
    # >>> milabench compare results/ --last 3 --metric train_rate --stat median
    #                                        |   rufidoko |   sokilupa
    #                                        | 2023-02-23 | 2023-03-09
    # bench                |          metric |   16:16:31 |   16:02:53
    # ----------------------------------------------------------------
    # bert                 |      train_rate |     243.05 |     158.50
    # convnext_large       |      train_rate |     210.11 |     216.23
    # dlrm                 |      train_rate |  338294.94 |  294967.41
    # efficientnet_b0      |      train_rate |     223.56 |     223.48

    if args.folder is None:
        base = os.environ.get("MILABENCH_BASE", None)

        if base is not None:
            args.folder = os.path.join(base, "runs")

    runs = fetch_runs(args.folder, args.filter)

    for run in runs:
        all_data = _read_reports(run.path)
        run.summary = make_summary(all_data)

    compare(runs, args.last, args.metric, args.stat)
