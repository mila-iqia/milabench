import os
import sys
from dataclasses import dataclass, field

from coleo import Option, config as configuration, tooled

from ..common import Option, _error_report, _get_multipack, _read_reports
from ..report import make_report
from ..summary import make_summary


# fmt: off
@dataclass
class Arguments:
    runs:        list = field(default_factory=list)
    config      : str = os.environ.get("MILABENCH_CONFIG", None)
    compare     : str = None
    compare_gpus: bool = False
    html        : str = None
    price       : int = None
# fmt: on


@tooled
def arguments():
    # Runs directory
    # [action: append]
    runs: Option = []

    # Configuration file (for weights)
    config: Option & str = os.environ.get("MILABENCH_CONFIG", None)

    # Comparison summary
    compare: Option & configuration = None

    # Compare the GPUs
    compare_gpus: Option & bool = False

    # HTML report file
    html: Option = None

    # Price per unit
    price: Option & int = None

    return Arguments(runs, config, compare, compare_gpus, html, price)


@tooled
def cli_report(args = arguments()):
    """Generate a report aggregating all runs together into a final report."""

    # Examples
    # --------
    # >>> milabench report --runs results/
    # Source: /home/newton/work/milabench/milabench/../tests/results
    # =================
    # Benchmark results
    # =================
    #                    n fail       perf   perf_adj   std%   sem%% peak_memory
    # bert               2    0     201.06     201.06  21.3%   8.7%          -1
    # convnext_large     2    0     198.62     198.62  19.7%   2.5%       29878
    # td3                2    0   23294.73   23294.73  13.6%   2.1%        2928
    # vit_l_32           2    1     548.09     274.04   7.8%   0.8%        9771
    # <BLANKLINE>
    # Errors
    # ------
    # 1 errors, details in HTML report.


    reports = None
    if args.runs:
        reports = _read_reports(*args.runs)
        summary = make_summary(reports.values())

    if config:
        config = _get_multipack(config, return_config=True)

    make_report(
        summary,
        compare=args.compare,
        weights=config,
        html=args.html,
        compare_gpus=args.compare_gpus,
        price=args.price,
        title=None,
        sources=args.runs,
        errdata=reports and _error_report(reports),
        stream=sys.stdout,
    )