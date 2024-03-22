import glob
import os
import sys
from dataclasses import dataclass, field

from coleo import Option, config as configuration, tooled

from ..common import Option, _error_report, _get_multipack, _push_reports, _read_reports
from ..fs import XPath
from ..report import make_report
from ..summary import make_summary


# fmt: off
@dataclass
class Arguments:
    runs        : list = field(default_factory=list)
    config      : str = os.environ.get("MILABENCH_CONFIG", None)
    compare     : str = None
    compare_gpus: bool = False
    html        : str = None
    price       : int = None
    push        : bool = False
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

    # Push reports to repo
    push: Option & bool = False

    return Arguments(runs, config, compare, compare_gpus, html, price, push)


@tooled
def cli_report(args=None):
    """Generate a report aggregating all runs together into a final report."""
    if args is None:
        args = arguments()

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

    if args.config:
        from milabench.common import arguments as multipack_args

        margs = multipack_args()
        margs.config = args.config

        args.config = _get_multipack(margs, return_config=True)

    assert args.config if args.push else None

    if not args.runs and args.config:
        run_dirs = {XPath(pack_config["dirs"]["runs"]) for pack_config in args.config.values()}
        filter = lambda _p: not any([XPath(_p).name.startswith(f"{prefix}.") for prefix in ("install", "prepare")])
        args.runs = sorted(
            {_r
             for _rd in run_dirs
             for _r in glob.glob(str(_rd / "*.*.*/"))
             if filter(_r)
            },
            key=lambda _p: XPath(_p).name.split(".")[-2:]
        )

    reports = None
    if args.runs:
        reports = _read_reports(*args.runs)
        summary = make_summary(reports.values())

    make_report(
        summary,
        compare=args.compare,
        weights=args.config,
        html=args.html,
        compare_gpus=args.compare_gpus,
        price=args.price,
        title=None,
        sources=args.runs,
        errdata=reports and _error_report(reports),
        stream=sys.stdout,
    )

    if len(reports) and args.push:
        reports_repo = next(iter(
            XPath(pack_config["dirs"]["base"]) / "reports"
            for pack_config in args.config.values()
        ))
        _push_reports(reports_repo, args.runs, summary)
