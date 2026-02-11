import os
from dataclasses import dataclass

from coleo import Option, tooled

from milabench.utils import validation_layers

from ..common import (
    _error_report,
    _read_reports,
    get_multipack,
    init_arch,
    run_with_loggers,
    validation_names,
)
from ..log import (
    DataReporter,
    LongDashFormatter,
    ShortDashFormatter,
    TerminalFormatter,
    TextReporter,
)
from ..report import make_report
from ..sizer import MemoryUsageExtractor
from ..summary import make_summary
from ..system import multirun, apply_system, SizerOptions, option


# fmt: off
@dataclass
class Arguments:
    run_name    : str = None
    repeat      : int = 1
    fulltrace   : bool = False
    report      : bool = True
    dash        : str = os.getenv("MILABENCH_DASH", "long")
    noterm      : bool = os.getenv("MILABENCH_NOTERM", "0") == "1"
    validations : str = None
# fmt: on


@tooled
def arguments():
    # Name of the run
    run_name: Option = None

    # Number of times to repeat the benchmark
    repeat: Option & int = 1

    # On error show full stacktrace
    fulltrace: Option & bool = False

    # Do not show a report at the end
    # [negate]
    report: Option & bool = True

    # Which type of dashboard to show (short, long, or no)
    dash: Option & str = os.getenv("MILABENCH_DASH", "long")

    noterm: Option & bool = os.getenv("MILABENCH_NOTERM", "0") == "1"

    validations: Option & str = None

    return Arguments(run_name, repeat, fulltrace, report, dash, noterm, validations)



def _fetch_arch(mp):
    try:
        arch = next(iter(mp.packs.values())).config["system"]["arch"]
    except StopIteration:
        print("no selected bench")
        return None
    

def run(mp, args, name):
    layers = validation_names(args.validations)

    dash_class = {
        "short": ShortDashFormatter,
        "long": LongDashFormatter,
        "no": None,
    }.get(args.dash, None)
        
    success = run_with_loggers(
        mp.do_run(repeat=args.repeat),
        loggers=[
            # Terminal Formatter slows down the dashboard,
            # if lots of info needs to be printed
            # in particular rwkv
            TerminalFormatter() if not args.noterm else None,
            dash_class and dash_class(),
            TextReporter("stdout"),
            TextReporter("stderr"),
            DataReporter(),
            MemoryUsageExtractor(),
            *validation_layers(*layers, short=not args.fulltrace),
        ],
        mp=mp,
    )

    if args.report:
        runs = {pack.logdir for pack in mp.packs.values()}
        compare = None
        compare_gpus = False
        html = None
        price = None

        reports = None
        if runs:
            reports = _read_reports(*runs)
            assert len(reports) != 0, "No reports found"

            summary = make_summary(reports)
            assert len(summary) != 0, "No summaries"

            make_report(
                summary,
                compare=compare,
                html=html,
                compare_gpus=compare_gpus,
                price=price,
                title=None,
                sources=runs,
                errdata=reports and _error_report(reports),
            )

    return success


@tooled
def cli_run(args=None):
    """Run the benchmarks."""
    if args is None:
        args = arguments()

    # Load the configuration and system
    mp = get_multipack(run_name=args.run_name)
    arch = _fetch_arch(mp)

    # Initialize the backend here so we can retrieve GPU stats
    init_arch(arch)
    
    success = 0
    for name, conf in multirun():
        run_name = name or args.run_name
        
        # Note that this function overrides the system config
        mp = get_multipack(run_name=run_name)
        
        with apply_system(conf):
            try:
                success += run(mp, args, run_name)
            except AssertionError as err:
                print(err)

    return success
