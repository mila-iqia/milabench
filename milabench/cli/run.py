import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

from coleo import Option, tooled

from milabench.loggers.http import HTTPMetricPusher
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
from ..config import get_config_global
from benchmate.ipmi import IPMIParallelMonitor


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
    plugins     : list[str] = field(default_factory=list)
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

    # [action: append]
    # [nargs: +]
    plugin: Option & str = []

    return Arguments(run_name, repeat, fulltrace, report, dash, noterm, validations, plugin)



def _fetch_arch(mp):
    from milabench.system import system_global
    
    try:
        arch = system_global.get()["arch"]
    except StopIteration:
        print("no selected bench")
        return None


def dashboard(dash):
    dash_class = {
        "short": ShortDashFormatter,
        "long": LongDashFormatter,
        "no": None,
    }

    return dash_class.get(dash, None)


PLUGINS = {
    "dash": dashboard,
    "http": HTTPMetricPusher,
    "term": TerminalFormatter,
    "mem": MemoryUsageExtractor,
    "txt": TextReporter,
    "ipmi": IPMIParallelMonitor
}

try:
    from milabench.metrics.sqlalchemy import SQLAlchemy

    PLUGINS["sql"] = SQLAlchemy

except ImportError as err:
    def sql_error(*args, **kwargs):
        raise err

    PLUGINS["sql"] = sql_error


def instantiate_loggers(*plugins):
    objects = []

    for logger in plugins:
        name, *rest = logger

        args = []
        kwargs = {}

        for p in rest:
            if "=" not in p:
                if len(kwargs) == 0:
                    args.append(p)
                else:
                    raise RuntimeError("Positional argument after keyword argument")
            
            else:
                k, v = p.split("=")
                kwargs[k] = v

        cls = PLUGINS.get(name, None)

        try:
            objects.append(cls(*args, **kwargs))
        except Exception:
            raise

    return objects


def fetch_plugins(args):
    # milabench run \
    # --plugin term                                         # TerminalFormatter
    # --plugin dash short                                   # ShortDashFormatter
    # --plugin dash long                                    # LongDashFormatter
    # --plugin txt stdout                                   # TextReporter("stdout")
    # --plugin txt stderr                                   # TextReporter("stderr")
    # --plugin mem                                          # MemoryUsageExtractor
    # --plugin http localhost:5000                          # HTTPMetricPusher              # Push to a dashboard
    # --plugin sql postgresql://<user>:<pwd>@host:port/db   # SQLAlchemyMetricPusher        # Push to a database
    #   testing url: postgresql://milabench_write:1234@localhost:5432/milabench 

    # ====
    # --plugin sql postgresql://milabench_write:1234@127.0.0.1:5432/milabench --plugin term 

    # Layer logic and plugin logic could be merged
    layer_names = validation_names(args.validations)
    layers = validation_layers(*layer_names, short=not args.fulltrace)

    plugins = instantiate_loggers(*args.plugins)

    if len(plugins) == 0:
        dash_class = dashboard(args.dash)

        plugins = [
            TerminalFormatter() if not args.noterm else None,
            dash_class and dash_class(),
            TextReporter("stdout"),
            TextReporter("stderr"),
            MemoryUsageExtractor(),
            IPMIParallelMonitor(),
        ]
    
    return plugins + layers


def run(mp, args, name):
    plugins = fetch_plugins(args)

    success = run_with_loggers(
        mp.do_run(repeat=args.repeat),
        loggers=[
            DataReporter(),
            *plugins,
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

            # This gets config BEFORE filters (select & exclude)
            weights = get_config_global()

            make_report(
                summary,
                compare=compare,
                html=html,
                compare_gpus=compare_gpus,
                price=price,
                title=None,
                sources=runs,
                errdata=reports and _error_report(reports),
                weights=weights,
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
