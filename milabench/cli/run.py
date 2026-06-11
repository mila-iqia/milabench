import os
from dataclasses import dataclass
from typing import Optional

from argklass.command import Command
from argklass.arguments import argument, group

from milabench.loggers.http import HTTPMetricPusher
from milabench.utils import validation_layers

from ..common import (
    CommonArguments,
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
from ..system import multirun, apply_system
from ..config import get_config_global
from benchmate.ipmi import ipmi_logger


def _fetch_arch(mp):
    from milabench.system import system_global

    try:
        arch = system_global.get()["arch"]
    except StopIteration:
        print("no selected bench")
        return None


def _dashboard(dash):
    dash_class = {
        "short": ShortDashFormatter,
        "long": LongDashFormatter,
        "no": None,
    }
    return dash_class.get(dash, None)


PLUGINS = {
    "dash": _dashboard,
    "http": HTTPMetricPusher,
    "term": TerminalFormatter,
    "mem": MemoryUsageExtractor,
    "txt": TextReporter,
    "ipmi": ipmi_logger,
}

try:
    from milabench.metrics.sqlalchemy import SQLAlchemy
    PLUGINS["sql"] = SQLAlchemy
except ImportError as err:
    def _sql_error(*args, **kwargs):
        raise err
    PLUGINS["sql"] = _sql_error


def _instantiate_loggers(*plugins):
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


def _fetch_plugins(args):
    layer_names = validation_names(args.validations)
    layers = validation_layers(*layer_names, short=not args.fulltrace)
    plugins = _instantiate_loggers(*args.plugin)
    if len(plugins) == 0:
        dash_class = _dashboard(args.dash)
        plugins = [
            TerminalFormatter() if not args.noterm else None,
            dash_class and dash_class(),
            TextReporter("stdout"),
            TextReporter("stderr"),
            MemoryUsageExtractor(),
            ipmi_logger(),
        ]
    return plugins + layers


def _run(mp, args, name):
    plugins = _fetch_plugins(args)

    success = run_with_loggers(
        mp.do_run(repeat=args.repeat),
        loggers=[
            DataReporter(),
            *plugins,
        ],
        mp=mp,
        short=not args.fulltrace,
        github_issues=args.github_issues,
    )

    if args.report:
        runs = {pack.logdir for pack in mp.packs.values()}
        reports = None
        if runs:
            reports = _read_reports(*runs)
            assert len(reports) != 0, "No reports found"

            summary = make_summary(reports)
            assert len(summary) != 0, "No summaries"

            weights = get_config_global()

            make_report(
                summary,
                compare=None,
                html=None,
                compare_gpus=False,
                price=None,
                title=None,
                sources=runs,
                errdata=reports and _error_report(reports),
                weights=weights,
            )

    return success


class Run(Command):
    """Run the benchmarks."""

    name = "run"

    # fmt: off
    @dataclass
    class Arguments:
        """Run the benchmarks."""
        shared        : CommonArguments = group(CommonArguments)
        run_name      : Optional[str]   = None                                       # Name of the run
        repeat        : int             = 1                                          # Number of times to repeat the benchmark
        fulltrace     : bool            = False                                      # On error show full stacktrace
        report        : bool            = True                                       # Show a report at the end
        dash          : str             = os.getenv("MILABENCH_DASH", "long")        # Dashboard type (short, long, no)
        noterm        : bool            = os.getenv("MILABENCH_NOTERM", "0") == "1"  # Disable terminal output
        validations   : Optional[str]   = None                                       # Validation layers to enable
        plugin        : list[str]       = argument(default=[], action="append", nargs="+")  # Logger plugin
        publish       : Optional[str]   = os.getenv("MILABENCH_PUBLISH_KEY", None)   # Push key to publish results
        dashboard_url : Optional[str]   = os.getenv("MILABENCH_DASHBOARD_URL", None) # Dashboard URL to publish results to
        github_issues : bool            = False                                      # Generate GitHub issue links for failures
    # fmt: on

    @staticmethod
    def execute(args):
        mp = get_multipack(args, run_name=args.run_name)
        arch = _fetch_arch(mp)
        init_arch(arch)

        all_run_folders = set()
        success = 0
        for name, conf in multirun():
            run_name = name or args.run_name

            mp = get_multipack(args, run_name=run_name)

            with apply_system(conf):
                try:
                    success += _run(mp, args, run_name)
                except AssertionError as err:
                    print(err)

                all_run_folders.update(
                    pack.logdir for pack in mp.packs.values() if pack.logdir
                )

        if args.publish and all_run_folders:
            from ._push_results import publish_results

            publish_results(
                all_run_folders,
                push_key=args.publish,
                dashboard_url=args.dashboard_url,
            )

        return success


COMMANDS = Run
