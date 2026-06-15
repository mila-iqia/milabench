"""Generate a report aggregating all runs together."""

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from argklass.command import Command
from argklass.arguments import argument

from ...common import _error_report, _read_reports
from ...report import make_report
from ...summary import make_summary


def _consolidate_runs(runs_dir, run_name=None):
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        return runs_dir

    subdirs = [d for d in runs_path.iterdir() if d.is_dir()]

    for d in subdirs:
        if d.name.startswith("install") or d.name.startswith("prepare"):
            print(f"[consolidate] Removing {d.name}/")
            shutil.rmtree(d)

    run_dirs = sorted(
        [d for d in runs_path.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )

    if not run_dirs:
        return runs_dir

    target = run_dirs[-1]

    for source in run_dirs[:-1]:
        for f in source.iterdir():
            if f.is_file():
                dest = target / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
        shutil.rmtree(source)
        print(f"[consolidate] Merged {source.name}/ into {target.name}/")

    if run_name:
        final = runs_path / run_name
        if final != target:
            if final.exists():
                shutil.rmtree(final)
            target.rename(final)
            target = final
            print(f"[consolidate] Renamed to {run_name}/")

    print(f"[consolidate] Result: {target}")
    return str(target)


def cli_report(args):
    """Generate a report aggregating all runs together into a final report.

    Kept as a standalone function because it is also called from
    ``reporting/__init__.py`` when ``milabench report`` is invoked
    without a subcommand.
    """
    if args.consolidate and args.runs:
        args.runs = [_consolidate_runs(r, run_name=args.run_name) for r in args.runs]

    reports = None
    summary = None
    if args.runs:
        reports = _read_reports(*args.runs)
        summary = make_summary(
            reports,
            filter_failures=args.filter_failures,
            latest_only=args.latest,
        )

    config = None
    if args.config:
        from ...common import CommonArguments, _get_multipack
        margs = CommonArguments(config=args.config)
        config = _get_multipack(margs, return_config=True)

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

    if args.publish and args.runs:
        from .._push_results import publish_results

        success = publish_results(
            args.runs,
            push_key=args.publish,
            dashboard_url=args.dashboard_url,
        )
        if not success:
            sys.exit(1)


class Generate(Command):
    """Generate a report aggregating all runs together into a final report."""

    name = "generate"

    # fmt: off
    @dataclass
    class Arguments:
        """Generate a report aggregating all runs together into a final report."""
        runs            : list[str]     = argument(default=[], nargs="*")             # Runs directory
        config          : Optional[str] = os.getenv("MILABENCH_CONFIG")              # Configuration file
        compare         : Optional[str] = None                                       # Comparison summary
        compare_gpus    : bool          = False                                      # Compare the GPUs
        html            : Optional[str] = None                                       # HTML report file
        price           : Optional[int] = None                                       # Price per unit
        filter_failures : bool          = False                                      # Filter out failed runs
        latest          : bool          = False                                      # Only consider the latest run
        publish         : Optional[str] = os.getenv("MILABENCH_PUBLISH_KEY", None)   # Push key to publish results
        dashboard_url   : Optional[str] = os.getenv("MILABENCH_DASHBOARD_URL", None) # Dashboard URL
        consolidate     : bool          = False                                      # Consolidate run folders
        run_name        : Optional[str] = None                                       # Name for consolidated run folder
    # fmt: on

    @staticmethod
    def execute(args):
        return cli_report(args)


COMMANDS = Generate
