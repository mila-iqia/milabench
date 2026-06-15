"""Compare all runs with each other."""

import os
from dataclasses import dataclass
from typing import Optional

from argklass.command import Command

from ...common import _read_reports
from ...compare import compare, fetch_runs
from ...summary import make_summary


class Compare(Command):
    """Compare all runs with each other."""

    name = "compare"

    # fmt: off
    @dataclass
    class Arguments:
        """Compare all runs with each other."""
        folder : Optional[str] = None          # Folder with milabench results
        filter : Optional[str] = None          # Filter runs
        last   : Optional[int] = None          # Number of runs to compare
        metric : str           = "train_rate"  # Metric to compare
        stat   : str           = "median"      # Statistic to compare
    # fmt: on

    @staticmethod
    def execute(args):
        folder = args.folder

        if folder is None:
            base = os.environ.get("MILABENCH_BASE", None)
            if base is not None:
                folder = os.path.join(base, "runs")

        runs = fetch_runs(folder, args.filter)

        for run in runs:
            all_data = _read_reports(run.path)
            run.summary = make_summary(all_data)

        compare(runs, args.last, args.metric, args.stat)


COMMANDS = Compare
