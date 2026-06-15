"""Generate a report aggregating all runs together."""

import os
from dataclasses import dataclass
from typing import Optional

from argklass.command import Command
from argklass.arguments import argument

from .reporting.generate import cli_report


class Report(Command):
    """Generate a report aggregating all runs together."""

    name = "report"

    # fmt: off
    @dataclass
    class Arguments:
        """Generate a report aggregating all runs together."""
        runs            : list[str]     = argument(default=[], action="append")      # Runs directory
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


COMMANDS = Report
