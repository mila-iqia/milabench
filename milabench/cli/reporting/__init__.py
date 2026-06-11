"""Reporting and analysis commands."""

import os
from dataclasses import dataclass
from typing import Optional

from argklass.command import ParentCommand
from argklass.arguments import argument


class Reporting(ParentCommand):
    """Generate reports, compare runs, publish results."""

    name: str = "report"

    # fmt: off
    @dataclass
    class Arguments:
        """Generate reports, compare runs, publish results."""
        runs            : list[str]     = argument(default=[], action="append")      # Runs directory
        config          : Optional[str] = os.getenv("MILABENCH_CONFIG")              # Configuration file
        compare         : Optional[str] = None                                       # Comparison summary
        compare_gpus    : bool          = False                                      # Compare the GPUs
        html            : Optional[str] = None                                       # HTML report file
        price           : Optional[int] = None                                       # Price per unit
        filter_failures : bool          = False                                      # Filter out failed runs
        latest          : bool          = False                                      # Only consider the latest run
        publish         : Optional[str] = os.getenv("MILABENCH_PUBLISH_KEY", None)   # Push key
        dashboard_url   : Optional[str] = os.getenv("MILABENCH_DASHBOARD_URL", None) # Dashboard URL
        consolidate     : bool          = False                                      # Consolidate run folders
        run_name        : Optional[str] = None                                       # Name for consolidated run folder
    # fmt: on

    @staticmethod
    def module():
        import milabench.cli.reporting
        return milabench.cli.reporting

    @classmethod
    def execute(cls, args):
        cmd = cls.module().__name__
        subcmd = vars(args).get(cls.command_field())

        if subcmd is None:
            from .generate import cli_report
            return cli_report(args)

        vars(args).pop(cls.command_field())
        dispatch_cmd = cls.dispatch.get((cmd, subcmd), None)
        if dispatch_cmd:
            return dispatch_cmd.execute(args)

        raise RuntimeError(f"Subcommand report {subcmd} is not defined")


COMMANDS = Reporting
