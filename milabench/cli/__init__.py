import os
import sys

from coleo import run_cli

from .compare import cli_compare
from .dev import cli_dev
from .install import cli_install
from .machine import cli_machine
from .pin import cli_pin
from .pip import cli_pip
from .pr import cli_write_report_to_pr
from .prepare import cli_prepare
from .publish import cli_publish
from .report import cli_report
from .run import cli_run
from .schedule import cli_schedule
from .slurm import cli_slurm_system
from .sql import cli_sqlsetup
from .summary import cli_summary


class Main:
    def run():
        """Run the benchmarks."""
        cli_run()

    def prepare():
        """Prepare a benchmark: download datasets, weights etc."""
        cli_prepare()

    def install():
        """Install the benchmarks' dependencies."""
        cli_install()

    def pin():
        """Pin the benchmarks' dependencies."""
        cli_pin()

    def dev():
        """Create a shell in a benchmark's environment for development."""
        cli_dev()

    def summary():
        """Produce a JSON summary of a previous run."""
        cli_summary()

    def compare():
        """Compare all runs with each other."""
        cli_compare()

    def report():
        """Generate a report aggregating all runs together into a final report."""
        cli_report()

    def pip():
        """Run pip on every pack"""
        cli_pip()

    def slurm_system():
        """Generate a system file based of slurm environment variables"""
        cli_slurm_system()

    def machine():
        """Display machine metadata.
        Used to generate metadata json to back populate archived run

        """
        cli_machine()

    def publish():
        """Publish an archived run to a database"""
        cli_publish()

    def schedule():
        """Launch a slurm job to run milabench"""
        cli_schedule()

    def sqlsetup():
        cli_sqlsetup()

    def write_report_to_pr():
        cli_write_report_to_pr()


def main(argv=None):
    sys.path.insert(0, os.path.abspath(os.curdir))
    if argv is None:
        argv = sys.argv[1:]
    argv = [str(x) for x in argv]
    try:
        sys.exit(run_cli(Main, argv=argv))
    except KeyboardInterrupt:
        pass
