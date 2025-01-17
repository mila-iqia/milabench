import os
import sys

from coleo import run_cli

from .compare import cli_compare
from .dev import cli_dev
from .install import cli_install
from .machine import cli_machine
from .matrix import cli_matrix_run
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
from .resolve import cli_resolve
from .new import cli_new
from .env import cli_env
from .prepare_run import cli_prepare_run
from .gated import cli_gated
from .sharedsetup import cli_shared_setup
from .docker import cli_docker


class Main:
    def new():
        """Create a new benchmark from template"""
        return cli_new()

    def run():
        """Run the benchmarks."""
        return cli_run()

    def prepare():
        """Prepare a benchmark: download datasets, weights etc."""
        return cli_prepare()

    def install():
        """Install the benchmarks' dependencies."""
        return cli_install()

    def pin():
        """Pin the benchmarks' dependencies."""
        return cli_pin()

    def dev():
        """Create a shell in a benchmark's environment for development."""
        return cli_dev()

    def summary():
        """Produce a JSON summary of a previous run."""
        return cli_summary()

    def compare():
        """Compare all runs with each other."""
        return cli_compare()

    def report():
        """Generate a report aggregating all runs together into a final report."""
        return cli_report()

    def pip():
        """Run pip on every pack"""
        return cli_pip()

    def slurm_system():
        """Generate a system file based of slurm environment variables"""
        return cli_slurm_system()

    def machine():
        """Display machine metadata.
        Used to generate metadata json to back populate archived run

        """
        return cli_machine()

    def publish():
        """Publish an archived run to a database"""
        return cli_publish()

    def schedule():
        """Launch a slurm job to run milabench"""
        return cli_schedule()

    def sqlsetup():
        return cli_sqlsetup()

    def write_report_to_pr():
        return cli_write_report_to_pr()

    def matrix():
        return cli_matrix_run()

    def resolve():
        return cli_resolve()
    
    def env():
        """Print milabench environment variables"""
        cli_env()

    def prepare_run():
        cli_prepare_run()

    def gated():
        cli_gated()

    def sharedsetup():
        cli_shared_setup()

    def docker():
        cli_docker()


def main(argv=None):
    sys.path.insert(0, os.path.abspath(os.curdir))
    if argv is None:
        argv = sys.argv[1:]
    argv = [str(x) for x in argv]
    try:
        sys.exit(run_cli(Main, argv=argv))
    except KeyboardInterrupt:
        pass
